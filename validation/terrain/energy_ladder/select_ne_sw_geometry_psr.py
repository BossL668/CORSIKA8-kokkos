#!/usr/bin/env python3
"""Select and audit a NE-to-SW mountain crossing using the actual DEM on PSR."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import hashlib
import json
import socket
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

assert socket.gethostname() == 'psrpku2025', 'Numerical work runs only on PSR'
base = Path('/data/yhlu/CorsikaData/corsika_validation_results')
old = base/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916'
out = base/'beta5_ne_sw_geometry_20260916'
out.mkdir(exist_ok=True)
scene = yaml.safe_load((old/'scenes/screen.yaml').read_text())
mesh = Path(scene['geometry']['mesh_path'])
with mesh.open('rb') as f:
    header = []
    while True:
        line = f.readline().decode().strip(); header.append(line)
        if line == 'end_header': break
    nv = int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
    nf = int(next(x.split()[-1] for x in header if x.startswith('element face ')))
    vertices = np.fromfile(f, dtype='<f8', count=nv*3).reshape(nv, 3)
    faces = np.fromfile(f, dtype=np.dtype([('count', 'u1'), ('indices', '<u4', (3,))]), count=nf)['indices']
triangles = vertices[faces]
e1 = triangles[:, 1]-triangles[:, 0]; e2 = triangles[:, 2]-triangles[:, 0]
normal = np.cross(e1, e2)
top = faces[normal[:, 2] > 1e-10]
used = np.unique(top); v = vertices[used]
tri = mtri.Triangulation(v[:, 0], v[:, 1], triangles=np.searchsorted(used, top))
height = mtri.LinearTriInterpolator(tri, v[:, 2])
stations = scene['radio']['observers']; xyz = np.array([s['position_enu_m'] for s in stations])
by_name = {s['name']:np.array(s['position_enu_m']) for s in stations}
print('Mesh bounds', v.min(axis=0), v.max(axis=0), 'array centre', xyz.mean(axis=0), flush=True)

def save(name, value): (out/name).write_text(json.dumps(value, indent=2)+'\n')

def hits(origin, direction, lo, hi):
    h = np.cross(direction, e2); det = np.einsum('ij,ij->i', e1, h)
    valid = abs(det)>1e-12; safe = np.where(valid, det, 1)
    rel = origin-triangles[:, 0]; u = np.einsum('ij,ij->i', rel, h)/safe
    q = np.cross(rel, e1); w = q@direction/safe
    s = np.einsum('ij,ij->i', e2, q)/safe
    take = valid & (u>=-1e-10) & (w>=-1e-10) & (u+w<=1+1e-10) & (s>lo) & (s<hi)
    ids = np.flatnonzero(take); ids = ids[np.argsort(s[ids])]
    ids = ids[np.r_[True, np.diff(s[ids])>1e-6]] if len(ids) else ids
    return [(float(s[j]), int(j)) for j in ids]

targets = {'array_centre':xyz.mean(axis=0), 'E18':by_name['E18'], 'E01':by_name['E01'], 'N10':by_name['N10'], 'arm_junction':np.array([-1161., -2029., -168.])}
candidates = []
samples = np.arange(-14000., 14000.01, 5.)
for target_name, target in targets.items():
    for azimuth in np.arange(200., 250.01, 5.):
        az = np.radians(azimuth)
        for elevation in [-8., -6., -4., -3., -2., -1., 0., 1., 2.]:
            el = np.radians(elevation)
            direction = np.array([np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), np.sin(el)])
            for offset in [30., 75., 150., 250., 400., 600.]:
                anchor = target.copy(); anchor[2] += offset
                p = anchor+samples[:, None]*direction
                terrain = height(p[:, 0], p[:, 1]); good = ~np.ma.getmaskarray(terrain)
                indices = np.flatnonzero(good)
                if not len(indices) or not np.all(good[indices[0]:indices[-1]+1]): continue
                rock = good & (p[:, 2]<terrain.filled(-np.inf))
                edges = np.diff(np.r_[False, rock, False].astype(int))
                starts = np.flatnonzero(edges==1); ends = np.flatnonzero(edges==-1)-1
                upstream = np.flatnonzero(samples[ends]<-1200)
                if not len(upstream): continue
                j = upstream[-1]; a, b = starts[j], ends[j]
                if a<=indices[0]+20 or b>=indices[-1]-20: continue
                entry = samples[a]; exit = samples[b]
                next_ground = samples[starts[j+1]] if j+1<len(starts) else samples[indices[-1]]
                chord = exit-entry; tail = next_ground-exit
                if not (200<=chord<=4000 and -7500<exit<-1200 and next_ground>500 and tail>=3500): continue
                if np.any(rock[a-20:a]): continue
                # A cheap DEM-height prescreen, followed below by exact all-face intersections.
                exit_point = anchor+(exit+5)*direction
                fractions = np.linspace(.002, 1., 450)
                paths = exit_point[None,None,:]+fractions[None,:,None]*(xyz[:,None,:]-exit_point)
                h = height(paths[:,:,0].ravel(), paths[:,:,1].ravel()).reshape(80,-1)
                visible = (~np.any(np.ma.getmaskarray(h), axis=1)) & np.all(paths[:,:,2]>h.filled(np.inf)-1e-5, axis=1)
                nvisible = int(visible.sum())
                score = nvisible*1000-abs(azimuth-225)*10-abs(chord-1500)*.02-abs(offset-150)*.1
                candidates.append(dict(target=target_name, azimuth_deg=float(azimuth), elevation_deg=elevation,
                    target_offset_m=offset, anchor_m=anchor.tolist(), direction=direction.tolist(),
                    sampled_entry_m=float(entry), sampled_exit_m=float(exit), sampled_rock_length_m=float(chord),
                    sampled_air_tail_m=float(tail), sampled_visible_stations=nvisible, score=score,
                    sampled_next_ground_or_boundary_m=float(next_ground),
                    sampled_coverage_min_m=float(samples[indices[0]]), sampled_coverage_max_m=float(samples[indices[-1]])))
candidates.sort(key=lambda x:-x['score'])
save('candidates.json', candidates)
print('Prescreened', len(candidates), 'candidates; best', candidates[:3], flush=True)
assert candidates
preferred = []
for c in candidates:
    entry_point = np.array(c['anchor_m'])+c['sampled_entry_m']*np.array(c['direction'])
    if (entry_point[0]>1000 and entry_point[1]>1000 and 220<=c['azimuth_deg']<=230
            and c['target_offset_m']<=250 and 300<=c['sampled_rock_length_m']<=2500):
        preferred.append(c)
save('preferred_candidates.json', preferred)
for candidate in preferred[:20]:
    anchor = np.array(candidate['anchor_m']); direction = np.array(candidate['direction'])
    intersections = hits(anchor, direction, candidate['sampled_coverage_min_m'], candidate['sampled_coverage_max_m'])
    a = min(intersections, key=lambda x:abs(x[0]-candidate['sampled_entry_m']))
    b = min(intersections, key=lambda x:abs(x[0]-candidate['sampled_exit_m']))
    entry, exit = a[0], b[0]; injection = entry-100
    exact = hits(anchor, direction, injection, 500.)
    if len(exact)!=2 or exact[0][0]!=entry or exact[1][0]!=exit: continue
    point = anchor+exit*direction; after = point+.01*direction
    station_clear = {}
    for s in stations:
        delta = np.array(s['position_enu_m'])-after; length = np.linalg.norm(delta)
        crossed = hits(after, delta/length, 1e-6, length-1e-6)
        station_clear[s['name']] = not crossed
    print('Exact visibility', candidate['target'], candidate['azimuth_deg'], candidate['elevation_deg'], sum(station_clear.values()), flush=True)
    if sum(station_clear.values())<20: continue
    future = hits(anchor, direction, exit+1e-6, candidate['sampled_coverage_max_m'])
    air_end = future[0][0] if future else candidate['sampled_coverage_max_m']
    selected = dict(**candidate, position_m=(anchor+injection*direction).tolist(),
        rock_entry_m=(anchor+entry*direction).tolist(), rock_exit_m=point.tolist(),
        entry_face=a[1], exit_face=b[1], rock_length_m=exit-entry,
        distance_to_array_target_after_exit_m=-exit,
        air_after_exit_within_dem_m=air_end-exit,
        next_ground_or_coverage_point_m=(anchor+air_end*direction).tolist(),
        air_end_reason='terrain' if future else 'DEM coverage',
        real_station_visibility_from_exit=station_clear,
        clear_stations_from_exit=sum(station_clear.values()), axis_intersections_after_injection=exact,
        model='Actual DEM, straight reference axis; station visibility alone is not a refracted-radio or double-bang guarantee.',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
    save('selected.json', selected)
    print('SELECTED', json.dumps(selected), flush=True)
    break
else: raise RuntimeError('No candidate passed exact triangle intersections')
