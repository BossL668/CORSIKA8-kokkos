#!/usr/bin/env python3
"""Current-event DEM geometry, transported tau and saved shower deposition."""
import argparse
import hashlib
import json
import os
import pathlib
import socket

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import yaml


def surface(path):
    with path.open('rb') as stream:
        lines = []
        while True:
            line = stream.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                break
        assert 'format binary_little_endian 1.0' in lines
        nv = int(next(line.split()[-1] for line in lines if line.startswith('element vertex ')))
        nf = int(next(line.split()[-1] for line in lines if line.startswith('element face ')))
        vertices = np.fromfile(stream, dtype='<f8', count=nv * 3).reshape(nv, 3)
        faces = np.fromfile(stream, dtype=np.dtype([('count', 'u1'), ('indices', '<u4', (3,))]), count=nf)
    assert np.all(faces['count'] == 3)
    faces = faces['indices'].astype(np.int64)
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    top = faces[cross[:, 2] > 1e-10]
    used = np.unique(top)
    v = vertices[used]
    tri = mtri.Triangulation(v[:, 0], v[:, 1], triangles=np.searchsorted(used, top))
    return v, mtri.LinearTriInterpolator(tri, v[:, 2])


def save(fig, output, name):
    for extension in ['png', 'pdf']:
        fig.savefig(output / (name + '.' + extension), dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('Geometry and numerical plotting run only on PSR')
    root = args.root
    assert (root / 'ALL_NINE_PASSED').exists()
    manifest = json.loads((root / 'campaign.json').read_text())
    out = root / 'report/geometry'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    scene = yaml.safe_load(pathlib.Path(manifest['cases'][0]['command'][5]).read_text())
    mesh = pathlib.Path(scene['geometry']['mesh_path'])
    mesh_hash = hashlib.sha256(mesh.read_bytes()).hexdigest()
    assert mesh_hash == scene['provenance']['mesh_sha256']
    v, height_at = surface(mesh)
    first = yaml.safe_load((root / 'runs' / manifest['cases'][0]['tag'] / 'output/terrain_run.yaml').read_text())
    origin = np.asarray(first['position_m'], dtype=float)
    east = np.linspace(v[:, 0].min(), v[:, 0].max(), 420)
    north = np.linspace(v[:, 1].min(), v[:, 1].max(), 480)
    x, y = np.meshgrid(east, north)
    z = height_at(x, y)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.pcolormesh(x / 1000, y / 1000, z / 1000, shading='auto', cmap='terrain', rasterized=True)
    fig.colorbar(im, ax=ax, label='Surface ENU height (km)')
    ax.plot([origin[0] / 1000] * 2, [origin[1] / 1000, north.min() / 1000], '--', color='#222222', lw=1.2,
            label='Primary axis / section guide')
    ax.annotate('', xy=(origin[0] / 1000, origin[1] / 1000 - 1), xytext=(origin[0] / 1000, origin[1] / 1000),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.6))
    ax.scatter([origin[0] / 1000], [origin[1] / 1000], marker='v', s=85, color='red', label='100 PeV nu_tau injection')
    ax.annotate('Injection', origin[:2] / 1000, xytext=(10, 8), textcoords='offset points')
    for observer in scene['radio']['observers']:
        p = np.asarray(observer['position_enu_m']) / 1000
        ax.scatter(p[0], p[1], marker='^', color='#8c00a0', edgecolors='white', s=85)
        ax.annotate(observer['name'], p[:2], xytext=(8, 5), textcoords='offset points', weight='bold')
    ax.set(xlabel='East (km)', ylabel='North (km)', aspect='equal',
           title='Common input DEM and receiving stations | all 9 events')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=.2)
    save(fig, out / 'figures', 'terrain_and_stations')
    results = []
    for case in manifest['cases']:
        folder = root / 'runs' / case['tag']
        s = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
        assert s['complete'] and s['position_m'] == first['position_m'] and s['direction'] == first['direction']
        vertices = [p for p in s['neutrino']['interactions'] if p['current'] == 'CC' and any(abs(d['pdg']) == 15 for d in p['daughters'])]
        tau_ids = {d['history_id'] for p in vertices for d in p['daughters'] if abs(d['pdg']) == 15}
        all_tau_tracks = json.loads((folder / 'tau_tracks.json').read_text())
        # Scalar stochastic interactions can give the continuing tau a new
        # history ID. Follow its recorded parent links back to the CC tau.
        children = {}
        for track in all_tau_tracks:
            children.setdefault(int(track['parent_history_id']), set()).add(int(track['history_id']))
        pending = list(tau_ids)
        while pending:
            for child in children.get(pending.pop(), ()):
                if child not in tau_ids:
                    tau_ids.add(child)
                    pending.append(child)
        tracks = [t for t in all_tau_tracks if int(t['history_id']) in tau_ids]
        decays = [d for d in s['tau']['decays'] if d['history_id'] in tau_ids]
        assert vertices and tracks and decays
        start = np.array([[float(t[k]) for k in ['x0_m', 'y0_m', 'z0_m']] for t in tracks])
        end = np.array([[float(t[k]) for k in ['x1_m', 'y1_m', 'z1_m']] for t in tracks])
        ds0, ds1 = (origin[1] - start[:, 1]) / 1000, (origin[1] - end[:, 1]) / 1000
        energy0 = np.array([float(t['E0_GeV']) / 1e6 for t in tracks])
        energy1 = np.array([float(t['E1_GeV']) / 1e6 for t in tracks])
        colors = ['#806443' if t['medium'] == 'rock' else '#0085bc' for t in tracks]
        maximum_s = (origin[1] - v[:, 1].min()) / 1000 + .3
        distances = np.linspace(-.2, maximum_s, 2400)
        h = height_at(np.full_like(distances, origin[0]), origin[1] - distances * 1000) / 1000
        arrays = np.load(root / 'report' / (case['tag'] + '_arrays.npz'))
        edges = arrays['deposit_edges_m'] / 1000
        centers = .5 * (edges[1:] + edges[:-1])
        deposition = arrays['deposited_GeV'] / 1e6 / np.diff(edges)
        selected = (centers >= distances[0]) & (centers <= distances[-1])
        shown_fraction = float(np.sum(arrays['deposited_GeV'][selected]) / np.sum(arrays['deposited_GeV']))
        fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[1.2, .75, 1.]))
        axes[0].fill_between(distances, v[:, 2].min() / 1000 - .1, h, color='#c4ac84', alpha=.7)
        axes[0].plot(distances, h, color='#665541', lw=.8)
        cc_distance = (origin[1] - vertices[0]['position_enu_m'][1]) / 1000
        axes[0].plot([0, cc_distance], [origin[2] / 1000, vertices[0]['position_enu_m'][2] / 1000], '--', color='#555555')
        segments = np.stack([np.column_stack([ds0, start[:, 2] / 1000]), np.column_stack([ds1, end[:, 2] / 1000])], axis=1)
        axes[0].add_collection(LineCollection(segments, colors='#154d96', linewidths=1.7))
        esegments = np.stack([np.column_stack([ds0, energy0]), np.column_stack([ds1, energy1])], axis=1)
        axes[1].add_collection(LineCollection(esegments, colors=colors, linewidths=1.3))
        axes[1].set_ylim(0, max(energy0.max(), energy1.max()) * 1.12)
        modes = []
        for vertex in vertices:
            p = np.asarray(vertex['position_enu_m'])
            d = (origin[1] - p[1]) / 1000
            axes[0].scatter(d, p[2] / 1000, marker='*', s=120, color='#be202a', zorder=5)
            axes[2].axvline(d, color='#be202a', ls=':', lw=1)
        for decay in decays:
            p = np.asarray(decay['position_enu_m'])
            d = (origin[1] - p[1]) / 1000
            pdgs = [abs(x['pdg']) for x in decay['daughters']]
            mode = 'muonic' if 13 in pdgs else 'electronic' if 11 in pdgs else 'hadronic'
            modes.append(mode + ' / ' + decay['medium'])
            axes[0].scatter(d, p[2] / 1000, marker='D', s=42, color='#8b3792', zorder=5)
            axes[0].annotate(mode + ' tau decay (' + decay['medium'] + ')', (d, p[2] / 1000),
                             xytext=(8, 18), textcoords='offset points', fontsize=8)
            axes[2].axvline(d, color='#8b3792', ls='--', lw=1)
        axes[0].set(ylabel='ENU height (km)', title='Actual tau tracks and vertices | fixed East = %.3f km' % (origin[0] / 1000))
        axes[0].set_ylim(float(np.ma.min(h)) - .15, max(float(np.ma.max(h)), float(start[:, 2].max() / 1000)) + .2)
        axes[0].legend(handles=[Line2D([], [], color='#665541', label='DEM surface'),
            Line2D([], [], color='#154d96', label='Transported tau'),
            Line2D([], [], marker='*', color='#be202a', ls='', label='CC vertex'),
            Line2D([], [], marker='D', color='#8b3792', ls='', label='Tau decay')], loc='lower left', ncol=4, fontsize=8)
        axes[1].set(ylabel='Tau energy (PeV)', title='Recorded tau transport energy')
        axes[1].legend(handles=[Line2D([], [], color='#806443', label='Tau in rock'),
                               Line2D([], [], color='#0085bc', label='Tau in air')], loc='upper right', fontsize=8)
        axes[2].plot(centers[selected], deposition[selected], color='#246991', lw=1.)
        axes[2].set(xlabel='Southward distance from injection (km)', ylabel='Deposited energy (PeV/km)',
                    title='Shower deposition | 25 m bins | %.5f%% of deposited energy shown' % (shown_fraction * 100))
        for ax in axes:
            ax.set_xlim(distances[0], distances[-1])
            ax.grid(alpha=.2)
        fig.suptitle('%s | seed %d | 100 PeV nu_tau' % (case['label'], case['seed']), fontsize=11)
        save(fig, out / 'figures', case['tag'] + '_geometry_profile')
        projection_offset = float(max(np.abs(start[:, 0] - origin[0]).max(), np.abs(end[:, 0] - origin[0]).max()))
        results.append(dict(tag=case['tag'], label=case['label'], seed=case['seed'], tau_modes=modes,
                            CC_vertices=vertices, tau_decays=decays, tau_segments=len(tracks),
                            fixed_east_m=float(origin[0]), maximum_east_projection_offset_m=projection_offset,
                            displayed_deposition_fraction=shown_fraction))
        np.savetxt(out / (case['tag'] + '_section.csv'), np.column_stack([distances, h.filled(np.nan)]),
                   delimiter=',', header='distance_from_injection_km,DEM_ENU_height_km', comments='')
        print('GEOMETRY', case['tag'], 'maximum projection offset (m)', projection_offset, flush=True)
    (out / 'geometry.json').write_text(json.dumps(dict(mesh=str(mesh), mesh_sha256=mesh_hash,
        section_definition='Fixed-East slice of original upward DEM triangles, no smoothing; tau is projected from actual recorded 3-D segments. These lines are particle tracks, not radio rays.',
        events=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
