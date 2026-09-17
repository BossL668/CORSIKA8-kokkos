"""Configuration-driven geographic terrain and antenna preparation (no shower).

The generated PLY and YAML use one true geographic ENU frame, not magnetic NWU.
All metre-valued coordinates entering ECEF use WGS84 ellipsoidal heights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pyproj import Transformer
import yaml

from .terrain import build_watertight_solid, projected_to_enu, sha256_file, write_binary_ply
from .terrain_array import enu_rotation, load_station_centres, packed_dms_to_degrees, place_on_dem
from .terrain_download import acquire_hgt, geoid_transform, required_tiles, validate_bounds
from .terrain_validation import _terrain_interpolator, _terrain_triangulation


def read_settings(path):
    path=Path(path).resolve()
    whole=yaml.safe_load(path.read_text()) or {}
    cfg=whole.get('terrain',{})
    if not isinstance(cfg.get('enabled'),bool):
        raise ValueError('terrain.enabled must explicitly be true or false')
    if not cfg['enabled']: return cfg
    cfg=json.loads(json.dumps(cfg))
    def resolve(value):
        p=Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent/p).resolve())
    for key in ('output_directory','cache_directory'):
        if key not in cfg: raise ValueError(f'terrain.{key} must be explicit')
        cfg[key]=resolve(cfg[key])
    cfg.setdefault('source',{})
    cfg['source']['local_hgt_directories']=[resolve(p) for p in cfg['source'].get('local_hgt_directories',[])]
    a=cfg.setdefault('antennas',{})
    for key in ('path','station_directory'):
        if key in a: a[key]=resolve(a[key])
    return cfg


def estimate(cfg):
    if not cfg['enabled']: return dict(status='terrain_disabled',network_requested=False)
    bounds=cfg['bounds_deg'];w,s,e,n=validate_bounds(bounds)
    tiles=required_tiles(bounds)
    provider=cfg.get('source',{}).get('provider','grand_srtmgl1')
    if provider not in ('grand_srtmgl1','skadi'): raise ValueError('unsupported DEM provider')
    if provider=='grand_srtmgl1' and not (-56<=s<n<=60): raise ValueError('outside SRTM latitude coverage')
    spacing=float(cfg.get('mesh',{}).get('spacing_arcsec',1.))
    if not math.isfinite(spacing) or spacing<1: raise ValueError('mesh.spacing_arcsec must be >= 1; no claimed sub-SRTM resolution')
    rows=math.ceil((n-s)*3600/spacing-1e-9)+1
    cols=math.ceil((e-w)*3600/spacing-1e-9)+1
    vertices=rows*cols+2*(rows+cols)-3
    faces=2*(rows-1)*(cols-1)+6*(rows+cols-2)
    # Conservative planning allowance, not an OS-enforced resident-set limit.
    # Includes matplotlib's transient triangle/path objects. The first real
    # 129k-face run used 718 MiB RSS, so an array-only estimate was insufficient.
    memory=512+vertices*5000/1024**2+len(tiles)*26
    limits=cfg.get('limits',{})
    if len(tiles)>int(limits.get('max_tiles',4)): raise ValueError('terrain tile-count budget exceeded')
    if vertices>int(limits.get('max_vertices',120000)): raise ValueError('terrain vertex budget exceeded')
    if faces>int(limits.get('max_faces',250000)): raise ValueError('terrain face budget exceeded')
    if memory>float(limits.get('max_estimated_memory_mb',1024)): raise ValueError('terrain estimated memory budget exceeded')
    if not math.isfinite(float(cfg.get('mesh',{}).get('bottom_depth_m',500))) or float(cfg.get('mesh',{}).get('bottom_depth_m',500))<=0:
        raise ValueError('mesh bottom depth must be finite and positive')
    for name,value in [('density',cfg.get('medium',{}).get('density_g_cm3',2.65)),
                       ('padding',cfg.get('mesh',{}).get('boundary_padding_m',1e-6))]:
        if not math.isfinite(float(value)) or float(value)<=0: raise ValueError(f'{name} must be finite and positive')
    return dict(status='estimate_only',network_requested=False,provider=provider,
        tile_count=len(tiles),grid_shape=[rows,cols],estimated_vertices=vertices,
        estimated_faces=faces,estimated_memory_mb=memory,
        maximum_download_mb=32*len(tiles)+4,
        output_directory=cfg['output_directory'],cache_directory=cfg['cache_directory'])


def read_antennas(settings):
    if not settings: return [],{}
    kind=settings.get('format','geodetic_csv')
    if kind=='21cma_station_centres':
        names,xyz,_,provenance=load_station_centres(settings['station_directory'])
        lon,lat,height=Transformer.from_crs(4978,4979,always_xy=True).transform(*xyz.T)
        rows=[dict(name=name,longitude_deg=float(x),latitude_deg=float(y),
                   measured_height_m=float(h),measured_height_datum='ellipsoidal')
              for name,x,y,h in zip(names,lon,lat,height)]
    elif kind in ('geodetic_csv', 'geodetic_points'):
        if kind == 'geodetic_csv':
            path=Path(settings['path'])
            with path.open() as f: table=list(csv.DictReader(f))
            provenance=dict(path=str(path),sha256=sha256_file(path))
        else:
            table=settings.get('points', [])
            if not isinstance(table, list): raise ValueError('antennas.points must be a list')
            provenance=dict(inline_points=table,
                sha256=hashlib.sha256(json.dumps(table,sort_keys=True).encode()).hexdigest())
        rows=[]
        for row in table:
            lon,lat=[float(row[k]) for k in ('longitude_deg','latitude_deg')]
            if settings.get('angle_format','decimal_degrees')=='packed_dms':
                lon,lat=map(float,packed_dms_to_degrees([lon,lat]))
            elif settings.get('angle_format','decimal_degrees')!='decimal_degrees':
                raise ValueError('unknown angle_format')
            height=row.get('height_m')
            if isinstance(height,str): height=height.strip()
            rows.append(dict(name=row['name'],longitude_deg=lon,latitude_deg=lat,
                measured_height_m=float(height) if height not in (None, '') else None,
                measured_height_datum=settings.get('measured_height_datum','ellipsoidal')))
    else: raise ValueError('unsupported antenna input format')
    if not rows or len({r['name'] for r in rows})!=len(rows): raise ValueError('antenna names must be nonempty and unique')
    for row in rows:
        lon,lat=row['longitude_deg'],row['latitude_deg']
        if not row['name'] or not (-180<=lon<=180 and -90<=lat<=90): raise ValueError('invalid antenna geodetic coordinate/name')
        h=row['measured_height_m']
        if h is not None and not math.isfinite(h): raise ValueError('nonfinite antenna height')
    if 'selected_names' in settings:
        selected=settings['selected_names']
        if not selected or len(set(selected))!=len(selected) or not set(selected)<={r['name'] for r in rows}:
            raise ValueError('unknown, duplicate or empty antenna selection')
        rows=[r for r in rows if r['name'] in selected]
    if len(rows)>int(settings.get('max_count',10000)): raise ValueError('antenna-count budget exceeded')
    return rows,provenance


def position_antennas(rows,settings,bounds,source,geoid,rotation,origin,surface):
    w,s,e,n=validate_bounds(bounds)
    excluded=[r['name'] for r in rows if not(w<=r['longitude_deg']<=e and s<=r['latitude_deg']<=n)]
    policy=settings.get('outside_policy','error')
    if policy not in ('error','exclude'): raise ValueError('unknown outside_policy')
    if excluded and policy=='error': raise ValueError('antennas outside requested DEM bounds: '+', '.join(excluded))
    rows=[r for r in rows if r['name'] not in excluded]
    if not rows: return [],dict(excluded_names=excluded)
    lon=np.array([r['longitude_deg'] for r in rows]);lat=np.array([r['latitude_deg'] for r in rows])
    H=source.sample(lon,lat)
    _,_,h_dem=geoid.transform(lon,lat,H,errcheck=True)
    h_dem=np.asarray(h_dem)
    N=h_dem-H
    mode=settings.get('height_mode','dem_agl')
    raw_h=np.array([np.nan if r['measured_height_m'] is None else r['measured_height_m'] for r in rows])
    for i,r in enumerate(rows):
        if r['measured_height_datum']=='EGM96': raw_h[i]+=N[i]
        elif r['measured_height_datum']!='ellipsoidal': raise ValueError('measured datum must be ellipsoidal or EGM96')
    if mode=='dem_agl':
        agl=float(settings.get('height_above_ground_m',1.))
        if not math.isfinite(agl) or agl<0: raise ValueError('antenna AGL must be finite and nonnegative')
        enu,h=place_on_dem(lon,lat,h_dem,rotation,origin,surface,agl)
    elif mode=='measured':
        if not np.all(np.isfinite(raw_h)): raise ValueError('measured height mode requires height for every antenna')
        h=raw_h
        xyz=np.column_stack(Transformer.from_crs(4979,4978,always_xy=True).transform(lon,lat,h))
        enu=(xyz-origin)@rotation.T
    else: raise ValueError('height_mode must be dem_agl or measured')
    mesh_u=surface(*enu[:,:2].T)
    if np.any(np.ma.getmaskarray(mesh_u)): raise ValueError('antenna outside actual mesh after ENU projection')
    clearance=enu[:,2]-np.asarray(mesh_u)
    if np.any(clearance < -1e-7): raise ValueError('measured antenna below mesh; reconcile height datum/DEM, do not silently move it')
    xyz=enu@rotation+origin
    back=Transformer.from_crs(4978,4979,always_xy=True).transform(*xyz.T)
    xy_error=float(np.max(np.abs(np.array(back[:2])-np.array([lon,lat]))))
    output=[]
    for i,r in enumerate(rows):
        output.append(dict(**r,dem_orthometric_height_m=float(H[i]),geoid_undulation_m=float(N[i]),
            model_ellipsoidal_height_m=float(h[i]),east_m=float(enu[i,0]),north_m=float(enu[i,1]),up_m=float(enu[i,2]),
            mesh_surface_u_m=float(mesh_u[i]),clearance_enu_up_m=float(clearance[i]),height_mode=mode,
            measured_ellipsoidal_minus_dem_m=float(raw_h[i]-h_dem[i]) if np.isfinite(raw_h[i]) else None))
    return output,dict(selected_count=len(output),excluded_names=excluded,
        geodetic_horizontal_roundtrip_max_deg=xy_error,
        clearance_range_m=[float(clearance.min()),float(clearance.max())],
        height_mode=mode,measured_heights_modified=False,
        model_heights_are_measurements=mode=='measured')


def plot_product(out,lon,lat,H,east,north,up,rows):
    tri=_terrain_triangulation(east,north,up)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
    im=ax.pcolormesh(lon,lat,H,shading='auto',cmap='terrain')
    fig.colorbar(im,ax=ax,label='SRTM orthometric height H [m], EGM96')
    bx.tripcolor(tri,up.ravel(),cmap='terrain')
    for arm,color in [('E','orange'),('W','royalblue'),('N','purple'),('S','crimson'),('other','black')]:
        selected=[r for r in rows if (r['name'][:1]==arm if arm!='other' else r['name'][:1] not in 'EWNS')]
        if not selected: continue
        ax.scatter([r['longitude_deg'] for r in selected],[r['latitude_deg'] for r in selected],s=12,c=color,label=f'{arm}: {len(selected)}')
        bx.scatter([r['east_m'] for r in selected],[r['north_m'] for r in selected],s=12,c=color)
    for r in rows:
        if r['name'] in ('E01','E20','W01','W20','N01','N20','S01','S13','S20'):
            bx.annotate(r['name'],(r['east_m'],r['north_m']),xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Longitude [deg]',ylabel='Latitude [deg]',title=f'Requested geographic patch; {len(rows)} antenna centres')
    ax.set_aspect(1/np.cos(np.deg2rad(np.mean(lat))));ax.ticklabel_format(useOffset=False,style='plain');ax.legend(fontsize=8)
    bx.set(xlabel='East [m]',ylabel='North [m]',title='One geographic ENU frame: x=E, y=N, z=U');bx.set_aspect('equal');bx.grid(alpha=.2)
    fig.savefig(out/'01_geographic_and_enu_array.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(12,9),constrained_layout=True)
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    ax.plot_trisurf(tri,up.ravel(),cmap='terrain',alpha=.65,linewidth=0,zorder=0)
    if rows:
        positions=np.array([[r[k] for k in ('east_m','north_m','up_m')] for r in rows])
        ax.scatter(*positions.T,c='magenta',s=18,depthshade=False,zorder=3,label=f'{len(rows)} antenna centres ({rows[0]["height_mode"]})')
        for r in rows:
            if r['name'] in ('E01','E20','W20','N20','S13','S20'):
                ax.text(r['east_m'],r['north_m'],r['up_m']+40,r['name'],fontsize=8,zorder=4)
        ax.legend(loc='upper right')
    ax.set(xlabel='East [m]',ylabel='North [m]',zlabel='Up [m]',title='Downloaded terrain → CORSIKA closed mesh + geodetic antennas\nDisplay vertical scale exaggerated; stored coordinates are metres')
    ax.set_box_aspect([np.ptp(east),np.ptp(north),max(np.ptp(up),.25*np.ptp(north))])
    ax.zaxis.set_major_locator(MaxNLocator(5));ax.view_init(35,-60)
    fig.savefig(out/'02_terrain_mesh_and_antennas_3d.png',dpi=180);plt.close(fig)
    if rows:
        fig,(ax,bx)=plt.subplots(2,1,figsize=(13,7),constrained_layout=True,sharex=True)
        idx=np.arange(len(rows));ax.plot(idx,[r['geoid_undulation_m'] for r in rows],'.-')
        ax.set(ylabel='N = h − H [m]',title='Explicit EGM96 → ellipsoidal height conversion (not an arbitrary offset)')
        bx.plot(idx,[r['clearance_enu_up_m'] for r in rows],'.-',label='Model U − exact PLY surface U')
        bx.axhline(0,c='black',lw=.6);bx.set(ylabel='Clearance [m]');bx.legend(fontsize=8)
        bx.set_xticks(idx[::2],[r['name'] for r in rows[::2]],rotation=90,fontsize=7)
        fig.savefig(out/'03_height_datum_and_mesh_clearance.png',dpi=180);plt.close(fig)


def prepare(config_path,estimate_only=False,offline=False):
    cfg=read_settings(config_path);plan=estimate(cfg)
    if not cfg['enabled'] or estimate_only: return plan
    rows,antenna_provenance=read_antennas(cfg.get('antennas',{}))
    # Fail outside-coverage errors before downloads or writing a product directory.
    w,s,e,n=validate_bounds(cfg['bounds_deg'])
    if cfg.get('antennas',{}).get('outside_policy','error')=='error':
        outside=[r['name'] for r in rows if not(w<=r['longitude_deg']<=e and s<=r['latitude_deg']<=n)]
        if outside: raise ValueError('antennas outside requested bounds: '+', '.join(outside))
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,antennas=antenna_provenance),sort_keys=True).encode()).hexdigest()
    out=Path(cfg['output_directory']);manifest=out/'terrain_manifest.yaml'
    if out.exists():
        if not manifest.exists(): raise FileExistsError(f'incomplete/existing output; use a new output_directory: {out}')
        old=yaml.safe_load(manifest.read_text())
        if old.get('request_sha256')!=fingerprint: raise FileExistsError('terrain request changed; use a new output_directory')
        for filename,digest in old['output_sha256'].items():
            if sha256_file(out/filename)!=digest: raise ValueError(f'terrain product checksum mismatch: {filename}')
        # Source files are checked on reuse as well; no silent stale DEM/geoid.
        for filename,digest in old['input_sha256'].items():
            if sha256_file(Path(filename))!=digest: raise ValueError(f'terrain input changed: {filename}')
        return dict(old,reused=True)
    begin=time.monotonic()
    source,tiles=acquire_hgt(cfg.get('source',{}),cfg['bounds_deg'],cfg['cache_directory'],offline)
    geoid,geoid_meta=geoid_transform(cfg['cache_directory'],offline)
    nrows,ncols=plan['grid_shape']
    lon,lat=np.meshgrid(np.linspace(w,e,ncols),np.linspace(n,s,nrows))
    H=source.sample(lon,lat)
    _,_,h=geoid.transform(lon,lat,H,errcheck=True);h=np.asarray(h)
    origin_cfg=cfg.get('origin',{})
    lat0=float(origin_cfg.get('latitude_deg',(s+n)/2));lon0=float(origin_cfg.get('longitude_deg',(w+e)/2))
    if not (w<lon0<e and s<lat0<n): raise ValueError('ENU origin must be strictly within patch')
    H0=float(source.sample(lon0,lat0))
    _,_,default_h0=geoid.transform(lon0,lat0,H0,errcheck=True)
    h0=float(origin_cfg.get('ellipsoidal_height_m',default_h0))
    if not math.isfinite(h0): raise ValueError('nonfinite origin height')
    east,north,up=projected_to_enu('EPSG:4979',lon,lat,h,lat0,lon0,h0)
    vertices,faces,qa=build_watertight_solid(east,north,up,float(cfg.get('mesh',{}).get('bottom_depth_m',500)))
    top=faces[:qa['top_face_count']]
    normals=np.cross(vertices[top[:,1]]-vertices[top[:,0]],vertices[top[:,2]]-vertices[top[:,0]])
    qa['all_top_normals_up']=bool(np.all(normals[:,2]>0))
    if not qa['watertight'] or qa['degenerate_face_count'] or not qa['all_top_normals_up']:
        raise ValueError('mesh topology/winding validation failed')
    rotation=enu_rotation(lat0,lon0)
    origin=np.array(Transformer.from_crs(4979,4978,always_xy=True).transform(lon0,lat0,h0))
    surface=_terrain_interpolator(east,north,up)
    placed,ant_qa=position_antennas(rows,cfg.get('antennas',{}),cfg['bounds_deg'],source,geoid,rotation,origin,surface)
    xyz=np.column_stack([east.ravel(),north.ravel(),up.ravel()])@rotation+origin
    lon_back,lat_back,h_back=Transformer.from_crs(4978,4979,always_xy=True).transform(*xyz.T)
    roundtrip_deg=float(np.max(np.abs(np.array([lon_back,lat_back])-np.array([lon.ravel(),lat.ravel()]))))
    roundtrip_h=float(np.max(np.abs(np.asarray(h_back)-h.ravel())))
    if roundtrip_deg>1e-9 or roundtrip_h>1e-4: raise ValueError('geographic/ENU round-trip failed')
    out.mkdir(parents=True,exist_ok=False)
    write_binary_ply(out/'terrain_enu.ply',vertices,faces)
    np.savez_compressed(out/'terrain_grid.npz',longitude_deg=lon,latitude_deg=lat,
        orthometric_height_m=H,ellipsoidal_height_m=h,geoid_undulation_m=h-H,
        east_m=east,north_m=north,up_m=up)
    if placed:
        with (out/'antenna_positions_enu.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(placed[0]));writer.writeheader();writer.writerows(placed)
    observers=[dict(name=r['name'],position_enu_m=[r['east_m'],r['north_m'],r['up_m']]) for r in placed]
    (out/'observers.yaml').write_text(yaml.safe_dump(dict(radio=dict(observers=observers)),sort_keys=False))
    geometry=dict(kind='terrain_mesh',mesh_path=str((out/'terrain_enu.ply').resolve()),
        transport_boundary=dict(type='dem_coverage'),
        boundary_padding_m=float(cfg.get('mesh',{}).get('boundary_padding_m',1e-6)),
        rock_density_g_cm3=float(cfg.get('medium',{}).get('density_g_cm3',2.65)),
        material='SiO2',rock_hadronic_target_approximation='sio2_stoichiometric_fluka_qgsjetii',
        rock_refractive_index=2.,air_refractive_index=1.,
        air_world_radius_m=max(100000.,float(np.max(np.linalg.norm(vertices,axis=1)))*1.25))
    fragment=dict(site=dict(latitude_deg=lat0,longitude_deg=lon0,altitude_m=h0),geometry=geometry)
    (out/'corsika_geometry.yaml').write_text(yaml.safe_dump(fragment,sort_keys=False))
    plot_product(out,lon,lat,H,east,north,up,placed)
    source_files={p.get('uncompressed_path',p['path']):p.get('uncompressed_sha256',p['sha256']) for p in tiles}
    source_files[geoid_meta['path']]=geoid_meta['sha256']
    summary=dict(status='geometry_and_antenna_checks_passed',reused=False,
        request_sha256=fingerprint,configuration=cfg,resource_plan=plan,
        runtime_s=time.monotonic()-begin,tiles=tiles,geoid=geoid_meta,input_sha256=source_files,
        vertical_conversion='WGS84 ellipsoidal h = EGM96 orthometric H + geoid undulation N',
        frame=dict(kind='geographic_ENU',origin_latitude_deg=lat0,origin_longitude_deg=lon0,
                   origin_ellipsoidal_height_m=h0,origin_ecef_m=origin.tolist(),ecef_to_enu=rotation.tolist()),
        geoid_undulation_range_m=[float(np.min(h-H)),float(np.max(h-H))],
        mesh_qa=qa,coordinate_qa=dict(roundtrip_max_deg=roundtrip_deg,roundtrip_height_max_m=roundtrip_h),
        antenna_source=antenna_provenance,antenna_qa=ant_qa,
        physics_scope='terrain_geometry_and_antenna_positions_only; no shower/radio acceptance',
        attribution='GRAND SRTMGL1 store / Mapzen Skadi (as configured); NASA/NGA/USGS elevation; NGA EGM96 via OSGeo PROJ-data',
        limitations=['DEM resolution is not survey accuracy; SRTM can represent vegetation/buildings',
            'bottom and vertical side walls are artificial finite-domain closure',
            'dem_agl is a model height, not a measured antenna height',
            'geographic patches only, no antimeridian, polar or >2-degree axis support'],
        output_sha256={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
    manifest.write_text(yaml.safe_dump(summary,sort_keys=False))
    return summary


def prepare_run_config(config_path):
    """Opt-in preprocessor for c8_mountain_shower; stock configurations unchanged.

    Terrain preparation itself does not choose an injection point or enable radio.
    Relative executable paths keep their original configuration-file anchor.
    """
    path=Path(config_path).resolve()
    raw=yaml.safe_load(path.read_text()) or {}
    if 'terrain' not in raw: return str(path)
    if raw['terrain'].get('enabled') is False: return str(path)
    if raw['terrain'].get('enabled') is not True: raise ValueError('terrain.enabled must be boolean')
    if raw.get('event',{}).get('backend')!='corsika':
        raise ValueError('terrain shower requires explicit event.backend: corsika; use c8-mountain-terrain-prepare for geometry only')
    if raw.get('radio',{}).get('solver')!='none':
        raise ValueError('automatic geographic preparation first supports transport only: explicitly set radio.solver: none')
    layered = raw.get('atmosphere',{}).get('model') == 'us_standard_bk'
    if layered:
        executable=raw['event'].get('corsika',{}).get('executable','')
        if Path(executable).name != 'c8_terrain_atmosphere_shower':
            raise ValueError('layered terrain requires independent c8_terrain_atmosphere_shower')
        if 'air_world_radius_m' in raw.get('geometry',{}):
            raise ValueError('remove explicit air_world_radius_m for native layered atmosphere')
    result=prepare(path)
    directory=Path(result['configuration']['output_directory'])
    fragment=yaml.safe_load((directory/'corsika_geometry.yaml').read_text())
    raw.setdefault('geometry',{}).update(fragment['geometry'])
    raw['site']=fragment['site']
    if layered:
        # The reusable geometry fragment also serves uniform-air programs.
        # Only this new target replaces its generated air sphere with native
        # atmospheric layers, and explicitly converts ellipsoidal height to MSL.
        raw['geometry'].pop('air_world_radius_m',None)
        frame=result['frame']
        geoid,_=geoid_transform(result['configuration']['cache_directory'],offline=True)
        _,_,undulation=geoid.transform(frame['origin_longitude_deg'],frame['origin_latitude_deg'],0.,errcheck=True)
        asl=float(frame['origin_ellipsoidal_height_m']-undulation)
        supplied=raw['atmosphere'].get('origin_altitude_asl_m')
        if supplied is not None and (not math.isfinite(float(supplied)) or abs(float(supplied)-asl)>1e-3):
            raise ValueError('atmosphere origin altitude disagrees with DEM ENU/geoid datum')
        raw['atmosphere']['origin_altitude_asl_m']=asl
    raw['radio']['observers']=yaml.safe_load((directory/'observers.yaml').read_text())['radio']['observers']
    executable=raw['event'].get('corsika',{}).get('executable')
    if executable and '/' in executable and not Path(executable).is_absolute():
        raw['event']['corsika']['executable']=str((path.parent/executable).resolve())
    raw['terrain']['enabled']=False
    raw['terrain_preparation']=dict(original_config=str(path),manifest=str(directory/'terrain_manifest.yaml'),
                                   request_sha256=result['request_sha256'])
    encoded=yaml.safe_dump(raw,sort_keys=False)
    digest=hashlib.sha256(encoded.encode()).hexdigest()[:16]
    resolved=directory/f'resolved_run_{digest}.yaml'
    if resolved.exists() and resolved.read_text()!=encoded: raise ValueError('resolved terrain configuration hash conflict')
    if not resolved.exists(): resolved.write_text(encoded)
    return str(resolved)


def main():
    p=argparse.ArgumentParser(description='Download/cache DEM by geographic bounds, build ENU terrain and place antennas')
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--estimate-only',action='store_true',help='no download or output writes')
    p.add_argument('--offline',action='store_true',help='require existing DEM and geoid caches')
    a=p.parse_args()
    result=prepare(a.config,a.estimate_only,a.offline)
    print(yaml.safe_dump(result,sort_keys=False))


if __name__=='__main__': main()
