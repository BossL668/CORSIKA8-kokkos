"""Geographic bounds -> verified DEM/cache -> native scene -> terrain application.

The downloader/height/mesh algorithms are the migrated mountain implementation.
This layer only orchestrates them. It never starts a shower without --run, chooses
an injection point, changes a backend, or invokes a shell.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import yaml

from .geographic_terrain import prepare, read_settings, estimate, read_antennas
from .scene import scene_from_product
from .terrain import sha256_file
from .terrain_download import validate_bounds


def default_cache():
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'corsika8/terrain'


@contextmanager
def preparation_lock(directory, timeout=30):
    """Serialize cooperating preparations; crash releases lock, cache is retained."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.c8-terrain.lock').open('a') as stream:
        deadline=time.monotonic()+timeout
        while True:
            try:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=deadline:
                    raise TimeoutError(f'terrain preparation lock busy: {directory}')
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def write_identical_or_new(path, content):
    """Idempotent admission: do not overwrite a conflicting approved artifact."""
    path=Path(path)
    if path.exists():
        if path.read_text()!=content:
            raise FileExistsError(f'existing artifact differs; choose a new output: {path}')
        return
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-clobber publication, including non-cooperating writers.
        os.link(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def region_settings(bounds, output, cache=None, provider='skadi', spacing_arcsec=3.,
                    bottom_depth_m=500., antennas=None):
    w,s,e,n=validate_bounds(dict(zip(('west','south','east','north'),bounds)))
    output=Path(output).expanduser().resolve()
    if antennas is None:
        # A clearly named demonstration observer, NOT the measured 21CMA array.
        antennas=dict(format='geodetic_points',points=[dict(name='reference_centre',
            longitude_deg=(w+e)/2,latitude_deg=(s+n)/2)],
            height_mode='dem_agl',height_above_ground_m=1.,outside_policy='error')
    return dict(enabled=True,bounds_deg=dict(west=w,south=s,east=e,north=n),
        source=dict(provider=provider),output_directory=str(output/'terrain'),
        cache_directory=str(Path(cache or default_cache()).expanduser().resolve()),
        mesh=dict(spacing_arcsec=spacing_arcsec,bottom_depth_m=bottom_depth_m,
                  boundary_padding_m=1.e-6),medium=dict(density_g_cm3=2.65),antennas=antennas,
        limits=dict(max_tiles=4,max_vertices=120000,max_faces=250000,max_estimated_memory_mb=1024))


def prepare_scene(config, offline=False, estimate_only=False, check_with=None):
    """Reusable API for a YAML terrain request; returns a C++-ready scene path.

    Product retains source hashes; changing bounds/antennas requires a new output.
    Optional check_with invokes the native environment admission executable.
    """
    cfg=read_settings(config)
    plan=estimate(cfg)
    if not cfg['enabled'] or estimate_only:
        return plan
    rows,_=read_antennas(cfg.get('antennas',{}))
    if not rows:
        raise ValueError('scene needs observers: supply geodetic points/CSV/stations')
    product=Path(cfg['output_directory'])
    if Path(cfg['cache_directory']).resolve()==product.parent.resolve():
        raise ValueError('cache directory must differ from product parent (independent locks)')
    with preparation_lock(cfg['cache_directory']), preparation_lock(product.parent):
        result=prepare(config,offline=offline)
        scene=scene_from_product(product)
        scene['geometry']['mesh_path']='terrain_enu.ply'
        # The scene is colocated with its mesh: this pair can move together.
        path=product/'scene.yaml'
        write_identical_or_new(path,yaml.safe_dump(scene,sort_keys=False))
    report=dict(scene=str(path),prepared_product=str(product),reused=result['reused'],
                manifest_sha256=sha256_file(product/'terrain_manifest.yaml'),
                scope='prepared native USStdBK + local rock DEM; radio is not enabled',
                native_geometry_checked=False)
    if check_with:
        executable=resolve_executable(check_with)
        # Environment probe uses exclusive output; each audit gets its own name.
        with tempfile.TemporaryDirectory(prefix='c8-terrain-check-') as temp:
            checked=Path(temp)/'admission.yaml'
            subprocess.run([executable,'--config',str(path),'--output',str(checked)],check=True)
            admission=yaml.safe_load(checked.read_text())
            if admission.get('complete') is not True:
                raise RuntimeError('native environment admission incomplete')
            destination=product/'native_environment_check.yaml'
            with preparation_lock(product.parent):
                write_identical_or_new(destination,checked.read_text())
        report.update(native_geometry_checked=True,native_check=str(destination))
    return report


def resolve_executable(executable):
    resolved=shutil.which(str(executable))
    if not resolved:
        raise FileNotFoundError(f'cannot execute {executable}; build/install the terrain application first')
    return str(Path(resolved).resolve())


def run_scene(scene, application, arguments):
    """Explicit user arguments select CPU/Kokkos, injection and output, unchanged."""
    arguments=list(arguments)
    if any(a=='--scene' or a.startswith('--scene=') for a in arguments):
        raise ValueError('--scene is managed by preparation, do not override it')
    command=[resolve_executable(application),'--scene',str(Path(scene).resolve()),*arguments]
    print('Running terrain application:',json.dumps(command),flush=True)
    return subprocess.run(command,check=False).returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--config',type=Path,help='existing terrain YAML (paths relative to YAML)')
    choice.add_argument('--bounds',type=float,nargs=4,metavar=('WEST','SOUTH','EAST','NORTH'),help='decimal degrees, longitude first')
    parser.add_argument('--output',type=Path,help='new preparation directory, required with --bounds')
    parser.add_argument('--cache',type=Path,help='default XDG_CACHE_HOME/corsika8/terrain')
    parser.add_argument('--provider',choices=['skadi','grand_srtmgl1'],default='skadi')
    parser.add_argument('--spacing-arcsec',type=float,default=3.)
    parser.add_argument('--bottom-depth-m',type=float,default=500.)
    antennas=parser.add_mutually_exclusive_group()
    antennas.add_argument('--antennas-csv',type=Path)
    antennas.add_argument('--station-directory',type=Path,help='original measured 21CMA station-centre files')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--estimate-only',action='store_true',help='no downloads or output creation')
    parser.add_argument('--check-with',help='path/name of c8_terrain_environment (native admission)')
    parser.add_argument('--run',action='store_true',help='explicitly launch terrain transport after preparing')
    parser.add_argument('--application',default='c8_terrain_cascade')
    parser.add_argument('application_args',nargs=argparse.REMAINDER,help='after --, exact terrain application arguments')
    args=parser.parse_args(argv)
    try:
        extra=args.application_args
        if extra[:1]==['--']: extra=extra[1:]
        if extra and not args.run: parser.error('application arguments require --run')
        if args.run and args.estimate_only: parser.error('--run and --estimate-only are incompatible')
        if args.run: resolve_executable(args.application)  # before any download
        if args.check_with: resolve_executable(args.check_with)
        if args.config:
            if any((args.output,args.cache,args.antennas_csv,args.station_directory)):
                parser.error('configure output/cache/antennas inside YAML when using --config')
            if args.provider!='skadi' or args.spacing_arcsec!=3. or args.bottom_depth_m!=500.:
                parser.error('configure provider/spacing/bottom inside YAML when using --config')
            report=prepare_scene(args.config,args.offline,args.estimate_only,args.check_with)
        else:
            if not args.output: parser.error('--bounds requires --output')
            antenna_cfg=None
            if args.antennas_csv or args.station_directory:
                antenna_cfg=dict(height_mode='dem_agl',height_above_ground_m=1.,outside_policy='error')
                if args.antennas_csv:
                    antenna_cfg.update(format='geodetic_csv',path=str(args.antennas_csv.expanduser().resolve()))
                else:
                    antenna_cfg.update(format='21cma_station_centres',station_directory=str(args.station_directory.expanduser().resolve()))
            cfg=region_settings(args.bounds,args.output,args.cache,args.provider,args.spacing_arcsec,args.bottom_depth_m,antenna_cfg)
            plan=estimate(cfg)
            if args.estimate_only:
                report=plan
            else:
                output=args.output.expanduser().resolve()
                with preparation_lock(output):
                    request=output/'terrain_request.yaml'
                    write_identical_or_new(request,yaml.safe_dump(dict(terrain=cfg),sort_keys=False))
                report=prepare_scene(request,args.offline,check_with=args.check_with)
        print(yaml.safe_dump(report,sort_keys=False),flush=True)
        if args.run:
            if 'scene' not in report: raise ValueError('no prepared scene to run')
            return run_scene(report['scene'],args.application,extra)
        return 0
    except (OSError,ValueError,subprocess.CalledProcessError,RuntimeError) as error:
        print(f'terrain workflow: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
