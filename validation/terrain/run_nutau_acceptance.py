#!/usr/bin/env python3
"""Reuse the original mountain pilot's DEM, rays, seeds and cuts (radio off).

No production process is stopped. Each child is resource/time guarded. An
existing output is never overwritten, including a failed run. GPU jobs are
serial; an explicit VRAM increment guard is always capped at 10% of the device.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


def load(path):
    return yaml.safe_load(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    previous = args.mountain / 'diagnostic_results/figures/terrain_nutau_v1'
    manifest = load(previous/'manifest.yaml')
    cfg = load(previous/'nutau10tev_selected_cc.yaml')
    mesh = Path(cfg['geometry']['mesh_path'])
    assert sha(mesh) == manifest['mesh_sha256'], 'original DEM hash mismatch'
    atmosphere = cfg['atmosphere']
    altitude = atmosphere['origin_altitude_asl_m']
    scene = dict(schema_version=1, site=dict(
        latitude_deg=cfg['site']['latitude_deg'], longitude_deg=cfg['site']['longitude_deg'],
        origin_ellipsoidal_height_m=cfg['site']['altitude_m'],
        geoid_undulation_m=cfg['site']['altitude_m']-altitude,
        coordinate_frame='geographic_ENU'),
        atmosphere=dict(model='us_standard_bk',origin_altitude_asl_m=altitude,
                        sea_level_refractive_index=1.000327),
        geometry=cfg['geometry'],radio=dict(observers=cfg['radio']['observers']),
        provenance=dict(mesh_sha256=manifest['mesh_sha256'],
                        source=str(previous.resolve()),radio_disabled=True))
    scene['geometry']['mesh_path'] = str(mesh.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    scene_path = args.output/'original_expanded_scene.yaml'
    if scene_path.exists():
        if load(scene_path) != scene:
            raise RuntimeError('existing scene differs; choose new output')
    else:
        scene_path.write_text(yaml.safe_dump(scene, sort_keys=False))
    jobs = []
    for old in manifest['jobs']:
        c = load(previous/(old['name']+'.yaml'))
        e = c['event']
        jobs.append(dict(name=old['name'], role=old['role'],
            primary='nu_tau' if old['primary_pdg']==16 else 'tau_minus',
            energy_GeV=old['energy_GeV'], seed=old['seed'],
            position=e['start_enu_m'], direction=e['direction_enu'],
            emcut=e['em_cut_eV']/1e9, hadcut=e['hadron_cut_eV']/1e9,
            mucut=e['muon_cut_eV']/1e9, emthin=e['thinning_fraction'],
            max_weight=e.get('max_weight',1e30), max_step=e['max_step_m'],
            window_ns=e['transport_window_ns']))
    spec = dict(jobs=jobs, chord=manifest['chord'], mesh_sha256=manifest['mesh_sha256'],
                original=str(previous.resolve()), radio='disabled',
                limitations=['CC-only; no NC or validated tau spin transfer',
                             'selected CC seed is conditioned, not an unbiased efficiency sample',
                             'different schedulers need not generate identical shower trees',
                             'finite window and explicit diagnostic cuts/thinning',
                             'deposition plus surviving energy is not a certified complete rest-mass ledger'])
    path=args.output/'pilot_manifest.yaml'
    if path.exists() and load(path)!=spec:
        raise RuntimeError('manifest mismatch')
    path.write_text(yaml.safe_dump(spec,sort_keys=False))
    return spec


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mountain',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path)
    p.add_argument('--backend',choices=['cpu','openmp','cuda'],default='cpu')
    p.add_argument('--magnetic-field',choices=['none','igrf14'],default='none')
    p.add_argument('--cases',nargs='+')
    p.add_argument('--timeout',type=float,default=360)
    p.add_argument('--label', default='', help='Optional non-overwriting attempt suffix')
    p.add_argument('--compare-group',help='Require exact physical records against an earlier group in this output')
    p.add_argument('--gpu-increment-MiB',type=float,default=512.)
    p.add_argument('--device-memory-MiB',type=int,choices=[128,192,256],default=192,
                   help='Accounted transport allocation; context is additionally monitored by the guard')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.label and not all(c.isalnum() or c in '-_' for c in a.label):
        p.error('--label may contain only letters, digits, hyphens and underscores')
    if a.compare_group and not all(c.isalnum() or c in '-_' for c in a.compare_group):
        p.error('--compare-group must be a group name, not a path')
    spec=prepare(a)
    if a.prepare_only:return
    if not a.binary:p.error('--binary required to run')
    run_root=a.output/(a.backend+'_'+a.magnetic_field+('_'+a.label if a.label else ''))
    run_root.mkdir(exist_ok=True)
    for job in spec['jobs']:
        if a.cases and job['name'] not in a.cases:continue
        dest=run_root/job['name']
        guard=run_root/(job['name']+'_guard')
        if dest.exists() or guard.exists():
            raise RuntimeError(f'output already exists: {dest}')
        work=run_root/(job['name']+'_work')
        work.mkdir(exist_ok=False)  # Isolate Fortran fort.* and timer files.
        command=[str(a.binary.resolve()),'--scene',str(a.output/'original_expanded_scene.yaml'),
                 '--output',str(dest),'--primary',job['primary'],'--energy-GeV',str(job['energy_GeV']),
                 '--seed',str(job['seed']),'--position-m',*map(str,job['position']),
                 '--direction',*map(str,job['direction']),'--emcut-GeV',str(job['emcut']),
                 '--hadcut-GeV',str(job['hadcut']),'--mucut-GeV',str(job['mucut']),
                 '--emthin',str(job['emthin']),'--max-weight',str(job['max_weight']),
                 '--max-step-m',str(job['max_step']),'--transport-window-ns',str(job['window_ns']),
                 '--magnetic-field',a.magnetic_field,'--em-backend','proposal' if a.backend=='cpu' else 'kokkos',
                 '--threads','2' if a.backend=='openmp' else '1',
                 '--batch','64','--device-memory-MiB',str(a.device_memory_MiB),'--track-row-limit','2000000',
                 '--aux-cache',str(a.output/'auxiliary')]
        receipt=dict(command=command,binary_sha256=sha(a.binary),field=a.magnetic_field,
                     backend=a.backend,working_directory=str(work))
        (run_root/(job['name']+'_command.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        wrapper=[sys.executable,str(HERE/'run_guarded_diagnostic.py'),'--output',str(guard),
                 '--timeout',str(a.timeout)]
        if a.backend=='cuda':wrapper+=['--gpu','--gpu-increment-MiB',str(a.gpu_increment_MiB)]
        env=dict(os.environ,OMP_NUM_THREADS='2' if a.backend=='openmp' else '1',
                 OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
        print(f'RUN {a.backend} {a.magnetic_field} {job["name"]}',flush=True)
        subprocess.run(wrapper+['--']+command,env=env,check=True,cwd=work)
        if not load(dest/'terrain_run.yaml')['complete']:
            raise RuntimeError('incomplete terrain event')
        if a.compare_group:
            import pandas as pd
            ref=a.output/a.compare_group/job['name']
            old=load(ref/'terrain_run.yaml');new=load(dest/'terrain_run.yaml')
            checks={key:old[key]==new[key] for key in [
                'seed','primary','energy_GeV','emcut_GeV','hadcut_GeV','mucut_GeV',
                'emthin','max_weight','max_step_m','transport_window_ns','mesh_sha256',
                'air_magnetic_field_enu_T','rock_magnetic_field_enu_T','diagnostics','neutrino','tau']}
            for name in ['tracks.csv','deposits.csv','window_survivors.csv']:
                checks[name]=pd.read_csv(ref/'terrain'/name).equals(pd.read_csv(dest/'terrain'/name))
            comparison=dict(reference=str(ref),run=str(dest),checks=checks,passed=all(checks.values()))
            (run_root/(job['name']+'_comparison.json')).write_text(json.dumps(comparison,indent=2)+'\n')
            print('rebuilt physical records exact:',comparison['passed'],flush=True)
            if not comparison['passed']:raise RuntimeError('rebuild physical regression failed')


if __name__=='__main__':main()
