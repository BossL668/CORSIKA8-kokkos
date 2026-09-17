#!/usr/bin/env python3
"""Run interface radio on PSR using frozen accepted shower commands and hashes."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from validate_interface_radio_atmosphere import check as check_atmosphere

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def replace(command,key,value):
    command[command.index(key)+1]=str(value)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--backend',choices=['openmp','cuda'],required=True)
    p.add_argument('--cases',nargs='+',default=['photon_up','photon_down','electron_rock','nue_forced'])
    p.add_argument('--variants',nargs='+',default=['batched','resident'])
    p.add_argument('--timeout',type=float,default=3600.)
    p.add_argument('--radio-off',action='store_true')
    args=p.parse_args();args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((args.stage/f'air-standard/acceptance-{args.backend}/acceptance.json').read_text())
    original_scene=args.stage/'psr_inputs/scene.yaml'
    scene=yaml.safe_load(original_scene.read_text())
    mesh=Path(scene['geometry']['mesh_path'])
    if not mesh.is_absolute():scene['geometry']['mesh_path']=str((original_scene.parent/mesh).resolve())
    radio=scene['radio'];radio['enabled']=not args.radio_off
    # Keep a real station (which may be shadowed) and add two explicit elevated
    # diagnostics so a downward in-rock shower exercises transmitted signals.
    radio['observers']=[dict(name='diagnostic_overhead',position_enu_m=[0.,0.,1000.]),
        dict(name='diagnostic_offset',position_enu_m=[100.,0.,1000.]),radio['observers'][0]]
    radio.update(samples=65536,start_ns=-1000.,sample_rate_GHz=1.,moment_order=12,
                 subdivision_frequency_GHz=2.,fraunhofer_limit=.025,memory_MiB=256,
                 air_index_model='native',index_table_step_m=10.,optical_integration_samples=64)
    radio['maximum_subdivision_depth']=20
    scene_path=args.root/'scene.yaml';scene_path.write_text(yaml.safe_dump(scene,sort_keys=False))
    binary=args.stage/f'build-{args.backend}/applications/c8_terrain_cascade'
    guard=args.stage/'source/validation/terrain/run_interface_radio_guard.py'
    env=dict(os.environ,OMP_NUM_THREADS='256',OMP_PROC_BIND='spread',OMP_PLACES='threads',OPENBLAS_NUM_THREADS='1')
    rows=[]
    for case in args.cases:
        selected_scene=scene_path
        if case=='nue_forced':
            # The frozen full shower's latest t + n_max R/c is 1.501941 ms.
            # 196608 samples at 128 MHz cover 1.536 ms while fitting the
            # explicitly owned array budget with a bounded radio runtime guard.
            wide=copy.deepcopy(scene);wide['radio'].update(samples=196608,sample_rate_GHz=.128,subdivision_frequency_GHz=.256)
            wide['radio']['observers']=wide['radio']['observers'][:1]
            selected_scene=args.root/'scene_nue.yaml';selected_scene.write_text(yaml.safe_dump(wide,sort_keys=False))
        for variant in args.variants:
            old=next(r for r in frozen if r['case']==case and r['variant']==variant)
            command=list(old['command']);command[0]=str(binary)
            tag=case+'_'+variant;output=args.root/tag;work=args.root/(tag+'_work');work.mkdir()
            replace(command,'--scene',selected_scene);replace(command,'--output',output);replace(command,'--device-memory-MiB',512 if case=='nue_forced' else 256)
            cmd=[sys.executable,str(guard),'--output',str(args.root/(tag+'_guard')),'--timeout',str(args.timeout)]
            if args.backend=='cuda':cmd.append('--gpu')
            print('RUN',args.backend,tag,'radio',not args.radio_off,flush=True)
            binary_hash=digest(binary)
            status=subprocess.run(cmd+['--']+command,cwd=work,env=env)
            summary=yaml.safe_load((output/'terrain_run.yaml').read_text()) if (output/'terrain_run.yaml').exists() else {}
            summary=json.loads(json.dumps(summary))
            hashes={f.name:digest(f) for f in (output/'terrain').glob('*.csv')}
            accelerated=summary.get('accelerator',{});radio_result=summary.get('radio_result',{})
            checks=dict(binary_unchanged=digest(binary)==binary_hash,completed=status.returncode==0 and summary.get('complete') is True,
                no_pending=accelerated.get('pending_particles')==0,
                backend=accelerated.get('execution_space')==('OpenMP' if args.backend=='openmp' else 'Cuda'),
                frozen_full_csv=hashes==old['csv_sha256'],
                frozen_diagnostics=summary.get('diagnostics')==old['diagnostics'],
                frozen_material_tables=summary.get('material_tables')==old['material_tables'],
                frozen_fallbacks=accelerated.get('specified_cpu_fallbacks')==old['accelerator']['specified_cpu_fallbacks'])
            radio_files={}
            if not args.radio_off:
                radio_files={f.name:digest(f) for f in (output/'radio').glob('*') if f.is_file()}
                checks.update(radio_complete=radio_result.get('complete') is True,
                    radio_device_tracks=radio_result.get('device_tracks',0)>0,
                    radio_paths=radio_result.get('paths',0)>0,
                    one_radio_download=radio_result.get('downloads')==1,
                    no_radio_error=radio_result.get('errors')==0 and radio_result.get('out_of_window')==0,
                    radio_outputs=all(k in radio_files for k in ['moments.bin','config.json','field.csv','spectrum.csv']))
                if checks['completed'] and (output/'radio/config.json').exists():
                    atmosphere,_,_,_,_=check_atmosphere(json.loads((output/'radio/config.json').read_text()))
                    checks['native_refractivity_normalization']=atmosphere['passed']
                # Charge inventory is independent of the new device counters.
                charged_pdg={11,13,15,211,321,2212,3222,3112,3312,3334}
                charged=0;cpu_charged=0
                if checks['completed']:
                    with (output/'terrain/tracks.csv').open() as stream:
                        for track in csv.DictReader(stream):
                            pdg=abs(int(track['pdg']))
                            nonzero=any(float(track[a])!=float(track[b]) for a,b in [('x0_m','x1_m'),('y0_m','y1_m'),('z0_m','z1_m')])
                            if (pdg in charged_pdg or pdg>=1000000000) and nonzero and float(track['t1_s'])>float(track['t0_s']) and float(track['weight'])>0:
                                charged+=1;cpu_charged+=pdg!=11
                checks['charged_sources_once']=checks['completed'] and radio_result.get('device_tracks',0)+radio_result.get('cpu_tracks',0)==charged
                checks['cpu_sources_once']=checks['completed'] and radio_result.get('cpu_tracks',0)==cpu_charged
            rows.append(dict(case=case,variant=variant,command=command,binary_sha256=binary_hash,
                checks=checks,csv_sha256=hashes,radio_sha256=radio_files,accelerator=accelerated,radio=radio_result,
                diagnostics=summary.get('diagnostics'),material_tables=summary.get('material_tables'),error=summary.get('error')))
            (args.root/'acceptance.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('PASS' if all(checks.values()) else 'FAIL',tag,checks,summary.get('error'),flush=True)
            if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
