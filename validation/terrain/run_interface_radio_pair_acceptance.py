#!/usr/bin/env python3
"""Simultaneous CoREAS/ZHS acceptance against c4 single-algorithm outputs; PSR only."""
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
import numpy as np
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
    p.add_argument('--cases',nargs='+',default=['electron_rock','nue_forced'])
    p.add_argument('--variants',nargs='+',default=['resident'])
    p.add_argument('--timeout',type=float,default=3600.)
    p.add_argument('--radio-off',action='store_true')
    args=p.parse_args();args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((args.stage/f'air-standard/acceptance-{args.backend}/acceptance.json').read_text())
    original_scene=args.stage/'psr_inputs/scene.yaml'
    scene=yaml.safe_load(original_scene.read_text())
    mesh=Path(scene['geometry']['mesh_path'])
    if not mesh.is_absolute():scene['geometry']['mesh_path']=str((original_scene.parent/mesh).resolve())
    radio=scene['radio'];radio['enabled']=not args.radio_off;radio.pop('algorithm',None)
    # Keep a real station (which may be shadowed) and add two explicit elevated
    # diagnostics so a downward in-rock shower exercises transmitted signals.
    radio['observers']=[dict(name='diagnostic_overhead',position_enu_m=[0.,0.,1000.]),
        dict(name='diagnostic_offset',position_enu_m=[100.,0.,1000.]),radio['observers'][0]]
    radio.update(samples=65536,start_ns=-1000.,sample_rate_GHz=1.,moment_order=12,
                 subdivision_frequency_GHz=2.,fraunhofer_limit=.025,memory_MiB=384,
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
        if case=='electron_rock' and not args.radio_off:
            cli_scene=copy.deepcopy(scene);cli_scene['radio']['enabled']=False
            selected_scene=args.root/'scene_cli.yaml';selected_scene.write_text(yaml.safe_dump(cli_scene,sort_keys=False))
        for variant in args.variants:
            old=next(r for r in frozen if r['case']==case and r['variant']==variant)
            command=list(old['command']);command[0]=str(binary)
            tag=case+'_'+variant;output=args.root/tag;work=args.root/(tag+'_work');work.mkdir()
            replace(command,'--scene',selected_scene);replace(command,'--output',output);replace(command,'--device-memory-MiB',512)
            if case=='electron_rock' and not args.radio_off:command+=['--radio']
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
            radio_files={};comparison={}
            if not args.radio_off:
                radio_files={str(f.relative_to(output/'radio')):digest(f) for f in (output/'radio').rglob('*') if f.is_file()}
                checks.update(radio_complete=radio_result.get('complete') is True,
                    algorithms=radio_result.get('algorithms')==['CoREAS','ZHS'],
                    radio_device_tracks=radio_result.get('device_tracks',0)>0,
                    one_radio_finalization=radio_result.get('downloads')==1,
                    three_moment_arrays=radio_result.get('moment_arrays')==3,
                    no_radio_error=radio_result.get('errors')==0 and radio_result.get('out_of_window')==0,
                    both_outputs=all(algorithm+'/'+f in radio_files for algorithm in ['CoREAS','ZHS'] for f in ['moments.bin','config.json','field.csv','spectrum.csv']),
                    actual_endpoint_terms=radio_result.get('CoREAS',{}).get('endpoint_contributions',0)>0)
                comparison={};spectra=[]
                for algorithm in ['CoREAS','ZHS']:
                    current=output/'radio'/algorithm
                    reference_manifest=args.stage/'interface-coreas-20260912'/('app-'+algorithm+'-'+args.backend+('-c4' if algorithm=='CoREAS' else '-c2'))/'acceptance.json'
                    reference=next(r for r in json.loads(reference_manifest.read_text()) if (r['case'],r['variant'])==(case,variant))
                    oldroot=Path(reference['radio']['directory'])
                    current_config=json.loads((current/'config.json').read_text());oldconfig=json.loads((oldroot/'config.json').read_text())
                    current_config.pop('maximum_device_bytes');oldconfig.pop('maximum_device_bytes')
                    checks[algorithm+'_same_config']=current_config==oldconfig
                    checks[algorithm+'_same_tracks']=hashes==reference['csv_sha256']
                    current_stats=radio_result[algorithm]
                    keys=['device_tracks','cpu_tracks','track_observer_pairs','paths','direct_paths','transmitted_paths','blocked_paths','leaves','downloads','wavefronts','errors','out_of_window']
                    checks[algorithm+'_same_counters']=all(current_stats[k]==reference['radio'][k] for k in keys)
                    errors={}
                    for filename in ['moments.bin']+(['regularized_moments.bin'] if algorithm=='CoREAS' else []):
                        values=np.fromfile(current/filename,dtype=np.float64);expected=np.fromfile(oldroot/filename,dtype=np.float64)
                        errors[filename]=float(np.linalg.norm(values-expected)/max(np.linalg.norm(expected),1.e-100)) if values.shape==expected.shape else float('inf')
                    checks[algorithm+'_same_moments']=all(v<1.e-10 for v in errors.values())
                    atmosphere,_,_,_,_=check_atmosphere(json.loads((current/'config.json').read_text()))
                    checks[algorithm+'_native_refractivity']=atmosphere['passed']
                    fft=np.genfromtxt(current/'spectrum.csv',delimiter=',',names=True)
                    spectra.append(np.stack([fft[x+'_real']+1j*fft[x+'_imag'] for x in ['Ex','Ey','Ez']]))
                    comparison[algorithm]=errors
                cross=float(np.linalg.norm(spectra[0]-spectra[1])/max(np.linalg.norm(spectra[1]),1.e-100))
                comparison['coreas_zhs_relative_l2']=cross;checks['coreas_zhs_equivalence']=cross<1.e-6
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
                checks['both_use_every_source']=all(radio_result[k]['device_tracks']+radio_result[k]['cpu_tracks']==charged and radio_result[k]['cpu_tracks']==cpu_charged for k in ['CoREAS','ZHS'])
            rows.append(dict(case=case,variant=variant,command=command,binary_sha256=binary_hash,
                checks=checks,comparison=comparison,csv_sha256=hashes,radio_sha256=radio_files,accelerator=accelerated,radio=radio_result,
                diagnostics=summary.get('diagnostics'),material_tables=summary.get('material_tables'),error=summary.get('error')))
            (args.root/'acceptance.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('PASS' if all(checks.values()) else 'FAIL',tag,checks,summary.get('error'),flush=True)
            if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
