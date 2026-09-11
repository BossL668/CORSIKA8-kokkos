#!/usr/bin/env python3
"""Small CPU/OpenMP interface regressions. No GPU or production job access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--reference-command',type=Path,required=True)
p.add_argument('--stage',choices=['baseline','applications'],required=True)
p.add_argument('--materials',nargs='+',choices=['SiO2','Water','Ice'])
p.add_argument('--timeout',type=float,default=1800.,
               help='Per-event limit, including first-use PROPOSAL cache creation')
a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
base=json.loads(a.reference_command.read_text())['command']
scene=yaml.safe_load(Path(base[base.index('--scene')+1]).read_text())
env=dict(os.environ,OMP_NUM_THREADS='2',OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1',
         FLUPRO=os.environ.get('FLUPRO',str(Path.home()/'fluka')))
guard=Path(__file__).with_name('run_guarded_diagnostic.py').resolve()
results=[]
materials=a.materials or (['SiO2'] if a.stage=='baseline' else ['SiO2','Water','Ice'])
for material in materials:
    for backend in ['proposal','kokkos']:
        tag=f'{a.stage}_{material}_{backend}'
        dest=a.root/tag;config=a.root/(tag+'.yaml')
        current=json.loads(json.dumps(scene))
        if material!='SiO2':
            current['geometry'].update(material=material,rock_density_g_cm3=1. if material=='Water' else .919,
                                       rock_refractive_index=1.33 if material=='Water' else 1.78,
                                       rock_hadronic_target_approximation='H2O_number_fractions')
        config.write_text(yaml.safe_dump(current,sort_keys=False))
        cmd=list(base);cmd[0]=str(a.binary.resolve());cmd.remove('--force-vertex-cc')
        for key,value in {'--scene':str(config.resolve()),'--output':str(dest.resolve()),'--primary':'photon',
                          '--energy-GeV':'.1','--emcut-GeV':'.0005','--emthin':'0',
                          '--max-step-m':'.1','--transport-window-ns':'20','--em-backend':backend,
                          '--threads':'2','--seed':'67101'}.items():
            cmd[cmd.index(key)+1]=value
        print('RUN',tag,flush=True)
        work=a.root/(tag+'_work');work.mkdir(exist_ok=False)
        subprocess.run([sys.executable,str(guard),'--output',str(a.root/(tag+'_guard')),
                        '--timeout',str(a.timeout),'--']+cmd,check=True,env=env,cwd=work)
        s=yaml.safe_load((dest/'terrain_run.yaml').read_text())
        checks=dict(complete=s['complete'],material_mismatches=s['diagnostics']['material_mismatches']==0,
                    no_truncated_tracks=not s['diagnostics']['csv_truncated'])
        if backend=='kokkos':checks['no_pending']=s['accelerator']['pending_particles']==0
        if material=='SiO2' and a.stage=='applications':
            for f in (a.root/f'baseline_SiO2_{backend}'/'terrain').glob('*.csv'):
                checks['unchanged_'+f.name]=hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((dest/'terrain'/f.name).read_bytes()).digest()
        result=dict(tag=tag,command=cmd,checks=checks,diagnostics=s['diagnostics'],material_tables=s.get('material_tables'))
        results.append(result)
        (a.root/(a.stage+'_'+('_'.join(materials))+'_results.json')).write_text(json.dumps(results,indent=2)+'\n')
        if not all(checks.values()):raise RuntimeError(f'{tag}: {checks}')
print('PASS',a.stage,len(results),flush=True)
