#!/usr/bin/env python3
"""Bounded OpenMP-only TAUOLA interface acceptance. No production/GPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();binary=a.binary.resolve();ref=a.reference.resolve();out=a.output.resolve()
    if 'CORSIKA_KOKKOS_BACKEND:STRING=OPENMP\n' not in (binary.parents[1]/'CMakeCache.txt').read_text():
        raise RuntimeError('OpenMP-only build required')
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads((ref/'runs/conditional_cc_command.json').read_text())
    base=original['command'];base[0]=str(binary)
    env=dict(os.environ,OMP_NUM_THREADS='4',OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
    checks={};runs=[]
    guard=Path(__file__).with_name('run_guarded_diagnostic.py').resolve()
    # Gates must fail before creating output or initializing expensive physics.
    for flag in ['--require-complete-weak-transport','--require-event-cc-polarization','--require-tau-depolarization']:
        cmd=list(base);dest=out/flag[2:];cmd[cmd.index('--output')+1]=str(dest);cmd.append(flag)
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=20,env=env)
        (out/(flag[2:]+'.log')).write_text(r.stdout+r.stderr)
        checks[flag]=r.returncode!=0 and not dest.exists() and 'unavailable' in r.stdout+r.stderr
        if not checks[flag]:raise RuntimeError('gate failed: '+flag)
    cmd=list(base);dest=out/'conflicting_spin';cmd[cmd.index('--output')+1]=str(dest)
    cmd+=['--tau-minus-polarization','-.5','--tauola-helicity','left']
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=20,env=env)
    checks['conflicting_spin']=r.returncode!=0 and not dest.exists() and 'select either' in r.stdout+r.stderr
    if not checks['conflicting_spin']:raise RuntimeError('spin selection gate failed')
    for name,pid in [('cc_default','nu_tau'),('tau_minus_partial','tau_minus'),('tau_plus_partial','tau_plus')]:
        cmd=list(base);dest=out/name;cmd[cmd.index('--output')+1]=str(dest)
        cmd[cmd.index('--primary')+1]=pid
        if pid!='nu_tau':
            cmd.remove('--force-vertex-cc');cmd+=['--tau-minus-polarization','-.5']
        work=out/(name+'_work');work.mkdir()
        (out/(name+'_command.json')).write_text(json.dumps(dict(command=cmd),indent=2))
        print('RUN '+name,flush=True)
        subprocess.run([sys.executable,str(guard),'--output',str(out/(name+'_guard')),
                        '--timeout','600','--']+cmd,check=True,env=env,cwd=work)
        s=yaml.safe_load((dest/'terrain_run.yaml').read_text())
        passed=s['complete'] and s['accelerator']['execution_space']=='OpenMP'
        passed=passed and s['diagnostics']['material_mismatches']==0
        if pid!='nu_tau':
            decays=s['tau']['decays'];passed=passed and len(decays)==1
            passed=passed and decays[0]['applied_longitudinal_polarization_at_decay']==(-.5 if pid=='tau_minus' else .5)
            passed=passed and s['tau']['spin_density_matrix_transferred_from_CC'] is False
            passed=passed and s['tau']['energy_loss_depolarization_included'] is False
        else:
            for f in (ref/'runs/conditional_cc/terrain').glob('*.csv'):
                checks['default_CSV_'+f.name]=digest(f)==digest(dest/'terrain'/f.name)
        checks[name]=passed
        runs.append(dict(name=name,complete=s['complete'],tau=s['tau']))
    result=dict(passed=all(checks.values()),checks=checks,runs=runs,binary_sha256=digest(binary),
                reference_binary_sha256=original['binary_sha256'],gpu_used=False,
                scope='decay-polarization interface, gates and unchanged-default regression; NOT full weak/spin physics')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=checks),indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
