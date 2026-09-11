#!/usr/bin/env python3
"""Three sequential, one-seed full-radio pilots, not a statistical benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only', choices=('old-cooperative20','independent20','single-cuda'),
                   help='Run one arm in a fresh output directory; do not overwrite earlier pilots')
    p.add_argument('--seed',type=int,help='Override the reference seed for a new pilot')
    p.add_argument('--energy-gev', default='1e5')
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--sample-threads', action='store_true')
    p.add_argument('--resume',action='store_true',
                   help='Reuse only closed outputs with an exactly matching recorded command')
    a=p.parse_args()
    if not os.environ.get('FLUPRO'):p.error('FLUPRO is required')
    root=Path(__file__).resolve().parents[2]
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=a.resume)
    old=json.loads(a.reference_manifest.read_text())
    template=old['command']
    results={}
    for label,binary,mode,threads in (
        ('old-cooperative20',a.before,'cuda-openmp',20),
        ('independent20',a.after,'cuda-openmp',20),
        ('single-cuda',a.after,'cuda',1)):
        if a.only and a.only!=label: continue
        command=list(template);command[0]=str(binary.resolve())
        for flag,value in (('-E',a.energy_gev),('-f',str(out/label)),
                           ('--kokkos-execution',mode),('--kokkos-num-threads',str(threads))):
            command[command.index(flag)+1]=value
        if a.seed is not None:command[command.index('-s')+1]=str(a.seed)
        binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
        env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),
                 OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1',
                 CORSIKA_DATA=str(root/'modules/data'))
        saved=out/(label+'-guard/summary.json')
        provenance=out/(label+'-command.json')
        if provenance.exists():
            recorded=json.loads(provenance.read_text())
            assert a.resume and recorded['command']==command
            assert recorded['binary_sha256']==binary_hash, 'binary differs from saved pilot'
        else:
            provenance.write_text(json.dumps(dict(command=command,binary_sha256=binary_hash,
                reference_manifest=str(a.reference_manifest.resolve())),indent=2)+'\n')
        if a.resume and saved.exists():
            previous=json.loads(saved.read_text())
            assert previous['pass'] and previous['command']==command
            r=subprocess.CompletedProcess(command,0)
        else:
            r=subprocess.run([sys.executable,str(Path(__file__).with_name('run_overlap_guarded.py')),
                '--output',str(out/(label+'-guard')),'--timeout',str(a.timeout),
                '--rss-limit-gib','4','--sample-resources',
                *(['--sample-threads'] if a.sample_threads else []),'--',*command],cwd=root,env=env)
        monitor=json.loads((out/(label+'-guard/summary.json')).read_text())
        result=dict(returncode=r.returncode,monitor=monitor,
                    binary_sha256=binary_hash)
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==binary_hash, 'binary changed during pilot'
        if r.returncode==0:
            timing=yaml.safe_load((out/label/'simulation_timing/summary.yaml').read_text())
            gpu=yaml.safe_load((out/label/'gpu_em/summary.yaml').read_text())
            assert len(timing)==len(gpu)==1
            assert all(x['closed'] for x in timing.values())
            assert all(x['complete'] for x in gpu.values())
            result['shower_wall_s']=next(iter(timing.values()))['wall_time_ms']/1000
            result['physical_statistics']=next(iter(gpu.values()))['statistics']
        results[label]=result
        (out/'timing.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
        if r.returncode:raise RuntimeError(label+' failed; pilot stopped')
    print(json.dumps({k:dict(shower_s=v['shower_wall_s'],process_s=v['monitor']['elapsed_s'])
                      for k,v in results.items()},indent=2))

if __name__=='__main__':main()
