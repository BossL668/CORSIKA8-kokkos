#!/usr/bin/env python3
"""Frozen-before/after single-endpoint outputs and isolated dual N=32 lifecycle."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lifecycle-energy-gev',default='1',
                   help='Use e.g. 100 to exercise both independent queues in every N=32 event')
    p.add_argument('--allow-adaptive-option',action='store_true',
                   help='Allow only the added adaptive CLI option and citation source-line shift')
    p.add_argument('--cooperative-policy',choices=('legacy','adaptive'),default='legacy')
    p.add_argument('--source-root',type=Path,
                   help='Frozen simulator source when using separately versioned diagnostic tools')
    a=p.parse_args()
    if not os.environ.get('FLUPRO'):
        p.error('FLUPRO must point to the existing FLUKA installation')
    root=(a.source_root or Path(__file__).resolve().parents[2]).resolve()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    tools=Path(__file__).resolve().parent
    env=dict(os.environ,OMP_PROC_BIND='false',OMP_NUM_THREADS='20',
             OMP_THREAD_LIMIT='20',OPENBLAS_NUM_THREADS='1',
             CORSIKA_DATA=str(root/'modules/data'))
    records={}
    # The usage prefix normally contains argv[0]; normalize it without
    # rewriting the actual help output, then compare every byte.
    help_text=[]
    for binary in (a.before,a.after):
        r=subprocess.run(['c8_air_shower','--help'],executable=str(binary.resolve()),
                         cwd=root,env=env,capture_output=True,check=True)
        help_text.append(r.stdout)
    if a.allow_adaptive_option:
        help_text=[re.sub(rb'(?m)^\[corsika:info    \(c8_air_shower\.cpp:\d+\)\] (?=Please cite)',
                          b'[corsika:info (source line)] ',text) for text in help_text]
        help_text[1]=re.sub(rb'(?m)^  --kokkos-cooperative-policy[^\n]*\n(?: {4,}\S[^\n]*\n)*',
                            b'',help_text[1])
    records['help-exact']=help_text[0]==help_text[1]
    if not records['help-exact']:raise RuntimeError('CLI help changed')
    def run(label,command,timeout=240):
        r=subprocess.run([sys.executable,str(tools/'run_overlap_guarded.py'),
            '--output',str(out/(label+'-guard')),'--timeout',str(timeout),
            '--rss-limit-gib','3','--sample-resources','--',*command],cwd=root,env=env)
        records[label]=r.returncode
        (out/'status.json').write_text(json.dumps(records,indent=2)+'\n')
        if r.returncode: raise RuntimeError(label+' failed')
    for mode in ('proposal','cuda','openmp'):
        for phase,binary in (('before',a.before),('after',a.after)):
            label=phase+'-'+mode
            cmd=[str(binary.resolve()),'-p','22','-E','1','-N','2','-s','26091021',
                 '-z','0','-a','0','--emthin','1e-6','--antenna-file',
                 str(root/'validation/accelerator/overlap_probe/antennas.txt'),
                 '--geomagnetic-model','IGRF14','--geomagnetic-year','2027',
                 '--verbosity','warn','-f',str(out/label)]
            if mode=='proposal':
                cmd+=['--em-backend','proposal','--radio-backend','cpu']
            else:
                cmd+=['--em-backend','kokkos','--radio-backend','kokkos',
                      '--kokkos-execution',mode,'--kokkos-num-threads',
                      '1' if mode=='cuda' else '20','--gpu-min-batch','16',
                      '--gpu-resident-batch-limit','1024','--gpu-memory-fraction','.1',
                      '--cuda-replay-trace',str(out/(label+'-trace.csv'))]
            run(label,cmd)
        r=subprocess.run([sys.executable,str(tools/'compare_backend_build_outputs.py'),
            str(out/('before-'+mode)),str(out/('after-'+mode)),
            '--report',str(out/(mode+'-comparison.json'))],cwd=root,env=env)
        records[mode+'-compare']=r.returncode
        if r.returncode: raise RuntimeError(mode+' fixed-seed regression')
        if mode!='proposal':
            same=(out/('before-'+mode+'-trace.csv')).read_bytes()==(out/('after-'+mode+'-trace.csv')).read_bytes()
            records[mode+'-trace-exact']=same
            if not same: raise RuntimeError(mode+' decision trace differs')
    cmd=[str(a.after.resolve()),'-p','22','-E',a.lifecycle_energy_gev,'-N','32','-s','26091021',
         '--emthin','1e-6','--em-backend','kokkos','--radio-backend','kokkos',
         '--kokkos-execution','cuda-openmp','--kokkos-num-threads','20',
         '--gpu-min-batch','16','--gpu-resident-batch-limit','4096',
         '--gpu-memory-fraction','.1','--antenna-file',
         str(root/'validation/accelerator/overlap_probe/antennas.txt'),
         '--verbosity','warn','-f',str(out/'independent-N32')]
    if a.cooperative_policy!='legacy':
        cmd+=['--kokkos-cooperative-policy',a.cooperative_policy]
    run('independent-N32',cmd,600)
    (out/'status.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records))

if __name__=='__main__':
    main()
