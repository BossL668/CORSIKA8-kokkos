#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args()
    assert socket.gethostname()=='psrpku2025'
    r=a.root;report=r/'report/checks.json';results=[]
    env=dict(os.environ,CORSIKA_DATA=str(r.parent/'beta5_material_models_20260913/source/modules/data'),C8_INTERFACE_TEST_THREADS='256')
    tasks=[('build',[sys.executable,str(r/'source/validation/terrain/batch_performance/build_checks_psr.py'),'--root',str(r)],{}),
        ('buffered_radio',[str(r/'checks/BufferedRadioChecks')],{}),
        ('resident_reference',[str(r/'checks/testInterfaceTransport'),'ordinary'],{}),
        ('resident_wide_reference',[str(r/'checks/testInterfaceTransport'),'ordinary'],{'C8_INTERFACE_TEST_WIDE_FRONT':'16384'})]
    for name,command,extra in tasks:
        start=time.monotonic();folder=r/'checks'/('run_'+name);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.log').open('w') as log:
            try:
                result=subprocess.run(command,cwd=folder,env=dict(env,**extra),stdout=log,stderr=subprocess.STDOUT,timeout=900)
                code=result.returncode
            except subprocess.TimeoutExpired:code=-999
        results.append(dict(name=name,command=command,returncode=code,wall_s=time.monotonic()-start,passed=code==0))
        report.write_text(json.dumps(dict(complete=len(results)==len(tasks),passed=all(x['passed'] for x in results),results=results),indent=2)+'\n')
        print(name,code,flush=True)
        if code:raise SystemExit(1)
    if a.benchmark:
        subprocess.run([sys.executable,str(r/'code/benchmark_psr.py'),'--root',str(r)],check=True)

if __name__=='__main__':main()
