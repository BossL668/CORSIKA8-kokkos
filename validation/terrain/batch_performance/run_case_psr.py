#!/usr/bin/env python3
"""Complete low-energy terrain cases, isolated outputs and PSR-only execution."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import yaml

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--energy',type=float,default=1.)
    p.add_argument('--primary',default='electron')
    p.add_argument('--position',type=float,nargs=3,default=[2516.535431729868,3921.7316894076384,500.529492016602])
    p.add_argument('--direction',type=float,nargs=3,default=[0.,-1.,0.])
    p.add_argument('--seed',type=int,default=39101)
    p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--records',type=int)
    p.add_argument('--radio-batch',type=int)
    p.add_argument('--wavefront',type=int)
    p.add_argument('--no-radio',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--profile-output',action='store_true')
    p.add_argument('--output-threads',type=int)
    p.add_argument('--timeout',type=float,default=900.)
    a=p.parse_args()
    assert socket.gethostname()=='psrpku2025','PSR only'
    f=a.root/'runs'/a.tag;f.mkdir(exist_ok=False)
    scene=yaml.safe_load((a.root/'scenes/production_reference.yaml').read_text())
    # Keep optics and every station. Low-energy full event reception spans 128 us.
    scene['radio'].update(enabled=not a.no_radio,samples=32768,memory_MiB=8192)
    (f/'scene.yaml').write_text(yaml.safe_dump(scene,sort_keys=False))
    command=['taskset','-c','0-255',str(a.binary),'--scene',str(f/'scene.yaml'),'--output',str(f/'output'),
        '--primary',a.primary,'--energy-GeV',str(a.energy),'--seed',str(a.seed),
        '--position-m']+list(map(str,a.position))+['--direction']+list(map(str,a.direction))+[
        '--emcut-GeV','.0005','--hadcut-GeV','.3','--mucut-GeV','.3',
        '--emthin','1e-6','--magnetic-field','igrf14','--magnetic-year','2027',
        '--em-backend','kokkos','--threads','256','--batch',str(a.batch),'--device-memory-MiB','16384',
        '--track-row-limit','1000000000','--transport-step-limit','1000000000',
        '--aux-cache','/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913/auxiliary',
        '--energy-ledger','--em-scheduler','resident','--resident-capacity','262144','--compress-terrain-csv']
    for name,value in [('resident-record-capacity',a.records),('radio-batch',a.radio_batch),('resident-wavefront-capacity',a.wavefront),('device-output-threads',a.output_threads)]:
        if value is not None:command.extend(['--'+name,str(value)])
    if a.profile_output:command.append('--profile-device-output')
    (f/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='256',OMP_PROC_BIND='spread',OMP_PLACES='threads',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    env.pop('KOKKOS_TOOLS_LIBS',None)
    if a.profile:
        env['KOKKOS_TOOLS_LIBS']=str(a.root/'tools/libKokkosTiming.so')
        env['C8_BATCH_PROFILE_FILE']=str(f/'kokkos_timing.csv')
    start=time.monotonic();reason=None;peak=0
    (f/'work').mkdir()
    with (f/'run.log').open('x') as log:
        child=subprocess.Popen(command,cwd=f/'work',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (f/'pid').write_text(str(child.pid))
        try:
            while child.poll() is None:
                try:
                    status=Path(f'/proc/{child.pid}/status').read_text()
                    rss=int(next(s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:')))*1024
                    peak=max(peak,rss)
                except (FileNotFoundError,StopIteration):pass
                if time.monotonic()-start>a.timeout:reason='diagnostic time limit';break
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    path=f/'output/terrain_run.yaml';s=yaml.safe_load(path.read_text()) if path.exists() else {}
    result=dict(tag=a.tag,complete=child.returncode==0 and bool(s.get('complete')),returncode=child.returncode,
        stop_reason=reason,wall_s=time.monotonic()-start,peak_rss_GiB=peak/1024**3,
        binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),seed=a.seed,energy_GeV=a.energy,
        emthin=1e-6,automatic_maximum_weight=.5e-6*a.energy,profile=a.profile,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=s)
    (f/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2),flush=True)
    if not result['complete']:raise SystemExit(1)

if __name__=='__main__':main()
