#!/usr/bin/env python3
"""Recheck the complete terminal segment at the real DEM perimeter."""
import argparse
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
assert socket.gethostname()=='psrpku2025'
while True:
    state=json.loads((r/'phase2_progress.json').read_text())['state']
    if state=='runs_complete':break
    assert state=='running',state
    time.sleep(5)
old=r.parent/'beta5_dem_boundary_20260914/runs/openmp/real_air_electron_80stations/command.json'
command=json.loads(old.read_text())
def triple(flag):
    i=command.index(flag);return command[i:i+4]
position=triple('--position-m')[1:];direction=triple('--direction')[1:]
for tag,binary in [('before_boundary_1GeV',r/'before/c8_terrain_cascade'),('final_boundary_1GeV',r/'candidate-v4/c8_terrain_cascade')]:
    call=[sys.executable,str(r/'source/validation/terrain/batch_performance/run_case_psr.py'),'--root',str(r),'--tag',tag,'--binary',str(binary),'--energy','1','--position']+position+['--direction']+direction+['--profile']
    with (r/(tag+'.log')).open('x') as log:subprocess.run(call,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(tag,'complete',flush=True)
(r/'boundary_progress.json').write_text(json.dumps(dict(state='runs_complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
