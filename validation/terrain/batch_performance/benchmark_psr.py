#!/usr/bin/env python3
"""Sequential complete-shower controls after all scheduler/radio checks pass."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert socket.gethostname()=='psrpku2025';r=a.root
    checks=json.loads((r/'report/checks.json').read_text())
    assert checks['complete'] and checks['passed']
    runner=r/'source/validation/terrain/batch_performance/run_case_psr.py'
    before=r/'before/c8_terrain_cascade';after=r/'candidate-v1/c8_terrain_cascade'
    after.parent.mkdir(exist_ok=True)
    if not after.exists():shutil.copy2(r/'build-openmp/c8_terrain_cascade',after)
    assert hashlib.sha256(after.read_bytes()).digest()==hashlib.sha256((r/'build-openmp/c8_terrain_cascade').read_bytes()).digest()
    # Explicit legacy settings give the candidate an exact before/after control.
    cases=[
      ('candidate_legacy_electron_1GeV',after,['--profile','--records','4096','--radio-batch','0']),
      ('candidate_electron_1GeV',after,['--profile']),
      ('before_electron_10GeV',before,['--energy','10','--profile']),
      ('candidate_electron_10GeV',after,['--energy','10','--profile']),
      ('before_electron_1TeV_transport',before,['--energy','1000','--no-radio','--profile']),
      ('candidate_electron_1TeV_transport',after,['--energy','1000','--no-radio','--profile']),
      ('candidate_wide_electron_1TeV_transport',after,['--energy','1000','--no-radio','--profile','--wavefront','65536']),
      ('before_surface_20MeV',before,['--energy','.02','--position','397.88140553660566','-2028.7543838949','-146.71048570153665','--direction','0','0','1','--profile']),
      ('candidate_surface_20MeV',after,['--energy','.02','--position','397.88140553660566','-2028.7543838949','-146.71048570153665','--direction','0','0','1','--profile'])]
    manifest=dict(before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),cases=[dict(tag=t,binary=str(b),extra=x) for t,b,x in cases])
    (r/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    progress=dict(state='running',completed=[],total=len(cases))
    for tag,binary,extra in cases:
      progress.update(current=tag,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
      (r/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
      command=[sys.executable,str(runner),'--root',str(r),'--binary',str(binary),'--tag',tag]+extra
      with (r/(tag+'.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
      print(tag,result.returncode,flush=True)
      if result.returncode:
        progress.update(state='failed',error='Case incomplete: '+tag)
        (r/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');raise SystemExit(1)
      progress['completed'].append(tag)
    progress.update(state='runs_complete',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (r/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')

if __name__=='__main__':main()
