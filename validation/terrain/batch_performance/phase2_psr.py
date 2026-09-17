#!/usr/bin/env python3
"""Complete controls for parallel output audits; never reuse a partial event."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
assert socket.gethostname()=='psrpku2025'
assert json.loads((r/'report/checks.json').read_text())['passed']
assert (r/'checks/run_output_audit_v4/run.log').read_text().count('PASS ')==5
assert 'PASS CPU mesh/tree oracle' in (r/'checks/run_output_audit_v4/dem.log').read_text()
before=r/'before/c8_terrain_cascade';after=r/'candidate-v4/c8_terrain_cascade'
surface=['--energy','.02','--position','397.88140553660566','-2028.7543838949','-146.71048570153665','--direction','0','0','1','--profile']
cases=[
 ('final_electron_1TeV_transport',after,['--energy','1000','--no-radio','--profile','--profile-output']),
 ('final_repeat_electron_1TeV_transport',after,['--energy','1000','--no-radio','--profile']),
 ('final_wide_electron_1TeV_transport',after,['--energy','1000','--no-radio','--profile','--wavefront','65536']),
 ('final_electron_10GeV',after,['--energy','10','--profile']),
 ('before_surface_20MeV',before,surface),
 ('final_surface_20MeV',after,surface)]
manifest=dict(cases=[dict(tag=t,binary=str(b),binary_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),extra=x) for t,b,x in cases])
(r/'phase2_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
progress=dict(state='running',completed=[],total=len(cases))
for tag,binary,extra in cases:
    progress.update(current=tag,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (r/'phase2_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    command=[sys.executable,str(r/'source/validation/terrain/batch_performance/run_case_psr.py'),'--root',str(r),'--binary',str(binary),'--tag',tag]+extra
    with (r/(tag+'.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    print(tag,result.returncode,flush=True)
    if result.returncode:
        progress.update(state='failed',error='Case incomplete: '+tag)
        (r/'phase2_progress.json').write_text(json.dumps(progress,indent=2)+'\n');raise SystemExit(1)
    progress['completed'].append(tag)
progress.update(state='runs_complete',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(r/'phase2_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
