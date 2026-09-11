#!/usr/bin/env python3
"""Require a real regenerated low-energy neutrino to mark the run incomplete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve()
command=json.loads((root/'terrain/cpu/nu_cc_command.json').read_text())['command']
output=root/'strict_coverage';work=root/'strict_work';guard=root/'strict_guard'
if output.exists() or work.exists() or guard.exists():raise RuntimeError('refusing overwrite')
work.mkdir();command[command.index('--output')+1]=str(output)
command+=['--require-neutrino-model-coverage']
(root/'strict_command.json').write_text(json.dumps(command,indent=2)+'\n')
wrapper=[sys.executable,str(Path(__file__).with_name('run_guarded_diagnostic.py').resolve()),
         '--output',str(guard),'--timeout','600','--']
result=subprocess.run(wrapper+command,cwd=work,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
s=yaml.safe_load((output/'terrain_run.yaml').read_text())
assert result.returncode!=0 and not s['complete'] and 'outside CTW2011' in s['error']
print('PASS: actual secondary outside the weak model domain rejects shower completion')
