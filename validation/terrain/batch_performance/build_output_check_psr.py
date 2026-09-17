#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
assert socket.gethostname()=='psrpku2025'
old=a.root.parent/'beta5_material_models_20260913';build=old/'build-openmp';source=a.root/'source'
target=build/'applications/CMakeFiles/c8_terrain_cascade.dir';out=a.root/'checks'
flags={}
for line in (target/'flags.make').read_text().splitlines():
    if ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
obj=out/'OutputAuditChecks.o'
cmd=['/home/yuhanglu/miniconda/envs/corsika_venv/bin/g++']+flags['CXX_DEFINES']+['-I'+str(source)]+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+[
    '-c',str(source/'validation/terrain/batch_performance/OutputAuditChecks.cpp'),'-o',str(obj)]
link=shlex.split((target/'link.txt').read_text())
link[next(i for i,x in enumerate(link) if x.endswith('c8_terrain_cascade.cpp.o'))]=str(obj)
link[link.index('-o')+1]=str(out/'OutputAuditChecks')
replacements={n:str(a.root/'build-openmp'/n) for n in ['libCORSIKA8InterfaceEm.a','libCORSIKA8InterfaceRadio.a']}
link=[replacements.get(Path(x).name,x) for x in link]
(out/'output_audit_commands.json').write_text(json.dumps([cmd,link],indent=2))
for command,label in [(cmd,'compile'),(link,'link')]:
    with (out/('output_audit_'+label+'.log')).open('w') as log:
        subprocess.run(command,cwd=build/'applications',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    print(label,flush=True)
