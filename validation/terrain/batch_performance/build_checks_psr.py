#!/usr/bin/env python3
"""Build isolated replay and buffering checks against the candidate libraries."""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert socket.gethostname()=='psrpku2025'
    old=a.root.parent/'beta5_material_models_20260913';build=old/'build-openmp';source=a.root/'source'
    out=a.root/'checks';out.mkdir(exist_ok=True)
    cxx='/home/yuhanglu/miniconda/envs/corsika_venv/bin/g++'
    nvcc='/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/bin/nvcc'
    commands=[]
    def run(cmd,cwd,label):
        commands.append(dict(argv=cmd,cwd=str(cwd),label=label))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        print(label,flush=True)
    for target,relative,name,cuda in [
        ('tests/modules/CMakeFiles/testInterfaceTransport.dir','tests/modules/testInterfaceTransport.cpp','testInterfaceTransport',False),
        ('tests/accelerator/CMakeFiles/testKokkosInterfaceRadioWaveforms.dir','validation/terrain/batch_performance/BufferedRadioChecks.cpp','BufferedRadioChecks',True)]:
        flags={}
        for line in (build/target/'flags.make').read_text().splitlines():
            if ' = ' in line:
                k,v=line.split(' = ',1);flags[k]=shlex.split(v)
        lang='CUDA' if cuda else 'CXX'
        command=([nvcc,'-forward-unknown-to-host-compiler','-ccbin='+cxx] if cuda else [cxx])
        command+=flags[lang+'_DEFINES']+['-I'+str(source)]+flags[lang+'_INCLUDES']+flags[lang+'_FLAGS']
        if cuda:command+=['-x','cu']
        obj=out/(name+'.o');command+=['-c',str(source/relative),'-o',str(obj)]
        run(command,(build/target).parent.parent,name+'-compile')
        link=shlex.split((build/target/'link.txt').read_text())
        original=Path(target).name[:-4]+'.cpp.o'
        link[next(i for i,x in enumerate(link) if x.endswith(original))]=str(obj)
        link[link.index('-o')+1]=str(out/name)
        replacements={n:str(a.root/'build-openmp'/n) for n in ['libCORSIKA8InterfaceEm.a','libCORSIKA8InterfaceRadio.a']}
        link=[replacements.get(Path(x).name,x) for x in link]
        run(link,(build/target).parent.parent,name+'-link')

if __name__=='__main__':main()
