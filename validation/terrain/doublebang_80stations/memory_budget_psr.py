#!/usr/bin/env python3
"""Admit a CLI-only memory-cap extension and retain the initial failed attempt."""
import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert socket.gethostname()=='psrpku2025'
    lock=(root/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=root.parent/'beta5_dem_boundary_20260914'
    original=(base/'source/applications/c8_terrain_cascade.cpp').read_text()
    actual=(root/'source/applications/c8_terrain_cascade.cpp').read_text()
    old='->check(CLI::Range(32, 16384));';new='->check(CLI::Range(32, 131072));'
    assert original.count(old)==1 and original.replace(old,new)==actual
    libraries={}
    for name in ['libCORSIKA8InterfaceEm.a','libCORSIKA8InterfaceRadio.a']:
        a=digest(root/'build-openmp'/name);assert a==digest(base/'build-openmp'/name);libraries[name]=a
    accepted=base/'runs/openmp/real_air_electron_80stations'
    folder=root/'control_check/runs/memory_budget_80control';folder.mkdir(exist_ok=False)
    command=json.loads((accepted/'command.json').read_text())
    command[3]=str(root/'build-openmp/c8_terrain_cascade')
    command[command.index('--output')+1]=str(folder/'output')
    command[command.index('--device-memory-MiB')+1]='98304'
    (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    env=dict(os.environ,CORSIKA_DATA=str(root.parent/'beta5_material_models_20260913/source/modules/data'))
    with (folder/'run.log').open('x') as log:
        subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    subprocess.run([sys.executable,str(root/'code/audit_psr.py'),'--root',str(root/'control_check'),'--folder',str(folder),'--control'],check=True,
                   env=dict(env,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    csv_hashes={}
    for name in ['tracks.csv.gz','deposits.csv.gz','domain_exits.csv.gz','window_survivors.csv.gz']:
        a=gzip.open(folder/'output/terrain'/name,'rb').read();b=gzip.open(accepted/'output/terrain'/name,'rb').read()
        assert a==b;csv_hashes[name]=hashlib.sha256(a).hexdigest()
    radio={}
    for algorithm in ['CoREAS','ZHS']:
        paths=[f/'output/radio'/algorithm/'field.csv' for f in [folder,accepted]]
        numerator=0.;denominator=0.;count=0
        for a,b in zip(pd.read_csv(paths[0],chunksize=32768),pd.read_csv(paths[1],chunksize=32768)):
            assert np.array_equal(a[['observer','time_s']].to_numpy(),b[['observer','time_s']].to_numpy())
            x=a.iloc[:,2:].to_numpy();y=b.iloc[:,2:].to_numpy();assert np.isfinite(x).all() and np.isfinite(y).all()
            numerator+=np.sum((x-y)**2);denominator+=np.sum(y**2);count+=len(a)
        assert count==80*32768
        relative=float(np.sqrt(numerator/denominator));assert relative<1e-12
        radio[algorithm]=relative
    proof=dict(passed=True,binary_sha256=digest(root/'build-openmp/c8_terrain_cascade'),
        parent_binary_sha256=digest(base/'build-openmp/c8_terrain_cascade'),
        only_cpp_change='CLI --device-memory-MiB upper bound 16384 -> 131072 MiB; no physical library changes.',
        unchanged_physics_libraries=libraries,exact_decompressed_terrain_csv=csv_hashes,
        radio_80_station_relative_l2_to_parent=radio,admitted_budget_MiB=98304,
        scope='Short-window 80-station control validates admission and regression; full 2048 us allocation is monitored during production.')
    (root/'build-openmp/memory_budget_acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
    manifest=json.loads((root/'campaign.json').read_text());progress=json.loads((root/'progress.json').read_text())
    assert not progress['completed'] and not manifest.get('memory_budget_admission')
    revision=root/'revisions/initial_16GiB_cap';revision.mkdir(parents=True)
    for name in ['campaign.json','progress.json','driver.log','driver.pid','launch.json']:
        if (root/name).exists():shutil.copy2(root/name,revision/name)
    tag=manifest['cases'][0]['tag'];failed=root/'runs'/tag
    if failed.exists():
        status=json.loads((failed/'status.json').read_text());assert status['returncode']==105 and not status['complete']
        (root/'failed_attempts').mkdir(exist_ok=True);failed.rename(root/'failed_attempts'/('initial_argument_cap_'+tag))
    binary=root/'bundle/c8_terrain_cascade_128GiB_budget';shutil.copy2(root/'build-openmp/c8_terrain_cascade',binary)
    shutil.copy2(root/'build-openmp/memory_budget_acceptance.json',root/'bundle/memory_budget_acceptance.json')
    for case in manifest['cases']:case['command'][3]=str(binary)
    manifest['boundary_reference_binary_sha256']=manifest['binary_sha256'];manifest['binary_sha256']=proof['binary_sha256']
    manifest['memory_budget_admission']=proof;manifest['revised_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    by_path={r['path']:r for r in manifest['files']}
    for directory in ['code','bundle']:
        for path in (root/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:by_path[str(path)]=dict(path=str(path),sha256=digest(path))
    manifest['files']=list(by_path.values());(root/'campaign.json').write_text(json.dumps(manifest,indent=2)+'\n')
    progress=dict(state='prepared',completed=[],qualified={},total=len(manifest['cases']),updated_utc=manifest['revised_utc'],note='CLI memory cap extended and 80-station regression passed; initial argument-only failure preserved.')
    (root/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print('MEMORY BUDGET ADMITTED',json.dumps(proof),flush=True)


if __name__=='__main__':main()
