#!/usr/bin/env python3
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import yaml

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert socket.gethostname()=='psrpku2025'
    spec=importlib.util.spec_from_file_location('boundary_runner',root/'source/validation/terrain/dem_boundary/run_psr.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    results={};out=root/'extra_checks';out.mkdir(exist_ok=True)
    def contents(p):
        return gzip.open(p,'rb').read() if p.suffix=='.gz' else p.read_bytes()
    def launch(tag,command):
        folder=out/tag;folder.mkdir();command=list(command)
        command[command.index('--output')+1]=str(folder/'output')
        (folder/'command.json').write_text(json.dumps(command,indent=2))
        env=os.environ.copy();env['CORSIKA_DATA']=str(runner.OLD/'source/modules/data')
        start=time.monotonic();print('START',tag,flush=True)
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        (folder/'status.json').write_text(json.dumps(dict(code=code,wall_s=time.monotonic()-start)))
        return folder,code
    for mode in ['openmp','cuda']:
        original=root/'runs'/mode/'profile_crossing'
        cmd=json.loads((original/'command.json').read_text())
        folder,code=launch(mode+'_final_binary',cmd);assert code==0
        result=runner.audit(folder)
        for name in ['tracks.csv.gz','deposits.csv.gz','domain_exits.csv.gz','window_survivors.csv.gz']:
            assert contents(folder/'output/terrain'/name)==contents(original/'output/terrain'/name)
        result['all_four_csv_exact_after_fail_fast_guard']=True;results[mode+'_final_binary']=result
        cmd[cmd.index('--track-row-limit')+1]='1'
        folder,code=launch(mode+'_profile_guard',cmd);assert code!=0
        s=yaml.safe_load((folder/'output/terrain_run.yaml').read_text())
        assert not s['complete'] and 'profile row budget exceeded' in (folder/'run.log').read_text()
        results[mode+'_profile_guard']=dict(passed=True,explicit_incomplete=True)
        source=root/'runs'/mode/'contained_off'
        oldcmd=json.loads((source/'command.json').read_text());oldcmd[3]=str(runner.OLD/('build-'+mode)/'applications/c8_terrain_cascade')
        oldcmd.remove('--compress-terrain-csv')
        folder,code=launch(mode+'_accepted_original_binary',oldcmd);assert code==0
        for name in ['tracks.csv','deposits.csv','window_survivors.csv']:
            assert contents(folder/'output/terrain'/name)==contents(source/'output/terrain'/(name+'.gz'))
        results[mode+'_accepted_original_binary']=dict(passed=True,all_three_csv_exact=True)
        (root/'report/extra_checks.json').write_text(json.dumps(results,indent=2))
        print('PASS',mode,'final binary, explicit profile failure, accepted original replay',flush=True)
    cmd=json.loads((root/'runs/openmp/cpu_electron/command.json').read_text())
    cmd[cmd.index('--position-m')+1]='10.1'
    folder,code=launch('outside_injection_rejected',cmd);assert code!=0
    assert 'injection outside DEM coverage' in (folder/'run.log').read_text()
    results['outside_injection_rejected']=dict(passed=True)
    (root/'report/extra_checks.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
