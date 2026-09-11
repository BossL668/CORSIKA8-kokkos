#!/usr/bin/env python3
"""Replay existing default terrain controls without adding diagnostic caps.

Only output/work paths change. Compare all old diagnostic fields and track
columns; new history/energy-audit columns may be added. Never overwrite data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import yaml


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['cpu','openmp','cuda'],required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    for mode in a.modes:
        before=a.baseline/f'magnetic_{mode}_final'
        record=next(r for r in json.loads((before/'acceptance.json').read_text()) if r['case']=='photon_up')
        cmd=record['command'].copy();dest=a.output/mode
        cmd[cmd.index('--output')+1]=str(dest)
        if dest.exists():raise RuntimeError(f'refuse to overwrite {dest}')
        work=a.output/(mode+'_work');work.mkdir(exist_ok=False)
        env=dict(os.environ,OMP_NUM_THREADS='2' if mode=='openmp' else '1',
                 OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
        wrapper=[sys.executable,str(Path(__file__).with_name('run_guarded_diagnostic.py').resolve()),
                 '--output',str(a.output/(mode+'_guard')),'--timeout','360']
        if mode=='cuda':wrapper+=['--gpu']
        subprocess.run(wrapper+['--']+cmd,cwd=work,env=env,check=True)
        old=yaml.safe_load((before/'photon_up/terrain_run.yaml').read_text())
        new=yaml.safe_load((dest/'terrain_run.yaml').read_text())
        x=pd.read_csv(before/'photon_up/terrain/tracks.csv')
        y=pd.read_csv(dest/'terrain/tracks.csv')
        checks=dict(complete=new['complete'],
                    old_diagnostics_unchanged=all(new['diagnostics'].get(k)==v for k,v in old['diagnostics'].items()),
                    old_track_columns_exact=x.equals(y[x.columns]),
                    same_mesh=new['mesh_sha256']==old['mesh_sha256'],
                    no_new_window_survivors=new['diagnostics']['finite_window_survivors']==0)
        result=dict(mode=mode,passed=all(checks.values()),checks=checks,
                    baseline=str(before/'photon_up'),command=cmd,
                    binary_sha256=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest())
        (a.output/(mode+'_comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
