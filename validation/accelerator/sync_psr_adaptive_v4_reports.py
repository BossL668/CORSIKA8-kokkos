#!/usr/bin/env python3
"""Copy only small PSR reports/figures through an already authenticated socket."""
import json
from pathlib import Path
import subprocess
import time


def main():
    root=Path('/mnt/d/CorsikaData/corsika_validation_results/psr_t400_adaptive_v4_batching130_20260911')
    root.mkdir(parents=True,exist_ok=True)
    filters=['/MONITOR_REPORT*','/CONFIG.json','/STATUS.json','/CORRECTNESS_GATES.json',
             '/*-resources.png','/*-batch-inputs.png','/PERFORMANCE_RESULT.json','/TIMING*json']
    cmd=['rsync','-a','-e','ssh -S /tmp/c8-psr-adaptive-v4-20260911.sock -o BatchMode=yes -o ConnectTimeout=15']
    cmd += ['--include='+f for f in filters]+['--exclude=*',
        'psrpku2025_PKU:/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v4_batching130_20260911/',str(root)+'/']
    failures=0
    while True:
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
            passed=result.returncode==0
            message=result.stderr.strip()
        except subprocess.TimeoutExpired:
            passed=False;message='report sync timed out; remote simulation unaffected'
        failures=0 if passed else failures+1
        (root/'LOCAL_SYNC_STATUS.json').write_text(json.dumps(dict(
            updated_unix=time.time(),success=passed,consecutive_failures=failures,message=message),indent=2)+'\n')
        if failures>=3:
            raise RuntimeError('report sync stopped after repeated connection failures; remote simulation unaffected')
        if passed:
            try:state=json.loads((root/'STATUS.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):state={}
            if state.get('complete') or 'failed' in state.get('phase',''):
                # Final report writer can lag STATUS by one update interval.
                time.sleep(40)
                subprocess.run(cmd,check=True,timeout=90)
                return
        time.sleep(60)


if __name__=='__main__':main()
