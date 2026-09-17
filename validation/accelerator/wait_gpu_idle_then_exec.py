#!/usr/bin/env python3
"""Queue an isolated GPU test without stopping unrelated work.

Optional workflow PID includes its creation timestamp, so a reused PID cannot
hold the queue indefinitely. GPU observation failures never count as idle.
This is a launch guard, not a GPU reservation or a restart policy.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil


def gpu_processes():
    # User systemd services need not inherit WSL's interactive PATH.
    executable = shutil.which('nvidia-smi')
    if executable is None and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        executable = '/usr/lib/wsl/lib/nvidia-smi'
    if executable is None:
        raise FileNotFoundError('nvidia-smi is absent from PATH and the WSL driver directory')
    result = subprocess.run(
        [executable, '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15, check=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def same_live_process(pid, created):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time()-created) < .001 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--wait-created', type=float)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--idle-seconds', type=float, default=10)
    parser.add_argument('--minimum-available-gib', type=float, default=32)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (args.wait_pid is None) != (args.wait_created is None):
        parser.error('--wait-pid and --wait-created must be specified together')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.idle_seconds < 0 or args.minimum_available_gib < 0:
        parser.error('a command and nonnegative safety settings are required')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    idle_since = None
    last_log = 0
    while True:
        now = time.monotonic()
        record = dict(phase='waiting', time=time.time(), pid=os.getpid(), command=command)
        try:
            alive = args.wait_pid is not None and same_live_process(args.wait_pid, args.wait_created)
            pids = gpu_processes()
            available = psutil.virtual_memory().available / 2**30
            record.update(workflow_live=alive, gpu_pids=pids, available_gib=available)
            safe = not alive and not pids and available >= args.minimum_available_gib
        except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
            safe = False
            record['observation_error'] = repr(error)
        if safe:
            if idle_since is None:
                idle_since = now
            record['idle_seconds'] = now-idle_since
        else:
            idle_since = None
        ready = idle_since is not None and now-idle_since >= args.idle_seconds
        if ready:
            record['phase'] = 'exec'
        tmp = args.state.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(args.state)
        if ready or now-last_log >= 30:
            print(json.dumps(record), flush=True)
            last_log = now
        if ready:
            os.execvp(command[0], command)
        time.sleep(2)


if __name__ == '__main__':
    main()
