#!/usr/bin/env python3
"""Drain a deliberately suspended pilot coordinator, then resume one mode.

The caller has suspended only the verified coordinator, not its current event.
No shower is signalled. A PID and creation-time match prevents PID-reuse errors.
Run this handoff in a durable service; the active guard remains responsible for
the current event's timeout, memory limits and complete output bookkeeping.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from run_priority_endpoints_acceptance import save, sha


def unit_state(unit):
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Restart'], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-unit', required=True)
    parser.add_argument('--previous-pid', type=int, required=True)
    parser.add_argument('--previous-created', type=float, required=True)
    parser.add_argument('--only-mode', choices=('cuda-openmp', 'openmp-cuda'), required=True)
    parser.add_argument('--standalone', choices=('cuda', 'openmp'), required=True)
    parser.add_argument('--timing-seeds', type=int, default=5)
    parser.add_argument('--libraries', type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve(strict=True)
    tools = Path(__file__).resolve().parent
    identity = json.loads((out/'PROVENANCE.json').read_text())
    binary = out/'binaries/c8_air_shower'
    if sha(binary) != identity['binary_sha256'] or sha(tools/'run_overlap_guarded.py') != identity['guard_sha256']:
        raise RuntimeError('Frozen binary or guard identity changed')
    process = psutil.Process(args.previous_pid)
    if (process.create_time() != args.previous_created or
            not process.cmdline()[1].endswith('/tools/run_priority_endpoints_acceptance.py')):
        raise RuntimeError('Coordinator identity mismatch')
    if process.status() != psutil.STATUS_STOPPED:
        raise RuntimeError('Coordinator must already be deliberately suspended')
    unit = unit_state(args.previous_unit)
    if unit['MainPID'] != str(process.pid) or unit['Restart'] != 'no':
        raise RuntimeError('Unit identity or restart policy mismatch')
    record = dict(previous_unit=args.previous_unit, previous_pid=process.pid,
        previous_created=args.previous_created, only_mode=args.only_mode,
        started_unix=time.time(), binary_sha256=sha(binary),
        scope='scheduler only; allow current event and guard to finish without signals')
    deadline = time.monotonic()+3660
    while True:
        if process.create_time() != args.previous_created or process.status() != psutil.STATUS_STOPPED:
            raise RuntimeError('Paused coordinator changed while draining')
        active = [child for child in process.children(recursive=True)
                  if child.status() != psutil.STATUS_ZOMBIE]
        record.update(phase='draining-current-event', updated_unix=time.time(),
                      active_children=[child.pid for child in active])
        save(out/'MODE_HANDOFF.json', record)
        if not active:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Guard did not finish in bounded drain interval; no shower signalled')
        time.sleep(10)
    # The only remaining task is the stopped Python parent. Its last state is
    # intentionally retained; the continuation adopts the finished guard result
    # before launching anything. SIGKILL prevents resuming its obsolete queue.
    if unit_state(args.previous_unit)['MainPID'] != str(process.pid):
        raise RuntimeError('Unit coordinator changed at handoff')
    process.kill()
    process.wait(timeout=15)
    subprocess.run(['systemctl', '--user', 'stop', args.previous_unit], check=True, timeout=30)
    record.update(phase='scheduler-retired-no-active-children', updated_unix=time.time(),
                  scheduler_signal='SIGKILL (already stopped; no simulation children alive)')
    save(out/'MODE_HANDOFF.json', record)
    command = [sys.executable, str(tools/'resume_priority_endpoints_acceptance.py'),
        '--run', str(out), '--previous-unit', args.previous_unit,
        '--standalone', args.standalone, '--only-mode', args.only_mode,
        '--timing-seeds', str(args.timing_seeds), '--libraries', str(args.libraries),
        '--idle-seconds', '10']
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
