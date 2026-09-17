#!/usr/bin/env python3
"""Fail-closed launch gate for a NEW build; never signal other processes.

Wait for a known earlier benchmark unit to complete successfully, its STATUS
to close, its binary to disappear, and foreign workflows/GPU work to remain
absent for the requested idle window. This is not a global GPU reservation:
another unrelated user may start work after the final observation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def prior_terminal(unit):
    if unit.get('LoadState') != 'loaded':
        raise RuntimeError('prior unit missing or unreadable; not proof of completion')
    if unit.get('MainPID') != '0':
        return False
    if unit.get('ActiveState') in ('activating', 'deactivating', 'reloading'):
        return False
    terminal = (unit.get('ActiveState'), unit.get('SubState')) in (
        ('active', 'exited'), ('inactive', 'dead'), ('failed', 'failed'))
    if not terminal:
        return False
    if unit.get('Result') != 'success' or unit.get('ExecMainStatus') != '0':
        raise RuntimeError('prior benchmark failed; review it before building')
    # An inactive unit that has never executed is not a completed workflow.
    if unit.get('ExecMainCode') != '1' or unit.get('ExecMainStartTimestampMonotonic') in (None, '', '0'):
        raise RuntimeError('prior unit has no successful execution evidence')
    return True


def verify_bundle(bundle):
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    names = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in names:
            raise ValueError('unsafe or duplicate bundle path')
        names.add(row['path'])
        path = bundle / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('bundle file is missing or a symlink: ' + str(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('bundle SHA-256 mismatch: ' + str(relative))
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}
    if actual != names | {'MANIFEST.json'}:
        raise ValueError('unexpected bundle files')
    return hashlib.sha256((bundle / 'MANIFEST.json').read_bytes()).hexdigest()


def observe(args):
    fields = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result',
              'ExecMainStatus', 'ExecMainCode', 'ExecMainStartTimestampMonotonic')
    raw = subprocess.check_output(['systemctl', '--user', 'show', args.wait_unit,
        *['--property=' + f for f in fields]], text=True, timeout=10)
    unit = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    terminal = prior_terminal(unit)
    state = json.loads(args.wait_status.read_text())
    if terminal and state.get('complete') is not True:
        raise RuntimeError('prior unit terminated but benchmark STATUS is incomplete')
    protected = [str(args.wait_binary), *args.wait_workflow_path]
    process_pids = []
    for process in psutil.process_iter(['pid', 'uids', 'cmdline', 'status']):
        if process.pid == os.getpid():
            continue
        try:
            info = process.info
            if info['uids'].real != os.getuid() or info['status'] == psutil.STATUS_ZOMBIE:
                continue
            if any(token in argument for token in protected for argument in info['cmdline']):
                process_pids.append(process.pid)
        except psutil.NoSuchProcess:
            continue
    raw_gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpu = sorted({int(line.strip()) for line in raw_gpu.splitlines() if line.strip()})
    available = psutil.virtual_memory().available
    return dict(systemd=unit, prior_terminal=terminal,
        prior_complete=state.get('complete') is True,
        workflow_pids=sorted(process_pids), gpu_pids=gpu,
        available_bytes=available,
        safe=terminal and state.get('complete') is True and not process_pids and not gpu
             and available >= args.minimum_available_gib * 2**30)


def wait_isolated(args, publish, observe_fn=observe, clock=time.monotonic,
                  sleep=time.sleep):
    deadline = clock() + args.timeout
    idle_since = None
    while True:
        try:
            record = observe_fn(args)
        except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
            record = dict(safe=False, observation_error=repr(error))
        # systemctl/NVML/process inspection can take time. A timestamp from
        # before observation must not extend the idle window or permit a late
        # successful observation to bypass the absolute wait deadline.
        now = clock()
        if record.get('safe') is True:
            if idle_since is None:
                idle_since = now
        else:
            idle_since = None
        idle = now - idle_since if idle_since is not None else 0.
        ready = idle_since is not None and idle >= args.idle_seconds
        expired = now >= deadline
        record.update(phase='timed-out' if expired else 'ready' if ready else 'waiting',
            observed_unix=time.time(), wait_deadline_monotonic=deadline,
            idle_seconds=idle, wait_remaining_seconds=max(0., deadline-now),
            signals_sent=0, launches_started=False)
        publish(record)
        if expired:
            raise TimeoutError('build isolation wait expired; no build started and no signals sent')
        if ready:
            return record
        sleep(min(10., deadline-now))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-unit', required=True)
    p.add_argument('--wait-status', type=Path, required=True)
    p.add_argument('--wait-binary', type=Path, required=True)
    p.add_argument('--wait-workflow-path', action='append', required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--idle-seconds', type=float, default=120)
    p.add_argument('--timeout', type=float, default=32400)
    p.add_argument('--minimum-available-gib', type=float, default=32)
    p.add_argument('--execute', action='store_true', help='After the gate, execute the supplied command; otherwise check once only')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if min(args.idle_seconds, args.timeout, args.minimum_available_gib) <= 0 or not command:
        p.error('positive bounds and a command are required')
    manifest = verify_bundle(args.bundle)
    if not args.execute:
        print(json.dumps(dict(bundle_sha256=manifest, command=command,
                              snapshot=observe(args), launches_started=False), indent=2))
        return
    if args.state.exists():
        raise ValueError('state already exists; choose a fresh queue attempt')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    def publish(record):
        record.update(bundle_sha256=manifest, command=command)
        temporary = args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(args.state)
    try:
        record = wait_isolated(args, publish)
        if verify_bundle(args.bundle) != manifest:
            raise ValueError('bundle manifest changed while waiting')
        # No GPU reservation is claimed; the final snapshot narrows the race.
        if observe(args).get('safe') is not True:
            raise RuntimeError('resources changed at final check; no build started')
        if time.monotonic() >= record['wait_deadline_monotonic']:
            raise TimeoutError('final resource check exceeded build isolation deadline; no build started')
    except Exception as error:
        publish(dict(phase='refused', error=repr(error), signals_sent=0, launches_started=False))
        raise
    record.update(phase='exec', launches_started=True)
    publish(record)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
