#!/usr/bin/env python3
"""Pause exactly the approved g01 production process trees, with timed recovery.

SIGSTOP preserves all state and files. CUDA allocations remain allocated: test
workers use at most 50% of the REMAINING free VRAM. The independent watchdog
restores SIGCONT after the deadline even if the test/coordinator disconnects.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PRODUCTION = '/home/member/yuhanglu/workspace/c7c8-rad0001-untagged-20260923'

def info(pid):
    base = Path('/proc') / str(pid)
    stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), ppid=int(stat[1]), start=stat[19], state=stat[0],
                command=(base / 'cmdline').read_bytes().replace(b'\0', b' ').decode())

def matches(item):
    try:
        now = info(item['pid'])
        return now['start'] == item['start'] and now['command'] == item['command']
    except (FileNotFoundError, ProcessLookupError):
        return False

def resume(items):
    for item in reversed(items):
        if item['state'] not in ('T', 't') and matches(item):
            os.kill(item['pid'], signal.SIGCONT)

def watchdog(path, timeout):
    lease = Path(path)
    record = json.loads(lease.read_text())
    items = record['processes']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not lease.with_suffix('.released').exists():
        time.sleep(2)
    if lease.with_suffix('.released').exists():
        return  # normal owner restored jobs; don't interfere with a later lease
    if not lease.with_suffix('.released').exists():
        # The normal owner has not returned. Stop ONLY matching test workers
        # before resuming production, including if their coordinator crashed.
        for file in Path(record['test_output']).glob('*/OWNED_PROCESS.json'):
            owned = json.loads(file.read_text())
            try:
                if info(owned['pid'])['start'] == owned['start']:
                    os.killpg(owned['pid'], signal.SIGTERM)
            except (ProcessLookupError, FileNotFoundError):
                pass
        time.sleep(5)
        for file in Path(record['test_output']).glob('*/OWNED_PROCESS.json'):
            owned = json.loads(file.read_text())
            try:
                if info(owned['pid'])['start'] == owned['start']:
                    os.killpg(owned['pid'], signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
    resume(items)

def run(config, lease_path, timeout):
    lease = Path(lease_path)
    records = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid == os.getuid():
                    records.append(info(p.name))
            except (OSError, ValueError):
                pass
    parents = [p for p in records if (PRODUCTION + '/run_qgs78050_c8_gate.py ') in p['command']
               and any((PRODUCTION + '/gpu%d.json' % i) in p['command'] for i in range(4))]
    if len(parents) != 4:
        raise RuntimeError('Expected exactly four approved production supervisors; found %d' % len(parents))
    selected = list(parents)
    ids = {p['pid'] for p in parents}
    while True:
        children = [p for p in records if p['ppid'] in ids and p['pid'] not in ids]
        if not children:
            break
        selected += children
        ids.update(p['pid'] for p in children)
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(int(p.strip()) not in ids for p in gpu_pids.splitlines() if p.strip()):
        raise RuntimeError('Another GPU application is present; will not interfere')
    with lease.open('x') as f:
        json.dump(dict(time=time.time(), processes=selected, deadline_s=timeout,
                       test_output=json.loads(Path(config).read_text())['output']), f, indent=2)
    with lease.with_suffix('.watchdog.log').open('x') as log:
        subprocess.Popen([sys.executable, __file__, '--watchdog', str(lease), '--timeout', str(timeout)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    child = None
    try:
        for item in selected:
            if not matches(item):
                raise RuntimeError('Production process changed before suspension')
            os.kill(item['pid'], signal.SIGSTOP)
        # Do not race a still-running asynchronous GPU kernel.
        time.sleep(3)
        if any(info(p['pid'])['state'] not in ('T', 't') for p in selected):
            raise RuntimeError('Production suspension was not acknowledged')
        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('run_multigpu.py')), config], start_new_session=True)
        return child.wait(timeout=timeout-30)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        resume(selected)
        lease.with_suffix('.released').write_text(str(time.time()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog')
    parser.add_argument('--config')
    parser.add_argument('--lease')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog, args.timeout)
    else:
        def stop(signum, frame):
            raise KeyboardInterrupt('Termination requested')
        signal.signal(signal.SIGTERM, stop)
        sys.exit(run(args.config, args.lease, args.timeout))
