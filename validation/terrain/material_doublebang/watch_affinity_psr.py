#!/usr/bin/env python3
"""Independent /proc affinity evidence for an already running detached driver."""
import argparse
import datetime
import json
import os
import pathlib
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('PSR only')
    root = args.root
    cases = json.loads((root / 'campaign.json').read_text())['cases']
    cores = set()
    for cpu in range(256):
        topology = pathlib.Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        cores.add(((topology / 'physical_package_id').read_text().strip(),
                   (topology / 'core_id').read_text().strip()))
    assert len(cores) == 256
    while True:
        progress = json.loads((root / 'progress.json').read_text())
        for case in cases:
            folder = root / 'runs' / case['tag']
            if (folder / 'affinity.json').exists() or not (folder / 'pid').exists():
                continue
            pid = int((folder / 'pid').read_text())
            proc = pathlib.Path('/proc/%d' % pid)
            try:
                command = (proc / 'cmdline').read_bytes().split(b'\0')
                if str(root / 'bundle/c8_terrain_cascade').encode() not in command:
                    continue
                tids = sorted(int(p.name) for p in (proc / 'task').iterdir())
                workers = {str(tid): sorted(os.sched_getaffinity(tid)) for tid in tids}
                good = len(tids) >= 256 and all(len(v) == 1 for v in workers.values())
                good = good and {v[0] for v in workers.values()} == set(range(256))
                if not good:
                    continue
                value = dict(pid=pid, verified=True, observed_threads=len(tids), physical_cores=256,
                    workers=workers, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    thread_names={str(tid): (proc / 'task' / str(tid) / 'comm').read_text().strip() for tid in tids},
                    note='256 distinct physical cores; runtime service threads may share these cores. Actual OpenMP team size is also checked against application execution_concurrency.')
                temp = folder / 'affinity.monitor.tmp'
                temp.write_text(json.dumps(value, indent=2) + '\n')
                temp.replace(folder / 'affinity.json')
                print(case['tag'], 'physical cores=256, total process threads=%d' % len(tids), flush=True)
            except (OSError, ProcessLookupError):
                continue
        if progress['state'] in ['complete', 'failed']:
            return
        driver_pid = int((root / 'driver.pid').read_text())
        try:
            if b'run_psr.py' not in pathlib.Path('/proc/%d/cmdline' % driver_pid).read_bytes():
                raise RuntimeError('Campaign driver is no longer running')
        except FileNotFoundError:
            raise RuntimeError('Campaign driver is no longer running')
        time.sleep(2.)


if __name__ == '__main__':
    main()
