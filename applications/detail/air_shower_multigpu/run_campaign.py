#!/usr/bin/env python3
"""Resumable single-shower multi-GPU ensemble, with continuous resource records.

One coordinator owns all GPUs per shower. Not four independent shower queues.
Never retries failed events silently. COMPLETE is an integrity, not physics gate.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_multigpu as mg

FIELDS = ['index', 'uuid', 'memory.total', 'memory.used', 'memory.free',
          'utilization.gpu', 'utilization.memory', 'power.draw', 'power.limit',
          'temperature.gpu', 'clocks.sm', 'clocks.mem', 'pstate']


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def gpu_inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + ','.join(FIELDS),
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [dict(zip(FIELDS, (v.strip() for v in row))) for row in csv.reader(raw.splitlines())]


def gpu_processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [(row[0].strip(), int(row[1])) for row in csv.reader(raw.splitlines()) if len(row) == 2]


def process_tree(pid):
    records = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            words = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            records[int(path.name)] = dict(ppid=int(words[1]), state=words[0],
                ticks=int(words[11])+int(words[12]), rss=int(words[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        new = {p for p, r in records.items() if r['ppid'] in owned} - owned
        if not new:
            break
        owned.update(new)
    return {p: records[p] for p in owned if p in records}


def terminate_owned(child, event):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    # Also recover worker process groups if the coordinator was killed by OOM.
    for path in event.glob('*/OWNED_PROCESS.json'):
        saved = json.loads(path.read_text())
        try:
            if mg.process_identity(saved['pid']) == saved:
                os.killpg(saved['pid'], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass


def verify_completed(event, config):
    if json.loads((event / 'CONFIG.json').read_text()) != config:
        raise ValueError('Completed configuration changed: ' + str(event))
    done = json.loads((event / 'COMPLETE.json').read_text())
    if not done.get('process_complete'):
        raise ValueError('Not a complete event')
    for name, product in done['products'].items():
        if mg.digest(event / 'merged' / name) != product['sha256']:
            raise ValueError('Completed data changed: ' + name)
    return done


def run(path):
    cfg = json.loads(path.read_text())
    root = Path(cfg['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'MANIFEST.json').exists():
        if json.loads((root / 'MANIFEST.json').read_text()) != cfg:
            raise ValueError('Frozen campaign manifest changed')
    else:
        mg.save(root / 'MANIFEST.json', cfg)
    for name, expected in cfg['frozen_files'].items():
        if mg.digest(name) != expected:
            raise ValueError('Frozen input/executable changed: ' + name)
    seeds = cfg['seeds']
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('Repeated/empty seed list')
    mg.memory_fraction(cfg['template']['gpu_memory_fraction'])
    atomic(root / 'RUNNER.json', dict(**mg.process_identity(os.getpid()), unix=time.time()))
    accepted = []
    with (root / 'telemetry.csv').open('a', buffering=1) as log:
        columns = ['unix', 'elapsed_s', 'seed', 'phase', 'host_available_GiB',
                   'owned_rss_GiB', 'owned_cpu_ticks', 'owned_processes'] + FIELDS
        writer = csv.DictWriter(log, fieldnames=columns)
        if log.tell() == 0:
            writer.writeheader()
        for seed in seeds:
            event = root / ('seed_%d' % seed)
            one = dict(cfg['template'], seed=seed, output=str(event))
            config = root / ('seed_%d.json' % seed)
            if (event / 'COMPLETE.json').exists():
                done = verify_completed(event, one)
                accepted.append(dict(seed=seed, wall_s=done['wall_s'], output=str(event)))
                continue
            if event.exists():
                raise RuntimeError('Partial event preserved; inspect before explicit rerun: ' + str(event))
            if (root / 'STOP_AFTER_EVENT').exists():
                atomic(root / 'STATE.json', dict(stage='stopped_at_boundary', accepted=accepted))
                return
            inventory = gpu_inventory()
            uuids = {r['uuid'] for r in inventory if r['index'] in map(str, one['devices'])
                     or r['uuid'] in one['devices']}
            if len(uuids) != len(one['devices']):
                raise RuntimeError('GPU identity mismatch')
            if any(uuid in uuids for uuid, pid in gpu_processes()):
                raise RuntimeError('Selected GPU has another compute process; will not interfere')
            if any(float(r['memory.free']) / float(r['memory.total']) < .92
                   for r in inventory if r['uuid'] in uuids):
                raise RuntimeError('Selected GPU does not have 92% free VRAM before launch')
            if not config.exists():
                mg.save(config, one)
            elif json.loads(config.read_text()) != one:
                raise ValueError('Event configuration changed')
            start = time.monotonic()
            atomic(root / 'STATE.json', dict(stage='running', current_seed=seed,
                accepted=accepted, updated_unix=time.time()))
            with (root / ('seed_%d.driver.log' % seed)).open('x') as driver:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run_multigpu.py')), str(config)],
                    stdin=subprocess.DEVNULL, stdout=driver, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    tree = process_tree(child.pid)
                    memory = {s.split(':')[0]: int(s.split()[1]) for s in
                              Path('/proc/meminfo').read_text().splitlines() if ':' in s}
                    available = memory['MemAvailable']/1024**2
                    rss = sum(r['rss'] for r in tree.values())/1024**3
                    phase = 'workers' if (event / 'PARTITION.json').exists() else 'prefix'
                    sample = dict(unix=time.time(), elapsed_s=elapsed, seed=seed, phase=phase,
                        host_available_GiB=available, owned_rss_GiB=rss,
                        owned_cpu_ticks=sum(r['ticks'] for r in tree.values()), owned_processes=len(tree))
                    for record in gpu_inventory():
                        if record['uuid'] in uuids:
                            writer.writerow(dict(sample, **record))
                    if available < cfg.get('min_host_available_GiB', 64) or rss > cfg.get('max_owned_rss_GiB', 256):
                        raise RuntimeError('Host memory safety threshold reached')
                    if shutil.disk_usage(root).free/1024**3 < cfg.get('min_disk_free_GiB', 100):
                        raise RuntimeError('Disk safety threshold reached')
                    if elapsed > cfg.get('event_timeout_s', 86400):
                        raise TimeoutError('Full-event wall-time limit')
                    # Re-read ownership AFTER nvidia-smi: workers may have been
                    # spawned after the earlier RSS snapshot.
                    active_gpu_processes = gpu_processes()
                    fresh_tree = process_tree(child.pid)
                    if any(uuid in uuids and pid not in fresh_tree and Path('/proc/%d' % pid).exists()
                           for uuid, pid in active_gpu_processes):
                        raise RuntimeError('Another compute process entered the reserved GPUs')
                    # Polling interval is a monitoring interval, never a physics synchronization.
                    time.sleep(cfg.get('telemetry_interval_s', 5))
                if child.returncode:
                    raise RuntimeError('Coordinator failed with code %d' % child.returncode)
                done = verify_completed(event, one)
                accepted.append(dict(seed=seed, wall_s=done['wall_s'], output=str(event)))
                atomic(root / 'STATE.json', dict(stage='event_complete', accepted=accepted, updated_unix=time.time()))
            except BaseException as exc:
                atomic(root / 'FAILED.json', dict(seed=seed, error=repr(exc), accepted=accepted, unix=time.time()))
                raise
            finally:
                terminate_owned(child, event)
    atomic(root / 'ALL_COMPLETE.json', dict(accepted=accepted, count=len(accepted),
        integrity_complete=True, physics_acceptance=False, unix=time.time()))


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt('Campaign termination requested')
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    run(parser.parse_args().config)
