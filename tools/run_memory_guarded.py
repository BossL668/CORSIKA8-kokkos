#!/usr/bin/env python3
"""Monitor a campaign and its descendants, including validation between showers.

No physics or sampling changes. Run inside a systemd service with MemoryMax,
MemorySwapMax=0, KillMode=control-group, OOMPolicy=stop and Restart=no. The
kernel limit remains the backstop for growth faster than the polling interval.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def process_info(pid):
    try:
        root = Path('/proc') / str(pid)
        # comm can contain spaces/parentheses; fields after the last ')' are stable.
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(stat[19]), 'state': stat[0],
                'rss_mib': int(stat[21]) * os.sysconf('SC_PAGE_SIZE') / 1048576,
                'name': (root / 'comm').read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_tree(root_pid):
    result = []
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        info = process_info(pid)
        if info is None:
            continue
        result.append(info)
        # Child processes can have their own sessions and still belong to this tree.
        for task in (Path('/proc') / str(pid) / 'task').glob('*'):
            try:
                pending.extend(int(x) for x in (task / 'children').read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                pass
    return result


def sample(pid):
    mem = dict(line.replace(':', '').split()[:2]
               for line in Path('/proc/meminfo').read_text().splitlines())
    tree = process_tree(pid)
    runner = next((p['rss_mib'] for p in tree if p['pid'] == pid), 0.)
    children = [p['rss_mib'] for p in tree if p['pid'] != pid]
    return {'host_available_mib': int(mem['MemAvailable']) / 1024,
            'swap_used_mib': (int(mem['SwapTotal']) - int(mem['SwapFree'])) / 1024,
            'runner_rss_mib': runner, 'largest_child_rss_mib': max(children, default=0.),
            # Conservative sum: shared pages may be counted more than once.
            'tree_rss_mib': sum(p['rss_mib'] for p in tree), 'processes': tree}


def reason(value, args):
    if value['host_available_mib'] < args.host_reserve_mib:
        return 'host_available_below_reserve'
    for key, limit in (('runner_rss_mib', args.max_runner_rss_mib),
                       ('largest_child_rss_mib', args.max_child_rss_mib),
                       ('tree_rss_mib', args.max_tree_rss_mib)):
        if value[key] > limit:
            return key + '_over_limit'
    return None


def signal_same_process(info, sig):
    current = process_info(info['pid'])
    if current is not None and current['start_ticks'] == info['start_ticks']:
        try:
            os.kill(info['pid'], sig)
        except ProcessLookupError:
            pass


def stop_tree(proc):
    # Only signal descendants of the process this watchdog launched. Never pgrep
    # and kill unrelated CORSIKA/IDE processes. Start times guard against PID reuse.
    tracked = {p['pid']: p for p in process_tree(proc.pid)}
    root = tracked.get(proc.pid)
    if root:
        signal_same_process(root, signal.SIGSTOP)
    tracked.update({p['pid']: p for p in process_tree(proc.pid)})
    for info in tracked.values():
        signal_same_process(info, signal.SIGTERM)
    if root:
        signal_same_process(root, signal.SIGCONT)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        proc.poll()
        if not any((process_info(p['pid']) or {}).get('state') not in (None, 'Z')
                   for p in tracked.values()):
            break
        time.sleep(0.05)
    for info in tracked.values():
        signal_same_process(info, signal.SIGKILL)
    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-reserve-mib', type=float, default=3072)
    parser.add_argument('--max-child-rss-mib', type=float, default=3072)
    parser.add_argument('--max-runner-rss-mib', type=float, default=1024)
    parser.add_argument('--max-tree-rss-mib', type=float, default=4096)
    parser.add_argument('--interval', type=float, default=.25)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or min(args.interval, args.host_reserve_mib, args.max_child_rss_mib,
                          args.max_runner_rss_mib, args.max_tree_rss_mib) <= 0:
        parser.error('a command and positive limits are required')
    args.output.mkdir(parents=True, exist_ok=True)
    incident = args.output / 'MEMORY_GUARD_STOP.json'
    if incident.exists():
        raise RuntimeError('memory incident marker exists; diagnose and archive it before resuming')
    with (args.output / '.memory_guard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fields = ('time_utc', 'elapsed_s', 'host_available_mib', 'swap_used_mib',
                  'runner_rss_mib', 'largest_child_rss_mib', 'tree_rss_mib', 'child_pids')
        state = {'started_utc': now(), 'status': 'starting', 'command': command,
                 'limits': {k: v for k, v in vars(args).items() if k not in ('output', 'command')}}
        status_path = args.output / 'memory_guard_status.json'
        save(status_path, state)
        initial = sample(os.getpid())
        if initial['host_available_mib'] < args.host_reserve_mib:
            state.update(status='stopped_memory_guard', reason='insufficient_start_memory', last_sample=initial)
            save(incident, state)
            save(status_path, state)
            return 75
        proc = subprocess.Popen(command, start_new_session=True)
        interrupted = []
        def on_signal(signum, _frame):
            interrupted.append(signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)
        start = time.monotonic()
        last_update = start
        state.update(status='running', runner_pid=proc.pid)
        save(status_path, state)
        peaks = {k: 0. for k in ('runner_rss_mib', 'largest_child_rss_mib', 'tree_rss_mib')}
        minimum_available = initial['host_available_mib']
        log_path = args.output / 'memory_guard.csv'
        try:
            with log_path.open('a') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if stream.tell() == 0:
                    writer.writeheader()
                while proc.poll() is None:
                    value = sample(proc.pid)
                    for key in peaks:
                        peaks[key] = max(peaks[key], value[key])
                    minimum_available = min(minimum_available, value['host_available_mib'])
                    state.update(last_sample=value, updated_utc=now(), peaks_mib=peaks.copy(),
                                 minimum_host_available_mib=minimum_available)
                    row = {key: value[key] for key in fields if key in value}
                    row.update(time_utc=state['updated_utc'], elapsed_s=time.monotonic() - start,
                               child_pids=' '.join(str(p['pid']) for p in value['processes'] if p['pid'] != proc.pid))
                    writer.writerow(row)
                    stream.flush()
                    violation = reason(value, args)
                    if violation:
                        # Stop first; diagnostics and marker writing happen afterwards.
                        stop_tree(proc)
                        state.update(status='stopped_memory_guard', reason=violation, returncode=proc.returncode)
                        save(incident, state)
                        save(status_path, state)
                        print('MEMORY GUARD STOP:', violation, flush=True)
                        return 75
                    if interrupted:
                        stop_tree(proc)
                        state.update(status='interrupted', signal=interrupted[0], returncode=proc.returncode)
                        save(status_path, state)
                        return 128 + interrupted[0]
                    if time.monotonic() - last_update >= 5:
                        save(status_path, state)
                        last_update = time.monotonic()
                    time.sleep(args.interval)
            state.update(status='runner_finished' if proc.returncode == 0 else 'runner_failed',
                         returncode=proc.returncode, updated_utc=now())
            save(status_path, state)
            return proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
        except BaseException:
            stop_tree(proc)
            state.update(status='watchdog_error', updated_utc=now())
            save(status_path, state)
            raise
        finally:
            if proc.poll() is None:
                stop_tree(proc)


if __name__ == '__main__':
    raise SystemExit(main())
