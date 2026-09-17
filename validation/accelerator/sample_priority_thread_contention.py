#!/usr/bin/env python3
"""Read-only /proc sampler for one frozen priority acceptance binary.

No subprocess, GPU API, signal, affinity write or run-directory write is used.
The explicit output is created exclusively and capped at 64 MiB. A last_cpu
sample is a placement observation, NOT a per-CPU accounting history. schedstat
runqueue delay is not OpenMP barrier time; zero values may mean disabled kernel
scheduler statistics (the setting is included in the header).
Optional queued prewait checks only run status and exact executable identity;
the active sampling duration starts when that binary is first observed running.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


MAX_OUTPUT_BYTES = 64 * 1024 * 1024


def error_record(path, error):
    return {'path': str(path), 'type': type(error).__name__,
            'errno': getattr(error, 'errno', None), 'message': str(error)}


def parse_proc_stat(text):
    """Parse Linux stat without splitting its arbitrary parenthesized comm."""
    left, right = text.find('('), text.rfind(')')
    if left < 1 or right <= left:
        raise ValueError('Malformed /proc stat comm')
    fields = text[right + 1:].split()  # starts with field 3 (state)
    if len(fields) < 37:
        raise ValueError('Truncated /proc stat: last_cpu field is absent')
    return {'tid': int(text[:left].strip()), 'comm': text[left + 1:right],
            'state': fields[0], 'user_ticks': int(fields[11]),
            'system_ticks': int(fields[12]), 'start_ticks': int(fields[19]),
            'last_cpu': int(fields[36])}


def parse_schedstat(text):
    fields = text.split()
    if len(fields) < 3:
        raise ValueError('Truncated /proc schedstat')
    values = [int(value) for value in fields[:3]]
    if min(values) < 0:
        raise ValueError('Negative scheduler counter')
    return dict(zip(('runtime_ns', 'runqueue_ns', 'sched_slices'), values))


def counter_deltas(previous, current, hz):
    """Never subtract counters across PID/TID reuse or unreadable samples."""
    identity = ('pid', 'process_start_ticks', 'tid', 'start_ticks')
    if previous is None or any(previous.get(k) != current.get(k) for k in identity):
        return {'counter_baseline': True}
    result = {'counter_baseline': False,
              'delta_elapsed_s': current['sample_time'] - previous['sample_time']}
    resets = []
    for key in ('cpu_ticks', 'runtime_ns', 'runqueue_ns', 'sched_slices'):
        if key not in current or key not in previous:
            continue
        delta = current[key] - previous[key]
        if delta < 0:
            result['delta_' + key] = None
            resets.append(key)
        elif key == 'cpu_ticks':
            result['delta_cpu_s'] = delta / hz
        else:
            result['delta_' + key] = delta
    if resets:
        result['counter_regressions'] = resets
    return result


def thread_role(comm, tid, pid):
    if comm == 'c8-cuda-driver':
        return 'cuda_driver'  # distinguish active/inactive by counters, not name
    if tid == pid:
        return 'coordinator'
    if comm == 'c8_air_shower':
        return 'openmp_candidate'  # comm alone cannot prove an OpenMP worker
    return 'other'


def read_thread(proc, pid, process_start, tid, now, hz, previous):
    directory = proc / str(pid) / 'task' / str(tid)
    errors = []
    try:
        raw = parse_proc_stat((directory / 'stat').read_text())
        if raw['tid'] != tid:
            raise ValueError('Thread stat identity changed')
    except (OSError, ValueError) as error:
        return {'tid': tid, 'errors': [error_record(directory / 'stat', error)]}, None
    snapshot = dict(raw, pid=pid, process_start_ticks=process_start,
                    sample_time=now, cpu_ticks=raw['user_ticks'] + raw['system_ticks'])
    for filename, parser in (('schedstat', parse_schedstat), ('wchan', lambda s: {'wchan': s.strip()})):
        try:
            snapshot.update(parser((directory / filename).read_text()))
        except (OSError, ValueError) as error:
            errors.append(error_record(directory / filename, error))
    try:
        final = parse_proc_stat((directory / 'stat').read_text())
        if (final['tid'], final['start_ticks']) != (tid, raw['start_ticks']):
            raise ValueError('Thread identity changed during sampling; discard deltas')
    except (OSError, ValueError) as error:
        errors.append(error_record(directory / 'stat', error))
        return {'tid': tid, 'errors': errors, 'identity_verified_after_sample': False}, None
    result = {key: value for key, value in snapshot.items()
              if key not in ('pid', 'process_start_ticks', 'sample_time',
                             'user_ticks', 'system_ticks', 'cpu_ticks')}
    result['cpu_s'] = snapshot['cpu_ticks'] / hz
    result['role'] = thread_role(raw['comm'], tid, pid)
    result.update(counter_deltas(previous, snapshot, hz))
    if errors:
        result['errors'] = errors
    return result, snapshot


def fingerprint(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def validate_binary(run):
    provenance = json.loads((run / 'PROVENANCE.json').read_text())
    expected = provenance.get('binary_sha256', '')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError('PROVENANCE.json lacks a valid binary_sha256')
    binary = (run / 'binaries' / 'c8_air_shower').resolve(strict=True)
    before = fingerprint(binary)
    digest = hashlib.sha256()
    with binary.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if before != fingerprint(binary) or digest.hexdigest() != expected.lower():
        raise ValueError('Frozen binary changed or does not match PROVENANCE SHA-256')
    return binary, before, expected.lower()


def matching_pids(proc, binary, binary_fingerprint):
    matches, errors = [], []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
            executable = directory / 'exe'
            # Match path AND inode: do not monitor another build with the same
            # filename, a replaced binary or an unrelated same-hash copy.
            if os.readlink(executable) != str(binary):
                continue
            if fingerprint(executable) != binary_fingerprint:
                raise ValueError('Live executable differs from the frozen binary')
            matches.append(int(directory.name))
        except (OSError, ValueError) as error:
            errors.append(error_record(directory / 'exe', error))
    return sorted(matches), errors


def external_cpu_activity(proc, excluded, now, hz, previous):
    """Only same-user comm/PID/counters: never record command lines or environ."""
    records, current, errors = [], {}, []
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
            stat = parse_proc_stat((directory / 'stat').read_text())
            key = (stat['tid'], stat['start_ticks'])
            ticks = stat['user_ticks'] + stat['system_ticks']
            current[key] = (now, ticks)
            old = previous.get(key)
            # A first observation is only a baseline, not evidence that an old
            # process consumed its lifetime CPU time during this interval.
            if old is not None and ticks > old[1]:
                dt = now - old[0]
                seconds = (ticks - old[1]) / hz
                records.append({'pid': stat['tid'], 'start_ticks': stat['start_ticks'],
                                'comm': stat['comm'], 'delta_cpu_s': seconds,
                                'delta_elapsed_s': dt,
                                'cpu_percent': 100 * seconds / dt if dt > 0 else None})
        except (OSError, ValueError) as error:
            errors.append(error_record(directory / 'stat', error))
    return sorted(records, key=lambda row: row['delta_cpu_s'], reverse=True), current, errors


def sample_process(proc, pid, now, hz, previous):
    directory = proc / str(pid)
    result = {'pid': pid, 'threads': []}
    try:
        initial = parse_proc_stat((directory / 'stat').read_text())
        result['start_ticks'] = initial['start_ticks']
        tids = sorted(int(path.name) for path in (directory / 'task').iterdir()
                      if path.name.isdigit())
    except (OSError, ValueError) as error:
        result['errors'] = [error_record(directory, error)]
        return result, previous
    process_key = (pid, initial['start_ticks'])
    # Retain previous readable values for temporarily unreadable live threads,
    # but not for retired TIDs or previous process lifetimes.
    kept = {key: value for key, value in previous.items()
            if key[:2] == process_key and key[2] in tids}
    for tid in tids:
        old = next((value for key, value in kept.items() if key[2] == tid), None)
        record, snapshot = read_thread(proc, pid, initial['start_ticks'], tid, now, hz, old)
        result['threads'].append(record)
        if snapshot is not None:
            kept = {key: value for key, value in kept.items() if key[2] != tid}
            kept[(pid, initial['start_ticks'], tid, snapshot['start_ticks'])] = snapshot
    try:
        final = parse_proc_stat((directory / 'stat').read_text())
        if final['start_ticks'] != initial['start_ticks']:
            raise ValueError('Process identity changed while sampling; discard deltas')
    except (OSError, ValueError) as error:
        result.setdefault('errors', []).append(error_record(directory / 'stat', error))
        result['identity_verified_after_sample'] = False
        return result, {}
    result['identity_verified_after_sample'] = True
    return result, kept


def observe_start(proc, run, binary, binary_fingerprint):
    """Cheap queued check: no thread counters, GPU calls or external CPU scan."""
    errors = []
    try:
        if fingerprint(binary) != binary_fingerprint:
            raise ValueError('Frozen binary changed during queued prewait')
    except (OSError, ValueError) as error:
        return {'state': 'binary_identity_changed', 'pids': [],
                'errors': [error_record(binary, error)]}
    try:
        if json.loads((run / 'STATUS.json').read_text()).get('complete') is True:
            return {'state': 'status_complete', 'pids': [], 'errors': errors}
    except (OSError, ValueError) as error:
        errors.append(error_record(run / 'STATUS.json', error))
    pids, discovery_errors = matching_pids(proc, binary, binary_fingerprint)
    errors.extend(discovery_errors)
    return {'state': 'started' if pids else 'waiting', 'pids': pids, 'errors': errors}


def wait_for_start(timeout, observe, emit, clock=time.monotonic, sleep=time.sleep):
    """Bounded two-second prewait; aggregate errors into sparse 30-second rows.

    Clock/sleep/observer injection keeps unit tests independent of processes and
    real waits. Returns a terminal reason or 'started'; never runs a workload.
    """
    started = clock()
    next_report = started
    pending_errors = {}
    while True:
        observation = observe()
        now = clock()
        for error in observation.get('errors', []):
            key = json.dumps(error, sort_keys=True)
            if key not in pending_errors:
                pending_errors[key] = dict(error, occurrences=0)
            pending_errors[key]['occurrences'] += 1
        state = observation['state']
        if state == 'waiting' and now - started >= timeout:
            state = 'wait_start_timeout'
        if now >= next_report or state != 'waiting':
            if not emit({'type': 'waiting', 'state': state,
                         'wait_elapsed_s': now - started, 'wait_limit_s': timeout,
                         'pids': observation.get('pids', []),
                         'errors': list(pending_errors.values())}):
                return {'reason': 'output_size_limit', 'wait_elapsed_s': clock() - started}
            pending_errors.clear()
            next_report = now + 30.
        if state != 'waiting':
            return {'reason': state, 'wait_elapsed_s': now - started}
        remaining = timeout - (clock() - started)
        if remaining > 0:
            sleep(min(2., remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=1800)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--wait-start-timeout', type=float, default=0,
                        help='Before active sampling, wait up to this many seconds for the exact frozen binary (0 disables prewait)')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 < args.duration <= 7200:
        parser.error('--duration must be finite and in (0,7200] seconds')
    if not math.isfinite(args.interval) or not 1 <= args.interval <= 60:
        parser.error('--interval must be finite and in [1,60] seconds')
    if not math.isfinite(args.wait_start_timeout) or not 0 <= args.wait_start_timeout <= 21600:
        parser.error('--wait-start-timeout must be finite and in [0,21600] seconds')
    run = args.run.resolve(strict=True)
    binary, binary_fingerprint, expected = validate_binary(run)
    proc = Path('/proc')
    hz = os.sysconf('SC_CLK_TCK')
    started = time.monotonic()
    previous = {}
    previous_external = {}
    with args.output.open('x', encoding='utf-8') as output:
        def emit(record, final=False):
            line = json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n'
            # Reserve space for an explicit final stop reason; never silently
            # truncate a JSON record when the bounded artifact is full.
            if not final and output.tell() + len(line.encode('utf-8')) > MAX_OUTPUT_BYTES - 1024:
                return False
            output.write(line)
            output.flush()
            return True
        header = {'type': 'header', 'schema': 1, 'run': str(run), 'binary': str(binary),
                  'binary_sha256': expected, 'clock_ticks_per_second': hz,
                  'unix_start_s': time.time(), 'interval_s': args.interval,
                  'duration_limit_s': args.duration, 'output_limit_bytes': MAX_OUTPUT_BYTES,
                  'wait_start_timeout_s': args.wait_start_timeout,
                  'absolute_limit_s': args.wait_start_timeout + args.duration,
                  'duration_semantics': 'active sampling after optional exact-binary prewait',
                  'last_cpu_semantics': 'instantaneous placement, not time attribution',
                  'runqueue_semantics': 'scheduler runnable wait; not OpenMP barrier duration',
                  'external_cpu_scope': 'same user, no command lines; deltas after baseline only'}
        try:
            header['kernel_sched_schedstats'] = (proc / 'sys/kernel/sched_schedstats').read_text().strip()
        except OSError as error:
            header['errors'] = [error_record(proc / 'sys/kernel/sched_schedstats', error)]
        emit(header)
        if args.wait_start_timeout:
            waited = wait_for_start(args.wait_start_timeout,
                lambda: observe_start(proc, run, binary, binary_fingerprint), emit)
            if waited['reason'] != 'started':
                emit({'type': 'stop', **waited, 'elapsed_s': time.monotonic() - started,
                      'active_elapsed_s': 0, 'signals_sent': 0, 'affinity_changes': 0}, final=True)
                return 1 if waited['reason'] == 'binary_identity_changed' else 0
        active_start = time.monotonic()
        # Even if a status/proc read is unexpectedly delayed, never grant an
        # additional full active duration beyond the absolute configured bound.
        deadline = min(active_start + args.duration,
                       started + args.wait_start_timeout + args.duration)
        if not emit({'type': 'active_start', 'elapsed_s': active_start - started,
                     'unix_time_s': time.time(), 'duration_remaining_s': max(0., deadline - active_start)}):
            emit({'type': 'stop', 'reason': 'output_size_limit',
                  'elapsed_s': time.monotonic() - started,
                  'signals_sent': 0, 'affinity_changes': 0}, final=True)
            return 0
        stop = 'duration_limit'
        while time.monotonic() < deadline:
            now = time.monotonic()
            sample = {'type': 'sample', 'elapsed_s': now - active_start,
                      'total_elapsed_s': now - started, 'processes': [], 'errors': []}
            try:
                if fingerprint(binary) != binary_fingerprint:
                    raise ValueError('Frozen binary changed during monitoring')
            except (OSError, ValueError) as error:
                emit({'type': 'error', **error_record(binary, error)})
                stop = 'binary_identity_changed'
                break
            try:
                status = json.loads((run / 'STATUS.json').read_text())
                if status.get('complete') is True:
                    stop = 'status_complete'
                    break
            except (OSError, ValueError) as error:
                sample['errors'].append(error_record(run / 'STATUS.json', error))
            pids, errors = matching_pids(proc, binary, binary_fingerprint)
            sample['errors'].extend(errors)
            current = {}
            for pid in pids:
                old = {key: value for key, value in previous.items() if key[0] == pid}
                record, snapshots = sample_process(proc, pid, now, hz, old)
                sample['processes'].append(record)
                current.update(snapshots)
            previous = current
            outside, previous_external, errors = external_cpu_activity(
                proc, set(pids) | {os.getpid()}, now, hz, previous_external)
            sample['external_cpu_activity'] = outside
            sample['errors'].extend(errors)
            if not emit(sample):
                stop = 'output_size_limit'
                break
            remaining = deadline - time.monotonic()
            delay = min(remaining, args.interval - (time.monotonic() - now))
            if delay > 0:
                time.sleep(delay)
        emit({'type': 'stop', 'reason': stop, 'elapsed_s': time.monotonic() - started,
              'active_elapsed_s': time.monotonic() - active_start,
              'signals_sent': 0, 'affinity_changes': 0}, final=True)
    return 1 if stop == 'binary_identity_changed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
