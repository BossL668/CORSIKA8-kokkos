#!/usr/bin/env python3
"""Read-only, recent-window resource diagnostics; not a kernel-overlap oracle."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resources', type=Path)
    parser.add_argument('--seconds', type=float, default=60)
    args = parser.parse_args()
    rows = []
    for line in args.resources.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # last append may still be in flight
    rows = [r for r in rows if 'cpu_seconds' in r]
    if len(rows) < 2:
        raise SystemExit('not enough completed telemetry samples')
    end = rows[-1]
    recent = [r for r in rows if r['elapsed_s'] >= end['elapsed_s'] - args.seconds]
    start = recent[0]
    duration = end['elapsed_s'] - start['elapsed_s']
    if duration <= 0:
        raise SystemExit('not enough time in requested window')
    threads = []
    for tid, seconds in end.get('thread_cpu_seconds', {}).items():
        before = start.get('thread_cpu_seconds', {}).get(tid, 0.)
        threads.append(dict(tid=int(tid), name=end['thread_names'].get(tid, ''),
                            cpu_core_equivalent=(seconds - before) / duration))
    report = dict(
        elapsed_s=end['elapsed_s'], measured_window_s=duration,
        process_cpu_core_equivalent=(end['cpu_seconds']-start['cpu_seconds']) / duration,
        latest_RSS_MiB=end['rss_bytes']/2**20,
        system_available_GiB=end['available_bytes']/2**30,
        latest_device_memory_MiB=end.get('device_used_mib'),
        sampled_device_utilization_mean_percent=statistics.mean(
            r['device_util_percent'] for r in recent if 'device_util_percent' in r),
        threads=threads,
        note='CPU seconds include scheduling/waits; device utilization is not direct kernel-overlap evidence.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
