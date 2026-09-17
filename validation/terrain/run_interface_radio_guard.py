#!/usr/bin/env python3
"""Independent radio diagnostic; kills only its own child process group.

VRAM is a conservative device-wide increment, not per-process NVML attribution.
The legacy EM guard is unchanged. Radio moments and CUDA kernel stack/context
need an explicit budget beyond that small probe's ten-percent limit.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_multimaterial_acceptance import gpu


def available_kib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu-increment-MiB', type=float, default=768.,
                        help='Device-wide radio increment limit, capped at 25%% of total VRAM')
    parser.add_argument('--timeout', type=float, default=240.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.gpu_increment_MiB < float('inf'):
        parser.error('--gpu-increment-MiB must be finite and positive')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    if available_kib() < 3 * 1024 * 1024:
        raise RuntimeError('less than 3 GiB available; do not start diagnostic')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = gpu() if args.gpu else None
    gpu_limit = min(args.gpu_increment_MiB, .25 * baseline[0]) if baseline else None
    samples, reason = [], None
    start = time.monotonic()
    with (args.output / 'stdout.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                sample = dict(elapsed_s=time.monotonic()-start, available_kib=available_kib())
                try:
                    fields = dict(line.split(':', 1) for line in Path(f'/proc/{child.pid}/status').read_text().splitlines() if ':' in line)
                    sample['rss_kib'] = int(fields.get('VmRSS', '0 kB').split()[0])
                except FileNotFoundError:
                    pass
                if baseline:
                    current = gpu()
                    sample.update(gpu_delta_mib=current[1]-baseline[1], gpu_util_percent=current[2])
                    if sample['gpu_delta_mib'] > gpu_limit:
                        reason = f'device-wide VRAM increment guard ({gpu_limit:g} MiB)'
                if sample.get('rss_kib', 0) > 2 * 1024 * 1024:
                    reason = '2 GiB child RSS guard'
                if sample['available_kib'] < 3 * 1024 * 1024:
                    reason = '3 GiB available memory reserve'
                if sample['elapsed_s'] > args.timeout:
                    reason = 'timeout'
                samples.append(sample)
                if reason:
                    break
                time.sleep(.25)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    result = dict(command=command, returncode=child.returncode, stop_reason=reason,
                  wall_seconds=time.monotonic()-start, gpu_baseline=baseline,
                  gpu_increment_limit_MiB=gpu_limit, samples=samples)
    (args.output / 'resources.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'exit={child.returncode}, stop={reason}, seconds={result["wall_seconds"]:.3f}', flush=True)
    if child.returncode or reason:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
