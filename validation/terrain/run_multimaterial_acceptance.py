#!/usr/bin/env python3
"""Small, bounded real-DEM transport cases; never launches air production.

WSL may not expose per-process VRAM. In that case record device-wide baseline
and incremental use, and stop our process group conservatively at 512 MiB.
The application's own fixed allocation gate remains 128 MiB. These two
measurements must not be mislabelled as exact NVML per-process memory.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


def gpu():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=10)
    return [float(x.strip()) for x in text.splitlines()[0].split(',')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--aux-cache', type=Path, required=True)
    p.add_argument('--mode', choices=['cpu', 'openmp', 'cuda'], required=True)
    p.add_argument('--cases', nargs='+', default=['photon_up', 'photon_down', 'electron_rock'])
    p.add_argument('--seed', type=int, default=67101)
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--track-row-limit', type=int, default=200000)
    p.add_argument('--magnetic-field', choices=['none','igrf14'], default='none',
                   help='Explicitly retain the old zero-field regression unless requested')
    args = p.parse_args()
    digest = hashlib.sha256()
    with args.binary.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    binary_hash = digest.hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = {
        'photon_up': ('photon', 1, -.01, 1, 1e-6, False),
        'photon_down': ('photon', 1, .01, -1, 1e-6, False),
        'electron_rock': ('electron', 1, -.01, 1, 1e-6, False),
        'positron_rock': ('positron', 1, -.01, 1, 1e-6, False),
        'electron_air_lowenergy': ('electron', .0011, .01, -1, 1e-6, False),
        'positron_air_lowenergy': ('positron', .0011, .01, -1, 1e-6, False),
        'nue_natural': ('nu_e', 1e4, -.01, 1, .1, False),
        'nue_forced': ('nu_e', 1e4, -.01, 1, .1, True),
        'nue_forced_coarse': ('nu_e', 1e4, -.01, 1, .5, True),
    }
    results = []
    for case in args.cases:
        primary, energy, height, direction, thinning, force = cases[case]
        output = args.output_root / case
        if output.exists():
            raise RuntimeError(f'refuse to overwrite {output}')
        command = [str(args.binary), '--scene', str(args.scene), '--output', str(output),
                   '--primary', primary, '--energy-GeV', str(energy), '--seed', str(args.seed),
                   '--position-m', '0', '0', str(height), '--direction', '0', '0', str(direction),
                   '--emthin', str(thinning), '--em-backend', 'proposal' if args.mode == 'cpu' else 'kokkos',
                   '--threads', '2' if args.mode == 'openmp' else '1', '--batch', '64',
                   '--device-memory-MiB', '128', '--aux-cache', str(args.aux_cache),
                   '--track-row-limit', str(args.track_row_limit),
                   '--magnetic-field', args.magnetic_field]
        if force:
            command.append('--force-vertex-cc')
        env = dict(os.environ, OMP_NUM_THREADS='2' if args.mode == 'openmp' else '1', OMP_PROC_BIND='false')
        initial_gpu = gpu() if args.mode == 'cuda' else None
        limit_mib = min(512, initial_gpu[0] * .10) if initial_gpu else None
        samples, stop_reason = [], None
        available = lambda: int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                    if line.startswith('MemAvailable:')))
        if available() < 3 * 1024 * 1024:
            raise RuntimeError('less than 3 GiB available; do not start terrain diagnostic')
        started = time.monotonic()
        with (args.output_root / f'{case}.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    sample = {'elapsed_s': elapsed}
                    sample['available_kib'] = available()
                    if sample['available_kib'] < 3 * 1024 * 1024:
                        stop_reason = '3 GiB global available memory reserve'
                    status = Path(f'/proc/{child.pid}/status')
                    if status.exists():
                        fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
                        sample['rss_kib'] = int(fields.get('VmRSS', '0 kB').split()[0])
                    if initial_gpu:
                        now = gpu()
                        sample.update(gpu_used_mib=now[1], gpu_util_percent=now[2],
                                      gpu_delta_mib=now[1]-initial_gpu[1])
                        if sample['gpu_delta_mib'] > limit_mib:
                            stop_reason = 'conservative global GPU memory increase guard'
                    if sample.get('rss_kib', 0) > 2 * 1024 * 1024:
                        stop_reason = '2 GiB child RSS guard'
                    if elapsed > args.timeout:
                        stop_reason = 'timeout'
                    samples.append(sample)
                    if stop_reason:
                        break
                    time.sleep(.25)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        summary_file = output / 'terrain_run.yaml'
        summary = yaml.safe_load(summary_file.read_text()) if summary_file.exists() else {}
        item = dict(case=case, mode=args.mode, command=command, binary_sha256=binary_hash,
                    magnetic_field=args.magnetic_field, returncode=child.returncode,
                    complete=bool(summary.get('complete')), stop_reason=stop_reason,
                    wall_seconds=time.monotonic()-started, diagnostics=summary.get('diagnostics'),
                    accelerator=summary.get('accelerator'), material_tables=summary.get('material_tables'),
                    gpu_baseline=initial_gpu, gpu_guard_increment_mib=limit_mib, samples=samples)
        results.append(item)
        (args.output_root / 'acceptance.json').write_text(json.dumps(results, indent=2)+'\n')
        print(case, child.returncode, item['complete'], item['wall_seconds'], flush=True)
        if child.returncode or not item['complete']:
            raise RuntimeError(f'{case} failed: {summary.get("error", stop_reason)}')


if __name__ == '__main__':
    main()
