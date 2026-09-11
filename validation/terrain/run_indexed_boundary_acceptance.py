#!/usr/bin/env python3
"""Run only standalone probes, under small resource limits, preserving evidence.

Device-wide VRAM deltas include any concurrent production activity; they are
conservative guards, not an attribution of memory to this probe's CUDA context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_multimaterial_acceptance import gpu


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(name, binary, mesh, output):
    initial = gpu() if name != 'host' else None
    started = time.monotonic()
    samples, reason = [], None
    with (output / f'{name}.json').open('x') as log, (output / f'{name}.stderr').open('x') as err:
        child = subprocess.Popen([str(binary), str(mesh)], stdout=log, stderr=err,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                sample = {'elapsed_s': time.monotonic() - started}
                try:
                    fields = dict(line.split(':', 1) for line in
                                  Path(f'/proc/{child.pid}/status').read_text().splitlines() if ':' in line)
                    sample['rss_kib'] = int(fields.get('VmRSS', '0 kB').split()[0])
                except FileNotFoundError:
                    pass
                mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                sample['available_kib'] = int(mem['MemAvailable'].split()[0])
                if initial:
                    current = gpu()
                    sample['gpu_delta_mib'] = current[1] - initial[1]
                    if sample['gpu_delta_mib'] > min(512, .1 * initial[0]):
                        reason = 'conservative device-wide VRAM increase guard'
                if sample.get('rss_kib', 0) > 512 * 1024:
                    reason = '512 MiB probe RSS guard'
                if sample['available_kib'] < 3 * 1024 * 1024:
                    reason = '3 GiB global available memory reserve'
                if sample['elapsed_s'] > 120:
                    reason = '120 second timeout'
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
    result = {'binary_sha256': digest(binary), 'returncode': child.returncode,
              'stop_reason': reason, 'wall_s': time.monotonic() - started,
              'gpu_baseline': initial, 'samples': samples}
    (output / f'{name}_resources.json').write_text(json.dumps(result, indent=2) + '\n')
    if reason or child.returncode:
        raise RuntimeError(f'{name} failed; see {output}')
    payload = json.loads((output / f'{name}.json').read_text())
    if not payload['tested_contract_passed'] or payload['one_ulp_edge_side_mismatches']:
        raise RuntimeError(f'{name} failed numerical checks')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in ('host', 'cuda', 'nofma'):
        results[name] = run(name, args.binary_dir / f'probe_indexed_boundary_{name}',
                            args.mesh, args.output)
        print(name, results[name]['result_fnv1a64'], flush=True)
    reference = dict(results['host'])
    reference.pop('host_device_comparisons')
    for name in ('cuda', 'nofma'):
        actual = dict(results[name])
        count = actual.pop('host_device_comparisons')
        if count < 1000000 or actual != reference:
            raise RuntimeError(f'{name} does not match host diagnostic')
    summary = {'scope': 'standalone straight-ray candidate; no production integration',
               'production_ready': False, 'tested_contract_passed': True,
               'mesh_sha256': digest(args.mesh), 'results': results}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
