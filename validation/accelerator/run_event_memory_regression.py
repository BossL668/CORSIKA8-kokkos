#!/usr/bin/env python3
"""Compare event-boundary cleanup against a preserved binary, with RSS guards.

One process per case, many showers per process. Uses the established memory
sampler and isolates Arrow/YAML comparisons in short-lived subprocesses.
No installed executable or running production campaign is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_nmulti_memory_acceptance import monitor, save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--antennas', type=Path, required=True)
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--backends', nargs='+', choices=('cuda', 'openmp', 'proposal'),
                   default=['openmp', 'cuda', 'proposal'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.count < 2:
        p.error('count must exceed one')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    helper = Path(__file__).with_name('run_nmulti_memory_acceptance.py')
    report = dict(binaries={v: dict(path=str(getattr(args, v).resolve()),
                                   sha256=sha256(getattr(args, v)))
                           for v in ('before', 'after')},
                  runs={}, comparisons={}, complete=False)
    if args.resume and (root / 'regression.json').exists():
        previous = json.loads((root / 'regression.json').read_text())
        if previous['binaries'] != report['binaries']:
            raise RuntimeError('Refusing to mix different binaries')
    try:
        for backend in args.backends:
            for primary, energy, count in [('photon', '1', args.count),
                                           ('proton', '10', 4)]:
                if backend == 'proposal':
                    count = min(count, 16)
                for version in ('before', 'after'):
                    label = f'{backend}_{primary}_{version}_N{count}'
                    threads = 4 if backend == 'openmp' else 1
                    env = os.environ.copy()
                    env.update(CORSIKA_DATA=str(args.data.resolve()),
                               OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                               OMP_PROC_BIND='spread', OMP_PLACES='cores',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    cmd = [str(getattr(args, version).resolve()), '--pdg',
                           '22' if primary == 'photon' else '2212', '-E', energy,
                           '-N', str(count), '-s', '26090711', '-z', '0', '-a', '0',
                           '--emthin', '1e-6', '--antenna-file', str(args.antennas.resolve()),
                           '--geomagnetic-model', 'IGRF14', '--geomagnetic-year', '2027',
                           '--verbosity', 'info', '-f', str(root / label)]
                    if backend == 'proposal':
                        cmd += ['--em-backend', 'proposal', '--radio-backend', 'cpu']
                    else:
                        cmd += ['--em-backend', 'kokkos-proposal', '--radio-backend', 'kokkos',
                                '--kokkos-execution', backend, '--kokkos-num-threads', str(threads),
                                '--gpu-min-batch', '16', '--gpu-resident-batch-limit', '4096',
                                '--gpu-memory-fraction', '0.15']
                    previous = root / (label + '_monitor.json')
                    if args.resume and previous.exists():
                        result = json.loads(previous.read_text())
                        if result['command'] != cmd or result['returncode'] or result['failure']:
                            raise RuntimeError('Cannot reuse failed or changed run ' + label)
                        print('REUSE ' + label, flush=True)
                    else:
                        if (root / label).exists():
                            raise RuntimeError('Refusing to overwrite ' + label)
                        print('RUN ' + label, flush=True)
                        result = monitor(cmd, root, env, label, timeout=1800)
                    report['runs'][label] = dict(pid=result['pid'], elapsed_s=result['elapsed_s'],
                                               returncode=result['returncode'], failure=result['failure'])
                    save(root / 'regression.json', report)
                    if backend != 'proposal':
                        inspection = root / (label + '_inspection.json')
                        subprocess.run([sys.executable, str(helper), '--inspect', str(root / label),
                                        backend, str(count), str(inspection)], check=True, timeout=180)
                comparison = root / f'{backend}_{primary}_comparison.json'
                subprocess.run([sys.executable, str(helper), '--compare',
                                str(root / f'{backend}_{primary}_before_N{count}'),
                                str(root / f'{backend}_{primary}_after_N{count}'),
                                str(comparison)], check=True, timeout=300)
                report['comparisons'][f'{backend}_{primary}'] = json.loads(comparison.read_text())
                save(root / 'regression.json', report)
                if not report['comparisons'][f'{backend}_{primary}']['pass']:
                    raise RuntimeError('Physics output changed')
        report['complete'] = True
    finally:
        save(root / 'regression.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
