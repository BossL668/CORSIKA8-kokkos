#!/usr/bin/env python3
"""Run small actual terrain showers with bounded, test-only host shadow replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--aux-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--cases', nargs='+', default=['photon_up', 'photon_down', 'electron_rock', 'positron_rock'])
    args = p.parse_args()
    cases = dict(photon_up=('photon', -.01, 1), photon_down=('photon', .01, -1),
                 electron_rock=('electron', -.01, 1), positron_rock=('positron', -.01, 1))
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with args.binary.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    (args.output / 'input.json').write_text(json.dumps(dict(
        binary=str(args.binary.resolve()), sha256=digest.hexdigest(),
        scene=str(args.scene.resolve()), auxiliary_cache=str(args.aux_cache.resolve()),
        scope='same-input portable-host/device step, NOT scalar-PROPOSAL shower tree'), indent=2)+'\n')
    for case in args.cases:
        primary, height, direction = cases[case]
        folder = args.output / case
        folder.mkdir()
        command = [str(args.binary.resolve()), '--scene', str(args.scene.resolve()),
                   '--output', str(folder.resolve() / 'shower'), '--primary', primary,
                   '--energy-GeV', '1', '--seed', '67101', '--position-m', '0', '0', str(height),
                   '--direction', '0', '0', str(direction), '--emthin', '1e-6',
                   '--em-backend', 'kokkos', '--threads', '1' if args.gpu else '2',
                   '--batch', '64', '--device-memory-MiB', '128',
                   '--aux-cache', str(args.aux_cache.resolve()), '--magnetic-field', 'igrf14',
                   '--track-row-limit', '300000']
        env = dict(os.environ, C8_TERRAIN_AUDIT_PREFIX=str(folder.resolve() / 'steps'),
                   OMP_NUM_THREADS='1' if args.gpu else '2', OMP_PROC_BIND='false',
                   OPENBLAS_NUM_THREADS='1')
        guard = [sys.executable, str(Path(__file__).with_name('run_guarded_diagnostic.py')),
                 '--output', str(folder / 'guard'), '--timeout', '120']
        if args.gpu:
            guard.append('--gpu')
        subprocess.run(guard + ['--'] + command, env=env, check=True)
        counts = json.loads((folder / 'steps.counts.json').read_text())
        print(case, counts, flush=True)


if __name__ == '__main__':
    main()
