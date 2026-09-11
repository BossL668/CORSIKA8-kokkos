#!/usr/bin/env python3
"""Three predeclared, bounded terrain illustrations; no radio or rate inference.

Reuse the audited full-resolution DEM/native auxiliary cache. Only the first
CC/NC vertex is conditioned; all subsequent transport and decays are simulated.
The transparent control starts in air and does not force any interaction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error('Use 1..8 OpenMP threads for this bounded local illustration')
    root = args.output.resolve()
    ref = args.reference.resolve()
    binary = args.binary.resolve()
    cache = binary.parents[1] / 'CMakeCache.txt'
    if not cache.is_file() or 'CORSIKA_KOKKOS_BACKEND:STRING=OPENMP\n' not in cache.read_text():
        raise RuntimeError('This local demo requires an audited OpenMP-only build, never a CUDA/dual binary')
    spec = yaml.safe_load((ref / 'pilot_manifest.yaml').read_text())
    scene = ref / 'original_expanded_scene.yaml'
    config = yaml.safe_load(scene.read_text())
    mesh = Path(config['geometry']['mesh_path'])
    if sha256(mesh) != config['provenance']['mesh_sha256']:
        raise RuntimeError('Reference DEM hash mismatch')
    point = spec['jobs'][0]['position']
    transparent = next(j for j in spec['jobs'] if j['name'] == 'nutau10tev_unselected_s79101')
    jobs = [dict(name='conditional_cc', seed=909350, current='cc', position=point),
            dict(name='conditional_nc', seed=909351, current='nc', position=point),
            dict(name='natural_throughgoing', seed=79101, current=None,
                 position=transparent['position'])]
    root.mkdir(parents=True, exist_ok=False)
    (root / 'runs').mkdir()
    manifest = dict(binary=str(binary), binary_sha256=sha256(binary), scene=str(scene),
                    reference=str(ref), mesh_sha256=sha256(mesh),
                    chord=spec['chord'], jobs=jobs, radio=False,
                    first_vertices_conditioned=True, full_neutrino_validation=False,
                    primary='nu_tau', energy_GeV=10000., window_ns=1100.,
                    emcut_GeV=.01, emthin=.001, max_weight=50.,
                    direction=transparent['direction'], backend='Kokkos OpenMP',
                    threads=args.threads, gpu_used=False)
    (root / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    for job in jobs:
        out = root / 'runs' / job['name']
        work = root / 'runs' / (job['name'] + '_work')
        work.mkdir()
        command = [str(binary), '--scene', str(scene), '--output', str(out),
                   '--primary', 'nu_tau', '--energy-GeV', '10000',
                   '--seed', str(job['seed']), '--position-m', *map(str, job['position']),
                   '--direction', *map(str, transparent['direction']),
                   '--neutrino-channels', 'cc+nc', '--tau-decay-model', 'tauola',
                   '--emcut-GeV', '.01', '--hadcut-GeV', '.3', '--mucut-GeV', '.3',
                   '--emthin', '.001', '--max-weight', '50', '--max-step-m', '1',
                   '--transport-window-ns', '1100', '--magnetic-field', 'igrf14',
                   '--magnetic-year', '2027', '--em-backend', 'kokkos',
                   '--threads', str(args.threads),
                   '--batch', '64', '--device-memory-MiB', '192',
                   '--track-row-limit', '2000000', '--aux-cache', str(ref / 'auxiliary')]
        if job['current']:
            command.append('--force-vertex-' + job['current'])
        (root / 'runs' / (job['name'] + '_command.json')).write_text(
            json.dumps(dict(command=command, binary_sha256=manifest['binary_sha256']), indent=2))
        guard = [sys.executable, str(Path(__file__).with_name('run_guarded_diagnostic.py')),
                 '--output', str(root / 'runs' / (job['name'] + '_guard')),
                 '--timeout', '600', '--']
        print('Running ' + job['name'], flush=True)
        subprocess.run(guard + command, cwd=work, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND='false',
                                OPENBLAS_NUM_THREADS='1'))
        result = yaml.safe_load((out / 'terrain_run.yaml').read_text())
        if not result['complete'] or result['diagnostics']['material_mismatches']:
            raise RuntimeError('Incomplete event or material mismatch: ' + job['name'])
        if result['accelerator']['execution_space'] != 'OpenMP':
            raise RuntimeError('Unexpected execution space: ' + job['name'])
    print('All three illustrations completed; analyze recorded tracks next.', flush=True)


if __name__ == '__main__':
    main()
