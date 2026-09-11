#!/usr/bin/env python3
"""Bounded constants, boundary and live scalar-module regressions.

Does not build/install, modify tables, or signal other jobs. Each child is
supervised by the existing RSS/VRAM/time guard. All results, including failures,
are preserved. Integration showers are run separately by the terrain runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    protected = source / 'applications/c8_air_shower.cpp'
    manifest = dict(backend='cuda' if args.gpu else 'openmp',
                    build=str(args.build.resolve()), air_source_before=digest(protected),
                    scope='local units/trajectory/module regression, not ensemble certification',
                    sources={}, tests=[], passed=True)
    for relative in ('corsika/accelerator/ScalarPhysicalConstants.hpp',
                     'corsika/accelerator/em/common/UniformMagneticField.hpp',
                     'corsika/accelerator/em/common/Types.hpp',
                     'corsika/accelerator/em/common/TransportMass.hpp',
                     'corsika/accelerator/em/detail/LeptonTransportStep.hpp',
                     'tests/accelerator/MagneticLinearGateChecks.hpp',
                     'src/terrain/TerrainEmSession.cpp'):
        manifest['sources'][relative] = digest(source / relative)
    for name in ('CMakeCache.txt',):
        (args.output / name).write_bytes((args.build / name).read_bytes())
    env = dict(os.environ, OMP_NUM_THREADS='1' if args.gpu else '2',
               OMP_PROC_BIND='false', OPENBLAS_NUM_THREADS='1')
    for name in ('testKokkosScalarConstants', 'testKokkosTerrainSession',
                 'testTerrainCurvedBoundary', 'testKokkosScalarRadioAlignment',
                 'testKokkosCpuTransportAlignment'):
        binary = args.build.resolve() / 'tests/accelerator' / name
        row = dict(name=name, binary=str(binary), binary_sha256=digest(binary))
        command = [sys.executable, str(source / 'validation/terrain/run_guarded_diagnostic.py'),
                   '--output', str(args.output / name), '--timeout', '240']
        if args.gpu:
            command.append('--gpu')
        command += ['--', str(binary)]
        row['returncode'] = subprocess.run(command, env=env).returncode
        row['passed'] = row['returncode'] == 0
        manifest['tests'].append(row)
        manifest['passed'] &= row['passed']
        manifest['air_source_after'] = digest(protected)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(name, row['passed'], flush=True)
    if manifest['air_source_after'] != manifest['air_source_before']:
        raise RuntimeError('air application source changed while auditing')
    if not manifest['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
