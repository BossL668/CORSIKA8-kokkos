#!/usr/bin/env python3
"""Small real-air integration check; no production files are overwritten.

Runs all profile modes using the same seed on each requested backend, then
requires all non-profile physical parquet arrays to be identical within that
backend. This is not a cross-backend identical-shower or performance claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq
import yaml


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('Cannot read available RAM')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--antenna', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--backends', nargs='+', default=['cuda', 'openmp'])
    p.add_argument('--full-step-records', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    env = dict(os.environ, CORSIKA_DATA=str(a.data.resolve()),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               OMP_PROC_BIND='spread', OMP_PLACES='threads')
    for backend in a.backends:
        baseline = None
        for mode in ['default', 'original-c8', 'forward', 'both']:
            case = a.out / f'{backend}_{mode}'
            cmd = [str(a.exe.resolve()), '-p', '22', '-E', '1', '-N', '2',
                   '-s', '12345', '-f', str(case.resolve()),
                   '--antenna-file', str(a.antenna.resolve())]
            if backend == 'proposal':
                cmd += ['--em-backend', 'proposal', '--radio-backend', 'cpu']
            else:
                cmd += ['--em-backend', 'kokkos-proposal', '--radio-backend', 'kokkos',
                        '--kokkos-execution', backend, '--gpu-min-batch', '32']
                if backend == 'openmp':
                    cmd += ['--kokkos-num-threads', '4']
                if a.full_step_records:
                    cmd += ['--gpu-full-step-records']
            if mode != 'default':
                cmd += ['--profile-crossings', mode]
            if available_gib() < 4:
                raise RuntimeError('Below 4 GiB available RAM; no event started')
            start = time.monotonic()
            with case.with_suffix('.log').open('w') as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if available_gib() < 4 or time.monotonic()-start > 900:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill(); child.wait()
                        raise RuntimeError('Stopped this test: RAM reserve or 15-minute timeout')
                    time.sleep(1)
            if child.returncode:
                raise RuntimeError(f'{backend}/{mode} failed: see {case}.log')
            config = yaml.safe_load((case/'profile/config.yaml').read_text())
            expected = 'original-c8' if mode == 'default' else mode
            assert config['profile-crossings'] == expected and not config['affects-transport']
            physics = {str(f.relative_to(case)): pq.read_table(f, use_threads=False)
                       for f in case.rglob('*.parquet')
                       if f.parent.name != 'profile' and 'timing' not in str(f.relative_to(case)).lower()}
            assert physics and any('coreas' in k.lower() for k in physics)
            for radio in ('CoREAS', 'ZHS'):
                table = physics[f'{radio}/observers.parquet']
                values = [v for component in ('Ex', 'Ey', 'Ez')
                          for v in table[component].to_pylist()]
                assert values and all(math.isfinite(v) for v in values)
                assert max(map(abs, values)) > 0, f'Empty {radio} waveform'
            if backend != 'proposal':
                summaries = yaml.safe_load((case/'gpu_em/summary.yaml').read_text())
                for event in range(2):
                    summary = summaries[f'shower_{event}']
                    assert summary['complete']
                    stats = summary['statistics']
                    assert stats['gpu_particles'] > 0
                    assert stats['accelerator']['backend'] == backend
                    assert stats['backend_lifecycle']['reused'] == (event > 0)
            profile = pq.read_table(case/'profile/profile.parquet')
            assert sorted(set(profile['shower'].to_pylist())) == [0, 1]
            if baseline is None:
                baseline, base_profile = physics, profile
            else:
                assert physics.keys() == baseline.keys()
                for key in physics:
                    assert physics[key].equals(baseline[key]), f'Non-profile array changed: {backend}/{mode}/{key}'
                if mode == 'original-c8':
                    assert profile.equals(base_profile), 'Default is not original C8'
            records.append(dict(backend=backend, mode=mode, command=cmd,
                                seconds=time.monotonic()-start, events=2,
                                unchanged_physics_arrays=list(physics), passed=True))
            print(json.dumps(records[-1]), flush=True)
            (a.out/'summary.json').write_text(json.dumps(dict(
                binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),
                records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
