#!/usr/bin/env python3
"""Isolated adaptive-v3 service-budget gates, Fe pilots, then one 100 PeV event.

No installation, production resume, table generation or overwrite of old runs.
Each job is guarded with a 4 GiB available-memory floor and persistent logs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import yaml


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'build', 'before', 'fe-manifest', 'proton-command'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--build-unit', required=True)
    parser.add_argument('--skip-high-energy', action='store_true')
    a = parser.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    tools = source/'validation/accelerator'
    state = dict(phase='waiting-for-build', complete=False, records=[])
    env = dict(os.environ, FLUPRO='/home/yuhanglu/fluka',
        CORSIKA_DATA=str(source/'modules/data'), OMP_NUM_THREADS='20', OMP_THREAD_LIMIT='20',
        OMP_PROC_BIND='false', OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    floor = 4*2**30

    def status(phase):
        state.update(phase=phase, updated_unix=time.time())
        save(root/'STATUS.json', state)

    def guard(label, command, timeout, rss=5):
        assert psutil.virtual_memory().available >= floor, 'available RAM <4 GiB'
        assert shutil.disk_usage(root).free >= 15*2**30, 'disk reserve <15 GiB'
        status(label)
        cmd = list(map(str, command))
        result = subprocess.run([sys.executable, str(tools/'run_overlap_guarded.py'),
            '--output', str(root/(label+'-guard')), '--timeout', str(timeout),
            '--rss-limit-gib', str(rss), '--sample-resources', '--sample-threads',
            '--', *cmd], env=env, cwd=source)
        report = json.loads((root/(label+'-guard/summary.json')).read_text())
        assert result.returncode == 0 and report['pass'], label+' failed'
        return report

    try:
        status('waiting-for-build')
        while subprocess.run(['systemctl','--user','is-active','--quiet',a.build_unit]).returncode == 0:
            if psutil.virtual_memory().available < floor:
                subprocess.run(['systemctl','--user','stop',a.build_unit], check=True)
                raise RuntimeError('build stopped: memory floor')
            time.sleep(2)
        code = subprocess.check_output(['systemctl','--user','show',a.build_unit,
            '-p','ExecMainStatus','--value'], text=True).strip()
        assert code == '0', 'build did not pass'
        assert subprocess.run(['pgrep','-x','c8_air_shower'], stdout=subprocess.DEVNULL).returncode == 1, 'GPU production active'
        (root/'binaries').mkdir()
        binary = root/'binaries/c8_air_shower'
        fixture = root/'binaries/testKokkosCooperativeBackend'
        shutil.copy2(a.build/'applications/c8_air_shower', binary)
        shutil.copy2(a.build/'tests/accelerator/testKokkosCooperativeBackend', fixture)
        manifest = json.loads(a.fe_manifest.read_text())
        proton = json.loads(a.proton_command.read_text())['command']
        identity = dict(binary_sha256=sha(binary), reference_sha256=sha(a.before),
            fixture_sha256=sha(fixture), fe_manifest=str(a.fe_manifest),
            proton_command=str(a.proton_command), threads=20, memory_fraction=.7,
            policy='adaptive-v3-service-budget', physics_changed=False,
            scheduling_is_dynamic=True, full_ensemble_acceptance=False)
        save(root/'PROVENANCE.json', identity)
        guard('regression', [sys.executable, tools/'run_independent_driver_regression.py',
            '--before', a.before, '--after', binary, '--output', root/'regression',
            '--lifecycle-energy-gev','100','--allow-adaptive-option',
            '--cooperative-policy','adaptive'], 1800)
        guard('real-EM-radio-N2', [fixture, source/'modules/data/PROPOSAL',
            '20','subshowers','.2','adaptive'], 600)
        save(root/'CORRECTNESS_GATES.json', dict(passed=True, statistical_acceptance=False))

        def put(cmd, flag, value):
            if flag in cmd:
                cmd[cmd.index(flag)+1] = str(value)
            else:
                cmd += [flag, str(value)]

        def event(family, mode, seed):
            label = family+'-'+str(seed)+'-'+mode
            cmd = [str(binary), *manifest['common_argv']] if family == 'Fe100TeV' else list(proton)
            cmd[0] = str(binary)
            for flag, value in (('-N',1),('-s',seed),('-f',root/label),
                ('--em-backend','kokkos'),('--radio-backend','kokkos'),
                ('--gpu-physics-source','proposal-native'),('--gpu-min-batch',4096),
                ('--gpu-memory-fraction',.7),('--hadronic-workers',1),
                ('--kokkos-device',0),('--kokkos-execution','cuda-openmp'),
                ('--kokkos-num-threads',20),('--kokkos-cooperative-policy',mode)):
                put(cmd,flag,value)
            assert sha(binary) == identity['binary_sha256']
            save(root/(label+'-command.json'), dict(command=cmd, binary_sha256=sha(binary)))
            report = guard(label, cmd, 7200 if family == 'proton100PeV' else 1200)
            gpu = yaml.safe_load((root/label/'gpu_em/summary.yaml').read_text())['shower_0']
            timing = yaml.safe_load((root/label/'simulation_timing/summary.yaml').read_text())['shower_0']
            assert gpu['complete'] and timing['closed'], 'output not closed'
            stats = gpu['statistics']
            coop = stats['accelerator']['cooperative']
            assert stats['queue_overflows'] == stats['profile']['fixed_point_overflows'] == stats['radio']['fixed_point_overflows'] == 0
            assert coop['subshower_cuda_submissions'] == coop['subshower_cuda_commits']
            if mode == 'adaptive':
                assert coop['scheduling_policy'] == 'adaptive-v3-service-budget'
            record = dict(family=family, mode=mode, seed=seed, process_s=report['elapsed_s'],
                timing=timing, cooperative=coop, complete=True)
            save(root/(label+'-result.json'), record)
            state['records'].append(record)
            status(label+'-complete')

        # Alternating same-binary control: old independent mode is unchanged.
        for seed in (85000001,85000002):
            modes = ('legacy','adaptive') if seed%2 else ('adaptive','legacy')
            for mode in modes:
                event('Fe100TeV', mode, seed)
        if not a.skip_high_energy:
            event('proton100PeV','adaptive',2026110001)
        state['complete'] = True
        status('complete-pilot-not-ensemble-acceptance')
    except BaseException as e:
        state['error'] = str(e)
        status('failed-production-remains-paused')
        raise


if __name__ == '__main__':
    main()
