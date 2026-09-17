#!/usr/bin/env python3
"""Isolated adaptive scheduling gates, Fe pilots, then one 100 PeV event.

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
    parser.add_argument('--policy-version', default='adaptive-v3-service-budget')
    parser.add_argument('--source-root', type=Path,
                        help='Full isolated source, when this launcher starts from a small frozen overlay')
    parser.add_argument('--tools-root', type=Path,
                        help='Independent monitor/analysis scripts; never modifies the frozen build source')
    parser.add_argument('--gpu-idle-seconds', type=float, default=0.,
                        help='Require repeated idle observations before each guarded run; excluded from timing')
    parser.add_argument('--timing-seeds', type=int, default=2)
    parser.add_argument('--timing-modes', nargs=2, choices=('legacy','adaptive','cuda','openmp'),
                        default=['legacy','adaptive'],
                        help='Alternate two same-binary modes; CUDA/adaptive tests net dual benefit')
    parser.add_argument('--strict-help', action='store_true',
                        help='Both frozen binaries already expose the same adaptive CLI')
    parser.add_argument('--require-exclusive-gpu', action='store_true',
                        help='Require uncontaminated GPU timing; unavailable samples invalidate timing')
    a = parser.parse_args()
    if a.timing_seeds < 1 or len(set(a.timing_modes)) != 2:
        parser.error('positive timing-seeds and two different timing modes required')
    if not 0 <= a.gpu_idle_seconds <= 300:
        parser.error('gpu-idle-seconds must be finite and between 0 and 300')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = (a.source_root or Path(__file__).resolve().parents[2]).resolve()
    tools = (a.tools_root or source/'validation/accelerator').resolve()
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
        if a.gpu_idle_seconds:
            from wait_gpu_idle_then_exec import gpu_processes
            idle = None
            deadline = time.monotonic()+600
            while True:
                assert psutil.virtual_memory().available >= floor, 'available RAM <4 GiB while waiting'
                # A failed/unknown observation raises; never reinterpret it as idle.
                pids = gpu_processes()
                now = time.monotonic()
                idle = (idle if idle is not None else now) if not pids else None
                if idle is not None and now-idle >= a.gpu_idle_seconds:
                    break
                if now >= deadline:
                    raise RuntimeError('GPU did not become observably idle; no simulation launched')
                time.sleep(2)
        guarded = [sys.executable, str(tools/'run_overlap_guarded.py'),
            '--output', str(root/(label+'-guard')), '--timeout', str(timeout),
            '--rss-limit-gib', str(rss), '--sample-resources', '--sample-threads']
        if a.require_exclusive_gpu:
            guarded += ['--require-exclusive-gpu']
        result = subprocess.run(guarded + ['--', *cmd], env=env, cwd=source)
        report = json.loads((root/(label+'-guard/summary.json')).read_text())
        assert result.returncode == 0 and report['pass'], label+' failed'
        if a.require_exclusive_gpu and report.get('performance_valid') is not True:
            print(label+' physical output completed, timing INVALID; retaining the sample without certifying speed', flush=True)
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
            policy=a.policy_version, physics_changed=False,
            scheduling_is_dynamic=True, full_ensemble_acceptance=False)
        identity.update(guard_sha256=sha(tools/'run_overlap_guarded.py'),
                        exclusive_gpu_required=a.require_exclusive_gpu,
                        gpu_idle_seconds=a.gpu_idle_seconds)
        save(root/'PROVENANCE.json', identity)
        regression=[sys.executable, tools/'run_independent_driver_regression.py',
            '--before', a.before, '--after', binary, '--output', root/'regression',
            '--lifecycle-energy-gev','100','--cooperative-policy','adaptive',
            '--source-root',source]
        if not a.strict_help: regression+=['--allow-adaptive-option']
        guard('regression', regression, 1800)
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
            dual = mode in ('legacy','adaptive')
            for flag, value in (('-N',1),('-s',seed),('-f',root/label),
                ('--em-backend','kokkos'),('--radio-backend','kokkos'),
                ('--gpu-physics-source','proposal-native'),('--gpu-min-batch',4096),
                ('--gpu-memory-fraction',.7),('--hadronic-workers',1),
                ('--kokkos-device',0),('--kokkos-execution','cuda-openmp' if dual else mode),
                ('--kokkos-num-threads',1 if mode=='cuda' else 20)):
                put(cmd,flag,value)
            if '--kokkos-cooperative-policy' in cmd:
                index=cmd.index('--kokkos-cooperative-policy');del cmd[index:index+2]
            if dual:
                put(cmd,'--kokkos-cooperative-policy',mode)
            assert sha(binary) == identity['binary_sha256']
            save(root/(label+'-command.json'), dict(command=cmd, binary_sha256=sha(binary)))
            report = guard(label, cmd, 7200 if family == 'proton100PeV' else 1200)
            gpu = yaml.safe_load((root/label/'gpu_em/summary.yaml').read_text())['shower_0']
            timing = yaml.safe_load((root/label/'simulation_timing/summary.yaml').read_text())['shower_0']
            assert gpu['complete'] and timing['closed'], 'output not closed'
            stats = gpu['statistics']
            accelerator = stats['accelerator']
            coop = accelerator.get('cooperative', {})
            assert accelerator['backend']==('cuda-openmp' if dual else mode)
            assert accelerator['host_threads']==(1 if mode=='cuda' else 20)
            assert stats['queue_overflows'] == stats['profile']['fixed_point_overflows'] == stats['radio']['fixed_point_overflows'] == 0
            if dual:
                assert coop['subshower_cuda_submissions'] == coop['subshower_cuda_commits']
            if mode == 'adaptive':
                assert coop['scheduling_policy'] == a.policy_version
                if a.policy_version in ('adaptive-v4-bounded-batching','adaptive-v5-work-quantum','adaptive-v6-tail-grain','adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    steps=waves=jobs=0
                    for endpoint in ('cuda','openmp'):
                        ep=coop['adaptive'][endpoint]
                        jobs+=sum(ep['job_input_histogram_floor_log2'])
                        for kind in ('photon','lepton'):
                            steps+=ep[kind]['transport_records']
                            waves+=ep[kind]['resident_wavefronts']
                    assert steps==stats['gpu_particles']==stats['profile']['steps'], 'endpoint physical step accounting'
                    assert waves==stats['resident_photon_wavefronts']+stats['resident_lepton_wavefronts'], 'endpoint wave accounting'
                    assert jobs==coop['subshower_cuda_commits']+coop['subshower_openmp_epochs'], 'job histogram accounting'
                    if a.policy_version in ('adaptive-v5-work-quantum','adaptive-v6-tail-grain','adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                        reasons=sum(sum(coop['adaptive'][e][k]['completion_reasons'])
                                    for e in ('cuda','openmp') for k in ('photon','lepton'))
                        assert reasons==jobs, 'each job must have one completion reason'
                    if a.policy_version in ('adaptive-v7-gpu-continuation','adaptive-v8-bounded-continuation','adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                        cap=2 if a.policy_version=='adaptive-v7-gpu-continuation' else 16
                        assert coop['cuda_completion_packets']+coop['subshower_cuda_autonomous_continuations']==coop['subshower_cuda_commits']
                        assert coop['subshower_cuda_autonomous_continuations']<=(cap-1)*coop['cuda_completion_packets']
                        assert coop['cuda_completion_buffer_delay_ms']>=coop['cuda_result_service_delay_ms']
                        if cap==16:
                            assert sum(coop['cuda_continuation_stops'])==coop['cuda_completion_packets']
                            assert coop['maximum_cuda_packet_calls']<=cap
                            assert coop['cuda_completion_retention_budget_bytes']==64*2**20
                        if a.policy_version in ('adaptive-v9-foreground-continuation','adaptive-v10-continuation-learning','adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                            assert coop['subshower_cuda_foreground_packets']<=coop['cuda_completion_packets']
                            assert coop['subshower_cuda_foreground_continuations']<=coop['subshower_cuda_autonomous_continuations']
                if a.policy_version in ('adaptive-v11-host-profile-shards','adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    assert 1 < coop['host_profile_shards'] <= 256
                    assert 0 < coop['host_profile_shard_bytes'] <= 16 * 2**20
                if a.policy_version in ('adaptive-v12-coalesced-fallback','adaptive-v13-independent-service-horizon'):
                    from run_adaptive_cooperative_acceptance import check_coalesced_fallbacks
                    check_coalesced_fallbacks(stats)
            record = dict(family=family, mode=mode, seed=seed, process_s=report['elapsed_s'],
                performance_valid=report.get('performance_valid',False),
                gpu_observation_failures=report.get('gpu_observation_failures',0),
                timing=timing, cooperative=coop, transport_records=stats['gpu_particles'],
                radio_tracks=stats['radio_tracks'], complete=True)
            save(root/(label+'-result.json'), record)
            state['records'].append(record)
            status(label+'-complete')

        # Same-binary paired input seeds; dynamic schedules may grow different
        # trees. Keep the default legacy comparison, but allow the meaningful
        # net-benefit comparison against the standalone CUDA/OpenMP endpoint.
        for index in range(a.timing_seeds):
            seed=85000001+index
            modes = a.timing_modes if index%2==0 else a.timing_modes[::-1]
            for mode in modes:
                event('Fe100TeV', mode, seed)
        if not a.skip_high_energy:
            event('proton100PeV','adaptive',2026110001)
        state['complete'] = True
        state['performance_evidence_complete'] = all(r['performance_valid'] for r in state['records'])
        status('complete-pilot-not-ensemble-acceptance')
    except BaseException as e:
        state['error'] = str(e)
        status('failed-production-remains-paused')
        raise


if __name__ == '__main__':
    main()
