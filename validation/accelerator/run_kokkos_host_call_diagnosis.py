#!/usr/bin/env python3
"""Profile two completed PSR same-binary pilot pairs, without changing physics.

Explicit isolated output, same CPU binding, bounded resource guard. This is a
host-call diagnosis, NOT a new uninstrumented performance measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import yaml


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_guard(guard):
    help_text = subprocess.check_output([sys.executable, str(guard), '--help'], text=True)
    for option in ('--require-exclusive-gpu', '--rss-limit-gib', '--sample-threads'):
        if option not in help_text:
            raise RuntimeError('Diagnostic guard lacks required option: ' + option)


def validate_comparison_tool(tools):
    # Import the complete comparison dependency chain before starting an
    # expensive shower. Merely copying the entry-point script is insufficient.
    subprocess.check_output([sys.executable,
        str(tools / 'compare_backend_build_outputs.py'), '--help'], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'plugin', 'tools', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--guard-script', type=Path, required=True,
                        help='Explicit reviewed resource guard; frozen physics source may have an older guard')
    parser.add_argument('--sample-log2', type=int, choices=range(11), default=0,
                        help='Time approximately 1/2^N calls; 6 selects about 1/64, not exact totals')
    parser.add_argument('--wait-unit', required=True,
                        help='This unit must already have ended successfully; queue outside this tool')
    args = parser.parse_args()
    properties = subprocess.check_output(['systemctl', '--user', 'show', args.wait_unit,
        '-p', 'ActiveState', '-p', 'ExecMainStatus', '-p', 'Result'], text=True)
    props = dict(line.split('=', 1) for line in properties.splitlines() if '=' in line)
    assert props.get('ActiveState') == 'inactive' and props.get('ExecMainStatus') == '0' and props.get('Result') == 'success', props
    pilot = args.pilot.resolve()
    assert read(pilot/'PERFORMANCE_RESULT.json')['complete']
    assert read(pilot/'CORRECTNESS_GATES.json')['passed']
    config = read(pilot/'CONFIG.json')
    options = config['arguments']
    assert options['threads'] == '130' and config['affinity'] == list(range(382, 512))
    assert sorted(os.sched_getaffinity(0)) == config['affinity']
    assert psutil.virtual_memory().available >= 32*2**30
    binary = Path(options['after'])
    expected_binary = config['hashes'][str(binary)]
    assert sha(binary) == expected_binary
    plugin = args.plugin.resolve()
    plugin_hash = sha(plugin)
    guard = args.guard_script.resolve()
    validate_guard(guard)  # reject incompatible diagnostics before creating an event
    validate_comparison_tool(args.tools)
    guard_hash = sha(guard)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CORSIKA_DATA=options['data'], LD_LIBRARY_PATH=options['libraries'],
        FLUPRO='/home/yuhanglu/fluka', OMP_NUM_THREADS='130', OMP_THREAD_LIMIT='130',
        OMP_PROC_BIND='spread', OMP_PLACES='threads', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', KOKKOS_TOOLS_LIBS=str(plugin))
    env['C8_KOKKOS_HOST_PROFILE_SAMPLE_LOG2'] = str(args.sample_log2)
    assert 'KOKKOS_PROFILE_LIBRARY' not in env, 'conflicting profiling plugin'
    state = dict(complete=False, binary_sha256=expected_binary, plugin_sha256=plugin_hash,
        guard_script=str(guard), guard_sha256=guard_hash,
        timing_sample_probability=2.**(-args.sample_log2),
        performance_benchmark=False, phase='prepared', records=[])
    save(out/'STATUS.json', state)
    try:
        for index, seed in enumerate((2026110001, 2026110002)):
            for mode in (('openmp', 'adaptive') if index == 0 else ('adaptive', 'openmp')):
                label = '100TeV-%d-%s' % (seed, mode)
                old = read(pilot/(label+'.json'))
                assert old['complete'] and old['performance_valid'] and old['gpu_exclusivity_checked']
                command = list(old['command'])
                assert command[0] == str(binary) and command.count('-f') == 1
                assert float(command[command.index('-E')+1]) == 1e5
                assert sha(binary) == expected_binary and sha(plugin) == plugin_hash
                assert sha(guard) == guard_hash
                command[command.index('-f')+1] = str(out/label)
                env['C8_KOKKOS_HOST_PROFILE'] = str(out/(label+'.host_calls.json'))
                state['phase'] = label
                save(out/'STATUS.json', state)
                recorded_environment = {key: env[key] for key in (
                    'CORSIKA_DATA', 'LD_LIBRARY_PATH', 'FLUPRO', 'OMP_NUM_THREADS',
                    'OMP_THREAD_LIMIT', 'OMP_PROC_BIND', 'OMP_PLACES',
                    'KOKKOS_TOOLS_LIBS', 'C8_KOKKOS_HOST_PROFILE', 'C8_KOKKOS_HOST_PROFILE_SAMPLE_LOG2')}
                save(out/(label+'.command.json'), dict(command=command, environment=recorded_environment,
                    note='Profiled diagnostic: do not pool into production speed measurements.'))
                subprocess.run([sys.executable, str(guard),
                    '--output', str(out/(label+'-guard')), '--timeout', '1200',
                    '--rss-limit-gib', '20', '--sample-resources', '--sample-threads',
                    '--require-exclusive-gpu', '--', *command], env=env, cwd=out, check=True)
                profile = read(out/(label+'.host_calls.json'))
                assert profile['complete'] and profile['errors'] == 0
                assert profile['sample_log2'] == args.sample_log2
                assert profile['settings_callback_seen'] and not profile['requires_global_fencing']
                event = yaml.safe_load((out/label/'gpu_em/summary.yaml').read_text())['shower_0']
                stats = event['statistics']
                assert event['complete'] and stats['queue_overflows'] == 0
                assert stats['profile']['fixed_point_overflows'] == stats['radio']['fixed_point_overflows'] == 0
                assert stats['profile']['steps'] == stats['gpu_particles']
                assert stats['accelerator']['backend'] == ('openmp' if mode == 'openmp' else 'cuda-openmp')
                if mode == 'openmp':
                    # Instrumentation may change dynamic dual scheduling, but
                    # must not change fixed-seed standalone OpenMP physics.
                    compare_env = {k: v for k, v in env.items() if k not in ('KOKKOS_TOOLS_LIBS', 'C8_KOKKOS_HOST_PROFILE')}
                    subprocess.run([sys.executable, str(args.tools/'compare_backend_build_outputs.py'),
                        str(pilot/label), str(out/label), '--report', str(out/(label+'.same_output.json'))],
                        env=compare_env, check=True)
                else:
                    coop = stats['accelerator']['cooperative']
                    assert coop['subshower_cuda_submissions'] == coop['subshower_cuda_commits']
                ranked = sorted(profile['rows'], key=lambda r: r['host_inclusive_s'], reverse=True)
                state['records'].append(dict(label=label, transport_steps=stats['gpu_particles'],
                    waves=stats['resident_photon_wavefronts']+stats['resident_lepton_wavefronts'],
                    profile_file=env['C8_KOKKOS_HOST_PROFILE'], top_host_calls=ranked[:30]))
                save(out/'STATUS.json', state)
        state.update(complete=True, phase='complete-host-call-diagnosis-not-speed-benchmark')
    except BaseException as error:
        state.update(complete=False, phase='failed-no-retry', error=repr(error))
        raise
    finally:
        save(out/'STATUS.json', state)


if __name__ == '__main__':
    main()
