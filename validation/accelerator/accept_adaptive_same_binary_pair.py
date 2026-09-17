#!/usr/bin/env python3
"""Read-only same-binary CUDA/adaptive pair audit (not ensemble certification).

Write only a new report directory. Stream particle files, keep Arrow/BLAS
single-threaded, and never stop/start a simulator or alter its output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from accept_adaptive_service_uhe import load, reserve, scan_parquet, resource_summary


def require(value, message):
    if not value:
        raise ValueError(message)


def command_flags(command):
    # The frozen campaign commands use one value per flag. Reject duplicates
    # rather than silently treating ambiguous argv as an identical experiment.
    require(len(command) % 2 == 1, 'Expected key/value campaign command')
    keys = command[1::2]
    require(len(keys) == len(set(keys)), 'Repeated CLI flag')
    require(all(k.startswith('-') for k in keys), 'Invalid CLI key')
    return dict(zip(keys, command[2::2]))


def physics_flags(command):
    flags = command_flags(command)
    for key in ('-f', '--kokkos-execution', '--kokkos-num-threads',
                '--kokkos-cooperative-policy'):
        flags.pop(key, None)
    return flags


def check_guard(guard, exclusive):
    require(guard['pass'] and guard['returncode'] == 0 and not guard.get('failure'),
            'Unsuccessful or incomplete guarded simulation')
    checked = (guard.get('exclusive_gpu_required') is True and
               guard.get('performance_valid') is True and
               guard.get('gpu_checks', 0) > 1 and
               guard.get('foreign_gpu_pids') == [])
    require(not exclusive or checked, 'Missing/failed exclusive-GPU evidence')
    return checked


def endpoint_steps(statistics):
    accelerator = statistics['accelerator']
    total = statistics['gpu_particles']
    require(total == statistics['profile']['steps'], 'Profile/transport step mismatch')
    if accelerator['backend'] == 'cuda':
        require(not accelerator['openmp'] and accelerator['host_threads'] == 1,
                'Single-CUDA reference has OpenMP enabled')
        return dict(cuda=total, openmp=0)
    require(accelerator['backend'] == 'cuda-openmp', 'Unexpected execution backend')
    cooperative = accelerator['cooperative']
    require(cooperative['subshower_cuda_submissions'] ==
            cooperative['subshower_cuda_commits'], 'Uncommitted CUDA jobs')
    result = {e: sum(cooperative['adaptive'][e][k]['transport_records']
                     for k in ('photon', 'lepton')) for e in ('cuda', 'openmp')}
    require(sum(result.values()) == total, 'Endpoint transport steps lost/doubled')
    return result


def fallback_counts(s):
    # PhysicalAcceleratedEmRouter::returnFallbacks() counts immediate specified
    # completions as returned_to_scalar in independent mode. Standalone CUDA
    # parks those SAME event types and counts them in deferred_* instead.
    # The historical cpu_generic_fallbacks name does not imply equal semantics.
    queued = s['deferred_cpu_fallbacks_queued']
    flushed = s['deferred_cpu_fallbacks_flushed']
    require(queued == flushed, 'Unflushed deferred fallback events')
    returned = s['cpu_generic_fallbacks']
    total = sum(s['cpu_fallbacks_by_reason_name'].values())
    require(total == returned + queued, 'Returned/deferred fallback count mismatch')
    require(total == sum(s['cpu_fallbacks_by_process'].values()), 'Fallback process count mismatch')
    immediate = s['cpu_specified_final_states'] - flushed
    require(0 <= immediate <= returned, 'Invalid immediate specified completion count')
    require(s['cpu_fallbacks_by_reason_name'].get('native_selection_replay', 0) ==
            s['cpu_completed_native_selection_replays'] == s['cpu_completed_selected_losses'],
            'Selected-loss replay mismatch')
    return dict(total_events=total, immediate_specified=immediate,
                deferred_specified=flushed, unspecified_scalar=returned - immediate)


def binary_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(path, exclusive):
    reserve()
    guard_path = path.parent / (path.name + '-guard')
    guard = json.loads((guard_path / 'summary.json').read_text())
    checked = check_guard(guard, exclusive)
    flags = command_flags(guard['command'])
    event = load(path / 'summary.yaml')
    gpu = load(path / 'gpu_em/summary.yaml')['shower_0']
    s = gpu['statistics']
    timing = load(path / 'simulation_timing/summary.yaml')['shower_0']
    require(event['showers'] == 1 and event.get('end time') and
            gpu['complete'] and timing['closed'], 'Event is not closed')
    require(event['seed'] == int(flags['-s']), 'Seed metadata differs from argv')
    require(s['process_registry']['accepted'], 'Rejected process registry')
    require(s['queue_overflows'] == s['profile']['fixed_point_overflows'] ==
            s['radio']['fixed_point_overflows'] == s['profile']['invalid_records'] ==
            s['proposal_native']['inverse_failures'] == 0, 'Invalid/overflowed output')
    require(s['cross_species']['final_pending_photons'] ==
            s['cross_species']['final_pending_leptons'] == 0, 'Undrained queues')
    endpoint = endpoint_steps(s)
    reasons = s['cpu_fallbacks_by_reason_name']
    fallback = fallback_counts(s)
    cfg = load(path / 'gpu_em/config.yaml')
    identity = dict(flags=physics_flags(guard['command']), seed=event['seed'],
        primary=load(path / 'primary/summary.yaml')['shower_0'],
        environment=cfg['environment'], magnetic_constant=cfg['magnetic_rigidity_GeV_per_T_m'],
        table={k: s['proposal_native'][k] for k in ('proposal_version',
               'cubic_interpolation_version', 'table_sha256', 'aux_sha256')},
        thinning={k: s['thinning'][k] for k in ('em_fraction', 'maximum_weight',
                                               'automatic_maximum_weight')},
        radio={a: load(path / a / 'config.yaml') for a in ('CoREAS', 'ZHS')})
    files = {str(p.relative_to(path)): scan_parquet(p) for p in sorted(path.rglob('*.parquet'))}
    require(len(files) == 7, 'Missing/unexpected output Parquet files')
    profile = pq.read_table(path / 'profile/profile.parquet', use_threads=False).to_pandas()
    require(np.all(np.diff(profile['X']) > 0), 'Invalid longitudinal grid')
    require(all(np.all(profile[k] >= 0) for k in profile if k not in ('X', 'shower')),
            'Negative longitudinal profile')
    for file in (path / 'interaction_hist').glob('*.npz'):
        with np.load(file, allow_pickle=False) as arrays:
            require(all(np.isfinite(arrays[k]).all() for k in arrays.files),
                    'Invalid interaction histogram')
    resources, _ = resource_summary(guard_path / 'resources.jsonl')
    c = s['accelerator'].get('cooperative', {})
    values = dict(process_s=guard['elapsed_s'], shower_s=timing['wall_time_ms'] / 1000,
        outside_shower_s=guard['elapsed_s'] - timing['wall_time_ms'] / 1000,
        transport_steps=s['gpu_particles'], cuda_steps=endpoint['cuda'], openmp_steps=endpoint['openmp'],
        resident_waves=s['resident_photon_wavefronts'] + s['resident_lepton_wavefronts'],
        radio_segments=s['radio_tracks'], scalar_steps=s['cpu_particle_steps'],
        scalar_stepper_s=s['hybrid_timing_ms']['scalar_stepper'] / 1000,
        cuda_jobs=c.get('subshower_cuda_submissions'), openmp_jobs=c.get('subshower_openmp_epochs'),
        gpu_completion_service_s=c.get('cuda_result_service_delay_ms', 0) / 1000 if c else None,
        peak_rss_GiB=guard['peak_tree_rss_bytes'] / 2**30,
        minimum_available_GiB=guard['minimum_available_bytes'] / 2**30, **resources)
    sha = binary_hash(guard['command'][0])
    provenance_file = path.parent / 'PROVENANCE.json'
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text())
        require(sha == provenance['binary_sha256'], 'Archived binary/provenance mismatch')
        if 'complete' in provenance:
            require(provenance['complete'], 'Campaign wrapper has not committed completion')
    return dict(path=str(path), backend=s['accelerator']['backend'],
        threads=s['accelerator']['host_threads'], policy=c.get('scheduling_policy'),
        identity=identity, binary_sha256=sha, values=values, files=files,
        exclusive_gpu_checked=checked, fallback_reasons=reasons, fallback_counts=fallback,
        gpu_monitor=dict(required=guard.get('exclusive_gpu_required', False),
            valid=guard.get('performance_valid', False),
            checks=guard.get('gpu_checks', 0),
            observation_failures=guard.get('gpu_observation_failures', 0),
            foreign_pids=guard.get('foreign_gpu_pids', [])),
        energy_ledger=s['energy_ledger']), profile


def plot_pair(out, records, profiles, pulse_package, analysis_module_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'savefig.dpi': 180})
    labels = ['Single CUDA', f"CUDA + OpenMP ({records[1]['threads']} threads)"]
    colors = ['#0072B2', '#D55E00']
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [r['values']['process_s'] / 60 for r in records], color=colors)
    ax.bar_label(bars, fmt='%.2f', padding=3); ax.margins(y=.18)
    ax.set_ylabel('End-to-end time [min]'); ax.grid(axis='y', alpha=.2)
    ax.set_title('Same binary and initial seed; one event per mode')
    fig.tight_layout(); fig.savefig(out / 'time_comparison.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, components, title in zip(axes.flat,
        [('photon', 'electron', 'positron'), ('photon',), ('electron',),
         ('positron',), ('muplus', 'muminus'), ('hadron',)],
        ['Electromagnetic', 'Photons', 'Electrons', 'Positrons', 'Muons', 'Hadrons']):
        end = 0.
        for profile, label, color in zip(profiles, labels, colors):
            x = profile['X'].to_numpy(); y = sum(profile[k].to_numpy() for k in components)
            ax.plot(x, y, label=label, color=color, lw=1.2)
            active = np.flatnonzero(y > y.max() * 1e-4)
            if len(active): end = max(end, x[active[-1]])
        ax.set_xlim(0, end + 20); ax.set_ylim(bottom=0); ax.set_title(title)
        ax.set_xlabel(r'Slant depth [g cm$^{-2}$]'); ax.set_ylabel('Weighted particle count')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.suptitle('Individual shower profiles, not an ensemble test')
    fig.tight_layout(); fig.savefig(out / 'single_event_profiles.png'); plt.close(fig)
    sys.path.insert(0, str(analysis_module_dir))
    import reproject_waveform_ensembles as waveforms
    flags = records[0]['identity']['flags']
    project, basis = waveforms.basis_for(dict(theta=float(flags['-z']), phi=float(flags['-a']),
                                            pulse_package=str(pulse_package)))
    identity = records[0]['identity']
    require(np.allclose([identity['primary'][k] for k in ('nx', 'ny', 'nz')],
                        basis['direction'], atol=2e-12, rtol=0), 'Pulse direction mismatch')
    field = np.array([identity['environment']['magnetic_field_T'][k] for k in ('x', 'y', 'z')]) * 1e6
    require(np.allclose(field, basis['B_NWU_microtesla'], rtol=1e-9, atol=0), 'Pulse field mismatch')
    radio = {}
    for algorithm in ('CoREAS', 'ZHS'):
        cfg = identity['radio'][algorithm]; observers = list(cfg['observers'].values())
        values = []
        for record in records:
            # Copy each yielded buffer before the iterator can overwrite it.
            events = [project(v.copy()) for v in waveforms.events(
                Path(record['path']) / algorithm / 'observers.parquet', cfg, 1, algorithm)]
            require(len(events) == 1 and np.isfinite(events[0]).all(), 'Invalid radio event')
            values.append(events[0])
        positions = np.array([o['location'] for o in observers]); positions -= positions[0]
        direction = np.array(basis['direction'])
        rperp = np.linalg.norm(positions - np.outer(positions @ direction, direction), axis=1)
        indices = sorted(set(int(np.argmin(abs(rperp - target))) for target in (0, 100, 300)))
        dt = 1 / observers[0]['sampling frequency']
        time = (np.arange(values[0].shape[1]) + (.5 if algorithm == 'ZHS' else 0)) * dt
        radio[algorithm] = []
        for array in values:
            power = np.sum(array**2, axis=2); total = power.sum()
            radio[algorithm].append(dict(nonzero_observers=int(np.sum(power.max(axis=1) > 0)),
                edge5_pooled_power_fraction=float((power[:, :5].sum() + power[:, -5:].sum()) / total)
                    if total else None, outside_recorded_window_unknown=True))
        fig, axes = plt.subplots(len(indices), 3, figsize=(12, 7), squeeze=False)
        for row, index in enumerate(indices):
            power = sum(np.sum(v[index]**2, axis=1) for v in values)
            if power.sum():
                lo, hi = np.searchsorted(np.cumsum(power) / power.sum(), [.001, .999])
                lower = max(time[0], time[lo] - 10 * dt)
                upper = min(time[-1], time[min(hi, len(time) - 1)] + 10 * dt)
            else: lower, upper = time[0], time[-1]
            for col, component in enumerate(('Ex′', 'Ey′', 'Ez′')):
                ax = axes[row, col]
                for value, label, color in zip(values, labels, colors):
                    ax.plot(time, value[index, :, col] * 1e6, color=color, label=label, lw=1)
                ax.set_xlim(lower, upper); ax.grid(alpha=.2)
                ax.set_title(f'{component}, r_perp={rperp[index]:.1f} m')
                ax.set_xlabel('Time from observer window start [ns]'); ax.set_ylabel('Field [µV/m]')
        axes[0, 0].legend(fontsize=8); fig.suptitle(algorithm + ': individual signed waveforms, not means')
        fig.tight_layout(); fig.savefig(out / (algorithm + '_single_event_prime.png')); plt.close(fig)
    return dict(basis=basis, window_diagnostics=radio)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--expected-energy-gev', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-exclusive-gpu', action='store_true')
    p.add_argument('--pulse-package', type=Path,
                   default=Path('/home/yuhanglu/21CMA/python/MCMCTidyUp/pulse_analysis_modular'))
    p.add_argument('--analysis-module-dir', type=Path, default=Path('/home/yuhanglu/21CMA/analysis_jobs'))
    args = p.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing report')
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    audited = [audit(path.resolve(), args.require_exclusive_gpu) for path in (args.reference, args.candidate)]
    records, profiles = list(zip(*audited)); a, b = records
    require(a['backend'] == 'cuda' and b['backend'] == 'cuda-openmp', 'Expected CUDA/adaptive pair')
    require(a['identity'] == b['identity'], 'Physical configuration/table/seed mismatch')
    require(a['binary_sha256'] == b['binary_sha256'], 'Not a same-binary comparison')
    require(a['identity']['primary']['total_energy'] == args.expected_energy_gev, 'Wrong energy')
    ratios = {k: b['values'][k] / a['values'][k] for k in a['values']
              if isinstance(a['values'][k], (int, float)) and a['values'][k] != 0
              and isinstance(b['values'][k], (int, float))}
    result = dict(operational_checks_passed=True, physical_settings_matched=True,
        binary_sha256=a['binary_sha256'], records=records, candidate_over_reference=ratios,
        descriptive_speedup=1 / ratios['process_s'],
        wall_time_per_transport_step_ratio=ratios['process_s'] / ratios['transport_steps'],
        exclusive_gpu_checked=all(r['exclusive_gpu_checked'] for r in records),
        performance_certified=False, ensemble_physics_certified=False,
        full_shower_energy_closure_certified=False,
        semantics={'transport_steps': 'Completed transport steps, not unique particles',
                   'cuda_jobs': 'Logical coordinator jobs; absent for standalone CUDA',
                   'resident_waves': 'Physical resident iterations, not logical jobs',
                   'gpu_completion_service_s': 'Completion-to-receipt delay, may overlap useful CPU work',
                   'outside_shower_s': 'Process minus shower wall time; not a measured startup breakdown'},
        limitations=['One event per mode; dynamic scheduling changes the shower tree/workload',
                     'Per-step cost is descriptive; process/energy mixtures are not identical',
                     'CPU utilization includes spinning; device utilization is not useful-work throughput',
                     'Accelerator energy ledger does not establish full hadronic shower energy closure'])
    args.output.mkdir(parents=True, exist_ok=False)
    result['radio'] = plot_pair(args.output, records, profiles, args.pulse_package, args.analysis_module_dir)
    (args.output / 'ACCEPTANCE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# 同版本单 CUDA／双端单事件诊断', '',
        '**运行完整性检查通过；不是大样本物理等价或性能认证。**', '',
        f"能量 {args.expected_energy_gev:g} GeV，seed {a['identity']['seed']}；物理配置、表和二进制哈希一致。", '',
        '| 指标 | 单 CUDA | 双端 |', '|---|---:|---:|']
    for title, key in [('进程耗时/s', 'process_s'), ('shower耗时/s', 'shower_s'),
        ('实际输运步数', 'transport_steps'), ('GPU输运步数', 'cuda_steps'),
        ('OpenMP输运步数', 'openmp_steps'), ('驻留波前数', 'resident_waves'),
        ('射电轨迹段数', 'radio_segments'), ('平均逻辑核当量', 'cpu_cores'),
        ('GPU平均利用率/%', 'gpu_utilization_percent'), ('RSS峰值/GiB', 'peak_rss_GiB')]:
        lines.append(f"| {title} | {a['values'][key]:,.3f} | {b['values'][key]:,.3f} |")
    lines += ['', f"描述性加速比 {result['descriptive_speedup']:.3f}×；相对耗时变化 {(ratios['process_s']-1)*100:+.2f}%。",
        '不把重复入口数当作独立粒子数；不将GPU领取等待直接从wall time扣除。',
        '图为单事件profile和Ex′/Ey′/Ez′有符号波形，不是系综均值。局部能量账本未用于认证全shower闭合。', '',
        '![耗时](time_comparison.png)', '', '![profile](single_event_profiles.png)', '',
        '![CoREAS](CoREAS_single_event_prime.png)', '', '![ZHS](ZHS_single_event_prime.png)']
    if not result['exclusive_gpu_checked']:
        warning = ('**严格GPU独占计时证据未通过或缺失，以下时间仅作描述性记录，不能作为已验证加速。** '
                   f"单CUDA／双端的GPU进程查询失败数分别为{a['gpu_monitor']['observation_failures']}／"
                   f"{b['gpu_monitor']['observation_failures']}；未知采样没有按GPU空闲处理。")
        lines[4:4] = [warning, '']
    (args.output / 'ACCEPTANCE_REPORT_CN.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(args.output), operational=True,
                          descriptive_speedup=result['descriptive_speedup'], certified=False)))


if __name__ == '__main__':
    main()
