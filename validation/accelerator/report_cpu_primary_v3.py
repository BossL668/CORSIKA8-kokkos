#!/usr/bin/env python3
"""Read-only completed-seed CPU-primary diagnostics; writes reports only on request."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import yaml

from summarize_priority_endpoints import resources


def read(path, issues, yaml_format=False):
    try:
        text = path.read_text()
        value = yaml.safe_load(text) if yaml_format else json.loads(text)
        if not isinstance(value, dict):
            raise ValueError('expected an object')
        return value
    except (OSError, ValueError, yaml.YAMLError) as error:
        issues.append('{}: {}'.format(path, error))
        return {}


def physics_flags(command):
    if not command or len(command) % 2 != 1:
        return None
    keys = command[1::2]
    if len(keys) != len(set(keys)) or not all(k.startswith('-') for k in keys):
        return None
    return {k: v for k, v in zip(keys, command[2::2])
            if k not in ('-f', '--kokkos-execution', '--kokkos-num-threads')}


def milliseconds(value):
    return value / 1000. if value is not None else None


def endpoint_work(stats, mode):
    cooperative = stats.get('accelerator', {}).get('cooperative', {})
    result = {}
    for endpoint in ('openmp', 'cuda'):
        if mode in ('openmp', 'cuda'):
            active = mode == endpoint
            steps = stats.get('gpu_particles') if active else 0
            waves = (stats.get('resident_photon_wavefronts', 0) +
                     stats.get('resident_lepton_wavefronts', 0)) if active else 0
            times = [stats.get('photon_backend_wall_time_ms'),
                     stats.get('lepton_backend_wall_time_ms')]
            seconds = milliseconds(sum(times)) if active and all(t is not None for t in times) else (None if active else 0.)
        else:
            columns = cooperative.get('priority_endpoints', cooperative.get('adaptive', {})).get(endpoint, {})
            kinds = ('photon', 'lepton')
            known = all(k in columns and 'transport_records' in columns[k] and
                        'resident_wavefronts' in columns[k] for k in kinds)
            steps = sum(columns[k]['transport_records'] for k in kinds) if known else None
            waves = sum(columns[k]['resident_wavefronts'] for k in kinds) if known else None
            seconds = milliseconds(cooperative.get('openmp_wall_ms' if endpoint == 'openmp' else 'cuda_driver_wall_ms'))
        result[endpoint] = dict(steps=steps, waves=waves, backend_call_s=seconds,
            steps_per_backend_call_s=steps/seconds if steps is not None and seconds else None,
            steps_per_wave=steps/waves if steps is not None and waves else None)
    return result


def event_record(run, entry):
    issues = []
    label = entry['label']
    if Path(label).name != label or label in ('.', '..'):
        raise ValueError('Unsafe event label: '+label)
    guard_dir = run/(label+'-guard')
    guard = read(guard_dir/'summary.json', issues)
    event = read(run/label/'gpu_em/summary.yaml', issues, True).get('shower_0', {})
    stats = event.get('statistics', {})
    timing = read(run/label/'simulation_timing/summary.yaml', issues, True).get('shower_0', {})
    cooperative = stats.get('accelerator', {}).get('cooperative', {})
    work = endpoint_work(stats, entry['mode'])
    measured = None
    try:
        # Reuse the existing integration semantics while owning/closing its
        # stream explicitly (the helper itself calls .open() without a with).
        with (guard_dir/'resources.jsonl').open() as stream:
            measured = resources(SimpleNamespace(open=lambda: stream))
        if measured['mean_cpu_core_equivalents'] is None:
            raise ValueError('insufficient CPU telemetry samples')
    except (OSError, ValueError, KeyError, ZeroDivisionError, IndexError) as error:
        issues.append('resource monitoring: '+str(error))
    queued, flushed = [stats.get(k) for k in
        ('deferred_cpu_fallbacks_queued', 'deferred_cpu_fallbacks_flushed')]
    total = stats.get('gpu_particles')
    endpoint_steps = [work[e]['steps'] for e in ('openmp', 'cuda')]
    endpoint_sum = sum(endpoint_steps) if all(v is not None for v in endpoint_steps) else None
    profile_steps = stats.get('profile', {}).get('steps')
    checks = dict(summary_closed=event.get('complete') is True,
        timing_closed=timing.get('closed') is True,
        guard_succeeded=guard.get('pass') is True and guard.get('returncode') == 0 and not guard.get('failure'),
        backend_matches=stats.get('accelerator', {}).get('backend') == entry['mode'],
        step_accounting=(endpoint_sum == total == profile_steps) if all(
            value is not None for value in (total, endpoint_sum, profile_steps)) else None,
        fallback_flushed=queued is not None and queued == flushed)
    return dict(label=label, family=entry['family'], seed=entry['seed'], mode=entry['mode'],
        process_s=guard.get('process_wall_s'), shower_s=milliseconds(timing.get('wall_time_ms')),
        completed=all(checks[k] for k in ('summary_closed', 'timing_closed', 'guard_succeeded', 'backend_matches')),
        summary_checks_passed=all(v is True for v in checks.values()), checks=checks,
        monitor_valid=guard.get('performance_valid') is True and measured is not None and not issues,
        guard_performance_valid=guard.get('performance_valid') is True,
        resources=measured, issues=issues, physics_flags=physics_flags(guard.get('command')),
        policy=cooperative.get('scheduling_policy'), endpoints=work, total_steps=total,
        cpu_step_share=work['openmp']['steps']/total if work['openmp']['steps'] is not None and total else None,
        scalar_s=milliseconds(stats.get('hybrid_timing_ms', {}).get('scalar_stepper')),
        below_minimum_checkpoints=stats.get('below_minimum_batch_checkpoints'),
        gpu_submissions=cooperative.get('subshower_cuda_submissions'),
        gpu_commits=cooperative.get('subshower_cuda_commits'),
        helper=dict(packets=cooperative.get('cuda_completion_packets'),
            continuations=cooperative.get('subshower_cuda_autonomous_continuations'),
            maximum_calls=cooperative.get('maximum_cuda_packet_calls'),
            retained_peak_bytes=cooperative.get('cuda_completion_peak_bytes'),
            blocking_enabled=cooperative.get('auxiliary_blocking_wait_enabled'),
            blocking_calls=cooperative.get('auxiliary_blocking_wait_calls'),
            blocking_wait_s=milliseconds(cooperative.get('auxiliary_blocking_wait_ms')),
            stop_order=cooperative.get('cuda_continuation_stop_order'),
            stops=cooperative.get('cuda_continuation_stops')),
        cpu_epochs=cooperative.get('subshower_openmp_epochs'),
        host_species_coalesces=cooperative.get('host_species_coalesces'),
        gpu_result_service_delay_s=milliseconds(cooperative.get('cuda_result_service_delay_ms')),
        coordinator_idle_wait_s=milliseconds(cooperative.get('coordinator_idle_wait_ms')),
        fallback=dict(queued=queued, flushed=flushed,
            flushes=stats.get('deferred_cpu_fallback_flushes'),
            maximum_batch=stats.get('maximum_deferred_cpu_fallback_batch'),
            configured_batch_limit=cooperative.get('specified_fallback_batch_limit')),
        energy_ledger_complete_coverage=stats.get('energy_ledger', {}).get('complete_coverage'))


def build_report(run):
    run = Path(run)
    issues = []
    state = read(run/'STATUS.json', issues)
    if not state:
        raise ValueError('; '.join(issues))
    identity = read(run/'PROVENANCE.json', issues)
    records = [event_record(run, entry) for entry in state.get('records', [])]
    keys = [(r['family'], r['seed'], r['mode']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate family/seed/mode in STATUS; refusing ambiguous pairing')
    groups = []
    for family, mode in sorted(set((r['family'], r['mode']) for r in records)):
        subset = [r for r in records if (r['family'], r['mode']) == (family, mode)]
        times = [r['process_s'] for r in subset if r['completed'] and r['process_s'] is not None]
        groups.append(dict(family=family, mode=mode, recorded=len(subset), completed=sum(r['completed'] for r in subset),
            monitor_valid=sum(r['monitor_valid'] for r in subset),
            median_process_s=statistics.median(times) if times else None,
            mean_process_s=statistics.mean(times) if times else None))
    refs = {(r['family'], r['seed']): r for r in records if r['mode'] == 'openmp'}
    pairs = []
    for dual in (r for r in records if r['mode'] == 'openmp-cuda'):
        base = refs.get((dual['family'], dual['seed']))
        if base is None:
            continue
        matched = base['physics_flags'] is not None and base['physics_flags'] == dual['physics_flags']
        usable = matched and base['completed'] and dual['completed'] and base['summary_checks_passed'] and dual['summary_checks_passed'] and bool(base['process_s']) and bool(dual['process_s'])
        a, b = [r['endpoints']['openmp']['steps_per_backend_call_s'] for r in (base, dual)]
        pairs.append(dict(family=dual['family'], seed=dual['seed'], physics_flags_match=matched,
            paired_completed=bool(usable), monitor_valid=base['monitor_valid'] and dual['monitor_valid'],
            openmp_process_s=base['process_s'], cpu_primary_process_s=dual['process_s'],
            process_ratio_openmp_over_dual=base['process_s']/dual['process_s'] if usable else None,
            cpu_throughput_ratio_dual_over_openmp=b/a if usable and a and b is not None else None))
    paired_groups = []
    for family in sorted(set(p['family'] for p in pairs)):
        selected = [p for p in pairs if p['family'] == family and p['paired_completed']]
        paired_groups.append(dict(family=family, n=len(selected), seeds=[p['seed'] for p in selected],
            monitor_valid=sum(p['monitor_valid'] for p in selected),
            ratio_of_paired_median_process_times=(statistics.median(p['openmp_process_s'] for p in selected)/
                statistics.median(p['cpu_primary_process_s'] for p in selected)) if selected else None))
    return dict(schema_revision=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), state_complete=state.get('complete') is True, phase=state.get('phase'),
        identity=identity, source_issues=issues, records=records, groups=groups, pairs=pairs,
        paired_groups=paired_groups, physics_equivalence_certified=False,
        performance_gate_certified=False,
        throughput_caveat='Backend-call time includes synchronization/copies/radio. Same seed does not imply identical endpoint workloads or shower trees.')


def markdown(report):
    def fmt(value):
        return 'NA' if value is None else '{:.3f}'.format(value)
    lines = ['# CPU 优先：逐种子性能与工作量诊断', '',
        '状态：{}；阶段：`{}`。'.format('运行器报告完成' if report['state_complete'] else '阶段性结果', report['phase']), '',
        '保留 STATUS 中所有记录，包括慢事件、监控无效和文件缺失记录。只对共有种子、相同物理参数、已关闭输出配对；不挑选最快事件。', '',
        '实际双端策略：'+', '.join(sorted({r['policy'] for r in report['records'] if r.get('policy')}))+'。', '',
        '| 事例 / seed | 模式 | 进程/s | shower/s | 已关闭 / 摘要检查 | 监控有效 | 平均CPU核当量 | GPU/% | 峰值RSS/GiB | 峰值显存/MiB |',
        '|---|---|---:|---:|---|---|---:|---:|---:|---:|']
    for r in report['records']:
        resource = r['resources'] or {}
        lines.append('| {} / {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['family'], r['seed'], r['mode'], fmt(r['process_s']), fmt(r['shower_s']),
            '{}/{}'.format(r['completed'], r['summary_checks_passed']), r['monitor_valid'], fmt(resource.get('mean_cpu_core_equivalents')),
            fmt(resource.get('mean_gpu_percent')), fmt(resource.get('rss_peak_gib')), fmt(resource.get('gpu_peak_mib'))))
    lines += ['', '## 端点有效工作', '',
        '| seed / 模式 | CPU步数 | CPU波前 | CPU调用/s | CPU百万步/s | GPU步数 | GPU波前 | GPU调用/s | GPU百万步/s | CPU步数占比 | scalar/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['records']:
        a, b = [r['endpoints'][e] for e in ('openmp', 'cuda')]
        rate = lambda e: fmt(e['steps_per_backend_call_s']/1e6 if e['steps_per_backend_call_s'] is not None else None)
        lines.append('| {} / {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['seed'], r['mode'], a['steps'], a['waves'], fmt(a['backend_call_s']), rate(a),
            b['steps'], b['waves'], fmt(b['backend_call_s']), rate(b), fmt(r['cpu_step_share']), fmt(r['scalar_s'])))
    if any(r.get('host_species_coalesces') is not None for r in report['records']):
        lines += ['', '## CPU小物种波前合并', '',
                  '| seed | 合并选择次数 | CPU工作段 |', '|---|---:|---:|']
        for r in report['records']:
            if r.get('host_species_coalesces') is not None:
                lines.append('| {} | {} | {} |'.format(
                    r['seed'], r['host_species_coalesces'], r['cpu_epochs']))
        lines += ['', '这是调度选择计数，不是丢弃粒子数、节省的波前数或时间；有合并并不保证净加速。']
    lines += ['', '## 回退与协调', '',
        '| seed / 模式 | 排队 | 已执行 | flush次数 | 最大批量 | batch上限 | GPU提交/领取 | CPU工作段 | GPU领取延迟/s | 协调器等待/s |',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in report['records']:
        f = r['fallback']
        lines.append('| {} / {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |'.format(
            r['seed'], r['mode'], f['queued'], f['flushed'], f['flushes'], f['maximum_batch'], f['configured_batch_limit'],
            r['gpu_submissions'], r['gpu_commits'], r['cpu_epochs'], fmt(r['gpu_result_service_delay_s']), fmt(r['coordinator_idle_wait_s'])))
    lines += ['', '## 辅助 GPU 自主续跑', '',
        '| seed / 模式 | 完成packet | 自主续跑调用 | packet最大调用数 | 保留结果峰值/MiB | 停止原因计数 |',
        '|---|---:|---:|---:|---:|---|']
    for r in report['records']:
        h = r['helper']
        if h['packets'] is None:
            continue
        stops = ', '.join('{}:{}'.format(name, count) for name, count in
                         zip(h['stop_order'] or [], h['stops'] or []))
        lines.append('| {} / {} | {} | {} | {} | {} | {} |'.format(
            r['seed'], r['mode'], h['packets'], h['continuations'], h['maximum_calls'],
            fmt(h['retained_peak_bytes']/2**20 if h['retained_peak_bytes'] is not None else None), stops))
    lines += ['', 'packet是协调器收取单位；续跑调用仍逐次记账。内存预算在下一次调用前检查，允许额外保留一个有界调用结果，不是严格峰值上限。', '',
        'CPU优先v5的阻塞等待仅作用于辅助GPU自己的stream；下列墙钟时间与CPU工作重叠，不代表自旋核时或纯kernel时间。', '']
    for r in report['records']:
        h = r['helper']
        if h['blocking_enabled'] is not None:
            lines.append('- {} / {}：blocking={}，调用{}次，等待墙钟{}s。'.format(
                r['seed'], r['mode'], h['blocking_enabled'], h['blocking_calls'], fmt(h['blocking_wait_s'])))
    lines += ['',
        '## 配对时间', '', '| 事例 | 共有完整种子数 | 监控有效对数 | 纯OpenMP/双端中位时间比 |', '|---|---:|---:|---:|']
    for p in report['paired_groups']:
        lines.append('| {} | {} | {} | {} |'.format(p['family'], p['n'], p['monitor_valid'], fmt(p['ratio_of_paired_median_process_times'])))
    lines += ['', '比值 > 1 表示本次双端时间较短；监控无效对仍保留，但不能用于严格性能通过结论。', '',
        '## 解释边界', '',
        '- 端点调用计时包含同步、复制及本端射电，不是纯 kernel 时间；两端计时有重叠，不能相加当成进程时间。',
        '- 相同输入种子不保证双端与单端拥有相同 shower 树或粒子状态组成。每步吞吐用于定位，不是同轨迹严格性能验收。',
        '- CPU 核当量包含等待/自旋；GPU 完成后领取延迟不等于 CPU 空闲等待。',
        '- 输出摘要检查仅验证关闭状态、步数账目和排队回退已执行，不替代全部输出数组、完整能量账本或物理统计验收。',
        '- 本报告不宣称 1% 物理等价、完整能量覆盖、3% 单端性能门禁或净加速验收通过。', '', '## 数据问题', '']
    problems = report['source_issues'] + ['{}: {}'.format(r['label'], item) for r in report['records'] for item in r['issues']]
    lines += ['- '+item for item in problems] if problems else ['未发现读取错误；监控和摘要检查状态仍以上表为准。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'REPORT_CN.md').write_text(markdown(report))
    print(json.dumps(dict(records=len(report['records']), complete=report['state_complete'], paired=report['paired_groups'])))


if __name__ == '__main__':
    main()
