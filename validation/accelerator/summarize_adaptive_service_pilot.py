#!/usr/bin/env python3
"""Read-only job inspection; write small timing/service diagnostics, never control jobs."""
import argparse
import json
from pathlib import Path
import random
import statistics


def paired_timing_summary(records):
    """Compare complete same-seed inputs, not necessarily identical shower trees."""
    comparisons = {}
    for family in sorted({r['family'] for r in records}):
        by_seed = {}
        for r in records:
            if r['family'] == family:
                modes = by_seed.setdefault(r['seed'], {})
                if r['mode'] in modes:
                    raise ValueError('duplicate completed mode/seed')
                if not r['process_s'] > 0:
                    raise ValueError('nonpositive completed timing')
                modes[r['mode']] = r
        for reference in ('cuda', 'openmp', 'legacy'):
            pairs = [(m[reference], m['adaptive']) for _, m in sorted(by_seed.items())
                     if reference in m and 'adaptive' in m]
            if not pairs:
                continue
            n = len(pairs)
            base = statistics.median(p[0]['process_s'] for p in pairs)
            dual = statistics.median(p[1]['process_s'] for p in pairs)
            result = dict(paired_seeds=n, seeds=[p[0]['seed'] for p in pairs],
                reference_median_s=base, adaptive_median_s=dual,
                reference_over_adaptive_median_ratio=base/dual,
                individual_time_ratios=[p[0]['process_s']/p[1]['process_s'] for p in pairs],
                five_seed_minimum_reached=n>=5,
                performance_certified=False, statistical_physics_acceptance=False)
            if n >= 5:
                rng = random.Random(20260912)
                boot = []
                for _ in range(10000):
                    draw = [pairs[rng.randrange(n)] for _ in range(n)]
                    boot.append(statistics.median(p[0]['process_s'] for p in draw)/
                                statistics.median(p[1]['process_s'] for p in draw))
                boot.sort()
                result['exploratory_pair_bootstrap95_median_ratio'] = [boot[249], boot[9749]]
                result['bootstrap_caveat'] = ('Few distinct seeds; conditional on these runs. '
                    'Does not cover drift, thermals, other work or cross-hardware uncertainty.')
            comparisons[family+'/'+reference+'_vs_adaptive'] = result
    return comparisons


def resources(path):
    previous = last = None
    weight = weighted_gpu = peak = 0.
    for line in path.open():
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue  # live writer may not have finished the final line
        if previous and 'device_util_percent' in sample:
            dt = sample['elapsed_s']-previous['elapsed_s']
            weight += dt
            weighted_gpu += dt*sample['device_util_percent']
        peak = max(peak, sample['rss_bytes'])
        previous = last = sample
    return dict(elapsed_sample_s=last['elapsed_s'] if last else None,
        mean_gpu_utilization_percent=weighted_gpu/weight if weight else None,
        mean_logical_cpu_cores=last.get('cpu_seconds',0)/last['elapsed_s'] if last else None,
        peak_rss_gib=peak/2**30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    state = json.loads((a.root/'STATUS.json').read_text())
    provenance = json.loads((a.root/'PROVENANCE.json').read_text())
    records = []
    for entry in state['records']:
        label = f"{entry['family']}-{entry['seed']}-{entry['mode']}"
        c = entry['cooperative']
        record=dict(family=entry['family'], seed=entry['seed'], mode=entry['mode'],
            process_s=entry['process_s'], **resources(a.root/(label+'-guard/resources.jsonl')),
            transport_records=entry.get('transport_records'), radio_tracks=entry.get('radio_tracks'),
            cuda_call_s=c.get('cuda_driver_wall_ms',0)/1000 if c else None,
            cuda_result_service_delay_s=c.get('cuda_result_service_delay_ms',0)/1000 if c else None,
            host_epoch_mean_ms=c.get('openmp_wall_ms',0)/max(1,c.get('subshower_openmp_epochs',0)) if c else None,
            host_epoch_max_ms=c.get('maximum_slice_ms'),
            cuda_submissions=c.get('subshower_cuda_submissions'),
            host_epochs=c.get('subshower_openmp_epochs'),
            migration_MB=c.get('migration_bytes',0)/1e6 if c else None,
            cuda_packets=c.get('cuda_completion_packets'),
            cuda_autonomous_calls=c.get('subshower_cuda_autonomous_continuations'))
        if 'adaptive' in c:
            record['physical_work']={}
            for endpoint in ('cuda','openmp'):
                ep=c['adaptive'][endpoint]
                record['physical_work'][endpoint]=dict(
                    transport_records=sum(ep[k]['transport_records'] for k in ('photon','lepton')),
                    resident_wavefronts=sum(ep[k]['resident_wavefronts'] for k in ('photon','lepton')),
                    job_input_histogram_floor_log2=ep['job_input_histogram_floor_log2'],
                    species_coalesces=ep['species_coalesces'])
        records.append(record)
    active = a.root/(state['phase']+'-guard/resources.jsonl')
    progress = resources(active) if active.exists() else None
    report = dict(state=state['phase'], complete=state['complete'], records=records,
        paired_timing=paired_timing_summary(records),
        unfinished_progress=progress, historical_best_100PeV_process_s=2176.256697,
        historical_best_100PeV_gpu_utilization_percent=85.59,
        policy=provenance['policy'], statistical_acceptance=False)
    (a.root/'SERVICE_PERFORMANCE.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# '+provenance['policy']+' 调度响应与性能试跑', '',
        '固定物理参数；双端20线程，单CUDA主机调度1线程；GPU预算70%。只纳入已完成配对，运行中样本不计入。', '',
        '| 样本 | 模式 | 总时间 s | GPU 平均 % | 平均逻辑核 | GPU 结果等待 s | CPU 平均批次 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        delay = '—' if r['cuda_result_service_delay_s'] is None else f"{r['cuda_result_service_delay_s']:.3f}"
        epoch = '—' if r['host_epoch_mean_ms'] is None else f"{r['host_epoch_mean_ms']:.3f}"
        lines.append(f"| {r['family']} / {r['seed']} | {r['mode']} | {r['process_s']:.2f} | "
            f"{r['mean_gpu_utilization_percent']:.1f} | {r['mean_logical_cpu_cores']:.2f} | "
            f"{delay} | {epoch} |")
    for name, comparison in report['paired_timing'].items():
        lines += ['', f"{name}：{comparison['paired_seeds']}个完整种子；中位时间 "
            f"{comparison['reference_median_s']:.3f} → {comparison['adaptive_median_s']:.3f}s，"
            f"描述性比值{comparison['reference_over_adaptive_median_ratio']:.3f}×。"]
        if 'exploratory_pair_bootstrap95_median_ratio' in comparison:
            lo, hi = comparison['exploratory_pair_bootstrap95_median_ratio']
            lines.append(f"按种子对重采样的探索性95%区间[{lo:.3f}, {hi:.3f}]；"
                         '少种子区间不覆盖温度、频率和其他任务的系统误差，不作为正式性能门禁通过。')
    lines += ['', 'GPU 利用率来自按采样间隔加权的设备级监测，不是内核重叠测量。',
        'GPU 结果等待是完成到协调器领取的时间，与 CPU 有用计算重叠，不能直接作为可扣除时间。',
        '同种子动态调度可能产生不同 shower tree。少量试跑不证明统计等价或稳定加速。', '',
        '100 PeV 最佳旧独立双端参考为 **2176.256697 s / 36.27 min**，不是 56.79 min。',
        '被中止的 adaptive-v2 >81 min 样本不计作已完成事件。', '',
        '当前状态：`'+state['phase']+'`。']
    if progress:
        lines += ['', f"当前未完成任务已采样 {progress['elapsed_sample_s']:.1f} s；"
            f"平均 GPU {progress['mean_gpu_utilization_percent']:.1f}%，"
            f"平均 {progress['mean_logical_cpu_cores']:.2f} 个逻辑核。"]
    (a.root/'SERVICE_PERFORMANCE_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
