#!/usr/bin/env python3
"""Read-only job inspection; write small timing/service diagnostics, never control jobs."""
import argparse
import json
from pathlib import Path


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
    records = []
    for entry in state['records']:
        label = f"{entry['family']}-{entry['seed']}-{entry['mode']}"
        c = entry['cooperative']
        records.append(dict(family=entry['family'], seed=entry['seed'], mode=entry['mode'],
            process_s=entry['process_s'], **resources(a.root/(label+'-guard/resources.jsonl')),
            cuda_call_s=c['cuda_driver_wall_ms']/1000,
            cuda_result_service_delay_s=c['cuda_result_service_delay_ms']/1000,
            host_epoch_mean_ms=c['openmp_wall_ms']/max(1,c['subshower_openmp_epochs']),
            host_epoch_max_ms=c['maximum_slice_ms'],
            cuda_submissions=c['subshower_cuda_submissions'],
            host_epochs=c['subshower_openmp_epochs'], migration_MB=c['migration_bytes']/1e6))
    active = a.root/(state['phase']+'-guard/resources.jsonl')
    progress = resources(active) if active.exists() else None
    report = dict(state=state['phase'], complete=state['complete'], records=records,
        unfinished_progress=progress, historical_best_100PeV_process_s=2176.256697,
        historical_best_100PeV_gpu_utilization_percent=85.59,
        policy='adaptive-v3-service-budget', statistical_acceptance=False)
    (a.root/'SERVICE_PERFORMANCE.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Adaptive-v3 调度响应与性能试跑', '',
        '仅双端 adaptive 调度修改；固定物理参数、20 线程、70% GPU 预算。', '',
        '| 样本 | 模式 | 总时间 s | GPU 平均 % | 平均逻辑核 | GPU 结果等待 s | CPU 平均批次 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['family']} / {r['seed']} | {r['mode']} | {r['process_s']:.2f} | "
            f"{r['mean_gpu_utilization_percent']:.1f} | {r['mean_logical_cpu_cores']:.2f} | "
            f"{r['cuda_result_service_delay_s']:.3f} | {r['host_epoch_mean_ms']:.3f} |")
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
