#!/usr/bin/env python3
"""Wait for one guarded pilot, then produce a small timing/resource report."""
import argparse
import json
from pathlib import Path
import time


def read_json(path):
    # The simulation writes in another process; a newly visible file may be partial.
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(1)


def write_failure(root, monitor):
    failure = dict(complete=False, failure=monitor.get('failure'),
                   returncode=monitor.get('returncode'), speedup_available=False)
    (root/'PERFORMANCE_RESULT.json').write_text(json.dumps(failure, indent=2)+'\n')
    (root/'PERFORMANCE_REPORT_CN.md').write_text(
        '# 100 PeV 双端测试未完成\n\n'
        f"监控器失败原因：{failure['failure']}；退出码：{failure['returncode']}。\n\n"
        '不使用不完整事件计算加速比。保留所有失败日志，不自动重试或替换种子。\n')
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=7800)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root/'run/timing.json'
    guard_path = root/'run/independent20-guard/summary.json'
    deadline = time.monotonic()+args.wait_seconds
    while not result.exists():
        if guard_path.exists():
            guard = read_json(guard_path)
            if not guard['pass']:
                return write_failure(root, guard)
        if time.monotonic()>deadline:
            (root/'REPORT_WAITER_TIMEOUT.json').write_text(json.dumps(
                dict(reason='No complete timing result within the bounded wait; simulation not signalled.')))
            return 2
        time.sleep(10)
    # The writer is a separate process; allow a just-created file to finish.
    for _ in range(10):
        try:
            current = json.loads(result.read_text())['independent20']
            break
        except (json.JSONDecodeError, KeyError):
            time.sleep(1)
    else:
        raise RuntimeError('incomplete timing JSON')
    if not current['monitor']['pass'] or current['returncode'] != 0:
        return write_failure(root, current['monitor'])
    stats = current['physical_statistics']
    assert stats['queue_overflows']==0
    assert stats['profile']['fixed_point_overflows']==0
    assert stats['radio']['fixed_point_overflows']==0
    c = stats['accelerator']['cooperative']
    assert c['independent_drivers'] and c['independent_joint_calls']>0
    assert c['cuda_input_particles']>0 and c['openmp_input_particles']>0
    subshowers = c.get('independent_subshowers', False)
    if subshowers:
        assert c['subshower_cuda_submissions']==c['subshower_cuda_commits']
        assert c['subshower_openmp_epochs']>0
    old = json.loads((args.reference/'TIMING_RESULT.json').read_text())
    assert old['monitor']['pass']
    manifest = json.loads((root/'run/independent20-command.json').read_text())
    def physics(command):
        command = list(command[1:])
        i=command.index('-f');del command[i:i+2]
        return command
    assert physics(manifest['command'])==physics(old['command'])
    rows = [json.loads(line) for line in
            (root/'run/independent20-guard/resources.jsonl').read_text().splitlines()]
    cpu = [r for r in rows if 'cpu_seconds' in r]
    average = ((cpu[-1]['cpu_seconds']-cpu[0]['cpu_seconds']) /
               (cpu[-1]['elapsed_s']-cpu[0]['elapsed_s']))
    process_s = current['monitor']['elapsed_s']
    reference_s = old['reference_process_wall_s']
    old_s = old['monitor']['elapsed_s']
    report = dict(complete=True, single_event_only=True, large_sample_acceptance=False,
                  new_process_s=process_s, new_shower_s=current['shower_wall_s'],
                  historical_cuda_process_s=reference_s, old_cooperative_process_s=old_s,
                  cuda_over_new_ratio=reference_s/process_s,
                  old_cooperative_over_new_ratio=old_s/process_s,
                  average_process_cpu_core_equivalent=average,
                  cooperative=c, binary_sha256=current['binary_sha256'],
                  peak_RSS_MiB=current['monitor']['peak_tree_rss_bytes']/2**20,
                  minimum_available_GiB=current['monitor']['minimum_available_bytes']/2**30)
    (root/'PERFORMANCE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True)
    minutes=[r['elapsed_s']/60 for r in cpu]
    cores=[(b['cpu_seconds']-a['cpu_seconds'])/(b['elapsed_s']-a['elapsed_s'])
           for a,b in zip(cpu,cpu[1:])]
    axes[0].plot(minutes[1:],cores,color='#377eb8',lw=.8)
    axes[0].axhline(20,color='grey',ls='--',lw=.7,label='20 OpenMP threads requested')
    axes[0].set_ylabel('Process CPU core equivalent')
    axes[0].legend(fontsize=9)
    axes[1].plot(minutes,[r.get('device_util_percent',float('nan')) for r in cpu],
                 color='#e69f00',lw=.8)
    axes[1].set_ylabel('Device-wide GPU utilization [%]')
    axes[1].set_ylim(0,105)
    axes[2].plot(minutes,[r['rss_bytes']/2**30 for r in cpu],label='Process RSS')
    axes[2].plot(minutes,[r.get('device_used_mib',float('nan'))/1024 for r in cpu],
                 label='Device-wide VRAM')
    axes[2].set_ylabel('Memory [GiB]');axes[2].set_xlabel('Elapsed time [min]')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2);ax.set_xlim(0,minutes[-1])
    fig.suptitle('100 PeV proton: independent CUDA + OpenMP (20 threads)')
    fig.tight_layout()
    fig.savefig(root/'resource_timeline.png',dpi=170,bbox_inches='tight')
    plt.close(fig)
    if subshowers:
        waiting = (
            f'协调器没有可运行主机工作时的轮询等待：{c["coordinator_idle_wait_ms"]/1000:.3f} s；'
            f'CUDA 结果完成至领取的累计延迟：{c["cuda_result_service_delay_ms"]/1000:.3f} s。\n\n'
            f'CUDA 提交/领取：{c["subshower_cuda_submissions"]}/{c["subshower_cuda_commits"]}；'
            f'OpenMP 工作段：{c["subshower_openmp_epochs"]}；单个 CUDA 任务在途期间最多 '
            f'{c["maximum_host_epochs_per_cuda_job"]} 个 OpenMP 工作段。\n\n'
            '旧成对批次 tail-wait 字段不适用，不能把其零值解释为零等待。'
            '上述为主机调度诊断，不是设备 kernel 的实际重叠时间。\n\n')
    else:
        waiting = (
            f'GPU 先结束后等 CPU：{c["cuda_finished_before_host_ms"]/1000:.3f} s；'
            f'CPU 先结束后等 GPU：{c["host_finished_before_cuda_ms"]/1000:.3f} s。'
            '两个数是调用窗口的等待诊断，不是设备 kernel 的实际重叠时间。\n\n')
    (root/'PERFORMANCE_REPORT_CN.md').write_text(
        '# 100 PeV 独立双端：单事件诊断\n\n'
        '质子，theta=47°，phi=180°，emthin=1e-6，seed=2026110001，20 OpenMP 线程，完整射电。\n\n'
        '| 路径 | 进程总耗时 [s] |\n|---|---:|\n'
        f'| 历史单 CUDA | {reference_s:.3f} |\n'
        f'| 旧回调式双端 | {old_s:.3f} |\n'
        f'| 新独立双端 | {process_s:.3f} |\n\n'
        f'新双端 shower 时间：{current["shower_wall_s"]:.3f} s。'
        f'历史单 CUDA/新双端耗时比：{reference_s/process_s:.4f}；'
        f'旧双端/新双端：{old_s/process_s:.4f}。\n\n'
        '仅同种子单事件、历史二进制不同；动态调度允许不同 shower 树。'
        '这些是描述性比值，不是多种子中位数验收或统计物理验收。\n\n'
        f'进程平均 CPU 核当量：{average:.2f}（包含驱动及忙等，不等于纯物理计算）；'
        f'峰值 RSS：{report["peak_RSS_MiB"]:.1f} MiB。\n\n'
        + waiting +
        '正常结束，且队列/profile/radio 溢出计数为零。加速器局部能量账本'
        '若未覆盖标量/强子过程，不能据它宣布全 shower 能量闭合。\n\n'
        '![资源时间线](resource_timeline.png)\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
