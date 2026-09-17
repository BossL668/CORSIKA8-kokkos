#!/usr/bin/env python3
"""Read small timing logs on PSR; report measured progress, never percent done."""
import argparse
import datetime
import json
import math
import os
import pathlib
import socket


def json_lines(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The writer may still be completing its last line.
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=pathlib.Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'psrpku2025', 'PSR only'
    root = args.root.resolve()
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    progress = json.loads((root / 'progress.json').read_text())
    cases = []
    plots = []
    for folder in sorted((root / 'timing/runs').glob('*')):
        samples = json_lines(folder / 'live_steps.jsonl')
        nonzero = [r for r in samples if r['steps'] > 0]
        if len(nonzero) < 2:
            continue
        last = nonzero[-1]
        recent = [r for r in nonzero if r['elapsed_s'] >= last['elapsed_s'] - 300]
        if len(recent) < 2:
            recent = nonzero
        first = recent[0]
        duration = last['elapsed_s'] - first['elapsed_s']
        rate = (last['steps'] - first['steps']) / duration
        affinity = json.loads((folder / 'affinity.json').read_text())
        row = dict(tag=folder.name, elapsed_s=last['elapsed_s'], steps=last['steps'],
                   recent_steps_per_s=rate, recent_interval_s=duration,
                   first_nonzero_log_s=nonzero[0]['elapsed_s'],
                   physical_cores=affinity.get('physical_cores'),
                   affinity_verified=affinity.get('verified'),
                   compressed_track_bytes=last['compressed_bytes'])
        if rate > 0:
            row['conditional_hours_for_1e8_steps'] = 1e8 / rate / 3600
            row['conditional_hours_for_1e9_steps'] = 1e9 / rate / 3600
        status = folder / 'status.json'
        row['process_status'] = json.loads(status.read_text()) if status.exists() else 'running'
        cases.append(row)
        plots.append((folder.name, samples))
    out = root / 'report'
    record = dict(updated=stamp, campaign_state=progress['state'], cases=cases,
                  scope='Recorded step throughput only. No completed-shower claim, completion percentage or primary-energy extrapolation.')
    (out / 'timing_progress_snapshot.json').write_text(json.dumps(record, indent=2) + '\n')
    state_text = '试跑仍在进行。'
    if progress['state'] == 'timing_pilot_incomplete':
        state_text = '**10 TeV 试跑到达 30 分钟上限后停止，清理进程另用了约 11 秒；完整 shower 和最终射电输出均未完成。更高能量没有启动。**'
    elif progress['state'] == 'timing_pilot_complete':
        state_text = '低能计时阶梯已结束，完整事件的验收结果见下方计时报告。'
    md = ['# 低能试跑的实时计时', '', '快照时间：' + stamp + '。', '', state_text, '',
          '正式高能事件的状态以 [当前计时报告](TIMING_CN.md) 和 [队列状态](../progress.json) 为准。', '',
          '| 试跑 | 已运行/min | 已记录步数 | 最近约 5 min 的步/秒 | 物理核绑定 |',
          '|---|---:|---:|---:|---|']
    for row in cases:
        md.append('| %s | %.2f | %s | %.0f | %s |' %
                  (row['tag'], row['elapsed_s'] / 60, format(row['steps'], ','),
                   row['recent_steps_per_s'], '256 核已核实' if row['affinity_verified'] else '未核实'))
    if cases and cases[-1]['recent_steps_per_s'] > 0:
        r = cases[-1]
        md += ['', '用最新实测速率换算：如果一个事件有 10⁸ 步，需要约 **%.1f 小时**；如果有 10⁹ 步，需要约 **%.1f 小时**。这是假设平均速度相同的条件估计，不是对 300 PeV / 1 EeV 总时间的预测。' %
               (r['conditional_hours_for_1e8_steps'], r['conditional_hours_for_1e9_steps'])]
    md += ['', '步数仍在增长表示计算仍有进展。输运采用队列调度，记录中的物理时间并不单调，也不能除以 500 μs 当作完成比例。最终仍须检查完整输出、CC/τ 事件记录、队列收尾以及射电结果。', '',
           '保持 EM 截断 0.5 MeV、强子/μ/τ 截断 0.3 GeV、emthin=1e-5 和原最大权重公式；10 TeV 的 Wmax=0.05，单位权重粒子实际上不触发 thinning。不能按初级能量倍数直接外推到高能。', '']
    witness = root / 'timing/live_cc_tau_witnesses.json'
    if witness.exists():
        evidence = json.loads(witness.read_text())
        taus = evidence['tau_tracks']
        children = evidence['daughter_first_tracks']
        if len(taus) == 1 and {int(x['pdg']) for x in children} == {16, -211, 111}:
            t = taus[0]
            distance = math.sqrt(sum((float(t[k+'1_m']) - float(t[k+'0_m']))**2 for k in 'xyz'))
            md += ['轨迹前缀已确认：10 TeV 样本的初级 ντ 产生了约 7.58 TeV 的 τ，τ 随后产生 π⁻、π⁰ 和 ντ；两处顶点相距约 %.2f cm。它确实有 shower，但不满足高能样本要求的 ≥100 m 顶点间距，只用于计时。这个前缀检查仍不能替代完整事件验收。' % (100*distance), '']
    if args.plot and plots:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
        for tag, samples in plots:
            energy, seed = tag.rsplit('_', 2)[-2:]
            assert energy.endswith('GeV')
            label = '%g TeV; %s' % (float(energy[:-3]) / 1000, seed.replace('seed', 'seed '))
            axes[0].plot([r['elapsed_s'] / 60 for r in samples],
                         [r['steps'] / 1e6 for r in samples], label=label)
            window = []
            for i in range(5, len(samples)):
                a, b = samples[i-5], samples[i]
                dt = b['elapsed_s'] - a['elapsed_s']
                if dt > 0:
                    window.append((b['elapsed_s'] / 60, (b['steps'] - a['steps']) / dt))
            axes[1].plot([x for x, _ in window], [y for _, y in window], label=label)
        axes[0].set_ylabel('Recorded transport steps (million)')
        axes[1].set_ylabel('Recorded steps / second (~30 s average)')
        for ax in axes:
            ax.set_xlabel('Process elapsed time (min)')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        title = 'SiO2 timing pilot | OpenMP, 256 physical cores | CoREAS + ZHS'
        if progress['state'] == 'timing_pilot_incomplete':
            title += '\nStopped at timing limit; incomplete event'
        fig.suptitle(title, fontsize=12)
        fig.savefig(out / 'timing_progress.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    if (out / 'timing_progress.png').exists():
        md += ['![Measured timing progress](timing_progress.png)', '',
               '左图：累计记录步数；右图：约 30 秒窗口内的吞吐率。曲线变平才表示没有新增轨迹，是否完成仍由最终状态判断。', '']
    (out / 'LIVE_TIMING_CN.md').write_text('\n'.join(md))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
