#!/usr/bin/env python3
"""Combine complete local/PSR pilots, without treating unlike hardware as a speed-up pair."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def priority_policy(records):
    policies={r['accelerator']['accelerator']['cooperative']['scheduling_policy']
              for r in records if r['family']=='Fe100TeV' and r['mode']=='openmp-cuda'}
    assert len(policies)==1, 'mixed or missing CPU-priority policy versions'
    return next(iter(policies))


def verify_rows(rows, modes, required_mode=None):
    seeds=set(range(85000001,85000006))
    keys=[(r['seed'],r['mode']) for r in rows]
    assert len(keys)==len(set(keys)), 'duplicate timing sample'
    assert all(seed in seeds and mode in modes for seed,mode in keys), 'unexpected timing sample'
    for mode in ([required_mode] if required_mode else modes):
        assert {seed for seed,m in keys if m==mode}==seeds, 'required mode has missing seeds'
    if required_mode is None:
        assert len(rows)==15, 'full matrix incomplete'


def load_run(root, required_mode=None):
    report=json.loads((root/'performance/REPORT.json').read_text())
    audit=json.loads((root/'COMPLETED_OUTPUT_AUDIT.json').read_text())
    gates=json.loads((root/'CORRECTNESS_GATES.json').read_text())
    life=json.loads((root/'N32_LIFECYCLE_AUDIT.json').read_text())
    assert report['complete'] and audit['all_planned_runs_complete'] and gates['passed']
    assert audit['completed_outputs_verified']
    for mode in ('cuda-openmp','openmp-cuda'):
        assert life[mode]['pass_'] and len(life[mode]['events'])==32
    groups=[g for g in report['groups'] if g['family']=='Fe100TeV']
    rows=[r for r in report['records'] if r['family']=='Fe100TeV']
    verify_rows(rows, report['identity']['modes'], required_mode)
    for group in groups:
        assert group['n']==sum(r['mode']==group['mode'] for r in rows)
    assert {r['label'] for r in rows}=={r['label'] for r in audit['records']
                                      if r['family']=='Fe100TeV'}, 'audit/report sample mismatch'
    report['cpu_priority_policy']=priority_policy(rows)
    hashes={r['physics']['sha256'] for r in audit['records'] if r['family']=='Fe100TeV'}
    assert len(hashes)==1
    return report,next(iter(hashes))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--placement-only',action='store_true',
        help='Require local GPU-primary and server CPU-primary only, per revised user queue')
    a=p.parse_args()
    local,lh=load_run(a.local, 'cuda-openmp' if a.placement_only else None)
    server,sh=load_run(a.server, 'openmp-cuda' if a.placement_only else None)
    assert lh==sh,'actual primary/geometry/radio/CLI conditions differ between hosts'
    assert local['cpu_priority_policy']==server['cpu_priority_policy'], 'different priority policies between hosts'
    a.output.mkdir(parents=True,exist_ok=True)
    lines=['# GPU 优先与 CPU 优先：本地 / PSR 短测', '',
        '两种模式的实现和两机短测完成不等于生产性能通过。每台机器用自身的单端结果比较；',
        '不将 RTX4060 Laptop 与 T400 服务器的秒数直接相除作为调度加速比。', '',
        'Fe56，100TeV，垂直，emthin=1e-6，默认自动 max-weight，完整 CoREAS/ZHS，GPU 预算70%。',
        '两机相同的五个输入种子；实际初级、磁场、天线、时间网格及非设备 CLI 配置已核对。', '',
        '两机 CPU 优先策略均为 `'+local['cpu_priority_policy']+'`，不混合不同调度版本。', '',
        '| 机器 | 模式 | 例数 | 中位进程/s | 平均进程/s | 中位shower/s | 完整计时监控 |',
        '|---|---|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,(name,report) in zip(axes,(('Local RTX4060 + 20 threads',local),('PSR T400 + 130 threads',server))):
        groups=[g for g in report['groups'] if g['family']=='Fe100TeV']
        selected='cuda-openmp' if report is local else 'openmp-cuda'
        modes=[m for m in report['identity']['modes'] if m not in ('cuda-openmp','openmp-cuda')]+(
            [selected] if a.placement_only else ['cuda-openmp','openmp-cuda'])
        for i,mode in enumerate(modes):
            g=next(g for g in groups if g['mode']==mode)
            rows=[r for r in report['records'] if r['family']=='Fe100TeV' and r['mode']==mode]
            bar=ax.bar(i,g['median_process_s'],.65,color=plt.get_cmap('tab10')(i))
            if g['monitor_valid']!=g['n']:bar[0].set_hatch('//')
            ax.scatter(np.linspace(i-.12,i+.12,len(rows)),[r['guard']['process_wall_s'] for r in rows],color='black',s=14,zorder=3)
            lines.append(f"| {name} | {mode} | {g['n']} | {g['median_process_s']:.3f} | {g['mean_process_s']:.3f} | {g['median_shower_s']:.3f} | {g['monitor_valid']}/{g['n']} |")
        labels=[f"{mode}\n(n={next(g['n'] for g in groups if g['mode']==mode)})" for mode in modes]
        ax.set_xticks(range(len(modes)),labels,rotation=10)
        ax.set_title(name);ax.set_ylabel('Process wall time [s]');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fe100TeV: '+('host-specific priority tests' if a.placement_only else 'five seeds per mode')+
                 ' (hatched = monitoring gaps)')
    fig.savefig(a.output/'priority_modes_two_hosts.png',dpi=180);plt.close(fig)
    if a.placement_only:
        lines+=['','按最新要求：本地只补 GPU 优先，PSR 只补 CPU 优先，所选模式各 5 例均完整。',
            '已完成的其他模式保留在原报告。本表单端参照数量可能不足 5 例；PSR 改队列后不再三模式交替运行。',
            '不将不同样本数的组中位数之比当作加速证据。下面仅列同机器、同种子的已完成配对：', '',
            '| 机器 | 配对 | 共有种子数 | 单端中位 / 双端中位 |', '|---|---|---:|---:|']
        for name,report,mode in [('本地',local,'cuda-openmp'),('PSR',server,'openmp-cuda')]:
            for pair in report['paired']:
                if pair['family']=='Fe100TeV' and pair['mode']==mode:
                    lines.append(f"| {name} | {pair['reference']} / {mode} | {pair['n']} | {pair['median_ratio']:.3f} |")
        lines+=['','比值小于 1 表示本组双端更慢。少量种子及监控缺口不满足严格性能认证。']
    lines+=['','时间是启动至 wait() 完成的进程时间，启动前等待显卡空闲不计入。',
        '保留所有已完成事件；有 GPU 监控缺口的样本不能用于宣布严格加速通过。',
        '完整 CPU 核时间包含忙等，GPU 采样非零也不等于两端持续有效重叠。',
        'CPU 优先的附加 profile 分片/回退合批是否启用，以各机原始 metadata 和本机报告为准；不能只按模式名推断。',
        '固定种子单端回归、真实双端 EM/射电通量、N=32、输出/队列检查均已检查；',
        '这不是 500 例物理统计等价，也不将完整强子 shower 的不完整能量账本认定为闭合。',
        '本轮没有复测 100PeV，不由这些短测推断已经恢复历史 36 分钟表现。', '',
        '![两机时间比较](priority_modes_two_hosts.png)']
    (a.output/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    (a.output/'SUMMARY.json').write_text(json.dumps(dict(physics_sha256=lh,
        placement_only=a.placement_only,
        cpu_priority_policy=local['cpu_priority_policy'],
        local_groups=local['groups'],server_groups=server['groups'],
        local_paired=local['paired'],server_paired=server['paired']),indent=2)+'\n')
    print(a.output/'REPORT_CN.md')


if __name__=='__main__':main()
