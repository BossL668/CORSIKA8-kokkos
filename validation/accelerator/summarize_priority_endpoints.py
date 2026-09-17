#!/usr/bin/env python3
"""Descriptive priority-mode timing report; never excludes a slow completed seed."""
import argparse
import json
from pathlib import Path
import statistics
from check_priority_lifecycle import correctness_source

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def resources(path):
    samples = [json.loads(s) for s in path.open()]
    cpu = [(s['elapsed_s'], s['cpu_seconds']) for s in samples if 'cpu_seconds' in s]
    mean_cpu = (cpu[-1][1]-cpu[0][1])/(cpu[-1][0]-cpu[0][0]) if len(cpu)>1 else None
    measured = [(a,b) for a,b in zip(samples,samples[1:]) if 'device_util_percent' in a]
    duration = sum(b['elapsed_s']-a['elapsed_s'] for a,b in measured)
    span = samples[-1]['elapsed_s']-samples[0]['elapsed_s'] if len(samples)>1 else 0
    gpu = sum((b['elapsed_s']-a['elapsed_s'])*a['device_util_percent'] for a,b in measured)/duration if duration else None
    return dict(mean_cpu_core_equivalents=mean_cpu, mean_gpu_percent=gpu,
        rss_peak_gib=max((s['rss_bytes'] for s in samples),default=0)/2**30,
        gpu_peak_mib=max((s.get('device_used_mib',0) for s in samples),default=0),
        gpu_missing_samples=sum('device_util_percent' not in s for s in samples),
        gpu_sample_time_coverage=duration/span if span else 0,
        min_available_gib=min((s['available_bytes'] for s in samples),default=0)/2**30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();state=json.loads((a.run/'STATUS.json').read_text())
    identity=json.loads((a.run/'PROVENANCE.json').read_text())
    regression_source, correctness_reuse = correctness_source(a.run, identity)
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for record in state['records']:
        r=dict(record);r['resources']=resources(a.run/(r['label']+'-guard/resources.jsonl'))
        records.append(r)
    summary=dict(complete=state['complete'], phase=state['phase'], identity=identity,
                 revised_plan=state.get('revised_plan'),
                 records=records, groups=[], paired=[])
    if correctness_reuse is not None:
        summary['correctness_reference_run'] = str(regression_source)
        summary['correctness_reuse_revalidated'] = True
    lines=['# 两种双端优先级：本机实测', '',
        '**'+('本轮运行已完成。' if state['complete'] else '阶段性结果；任务尚未完成。')+'**', '',
        '保留所有已完成种子，不剔除较慢事件。主时间采用 Popen 至独立 wait() 的完整进程时间；',
        'shower 内部时间另列。NVML 查询不确定的样本保留，但不用于宣布严格速度验收通过。', '',
        f"线程设置：{identity['threads']}；CPU affinity：`{identity['affinity']}`。", '',
        '| 事例 / seed | 模式 | 进程/s | shower/s | 计时监控完整 | 平均CPU核当量 | GPU/% | RSS/GiB |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    def fmt(x):return 'NA' if x is None else f'{x:.2f}'
    if state.get('revised_plan'):
        revised=state['revised_plan']
        lines[4:4]=[
            f"队列已按用户要求改为仅补齐 `{revised['only_mode']}`，目标 {revised['requested_cases']} 例。",
            '“完成”只针对修订后的目标；其余模式已完成样本保留，原三模式矩阵不再补跑。',
            '各组数量可能不等；只对共有种子计算配对时间比，不将不同样本数的中位数直接认定为加速。', '']
    for r in records:
        g=r['guard'];v=r['resources']
        lines.append(f"| {r['family']} / {r['seed']} | {r['mode']} | {g['process_wall_s']:.3f} | {r['shower_s']:.3f} | {'是' if g.get('performance_valid') else '否'} | {fmt(v['mean_cpu_core_equivalents'])} | {fmt(v['mean_gpu_percent'])} | {v['rss_peak_gib']:.3f} |")
    lines+=['','| 事例 | 模式 | 完成数 | 中位进程/s | 平均进程/s | 中位shower/s |','|---|---|---:|---:|---:|---:|']
    for family in sorted(set(r['family'] for r in records)):
        subset=[r for r in records if r['family']==family]
        modes=[m for m in identity['modes'] if any(r['mode']==m for r in subset)]
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        seeds=sorted(set(r['seed'] for r in subset))
        for i,mode in enumerate(modes):
            group=[r for r in subset if r['mode']==mode]
            wall=[r['guard']['process_wall_s'] for r in group]
            inside=[r['shower_s'] for r in group]
            group_summary=dict(family=family,mode=mode,n=len(group),
                median_process_s=statistics.median(wall),mean_process_s=statistics.mean(wall),
                median_shower_s=statistics.median(inside),
                monitor_valid=sum(r['guard'].get('performance_valid') is True for r in group))
            summary['groups'].append(group_summary)
            lines.append(f"| {family} | {mode} | {len(group)} | {statistics.median(wall):.3f} | {statistics.mean(wall):.3f} | {statistics.median(inside):.3f} |")
            xs=[seeds.index(r['seed'])+(i-(len(modes)-1)/2)*.23 for r in group]
            bars=axes[0].bar(xs,wall,.22,label=mode)
            for bar,r in zip(bars,group):
                if not r['guard'].get('performance_valid'):bar.set_hatch('//')
            axes[1].scatter([i]*len(group),wall,s=25)
            axes[1].plot([i-.18,i+.18],[statistics.median(wall)]*2,color='black')
        axes[0].set_xticks(range(len(seeds)),[str(s) for s in seeds],rotation=25)
        axes[0].set_xlabel('Input seed');axes[0].legend(fontsize=8)
        axes[1].set_xticks(range(len(modes)),modes,rotation=15)
        for ax in axes:
            ax.set_ylabel('Process wall time [s]');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{family}: priority policies (hatched = incomplete monitoring)')
        fig.savefig(a.output/(family+'_timing.png'),dpi=180);plt.close(fig)
        # Same seeds are paired for uncertainty estimates, not assumed identical trees.
        reference=next(m for m in identity['modes'] if m not in ('cuda-openmp','openmp-cuda'))
        refs={r['seed']:r for r in subset if r['mode']==reference}
        rng=np.random.default_rng(20260912)
        for mode in ('cuda-openmp','openmp-cuda'):
            pairs=[(refs[r['seed']],r) for r in subset if r['mode']==mode and r['seed'] in refs]
            if not pairs:continue
            x=np.array([r['guard']['process_wall_s'] for r,_ in pairs])
            y=np.array([r['guard']['process_wall_s'] for _,r in pairs])
            estimate=float(np.median(x)/np.median(y));ci=None
            if len(pairs)>=2:
                ids=rng.integers(0,len(pairs),(10000,len(pairs)))
                ci=np.quantile(np.median(x[ids],axis=1)/np.median(y[ids],axis=1),[.025,.975]).tolist()
            summary['paired'].append(dict(family=family,reference=reference,mode=mode,
                n=len(pairs),median_ratio=estimate,exploratory_95_interval=ci))
    lines+=['','## 实际工作与调度','',
        '| 事例 / seed | 模式 | 输运步数 | 驻留波前数 | GPU提交 | CPU工作段 | CPU工作/s | GPU调用/s | GPU结果领取延迟/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        work=r['accelerator'];c=work['accelerator'].get('cooperative',{})
        lines.append(f"| {r['family']} / {r['seed']} | {r['mode']} | {work['steps']:,} | {work['photon_waves']+work['lepton_waves']:,} | {c.get('subshower_cuda_commits','—')} | {c.get('subshower_openmp_epochs','—')} | {fmt(c['openmp_wall_ms']/1000 if 'openmp_wall_ms' in c else None)} | {fmt(c['cuda_driver_wall_ms']/1000 if 'cuda_driver_wall_ms' in c else None)} | {fmt(c['cuda_result_service_delay_ms']/1000 if 'cuda_result_service_delay_ms' in c else None)} |")
    lines+=['','GPU 调用时段包含其同步/传输，不是纯 kernel 时间；CPU 与 GPU 时段会重叠，不能直接相加得到总时间。',
        '领取延迟是 GPU 完成工作后至协调器领取结果的累计延迟。GPU 采样缺失及时间覆盖率见 REPORT.json。',
        '', '## 解释边界','',
        '- cuda-openmp 保留原 GPU 优先策略；openmp-cuda 是新增 CPU 优先策略。两者均为动态分配。',
        ('- 本批 CPU 优先存在额外的整数 profile 分片或指定回退合批；模式耗时差不能全部归因于 GPU 的额外贡献。'
         if any(r['accelerator']['accelerator'].get('cooperative',{}).get('host_profile_shards',0)>0
                or r['accelerator']['accelerator'].get('cooperative',{}).get('specified_fallback_batch_limit',1)>1
                for r in records if r['mode']=='openmp-cuda') else
         '- 本批 CPU 优先没有专属 profile 分片或指定回退合批；仍需区分调度开销、动态 shower 工作量及设备状态。'),
        '- CPU 核当量包含 OpenMP 自旋与驱动；GPU 利用率是设备采样，不等于有效物理工作重叠。',
        '- 比较时同时查看真实输运步数、驻留波前与提交次数，不把步数称为独立粒子数。',
        '- 五种子短测及探索性 bootstrap 不替代 500 例统计、1% 等价门禁或多次高能复测。',
        (f'- 新旧单端输出回归、两端 EM/射电能量账本和 N=32 记录复用并重新校验自 `{regression_source}`；本运行不重复生成这些记录。'
         if correctness_reuse is not None else
         '- 新旧单端输出回归、两端 EM/射电能量账本和 N=32 记录另见本运行目录。'), '']
    for family in sorted(set(r['family'] for r in records)):
        lines.append(f'![{family} 运行时间]({family}_timing.png)')
    (a.output/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    (a.output/'REPORT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=state['complete'],records=len(records),groups=summary['groups'])))


if __name__=='__main__':main()
