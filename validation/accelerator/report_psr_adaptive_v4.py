#!/usr/bin/env python3
"""Read-only streaming telemetry report. Does not control simulation processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def telemetry(path):
    count = sum(1 for _ in path.open())
    stride = max(1, count//1500)
    points = []
    start = prev = last = None
    gpu_sum = gpu_span = peak_cpu = peak_rss = peak_mem = 0.
    masks = set()
    driver_masks = set()
    max_threads = 0
    for i,line in enumerate(path.open()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        gpu = row.get('gpu','').split(',')
        try:
            mem,util = map(float, gpu[:2])
        except (ValueError,TypeError):
            mem = util = float('nan')
        start = start or row
        dt = row['elapsed_s']-prev['elapsed_s'] if prev else 0.
        cpu = 100*(row['cpu_s']-prev['cpu_s'])/dt if dt else 0.
        if dt and np.isfinite(util):
            gpu_sum += util*dt
            gpu_span += dt
        peak_cpu = max(peak_cpu,cpu)
        peak_rss = max(peak_rss,row['rss_bytes']/2**30)
        if np.isfinite(mem): peak_mem = max(peak_mem,mem)
        threads = row.get('threads',{})
        max_threads = max(max_threads,len(threads))
        for thread in threads.values():
            masks.update(thread['affinity'])
            if thread['name']=='c8-cuda-driver': driver_masks.update(thread['affinity'])
        if i%stride==0:
            points.append([row['elapsed_s'],cpu,util,mem,row['rss_bytes']/2**30])
        prev = last = row
    span = last['elapsed_s']-start['elapsed_s'] if last else 0.
    return dict(sample_elapsed_s=last['elapsed_s'] if last else 0.,
        mean_cpu_percent=100*(last['cpu_s']-start['cpu_s'])/span if span else 0.,
        peak_cpu_percent=peak_cpu, mean_gpu_percent=gpu_sum/gpu_span if gpu_span else None,
        peak_gpu_mib=peak_mem, peak_rss_gib=peak_rss,
        max_os_threads=max_threads, observed_cpus=sorted(masks),
        cuda_driver_allowed_cpus=sorted(driver_masks)), np.array(points)


def report_settings(config):
    """Use the pilot's recorded configuration, not this historical filename."""
    options = config.get('arguments', {})
    policy = options.get('policy_version', 'unknown-unrecorded')
    threads = int(options['threads']) if 'threads' in options else None
    affinity = config.get('affinity', [])
    if threads is not None and threads <= 0:
        raise ValueError('invalid recorded thread count')
    logical = options.get('expected_affinity', 'unrecorded')
    if logical in ('None', ''):
        logical = 'unrecorded'
    if affinity and logical != 'unrecorded':
        first, last = map(int, logical.split('-'))
        if affinity != list(range(first, last+1)):
            raise ValueError('recorded affinity and expected range disagree')
    return policy, threads, logical, affinity


def report(root, output=None):
    output = output or root
    output.mkdir(parents=True, exist_ok=True)
    status = load(root/'STATUS.json')
    config = load(root/'CONFIG.json')
    policy, threads, logical, affinity = report_settings(config)
    rows = []
    for path in sorted(root.glob('*.telemetry.jsonl')):
        label = path.name[:-len('.telemetry.jsonl')]
        if not (label.startswith('100TeV-') or label.startswith('100PeV-')):
            continue
        record = load(root/(label+'.json'))
        sampled,points = telemetry(path)
        c = record.get('accelerator',{}).get('cooperative',{})
        row = dict(label=label, complete=record.get('complete',False),
            process_s=record.get('process_s'), transport_records=record.get('transport_records'),
            telemetry=sampled)
        if c:
            row['batches'] = dict(cuda=c['subshower_cuda_commits'],openmp=c['subshower_openmp_epochs'])
            row['gpu_result_service_delay_s']=c['cuda_result_service_delay_ms']/1000
            row['host_epoch_mean_ms']=c['openmp_wall_ms']/max(1,c['subshower_openmp_epochs'])
            row['endpoints'] = c.get('adaptive',{})
        rows.append(row)
        if len(points):
            fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
            axes[0].plot(points[:,0],points[:,1],lw=.7)
            if threads is not None:
                axes[0].axhline(100*threads,color='gray',ls='--',lw=.8)
            axes[0].set_ylabel('Process CPU [%]\n100% = one logical CPU')
            axes[1].plot(points[:,0],points[:,2],color='tab:orange',lw=.8)
            axes[1].set_ylabel('Device GPU [%]');axes[1].set_ylim(0,105)
            axes[2].plot(points[:,0],points[:,3]/1024,label='GPU used (device-wide)')
            axes[2].plot(points[:,0],points[:,4],label='Process RSS')
            axes[2].set_ylabel('Memory [GiB]');axes[2].set_xlabel('Elapsed wall time [s]')
            axes[2].legend(fontsize=9)
            for ax in axes:ax.grid(alpha=.2);ax.margins(x=.01)
            fig.suptitle(label+(' (complete)' if row['complete'] else ' (running / incomplete)'))
            fig.tight_layout();fig.savefig(output/(label+'-resources.png'),dpi=150);plt.close(fig)
        endpoints=row.get('endpoints',{})
        if all(ep in endpoints for ep in ('cuda','openmp')):
            fig,ax=plt.subplots(figsize=(10,4))
            for i,ep in enumerate(('cuda','openmp')):
                hist=endpoints[ep]['job_input_histogram_floor_log2']
                ax.bar(np.arange(len(hist))+(i-.5)*.38,hist,width=.38,label=ep)
            ax.set_yscale('symlog',linthresh=1)
            ax.set_xlabel('Input batch size bin: k denotes [2^k, 2^(k+1)); last bin >=2^20')
            ax.set_ylabel('Completed backend jobs (not unique particles)')
            ax.set_title(label);ax.legend();ax.grid(axis='y',alpha=.2)
            occupied=[i for ep in ('cuda','openmp') for i,v in enumerate(
                endpoints[ep]['job_input_histogram_floor_log2']) if v]
            if occupied:ax.set_xlim(min(occupied)-.8,max(occupied)+.8)
            fig.tight_layout();fig.savefig(output/(label+'-batch-inputs.png'),dpi=150);plt.close(fig)
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),state=status,
        source=str(root),threads=threads,logical_cpus=logical,affinity=affinity,rows=rows,
        policy=policy,statistical_acceptance=False)
    tmp=output/'MONITOR_REPORT.tmp'
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(output/'MONITOR_REPORT.json')
    lines=[f'# PSR 双端 / OpenMP {threads} 线程：{policy} 监测','',
        '双端和纯 OpenMP 使用同一独立二进制，顺序运行；不与本地生产竞争。',
        f'记录的线程数：{threads}；逻辑CPU绑定：{logical}。',
        '质子，47°/180°，emthin=1e-6，max-weight 未指定（应用默认值），CoREAS/ZHS 全开启。','',
        '| 事件 | 完成 | 总时间 s | 平均 CPU % | 平均 GPU % | 峰值显存 MiB | CUDA jobs | OpenMP jobs |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t=r['telemetry'];b=r.get('batches',{})
        duration=('%.2f'%r['process_s']) if r['process_s'] is not None else '未完成'
        gpu=('%.1f'%t['mean_gpu_percent']) if t['mean_gpu_percent'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {r['complete']} | {duration} | {t['mean_cpu_percent']:.0f} | {gpu} | {t['peak_gpu_mib']:.0f} | {b.get('cuda','—')} | {b.get('openmp','—')} |")
    lines+=['','状态：`'+status.get('phase','unknown')+'`。',
        '批次/每端真实输运步数在事件结束并核对计数后统计；未完成事件不推算最终批次。',
        'CPU 百分比包含运行库忙等，GPU 是设备级采样；占用率高不等于有效吞吐或净加速。',
        '每个输入可被多个 job 推进，输入计数不是唯一粒子数。transport_records 是真实推进步数。',
        '少量同种子动态调度事件不保证相同 shower tree，也不构成大样本物理验收。','',
        '## 同种子同机时间比较','']
    for energy in ('100TeV','100PeV'):
        seeds={r['label'].split('-')[1] for r in rows if r['label'].startswith(energy+'-')}
        for seed in sorted(seeds):
            pair={r['label'].split('-')[-1]:r for r in rows
                  if r['label'].startswith(energy+'-'+seed+'-') and r['complete']}
            if 'adaptive' in pair and 'openmp' in pair:
                ratio=pair['openmp']['process_s']/pair['adaptive']['process_s']
                lines.append(f'- {energy}, seed {seed}: OpenMP/双端 = {ratio:.3f}；大于 1 才是本次试跑净加速。')
        # Only closed complete pairs enter the median; never speed-select the
        # fastest subset of a still-running campaign as a completed ensemble.
        pairs=[]
        for seed in sorted(seeds):
            pair={r['label'].split('-')[-1]:r for r in rows
                  if r['label'].startswith(energy+'-'+seed+'-') and r['complete']}
            if set(pair) >= {'openmp','adaptive'}:
                pairs.append((seed,pair['openmp']['process_s'],pair['adaptive']['process_s']))
        if pairs:
            single=np.asarray([p[1] for p in pairs]);dual=np.asarray([p[2] for p in pairs])
            ratio=float(np.median(single)/np.median(dual))
            lines += ['', f'{energy}当前{len(pairs)}对完成样本：纯OpenMP中位{np.median(single):.3f}s，'
                      f'双端中位{np.median(dual):.3f}s，比值{ratio:.3f}。',
                      '未完成campaign时以上只描述已完成配对，不作为整组加速验收。']
            fig,ax=plt.subplots(figsize=(10,4.7))
            for dx,values,name,color in ((-.19,single,'OpenMP','#0072B2'),(.19,dual,'CUDA + OpenMP','#D55E00')):
                bars=ax.bar(np.arange(len(pairs))+dx,values,.36,label=name,color=color)
                ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
            ax.set_xticks(np.arange(len(pairs)),[p[0] for p in pairs])
            ax.set_xlabel('Initial seed; dynamic scheduling permits different shower trees')
            ax.set_ylabel('End-to-end time [s]');ax.margins(y=.18)
            ax.grid(axis='y',alpha=.2);ax.legend()
            ax.set_title(f'{policy}, {energy}, {threads} threads + T400 / {threads} threads\n'
                         +('Completed pilot; not ensemble acceptance' if status.get('complete') else 'Completed pairs only; campaign incomplete'))
            fig.tight_layout();fig.savefig(output/(energy+'-paired-time.png'),dpi=180);plt.close(fig)
    (output/'MONITOR_REPORT_CN.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--watch',action='store_true')
    p.add_argument('--output',type=Path,help='Optional separate directory for analysis artifacts')
    a=p.parse_args()
    plt.rcParams.update({'font.family':'serif','font.size':10})
    while True:
        report(a.root,a.output)
        state=load(a.root/'STATUS.json')
        if not a.watch or state.get('complete') or 'failed' in state.get('phase',''):return
        time.sleep(30)


if __name__=='__main__':main()
