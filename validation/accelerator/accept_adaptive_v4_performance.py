#!/usr/bin/env python3
"""Read-only local v4 pilot acceptance. Never starts or alters a simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from accept_adaptive_service_uhe import load, reserve, scan_parquet, resource_summary


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def read_event(path):
    g=load(path/'gpu_em/summary.yaml')['shower_0']
    s=g['statistics'];c=s['accelerator']['cooperative']
    guard=json.loads((path.parent/(path.name+'-guard/summary.json')).read_text())
    timing=load(path/'simulation_timing/summary.yaml')['shower_0']
    cfg=load(path/'gpu_em/config.yaml')
    primary=load(path/'primary/summary.yaml')['shower_0']
    cmd=guard['command'];flags=dict(zip(cmd[1::2],cmd[2::2]))
    flags.pop('-f');flags.pop('--kokkos-cooperative-policy',None)
    identity=dict(primary=primary,flags=flags,environment=cfg['environment'],
        constants=cfg['magnetic_rigidity_GeV_per_T_m'],table={k:s['proposal_native'][k] for k in
        ('proposal_version','cubic_interpolation_version','table_sha256','aux_sha256')})
    assert guard['pass'] and guard['returncode']==0 and g['complete'] and timing['closed']
    assert c['subshower_cuda_submissions']==c['subshower_cuda_commits']>0
    assert s['gpu_particles']==s['profile']['steps']
    assert s['queue_overflows']==s['profile']['fixed_point_overflows']==s['radio']['fixed_point_overflows']==0
    assert s['profile']['invalid_records']==s['proposal_native']['inverse_failures']==0
    assert s['cross_species']['final_pending_photons']==s['cross_species']['final_pending_leptons']==0
    reasons=s['cpu_fallbacks_by_reason_name']
    assert set(reasons)<= {'unsupported_geometry','epair_rejection_envelope_exceeded','native_selection_replay'}
    assert sum(reasons.values())==s['cpu_generic_fallbacks']
    assert reasons.get('native_selection_replay',0)==s['cpu_completed_native_selection_replays']==s['cpu_completed_selected_losses']
    assert s['cpu_specified_final_states']==reasons.get('native_selection_replay',0)+reasons.get('epair_rejection_envelope_exceeded',0)
    ep={}
    if c.get('scheduling_policy')=='adaptive-v4-bounded-batching':
        for name in ('cuda','openmp'):
            a=c['adaptive'][name]
            ep[name]=dict(steps=sum(a[k]['transport_records'] for k in ('photon','lepton')),
                waves=sum(a[k]['resident_wavefronts'] for k in ('photon','lepton')),
                input_histogram=a['job_input_histogram_floor_log2'],
                photon=a['photon'],lepton=a['lepton'])
        assert sum(e['steps'] for e in ep.values())==s['gpu_particles']
        assert sum(e['waves'] for e in ep.values())==s['resident_photon_wavefronts']+s['resident_lepton_wavefronts']
        assert sum(ep['cuda']['input_histogram'])==c['subshower_cuda_commits']
        assert sum(ep['openmp']['input_histogram'])==c['subshower_openmp_epochs']
    resources,raw=resource_summary(path.parent/(path.name+'-guard/resources.jsonl'))
    values=dict(process_s=guard['elapsed_s'],shower_s=timing['wall_time_ms']/1000,
        transport_records=s['gpu_particles'],final_state_children=s['physical_secondaries'],
        radio_segments=s['radio_tracks'],cuda_jobs=c['subshower_cuda_commits'],openmp_jobs=c['subshower_openmp_epochs'],
        cuda_mean_job_inputs=c['cuda_input_particles']/c['subshower_cuda_commits'],
        openmp_mean_job_inputs=c['openmp_input_particles']/c['subshower_openmp_epochs'],
        service_delay_s=c['cuda_result_service_delay_ms']/1000,
        cuda_call_s=c['cuda_driver_wall_ms']/1000,openmp_call_s=c['openmp_wall_ms']/1000,
        scalar_stepper_s=s['hybrid_timing_ms']['scalar_stepper']/1000,
        set_nodes_s=s['hybrid_timing_ms']['set_nodes']/1000,
        router_advance_s=s['hybrid_timing_ms']['router_advance']/1000,
        resident_wavefronts=s['resident_photon_wavefronts']+s['resident_lepton_wavefronts'],
        migration_bytes=c['migration_bytes'],peak_rss_GiB=guard['peak_tree_rss_bytes']/2**30,
        minimum_available_GiB=guard['minimum_available_bytes']/2**30,**resources)
    return dict(path=str(path),complete=True,identity=identity,values=values,
        endpoints=ep,energy_ledger=s['energy_ledger'],fallbacks=reasons,thinning=s['thinning']),raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('/mnt/d/CorsikaData/corsika_validation_results'))
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pa.set_cpu_count(1);pa.set_io_thread_count(1);reserve()
    run=a.base/'beta5_adaptive_v4_batching_20260911/run'
    state=json.loads((run/'STATUS.json').read_text());assert state['complete'] and len(state['records'])==5
    provenance=json.loads((run/'PROVENANCE.json').read_text())
    assert hashlib.sha256((run/'binaries/c8_air_shower').read_bytes()).hexdigest()==provenance['binary_sha256']
    paths=dict(independent=a.base/'beta5_subshower_queues_20260911/pilot-100pev/run/independent20',
        v3=a.base/'beta5_adaptive_v3_service_20260911/run/proton100PeV-2026110001-adaptive',
        v4=run/'proton100PeV-2026110001-adaptive')
    records={};telemetry={}
    for name,path in paths.items():records[name],telemetry[name]=read_event(path)
    assert records['independent']['identity']==records['v3']['identity']==records['v4']['identity']
    assert records['v4']['identity']['primary']['total_energy']==1e8
    fresh={};fe={};sys.path.insert(0,'/home/yuhanglu/21CMA/analysis_jobs')
    import reproject_waveform_ensembles as w
    for entry in state['records']:
        path=run/f"{entry['family']}-{entry['seed']}-{entry['mode']}"
        record,_=read_event(path)
        event=load(path/'summary.yaml')
        assert event['showers']==1 and event['seed']==entry['seed'] and event.get('end time')
        scans={str(f.relative_to(path)):scan_parquet(f) for f in sorted(path.rglob('*.parquet'))}
        assert len(scans)==7
        for f in (path/'interaction_hist').glob('*.npz'):
            with np.load(f,allow_pickle=False) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
        profile=pq.read_table(path/'profile/profile.parquet').to_pandas()
        assert np.all(np.diff(profile['X'].to_numpy())>0)
        assert all(np.all(profile[k].to_numpy()>=0) for k in profile if k not in ('X','shower'))
        radio={}
        for alg in ('CoREAS','ZHS'):
            cfg=load(path/alg/'config.yaml')
            wave=list(w.events(path/alg/'observers.parquet',cfg,1,alg))
            assert len(wave)==1 and np.isfinite(wave[0]).all()
            power=(wave[0]**2).sum(axis=2);assert np.all(power.sum(axis=1)>0)
            edge=power[:,:5].sum()+power[:,-5:].sum()
            radio[alg]=dict(observers=power.shape[0],time_bins=power.shape[1],
                edge5_power_fraction=float(edge/power.sum()),
                peaks_at_edge_count=int(((power.argmax(axis=1)<5)|(power.argmax(axis=1)>=power.shape[1]-5)).sum()),
                outside_recorded_time_window_unknown=True)
        fresh[path.name]=dict(complete=True,parquet=scans,radio=radio)
        if entry['family']=='Fe100TeV':fe.setdefault(str(entry['seed']),{})[entry['mode']]=record
    for pair in fe.values():assert pair['legacy']['identity']==pair['adaptive']['identity']
    fe_medians={k:statistics.median(pair[k]['values']['process_s'] for pair in fe.values()) for k in ('legacy','adaptive')}
    cuda_record=a.base/'beta5_proton100PeV_cpu100_cuda100_openmp100_paired_20260908_v1/cuda/event_0000.json'
    cuda=json.loads(cuda_record.read_text());assert cuda['complete'] and cuda['seed']==2026110001
    attempt=cuda['attempts'][-1];assert attempt['returncode']==0
    cp=Path(attempt['output']);cs=load(cp/'gpu_em/summary.yaml')['shower_0']['statistics'];cc=load(cp/'gpu_em/config.yaml')
    assert load(cp/'primary/summary.yaml')['shower_0']==records['v4']['identity']['primary']
    assert cc['environment']==records['v4']['identity']['environment']
    assert all(cs['proposal_native'][k]==v for k,v in records['v4']['identity']['table'].items())
    assert all(cs['thinning'][k]==records['v4']['thinning'][k] for k in ('em_fraction','maximum_weight','automatic_maximum_weight'))
    historical_cuda=dict(process_s=attempt['simulation_process_wall_s'],transport_records=cs['gpu_particles'],
        source=str(cuda_record),binary_sha256=cuda['executable_sha256'],
        magnetic_constant_in_metadata=cc.get('magnetic_rigidity_GeV_per_T_m'),
        strict_same_binary_comparator=False)
    v4=records['v4']['values'];ratios={}
    for name in ('independent','v3'):
        old=records[name]['values']
        ratios[name]={k:v4[k]/old[k] for k in old if old[k]!=0}
        ratios[name]['time_per_step_ratio']=ratios[name]['process_s']/ratios[name]['transport_records']
    result=dict(operational_acceptance=True,all_five_new_events_valid=True,
        high_energy_performance_restored=False,statistical_acceptance=False,
        full_shower_energy_closure_certified=False,provenance=provenance,
        high_energy=records,new_over_reference=ratios,historical_cuda=historical_cuda,
        Fe=fe,Fe_median_s=fe_medians,fresh_output_checks=fresh,
        caveats=['Single high-energy seed per strategy; dynamic schedules allow different shower trees',
            'Total transport records are actual steps, not unique particles; job inputs count repeated states',
            'CPU utilization includes spinning; GPU utilization is device-wide; call windows can overlap',
            'Historical single CUDA metadata lacks the magnetic conversion constant; it is a timing reference, not a strict same-build comparison',
            'Per-step normalization does not correct for different process/energy mixtures'])
    save(a.output/'ACCEPTANCE.json',result)
    with (a.output/'workload.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['metric','independent','v3','v4'])
        for k in v4:writer.writerow([k]+[records[n]['values'][k] for n in ('independent','v3','v4')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':180})
    names=['Historical single CUDA','Independent dual','Adaptive v3','Adaptive v4']
    times=[historical_cuda['process_s']]+[records[n]['values']['process_s'] for n in ('independent','v3','v4')]
    fig,ax=plt.subplots(figsize=(9,4.5));bars=ax.bar(names,np.array(times)/60,color=['gray','#0072B2','#CC79A7','#D55E00'])
    for bar,t in zip(bars,times):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{t/60:.2f} min',ha='center')
    ax.set_ylabel('End-to-end wall time [min]');ax.set_ylim(0,85);ax.grid(axis='y',alpha=.2)
    ax.set_title('100 PeV proton, 47°/180°, seed 2026110001\nOne event per strategy; historical binaries, not a controlled ensemble benchmark')
    fig.tight_layout();fig.savefig(a.output/'time_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for name,color in zip(('independent','v3','v4'),('#0072B2','#CC79A7','#D55E00')):
        raw=telemetry[name];t=np.array([r['elapsed_s'] for r in raw]);cs=np.array([r['cpu_seconds'] for r in raw])
        tx=[];cpu=[];gpu=[]
        for start in np.arange(0,t[-1],15):
            lo=np.searchsorted(t,start);hi=min(len(t)-1,np.searchsorted(t,start+15))
            if hi<=lo:continue
            tx.append((t[lo]+t[hi])/120);cpu.append((cs[hi]-cs[lo])/(t[hi]-t[lo]))
            gpu.append(np.average([r.get('device_util_percent',0) for r in raw[lo+1:hi+1]],weights=np.diff(t[lo:hi+1])))
        axes[0].plot(tx,cpu,label=name,color=color);axes[1].plot(tx,gpu,color=color)
        axes[2].plot(t/60,[r['rss_bytes']/2**30 for r in raw],color=color)
    axes[0].set_ylabel('Logical CPU equivalent');axes[0].legend();axes[1].set_ylabel('Device GPU [%]')
    axes[2].set_ylabel('Process RSS [GiB]');axes[2].set_xlabel('Elapsed wall time [min]')
    for ax in axes:ax.grid(alpha=.2);ax.margins(x=.01)
    fig.tight_layout();fig.savefig(a.output/'resources.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    for i,n in enumerate(('cuda','openmp')):
        hist=records['v4']['endpoints'][n]['input_histogram']
        ax.bar(np.arange(len(hist))+(i-.5)*.4,hist,width=.4,label=n)
    ax.set_yscale('symlog',linthresh=1);ax.set_xlim(-.7,18.7)
    ax.set_xlabel('Input count bin k: [2^k, 2^(k+1)); last bin >= 2^20')
    ax.set_ylabel('Completed jobs');ax.set_title('Adaptive v4 input-batch distribution (not unique particles)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(a.output/'batch_input_distribution.png');plt.close(fig)
    lines=['# 本地 adaptive-v4：完成样本的加速验收','',
        '**运行完整性通过；相比 v3 改善，但没有恢复原独立双端的高能性能。不属于大样本统计验收。**','',
        '质子100 PeV，47°/180°，seed2026110001，emthin=1e-6，默认自动max-weight=50；双端20线程，GPU预算70%，完整CoREAS/ZHS。','',
        '| 模式 | 总时间 s | 分钟 |','|---|---:|---:|']
    for name,t in zip(names,times):lines.append(f'| {name} | {t:.3f} | {t/60:.2f} |')
    lines += ['',f"v4 比 v3 时间缩短 **{100*(1-ratios['v3']['process_s']):.2f}%**；比原独立双端慢 **{100*(ratios['independent']['process_s']-1):.2f}%**。",
        f"相对历史单CUDA的描述性比值为 **{historical_cuda['process_s']/v4['process_s']:.3f}×**（耗时缩短 {100*(1-v4['process_s']/historical_cuda['process_s']):.2f}%）。",
        '历史单CUDA的主粒子、表/辅助哈希、磁场、thinning和max-weight已匹配；但其metadata没有磁偏转换算常数字段，且二进制不同，不能当作严格同版本受控基准。','',
        '## 实际工作量与批次','',
        '| 指标 | 原独立双端 | v3 | v4 |','|---|---:|---:|---:|']
    for title,k in [('实际两端输运步数','transport_records'),('薄化后末态子状态','final_state_children'),('射电轨迹段','radio_segments'),
        ('CUDA jobs','cuda_jobs'),('OpenMP jobs','openmp_jobs'),('CUDA平均输入状态/job','cuda_mean_job_inputs'),
        ('OpenMP平均输入状态/job','openmp_mean_job_inputs'),('GPU完成到领取累计等待/s','service_delay_s')]:
        lines.append('| '+title+' | '+' | '.join(f"{records[n]['values'][k]:,.2f}" for n in ('independent','v3','v4'))+' |')
    lines += ['',f"v4 的真实输运步数比 v3 多 {100*(ratios['v3']['transport_records']-1):.2f}%，时间反而减少，因此不是靠更小的 shower 得到这次改善。",
        f"v4 的 OpenMP jobs 比 v3 减少 {100*(1-ratios['v3']['openmp_jobs']):.2f}%，CUDA jobs 减少 {100*(1-ratios['v3']['cuda_jobs']):.2f}%。",
        f"但相对原独立双端，CUDA/OpenMP jobs 仍为 {ratios['independent']['cuda_jobs']:.2f}/{ratios['independent']['openmp_jobs']:.2f} 倍；",
        f"按总推进步数粗略归一化，v4 相对原独立双端的每步墙钟成本仍高 {100*(ratios['independent']['time_per_step_ratio']-1):.2f}%。过程/能量组合不同，不能当成严格因果分解。",
        'GPU服务等待与CPU计算存在重叠，不能直接从总时间扣除；实际kernel/同步开销没有设备事件级时间线，不能精确分摊。','',
        '## 两端分工与内存','']
    for name,e in records['v4']['endpoints'].items():
        lines.append(f"- {name}: {e['steps']:,} 个真实输运步，占 {100*e['steps']/v4['transport_records']:.2f}%；{e['waves']:,} 个驻留物理波前。")
    lines += [f"- 平均GPU {v4['gpu_utilization_percent']:.2f}%，平均CPU {v4['cpu_cores']:.2f} 个逻辑核当量；峰值显存 {v4['peak_gpu_MiB']:.0f} MiB。",
        f"- 进程树RSS峰值 {v4['peak_rss_GiB']:.2f} GiB，系统可用内存最低 {v4['minimum_available_GiB']:.2f} GiB；未触发保护。单个N=1事件不能证明所有N>1场景无泄漏。",'',
        '## Fe 100 TeV 两例','', '| seed | legacy20/s | v4/s |','|---|---:|---:|']
    for seed,pair in fe.items():lines.append(f"| {seed} | {pair['legacy']['values']['process_s']:.3f} | {pair['adaptive']['values']['process_s']:.3f} |")
    lines += [f"同二进制交替测试的中位数 {fe_medians['legacy']:.3f} → {fe_medians['adaptive']:.3f} s，耗时缩短 {100*(1-fe_medians['adaptive']/fe_medians['legacy']):.2f}%。仅两个种子，不能替代500例验收。",'',
        '## 完整性边界','',
        '- 5个新事件全部正常结束；7类Parquet逐批扫描无NaN/Inf/空值，事件号正确；profile非负，interaction histogram有效。',
        '- CoREAS/ZHS实际数组有效，时间边缘诊断见ACCEPTANCE.json；窗口以外未记录的信号不能由文件内检查证明不存在。',
        '- 两端pending清零、CUDA提交全部回收，queue/profile/radio溢出与native反解失败为0；v4逐端步数、波前和批次直方图对账通过。',
        '- 指定native selected-loss、Epair及几何fallback逐项对账；不是零回退。',
        f"- 100PeV局部能量账本仍为 complete_coverage=false、accepted=false，残差 {records['v4']['energy_ledger']['relative_closure_error']*100:.3f}%。不据此宣布完整能量守恒通过，也不直接断言这些能量丢失。",
        '- 本轮只做读取与分析，没有改调度、物理参数、生产二进制或启动新任务。','',
        '![时间比较](time_comparison.png)','', '![CPU/GPU及内存](resources.png)','', '![批量分布](batch_input_distribution.png)']
    (a.output/'ACCEPTANCE_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(a.output),operational=True,high_energy_restored=False,
        v4_process_s=v4['process_s'],v4_vs_v3=ratios['v3']['process_s'],v4_vs_independent=ratios['independent']['process_s'],Fe_median_s=fe_medians),indent=2))


if __name__=='__main__':main()
