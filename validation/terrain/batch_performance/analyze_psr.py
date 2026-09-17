#!/usr/bin/env python3
"""Validate complete event ledgers/profiles and report measured scheduling costs."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import numpy as np
import pandas as pd

def digest(path,compressed=False):
    h=hashlib.sha256();lines=0
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block);lines+=block.count(b'\n')
    return h.hexdigest(),max(0,lines-1)

def audit(folder):
    status=json.loads((folder/'status.json').read_text());assert status['complete'],folder
    s=status['summary'];d=s['diagnostics'];a=s['accelerator'];ledger=s['energy_ledger']
    assert a['execution_space']=='OpenMP' and a['execution_concurrency']==256
    assert s['emthin']==1e-6 and s['emcut_GeV']==.0005 and s['hadcut_GeV']==s['mucut_GeV']==.3
    # yaml-cpp emits e.g. 1e-08, which PyYAML's YAML 1.1 loader treats as text.
    assert abs(float(s['max_weight'])-.5e-6*status['energy_GeV'])<1e-14
    assert d['material_mismatches']==0
    assert a['downloaded_records']==a['particles_advanced'],'resident records lost or repeated'
    assert a['pending_particles']==0 and not d['csv_truncated'] and not d['deposition_csv_truncated']
    assert abs(ledger['unexplained_over_initial'])<1e-9
    hashes={};counts={}
    for name in ['tracks.csv.gz','deposits.csv.gz','domain_exits.csv.gz','window_survivors.csv.gz']:
        hashes[name],counts[name]=digest(folder/'output/terrain'/name,True)
    assert counts['tracks.csv.gz']==d['steps']
    assert counts['deposits.csv.gz']==d['deposition_rows']
    assert counts['domain_exits.csv.gz']==d['domain_exits']
    assert counts['window_survivors.csv.gz']==d['finite_window_survivors']
    if d['domain_exits']:
        tracks=pd.read_csv(folder/'output/terrain/tracks.csv.gz')
        exits=pd.read_csv(folder/'output/terrain/domain_exits.csv.gz')
        assert exits.history_id.is_unique,'duplicate terminal history'
        last=tracks.drop_duplicates('history_id',keep='last').set_index('history_id')
        for row in exits.itertuples():
            end=last.loc[row.history_id]
            assert end.pdg==row.pdg and end.weight==row.weight and end.parent_history_id==row.parent_history_id
            for axis in ['x','y','z']:assert abs(end[axis+'1_m']-getattr(row,axis+'_m'))<1e-10
            assert abs(end.E1_GeV-row.total_GeV)<1e-10
        assert abs(float((exits.weight*exits.total_GeV).sum())-d['domain_escaped_total_GeV'])<1e-10*max(1.,d['domain_escaped_total_GeV'])
    deposition=0.
    for chunk in pd.read_csv(folder/'output/terrain/deposits.csv.gz',usecols=['weighted_deposited_GeV'],chunksize=131072):
        values=chunk['weighted_deposited_GeV'].to_numpy();assert np.isfinite(values).all()
        deposition+=float(values.sum())
    assert abs(deposition-d['deposited_energy_GeV'])<1e-10*max(1.,abs(deposition))
    kernels=[];profile=folder/'kokkos_timing.csv'
    if profile.exists():kernels=list(csv.DictReader(profile.open()))
    radio_kernel=sum(float(k['host_call_seconds']) for k in kernels if k['kernel'].startswith('for:interface_radio_') and 'build_transmission' not in k['kernel'])
    transport_kernel=sum(float(k['host_call_seconds']) for k in kernels if k['kernel']=='for:interface_material_em_step')
    queue_kernel=sum(float(k['host_call_seconds']) for k in kernels if 'interface_resident_' in k['kernel'])
    radio=s.get('radio_result',{});paths=0;launches=0
    if radio:
        assert radio['complete'] and radio['errors']==0 and radio['out_of_window']==0
        assert len(s['scene']['radio']['observers'])==80
        x,y=radio['CoREAS'],radio['ZHS'];paths=x['paths'];launches=radio['radio_kernel_launches']
        for key in ['device_tracks','cpu_tracks','track_observer_pairs','paths','leaves']:
            assert x[key]==y[key],(folder,key)
        assert x['track_observer_pairs']==80*(x['device_tracks']+x['cpu_tracks'])
    return dict(tag=folder.name,complete=True,energy_GeV=status['energy_GeV'],profiled=status['profile'],wall_s=status['wall_s'],
        shower_s=s['shower_seconds'],steps=d['steps'],deposited_GeV=deposition,
        unexplained_over_initial=ledger['unexplained_over_initial'],csv_sha256=hashes,
        accelerator=a,execution_timing=s.get('execution_timing'),radio_launches=launches,radio_paths=paths,
        radio_kernel_s=radio_kernel,transport_kernel_s=transport_kernel,queue_kernel_s=queue_kernel,
        max_weight=float(s['max_weight']),output_timing=d.get('device_output_timing'),geometry_audit=d.get('device_geometry_audit'))

def compare_fields(left,right):
    result={}
    for algorithm in ['CoREAS','ZHS']:
        a=left/'output/radio'/algorithm/'field.csv';b=right/'output/radio'/algorithm/'field.csv'
        if not a.exists() and not b.exists():continue
        delta=norm=0.;peak=0.;rows=0
        with pd.read_csv(a,chunksize=65536) as xa,pd.read_csv(b,chunksize=65536) as xb:
            from itertools import zip_longest
            for x,y in zip_longest(xa,xb):
                assert x is not None and y is not None
                assert np.array_equal(x[['observer','time_s']].to_numpy(),y[['observer','time_s']].to_numpy())
                u=x.iloc[:,2:].to_numpy();v=y.iloc[:,2:].to_numpy()
                assert np.isfinite(u).all() and np.isfinite(v).all()
                delta+=float(np.sum((u-v)**2));norm+=float(np.sum(v*v));peak=max(peak,float(np.max(np.abs(v))));rows+=len(x)
        relative=float(np.sqrt(delta/norm)) if norm else 0.
        assert (norm and relative<1e-11) or (not norm and delta==0.)
        result[algorithm]=dict(relative_l2=relative,reference_peak_V_m=peak,rows=rows)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert socket.gethostname()=='psrpku2025';r=a.root
    progress=json.loads((r/'phase2_progress.json').read_text());assert progress['state']=='runs_complete'
    assert json.loads((r/'boundary_progress.json').read_text())['state']=='runs_complete'
    results={f.name:audit(f) for f in sorted((r/'runs').iterdir()) if (f/'status.json').exists()}
    import yaml
    stations=yaml.safe_load((r/'scenes/production_reference.yaml').read_text())['radio']['observers']
    assert len(stations)==80
    for tag in results:
        s=json.loads((r/'runs'/tag/'status.json').read_text())['summary']
        assert s['scene']['radio']['observers']==stations,'station positions changed'
    pairs=[('before_electron_1GeV_profile','candidate_legacy_electron_1GeV'),
           ('before_electron_1GeV_profile','candidate_electron_1GeV'),
           ('before_electron_10GeV','candidate_electron_10GeV'),
           ('before_electron_1TeV_transport','candidate_electron_1TeV_transport'),
           ('before_electron_1TeV_transport','final_electron_1TeV_transport'),
           ('before_electron_1TeV_transport','final_repeat_electron_1TeV_transport'),
           ('before_electron_10GeV','final_electron_10GeV'),
           ('before_boundary_1GeV','final_boundary_1GeV'),
           ('before_surface_20MeV','final_surface_20MeV')]
    comparisons=[]
    for before,after in pairs:
        x,y=results[before],results[after]
        assert x['csv_sha256']==y['csv_sha256'],(before,after,'per-step CSV changed at same wavefront capacity')
        fields=compare_fields(r/'runs'/before,r/'runs'/after)
        comparisons.append(dict(before=before,after=after,exact_csv=True,fields=fields,
            shower_speedup=x['shower_s']/y['shower_s'],wall_speedup=x['wall_s']/y['wall_s'],
            radio_kernel_speedup=x['radio_kernel_s']/y['radio_kernel_s'] if y['radio_kernel_s'] else None))
    checks=json.loads((r/'report/checks.json').read_text());assert checks['passed'] and checks['complete']
    audit_checks={name:(r/'checks/run_output_audit_v4'/name).read_text() for name in ['run.log','dem.log']}
    assert audit_checks['run.log'].count('PASS ')==5 and 'PASS CPU mesh/tree oracle' in audit_checks['dem.log']
    for tag in ['before_surface_20MeV','final_surface_20MeV']:
        assert results[tag]['radio_paths']>0,'real surface control must produce paths'
    assert results['final_boundary_1GeV']['accelerator']['domain_exits']>0
    assert comparisons[-1]['fields']['CoREAS']['reference_peak_V_m']>0
    assert comparisons[-1]['fields']['ZHS']['reference_peak_V_m']>0
    parent=r.parent/'beta5_doublebang_21CMA80_dem_openmp256_20260914/source'
    inherited=r.parent/'beta5_material_models_20260913/source'
    unchanged={}
    for name in ['corsika/modules/radio/interface/Propagation.hpp','corsika/modules/radio/interface/ZhsIntervals.hpp',
                 'corsika/modules/radio/interface/CoreasEndpoints.hpp','corsika/modules/transport/detail/InterfaceEmStep.hpp',
                 'corsika/modules/terrain/TerrainMagneticTracking.hpp']:
        before_file=parent/name if (parent/name).exists() else inherited/name
        after_file=r/'source'/name if (r/'source'/name).exists() else inherited/name
        old=digest(before_file)[0];new=digest(after_file)[0];assert old==new
        unchanged[name]=dict(sha256=new,before_source=str(before_file),after_source=str(after_file))
    proof=dict(passed=True,scope='Complete low-energy events and exact replay; not a high-energy production timing claim.',
        events=results,comparisons=comparisons,checks=checks,output_audit_checks=audit_checks,unchanged_physical_sources=unchanged)
    # Figures are generated only on PSR; English labels throughout.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[('before_electron_1GeV_profile','candidate_electron_1GeV','1 GeV, buffering only'),
              ('before_electron_10GeV','final_electron_10GeV','10 GeV, radio'),
              ('before_electron_1TeV_transport','final_electron_1TeV_transport','1 TeV, transport only')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    for ax,key,title in zip(axes,['shower_s','radio_launches','record_downloads'],['Shower phase (s)','Radio kernel launches','Host record checkpoints']):
        xx=np.arange(len(selected));before=[];after=[]
        for b,c,label in selected:
            before.append(results[b]['accelerator'][key] if key=='record_downloads' else results[b][key])
            after.append(results[c]['accelerator'][key] if key=='record_downloads' else results[c][key])
        ax.bar(xx-.19,before,.38,label='Before');ax.bar(xx+.19,after,.38,label='After')
        ax.set_xticks(xx,[v[2].replace(', ','\n') for v in selected]);ax.set_title(title);ax.grid(axis='y',alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('SiO2 | OpenMP 256 cores | EM thinning 1e-6 | same-wavefront comparisons')
    fig.savefig(r/'report/batching_comparison.png',dpi=170);plt.close(fig)
    # Complete deposition profile: raw rows already required to match byte for byte.
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    profiles={}
    for tag in ['before_electron_1TeV_transport','final_electron_1TeV_transport']:
        depths=[];weights=[]
        for chunk in pd.read_csv(r/'runs'/tag/'output/terrain/deposits.csv.gz',usecols=['y0_m','y1_m','weighted_deposited_GeV'],chunksize=131072):
            depths.append(3921.7316894076384-.5*(chunk.y0_m.to_numpy()+chunk.y1_m.to_numpy()))
            weights.append(chunk.weighted_deposited_GeV.to_numpy())
        profiles[tag]=(np.concatenate(depths),np.concatenate(weights))
    low=min(x[0].min() for x in profiles.values());high=max(x[0].max() for x in profiles.values())
    profile_bins=np.linspace(low-1e-8,high+1e-8,101)
    for tag,label,style in [('before_electron_1TeV_transport','Before','-'),('final_electron_1TeV_transport','After','--')]:
        depth,weight=profiles[tag];hist=np.histogram(depth,bins=profile_bins,weights=weight)[0]
        assert abs(hist.sum()-weight.sum())<1e-10*max(1.,weight.sum()),'profile figure omitted deposition'
        axes[0].stairs(hist,profile_bins,label=label,linestyle=style)
    axes[0].set(xlabel='Depth along initial axis (m)',ylabel='Deposited energy / bin (GeV)',title='1 TeV SiO2: complete recorded profile');axes[0].legend()
    timing_tags=['output_profile_electron_10GeV','audit8_electron_10GeV','audit32_electron_10GeV','audit256_electron_10GeV','flat_audit_electron_10GeV']
    measured=[results[t]['execution_timing']['transport_loop_seconds'] for t in timing_tags]
    axes[1].bar(np.arange(5),measured,color=['#777777','#c49c94','#c49c94','#c49c94','#55a868'])
    axes[1].set_xticks(np.arange(5),['Original\nserial','Objects\n8 threads','Objects\n32 threads','Objects\n256 threads','Final\n32 threads'])
    axes[1].set(ylabel='Transport loop (s)',title='10 GeV: host audit implementation')
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(r/'report/profile_and_audit.png',dpi=170);plt.close(fig)
    # A real, nonzero DEM/80-station control complements the deep-rock zero-field tests.
    fields={};peak_station=0;peak=0.
    for algorithm in ['CoREAS','ZHS']:
        path=r/'runs/final_surface_20MeV/output/radio'/algorithm/'field.csv'
        frame=pd.read_csv(path,usecols=['observer','time_s','Ex_V_m','Ey_V_m','Ez_V_m'])
        fields[algorithm]=frame
        amplitude=np.linalg.norm(frame[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy(),axis=1)
        index=int(np.argmax(amplitude))
        if amplitude[index]>peak:peak=float(amplitude[index]);peak_station=int(frame.observer.iloc[index])
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for algorithm,style in [('CoREAS','-'),('ZHS','--')]:
        frame=fields[algorithm];frame=frame[frame.observer==peak_station]
        amplitude=np.linalg.norm(frame[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy(),axis=1)
        ax.plot(frame.time_s.to_numpy()*1e9,amplitude,label=algorithm,linestyle=style)
    frame=fields['CoREAS'];frame=frame[frame.observer==peak_station]
    amplitude=np.linalg.norm(frame[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy(),axis=1)
    center=float(frame.time_s.iloc[int(np.argmax(amplitude))])*1e9
    station=json.loads((r/'runs/final_surface_20MeV/status.json').read_text())['summary']['scene']['radio']['observers'][peak_station]['name']
    ax.set(xlim=(center-100,center+100),xlabel='Observer time (ns)',ylabel='Electric-field magnitude (V/m)',title=f'20 MeV surface control | {station} | both algorithms | full 80-station run')
    ax.legend();ax.grid(alpha=.25);fig.savefig(r/'report/surface_radio_control.png',dpi=170);plt.close(fig)
    b=results['before_electron_1TeV_transport'];c=results['final_electron_1TeV_transport'];repeat=results['final_repeat_electron_1TeV_transport'];wide=results['final_wide_electron_1TeV_transport']
    br=results['before_electron_10GeV'];cr=results['final_electron_10GeV']
    lines=['# 山体常驻队列与主机输出优化验收','',
      'PSR 的未完成高能山体任务已停止；只删除该事件约 2.73 GiB 的未完成 CSV，保留输入、日志及已有诊断。本地空气 CUDA 任务保留。此次编译、测试、分析和绘图全部在 PSR。','',
      '本次使用 SiO₂、真实 DEM、OpenMP 256 个物理核。电子从原 double bang 第一顶点注入，用于隔离 EM 调度开销；这些低能控制事件本身不是 double bang。射电对照始终计算全部 80 站和 CoREAS/ZHS 两种算法。','',
      '## 看运行时间与返回次数','',
      '![Timing and checkpoints](batching_comparison.png)','',
      '左图越低越快，中图是射电内核启动次数，右图是批量记录返回主机的次数。减少返回次数本身不保证提速：仅合并缓冲时，10 GeV 射电 shower 从 93.52 s 变为 92.43 s，改善很小。真正的瓶颈是主机逐条几何复核中的重复对象构造。','',
      '|完整事件|优化前 shower / 总耗时|优化后 shower / 总耗时|shower 加速|',
      '|---|---:|---:|---:|',
      f'|1 TeV，仅输运|{b["shower_s"]:.2f} / {b["wall_s"]:.2f} s|{c["shower_s"]:.2f} / {c["wall_s"]:.2f} s|{b["shower_s"]/c["shower_s"]:.2f}×|',
      f'|10 GeV，带射电|{br["shower_s"]:.2f} / {br["wall_s"]:.2f} s|{cr["shower_s"]:.2f} / {cr["wall_s"]:.2f} s|{br["shower_s"]/cr["shower_s"]:.2f}×|','',
      '带射电的单次前后差别约 1%，不能视为显著提速；该事件主要受光路计算和射电网格输出支配。7 倍加速仅指上述 1 TeV 纯输运 shower 阶段。','',
      f'1 TeV 优化版重复一次：shower {repeat["shower_s"]:.2f} s。将常驻波前上限提高到 65536 的完整事件：{wide["shower_s"]:.2f} s、{wide["steps"]:,} 步，队列清空且账本通过。宽波前可能改变 CPU 回退和随机历史调度，不能用单个事件的时间决定所有高能事件的最佳上限。','',
      '## 看 profile 是否完整','',
      '![Profile and audit](profile_and_audit.png)','',
      f'左图是 1 TeV 事件的沉积纵向分布；两条线应重合。更严格的验收是全部 {c["steps"]:,} 条 track 以及全部 deposit、边界逃逸和末端记录，解压后的 CSV 与原版逐字节一致。右图说明直接增加旧几何对象查询的线程数会更慢；最终版保存既有 BVH 的只读数值，执行同样的三射线判据和 10 nm 边界复核，再按原顺序写记录和累计能量。','',
      '完整能量账本、CSV 行数与沉积求和通过；队列没有剩余粒子，没有 profile 截断。这里的能量验收会单独核算截止静质量、生成器交换项及 thinning 项，不把这些物理账项误报成数值损失。','',
      '## 看非零射电对照','',
      '![Nonzero surface radio](surface_radio_control.png)','',
      f'20 MeV 电子在真实 DEM 表面下约 1 cm 注入，图中显示全阵列峰值所在站 {station} 的场强模。它用于确认新旧处理在非零信号上也一致；两个算法各自的新旧波形相对 L2 差必须小于 1e-11。深部 1–10 GeV 控制事件没有到达阵列的有效透射路径，其零场不能单独验证射电正确性。','',
      '## 实现与适用范围','',
      '- 独立山体模块新增常驻波前上限（最高 65536）、记录缓冲（最高 1048576）与 OpenMP 射电源缓冲；容量纳入内存准入，尾批在结束时排空。',
      '- 默认记录缓冲为 16×波前容量，射电源缓冲为 8192；保留必须由 CPU 处理的物理回退。输出几何复核最多用 32 线程，输运和射电仍配置 256 线程。',
      '- 原 CPU 几何与快速复核在立方体约 1 万点、真实 DEM 约 2.8 万点上对照，覆盖面、顶点、纳米偏移及坐标变换；错误注入检查首个报错位置和已有输出完全一致。',
      '- 常驻队列约 297 万条记录的独立重放、宽波前，以及射电 CPU/设备混合来源、尾批和重复结束保护均通过。',
      '- 另用真实 DEM 边缘的电子逃逸事件复查：末段 track、终止位置和逃逸能量均完整记录，前后 CSV 相同。每个波前仍读取少量队列控制计数；优化减少的是完整记录返回和射电源提交次数，并不表示主机完全不参与控制。',
      '- 原空气模块未修改；山体 EM、磁场、折射、Fresnel、衰减及两种射电算法的物理内核未修改。','',
      '统一数值参数：EM 动能截止 0.5 MeV，强子/μ/τ 截止 0.3 GeV，emthin=1e-6，Wmax=0.5×emthin×E/GeV，LPM 等物理保持原设置。低能测试的 Wmax<1，单位权重粒子实际上不触发 thinning；没有人为放大权重。射电低能验收窗为 128 μs、256 MHz，前后完全相同；原高能生产窗为 2048 μs。因此这里不承诺高能 double bang 同比例加速，也没有重启高能生产。','',
      '完整机器验收：[acceptance.json](acceptance.json)。所有命令和隔离二进制保存在 PSR 的 beta5_interface_batch_performance_20260915 目录。']
    (r/'report/README_CN.md').write_text('\n'.join(lines)+'\n')
    proof['profile_figure_includes_all_deposition']=True
    proof['nonzero_radio_control_station']=station
    (r/'report/acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(passed=True,events=len(results),comparisons=comparisons),indent=2))

if __name__=='__main__':main()
