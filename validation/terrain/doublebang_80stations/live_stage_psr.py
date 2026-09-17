#!/usr/bin/env python3
"""Read fixed byte prefixes of live gzip files, without changing the writer."""
import argparse
from collections import Counter,deque
import datetime
import io
import json
from pathlib import Path
import socket
import time
import zlib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def frames(path,limit,dtypes=None):
    decoder=zlib.decompressobj(31);pending=b'';header=None;offset=0
    with path.open('rb') as stream:
        while offset<limit:
            compressed=stream.read(min(1048576,limit-offset))
            if not compressed:break
            offset+=len(compressed);pending+=decoder.decompress(compressed)
            index=pending.rfind(b'\n')
            if index<0:continue
            complete=pending[:index+1];pending=pending[index+1:]
            if header is None:
                end=complete.find(b'\n');header=complete[:end+1];complete=complete[end+1:]
            if complete:
                yield pd.read_csv(io.BytesIO(header+complete),dtype=dtypes),offset
    # No gzip footer is expected in a running output. The final incomplete CSV
    # row is discarded. This is a flushed-prefix diagnostic, never a full audit.


def stats(frame,vertex,direction):
    p=frame[['x1_m','y1_m','z1_m']].to_numpy();s=(p-vertex)@direction
    return dict(rows=len(frame),pdg_counts={str(k):int(v) for k,v in frame.pdg.value_counts().items()},
        medium_counts={str(k):int(v) for k,v in frame.medium.value_counts().items()},
        longitudinal_distance_from_first_vertex_m_quantiles=np.quantile(s,[0,.05,.5,.95,1]).tolist(),
        total_energy_GeV_quantiles=np.quantile(frame.E0_GeV.to_numpy(),[0,.05,.5,.95,1]).tolist(),
        physical_time_s_quantiles=np.quantile(frame.t1_s.to_numpy(),[0,.05,.5,.95,1]).tolist())


def crossing_profile(frame,vertex,direction,centres):
    p=frame[['x0_m','y0_m','z0_m']].to_numpy();q=frame[['x1_m','y1_m','z1_m']].to_numpy()
    a=(p-vertex)@direction;b=(q-vertex)@direction
    i=np.searchsorted(centres,np.minimum(a,b),side='left');j=np.searchsorted(centres,np.maximum(a,b),side='left')
    diff=np.bincount(i,weights=frame.weight.to_numpy(),minlength=len(centres)+1)-np.bincount(j,weights=frame.weight.to_numpy(),minlength=len(centres)+1)
    return np.cumsum(diff[:-1])


def deposit_profile(frame,vertex,direction,edges):
    p=frame[['x0_m','y0_m','z0_m']].to_numpy();q=frame[['x1_m','y1_m','z1_m']].to_numpy()
    a=(p-vertex)@direction;b=(q-vertex)@direction;lo=np.minimum(a,b);hi=np.maximum(a,b)
    energy=frame.weighted_deposited_GeV.to_numpy();n=len(edges)-1
    assert np.all(lo>=edges[0]) and np.all(hi<=edges[-1])
    i=np.minimum(np.searchsorted(edges,lo,side='right')-1,n-1);j=np.minimum(np.searchsorted(edges,hi,side='right')-1,n-1)
    same=i==j;out=np.bincount(i[same],weights=energy[same],minlength=n)
    rate=energy[~same]/(hi[~same]-lo[~same]);l=i[~same];r=j[~same]
    out+=np.bincount(l,weights=rate*(edges[l+1]-lo[~same]),minlength=n)
    out+=np.bincount(r,weights=rate*(hi[~same]-edges[r]),minlength=n)
    diff=np.bincount(l+1,weights=rate,minlength=n+1)-np.bincount(r,weights=rate,minlength=n+1)
    out+=np.cumsum(diff[:-1])*np.diff(edges)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    assert socket.gethostname()=='psrpku2025'
    root=args.root;start=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc)
    progress=json.loads((root/'progress.json').read_text());folder=root/'runs'/progress['current'];terrain=folder/'output/terrain'
    summary=yaml.safe_load((folder/'output/terrain_run.yaml').read_text())
    out=root/'report'/('live_stage_'+stamp.strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    sizes={p.name:p.stat().st_size for p in terrain.glob('*.csv.gz')}
    (out/'input_snapshot.json').write_text(json.dumps(dict(captured_utc=stamp.isoformat(),progress=progress,byte_limits=sizes,
        warning='Independent flushed byte prefixes: track/deposit buffering differs. Neither the live stack nor unflushed resident records are inspected.'),indent=2)+'\n')
    direction=np.asarray(summary['direction']);direction/=np.linalg.norm(direction)
    first=next(frames(terrain/'tracks.csv.gz',min(sizes['tracks.csv.gz'],262144),{'history_id':'uint64','parent_history_id':'uint64'}))[0]
    initial=first[(first.history_id==1)&(first.pdg==16)]
    vertex=initial.iloc[-1][['x1_m','y1_m','z1_m']].to_numpy(dtype=float)
    entry=initial[initial.medium=='rock'].iloc[0][['x0_m','y0_m','z0_m']].to_numpy(dtype=float)
    vertex_time=float(initial.iloc[-1].t1_s)
    # Fine sampling around the first interaction, coarse coverage of the full
    # 500 us causal extent. Every recorded segment contributes to the profile.
    edges=np.unique(np.r_[np.arange(-160000.,-100.,25.),np.arange(-100.,300.,.05),np.arange(300.,160026.,25.)])
    centres=.5*(edges[:-1]+edges[1:]);crossing=np.zeros(len(centres));deposition=np.zeros(len(centres))
    pdgs=Counter();media=Counter();tail=deque();tail_rows=0;rows=0;sample=[];taus=[];neutrinos=[];root_children={};recent_cpu=deque(maxlen=100)
    previous_time=None;previous_s=None;back_time=0;back_depth=0;minimum_time=float('inf');maximum_time=0.;max_root_id=0
    snapshots={};cutoffs=[2000000,5000000,10000000]
    for frame,offset in frames(terrain/'tracks.csv.gz',sizes['tracks.csv.gz'],{'history_id':'uint64','parent_history_id':'uint64'}):
        old=rows;rows+=len(frame);assert int(frame.step.iloc[0])==old+1 and int(frame.step.iloc[-1])==rows
        pdgs.update({str(k):int(v) for k,v in frame.pdg.value_counts().items()});media.update(frame.medium)
        selected=frame[frame.parent_history_id==1]
        for identity,group in selected.groupby('history_id',sort=False):
            key=str(int(identity));max_root_id=max(max_root_id,int(identity))
            if key not in root_children:root_children[key]=json.loads(group.iloc[:1].to_json(orient='records',double_precision=15))[0]
        for code,records in [(15,taus),(16,neutrinos)]:
            selected=frame[np.abs(frame.pdg.to_numpy())==code]
            records.extend(json.loads(selected.to_json(orient='records',double_precision=15)))
        cpu=frame[~np.isin(frame.pdg.to_numpy(),[11,-11,22])]
        recent_cpu.extend(json.loads(cpu.tail(100).to_json(orient='records',double_precision=15)))
        t=frame.t0_s.to_numpy();s=(frame[['x1_m','y1_m','z1_m']].to_numpy()-vertex)@direction
        back_time+=int(np.count_nonzero(np.diff(np.r_[previous_time,t] if previous_time is not None else t)<-1e-15))
        back_depth+=int(np.count_nonzero(np.diff(np.r_[previous_s,s] if previous_s is not None else s)<-1e-6))
        previous_time=t[-1];previous_s=s[-1];minimum_time=min(minimum_time,float(t.min()));maximum_time=max(maximum_time,float(frame.t1_s.max()))
        for limit in cutoffs:
            if old<limit<=rows:
                electron=frame[(np.abs(frame.pdg.to_numpy())==11)&(frame.step<=limit)]
                snapshots[str(limit)]=crossing+crossing_profile(electron,vertex,direction,centres)
        electron=frame[np.abs(frame.pdg.to_numpy())==11]
        crossing+=crossing_profile(electron,vertex,direction,centres)
        chosen=frame[frame.step%5000==0][['step','pdg','medium','t0_s','t1_s','E0_GeV','x1_m','y1_m','z1_m']]
        if len(chosen):sample.append(chosen)
        tail.append(frame);tail_rows+=len(frame)
        while len(tail)>1 and tail_rows-len(tail[0])>=100000:tail_rows-=len(tail.popleft())
        if rows//1000000!=old//1000000:print('TRACKS',rows,'/',progress['latest']['recorded_steps_prefix'],flush=True)
    latest=pd.concat(tail,ignore_index=True).tail(100000)
    snapshots[str(rows)]=crossing.copy()
    total=np.longdouble(0);deposit_rows=0
    for frame,offset in frames(terrain/'deposits.csv.gz',sizes['deposits.csv.gz']):
        total+=np.sum(frame.weighted_deposited_GeV.to_numpy(),dtype=np.longdouble)
        deposition+=deposit_profile(frame,vertex,direction,edges);deposit_rows+=len(frame)
    assert abs(deposition.sum()-float(total))<1e-8*max(float(total),1.)
    peak=int(np.argmax(crossing));peak_dep=int(np.argmax(deposition/np.diff(edges)))
    result=dict(snapshot_utc=stamp.isoformat(),tag=folder.name,diagnostic_seconds=time.monotonic()-start,
        track_rows=rows,deposit_rows=deposit_rows,medium_counts=dict(media),pdg_counts=dict(pdgs),
        first_neutrino_vertex_m=vertex.tolist(),first_vertex_time_s=vertex_time,
        first_rock_entry_m=entry.tolist(),rock_distance_before_first_vertex_m=float((vertex-entry)@direction),
        recorded_tau_track_rows=len(taus),recorded_tau_histories=sorted({str(t['history_id']) for t in taus}),
        recorded_neutrino_track_rows=len(neutrinos),root_daughters_recorded=len(root_children),
        root_daughter_pdg_counts=dict(Counter(str(v['pdg']) for v in root_children.values())),
        sum_first_recorded_root_daughter_energy_GeV=sum(v['E0_GeV']*v['weight'] for v in root_children.values()),
        flushed_deposited_GeV=float(total),flushed_deposited_over_primary=float(total)/summary['energy_GeV'],
        recent_100000_tracks=stats(latest,vertex,direction),recent_scalar_tracks=list(recent_cpu),
        backwards_physical_time_transitions=back_time,backwards_axis_transitions=back_depth,
        recorded_physical_time_range_s=[minimum_time,maximum_time],
        partial_electron_crossing_peak_distance_from_first_vertex_m=float(centres[peak]),
        partial_electron_crossing_peak=float(crossing[peak]),
        partial_deposition_peak_distance_from_first_vertex_m=float(centres[peak_dep]),
        final_Xmax_determined=False,final_doublebang_topology_determined=False,
        caveat='Particle-stack/FIFO execution order is not monotone in shower depth/time. Flushed prefixes omit pending particles and unflushed records; partial profile maxima and deposited-energy fraction are not completion estimates.')
    (out/'stage.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'root_daughters_first_tracks.json').write_text(json.dumps(root_children,indent=2)+'\n')
    (out/'tau_tracks_prefix.json').write_text(json.dumps(taus,indent=2)+'\n')
    (out/'neutrino_tracks_prefix.json').write_text(json.dumps(neutrinos,indent=2)+'\n')
    latest.to_csv(out/'latest_100000_tracks.csv.gz',index=False,compression='gzip')
    np.savez_compressed(out/'partial_profiles.npz',edges_m=edges,electron_crossings=crossing,deposited_GeV=deposition,
                        **{'crossings_after_'+key:value for key,value in snapshots.items()})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    points=pd.concat(sample,ignore_index=True);distance=(points[['x1_m','y1_m','z1_m']].to_numpy()-vertex)@direction
    axes[0,0].scatter(points.step/1e6,distance,s=3,c=np.log10(np.maximum(points.E0_GeV,1e-12)),cmap='viridis')
    axes[0,0].set(xlabel='Recorded step number [million]',ylabel='Distance from first vertex [m]',title='Write order revisits shower depths')
    axes[0,1].scatter(points.step/1e6,points.t0_s*1e6,s=3)
    axes[0,1].set(xlabel='Recorded step number [million]',ylabel='Particle time [us]',title='Write order is not physical-time order')
    for key,values in snapshots.items():axes[1,0].plot(centres,values,label='First %.1f million steps'%(int(key)/1e6),lw=1)
    active=np.flatnonzero(crossing>max(float(crossing.max())*1e-4,1e-12))
    if len(active):axes[1,0].set_xlim(centres[active[0]]-.5,centres[active[-1]]+.5)
    axes[1,0].set(xlabel='Distance from first vertex [m]',ylabel='Weighted e+/e- plane crossings',title='Partial profiles only: not final Xmax');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(centres,deposition/np.diff(edges),lw=1)
    active=np.flatnonzero(deposition/np.diff(edges)>np.max(deposition/np.diff(edges))*1e-4)
    if len(active):axes[1,1].set_xlim(centres[active[0]]-.5,centres[active[-1]]+.5)
    axes[1,1].set(xlabel='Distance from first vertex [m]',ylabel='Recorded deposited energy [GeV/m]',title='Flushed deposits only; pending branches excluded')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('100 PeV nu_tau, seed 946 | live prefix, not a completed shower')
    fig.savefig(out/'live_stage.png',dpi=170);plt.close(fig)
    lines=['**运行阶段诊断：已落盘前缀，不是完整事例**','','快照 UTC：'+stamp.isoformat()+'。',
        '已扫描 %d 条轨迹和 %d 条沉积记录。程序继续运行，诊断未暂停或修改输运进程。'%(rows,deposit_rows),'',
        '![](live_stage.png)','','怎么看：上排按文件写入顺序画纵向位置和粒子物理时间，若反复回到较小深度/时间，就说明不能把最后一条轨迹当成全 shower 的进度。下排是部分电子/正电子穿越计数及部分能量沉积，峰值会随尚未处理的分支加入而变化，不能当成最终 Xmax。','',
        '首个中微子相互作用位置为 ENU (%.3f, %.3f, %.3f) m，距进入岩石约 %.3f m。'%(vertex[0],vertex[1],vertex[2],result['rock_distance_before_first_vertex_m']),
        '当前前缀中 τ 输运记录为 %d 条；首个中微子的已见直接子粒子有 %d 个。缺少某类记录不代表它未生成，可能还在栈中或尚未落盘。'%(len(taus),len(root_children)),
        '已落盘沉积 %.6g GeV，占初始能量 %.6g%%；该比例既不是完成百分比，也不是未完成事件的能量闭合验收。'%(float(total),float(total)/summary['energy_GeV']*100),'',
        '最近 10 万条记录的统计、完整计数与部分峰位见 [stage.json](stage.json)，初级子粒子证据见 [root_daughters_first_tracks.json](root_daughters_first_tracks.json)。','',
        '坐标横轴为沿初级方向的米数，不混称为克深度。尚无完整的两 bang 谱系与各自纵向分布，不能用这些部分峰位断言整个事件已越过 Xmax。轨迹和沉积分别按捕获时的字节上限读取，末尾未写完的行丢弃；它们的压缩缓冲落盘时刻并不完全一致。']
    (out/'README_CN.md').write_text('\n'.join(lines)+'\n')
    print('DIAGNOSTIC',str(out),json.dumps({k:v for k,v in result.items() if k!='recent_scalar_tracks'}),flush=True)


if __name__=='__main__':main()
