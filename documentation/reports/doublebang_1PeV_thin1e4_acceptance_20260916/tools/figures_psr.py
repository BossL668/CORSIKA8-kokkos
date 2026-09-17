#!/usr/bin/env python3
"""Readable plots of the completed 1 PeV event, using PSR data only."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket

assert socket.gethostname() == 'psrpku2025'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
ROOT = BASE/'beta5_doublebang_1PeV_thin1e4_queue4M_20260915'
TAG = 'openmp_SiO2_21CMA80_1PeV_seed22309'
RUN = ROOT/'runs'/TAG
REPORT = ROOT/'report'/TAG
OUT = REPORT/'figure_review_20260916'
OUT.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text())

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def save_figure(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/'figures'/(name+'.'+ext), dpi=160)
    plt.close(fig)
    print('FIGURE', name, flush=True)

plt.rcParams.update({'font.size':11, 'axes.titlesize':12, 'legend.fontsize':9})
s = yaml.safe_load((RUN/'output/terrain_run.yaml').read_text())
a = read(REPORT/'acceptance.json')
status = read(RUN/'status.json')
timing = read(RUN/'timing.json')
assert status['complete'] and status['returncode']==0 and a['passed'] and all(a['gates'].values())
assert not a['classification']['qualified_doublebang']
assert [v['current'] for v in s['neutrino']['interactions']]==['NC'] and s['tau']['tau_decay_count']==0
assert a['records']['all_tracks']==s['diagnostics']['steps']
assert a['records']['all_deposits']==s['diagnostics']['deposition_rows']
assert abs(s['energy_ledger']['unexplained_over_initial'])<1e-8

origin=np.array(s['position_m']);direction=np.array(s['direction'])
vertex=np.array(s['neutrino']['interactions'][0]['position_enu_m'])
vertex_distance=float((vertex-origin)@direction)
obs=s['scene']['radio']['observers'];xyz=np.array([x['position_enu_m'] for x in obs])
names=[x['name'] for x in obs]
helper_path=BASE/'beta5_material_doublebang_openmp256_20260913/code/geometry_figures_psr.py'
spec=importlib.util.spec_from_file_location('terrain_surface',helper_path)
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
mesh=Path(s['scene']['geometry']['mesh_path'])
assert hashlib.sha256(mesh.read_bytes()).hexdigest()==s['mesh_sha256']
vertices,height=geometry.surface(mesh)
surface_at_vertex=float(height(vertex[0],vertex[1]))
with np.load(ROOT/'bundle/terrain_grid.npz') as grid:
    east,north,up=[grid[k][::3,::3] for k in ['east_m','north_m','up_m']]
fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
im=axes[0].contourf(east/1000,north/1000,up,levels=30,cmap='terrain')
fig.colorbar(im,ax=axes[0],label='Terrain ENU Up [m]')
axes[0].scatter(xyz[:,0]/1000,xyz[:,1]/1000,s=7,c='purple',label='80 stations')
axes[0].plot([origin[0]/1000]*2,[origin[1]/1000,(origin[1]-2000)/1000],'r--',label='Section guide')
axes[0].scatter(origin[0]/1000,origin[1]/1000,c='red',marker='*',s=70,label='Injection')
axes[0].scatter(vertex[0]/1000,vertex[1]/1000,c='black',marker='x',s=60,label='NC vertex')
axes[0].set(xlabel='East [km]',ylabel='North [km]',title='Recorded geometry',aspect='equal')
axes[0].legend(loc='lower left',fontsize=8)
for ax,lo,hi,title in [(axes[1],-100,2000,'Section through injection axis'),
                       (axes[2],vertex_distance-60,vertex_distance+60,'NC vertex near the surface')]:
    distances=np.linspace(lo,hi,1500)
    z=height(origin[0]+direction[0]*distances,origin[1]+direction[1]*distances)
    low=min(float(z.min()),origin[2])-20;high=max(float(z.max()),origin[2])+20
    ax.fill_between(distances,low,z,color='#cdb795',label='SiO2 rock')
    ax.fill_between(distances,z,high,color='#dceef8',label='Air')
    ax.plot(distances,z,c='#6d5938',lw=1)
    ax.plot([0,vertex_distance],[origin[2],vertex[2]],c='#c0392b',lw=1.6,label='Incoming neutrino track')
    ax.scatter(vertex_distance,vertex[2],marker='x',c='black',s=65,zorder=5,label='NC interaction')
    ax.set(xlim=(lo,hi),ylim=(low,high),xlabel='Distance along injection axis [m]',ylabel='ENU Up [m]',title=title)
    ax.grid(alpha=.2)
axes[1].legend(fontsize=8,loc='upper right')
axes[2].annotate('No tau produced',(vertex_distance,vertex[2]),xytext=(8,-25),textcoords='offset points')
fig.suptitle('1 PeV | SiO2 | seed 22309 | Completed NC event')
save_figure(fig,'01_geometry_section')

with np.load(REPORT/'profile.npz') as p:
    edges=p['edges_m'];energy=p['deposited_GeV']
centres=(edges[:-1]+edges[1:])/2;density=energy/np.diff(edges)
profile_residual=float(energy.sum()-a['records']['deposited_GeV'])
assert abs(profile_residual)<1e-8*s['energy_GeV']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
axes[0].plot(centres/1000,density,lw=.9)
axes[0].set(xlim=(-.1,(origin[1]-vertices[:,1].min())/1000+.1),yscale='log',ylim=(1e-8,density.max()*4),
            xlabel='Distance along injection axis [km]',ylabel='Deposited energy [GeV/m]',title='Full domain profile; 1 m bins')
axes[1].stairs(density/1000,edges,lw=1.5)
axes[1].axvline(vertex_distance,c='black',ls='--',label='NC vertex')
axes[1].set(xlim=(vertex_distance-2,vertex_distance+14),xlabel='Distance along injection axis [m]',
            ylabel='Deposited energy [TeV/m]',title='Main shower, enlarged')
axes[1].legend()
for ax in axes:ax.grid(alpha=.2)
fig.suptitle(f'Complete deposition profile | Integral = {energy.sum()/1000:.3f} TeV | No second bang')
save_figure(fig,'02_shower_profile')

rows=a['radio']['stations'];byname={r['station']:r for r in rows}
peak=np.array([byname[name]['peak_50_100MHz_V_m']*1e12 for name in names])
strongest=max(rows,key=lambda r:r['peak_50_100MHz_V_m']);station=strongest['station']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
im=axes[0].scatter(xyz[:,0]/1000,xyz[:,1]/1000,c=peak,cmap='viridis',s=38,vmin=0)
fig.colorbar(im,ax=axes[0],label='Peak vector field [pV/m]')
imax=names.index(station);axes[0].annotate(station,xyz[imax,:2]/1000,xytext=(8,8),textcoords='offset points')
axes[0].set(xlabel='East [km]',ylabel='North [km]',title='All 80 station positions',aspect='equal')
for arm in 'EWNS':
    indices=np.arange(1,21)
    axes[1].plot(indices,[byname[f'{arm}{i:02d}']['peak_50_100MHz_V_m']*1e12 for i in indices],'o-',ms=3,label=arm+' arm')
axes[1].set(xlabel='Station number within arm',ylabel='Peak vector field [pV/m]',title='Every station, including zeros',xticks=[1,5,10,15,20])
axes[1].legend();axes[1].grid(alpha=.2)
fig.suptitle('50-100 MHz | 77 nonzero stations | S01, S02, S03 are zero')
save_figure(fig,'03_radio_array')

# Extract the strongest station from each archived, already fully audited field.
FIELDS=['Ex_V_m','Ey_V_m','Ez_V_m','Ex_outside','Ey_outside','Ez_outside','Ex_inside','Ey_inside','Ez_inside']
config=read(RUN/'output/radio/ZHS/config.json')
n=config['samples'];rate=config['sample_rate_Hz'];index=[x['name'] for x in config['observers']].index(station)
assert n==524288 and rate==256e6
fields=[]
for alg in ['CoREAS','ZHS']:
    path=RUN/'output/radio'/alg/'field.csv.gz'
    columns=['observer','time_s']+FIELDS
    frame=pd.read_csv(path,skiprows=1+index*n,nrows=n,header=None,names=columns,float_precision='round_trip')
    assert len(frame)==n and (frame.observer==index).all()
    field=frame[FIELDS].to_numpy();assert np.isfinite(field).all()
    times=frame.time_s.to_numpy();assert np.allclose(times,config['start_time_s']+np.arange(n)/rate,rtol=1e-14,atol=1e-16)
    assert np.linalg.norm(field[:,:3]-field[:,3:6]-field[:,6:])<=1e-13*np.linalg.norm(field)
    assert not np.any(field[:,6:])
    fields.append(field)
    print('EXTRACTED',alg,station,flush=True)
frequency=np.fft.rfftfreq(n,1/rate);band=(frequency>=50e6)&(frequency<=100e6)
transforms=[np.fft.rfft(field,axis=0) for field in fields]
filtered=[np.fft.irfft(ft*band[:,None],n=n,axis=0) for ft in transforms]
coreas,zhs=filtered
vector=np.linalg.norm(zhs[:,:3],axis=1);imax=int(np.argmax(vector));axis=int(np.argmax(abs(zhs[imax,:3])))
assert np.isclose(vector[imax],strongest['peak_50_100MHz_V_m'],rtol=1e-12,atol=0)
relative=float(np.linalg.norm(coreas[:,:3]-zhs[:,:3])/np.linalg.norm(zhs[:,:3]))
assert relative<1e-5 and np.isclose(relative,strongest['filtered_algorithm_relative_l2'],rtol=1e-4,atol=1e-14)
zoom=slice(max(0,imax-100),min(n,imax+101));relative_time_ns=(times-times[imax])*1e9
factor=64;blocks=zhs[:,axis].reshape(-1,factor);starts=np.arange(len(blocks))*factor
display=np.unique(np.r_[starts+np.argmax(blocks,axis=1),starts+np.argmin(blocks,axis=1),imax])
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
for field,label,ls in [(zhs,'ZHS','-'),(coreas,'CoREAS','--')]:
    axes[0,0].plot(times[display]*1e6,field[display,axis]*1e12,ls,lw=.8,label=label)
    axes[0,1].plot(relative_time_ns[zoom],field[zoom,axis]*1e12,ls,lw=1.2,label=label)
axes[0,0].set(xlabel='Arrival time [us]',ylabel=f'E{"xyz"[axis]} [pV/m]',title='50-100 MHz; full recorded window')
axes[0,1].set(xlabel='Time relative to strongest sample [ns]',ylabel=f'E{"xyz"[axis]} [pV/m]',title='Pulse zoom; every sample')
for start,label,color in [(3,'Air source','#1f77b4'),(6,'Rock source','#ff7f0e'),(0,'Coherent total','black')]:
    axes[1,0].plot(relative_time_ns[zoom],zhs[zoom,start+axis]*1e12,label=label,c=color,ls='--' if start==0 else '-')
axes[1,0].set(xlabel='Time relative to strongest sample [ns]',ylabel=f'E{"xyz"[axis]} [pV/m]',title='Received source components')
take=np.flatnonzero(band);stride=max(1,len(take)//8000)
for ft,label,ls in [(transforms[1],'ZHS','-'),(transforms[0],'CoREAS','--')]:
    amplitude=np.linalg.norm(ft[:,:3],axis=1)/rate*1e18
    show=np.unique(np.r_[take[::stride],take[np.argmax(amplitude[take])]])
    axes[1,1].plot(frequency[show]/1e6,amplitude[show],ls,label=label,lw=.8)
axes[1,1].set(xlabel='Frequency [MHz]',ylabel=r'$|\widetilde{\mathbf{E}}(f)|$ [pV m$^{-1}$ MHz$^{-1}$]',title='Fourier amplitude; dt-normalized',xlim=(50,100))
for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(f'{station} | Peak vector field = {vector[imax]*1e12:.3f} pV/m | Rock-source field = 0')
save_figure(fig,'04_waveform_spectrum_'+station)

fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for arm in 'EWNS':
    ids=[i for i in range(1,21) if byname[f'{arm}{i:02d}']['filtered_algorithm_relative_l2'] is not None]
    axes[0].semilogy(ids,[byname[f'{arm}{i:02d}']['filtered_algorithm_relative_l2'] for i in ids],'o-',ms=3,label=arm+' arm')
axes[0].set(xlabel='Station number within arm',ylabel=r'$\|\mathbf{E}_{CoREAS}-\mathbf{E}_{ZHS}\|_2/\|\mathbf{E}_{ZHS}\|_2$',title='50-100 MHz; all nonzero stations',xticks=[1,5,10,15,20])
axes[0].legend();axes[0].grid(alpha=.2)
residual=(coreas[:,axis]-zhs[:,axis])/vector[imax]
axes[1].plot(relative_time_ns[zoom],residual[zoom],lw=1)
axes[1].set(xlabel='Time relative to strongest sample [ns]',ylabel=r'$(E_{CoREAS,x}-E_{ZHS,x})/\max_t |\mathbf{E}_{ZHS}|$'.replace(',x',','+'xyz'[axis]),title=station+': pointwise component residual')
axes[1].grid(alpha=.2)
fig.suptitle('Algorithm agreement | Shared propagation; not an independent physics validation')
save_figure(fig,'05_algorithm_comparison')

resources=[json.loads(x) for x in (RUN/'resources.jsonl').read_text().splitlines()]
elapsed=np.array([r['elapsed_s'] for r in resources])/3600
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
axes[0].plot(elapsed,[r['rss_GiB'] for r in resources],c='#1f77b4',label='RSS')
axes[0].set(xlabel='Elapsed simulation time [h]',ylabel='Resident memory [GiB]',title='Measured memory and written track prefix')
twin=axes[0].twinx();twin.plot(elapsed,np.array([r['recorded_steps_prefix'] for r in resources])/1e6,c='#d35400',label='Track prefix')
twin.set_ylabel('Written track records [million]');axes[0].grid(alpha=.2)
axes[0].legend(loc='upper left');twin.legend(loc='lower right')
execution=timing['execution_timing']
durations=np.array([execution['transport_loop_seconds'],execution['radio_flush_download_write_seconds'],
                    timing['process_wall_seconds']-sum(execution[k] for k in ['transport_loop_seconds','radio_flush_download_write_seconds']),
                    timing['validation_and_figures_seconds'],timing['lossless_archive_seconds']])/60
labels=['Transport + in-loop radio','Final radio / file output','Other process work','Original audit + plots','Lossless archive']
bars=axes[1].barh(labels,durations,color=['#2c7fb8','#7fcdbb','#c7e9b4','#fec44f','#fc8d59'])
axes[1].invert_yaxis();axes[1].set(xlabel='Wall time [min]',title='Stage timings; labels give exact minutes',xlim=(0,max(durations)*1.22))
for bar,value in zip(bars,durations):axes[1].text(value+5,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center')
fig.suptitle(f'1 PeV | Simulation {timing["process_wall_seconds"]/3600:.3f} h | Peak RSS {timing["peak_rss_GiB"]:.2f} GiB')
save_figure(fig,'06_timing_resources')

np.savez_compressed(OUT/'E18_fields_and_spectra.npz',time_s=times,raw_coreas=fields[0],raw_zhs=fields[1],
                    frequency_Hz=frequency,filtered_coreas=coreas,filtered_zhs=zhs)
metrics=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_event=True,doublebang=False,
             source_full_audit='../acceptance.json',record_counts=a['records'],profile_sum_residual_GeV=profile_residual,
             energy_ledger_unexplained_relative=s['energy_ledger']['unexplained_over_initial'],
             geometry_mesh_sha256=s['mesh_sha256'],vertex_distance_m=vertex_distance,
             surface_at_vertex_ENU_m=surface_at_vertex,vertical_overburden_at_vertex_m=surface_at_vertex-vertex[2],
             strongest_station=station,peak_filtered_V_m=float(vector[imax]),strongest_station_filtered_relative_l2=relative,
             full_audit_relative_l2=a['radio']['relative_l2_by_source'],rock_receiver_field_zero=True,
             accepted_transmitted_paths=s['radio_result']['ZHS']['transmitted_paths'],
             effective_band_MHz=[50,100],sample_rate_Hz=rate,nyquist_Hz=rate/2,
             spectrum_definition='Two-sided Fourier amplitude at positive f: dt * rfft(E); vector norm over xyz, no one-sided factor of 2. Multiply V/m/Hz by 1e18 for pV/m/MHz.',
             display_only_reduction='Time overview retains per-block minima/maxima. Spectral display is decimated. Metrics, peak, zoom, and waveform FFT use all samples.',
             new_review_scope='Rechecked summary/gates/profile; strongest-station archived fields re-read and recomputed. Original complete audit examined all track/deposit rows and all 80 station samples.',
             timing=timing)
save_json(OUT/'figure_checks.json',metrics)
shutil.copy2(Path(__file__),OUT/'figures_psr.py')
pages=['---','marp: true','paginate: true','---','',
       '# 1 PeV SiO₂ 模拟：验收与读图','',
       '**计算完整性通过；实际是 NC 单次 shower，没有 τ，也没有 double bang。**', '',
       'seed 22309，emthin=1e-4，OpenMP 256 核，全部 80 站。当前 100 PeV/130 核的新任务另行记录，本报告不包含其未完成结果。', '',
       '![山体剖面](figures/01_geometry_section.png)', '',
       '从左到右看站点布局、入射轴上的山体剖面、NC 顶点附近的放大图。红线是实际入射中微子轨迹，不是射电光路。棕色为 SiO₂，浅蓝色为空气；没有画出不存在的 τ 轨迹。', '',
       '---','', '# Shower profile','', '![能量沉积](figures/02_shower_profile.png)','',
       '左图看域内尾部，右图看主要 shower。两图都是完整沉积记录的 1 m 分箱；纵轴单位分别为 GeV/m 和 TeV/m。积分约 421.588 TeV。', '',
       '主峰分箱中心在注入点前方 1420.5 m，NC 顶点在 1418.971 m；这是沉积能量峰，不能直接当作粒子数定义的 Xmax。左图对数轴下限为 1e-8 GeV/m，原始全分箱仍保留。', '',
       '---','', '# 哪些站信号更强','', '![80站信号](figures/03_radio_array.png)','',
       '左图看空间分布，右图按 E/W/N/S 四臂比较全部 80 站。颜色和纵轴都是 50–100 MHz 矢量场峰值，单位 pV/m。', '',
       'E18 最强，约 2.15 pV/m；77 站非零，S01/S02/S03 的结果为零，但并没有缺行。非零也不等于实验上可探测，这里展示的是传播后的电场。', '',
       '---','', '# 最强站的波形、频谱和源分量','', '![最强站](figures/04_waveform_spectrum_E18.png)','',
       '左上看整段时间窗，右上看峰值附近；实线 ZHS、虚线 CoREAS，横轴无需手动平移来匹配。峰值附近保留每一个采样点。纵轴改用 pV/m，避免 μV/m 后再乘 1e-6。', '',
       '左下的总场与空气源重合，山体源为零。右下是乘以采样间隔后的 Fourier 幅度，不是功率谱密度；频率图只展示 50–100 MHz。', '',
       '**有效透射光路数为 0，不能用这例证明山体没有辐射，也不能用于验收山体透射或衰减模型。** 当前采样率 256 MHz，不能覆盖完整 50–300 MHz。', '',
       '---','', '# CoREAS 与 ZHS 差多少','', '![算法对比](figures/05_algorithm_comparison.png)','',
       '左图纵轴为 ||E_CoREAS−E_ZHS||₂ / ||E_ZHS||₂，范数包含各时刻和三个分量；这里只比较 50–100 MHz。零信号站的比值未定义，所以没有画成“误差为零”。', '',
       '右图把 E18 的一个分量逐点差除以 ZHS 矢量峰值，查看差异是否集中在脉冲附近。原全频、全站总场相对 L2 差为 2.35e-9；带通后各非零站最大约 2.15e-10。', '',
       '两算法共享部分源项和传播处理，重合是数值一致性证据，不是独立的物理验证。', '',
       '---','', '# 耗时与验收边界','', '![资源与耗时](figures/06_timing_resources.png)','',
       '模拟进程耗时 18 小时 10 分 29 秒。主要时间花在输运循环及循环内射电；最终射电重建和写出约 10 分 40 秒。峰值 RSS 151.27 GiB 出现在最后阶段。', '',
       '17,636,988 条轨迹、16,784,148 条沉积记录、34 条出界末态完整；队列已清空。计入生成器、截止静质量与随机薄化账项后，未解释能量残差相对初始能量约 1.15e-11。该账本闭合不代表全部物理模型已经独立验证。', '',
       '[原验收结论](../review_20260916/README_CN.md) · [本次图件核对数值](figure_checks.json) · [80 站全部波形](../README_CN.md) · [80 站数值](../station_signals.csv)','']
(OUT/'FIGURES_CN.md').write_text('\n'.join(pages))
(OUT/'README_CN.md').write_text('# 1 PeV 完成事件图件\n\n[按顺序读图：PPT Markdown](FIGURES_CN.md)。新增 6 张英文图，各有 PNG 和独立 PDF；波形单位改为 pV/m，提供真实地形剖面、全 profile 与主峰放大、80 站峰值、E18 波形频谱、算法误差和耗时。\n\n本例完整性与数值一致性通过，但没有 double bang，山体源透射信号也不能由此例验收。[原验收](../review_20260916/README_CN.md)。\n\n[本次复核数值](figure_checks.json) · [E18 可复算数组](E18_fields_and_spectra.npz) · [PSR 绘图脚本](figures_psr.py)\n')
for path in [REPORT/'README_CN.md',REPORT/'review_20260916/README_CN.md']:
    relative='figure_review_20260916/FIGURES_CN.md' if path.parent==REPORT else '../figure_review_20260916/FIGURES_CN.md'
    text=path.read_text()
    if relative not in text:
        path.write_text('[新增六张图及简短读法]('+relative+')：真实山体剖面、profile、80 站信号、波形频谱、算法差异与耗时。\n\n'+text)
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
(OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in files))
print('COMPLETE',json.dumps({k:metrics[k] for k in ['complete_event','doublebang','peak_filtered_V_m','strongest_station_filtered_relative_l2','vertical_overburden_at_vertex_m']}),flush=True)
