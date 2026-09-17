#!/usr/bin/env python3
"""Diagnose the saved 1 PeV event by actual track medium, exclusively on PSR."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import socket
import time
import numpy as np
import pandas as pd
import yaml

assert socket.gethostname() == 'psrpku2025'
root = Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_1PeV_thin1e4_queue4M_20260915')
tag = 'openmp_SiO2_21CMA80_1PeV_seed22309'
run = root/'runs'/tag
report = root/'report'/tag
out = report/'figure_review_20260916'
summary = yaml.safe_load((run/'output/terrain_run.yaml').read_text())
assert summary['complete']
origin = np.array(summary['position_m'])
direction = np.array(summary['direction']); direction /= np.linalg.norm(direction)
sv = (np.array(summary['neutrino']['interactions'][0]['position_enu_m'])-origin)@direction
with np.load(out/'particle_crossing_profile.npz') as saved:
    planes = saved['planes_m']
    reference = saved['weighted_crossings'][0]
delta = np.zeros((2,4,len(planes)+1), dtype=np.longdouble)
steps = np.zeros(2, dtype=np.int64)
charged_steps = np.zeros(2, dtype=np.int64)
weighted_lengths = np.zeros(2, dtype=np.longdouble)
air_ids = set()
start = time.monotonic()
cols = ['step','pdg','weight','medium','x0_m','y0_m','z0_m','x1_m','y1_m','z1_m','t0_s','t1_s','history_id']
for f in pd.read_csv(run/'output/terrain/tracks.csv.gz', usecols=cols, dtype={'history_id':'uint64'}, chunksize=200000):
    a = f[['x0_m','y0_m','z0_m']].to_numpy()
    b = f[['x1_m','y1_m','z1_m']].to_numpy()
    s0 = (a-origin)@direction; s1 = (b-origin)@direction
    left = np.searchsorted(planes,s0,side='right'); right = np.searchsorted(planes,s1,side='right')
    pid = abs(f.pdg.to_numpy()); weight = f.weight.to_numpy()
    groups = [pid==11,pid==22,pid==13,np.isin(pid,[211,321,2212,3222,3112,3312,3334])|
              ((pid>=1000000000)&((pid//10000)%1000>0))]
    moving = np.any(a!=b,axis=1)&(f.t1_s.to_numpy()>f.t0_s.to_numpy())
    charged = groups[0]|groups[2]|groups[3]|(pid==15)
    assert f.medium.isin(['air','rock']).all()
    for i,medium in enumerate(['air','rock']):
        take = f.medium.to_numpy()==medium
        steps[i] += int(take.sum())
        charged_steps[i] += int(np.count_nonzero(take&moving&charged))
        weighted_lengths[i] += np.sum(np.linalg.norm(b[take&charged]-a[take&charged],axis=1)*weight[take&charged],dtype=np.longdouble)
        if i==0: air_ids.update(map(int,f.history_id[take&charged]))
        for j,g in enumerate(groups):
            keep = take&g&(s1>s0)
            delta[i,j] += np.bincount(left[keep],weights=weight[keep],minlength=len(planes)+1)
            delta[i,j] -= np.bincount(right[keep],weights=weight[keep],minlength=len(planes)+1)
counts = np.cumsum(delta[:,:,:-1],axis=2).astype(float)
error = float(abs(counts.sum(axis=0)-reference).max())
assert error < 1e-6
assert steps.tolist() == [summary['diagnostics']['air_steps'],summary['diagnostics']['rock_steps']]
radio = summary['radio_result']['ZHS']
assert int(charged_steps.sum()) == radio['cpu_tracks']+radio['device_tracks']
with np.load(out/'E18_fields_and_spectra.npz') as saved:
    raw = saved['raw_zhs']; filtered = saved['filtered_zhs']
    raw_peaks = [float(np.linalg.norm(raw[:,k:k+3],axis=1).max()) for k in [3,6]]
    filtered_peaks = [float(np.linalg.norm(filtered[:,k:k+3],axis=1).max()) for k in [3,6]]
assert raw_peaks[1] == 0 and np.array_equal(raw[:,:3],raw[:,3:6])
checks = dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_medium_track_steps=dict(zip(['air','rock'],map(int,steps))),
    actual_medium_moving_charged_steps=dict(zip(['air','rock'],map(int,charged_steps))),
    air_charged_history_count=len(air_ids),
    weighted_charged_path_m=dict(zip(['air','rock'],map(float,weighted_lengths))),
    maximum_air_forward_crossings_by_species=dict(zip(['electron_positron','photon','muon','charged_hadron'],map(float,counts[0].max(axis=1)))),
    maximum_air_electron_forward_crossings_distance_from_vertex_m=float(planes[np.argmax(counts[0,0])]-sv),
    medium_profile_sum_max_abs_error=error,tracks_scanned=int(steps.sum()),
    accepted_direct_paths=radio['direct_paths'],accepted_transmitted_paths=radio['transmitted_paths'],
    E18_raw_vector_peak_V_m=dict(zip(['air','rock'],raw_peaks)),
    E18_50_100MHz_vector_peak_V_m=dict(zip(['air','rock'],filtered_peaks)),
    scan_seconds=time.monotonic()-start,
    interpretation='Track steps and history counts are not N(X). Profiles sum weighted forward plane crossings, grouped by actual segment medium. Receiver-source fields are grouped by emission medium, not propagation medium. Zero accepted transmission is established, but the exact rejection causes have not been independently decomposed.')
np.savez_compressed(out/'actual_medium_particle_profile.npz',planes_m=planes,
    weighted_forward_crossings=counts,media=np.array(['air','rock']),reference_vertex_distance_m=sv)
(out/'source_medium_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'legend.fontsize':9})
fig,axs = plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
x = planes-sv
have = reference.max(axis=0)>1e-6
end = float(x[have].max())
for j in [0,1]:
    ax = axs[0,j]
    for y,label,color in [(reference[0],'All electron / positron segments','black'),
        (counts[1,0],'Rock segments','#a47a3b'),(counts[0,0],'Air segments','tab:blue')]:
        ax.plot(x,np.where(y>1e-6,y,np.nan) if j else np.maximum(y,0),label=label,color=color,lw=1.3)
    ax.set(xlim=(-2,25) if j==0 else (-20,end+20),ylabel='Weighted forward e-/e+ crossings',
        xlabel='Distance from NC vertex along reference axis [m]',title='Main shower: linear scale' if j==0 else 'Whole profile: logarithmic scale')
    if j: ax.set_yscale('log')
    ax.legend()
for i,label in enumerate(['Electrons + positrons','Photons','Muons','Charged hadrons / nuclei']):
    y=counts[0,i];axs[1,0].plot(x,np.where(y>1e-6,y,np.nan),label=label)
axs[1,0].set(xlim=(-20,end+20),yscale='log',ylabel='Weighted forward crossings in AIR',
    xlabel='Distance from NC vertex along reference axis [m]',title='Air segments only (actual recorded medium)')
axs[1,0].legend()
ax=axs[1,1]
ax.bar(['Air source','Rock source'],np.array(raw_peaks)*1e12,color=['tab:blue','#a47a3b'])
ax.set(ylabel='Raw received vector peak [pV/m]',title='Station E18 | saved bandwidth 0-128 MHz',ylim=(0,raw_peaks[0]*1e12*1.4))
for i,v in enumerate(raw_peaks):ax.text(i,v*1e12+.1,f'{v*1e12:.3f}',ha='center')
ax.text(.5,.93,f"Accepted transmitted paths: {radio['transmitted_paths']}",ha='center',va='top',transform=ax.transAxes)
for ax in axs.flat:ax.grid(alpha=.2)
fig.suptitle('1 PeV | SiO2 | seed 22309 | Particle profile and receiver field measure different quantities')
for ext in ['png','pdf']:fig.savefig(out/'figures'/('11_actual_medium_profile_and_radio.'+ext),dpi=155)
plt.close(fig)
note=r'''

---

空气粒子少，为什么接收信号来自空气？

![w:1020](figures/11_actual_medium_profile_and_radio.png)

左上用线性轴看主 shower，空气曲线几乎贴地；右上改为对数轴，左下只画实际处于空气中的轨迹。分组读取每段轨迹的真实介质，与参考轴的地形底色不同。

右下是站点收到的源分量，不是两种介质总共发射了多少。山体虽有大量粒子，这例却没有被当前传播模型接受的岩石→空气透射光路；只有少量空气源的直达贡献，最强站也只有 pV/m 量级。

零透射不能单凭此图归因于吸收、全反射或地形遮挡；具体拒绝原因尚未独立分解。本例不能用来证明山体透射物理正确。[逐项检查](source_medium_checks.json)
'''
deck=out/'FIGURES_CN.md';old=deck.read_text();marker='\n\n---\n\n空气粒子少，为什么接收信号来自空气？'
if marker in old:old=old.split(marker)[0]
deck.write_text(old.rstrip()+note+'\n')
shutil.copy2(__file__,out/'source_medium_profile_psr.py')
for directory in [out,report]:
    rows=[]
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(directory).as_posix())
    (directory/'SHA256SUMS').write_text('\n'.join(rows)+'\n')
print(json.dumps(checks),flush=True)
