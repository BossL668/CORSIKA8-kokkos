#!/usr/bin/env python3
"""Add unfiltered saved-band waveforms and track-crossing profiles on PSR."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time

assert socket.gethostname() == 'psrpku2025'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_1PeV_thin1e4_queue4M_20260915')
TAG = 'openmp_SiO2_21CMA80_1PeV_seed22309'
RUN = ROOT/'runs'/TAG
REPORT = ROOT/'report'/TAG
OUT = REPORT/'figure_review_20260916'
started = time.monotonic()
plt.rcParams.update({'font.size': 11, 'legend.fontsize': 9})
s = yaml.safe_load((RUN/'output/terrain_run.yaml').read_text())
a = json.loads((REPORT/'acceptance.json').read_text())
assert a['passed'] and s['complete']
c = json.loads((RUN/'output/radio/ZHS/config.json').read_text())
rate = c['sample_rate_Hz']
assert rate == 256e6

def figure(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/'figures'/(name+'.'+ext), dpi=160)
    plt.close(fig)
    print('FIGURE', name, flush=True)

def envelope(ax, times, values, **kwargs):
    # Every sample contributes to its display block; isolated pulses survive.
    block = 64
    n = len(values)//block*block
    v = values[:n].reshape(-1, block)
    t = times[:n].reshape(-1, block).mean(axis=1)
    ax.fill_between(t, v.min(axis=1), v.max(axis=1), **kwargs)

with np.load(OUT/'E18_fields_and_spectra.npz') as data:
    t = data['time_s']; z = data['raw_zhs']; x = data['raw_coreas']
    band_old = data['filtered_zhs']
assert len(t) == c['samples'] and np.isfinite(z).all() and np.isfinite(x).all()
assert np.allclose(np.diff(t), 1/rate, rtol=1e-10, atol=1e-18)
peak = int(np.argmax(np.linalg.norm(z[:, :3], axis=1)))
zoom = slice(max(0, peak-80), min(len(t), peak+81))
f = np.fft.rfftfreq(len(t), 1/rate)
fft = np.fft.rfft(z[:, :3], axis=0)
power_time = float(np.sum(z[:, :3]**2)/rate)
fac = np.full(len(f), 2.); fac[[0, -1]] = 1.
power_fft = float(np.sum(fac[:, None]*abs(fft)**2)/(len(t)*rate))
assert abs(power_time-power_fft) < 1e-12*power_time
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for j, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
    label = ['East', 'North', 'Up'][j]
    envelope(ax[0, 0], t*1e6, z[:, j]*1e12, color=color, alpha=.6, label=label)
    ax[0, 1].plot((t[zoom]-t[peak])*1e9, z[zoom, j]*1e12, color=color, label='ZHS '+label)
    ax[0, 1].plot((t[zoom]-t[peak])*1e9, x[zoom, j]*1e12, '--', color=color, lw=.8)
ax[0, 0].set(xlabel='Arrival time [us]', ylabel='Electric field [pV/m]', title='Entire stored window; min/max display envelope')
ax[0, 1].set(xlabel='Time relative to raw vector peak [ns]', ylabel='Electric field [pV/m]', title='Unfiltered pulse: ZHS solid, CoREAS dashed')
axis = int(np.argmax(abs(z[peak, :3])))
ax[1, 0].plot((t[zoom]-t[peak])*1e9, z[zoom, axis]*1e12, label='Unfiltered stored band (0-128 MHz)')
ax[1, 0].plot((t[zoom]-t[peak])*1e9, band_old[zoom, axis]*1e12, label='Previous 50-100 MHz band')
ax[1, 0].set(xlabel='Time relative to raw vector peak [ns]', ylabel='Electric field [pV/m]', title=['East', 'North', 'Up'][axis]+' component: effect of the display filter')
spectral = np.linalg.norm(fft, axis=1)/rate*1e18
ax[1, 1].plot(f[1::32]/1e6, spectral[1::32], lw=.8)
ax[1, 1].axvspan(128, 300, color='0.9', label='Not present in saved waveform')
ax[1, 1].axvline(128, color='crimson', ls='--', label='Nyquist: 128 MHz')
ax[1, 1].set(xlim=(0, 300), xlabel='Frequency [MHz]', ylabel=r'$|\widetilde{\mathbf{E}}(f)|$ [pV/m/MHz]', title='Fourier amplitude; no power-density normalization')
for a0 in ax.flat:
    a0.legend(); a0.grid(alpha=.2)
fig.suptitle('E18 | 1 PeV NC event | Saved bandwidth 0-128 MHz | No instrumental response')
figure(fig, '07_unfiltered_waveform_E18')

origin = np.array(s['position_m']); direction = np.array(s['direction'])
direction /= np.linalg.norm(direction)
vertex = np.array(s['neutrino']['interactions'][0]['position_enu_m'])
sv = float((vertex-origin)@direction)
cache = OUT/'particle_crossing_profile.npz'
if not cache.exists():
    limit = np.ceil(299792458*s['transport_window_ns']*1e-9+100)
    planes = np.unique(np.r_[np.arange(-limit, limit+10, 10.), sv+np.arange(-20., 100., .1)])
    labels = ['electrons_positrons', 'photons', 'muons', 'charged_hadrons']
    delta = np.zeros((2, 4, len(planes)+1), dtype=np.longdouble)
    count = 0; maximum_weight = 0.
    use = ['step', 'pdg', 'weight', 'x0_m', 'y0_m', 'z0_m', 'x1_m', 'y1_m', 'z1_m']
    for frame in pd.read_csv(RUN/'output/terrain/tracks.csv.gz', usecols=use, chunksize=150000):
        assert np.array_equal(frame.step.to_numpy(), np.arange(count+1, count+len(frame)+1))
        count += len(frame)
        s0 = (frame[['x0_m', 'y0_m', 'z0_m']].to_numpy()-origin)@direction
        s1 = (frame[['x1_m', 'y1_m', 'z1_m']].to_numpy()-origin)@direction
        w = frame.weight.to_numpy(); pid = abs(frame.pdg.to_numpy())
        assert np.isfinite(s0).all() and np.isfinite(s1).all() and np.all(w > 0)
        maximum_weight = max(maximum_weight, float(w.max()))
        # Half-open intervals exclude a segment start and include its end.
        left_f = np.searchsorted(planes, s0, side='right')
        right_f = np.searchsorted(planes, s1, side='right')
        left_b = np.searchsorted(planes, s1, side='left')
        right_b = np.searchsorted(planes, s0, side='left')
        groups = [pid == 11, pid == 22, pid == 13,
                  np.isin(pid, [211, 321, 2212, 3222, 3112, 3312, 3334]) |
                  ((pid >= 1000000000) & ((pid//10000) % 1000 > 0))]
        for k, (left, right, moving) in enumerate([(left_f, right_f, s1 > s0), (left_b, right_b, s1 < s0)]):
            for j, group in enumerate(groups):
                take = moving & group
                delta[k, j] += np.bincount(left[take], weights=w[take], minlength=len(planes)+1)
                delta[k, j] -= np.bincount(right[take], weights=w[take], minlength=len(planes)+1)
        if count % 1500000 == 0:
            print('TRACKS', count, flush=True)
    assert count == s['diagnostics']['steps'] == a['records']['all_tracks']
    profile = np.cumsum(delta[:, :, :-1], axis=2).astype(float)
    residue = float(max(0., -profile.min()))
    assert residue < 1e-7*max(1., profile.max())
    # Keep the accumulated values, including tiny subtraction roundoff, in data.
    np.savez_compressed(cache, planes_m=planes, weighted_crossings=profile,
                        labels=np.array(labels), tracks_scanned=count,
                        negative_roundoff=residue, maximum_weight=maximum_weight)
with np.load(cache) as data:
    planes = data['planes_m']; profile = data['weighted_crossings']
    count = int(data['tracks_scanned']); residue = float(data['negative_roundoff'])
assert count == s['diagnostics']['steps']
with np.load(REPORT/'profile.npz') as data:
    edges = data['edges_m']; energy = data['deposited_GeV']
centres = .5*(edges[:-1]+edges[1:])
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
for j, label in enumerate(['Electrons + positrons', 'Photons', 'Muons', 'Charged hadrons / nuclei']):
    for panel in [ax[0, 0], ax[0, 1]]:
        values = profile[0, j]
        panel.plot(planes-sv, np.where(values > max(1e-6, residue*10), values, np.nan), color=colors[j], label=label)
for panel in ax[0]:
    panel.set(xlabel='Distance from NC vertex along injection axis [m]', ylabel=r'Forward crossings $N_w(s)=\sum_{i\ crossing\ s}w_i$')
    panel.grid(alpha=.2); panel.legend()
ax[0, 0].set(xlim=(-2, 30), title='Particle shower profile; 0.1 m planes near vertex')
ax[0, 1].set(xlim=(-10, 8000), yscale='log', title='Extended tails; 10 m planes away from vertex')
ax[1, 0].plot(planes-sv, profile[0, 0], label='Forward electron / positron crossings')
ax[1, 0].plot(planes-sv, profile[1, 0], label='Backward electron / positron crossings')
ax[1, 0].set(xlim=(-2, 30), xlabel='Distance from NC vertex along injection axis [m]', ylabel='Weighted plane crossings', title='Scattering can produce repeated plane crossings')
ax[1, 1].stairs(energy/np.diff(edges)/1000, edges-sv, label='All weighted energy deposits')
ax[1, 1].set(xlim=(-2, 30), xlabel='Distance from NC vertex along injection axis [m]', ylabel='Deposited energy per axis length [TeV/m]', title='Energy deposition profile; 1 m bins')
for panel in ax[1]:
    panel.legend(); panel.grid(alpha=.2)
fig.suptitle('1 PeV SiO2 NC event | Longitudinal particle and energy-deposition profiles')
figure(fig, '08_particle_shower_profile')

metrics = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    stored_sample_rate_Hz=rate, stored_nyquist_Hz=rate/2,
    stored_raw_vector_peak_V_m=float(np.linalg.norm(z[:, :3], axis=1).max()),
    raw_coreas_zhs_relative_l2=float(np.linalg.norm(x[:, :3]-z[:, :3])/np.linalg.norm(z[:, :3])),
    parseval_relative_error=abs(power_time-power_fft)/power_time,
    tracks_scanned=count, electron_forward_profile_peak=float(profile[0, 0].max()),
    electron_forward_peak_distance_from_vertex_m=float(planes[np.argmax(profile[0, 0])]-sv),
    profile_negative_roundoff=residue, elapsed_s=time.monotonic()-started,
    profile_definition='Weighted forward/backward crossings of planes normal to the initial direction; repeated recrossings count. Not an instantaneous population or an Xmax in g/cm2.',
    unavailable_band='The saved 256 MHz waveform does not contain 128-250 MHz; no interpolation or zero-padding is used to claim that band.')
(OUT/'bandwidth_profile_checks.json').write_text(json.dumps(metrics, indent=2)+'\n')
addition = r'''

---

未经带通的时域波形：先看这里

![w:1000](figures/07_unfiltered_waveform_E18.png)

左上是完整接收窗，右上放大脉冲并比较三方向分量；左下比较原始输出与旧的 50–100 MHz 带通。滤波会改变峰值和振铃。

这次旧输出的采样率是 256 MHz，因此“未经带通”仅指已保存的 0–128 MHz。右下灰区没有数据，不能补零或插值冒充 50–250 MHz。后续配置改与当前空气任务一致的 1 GHz 采样，分析 50–250 MHz，并附 50–300 MHz。

---

Shower profile：粒子数和沉积能量一起看

![w:1000](figures/08_particle_shower_profile.png)

横轴是沿入射方向、相对实际 NC 顶点的距离。左上用 0.1 m 间隔统计电子/正电子、光子等的加权向前穿面数，右上用对数轴看长尾；每条实际轨迹都已扫描，使用 thinning 权重。

左下显示散射后的反向穿面数；同一粒子反复穿面会再次计数，因此这是穿面 profile。右下是所有沉积的 $dE_{dep}/ds$，不是粒子数，也不能直接称为以 g/cm² 表示的 $X_{max}$。本例只有一次 NC shower，没有 double bang。
'''
deck = OUT/'FIGURES_CN.md'
text = deck.read_text()
marker = '\n\n---\n\n未经带通的时域波形：先看这里'
if marker in text:
    text = text.split(marker)[0]
deck.write_text(text.rstrip()+addition+'\n')
(OUT/'README_CN.md').write_text('请打开 [图解 Markdown](FIGURES_CN.md)。已补充未经带通的时域波形与粒子穿面 shower profile；图中为英文标注。\n\n旧输出只能覆盖 0–128 MHz；后续 1 GHz 配置与空气任务对齐。\n')
import shutil
shutil.copy2(__file__, OUT/'bandwidth_profile_psr.py')
def manifest(directory):
    lines = []
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p != directory/'SHA256SUMS':
            h = hashlib.sha256()
            with p.open('rb') as handle:
                for b in iter(lambda: handle.read(1024*1024), b''):
                    h.update(b)
            lines.append(h.hexdigest()+'  '+str(p.relative_to(directory)))
    (directory/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
manifest(OUT)
manifest(REPORT)
print(json.dumps(metrics), flush=True)
