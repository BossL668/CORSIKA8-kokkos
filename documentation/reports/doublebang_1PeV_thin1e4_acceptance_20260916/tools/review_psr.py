#!/usr/bin/env python3
"""Review the completed pilot on PSR; retain the full-scan audit and raw data."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket

assert socket.gethostname() == 'psrpku2025', 'Run numerical analysis only on PSR'
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_1PeV_thin1e4_queue4M_20260915')
TAG = 'openmp_SiO2_21CMA80_1PeV_seed22309'
RUN = ROOT / 'runs' / TAG
REPORT = ROOT / 'report' / TAG
OUT = REPORT / 'review_20260916'
OUT.mkdir(exist_ok=True)
(OUT / 'evidence').mkdir(exist_ok=True)

def read_json(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

s = yaml.safe_load((RUN / 'output/terrain_run.yaml').read_text())
a = read_json(REPORT / 'acceptance.json')
t = read_json(RUN / 'timing.json')
status = read_json(RUN / 'status.json')
classification = read_json(RUN / 'doublebang_classification.json')
archive = read_json(RUN / 'radio_archive.json')
assert status['complete'] and status['returncode'] == 0
assert a['passed'] and all(a['gates'].values())
assert a['records']['all_tracks'] == s['diagnostics']['steps']
assert a['records']['all_deposits'] == s['diagnostics']['deposition_rows']
assert s['accelerator']['pending_particles'] == 0
assert classification == a['classification']
assert all(p['roundtrip_verified'] and Path(p['archive']).stat().st_size == p['gzip_bytes'] for p in archive)
assert all(s['radio_result'][alg]['complete'] for alg in ['CoREAS', 'ZHS'])

with np.load(REPORT / 'profile.npz') as p:
    edges, deposited = p['edges_m'], p['deposited_GeV']
centres = (edges[:-1] + edges[1:]) / 2
profile_residual = float(deposited.sum() - a['records']['deposited_GeV'])
assert abs(profile_residual) < 1e-8 * s['energy_GeV']
station_rows = a['radio']['stations']
strongest = max(station_rows, key=lambda r: r['peak_50_100MHz_V_m'])
with np.load(ROOT / 'bundle/terrain_grid.npz') as grid:
    east, north, up = [grid[key][::3, ::3] for key in ['east_m', 'north_m', 'up_m']]
origin, direction = np.array(s['position_m']), np.array(s['direction'])
axis_grid = (east-origin[0])*direction[0] + (north-origin[1])*direction[1] + (up-origin[2])*direction[2]
vertex = np.array(s['neutrino']['interactions'][0]['position_enu_m'])
vertex_distance = float((vertex-origin) @ direction)
station_xyz = np.array([o['position_enu_m'] for o in s['scene']['radio']['observers']])

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
im = axes[0].contourf(east/1000, north/1000, up, levels=35, cmap='terrain')
fig.colorbar(im, ax=axes[0], label='Terrain ENU Up [m]')
axes[0].scatter(station_xyz[:, 0]/1000, station_xyz[:, 1]/1000, s=8, c='purple', label='All 80 stations')
axes[0].plot([origin[0]/1000, vertex[0]/1000], [origin[1]/1000, vertex[1]/1000], 'r-', lw=1)
axes[0].scatter(origin[0]/1000, origin[1]/1000, c='red', marker='*', s=65, label='Injection')
axes[0].scatter(vertex[0]/1000, vertex[1]/1000, c='black', marker='x', s=65, label='NC interaction')
axes[0].set(xlabel='East [km]', ylabel='North [km]', title='Geometry: one NC vertex', aspect='equal')
axes[0].legend(fontsize=8)
density = deposited / np.diff(edges)
axes[1].plot(centres/1000, density, lw=.8)
axes[1].set(xlabel='Distance along injection axis [km]', ylabel='Deposited energy [GeV/m]',
            title='Full DEM projection; 1 m bins', yscale='log',
            xlim=(float(axis_grid.min()/1000), float(axis_grid.max()/1000)), ylim=(1e-8, float(density.max()*4)))
axes[2].stairs(density, edges, lw=1.2)
axes[2].axvline(vertex_distance, color='black', ls='--', label='NC vertex')
axes[2].set(xlabel='Distance along injection axis [m]', ylabel='Deposited energy [GeV/m]',
            title='Shower maximum: deposited energy', xlim=(vertex_distance-2, vertex_distance+14))
axes[2].legend(fontsize=8)
for ax in axes[1:]:
    ax.grid(alpha=.2)
fig.suptitle('1 PeV | SiO2 | seed 22309 | Complete NC shower, not double bang')
fig.savefig(OUT / 'geometry_profile_review.png', dpi=155)
plt.close(fig)

review = dict(
    reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Review of the existing full-row acceptance, summary, archive manifest, and complete profile; no new shower run.',
    event_integrity_passed=True,
    doublebang_passed=classification['qualified_doublebang'],
    interaction_currents=[v['current'] for v in s['neutrino']['interactions']],
    tau_tracks=read_json(RUN/'tau_tracks.json'),
    timing=t,
    records=a['records'],
    energy_ledger_relative_unexplained=s['energy_ledger']['unexplained_over_initial'],
    profile_sum_minus_deposit_csv_GeV=profile_residual,
    profile_beyond_15km_absolute_GeV=float(abs(deposited[abs(centres)>15000]).sum()),
    profile_peak_bin_centre_m=float(centres[deposited.argmax()]),
    profile_peak_bin_GeV=float(deposited.max()),
    peak_is_deposition_not_particle_number_Xmax=True,
    resident_peak_particles=s['accelerator']['peak_resident_particles'],
    resident_capacity=s['accelerator']['resident_capacity'],
    pending_particles=s['accelerator']['pending_particles'],
    radio_relative_l2_by_source=a['radio']['relative_l2_by_source'],
    zero_signal_stations=[r['station'] for r in station_rows if r['peak_50_100MHz_V_m']==0],
    nonzero_signal_stations=sum(r['peak_50_100MHz_V_m']>0 for r in station_rows),
    nonzero_rock_signal_stations=sum(r['rock_Q']>0 for r in station_rows),
    transmitted_paths=s['radio_result']['ZHS']['transmitted_paths'],
    rock_transmission_validated_by_this_event=False,
    strongest_station=strongest,
    archive_existing_and_recorded_roundtrip_verified=True,
    archive_compressed_bytes=sum(p['gzip_bytes'] for p in archive),
    archive_original_bytes=sum(p['bytes'] for p in archive),
    limitations=[
        'A complete NC shower is not a double-bang success.',
        'No accepted transmitted ray and zero rock-source receiver field: this event does not validate transmission; no specific rejection cause established in this review.',
        'Algorithm agreement is conditional on shared sources/geometry and does not establish physical accuracy.',
        'The energy ledger includes recorded generator balance, cut rest energies, and stochastic thinning changes; it is not a full microscopic energy-conservation proof.',
        '33 secondary neutrinos below the weak-model domain carry 87.03191499230233 GeV and are transported to escape without CC/NC interactions.',
        '256 MHz sampling gives a 128 MHz Nyquist limit; plotted waveforms use 50-100 MHz, not 50-300 MHz.',
        'Full profile is retained unchanged. Axis/scale adjustment avoids interpreting empty-bin floating-point tails as physical deposition far beyond DEM coverage.',
    ],
)
save(OUT/'review.json', review)
for source in [RUN/'status.json', RUN/'timing.json', RUN/'doublebang_classification.json',
               RUN/'radio_archive.json', RUN/'tau_tracks.json', RUN/'audit.log',
               RUN/'output/terrain_run.yaml', REPORT/'acceptance.json', Path(__file__)]:
    shutil.copy2(source, OUT/'evidence'/source.name)

readme = '''# 1 PeV、SiO₂、1e-4 薄化：完成事件验收

**计算完整性与记录一致性通过；double bang 未通过。** seed 22309 实际发生一次 NC（中性流）相互作用，没有 τ 轨迹或 τ 衰变，是一个完整的 NC shower。不能作为成功的 double bang 或接收站双脉冲案例。

北京时间 2026-09-16 10:55:15 模拟正常结束，退出码 0；11:07 完成验收、绘图和无损归档。本轮只运行这一例，未自动启动新种子。

| 核对项 | 结果 |
|---|---|
| 模拟进程总耗时 | 18 小时 10 分 28.8 秒 |
| 输运循环（含循环内射电） | 17 小时 58 分 27.1 秒 |
| 最后射电刷新、下载、重建与写出 | 10 分 39.8 秒 |
| 验收及绘图 / 无损归档 | 4 分 1.9 秒 / 7 分 46.4 秒 |
| 进程峰值内存 | 151.27 GiB；输运阶段约 74.4 GiB |
| 完整轨迹 / 沉积记录 | 17,636,988 / 16,784,148 条；未触发行数截断 |
| 终态边界记录 | 34 条，逐一与对应粒子的最终轨迹核对；时间窗存活粒子为 0 |
| 常驻队列 | 峰值 39,589 / 容量 4,194,304；最终待处理粒子为 0 |
| OpenMP / 射电累积结果下载 | 256 核 / 最后一次下载；不代表输运记录也只下载一次 |
| 80 站输出 | 每种算法 41,943,040 行，逐站逐样本检查；无非有限值、无超窗计数 |

域内加权沉积 421,587.812268 GeV，出界总能量 554,170.502889 GeV。加入生成器能量差、截止静质量和随机薄化跳变后，账本未解释残差为 −1.15086e-5 GeV，相对初始能量为 −1.15086e-11。这说明已记录账本闭合，不能把它表述成所有物理模型的精度或忽略其余账项直接相加等于 1 PeV。完整 profile 积分与沉积 CSV 相差约 8.95e-8 GeV。

![](geometry_profile_review.png)

左图是实际地形、80 站和唯一 NC 顶点；中图用对数纵轴查看 DEM 投影范围内的沉积；右图放大主要 shower。沉积峰的 1 m 分箱中心为注入点前方 1420.5 m，NC 顶点位于 1418.971 m。这个峰是沉积能量峰，不是粒子数定义的 Xmax，也不是第二个 bang。所有分箱仍完整保存在 [profile.npz](../profile.npz)。

原图横轴被极小的分箱累计舍入尾项延伸到约 150 km：15 km 之外的总量仅 8.19e-8 GeV。本图只调整显示范围和纵轴，没有修改 profile 数组、删除沉积或改变能量积分；中图纵轴下限 1e-8 GeV/m。

![](../figures/waveform_E18.png)

E18 是 50–100 MHz 最强站，矢量场峰值 2.14819e-12 V/m（约 2.15 pV/m）。上排 CoREAS/ZHS 重合，下排显示接收信号全部来自空气源。80 站中 77 站非零，S01/S02/S03 为零，但输出行完整。

全采样、全站总场的相对 L2 差为 ||E_CoREAS−E_ZHS||₂/||E_ZHS||₂ = 2.34547e-9；50–100 MHz 各非零站的最大相对差为 2.14745e-10。两算法共享几何和部分传播处理，因此一致性不等于独立物理验证。

**这例的山体源接收场全部为零，接受的透射光路数也为零。** 因而不能用它验收山体透射分支、山体/空气双脉冲或山体衰减模型；也不能把零值直接解释成山体中没有辐射或必然被吸收。具体光路拒绝原因尚未由本次复核确定。

当前输出采样率 256 MHz，奈奎斯特频率 128 MHz；本报告波形用 50–100 MHz，不能用于完整 50–300 MHz 分析。另有 33 条低于弱相互作用模型适用能区的次级中微子，合计约 87.03 GeV，按现有模型只输运至出界；不能把这次验收扩展为该低能弱物理已验证。

原始完整轨迹、沉积及射电数据在 PSR 保留，射电大文件已无损压缩并做解压 SHA-256 往返核对；没有删除这例完整事件数据。下一轮 double bang 候选应先核实 CC→τ→产生 shower 的衰变链，再投入完整 80 站计算；本次没有启动新任务。

[复核数值](review.json) · [原全量验收](../acceptance.json) · [80 站数值](../station_signals.csv) · [全部波形索引](../README_CN.md) · [运行证据](evidence/status.json)
'''
(OUT/'README_CN.md').write_text(readme)
parent = REPORT/'README_CN.md'
parent_text = parent.read_text()
if 'review_20260916/README_CN.md' not in parent_text:
    parent.write_text('**最终复核：这例是 NC shower，不是 double bang。** [验收结论、主峰放大图与适用范围](review_20260916/README_CN.md)。\n\n'+parent_text)
top = ROOT/'README_CN.md'
top_text = top.read_text()
link = f'report/{TAG}/review_20260916/README_CN.md'
if link not in top_text:
    top_text = top_text.replace('完整验收 1 例', '计算完整性验收 1 例')
    top_text = '\n'.join(line for line in top_text.splitlines() if not line.startswith('已耗时 '))+'\n'
    top.write_text(f'**本轮已经结束：完整 NC shower 1 例，double bang 0 例。** [最终验收与诊断图]({link})。模拟进程耗时 18 小时 10 分 28.8 秒；未启动下一例。\n\n'+top_text)
paths = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
(OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in paths))
print(json.dumps({k:review[k] for k in ['event_integrity_passed','doublebang_passed','interaction_currents','nonzero_signal_stations','nonzero_rock_signal_stations','profile_sum_minus_deposit_csv_GeV']}, indent=2))
