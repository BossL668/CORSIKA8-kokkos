#!/usr/bin/env python3
"""Incremental figures from completed PSR events; never plot partial outputs."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys

# Analysis does not use the shower's 256-thread OpenMP team. Prevent pandas'
# optional numexpr dependency from inheriting an unsupported 256-thread count.
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def make_case(root, case):
    folder = root / 'runs' / case['tag']
    out = root / 'report'
    cache = out / (case['tag'] + '.json')
    if cache.exists():
        return json.loads(cache.read_text())
    s = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
    checked = json.loads((folder / 'checks.json').read_text())
    assert all(checked['checks'].values())
    assert json.loads((folder / 'material_checks.json').read_text())['passed']
    config = json.loads((folder / 'output/radio/CoREAS/config.json').read_text())
    n = config['samples']
    rate = config['sample_rate_Hz']
    time_us = (np.arange(n) / rate + config['start_time_s']) * 1e6
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    band = (frequencies >= 50e6) & (frequencies <= 100e6)
    waves = {}
    for alg in ['CoREAS', 'ZHS']:
        frame = pd.read_csv(folder / 'output/radio' / alg / 'field.csv',
                            usecols=['Ex_V_m', 'Ey_V_m', 'Ez_V_m'])
        values = frame[['Ex_V_m', 'Ey_V_m', 'Ez_V_m']].to_numpy().reshape(-1, n, 3).transpose(0, 2, 1)
        waves[alg] = np.fft.irfft(np.fft.rfft(values, axis=-1) * band, n=n, axis=-1)
    fig, axes = plt.subplots(2, len(config['observers']), figsize=(15, 6.5),
                             squeeze=False, constrained_layout=True)
    peaks = []
    for i, observer in enumerate(config['observers']):
        norm = np.linalg.norm(waves['ZHS'][i], axis=0)
        peak_index = int(np.argmax(norm))
        component = int(np.argmax(np.abs(waves['ZHS'][i, :, peak_index])))
        peaks.append(dict(observer=observer['name'], peak_V_m=float(norm[peak_index]),
                          peak_time_us=float(time_us[peak_index]), component=['Ex', 'Ey', 'Ez'][component]))
        zoom = slice(max(0, peak_index - 100), min(n, peak_index + 101))
        for alg, style in [('ZHS', '-'), ('CoREAS', '--')]:
            axes[0, i].plot(time_us, waves[alg][i, component] * 1e6, style, lw=1, label=alg)
            axes[1, i].plot(time_us[zoom], waves[alg][i, component, zoom] * 1e6, style, lw=1.3, label=alg)
        for row in range(2):
            axes[row, i].set(xlabel='Arrival time (us)', ylabel=['Ex', 'Ey', 'Ez'][component] + ' (uV/m)',
                title=observer['name'] + (': full reception window' if row == 0 else ': strongest pulse'))
            axes[row, i].grid(alpha=.2)
            axes[row, i].legend()
    fig.suptitle('%s | seed %d | %g PeV %s | 50-100 MHz ideal bandpass' %
                 (case['label'], case['seed'], s['energy_GeV'] / 1e6, s['primary']))
    fig.savefig(out / 'figures' / (case['tag'] + '_waveforms.png'), dpi=160)
    plt.close(fig)
    # Project weighted deposits along the original primary axis. This is an
    # energy-deposition profile, not a particle-count longitudinal profile.
    edges = np.arange(-31000., 31000. + 25., 25.)
    histogram = np.zeros(len(edges) - 1)
    origin = np.asarray(s['position_m'], dtype=float)
    direction = np.asarray(s['direction'], dtype=float)
    direction /= np.linalg.norm(direction)
    total = 0.
    for chunk in pd.read_csv(folder / 'output/terrain/deposits.csv', chunksize=250000):
        midpoint = .5 * (chunk[['x0_m', 'y0_m', 'z0_m']].to_numpy() + chunk[['x1_m', 'y1_m', 'z1_m']].to_numpy())
        distance = (midpoint - origin) @ direction
        weights = chunk['weighted_deposited_GeV'].to_numpy()
        assert np.isfinite(distance).all() and np.isfinite(weights).all()
        histogram += np.histogram(distance, bins=edges, weights=weights)[0]
        total += float(np.sum(weights))
    assert abs(np.sum(histogram) - total) <= 1e-8 * max(abs(total), 1.)
    decays = []
    for decay in s['tau'].get('decays', []):
        pdgs = [abs(d['pdg']) for d in decay['daughters']]
        category = 'muonic' if 13 in pdgs else 'electronic' if 11 in pdgs else 'hadronic'
        decays.append(dict(channel=category, medium=decay['medium'], energy_PeV=decay['energy_GeV'] / 1e6,
            time_us=decay['time_ns'] / 1000,
            axis_distance_km=float((np.asarray(decay['position_enu_m']) - origin) @ direction / 1000)))
    np.savez_compressed(out / (case['tag'] + '_arrays.npz'), time_us=time_us,
        CoREAS=waves['CoREAS'], ZHS=waves['ZHS'], deposit_edges_m=edges, deposited_GeV=histogram)
    row = dict(tag=case['tag'], label=case['label'], material=case['material'], seed=case['seed'],
        peaks=peaks, decays=decays, coreas_zhs_relative_l2=checked['coreas_zhs_relative_l2'],
        checks=checked['checks'], material_model=s['resolved_material'], diagnostics=s['diagnostics'],
        accelerator=s['accelerator'], energy_ledger=s['energy_ledger'],
        neutrino_interaction_count=s['neutrino']['interaction_count'],
        deposited_GeV=total, status=json.loads((folder / 'status.json').read_text()))
    cache.write_text(json.dumps(row, indent=2) + '\n')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('PSR only: no local numerical analysis')
    root = args.root
    out = root / 'report'
    (out / 'figures').mkdir(exist_ok=True)
    manifest = json.loads((root / 'campaign.json').read_text())
    rows = []
    for case in manifest['cases']:
        if (root / 'runs' / case['tag'] / 'material_checks.json').exists():
            rows.append(make_case(root, case))
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    for seed in [158, 946, 3605]:
        selected = [r for r in rows if r['seed'] == seed]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        occupied = []
        for row in selected:
            with np.load(out / (row['tag'] + '_arrays.npz')) as data:
                edges = data['deposit_edges_m']
                centers = .5 * (edges[1:] + edges[:-1]) / 1000
                energy = data['deposited_GeV'] / 1e6 / (np.diff(edges) / 1000)
                ax.plot(centers, energy, lw=1.2, label=row['label'])
                occupied.extend(centers[energy != 0].tolist())
        if occupied:
            ax.set_xlim(min(occupied) - .2, max(occupied) + .2)
        ax.set(xlabel='Distance along primary axis from injection (km)',
               ylabel='Weighted energy deposition (PeV/km)',
               title='100 PeV nu_tau | seed %d | 25 m bins | %d/3 materials complete' % (seed, len(selected)))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(out / 'figures' / ('seed%d_shower.png' % seed), dpi=170)
        plt.close(fig)
    md = ['---', 'marp: true', 'paginate: true', 'title: 不同介质的 100 PeV ντ shower 与射电', '---', '',
          '# 相同入射条件，更换山体介质', '',
          '已完成并通过检查：**%d / 9**。仅展示完成的真实模拟。' % len(rows), '',
          '沿用此前 100 PeV ντ 的 seed 158、946、3605，入射位置、方向、粒子截断、thinning、100 μs 输运窗和 E01/E20/N10 接收站。每例 OpenMP 256 个物理核，常驻队列，同时计算 CoREAS 与 ZHS。', '',
          '材料为 SiO₂、CaCO₃、花岗岩混合物（H、C、O、Na、Mg、Al、Si、K、Ca、Fe）；密度、折射率和电场衰减长度按所给材料文件。SiO₂ 为 2650 kg/m³、n=√5、L_E=100 m。LPM 开启。', '',
          '相同种子不固定 τ 衰变通道或位置；每种材料每个种子仅一个事件，不能据此推断统计上的材料优劣。保留原模型的有限输运窗、截断和 thinning，未进行这些参数的收敛扫描。', '',
          '射电常数是所选材料模型值，不是当地实测值。ν 核反应仍采用原自由核子近似，未包含核遮蔽；低能次级 ν 的适用域限制与原算例相同。', '']
    for seed in [158, 946, 3605]:
        selected = [r for r in rows if r['seed'] == seed]
        if not selected:
            continue
        md += ['', '---', '', '# seed %d：shower 能量沉积' % seed, '',
               '![](figures/seed%d_shower.png)' % seed, '',
               '横轴沿入射方向；纵轴为每公里的加权能量沉积，分箱 25 m。比较峰的位置、宽度和沉积能量；这不是粒子数曲线。', '',
               '| 材料 | 本次 τ 衰变 | CoREAS/ZHS 谱相对 L2 差异 |', '|---|---|---:|']
        for row in selected:
            classification = '; '.join(d['channel'] + ' / ' + d['medium'] for d in row['decays']) or 'No tau decay in window'
            md.append('| %s | %s | %.3g |' % (row['label'], classification, row['coreas_zhs_relative_l2']))
        for row in selected:
            md += ['', '---', '', '# %s，seed %d' % (row['label'], seed), '',
                   '![](figures/%s_waveforms.png)' % row['tag'], '',
                   '上排为完整接收窗，下排放大最强脉冲。实线 ZHS、虚线 CoREAS；两条线应重合。每站画最强脉冲处最强的笛卡尔分量，50–100 MHz 理想带通，未归一化或平移时间。', '',
                   '纵轴是电场而非天线电压；各面板刻度独立，比较幅度要读倍率。零信号或无第二峰如实保留；不能仅凭双顶点就宣称有可分辨双脉冲。']
    md += ['', '---', '', '# 结果位置与检查', '',
           '服务器：`%s`。' % root, '',
           '`runs/<完整材料名和种子>/output/terrain/` 保存 tracks.csv 和 deposits.csv；`output/radio/{CoREAS,ZHS}/` 保存原始时域电场与复数谱。', '',
           '每例保留输入、实际 256 核绑定、能量账本、介质及 LPM 参数、队列清空、完整轨迹来源和算法一致性检查。通过这些检查表示实现和数值一致，不代表全部物理模型已获独立实验验证。', '']
    (out / 'SLIDES_CN.md').write_text('\n'.join(md))
    print('Analyzed %d / 9 completed material events' % len(rows), flush=True)
    # The final detached event receives the same independent acceptance audit
    # as the completed cases inspected interactively on 2026-09-14.
    acceptance_script = root / 'code/accept_psr.py'
    if len(rows) == len(manifest['cases']) and acceptance_script.exists():
        subprocess.run([sys.executable, str(acceptance_script), '--root', str(root)], check=True)


if __name__ == '__main__':
    main()
