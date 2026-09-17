#!/usr/bin/env python3
"""Completed-event waveforms, source decomposition and energy comparison on PSR."""
import argparse
import importlib.util
import json
import os
import pathlib
import socket

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import yaml


def savefig(fig, out, name):
    for extension in ['png', 'pdf']:
        fig.savefig(out / 'figures' / (name + '.' + extension), dpi=150)
    plt.close(fig)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-100))


def analyze(root, case, surface):
    out = root / 'report'
    cached = out / (case['tag'] + '.json')
    if cached.exists():
        return json.loads(cached.read_text())
    folder = root / 'runs' / case['tag']
    summary = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
    checks = json.loads((folder / 'checks.json').read_text())
    assert summary['complete'] and all(checks['checks'].values())
    assert json.loads((folder / 'material_checks.json').read_text())['passed']
    config = json.loads((folder / 'output/radio/ZHS/config.json').read_text())
    n, rate = config['samples'], config['sample_rate_Hz']
    time = (np.arange(n) / rate + config['start_time_s']) * 1e6
    frequencies = np.fft.rfftfreq(n, 1 / rate)
    band = (frequencies >= 50e6) & (frequencies <= 100e6)
    waves = {}
    closure = []
    for algorithm in ['CoREAS', 'ZHS']:
        columns = ['E' + axis + '_' + suffix for suffix in ['V_m', 'inside', 'outside'] for axis in 'xyz']
        frame = pd.read_csv(folder / 'output/radio' / algorithm / 'field.csv',
                            usecols=columns, float_precision='round_trip')
        raw = {name: frame[['E' + axis + '_' + suffix for axis in 'xyz']].to_numpy().reshape(3, n, 3)
               for name, suffix in [('total', 'V_m'), ('rock', 'inside'), ('air', 'outside')]}
        assert all(np.isfinite(v).all() for v in raw.values())
        closure.append(rel(raw['air'] + raw['rock'], raw['total']))
        assert closure[-1] < 1e-13
        waves[algorithm] = {name: np.fft.irfft(np.fft.rfft(v, axis=1) * band[None, :, None], n=n, axis=1)
                            for name, v in raw.items()}
        del raw, frame
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), constrained_layout=True)
    figc, axc = plt.subplots(2, 3, figsize=(15, 6.5), constrained_layout=True)
    for i, observer in enumerate(config['observers']):
        parts = {k: v[i] for k, v in waves['ZHS'].items()}
        norm = np.linalg.norm(parts['total'], axis=1)
        peak = int(np.argmax(norm))
        component = int(np.argmax(np.abs(parts['total'][peak])))
        zoom = slice(max(0, peak - 100), min(n, peak + 101))
        # Plot the unshifted reception window and a full-resolution peak zoom.
        for algorithm, style in [('ZHS', '-'), ('CoREAS', '--')]:
            data = waves[algorithm]['total'][i, :, component] * 1e6
            axes[0, i].plot(time, data, style, lw=.9, label=algorithm)
            axes[1, i].plot(time[zoom], data[zoom], style, lw=1.2, label=algorithm)
        for j in range(2):
            axes[j, i].set(xlabel='Arrival time (us)', ylabel='E' + 'xyz'[component] + ' (uV/m)',
                           title=observer['name'] + (' | full window' if j == 0 else ' | strongest pulse'))
            axes[j, i].legend(fontsize=8)
            axes[j, i].grid(alpha=.2)
        qa, qr, qt = [float(np.sum(parts[k]**2) / rate) for k in ['air', 'rock', 'total']]
        cross = float(2 * np.sum(parts['air'] * parts['rock']) / rate)
        assert abs(qt - qa - qr - cross) / max(qt, 1e-100) < 1e-12
        row = dict(observer=observer['name'], peak_V_m=float(norm[peak]), peak_time_us=float(time[peak]),
                   air_Q=qa, rock_Q=qr, total_Q=qt, interference_Q=cross,
                   rock_self_percent=100 * qr / (qa + qr) if qa + qr > 0 else None)
        for name in ['air', 'rock', 'total']:
            error = rel(waves['CoREAS'][name][i], parts[name])
            assert error < 1e-5
            row[name + '_algorithm_relative_l2'] = error
            magnitude = np.linalg.norm(parts[name], axis=1)
            index = int(np.argmax(magnitude))
            row[name + '_peak_V_m'] = float(magnitude[index])
            row[name + '_peak_time_us'] = float(time[index]) if magnitude[index] > 0 else None
        # Medium-tagged source fields are not labels for first/second bang.
        for name, color, style in [('air', '#1976b9', '-'), ('rock', '#e67e22', '-'), ('total', '#222222', '--')]:
            axc[0, i].plot(time, np.linalg.norm(parts[name], axis=1) * 1e6, style, color=color, lw=.9, label=name)
            axc[1, i].plot(time[zoom], parts[name][zoom, component] * 1e6, style, color=color, lw=1.1, label=name)
        for j in range(2):
            axc[j, i].set(xlabel='Arrival time (us)', ylabel='|E| (uV/m)' if j == 0 else 'E' + 'xyz'[component] + ' (uV/m)',
                          title=observer['name'] + (' | source components' if j == 0 else ' | total-peak zoom'))
            axc[j, i].legend(fontsize=8)
            axc[j, i].grid(alpha=.2)
        rows.append(row)
    title = 'Silica (SiO2) | %g PeV nu_tau | seed %d | 50-100 MHz' % (case['energy_GeV'] / 1e6, case['seed'])
    fig.suptitle(title + ' | paired algorithms')
    figc.suptitle(title + ' | emission medium')
    savefig(fig, out, case['tag'] + '_waveforms')
    savefig(figc, out, case['tag'] + '_air_rock')
    del waves
    origin = np.asarray(summary['position_m'])
    direction = np.asarray(summary['direction'])
    direction /= np.linalg.norm(direction)
    # 500 us corresponds to at most 149.9 km from injection at c. Retain the
    # complete domain in the deposit histogram, including backward secondaries.
    edges = np.arange(-151000., 151000. + 25, 25.)
    histogram = np.zeros(len(edges) - 1)
    total = 0.
    for chunk in pd.read_csv(folder / 'output/terrain/deposits.csv.gz', chunksize=250000):
        midpoint = .5 * (chunk[['x0_m', 'y0_m', 'z0_m']].to_numpy() + chunk[['x1_m', 'y1_m', 'z1_m']].to_numpy())
        distance = (midpoint - origin) @ direction
        weights = chunk['weighted_deposited_GeV'].to_numpy()
        assert np.isfinite(distance).all() and np.isfinite(weights).all()
        histogram += np.histogram(distance, bins=edges, weights=weights)[0]
        total += float(weights.sum())
    assert abs(histogram.sum() - total) < 1e-8 * max(total, 1.)
    np.savez_compressed(out / (case['tag'] + '_profile.npz'), deposit_edges_m=edges, deposited_GeV=histogram)
    decays = []
    for decay in summary['tau'].get('decays', []):
        pdgs = [abs(d['pdg']) for d in decay['daughters']]
        channel = 'muonic' if 13 in pdgs else 'electronic' if 11 in pdgs else 'hadronic'
        decays.append(dict(channel=channel, medium=decay['medium'], energy_PeV=decay['energy_GeV'] / 1e6,
                           time_us=decay['time_ns'] / 1000,
                           axis_distance_km=float((np.asarray(decay['position_enu_m']) - origin) @ direction / 1000)))
    cc = [v for v in summary['neutrino']['interactions'] if v['current'] == 'CC']
    cc_dist = [float((np.asarray(v['position_enu_m']) - origin) @ direction / 1000) for v in cc]
    scene = summary['scene']
    vertices, height_at = surface(pathlib.Path(scene['geometry']['mesh_path']))
    local_end = (origin[1] - vertices[:, 1].min()) / 1000
    ds = np.linspace(-.2, local_end, 2200)
    h = height_at(np.full_like(ds, origin[0]), origin[1] - ds * 1000) / 1000
    tau = json.loads((folder / 'tau_tracks.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), constrained_layout=True)
    axes[0].fill_between(ds, vertices[:, 2].min() / 1000 - .1, h, color='#c4ac84')
    axes[0].plot(ds, h, color='#665541', lw=.8, label='DEM surface')
    if tau:
        segments, energies, colors = [], [], []
        for t in tau:
            x = [(origin[1] - float(t['y0_m'])) / 1000, (origin[1] - float(t['y1_m'])) / 1000]
            segments.append([[x[0], float(t['z0_m']) / 1000], [x[1], float(t['z1_m']) / 1000]])
            energies.append([[x[0], float(t['E0_GeV']) / 1e6], [x[1], float(t['E1_GeV']) / 1e6]])
            colors.append('#806443' if t['medium'] == 'rock' else '#0085bc')
        axes[0].add_collection(LineCollection(segments, colors='#154d96', linewidths=1))
        axes[1].add_collection(LineCollection(energies, colors=colors, linewidths=1))
        axes[1].set_ylim(0, max(float(t['E0_GeV']) / 1e6 for t in tau) * 1.1)
    for v, dist in zip(cc, cc_dist):
        axes[0].scatter(dist, v['position_enu_m'][2] / 1000, marker='*', color='#be202a', s=65)
    for d, original_decay in zip(decays, summary['tau'].get('decays', [])):
        axes[0].scatter(d['axis_distance_km'], original_decay['position_enu_m'][2] / 1000, marker='D', color='#8b3792', s=30)
    axes[0].set(xlim=(-.2, local_end), ylim=(float(np.ma.min(h)) - .15, max(float(np.ma.max(h)), origin[2] / 1000) + .2),
                ylabel='ENU height (km)', xlabel='Southward distance from injection (km)',
                title='Local DEM section | blue: transported tau; star: CC; diamond: tau decay')
    centers = .5 * (edges[1:] + edges[:-1]) / 1000
    occupied = centers[histogram != 0]
    extent = list(occupied) + cc_dist + [d['axis_distance_km'] for d in decays]
    if tau:
        extent.extend((origin[1] - float(t['y1_m'])) / 1000 for t in tau)
    left, right = (min(extent) - .2, max(extent) + .2) if extent else (-.2, 1.)
    axes[1].set(xlim=(left, right), xlabel='Distance from injection (km)', ylabel='Tau energy (PeV)',
                title='Full recorded tau extent | brown: rock; blue: air')
    axes[2].plot(centers, histogram / 1e6 / .025, lw=1, color='#246991')
    for j in [1, 2]:
        axes[j].axvline(local_end, color='#777777', ls=':', lw=1, label='South edge of available DEM')
        for dist in cc_dist:
            axes[j].axvline(dist, color='#be202a', ls=':', lw=.8)
        for d in decays:
            axes[j].axvline(d['axis_distance_km'], color='#8b3792', ls='--', lw=.8)
    axes[2].set(xlim=(left, right), xlabel='Distance along primary axis (km)', ylabel='Deposited energy (PeV/km)',
                title='Complete shower deposition | 25 m bins | stars/diamonds are particle vertices, not radio rays')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(title.replace(' | 50-100 MHz', '') + ' | actual recorded geometry')
    savefig(fig, out, case['tag'] + '_geometry_profile')
    result = dict(**{k: case[k] for k in ['tag', 'seed', 'energy_GeV', 'label', 'baseline_tag']},
                  peaks=rows, decays=decays, CC_distances_km=cc_dist, deposited_GeV=total,
                  decays_beyond_south_DEM_edge=sum(d['axis_distance_km'] > local_end for d in decays),
                  radio_addition_relative_l2=max(closure), checks=checks['checks'],
                  coreas_zhs_relative_l2=checks['coreas_zhs_relative_l2'],
                  diagnostics=summary['diagnostics'], energy_ledger=summary['energy_ledger'],
                  accelerator=summary['accelerator'], radio_result=summary['radio_result'],
                  doublebang=json.loads((folder / 'doublebang_classification.json').read_text()),
                  tau_summary=summary['tau'], status=json.loads((folder / 'status.json').read_text()))
    cached.write_text(json.dumps(result, indent=2) + '\n')
    print('ANALYZED', case['tag'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=pathlib.Path)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('Numerical analysis only on PSR')
    root, out = args.root, args.root / 'report'
    (out / 'figures').mkdir(exist_ok=True)
    manifest = json.loads((root / 'campaign.json').read_text())
    reference = pathlib.Path(manifest['baseline_root'])
    spec = importlib.util.spec_from_file_location('geometry_helpers', root / 'bundle/geometry_helpers.py')
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    rows = [analyze(root, case, geometry.surface) for case in manifest['cases']
            if (root / 'runs' / case['tag'] / 'material_checks.json').exists()]
    baseline = [r for r in json.loads((root / 'bundle/baseline_results.json').read_text()) if r['material'] == 'silica_SiO2']
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for ax, seed in zip(axes, [158, 946, 3605]):
        old = next(r for r in baseline if r['seed'] == seed)
        selected = [dict(old, energy_GeV=1e8)] + [r for r in rows if r['seed'] == seed]
        selected.sort(key=lambda r: r['energy_GeV'])
        for i, station in enumerate(['E01', 'E20', 'N10']):
            ax.plot([r['energy_GeV'] / 1e6 for r in selected],
                    [r['peaks'][i]['peak_V_m'] * 1e6 for r in selected], 'o-', label=station)
        ax.set(xscale='log', yscale='log', xlabel='Primary nu_tau energy (PeV)', ylabel='Peak vector E (uV/m)',
               title='SiO2 | seed %d' % seed)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('50-100 MHz | absolute field at original stations | 100 PeV: 100 us; new runs: 500 us transport')
    savefig(fig, out, 'energy_vs_received_peak')
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, station in enumerate(['E01', 'E20', 'N10']):
        for qualified, marker in [(True, 'o'), (False, 'x')]:
            selected = [r for r in rows if r['doublebang']['qualified_doublebang'] == qualified and r['peaks'][i]['peak_V_m'] > 0]
            if selected:
                ax.scatter([r['energy_GeV'] / 1e6 * (1 + .035 * (i - 1)) for r in selected],
                           [r['peaks'][i]['peak_V_m'] * 1e6 for r in selected], marker=marker,
                           label=station + (' | qualified topology' if qualified else ' | other outcome'))
    ax.set(xscale='log', yscale='log', xlabel='Primary energy (PeV; small station offsets for display)',
           ylabel='Peak vector E (uV/m)', title='All completed seeds | SiO2 | air-default cuts | emthin = 1e-5')
    ax.grid(alpha=.2)
    if ax.collections:
        ax.legend(fontsize=8)
    savefig(fig, out, 'all_seed_radio_peaks')
    for seed in sorted({r['seed'] for r in rows}):
        selected = [r for r in rows if r['seed'] == seed]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        old = next((r for r in baseline if r['seed'] == seed), None)
        paths = [(100., reference / 'report' / (old['tag'] + '_arrays.npz'))] if old else []
        paths += [(r['energy_GeV'] / 1e6, out / (r['tag'] + '_profile.npz')) for r in selected]
        occupied = []
        for energy, path in sorted(paths):
            with np.load(path) as data:
                x = (data['deposit_edges_m'][:-1] + data['deposit_edges_m'][1:]) / 2000
                y = data['deposited_GeV'] / 1e6 / .025
                occupied.extend(x[y != 0].tolist())
                ax.plot(x, y, lw=1, label='%g PeV' % energy)
        if occupied:
            ax.set_xlim(min(occupied) - .2, max(occupied) + .2)
        ax.set(xlabel='Distance along primary axis (km)', ylabel='Deposited energy (PeV/km)',
               title='Silica (SiO2) | seed %d | 25 m bins | 100 PeV has shorter transport window' % seed)
        ax.legend()
        ax.grid(alpha=.2)
        savefig(fig, out, 'seed%d_energy_profiles' % seed)
    table = []
    for r in rows:
        old = next((b for b in baseline if b['seed'] == r['seed']), None)
        for i, station in enumerate(r['peaks']):
            table.append(dict(tag=r['tag'], seed=r['seed'], energy_PeV=r['energy_GeV'] / 1e6,
                              **station, qualified_doublebang=r['doublebang']['qualified_doublebang'],
                              baseline_100PeV_peak_V_m=old['peaks'][i]['peak_V_m'] if old else None,
                              peak_ratio_to_100PeV=station['peak_V_m'] / old['peaks'][i]['peak_V_m'] if old else None))
    pd.DataFrame(table).to_csv(out / 'station_comparison.csv', index=False, float_format='%.17g')
    counts = {energy: sum(r['energy_GeV'] == energy and r['doublebang']['qualified_doublebang'] for r in rows) for energy in [300000000,1000000000]}
    md = ['# 更高能量的 SiO₂ double bang 试跑', '', '**完成并通过计算检查：%d 个事件。符合 double bang 谱系筛选：300 PeV %d 个、1 EeV %d 个；目标每档至少 3 个。**' % (len(rows),counts[300000000],counts[1000000000]), '',
          '每档预先准备最多 30 个不同种子，要求至少 3 个新种子通过 double bang 条件；原种子 158、946、3605 只作为对照，即使物理拓扑通过也不占用新种子名额。预筛仅检查近似相互作用柱深；是否发生 double bang 必须看完整 shower 的真实顶点、τ 谱系和次级轨迹。保留所有完成事件及失败筛选原因，不从这组条件样本推断天然发生率。', '',
          '原 DEM、入射位置、方向和 E01/E20/N10 接收站。OpenMP 256 个物理核、常驻队列、CoREAS/ZHS 同时计算。电磁粒子截断 0.5 MeV，强子/μ/τ 截断 0.3 GeV；emthin=1e-5，最大权重按大气规则取 0.5×emthin×E/GeV，分别为 1500 和 5000。大气代码本身默认 emthin=1e-6，这里采用用户指定的 1e-5。', '',
          '输运窗 500 μs；射电接收窗 2048 μs，采样率 256 MHz。旧 100 PeV 参考使用 EM 截断 100 MeV、强子截断 10 GeV、emthin=0.01、100 μs 输运窗。因此新旧幅度差包含数值设置的改进，不能当成只改变能量的对照。', '',
          'double bang 筛选要求：自然岩内 ντ CC，通过实际 τ 父子谱系连接到电子或强子衰变；两顶点各注入至少 1 PeV 的 shower 次级，直接女儿确实有输运记录；顶点距离至少 100 m，衰变后至少留 20 μs 输运。这些是本次选图条件，并非探测器可分辨双脉冲的判据，也不保证两条沉积轮廓完全不重叠。', '',
          '按 PDG 的 τ 质量与寿命，忽略能损时，平均衰变长度约为 49 km × Eτ/EeV；这是 τ 的能量，不是入射 ντ 的能量。因此更高能量不保证原站点收到更强或可分辨的双脉冲。[PDG 参数](https://pdg.lbl.gov/2025/download/db2024.pdf)', '',
          '## 所有新种子的信号', '', '![](figures/all_seed_radio_peaks.png)', '',
          '圆点为计入新种子目标的 double bang；叉号为未计入目标的结果（包括原种子对照，物理拓扑是否通过另列）。零信号保留在数值表中。横向少量错开仅为区分站点。这里每档使用相同的新截断和 thinning。', '',
          '## 与旧图的参考比较', '', '![](figures/energy_vs_received_peak.png)', '',
          '每个面板一个种子，每条线一个原接收站。横轴为入射能量，纵轴为 50–100 MHz 三维电场模的峰值；两轴为对数。读绝对幅度及相对 100 PeV 的倍率，不能把连线当成统计上的能量标度。尚未加入天线响应、噪声或触发。', '',
          '| 能量/PeV | seed | 站点 | 峰值/μV m⁻¹ | 相对旧100 PeV | 山体独立Q占比 |', '|---:|---:|---|---:|---:|---:|']
    for t in table:
        share = '无信号' if t['rock_self_percent'] is None else '%.4g%%' % t['rock_self_percent']
        ratio = '—' if t['peak_ratio_to_100PeV'] is None else '%.4g' % t['peak_ratio_to_100PeV']
        md.append('| %g | %d | %s | %.5g | %s | %s |' % (t['energy_PeV'], t['seed'], t['observer'], t['peak_V_m'] * 1e6, ratio, share))
    md += ['', '山体占比为 Qrock/(Qair+Qrock)，其中 Q=∫|E|²dt；干涉项另存在 CSV 中，不是总辐射能量占比，也不是第一/第二 bang 的占比。', '']
    for r in rows:
        name = '%g PeV，seed %d' % (r['energy_GeV'] / 1e6, r['seed'])
        modes = '; '.join('%s / %s，%.3f km，%.3f μs' % (d['channel'], d['medium'], d['axis_distance_km'], d['time_us']) for d in r['decays']) or '记录窗内没有 τ 衰变，不能当作完整 double bang'
        qualification = '通过 double bang 谱系筛选' if r['doublebang']['qualified_doublebang'] else '未通过 double bang 筛选（仍保留此事件）'
        if r['doublebang']['topology_passed'] and not r['doublebang']['eligible_for_new_seed_target']:
            qualification = '原种子对照：物理 double bang 拓扑通过，但不计入新种子目标'
        md += ['## ' + name, '', qualification + '。实际 τ 衰变：' + modes + '。', '',
               '![](figures/' + r['tag'] + '_geometry_profile.png)', '',
               '上图是原 DEM 的局部剖面；下两图展示完整 τ 轨迹能量和 shower 沉积范围，横轴范围可能更大。红星/红点线为 CC，紫色菱形/虚线为 τ 衰变。超过原 DEM 边缘的区段按现有环境中的大气处理，那里没有补造真实山体。无衰变或超出局部几何的事件照实保留。', '',
               '![](figures/' + r['tag'] + '_waveforms.png)', '',
               '上排完整接收窗，下排最强脉冲放大；实线 ZHS、虚线 CoREAS。纵轴是实际电场，未归一化或平移时间。各站纵轴独立。', '',
               '![](figures/' + r['tag'] + '_air_rock.png)', '',
               '蓝色为空气源、橙色为山体源、黑色为相干总场；上排是电场模，下排画相同笛卡尔分量。来源按辐射轨迹所在介质分类，不等同于两个 bang。', '']
    for seed in sorted({r['seed'] for r in rows}):
        if any(r['seed'] == seed for r in rows):
            md += ['## seed %d：能量沉积对比' % seed, '', '![](figures/seed%d_energy_profiles.png)' % seed, '',
                   '每条线一个入射能量，纵轴为每公里加权沉积，25 m 分箱；这是沉积曲线，不是粒子数曲线。先看峰的位置，再看幅度和宽度。', '']
    md += ['## 检查范围', '',
           '运行前把 SiO₂ 原生/可移植 LPM 抑制因子对照扩展到 1 EeV，覆盖轫致辐射和成对产生；程序启动时仍执行原来的表格适用域检查。每个完成事件检查实际 256 核绑定、队列清空、全部带电轨迹、无 CSV 截断、能量账本、自然衰变、CoREAS/ZHS 一致和接收窗外丢弃计数。', '',
           '轨迹和沉积 CSV 在写出时无损 gzip 压缩；已用 10 GeV SiO₂ 电子 shower 对照检查解压后的全部 CSV 和射电 moment 文件与原程序一致。压缩不删轨迹、不改变 thinning。', '',
           '这些检查说明实现与数值一致，并非完整物理独立验证。保留现有 CTW/DIS、TAUOLA、LPM、thinning，以及光路最多一次透射、无反射/绕射等模型限制；所筛选事件不用于估计探测率。', '',
           '[逐站数值](station_comparison.csv) · [完整检查与事件摘要](results.json)', '', '服务器：`' + str(root) + '`。', '']
    (out / 'README_CN.md').write_text('\n'.join(md))
    print('REPORT', len(rows), '/', len(manifest['cases']), flush=True)


if __name__ == '__main__':
    main()
