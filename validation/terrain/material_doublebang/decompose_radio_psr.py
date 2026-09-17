#!/usr/bin/env python3
"""PSR-only air/rock source decomposition of the already simulated fields."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import socket

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import yaml

EXAMPLES = [
    ('openmp_silica_SiO2_seed158', 'E20', 'Separated main pulses'),
    ('openmp_silica_SiO2_seed3605', 'E20', 'Overlapping main pulses'),
    ('openmp_granite_H_C_O_Na_Mg_Al_Si_K_Ca_Fe_seed158', 'N10', 'Separated, very weak fields'),
    ('openmp_granite_H_C_O_Na_Mg_Al_Si_K_Ca_Fe_seed3605', 'E20', 'A short rock pulse after a larger air signal'),
]


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-100))


def peak(field, times):
    amplitude = np.linalg.norm(field, axis=1)
    if not np.any(amplitude):
        return dict(peak_nV_m=0., peak_time_us=None, energy_q05_us=None, energy_q95_us=None)
    idx = int(np.argmax(amplitude))
    cumulative = np.cumsum(amplitude**2)
    cumulative /= cumulative[-1]
    return dict(peak_nV_m=float(amplitude[idx] * 1e9), peak_time_us=float(times[idx]),
                energy_q05_us=float(times[np.searchsorted(cumulative, .05)]),
                energy_q95_us=float(times[np.searchsorted(cumulative, .95)]))


def example_plot(out, row, title, times, parts):
    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    top = fig.add_subplot(grid[0, :])
    bottom = [fig.add_subplot(grid[1, i]) for i in range(2)]
    names = [('air', 'Air sources', '#1976b9', '-'),
             ('rock', 'Rock sources', '#e67e22', '-'),
             ('total', 'Coherent total', '#222222', '--')]
    energy_density = sum(np.sum(parts[name]**2, axis=1) for name in ['air', 'rock'])
    cumulative = np.cumsum(energy_density) / np.sum(energy_density)
    lo = times[np.searchsorted(cumulative, .0005)] - 1.
    hi = times[np.searchsorted(cumulative, .9995)] + 1.
    for name, label, color, style in names:
        top.plot(times, np.linalg.norm(parts[name], axis=1) * 1e9, style, color=color, lw=1., label=label)
    top.set(xlim=(max(times[0], lo), min(times[-1], hi)), xlabel='Arrival time (us)',
            ylabel='Vector field magnitude (nV/m)', title=title + ' | overview')
    top.legend(loc='upper center', fontsize=9)
    top.grid(alpha=.2)
    component = int(np.argmax(sum(np.sum(parts[name]**2, axis=0) for name in ['air', 'rock'])))
    for ax, around in zip(bottom, ['air', 'rock']):
        center = row[around + '_peak_time_us']
        if center is None:
            ax.text(.5, .5, 'No source component at this observer', ha='center', transform=ax.transAxes)
            continue
        selected = (times >= center - .35) & (times <= center + .35)
        for name, label, color, style in names:
            ax.plot(times[selected], parts[name][selected, component] * 1e9, style, color=color, lw=1.2, label=label)
        ax.set(xlabel='Arrival time (us)', ylabel=['Ex', 'Ey', 'Ez'][component] + ' (nV/m)',
               title='Near ' + around + ' peak | original time and amplitude')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('%s | seed %d | %s | 50-100 MHz' % (row['label'], row['seed'], row['observer']), fontsize=11)
    filename = row['tag'] + '_' + row['observer'] + '_components'
    for extension in ['png', 'pdf']:
        fig.savefig(out / 'figures' / (filename + '.' + extension), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('PSR only: no local numerical analysis')
    root = args.root
    assert (root / 'ALL_NINE_PASSED').exists()
    acceptance = json.loads((root / 'report/acceptance.json').read_text())
    assert acceptance['accepted_count'] == 9
    cases = json.loads((root / 'campaign.json').read_text())['cases']
    out = root / 'report/source_components'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    rows, checks, sources = [], [], []
    for case in cases:
        folder = root / 'runs' / case['tag']
        s = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
        assert s['complete'] and s['material_interface']['outside_region'] == 0 and s['material_interface']['inside_region'] == 1
        config = json.loads((folder / 'output/radio/ZHS/config.json').read_text())
        assert all(o['region'] == 0 for o in config['observers'])
        n, rate = config['samples'], config['sample_rate_Hz']
        times = (np.arange(n) / rate + config['start_time_s']) * 1e6
        frequencies = np.fft.rfftfreq(n, 1 / rate)
        mask = (frequencies >= 50e6) & (frequencies <= 100e6)
        algorithms = {}
        for algorithm in ['CoREAS', 'ZHS']:
            path = folder / 'output/radio' / algorithm / 'field.csv'
            frame = pd.read_csv(path, float_precision='round_trip')
            raw, parts = {}, {}
            for key, suffix in [('air', 'outside'), ('rock', 'inside'), ('total', 'V_m')]:
                raw[key] = frame[['E' + axis + '_' + suffix for axis in ['x', 'y', 'z']]].to_numpy().reshape(3, n, 3)
                parts[key] = np.fft.irfft(np.fft.rfft(raw[key], axis=1) * mask[None, :, None], n=n, axis=1)
                assert np.isfinite(parts[key]).all()
            raw_closure = rel(raw['air'] + raw['rock'], raw['total'])
            filtered_closure = rel(parts['air'] + parts['rock'], parts['total'])
            assert raw_closure < 1e-14 and filtered_closure < 1e-14
            checks.append(dict(tag=case['tag'], algorithm=algorithm, raw_addition_relative_l2=raw_closure,
                               filtered_addition_relative_l2=filtered_closure))
            sources.append(dict(path=str(path), sha256=sha(path)))
            algorithms[algorithm] = parts
        for i, observer in enumerate(config['observers']):
            parts = {name: values[i] for name, values in algorithms['ZHS'].items()}
            air, rock, total = [parts[name] for name in ['air', 'rock', 'total']]
            qa, qr, qt = [float(np.sum(v * v) / rate) for v in [air, rock, total]]
            cross = float(2 * np.sum(air * rock) / rate)
            assert qa + qr > 0 and qt > 0
            closure = abs(qt - qa - qr - cross) / qt
            assert closure < 1e-13
            row = dict(tag=case['tag'], material=case['material'], label=case['label'], seed=case['seed'],
                observer=observer['name'], air_Q_V2_s_m2=qa, rock_Q_V2_s_m2=qr,
                coherent_total_Q_V2_s_m2=qt, cross_Q_V2_s_m2=cross,
                rock_self_percent=100 * qr / (qa + qr), air_self_percent=100 * qa / (qa + qr),
                rock_over_coherent_total_percent=100 * qr / qt,
                air_over_coherent_total_percent=100 * qa / qt,
                cross_over_coherent_total_percent=100 * cross / qt,
                integral_closure_relative=closure)
            for name in ['air', 'rock', 'total']:
                row.update({name + '_' + key: value for key, value in peak(parts[name], times).items()})
                error = rel(algorithms['CoREAS'][name][i], parts[name])
                row[name + '_CoREAS_ZHS_relative_l2'] = error
                assert error < 1e-5, (case['tag'], observer['name'], name, error)
            if qa > 0 and qr > 0:
                pa = np.sum(air * air, axis=1) / (qa * rate)
                pr = np.sum(rock * rock, axis=1) / (qr * rate)
                row['normalized_time_power_overlap'] = float(np.minimum(pa, pr).sum())
                row['rock_minus_air_peak_us'] = row['rock_peak_time_us'] - row['air_peak_time_us']
                gate = np.abs(times - row['rock_peak_time_us']) <= .1
                ga, gr = [float(np.sum(v[gate]**2)) for v in [air, rock]]
                row['air_self_percent_in_rock_peak_0p2us_gate'] = 100 * ga / (ga + gr)
            else:
                row.update(normalized_time_power_overlap=None, rock_minus_air_peak_us=None,
                           air_self_percent_in_rock_peak_0p2us_gate=None)
            rows.append(row)
            for tag, station, title in EXAMPLES:
                if case['tag'] == tag and observer['name'] == station:
                    example_plot(out, row, title, times, parts)
        print('DECOMPOSED', case['tag'], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'contributions.csv', index=False, float_format='%.17g')
    data = dict(band_MHz=[50, 100], metric='Q = integral |vector E(t)|^2 dt, in V^2 s/m^2; received electric-field metric, not total emitted radio energy',
        source_definition='outside: emission track in air; inside: emission track in embedded material. Fields include the existing propagation, attenuation and transmission model.',
        fraction_definition='rock_self_percent = 100 Q_rock / (Q_air + Q_rock); interference excluded from this fraction and reported separately',
        coherent_identity='Q_total = Q_air + Q_rock + 2 integral E_air dot E_rock dt',
        sample_interval_ns=1e9 / rate, noise_or_antenna_response_included=False,
        evaluated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows, checks=checks, input_files=sources)
    (out / 'contributions.json').write_text(json.dumps(data, indent=2) + '\n')
    fraction = np.array([r['rock_self_percent'] for r in rows]).reshape(9, 3)
    strength = np.array([r['total_peak_nV_m'] for r in rows]).reshape(9, 3)
    labels = []
    for case in cases:
        short = 'Silica (SiO2)' if case['material'].startswith('silica') else 'Calcite (CaCO3)' if case['material'].startswith('limestone') else 'Granite mixture'
        labels.append(short + ' | ' + str(case['seed']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), constrained_layout=True)
    for ax, values, title in zip(axes, [fraction, strength],
            ['Rock share of independent component Q (%)', 'Coherent total peak magnitude (nV/m)']):
        is_share = ax is axes[0]
        im = ax.imshow(values, aspect='auto', cmap='viridis',
                       norm=Normalize(0, 100) if is_share else LogNorm(vmin=float(strength.min()), vmax=float(strength.max())))
        ax.set_xticks(range(3)); ax.set_xticklabels(['E01', 'E20', 'N10'])
        ax.set_yticks(range(9)); ax.set_yticklabels(labels if is_share else [''] * 9)
        ax.set_title(title, fontsize=10)
        for i in range(9):
            for j in range(3):
                value = float(values[i, j])
                ax.text(j, i, '0' if value == 0 else '%.3g' % value, ha='center', va='center', fontsize=9,
                        color='black' if im.norm(value) > .65 else 'white')
        fig.colorbar(im, ax=ax, shrink=.85)
    fig.suptitle('100 PeV nu_tau | 50-100 MHz | source medium at emission', fontsize=12)
    fig.supxlabel('Granite composition: H, C, O, Na, Mg, Al, Si, K, Ca, Fe. Independent Q shares exclude interference.', fontsize=9)
    for extension in ['png', 'pdf']:
        fig.savefig(out / 'figures' / ('rock_air_contributions.' + extension), dpi=170)
    plt.close(fig)
    md = ['---', 'marp: true', 'paginate: true', 'title: 山体源与大气源射电贡献', '---', '',
          '# 可以在模拟中分开；实际波形能否分开要逐例看', '',
          '九组模拟已全部完成。这里统计三个接收站的 **50–100 MHz** 电场，沿用之前报告的频带。所有接收站都在大气区域。', '',
          '`inside` 是山体内轨迹产生、传播到接收站的电场；`outside` 是大气中轨迹产生的电场。二者都已包含当前模型的传播和损耗。它们不是“光路有多少在山体中”的分类，也不等同于第一、第二个 bang。', '',
          r'$\mathbf E_{\rm total}=\mathbf E_{\rm air}+\mathbf E_{\rm rock}$', '',
          r'$Q_i=\int |\mathbf E_i(t)|^2dt,\quad Q_{\rm total}=Q_{\rm air}+Q_{\rm rock}+2\int\mathbf E_{\rm air}\cdot\mathbf E_{\rm rock}\,dt$', '',
          '**图表的山体占比 = Q_rock / (Q_air + Q_rock)**，大气占比为余下部分；干涉项另列。Q 是接收电场平方积分，不是山体或空气产生的总射电能量。', '',
          '---', '', '# 看贡献，也看绝对信号强度', '',
          '![](source_components/figures/rock_air_contributions.png)', '',
          '左图是山体源占比；大气源占比为 100% 减去该值。右图是总电场峰值，单位 nV/m，色标为对数。高占比不等于强信号。零值只表示当前模型在该站得到的山体源分量为零，不表示山体没有产生辐射。', '',
          '---', '', '# E20 站：定量贡献', '',
          '| 材料 | seed | 山体 Q 占比 | 大气 Q 占比 | 干涉 / Q_total |', '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['observer'] != 'E20':
            continue
        label = 'SiO₂' if row['material'].startswith('silica') else 'CaCO₃' if row['material'].startswith('limestone') else '花岗岩十元素混合物'
        md.append('| %s | %d | %.6g%% | %.6g%% | %+.4g%% |' % (label, row['seed'], row['rock_self_percent'], row['air_self_percent'], row['cross_over_coherent_total_percent']))
    md += ['', '花岗岩组分：H、C、O、Na、Mg、Al、Si、K、Ca、Fe。表中前两个占比采用独立分量之和为分母；干涉采用相干总量为分母，不能直接把三列相加。', '']
    captions = [
        '大气主峰约 22.984 μs，山体主峰约 41.160 μs，相隔 18.176 μs；在无噪声模拟中主脉冲可以按时间分开。山体峰值约 0.990 nV/m，大气峰值约 5.679 nV/m。',
        '山体与大气主峰约在 39.805、39.801 μs，只差 3.906 ns，即一个采样点。主峰处两者叠加，不能用简单时间窗完整分开。山体 Q 占 46.88%、大气占 53.12%；干涉 / Q_total 为 −2.48%。大气分量还有较早到达的信号。',
        '大气主峰约 21.102 μs、山体主峰约 56.750 μs，主脉冲在时间上分开。但峰值只有约 0.0357、0.0394 nV/m，是否能被真实探测器识别，需要噪声和天线响应。',
        '空气主峰约 23.195 μs，山体主峰约 39.801 μs。山体分量的积分占比仅约 0.0286%，但它是较短的脉冲，峰值约 8.00 nV/m；不能把积分占比直接当成峰值比。',
    ]
    for (tag, station, title), caption in zip(EXAMPLES, captions):
        md += ['---', '', '# ' + title, '', '![](source_components/figures/' + tag + '_' + station + '_components.png)', '',
               '蓝色：大气源；橙色：山体源；黑色虚线：相干总电场。上图为三维电场模，下图画同一个笛卡尔分量；未归一化或平移时间，下图各面板纵轴刻度独立。', '', caption, '']
    max_error = max(row[name + '_CoREAS_ZHS_relative_l2'] for row in rows for name in ['air', 'rock'])
    max_closure = max(c['filtered_addition_relative_l2'] for c in checks)
    md += ['---', '', '# 怎样理解“可分辨”', '',
           '**模拟来源分解：可以。** 两套算法都按轨迹产生位置分别累计；重加后恢复总场，相对 L2 残差最大 %.3g。山体／大气分量分别比较 CoREAS 与 ZHS，最大相对 L2 差异 %.3g。' % (max_closure, max_error), '',
           '**时间上分辨：部分事件可以看到分离的主脉冲，部分重叠。** 两个峰的时间差不是完整判据，还要看持续时间、弱分量幅度和重叠背景。图中分开的峰也不能脱离模型直接认定来自哪种介质。', '',
           '**实验上辨识来源：当前结果尚不能保证。** 本批电场尚未加入天线有效长度、接收机响应、噪声和触发。仅有总电场无法唯一恢复两个任意未知分量，需要利用传播／shower 模板、多站时延和偏振等额外约束。', '',
           '当前仍采用已说明的直线光路腿、最多一次透射，无反射或绕射；保留原截断、thinning 与有限输运窗。不能将这几组事件解释为材料的统计优劣。', '',
           '[全部 27 个事件—站点的数值表](source_components/contributions.csv) · [定义、相干项和核对结果](source_components/contributions.json)。分量核对和绘图均在 PSR 完成，无需重跑 shower。', '']
    (root / 'report/RADIO_COMPONENTS_CN.md').write_text('\n'.join(md))
    print(json.dumps(dict(rows=len(rows), maximum_component_algorithm_relative_l2=max_error,
                          maximum_addition_relative_l2=max_closure, output=str(out))), flush=True)


if __name__ == '__main__':
    main()
