#!/usr/bin/env python3
"""Bounded-memory local three-way comparison; missing raw CPU data stays explicit.

Reuse the audited beta4 scalar extractor and pulse_analysis_modular APIs. CPU
scalar rows are mapped by the archived source ordering, NOT by sorting seeds.
Pointwise profile bands from archived sufficient statistics are diagnostic;
they cannot replace a shower-level covariance/global bootstrap analysis.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

from run_paired_acceptance import digest, load, save

COLORS = {'scalar': '#333333', 'cuda': '#0072B2', 'openmp': '#D55E00'}
LABELS = {'scalar': 'Archived scalar CPU', 'cuda': 'Beta5 Kokkos-CUDA', 'openmp': 'Beta5 Kokkos-OpenMP'}


def paired_mean(a, b, rng, repeats=2500):
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired scalar data invalid')
    d = b - a
    mean_a = float(np.mean(a))
    if mean_a == 0:
        return {'n': len(a), 'mean_cpu': mean_a, 'mean_candidate': float(np.mean(b)), 'status': 'zero_reference'}
    boot = []
    for i in range(0, repeats, 50):
        indices = rng.integers(len(a), size=(min(50, repeats - i), len(a)))
        denom = a[indices].mean(axis=1)
        good = denom != 0
        boot.extend((d[indices].mean(axis=1)[good] / denom[good]).tolist())
    if len(boot) < repeats * .99:
        return {'n': len(a), 'mean_cpu': mean_a, 'mean_candidate': float(np.mean(b)),
                'status': 'insufficient_nonzero_bootstrap_reference', 'valid_replicates': len(boot)}
    low, high = np.quantile(boot, [.025, .975])
    p = float(stats.ttest_rel(b, a).pvalue) if np.std(d) else (1. if np.mean(d) == 0 else 0.)
    return {'n': len(a), 'mean_cpu': mean_a, 'mean_candidate': float(np.mean(b)),
            'relative_mean_difference': float(np.mean(d) / mean_a),
            'paired_bootstrap_95': [float(low), float(high)], 'paired_mean_p': p,
            'point_estimate_within_1pct': bool(abs(np.mean(d) / mean_a) <= .01),
            'bootstrap_interval_within_1pct': bool(low >= -.01 and high <= .01),
            'bootstrap_contains_zero': bool(low <= 0 <= high)}


def add_curves(accum, curves):
    for name, (axis, matrix) in curves.items():
        y = matrix[0]
        if name not in accum:
            accum[name] = [axis.copy(), y.copy(), y*y, 1]
        else:
            x, total, squares, n = accum[name]
            if not np.array_equal(x, axis):
                raise ValueError('profile axes differ within a backend')
            total += y
            squares += y*y
            accum[name][3] = n + 1


def finish_curves(accum):
    return {name: (x, total/n, np.sqrt(np.maximum(0, (squares-total*total/n) / (n-1)/n)), n)
            for name, (x, total, squares, n) in accum.items() if n > 1}


def plot_profiles(output, curves, archived):
    groups = {'longitudinal_em': ['profile_em', 'profile_electron_positron', 'profile_photon', 'energy_deposit'],
              'longitudinal_non_em': ['profile_hadron', 'profile_muon', 'profile_muplus', 'profile_muminus'],
              'muon_production_parent': ['muon_production_parent_pion', 'muon_production_parent_kaon',
                                         'muon_production_parent_hadron', 'muon_production_parent_all']}
    for filename, names in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            cpu = archived[archived.observable == name]
            if len(cpu):
                x = cpu.coordinate.to_numpy()
                y = cpu.proposal_mean.to_numpy()
                se = cpu.proposal_standard_error.to_numpy()
                ax.plot(x, y, color=COLORS['scalar'], label='Scalar CPU (2000)')
                ax.fill_between(x, np.maximum(0, y-1.96*se), y+1.96*se, color=COLORS['scalar'], alpha=.12)
            for backend in ('cuda', 'openmp'):
                if name not in curves[backend]:
                    continue
                x, y, se, n = curves[backend][name]
                ax.plot(x, y, color=COLORS[backend], label=f'{backend} ({n})')
                ax.fill_between(x, np.maximum(0, y-1.96*se), y+1.96*se, color=COLORS[backend], alpha=.15)
            ax.set(xlim=(0, 1400), xlabel=r'$X$ [g cm$^{-2}$]', ylabel=name.replace('_', ' '))
            ax.legend(fontsize=8)
        fig.suptitle('Ensemble means; pointwise 95% normal-approximation bands')
        fig.savefig(output / f'{filename}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--skip-radio', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve()
    m = json.loads((root / 'campaign.json').read_text())
    tools = Path(m['analysis_tools'])
    if digest(tools / 'compare_ensembles.py') != m['analysis_tools_sha256']:
        raise ValueError('archived scalar extraction implementation changed')
    sys.path.insert(0, str(tools))
    import compare_ensembles as compare
    import analyze_geomagnetic_pulse_distributions as pulse
    import analyze_geomagnetic_radial_comparison as radial
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    out = root / ('local_comparison' if args.limit == 2000 else f'diagnostic_first_{args.limit}')
    out.mkdir(exist_ok=True)
    cpu = pd.read_csv(root / 'reference/ensemble_comparison_2000/per_shower_observables.csv')
    cpu = cpu[cpu.backend == 'proposal'].copy().set_index('shower')
    if len(cpu) != 2000 or not np.array_equal(np.sort(cpu.index), np.arange(2000)):
        raise ValueError('CPU cached scalar IDs are not exactly 0..1999')
    for event in m['events']:
        cpu.loc[event['index'], 'seed'] = event['seed']
    cpu['backend'] = 'scalar'
    scalar_frames = [cpu.reset_index()]
    accumulated = {b: {} for b in ('cuda', 'openmp')}
    runtime = []
    summaries = {}
    count = {}
    candidates = {}
    radio_rows = []
    api = None
    if not args.skip_radio:
        pulse_root = Path('/home/yuhanglu/21CMA/python/MCMCTidyUp/pulse_analysis_modular')
        api = pulse.load_reference_apis(pulse_root)
    for backend in ('cuda', 'openmp'):
        scalar_rows, antenna_rows = [], []
        basis = None
        for event in m['events'][:args.limit]:
            slot = root / backend / f"event_{event['index']:04d}"
            if not (slot / 'result.json').is_file():
                continue
            result = json.loads((slot / 'result.json').read_text())
            if result['status'] != 'complete':
                continue
            library = slot / 'shower'
            for rel, record in result['validation']['parquet'].items():
                if digest(library / rel) != record['sha256']:
                    raise ValueError(f'output changed: {library / rel}')
            e = compare.extract_ensemble(backend, library, True, permitted_generic_fallback_reasons=('unsupported_geometry',))
            row = e.scalars.iloc[0].to_dict()
            row.update(shower=event['index'], seed=event['seed'], backend=backend)
            scalar_rows.append(row)
            add_curves(accumulated[backend], e.curves)
            runtime.append({'backend': backend, 'index': event['index'], 'seed': event['seed'],
                            'shower_seconds': result['validation']['simulation_wall_ms']/1000,
                            'process_seconds': result['process_wall_s'], 'peak_rss_mib': result['peak_rss_mib']})
            if api:
                build_basis, analyze, read_radio, band, mask, Filter = api
                if basis is None:
                    config = load(library / 'gpu_em/config.yaml')
                    mag = config['environment']['magnetic_field_T']
                    basis = build_basis(m['physics']['zenith_deg'], m['physics']['azimuth_deg'],
                                        np.array([mag[k] for k in ('x', 'y', 'z')]))
                for algorithm in ('CoREAS', 'ZHS'):
                    records = radial.load_records(output_directories=[library], algorithm=algorithm,
                        shower_axis_nwu=basis['n'], core_xy_m=(0., 0.), read_radio_records=read_radio)
                    for r in records:
                        r['shower'] = event['index']
                    radii = radial.clustered_available_radii(records, minimum_m=1., maximum_m=600.)
                    for radius in radii:
                        antenna_rows.extend(pulse.extract_antenna_rows(records=records, backend=backend,
                            algorithm=algorithm, radius_m=radius, yprime=basis['yprime'],
                            analyze_pulse_parameters=analyze, band_limited_waveform=band,
                            band_MHz=None, analysis_sampling_rate_GHz=None))
            if len(scalar_rows) % 100 == 0:
                print(f'{backend}: extracted {len(scalar_rows)} showers', flush=True)
        count[backend] = len(scalar_rows)
        if len(scalar_rows) < 2:
            raise ValueError(f'{backend}: at least two complete events needed')
        frame = pd.DataFrame(scalar_rows).set_index('shower')
        candidates[backend] = frame
        scalar_frames.append(frame.reset_index())
        summaries[backend] = {}
        rng = np.random.default_rng(2026090501)
        for name in compare.DEFAULT_KEY_SCALAR_METRICS:
            summaries[backend][name] = paired_mean(cpu.loc[frame.index, name].to_numpy(), frame[name].to_numpy(), rng)
        keys = [k for k, v in summaries[backend].items() if 'paired_mean_p' in v]
        rejects = pulse.holm_bonferroni_rejections([summaries[backend][k]['paired_mean_p'] for k in keys], .025)
        for key, rejected in zip(keys, rejects):
            summaries[backend][key]['paired_mean_holm_reject'] = rejected
        if api:
            pulse.apply_reference_width_filter(antenna_rows, robust_pulse_width_mask=mask, filter_config=Filter())
            radio_rows.extend(pulse.aggregate_by_shower(antenna_rows))
    pd.concat(scalar_frames, ignore_index=True).to_csv(out / 'per_shower_observables.csv', index=False)
    curve_results = {b: finish_curves(v) for b, v in accumulated.items()}
    archived = pd.read_csv(root / 'reference/ensemble_comparison_2000/curve_comparison.csv')
    plot_profiles(out, curve_results, archived)
    flattened = []
    for b, curves in curve_results.items():
        for name, (x, y, se, n) in curves.items():
            flattened.extend({'backend': b, 'observable': name, 'X': xx, 'mean': yy, 'standard_error': ss, 'n': n}
                             for xx, yy, ss in zip(x, y, se))
    pd.DataFrame(flattened).to_csv(out / 'new_profile_summary.csv', index=False)
    metrics = list(compare.DEFAULT_KEY_SCALAR_METRICS)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, metrics):
        values = {'scalar': cpu[name].to_numpy(), **{b: f[name].to_numpy() for b, f in candidates.items()}}
        bins = np.histogram_bin_edges(np.concatenate(list(values.values())), bins=45)
        for b, y in values.items():
            ax.hist(y, bins=bins, density=True, histtype='step', color=COLORS[b], label=f'{b} ({len(y)})')
        ax.set_title(name.replace('_', ' '), fontsize=8)
        ax.legend(fontsize=7)
    fig.savefig(out / 'shower_scalar_distributions.png', dpi=180)
    plt.close(fig)
    for event in m['events']:
        runtime.append({'backend': 'scalar', 'index': event['index'], 'seed': event['seed'],
                        'shower_seconds': event['cpu_wall_ms']/1000, 'process_seconds': np.nan, 'peak_rss_mib': np.nan})
    times = pd.DataFrame(runtime)
    times.to_csv(out / 'runtime_per_event.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ('shower_seconds', 'process_seconds')):
        for b, frame in times.groupby('backend'):
            values = frame[key].dropna().to_numpy()
            if len(values):
                ax.hist(values, bins=40, histtype='step', density=True, label=b, color=COLORS[b])
        ax.set(xlabel=key.replace('_', ' '), ylabel='Probability density', xscale='log')
        ax.legend()
    fig.savefig(out / 'runtime_distributions.png', dpi=180)
    plt.close(fig)
    if api:
        old = pd.read_csv(root / 'reference/geomagnetic_radial_validation/per_shower_radius_features.csv')
        old = old[old.backend == 'legacy_proposal'].copy()
        old['backend'] = 'scalar'
        radio = pd.concat([old, pd.DataFrame(radio_rows)], ignore_index=True)
        radio.to_csv(out / 'radio_per_shower_radius.csv', index=False)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        radio_summary = []
        for row, algorithm in enumerate(('CoREAS', 'ZHS')):
            for col, metric in enumerate(('geomagnetic_amplitude_geomean_V_per_m', 'pulse_width_mean_ns')):
                ax = axes[row, col]
                for b in ('scalar', 'cuda', 'openmp'):
                    selected = radio[(radio.backend == b) & (radio.algorithm == algorithm)]
                    points = []
                    for radius, frame in selected.groupby('radius_m'):
                        values = frame[metric].to_numpy()
                        values = values[np.isfinite(values) & (values > 0)]
                        if len(values) < 2:
                            continue
                        transformed = np.log(values) if col == 0 else values
                        center = transformed.mean()
                        se = transformed.std(ddof=1) / np.sqrt(len(values))
                        triplet = np.array([center, center-1.96*se, center+1.96*se])
                        if col == 0:
                            triplet = np.exp(triplet)
                        points.append((radius, *triplet))
                        radio_summary.append({'backend': b, 'algorithm': algorithm, 'metric': metric,
                            'r_perp_m': radius, 'valid_showers': len(values), 'total_showers': len(frame),
                            'mean': triplet[0], 'low': triplet[1], 'high': triplet[2]})
                    if points:
                        x, y, low, high = np.array(points).T
                        ax.plot(x, y, color=COLORS[b], label=b)
                        ax.fill_between(x, low, high, color=COLORS[b], alpha=.14)
                ax.set(xlabel=r'$r_\perp$ [m]', ylabel='Amplitude [V/m]' if col == 0 else 'Pulse width [ns]',
                       title=algorithm, xscale='log')
                if col == 0:
                    ax.set_yscale('log')
                ax.legend()
        fig.suptitle(f"CPU: 2000; CUDA: {count['cuda']}; OpenMP: {count['openmp']} showers\n"
                     'Pulse-analysis features; pointwise 95% bands, valid-pulse population')
        fig.savefig(out / 'radio_radial_profiles.png', dpi=180)
        plt.close(fig)
        pd.DataFrame(radio_summary).to_csv(out / 'radio_radial_summary.csv', index=False)
    report = {'status': 'local_diagnostics_complete_not_full_acceptance', 'counts': count,
              'cpu_cached_events': 2000, 'paired_scalar_mean_tests': summaries,
              'bootstrap_repetitions': 2500, 'resampling_unit': 'whole matched seed pair',
              'paired_mean_familywise_alpha': .05, 'families': 'nine observables per backend; alpha=.025 each',
              'cpu_raw_profiles_available': m['raw_cpu_profiles_available'],
              'pending': ['raw CPU profile covariance/global distribution acceptance',
                          'full radio equivalence and multiplicity-controlled distribution tests',
                          'thinning-active validation: default maximum weight disables thinning at this energy']}
    save(out / 'comparison.json', report)
    text = '# Beta5 三组比较：本地诊断\n\n'
    text += f"样本：历史单核 CPU 2000；Kokkos-CUDA {count['cuda']}；Kokkos-OpenMP {count['openmp']}。\n\n"
    text += ('CPU 缓存逐事件标量按原归档来源顺序映射种子。均值比较使用完整配对 shower bootstrap，'
             '不是把粒子或天线视为独立样本。原 CPU 完整 profile/波形当前不在本地，'
             '所以这些图不能替代全局协方差与完整分布验收，不把“无显著差异”写成“1% 等价”。\n\n'
             '纵向图：历史 CPU2000 的均值/标准误与新样本均值及点态 95% 正态近似阴影。'
             '射电继续调用 pulse_analysis_modular，横轴 r_perp；宽度筛选按整个后端样本执行，'
             '有效率保存在 radio_radial_summary.csv。小样本结果仅为诊断。\n\n'
             '运行时间分开保存 shower 区间与新进程端到端用时；历史 CPU 是并发单核生产数据，'
             '不将其比值标注为隔离单核硬件基准。\n\n')
    for b, frame in times.groupby('backend'):
        text += f"- {LABELS[b]}：{len(frame)} 例，shower 中位 {frame.shower_seconds.median():.3f} s。\n"
    text += '\n图：shower_scalar_distributions.png、longitudinal_em.png、longitudinal_non_em.png、muon_production_parent.png、runtime_distributions.png、radio_radial_profiles.png。\n'
    (out / 'LOCAL_COMPARISON_REPORT_CN.md').write_text(text)
    print(f'Local diagnostic comparison written to {out}; full physics acceptance remains pending.', flush=True)


if __name__ == '__main__':
    main()
