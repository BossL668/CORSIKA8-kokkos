#!/usr/bin/env python3
"""Fe cooperative500: reuse validated CPU/CUDA packs; never rerun reference showers.

Only complete prescribed seed prefixes may be inspected. Partial plots explicitly
say interim. Finite-window radio means are projected with the original pulse code
BEFORE accumulating moments; stored marginal errors are never rotated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

METRICS = ['profile_xmax_charged_gcm2', 'profile_charged_max',
           'profile_charged_integral', 'profile_em_integral',
           'profile_photon_integral', 'profile_muon_integral',
           'profile_hadron_integral', 'energy_deposit_sum_GeV',
           'energy_deposit_xmax_gcm2', 'ground_em_weighted_count',
           'ground_em_kinetic_energy_GeV', 'energy_closure_fraction']
LABELS = {'cpu': 'Scalar CPU', 'cuda': 'Single Kokkos CUDA',
          'cooperative': 'CUDA + OpenMP16'}
COLORS = {'cpu': '#444444', 'cuda': '#0072B2', 'cooperative': '#D55E00'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p, kind='stable')
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def welch_from_moments(x, sx, nx, y, sy, ny):
    """SE, not per-event SD. Bonferroni later needs no independent-bin assumption."""
    vx, vy = sx*sx, sy*sy
    variance = vx+vy
    df = np.divide(variance**2, vx*vx/(nx-1)+vy*vy/(ny-1),
                   out=np.full_like(variance, np.inf), where=variance > 0)
    z = np.divide(np.abs(x-y), np.sqrt(variance), out=np.zeros_like(variance), where=variance > 0)
    p = 2*stats.t.sf(z, df)
    return np.where(variance > 0, p, np.where(x == y, 1., 0.))


def available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    raise RuntimeError('cannot determine host reserve')


def wait_complete(root, unit, hours):
    deadline = time.monotonic()+hours*3600
    marker = root/'simulation_complete_cuda-openmp.json'
    while not marker.exists():
        state = subprocess.run(['systemctl', '--user', 'is-active', unit],
                               capture_output=True, text=True).stdout.strip()
        if state not in ('active', 'activating'):
            raise RuntimeError('simulation stopped before all500 complete: '+state)
        if time.monotonic() > deadline:
            raise RuntimeError('analysis wait deadline exceeded; simulations untouched')
        time.sleep(15)
    status = read(marker)
    assert status['complete'] and status['complete_events'] == 500


def extract(root, out, count, pulse_package):
    import compare_ensembles as c
    import reproject_waveform_ensembles as w
    import run_cooperative_fe_campaign as runner
    # Backend control only; physics token comparison retains every physics option.
    c.NON_PHYSICS_OPTIONS_WITH_VALUE |= {'--kokkos-execution'}
    m, digest, _ = runner.read_manifest(root, 'cuda-openmp')
    project, basis = w.basis_for(dict(theta=0, phi=0, pulse_package=str(pulse_package)))
    rows, times, axes, curves, configs, sources = [], [], {}, {}, {}, []
    moments = {a: w.Moments() for a in ('CoREAS', 'ZHS')}
    first_config = None
    for i, seed in enumerate(m['seeds'][:count]):
        if available_mib() < 4096:
            raise RuntimeError('analysis host reserve below4GiB')
        record = runner.existing_record(root/'cuda-openmp'/f'event_{i:04d}.json',
                                       i, seed, 'cuda-openmp', digest)
        assert record['complete'], ('not a completed prescribed seed', seed)
        path = Path(record['output'])
        ens = c.extract_ensemble('cooperative', path, expect_gpu=True,
                                allow_legacy_provenance=True,
                                permitted_generic_fallback_reasons=(
                                    'unsupported_particle', 'unsupported_medium', 'unsupported_geometry'))
        cfg = ens.metadata['physics_configuration']
        if first_config is None:
            first_config = cfg
        assert cfg == first_config
        row = ens.scalars.iloc[0].to_dict()
        row.update(seed=seed, index=i, backend='cooperative')
        rows.append(row)
        for key, (x, y) in {**ens.curves, **ens.histograms}.items():
            if i == 0:
                axes[key] = x.copy()
                curves[key] = np.empty((count, y.shape[1]))
            assert np.array_equal(x, axes[key]) and y.shape[0] == 1
            curves[key][i] = y[0]
        for alg in moments:
            cfg = w.load(path/alg/'config.yaml')
            if i == 0:
                configs[alg] = cfg
            assert cfg == configs[alg]
            arrays = list(w.events(path/alg/'observers.parquet', cfg, 1, alg))
            assert len(arrays) == 1
            moments[alg].add(project(arrays[0]))
        attempt = record['attempts'][-1]
        timing = record['integrity']['simulation_timing']
        times.append(dict(seed=seed, backend='cooperative',
                          shower_s=timing['wall_time_ms']/1000,
                          process_s=attempt['simulation_process_wall_s'],
                          validation_s=attempt['validation_s'],
                          peak_rss_mib=attempt['peak_rss_bytes']/2**20,
                          **record['integrity']['cooperative']))
        sources.append(dict(seed=seed, path=str(path), marker_sha256=runner.sha(
            root/'cuda-openmp'/f'event_{i:04d}.json')))
        if i % 25 == 0:
            print(f'extracted {i+1}/{count}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'cooperative_scalars.csv', index=False)
    times = pd.DataFrame(times)
    times.to_csv(out/'cooperative_runtime.csv', index=False)
    np.savez_compressed(out/'cooperative_profiles.npz',
                        **{k+'__x': v for k, v in axes.items()},
                        **{k+'__y': v for k, v in curves.items()})
    wave = {k+'__'+s: getattr(v, s)() if s == 'se' else v.mean
            for k, v in moments.items() for s in ('mean', 'se')}
    np.savez_compressed(out/'cooperative_prime.npz', **wave)
    meta = dict(events=count, basis=basis, configs=configs, sources=sources,
                physics_configuration=first_config, manifest_sha256=digest)
    write(out/'cooperative_extraction.json', meta)
    return frame, times, axes, curves, wave, meta


def analyze(root, count, output, pulse_package, overview_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    plots = output/'plots'
    plots.mkdir()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    frame, times, axes, curves, wave, meta = extract(root, output, count, pulse_package)
    references = pd.read_csv(root/'reference/per_shower_observables.csv')
    frames = {b: references[references.backend == b].sort_values('seed') for b in ('cpu', 'cuda')}
    frames['cooperative'] = frame
    packs, prime, radio_meta = {}, {}, {}
    for b in ('cpu', 'cuda'):
        assert frames[b].seed.tolist() == list(range(85000001, 85000501))
        complete = read(root/'reference'/('profile_'+b)/'complete.json')
        assert complete['events'] == 500 and complete.get('success', True)
        with np.load(root/'reference'/('profile_'+b)/'profiles.npz') as z:
            packs[b] = {k: z[k] for k in z.files}
        with np.load(root/'reference'/('prime_'+b)/'prime_statistics.npz') as z:
            prime[b] = {k: z[k] for k in z.files}
        radio_meta[b] = read(root/'reference'/('prime_'+b)/'metadata.json')
        assert radio_meta[b]['complete'] and radio_meta[b]['events'] == 500
        for key in ('geometry_sha256', 'magnetic_field_sha256', 'cof_sha256'):
            assert radio_meta[b]['basis'][key] == meta['basis'][key]
        assert np.allclose(radio_meta[b]['basis']['rotation_rows'],
                           meta['basis']['rotation_rows'], rtol=0, atol=16*np.finfo(float).eps)
        current_tokens = dict(zip(meta['physics_configuration'][::2], meta['physics_configuration'][1::2]))
        ref_tokens = dict(zip(complete['physics_configuration'][::2], complete['physics_configuration'][1::2]))
        for key, value in ref_tokens.items():
            if key != '--observer-layout-sha256':  # Numerical observer check below.
                assert current_tokens.get(key) == value, (b, key, value, current_tokens.get(key))
    packs['cooperative'] = {**{k+'__x': v for k, v in axes.items()},
                            **{k+'__y': v for k, v in curves.items()}}
    prime['cooperative'] = wave
    radio_meta['cooperative'] = meta
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .15,
                         'savefig.dpi': 165, 'lines.linewidth': 1.2})
    labels = {b: f'{LABELS[b]} ({len(f)})' for b, f in frames.items()}
    profile_tests = []
    for key, y in curves.items():
        for b in ('cpu', 'cuda'):
            x = packs[b][key+'__y']
            p = welch_from_moments(y.mean(0), y.std(0, ddof=1)/np.sqrt(count), count,
                                   x.mean(0), x.std(0, ddof=1)/np.sqrt(len(x)), len(x))
            profile_tests.append(dict(reference=b, metric=key, bins_compared=p.size,
                min_pointwise_p=float(p.min()), conservative_global_p=float(min(1., p.min()*p.size))))
    pt = pd.DataFrame(profile_tests)
    pt['family_p'] = holm(pt.conservative_global_p)
    pt.to_csv(output/'profile_statistics.csv', index=False)
    title = 'Fe-56, 100 TeV, vertical'+(' — INTERIM' if count != 500 else '')
    rng = np.random.default_rng(20260910)
    scalar_tests = []
    fig, axs = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True)
    for ax, key in zip(axs.flat, METRICS):
        values = {b: f[key].to_numpy(float) for b, f in frames.items()}
        edges = np.histogram_bin_edges(np.concatenate(list(values.values())), bins=35)
        for b, v in values.items():
            assert np.isfinite(v).all()
            ax.hist(v, bins=edges, histtype='step', density=True, label=labels[b], color=COLORS[b])
        y = values['cooperative']
        for b in ('cpu', 'cuda'):
            x = values[b]
            bx = x[rng.integers(len(x), size=(1999, len(x)))].mean(1)
            by = y[rng.integers(len(y), size=(1999, len(y)))].mean(1)
            if abs(x.mean()) < 1e-100 or np.any(bx == 0):
                raise ValueError('zero reference denominator: '+key)
            low, high = np.quantile(by/bx-1, [.025, .975])
            p = float(stats.ttest_ind(y, x, equal_var=False).pvalue)
            if not np.isfinite(p):
                p = float(y.mean() == x.mean())
            ks = stats.ks_2samp(y, x)
            scalar_tests.append(dict(reference=b, metric=key, reference_mean=x.mean(),
                cooperative_mean=y.mean(), relative_difference=y.mean()/x.mean()-1,
                bootstrap_low=low, bootstrap_high=high, welch_p=p,
                ks_distance=float(ks.statistic), ks_p=float(ks.pvalue),
                within_1pct=bool(low >= -.01 and high <= .01)))
        label = key.replace('_', ' ')
        if key == 'energy_closure_fraction':
            label = 'Deposit + ground EM fraction (NOT full closure)'
        ax.set(xlabel=label, ylabel='Probability density')
        ax.xaxis.label.set_size(8)
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(plots/'shower_scalar_distributions.png')
    plt.close(fig)
    table = pd.DataFrame(scalar_tests)
    table['holm_welch_p'] = holm(table.welch_p)
    table['holm_ks_p'] = holm(table.ks_p)
    table.to_csv(output/'scalar_statistics.csv', index=False)
    pd.concat(frames.values()).to_csv(output/'per_shower_observables.csv', index=False)

    def profile(ax, key):
        x = packs['cpu'][key+'__x']
        occupied = None
        for b, p in packs.items():
            # Archived NumPy logspace edges differ by <=1 ULP across hosts.
            # This is an axis serialization check, not a rebinning operation.
            assert np.allclose(x, p[key+'__x'], rtol=8*np.finfo(float).eps, atol=0), key
            y = p[key+'__y']
            mean, ci = y.mean(0), stats.t.ppf(.975, len(y)-1)*y.std(0, ddof=1)/np.sqrt(len(y))
            xx = (x[:-1]+x[1:])/2 if len(x) == len(mean)+1 else x
            if not np.isfinite(xx).all():
                xx = np.arange(len(mean))
            visible = (np.abs(mean) > 0) | (np.abs(ci) > 0)
            occupied = visible if occupied is None else occupied | visible
            ax.plot(xx, mean, color=COLORS[b], label=labels[b])
            ax.fill_between(xx, mean-ci, mean+ci, color=COLORS[b], alpha=.16)
        ax.set(title=key.replace('_', ' '), xlabel='Slant depth [g cm$^{-2}$]',
               ylabel='Mean weighted count' if key.startswith('profile') else 'Mean per bin')
        if key.startswith('ground_'):
            ax.set_xlabel('Recorded histogram coordinate (overflow retained)')
        if key.startswith('ground_'):
            ax.set_xlim(float(xx[0]), float(xx[-1]))
            if np.all(xx > 0):
                ax.set_xscale('log')
        else:
            indices = np.flatnonzero(occupied)
            last = min(len(xx)-1, int(indices[-1])+3) if len(indices) else len(xx)-1
            # Stored atmosphere profiles include thousands of g/cm2 of zero padding.
            # Zoom display only; all bins remain in statistical calculations/NPZ.
            ax.set_xlim(float(xx[0]), float(xx[max(1, last)]))

    for key in curves:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        profile(ax, key)
        ax.legend(fontsize=8)
        fig.savefig(plots/(key+'_mean.png'))
        plt.close(fig)
    fig, axs = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, key in zip(axs.flat, ('profile_em', 'profile_electron_positron', 'profile_photon',
                                'profile_muon', 'profile_hadron', 'energy_deposit')):
        profile(ax, key)
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(title+'; pointwise95% mean intervals')
    fig.savefig(plots/'longitudinal_all_components_mean.png')
    plt.close(fig)

    # Simultaneous family bound uses a Bonferroni union, not an independence assumption.
    # It is deliberately conservative, and cannot demonstrate 1% waveform equivalence.
    radio_tests = []
    for alg in ('CoREAS', 'ZHS'):
        obs = list(meta['configs'][alg]['observers'].values())
        for b in ('cpu', 'cuda'):
            old = list(radio_meta[b]['configs'][alg]['observers'].values())
            assert len(old) == len(obs)
            for x, y in zip(old, obs):
                for k in ('location', 'start time', 'duration', 'number of bins', 'sampling frequency'):
                    assert np.allclose(x[k], y[k], rtol=0, atol=1e-8), (b, alg, k)
            p = welch_from_moments(wave[alg+'__mean'], wave[alg+'__se'], count,
                                   prime[b][alg+'__mean'], prime[b][alg+'__se'], 500)
            radio_tests.append(dict(algorithm=alg, reference=b, bins_compared=p.size,
                min_pointwise_p=float(p.min()), conservative_global_p=float(min(1., p.min()*p.size))))
        bins = wave[alg+'__mean'].shape[1]
        t = (np.arange(bins)+(.5 if alg == 'ZHS' else 0))/obs[0]['sampling frequency']
        direction = np.asarray(meta['basis']['direction'])
        groups = ([] if overview_only else [[i] for i in range(len(obs))])+[[0, 1, 5, 10, 20, 40]]
        for group in groups:
            fig, axs = plt.subplots(len(group), 3, figsize=(13, 2.8*len(group)),
                                    constrained_layout=True, squeeze=False)
            for row, i in enumerate(group):
                d = np.asarray(obs[i]['location'])-np.asarray(obs[0]['location'])
                radius = np.linalg.norm(d-direction*np.dot(d, direction))
                for j, component in enumerate(("Ex′", "Ey′", "Ez′")):
                    ax = axs[row, j]
                    envelope = np.zeros(bins)
                    for b in frames:
                        n = len(frames[b])
                        mean = prime[b][alg+'__mean'][i, :, j]*1e6
                        ci = stats.t.ppf(.975, n-1)*prime[b][alg+'__se'][i, :, j]*1e6
                        envelope = np.maximum(envelope, np.abs(mean))
                        ax.plot(t, mean, color=COLORS[b], label=labels[b])
                        ax.fill_between(t, mean-ci, mean+ci, color=COLORS[b], alpha=.16)
                    visible = np.flatnonzero(envelope > envelope.max()*.01)
                    lo, hi = (max(0, visible[0]-10), min(bins-1, visible[-1]+10)) if len(visible) else (0, bins-1)
                    ax.set_xlim(t[lo], t[max(lo+1, hi)])
                    ax.set(title=f'{alg} antenna{i}, r_perp={radius:.1f}m, {component}',
                           xlabel='Time from recorded window start [ns]', ylabel='Mean field [microV/m]')
            axs[0, 0].legend(fontsize=7)
            fig.suptitle(title+'; original polarization basis')
            suffix = 'overview' if len(group) > 1 else f'antenna_{group[0]:03d}'
            fig.savefig(plots/(alg+'_'+suffix+'_prime.png'))
            plt.close(fig)
    rt = pd.DataFrame(radio_tests)
    rt['family_p'] = holm(rt.conservative_global_p)
    rt.to_csv(output/'radio_statistics.csv', index=False)

    old_time = pd.read_csv(root/'reference/cuda_runtime.csv')
    assert sorted(old_time.seed) == list(range(85000001, 85000501))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, k in zip(axs, ('shower_s', 'process_s')):
        edges = np.histogram_bin_edges(np.r_[times[k], old_time[k]], bins=35)
        for b, v in (('cuda', old_time[k]), ('cooperative', times[k])):
            ax.hist(v, bins=edges, histtype='step', density=True, color=COLORS[b], label=labels[b])
        ax.set(xlabel=k+' [s]', ylabel='Probability density')
        ax.legend(fontsize=8)
    fig.savefig(plots/'runtime_distribution.png')
    plt.close(fig)
    timing = {b: {k: {'mean': float(v[k].mean()), 'median': float(v[k].median())}
                  for k in ('shower_s', 'process_s')}
              for b, v in (('cooperative', times), ('cuda', old_time))}
    write(output/'timing_summary.json', timing)
    scalar_rejections = int(((table.holm_welch_p < .05)|(table.holm_ks_p < .05)).sum())
    radio_rejections = int((rt.family_p < .05).sum())
    result = dict(status='ANALYSIS_COMPLETE_REVIEW_REQUIRED' if count == 500 else 'INTERIM_DIAGNOSTIC',
                  events=count, reference_events=500, scalar_rejections=scalar_rejections,
                  scalar_1pct_intervals=int(table.within_1pct.sum()), radio_rejections=radio_rejections,
                  profile_rejections=int((pt.family_p < .05).sum()),
                  production_release_accepted=False,
                  note='Neither non-significance nor shading overlap proves equivalence. Finite400ns waveform only.')
    write(output/'analysis_summary.json', result)
    lines = ['# Fe56 100TeV：CUDA＋OpenMP16 系综比较', '',
             f'新协同样本{count}例；已有CPU500例、单CUDA500例。'+('本次仅为小样本分析流程检查。' if count != 500 else ''), '',
             '保留修复后的磁偏转常数；相同种子不要求相同shower树。不回退常数追求旧二进制一致。',
             '使用独立事件bootstrap均值差区间、Welch与KS检验，各检验家族Holm校正。',
             f'标量校正后拒绝数{scalar_rejections}/24；bootstrap区间全部落入±1%的{int(table.within_1pct.sum())}/24。',
             f'profile全曲线保守检验拒绝数{int((pt.family_p < .05).sum())}/{len(pt)}；详见profile_statistics.csv。',
             f'射电4组保守全局检验拒绝数{radio_rejections}/4。原CPU仅有投影后均值/SE缓存，',
             '故采用逐点Welch及Bonferroni全曲线界，不冒充事件级波形bootstrap或严格1%等价。', '',
             '波形逐事件调用原pulse_analysis投影到Ex′/Ey′/Ez′后再平均；阴影为逐点Student-t95%均值区间。',
             '无峰值对齐、滤波、归一化或删尾点。显示范围按三组均值包络缩放；统计使用完整400ns记录。',
             '旧单CUDA/CPU来自历史二进制与批次；ZHS边缘修复、计账等版本变化不能归咎于协同调度。',
             'max-weight=0的自动值0.05；保持原样，不用改变薄化获取加速。',
             'energy_closure_fraction是历史提取器的沉积+地面EM份额，并非完整能量守恒检验。', '',
             '| 后端 | shower均值/中位[s] | 进程均值/中位[s] |', '|---|---:|---:|']
    for b, v in timing.items():
        lines.append(f'| {LABELS[b]} | {v["shower_s"]["mean"]:.2f}/{v["shower_s"]["median"]:.2f} | '
                     f'{v["process_s"]["mean"]:.2f}/{v["process_s"]["median"]:.2f} |')
    lines += ['', '时间比较为历史生产批次对照，不是受控交替运行性能验收；不以一个快事件代表稳定加速。', '',
              '[纵向组分总览](plots/longitudinal_all_components_mean.png) · [标量分布](plots/shower_scalar_distributions.png)',
              '[CoREAS](plots/CoREAS_overview_prime.png) · [ZHS](plots/ZHS_overview_prime.png) · [时间分布](plots/runtime_distribution.png)', '',
              '本报告完成数据处理，不自动授予生产发布通过。详细数值见scalar_statistics.csv与radio_statistics.csv。']
    (output/'VALIDATION_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pulse-package', type=Path, required=True)
    p.add_argument('--wait-unit')
    p.add_argument('--overview-only', action='store_true', help='smoke-check plotting without all81 antenna figures')
    p.add_argument('--wait-hours', type=float, default=36)
    a = p.parse_args()
    assert 2 <= a.count <= 500
    sys.path[:0] = [str(a.root/'scripts/analysis'), str(a.root/'scripts')]
    if a.wait_unit:
        assert a.count == 500
        wait_complete(a.root, a.wait_unit, a.wait_hours)
    if a.count == 500:
        assert read(a.root/'simulation_complete_cuda-openmp.json')['complete']
    analyze(a.root, a.count, a.output, a.pulse_package, a.overview_only)


if __name__ == '__main__':
    main()
