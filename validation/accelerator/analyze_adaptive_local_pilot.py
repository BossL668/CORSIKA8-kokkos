#!/usr/bin/env python3
"""Read-only, small-sample Fe acceptance of single/legacy/adaptive local pilots.

Reuses the established profile extractor and ORIGINAL pulse polarization code.
No simulations or production sources are changed. All resampling units are
whole showers; a shared seed is a block, not a promise of identical trees.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from scipy import stats


MODES = ("single-cuda", "legacy20", "adaptive20")
LABELS = {"single-cuda": "Single CUDA", "legacy20": "Legacy CUDA + OMP20",
          "adaptive20": "Adaptive CUDA + OMP20"}
COLORS = {"single-cuda": "#0072B2", "legacy20": "#666666", "adaptive20": "#D55E00"}
SCALARS = ("profile_xmax_em_gcm2", "profile_em_max", "profile_em_integral",
           "profile_electron_positron_integral", "profile_photon_integral",
           "profile_muon_integral", "profile_hadron_integral", "energy_deposit_sum_GeV",
           "ground_em_weighted_count", "ground_em_kinetic_energy_GeV",
           "ground_muon_weighted_count", "ground_hadron_weighted_count")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p, kind="stable")
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    return result


def check_reserve():
    if psutil.virtual_memory().available < 4*2**30:
        raise RuntimeError("analysis stopped: less than 4 GiB available; simulation untouched")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--analysis-module-dir", type=Path, required=True)
    ap.add_argument("--prime-module-dir", type=Path, required=True)
    ap.add_argument("--pulse-package", type=Path, required=True)
    args = ap.parse_args()
    sys.path[:0] = [str(args.analysis_module_dir), str(args.prime_module_dir)]
    import compare_ensembles as c
    import reproject_waveform_ensembles as w
    import summarize_local_adaptive_pilots as timing_helper
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root, out = args.root, args.output
    check_reserve()
    out.mkdir(parents=True, exist_ok=False)
    plots = out / "plots"
    plots.mkdir()
    snapshot = read(root / "STATUS.json")
    write(out / "INPUT_STATUS_SNAPSHOT.json", snapshot)
    source = read(root / "PROVENANCE.json")
    assert sha(root / "binaries/c8_air_shower") == source["binary_sha256"]
    events = [r for r in snapshot["records"] if r["complete"] and not r["warmup"]
              and r["family"] == "Fe100TeV"]
    assert len(events) == 15, "this pilot requires all prescribed five seeds in all three modes"
    for mode in MODES:
        assert sorted(r["seed"] for r in events if r["mode"] == mode) == list(range(85000001, 85000006))
    project, basis = w.basis_for(dict(theta=0, phi=0, pulse_package=str(args.pulse_package)))
    c.NON_PHYSICS_OPTIONS_WITH_VALUE |= {"--kokkos-execution", "--kokkos-cooperative-policy",
                                       "--kokkos-num-threads", "--kokkos-device"}
    frames, curves, waves, axes, configs, integrity = {}, {}, {}, {}, {}, []
    configuration = environment = table_id = None
    times = []
    for mode in MODES:
        rows, ys, ws = [], {}, {alg: [] for alg in w.ALGORITHMS}
        for record in sorted((r for r in events if r["mode"] == mode), key=lambda r: r["seed"]):
            check_reserve()
            label = f"Fe100TeV-{record['seed']}-{mode}"
            path = root / label
            guard = read(root / (label + "-guard/summary.json"))
            assert guard["pass"] and guard["command"] == record["command"]
            assert str(path) == record["command"][record["command"].index("-f")+1]
            summary = w.load(path / "summary.yaml")
            assert summary.get("end time") and summary["showers"] == 1 and summary["seed"] == record["seed"]
            primary = w.load(path / "primary/summary.yaml")["shower_0"]
            assert primary["pdg"] == 1000260560 and primary["total_energy"] == 100000
            assert np.allclose([primary[k] for k in ("nx", "ny", "nz")], basis["direction"], atol=2e-12, rtol=0)
            stats_gpu = w.load(path / "gpu_em/summary.yaml")["shower_0"]["statistics"]
            cfg_gpu = w.load(path / "gpu_em/config.yaml")
            env = cfg_gpu["environment"]
            if environment is None:
                environment = env
            assert environment == env
            # The established Python pulse code independently evaluates IGRF.
            # Compare orientation and magnitude rather than demanding bitwise
            # equality with the C++ implementation (measured ~1.6e-11 relative).
            # Inter-mode simulation fields themselves remain EXACTLY equal above.
            field = np.array([env["magnetic_field_T"][k]*1e6 for k in ("x", "y", "z")])
            pulse_field = np.asarray(basis["B_NWU_microtesla"])
            field_delta = np.linalg.norm(field-pulse_field)/np.linalg.norm(field)
            field_angle = np.arctan2(np.linalg.norm(np.cross(field,pulse_field)), field@pulse_field)
            assert field_delta < 1e-9 and field_angle < 1e-9
            native = stats_gpu["proposal_native"]
            identity = (native["proposal_version"], native["cubic_interpolation_version"],
                        native["table_sha256"], native["aux_sha256"])
            if table_id is None:
                table_id = identity
            assert table_id == identity and native["inverse_failures"] == 0
            # The router's independent-subshower branch completes specified
            # losses immediately, whereas single CUDA defers them. Its legacy
            # "cpu_generic_fallbacks" name counts immediate scalar returns, not
            # the same physical subset in both modes. Audit each identity/count.
            reasons = stats_gpu["cpu_fallbacks_by_reason_name"] or {}
            replay = reasons.get("native_selection_replay", 0)
            geometry = reasons.get("unsupported_geometry", 0)
            epair = reasons.get("epair_rejection_envelope_exceeded", 0)
            assert not (set(reasons)-{"native_selection_replay", "unsupported_geometry", "epair_rejection_envelope_exceeded"})
            assert replay == stats_gpu["cpu_completed_native_selection_replays"] == stats_gpu["cpu_completed_selected_losses"]
            assert replay + epair == stats_gpu["cpu_specified_final_states"]
            if mode == "single-cuda":
                assert stats_gpu["cpu_generic_fallbacks"] == geometry
                assert replay + epair == stats_gpu["deferred_cpu_fallbacks_queued"] == stats_gpu["deferred_cpu_fallbacks_flushed"]
                permitted_generic = ("unsupported_geometry",)
            else:
                assert stats_gpu["cpu_generic_fallbacks"] == replay + epair + geometry
                assert stats_gpu["deferred_cpu_fallbacks_queued"] == stats_gpu["deferred_cpu_fallbacks_flushed"] == 0
                permitted_generic = ("native_selection_replay", "unsupported_geometry", "epair_rejection_envelope_exceeded")
            assert stats_gpu["profile"]["fixed_point_overflows"] == stats_gpu["radio"]["fixed_point_overflows"] == 0
            coop = stats_gpu["accelerator"].get("cooperative", {})
            if mode != "single-cuda":
                assert coop["subshower_cuda_submissions"] == coop["subshower_cuda_commits"] > 0
                assert coop["subshower_openmp_epochs"] > 0
            if mode == "adaptive20":
                assert coop["scheduling_policy"] == "adaptive-v2"
            ensemble = c.extract_ensemble(mode, path, True, allow_legacy_provenance=True,
                                          permitted_generic_fallback_reasons=permitted_generic)
            if configuration is None:
                configuration = ensemble.metadata["physics_configuration"]
            assert ensemble.metadata["physics_configuration"] == configuration
            scalars = ensemble.scalars.iloc[0].to_dict()
            x, y = ensemble.curves["profile_em"]
            xmax, maximum = c.quadratic_peak(x, y[0])
            scalars.update(seed=record["seed"], mode=mode, profile_xmax_em_gcm2=xmax, profile_em_max=maximum)
            rows.append(scalars)
            for key, (x, y) in {**ensemble.curves, **ensemble.histograms}.items():
                if key not in axes:
                    axes[key] = x.copy()
                assert np.array_equal(axes[key], x) and y.shape[0] == 1
                ys.setdefault(key, []).append(y[0].copy())
            edges = {}
            for alg in w.ALGORITHMS:
                cfg = w.load(path / alg / "config.yaml")
                if alg not in configs:
                    configs[alg] = cfg
                assert cfg == configs[alg]
                raw = list(w.events(path / alg / "observers.parquet", cfg, 1, alg))
                assert len(raw) == 1 and np.isfinite(raw[0]).all()
                z = project(raw[0])
                ws[alg].append(z.copy())
                energy = np.sum(z*z, axis=(1, 2))
                edge_energy = np.sum(z[:, :5]**2, axis=(1, 2)) + np.sum(z[:, -5:]**2, axis=(1, 2))
                ratios = np.divide(edge_energy, energy, out=np.zeros_like(energy), where=energy>0)
                edges[alg] = dict(max_first_last_5bin_squared_field_fraction=float(ratios.max()),
                                  weighted_fraction=float(edge_energy.sum()/energy.sum()) if energy.sum() else 0.)
            integrity.append(dict(mode=mode, seed=record["seed"], completed=True,
                native_selection_replays=replay, cpu_generic_fallbacks_accounted=True,
                scalar_geometry_returns=geometry, specified_epair_envelope_completions=epair,
                fallback_reasons=reasons, raw_cpu_generic_fallbacks=stats_gpu["cpu_generic_fallbacks"],
                unexpected_fallbacks=0, queue_overflows=stats_gpu["queue_overflows"],
                native_inverse_failures=native["inverse_failures"], energy_ledger=stats_gpu["energy_ledger"],
                radio_edges=edges, pulse_vs_simulation_field_relative_difference=float(field_delta),
                pulse_vs_simulation_field_angle_rad=float(field_angle),
                input_summary_sha256=sha(path / "summary.yaml")))
            t = dict(mode=mode, seed=record["seed"], process_s=record["process_s"], shower_s=record["shower_s"],
                     peak_rss_gib=guard["peak_tree_rss_bytes"]/2**30)
            t.update(timing_helper.telemetry(root / (label+"-guard/resources.jsonl")))
            times.append(t)
            print("checked", label, flush=True)
        frames[mode] = pd.DataFrame(rows)
        curves[mode] = {k: np.asarray(v) for k, v in ys.items()}
        waves[mode] = {k: np.asarray(v) for k, v in ws.items()}
        np.savez_compressed(out / (mode+"_profiles.npz"), **{
            **{k+"__x": v for k, v in axes.items()}, **{k+"__y": v for k, v in curves[mode].items()}})
        np.savez_compressed(out / (mode+"_prime_waveforms.npz"), **waves[mode])
    pd.concat(frames.values()).to_csv(out / "per_shower_observables.csv", index=False)
    times = pd.DataFrame(times)
    times.to_csv(out / "per_shower_runtime.csv", index=False)
    write(out / "integrity.json", integrity)
    write(out / "provenance.json", dict(physics_configuration=configuration, environment=environment,
          basis=basis, radio_configs=configs, table_identity=table_id, binary=source,
          analysis_sha256=sha(Path(__file__)), extractor_sha256=sha(Path(c.__file__)),
          projection_wrapper_sha256=sha(Path(w.__file__))))

    # Enumerate all 5^5 ordinary paired bootstrap draws; all endpoints of the
    # same original seed retain their pairing. Percentiles are exploratory.
    ix = np.array(list(itertools.product(range(5), repeat=5)))
    weights = np.eye(5)[ix].mean(axis=1)
    signs = np.array(list(itertools.product((-1., 1.), repeat=5)))

    def comparison(x, y):
        bx, by = weights @ x, weights @ y
        delta = y-x
        permutations = signs @ delta/5
        observed = abs(delta.mean())
        p = float(np.mean(abs(permutations) >= observed-1e-12*max(1., observed)))
        valid = np.abs(bx) > 1e-100
        interval = np.quantile(by[valid]/bx[valid]-1, [.025, .975]) if valid.all() else [None, None]
        return dict(reference_mean=float(x.mean()), adaptive_mean=float(y.mean()),
                    mean_difference=float(delta.mean()),
                    relative_mean_difference=float(y.mean()/x.mean()-1) if abs(x.mean())>1e-100 else None,
                    paired_bootstrap_low=interval[0], paired_bootstrap_high=interval[1],
                    paired_sign_flip_p=p,
                    entire_interval_within_1pct=bool(interval[0] is not None and interval[0]>=-.01 and interval[1]<=.01))

    tests = []
    for base in MODES[:2]:
        for key in SCALARS:
            tests.append(dict(reference=base, metric=key,
                              **comparison(frames[base][key].to_numpy(), frames["adaptive20"][key].to_numpy())))
    tests = pd.DataFrame(tests)
    tests["holm_sign_flip_p"] = holm(tests.paired_sign_flip_p)
    tests.to_csv(out / "scalar_comparisons.csv", index=False)
    runtime = []
    for base in MODES[:2]:
        for metric in ("process_s", "shower_s"):
            x = times[times["mode"] == base][metric].to_numpy()
            y = times[times["mode"] == "adaptive20"][metric].to_numpy()
            d = 100*(np.median(y[ix], axis=1)/np.median(x[ix], axis=1)-1)
            runtime.append(dict(reference=base, metric=metric, **comparison(x, y),
                reference_median=float(np.median(x)), adaptive_median=float(np.median(y)),
                median_time_change_percent=100*(np.median(y)/np.median(x)-1),
                median_change_bootstrap_low=float(np.quantile(d, .025)),
                median_change_bootstrap_high=float(np.quantile(d, .975))))
    write(out / "runtime_comparisons.json", runtime)
    radio_tests = []
    for base in MODES[:2]:
        for alg in w.ALGORITHMS:
            x, y = waves[base][alg].reshape(5,-1), waves["adaptive20"][alg].reshape(5,-1)
            delta = y-x
            gram = delta @ delta.T
            null = np.einsum("bi,ij,bj->b", signs, gram, signs)/25
            observed = float(gram.sum()/25)
            p = float(np.mean(null >= observed-1e-12*max(float(abs(null).max()), 1e-100)))
            radio_tests.append(dict(reference=base, algorithm=alg,
                relative_mean_waveform_l2=float(np.linalg.norm(delta.mean(0))/np.linalg.norm(x.mean(0))),
                paired_sign_flip_p=p))
    radiot = pd.DataFrame(radio_tests)
    radiot["holm_sign_flip_p"] = holm(radiot.paired_sign_flip_p)
    radiot.to_csv(out / "radio_comparisons.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                         "grid.alpha": .16, "savefig.dpi": 160, "lines.linewidth": 1.35})
    title = "Fe-56, 100 TeV, vertical, thinning $10^{-6}$ — 5 showers per mode (PILOT)"
    shade = stats.t.ppf(.975, 4)/np.sqrt(5)

    def means(ax, key):
        axis = axes[key]
        size = curves[MODES[0]][key].shape[1]
        x = (axis[:-1]+axis[1:])/2 if len(axis) == size+1 else axis
        support = np.zeros(size, bool)
        for mode in MODES:
            yy = curves[mode][key]
            mean, ci = yy.mean(0), shade*yy.std(0, ddof=1)
            ax.plot(x, mean, color=COLORS[mode], label=LABELS[mode])
            ax.fill_between(x, mean-ci, mean+ci, color=COLORS[mode], alpha=.15)
            support |= np.abs(mean) > np.max(np.abs(mean))*1e-5
        use = np.flatnonzero(support)
        if len(use):
            ax.set_xlim(x[max(0,use[0]-3)], x[min(size-1,use[-1]+3)])
        ax.set(title=key.replace("_", " "), xlabel="Slant depth [g cm$^{-2}$]",
               ylabel="Mean weighted count" if key != "energy_deposit" else "Mean deposited energy/bin [GeV]")

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax, key in zip(axs.flat, ("profile_em", "profile_electron_positron", "profile_photon",
                                 "profile_muon", "profile_hadron", "energy_deposit")):
        means(ax, key)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(title+"\nShading: pointwise 95% Student-t intervals for the mean")
    fig.savefig(plots / "longitudinal_all_components_mean.png")
    plt.close(fig)
    fig, axs = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    for ax, key in zip(axs.flat, [k for k in axes if k.startswith("muon_production_parent_")]):
        means(ax, key)
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(title+"\nMuon production parent profiles; pointwise 95% mean intervals")
    fig.savefig(plots / "muon_production_parent_mean.png")
    plt.close(fig)

    fig, axs = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
    for ax, key in zip(axs.flat, SCALARS):
        values = {mode: frames[mode][key].to_numpy() for mode in MODES}
        edges = np.histogram_bin_edges(np.concatenate(list(values.values())), bins=5)
        for mode, v in values.items():
            ax.hist(v, bins=edges, histtype="step", color=COLORS[mode], label=LABELS[mode])
            ax.plot(v, np.full_like(v, -.08*(1+MODES.index(mode))), "|", color=COLORS[mode])
        ax.set(xlabel=key.replace("_", " "), ylabel="Events/bin")
        ax.xaxis.label.set_size(8)
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(title+"\nFive-bin histograms and individual-event marks; not a distribution acceptance")
    fig.savefig(plots / "shower_scalar_distributions.png")
    plt.close(fig)

    for alg in w.ALGORITHMS:
        obs = list(configs[alg]["observers"].values())
        count = waves[MODES[0]][alg].shape[2]
        t = (np.arange(count)+(.5 if alg == "ZHS" else 0))/obs[0]["sampling frequency"]
        group = [i for i in (0,1,5,10,20,40) if i < len(obs)]
        direction = np.asarray(basis["direction"])
        fig, axs = plt.subplots(len(group), 3, figsize=(13, 2.5*len(group)), constrained_layout=True)
        for row, i in enumerate(group):
            d = np.asarray(obs[i]["location"])-np.asarray(obs[0]["location"])
            radius = np.linalg.norm(d-d.dot(direction)*direction)
            envelope = np.max([abs(waves[m][alg][:,i].mean(0)).max(axis=1) for m in MODES], axis=0)
            visible = np.flatnonzero(envelope > envelope.max()*.01)
            lo, hi = (max(0,visible[0]-8), min(count-1,visible[-1]+8)) if len(visible) else (0,count-1)
            for j, component in enumerate((r"$E_{x'}$",r"$E_{y'}$",r"$E_{z'}$")):
                ax = axs[row,j]
                for mode in MODES:
                    yy = waves[mode][alg][:,i,:,j]*1e6
                    mean, ci = yy.mean(0), shade*yy.std(0,ddof=1)
                    ax.plot(t, mean, color=COLORS[mode], label=LABELS[mode])
                    ax.fill_between(t, mean-ci, mean+ci, color=COLORS[mode], alpha=.15)
                ax.set(xlim=(t[lo], t[max(lo+1,hi)]), title=f"Antenna {i}, $r_\\perp$={radius:.1f} m, {component}",
                       xlabel="Time from recorded window start [ns]", ylabel=r"Mean field [$\mu$V/m]")
        axs[0,0].legend(fontsize=7)
        fig.suptitle(title+f"\n{alg}: signed waveform, original pulse basis, pointwise 95% mean intervals")
        fig.savefig(plots / (alg+"_overview_prime.png"))
        plt.close(fig)

    fig, axs = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax, metric in zip(axs,("process_s","shower_s")):
        bins = np.histogram_bin_edges(times[metric],bins=6)
        for mode in MODES:
            x = times[times["mode"] == mode][metric]
            ax.hist(x,bins=bins,histtype="step",color=COLORS[mode],label=LABELS[mode])
            ax.plot(x,np.full(len(x),-.08*(1+MODES.index(mode))),"|",color=COLORS[mode])
        ax.set(xlabel=metric+" [s]",ylabel="Events/bin")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(plots / "runtime_distribution.png")
    plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for mode in MODES:
        tm = times[times["mode"]==mode]
        axs[0].plot(np.arange(1,6), tm.process_s, "o-", color=COLORS[mode],label=LABELS[mode])
        axs[1].plot(np.arange(1,6), tm.average_logical_cores,"o-",color=COLORS[mode],label=LABELS[mode])
    axs[0].set(xlabel="Seed index (85000000 + index)",ylabel="End-to-end time [s]",xticks=range(1,6))
    axs[1].set(xlabel="Seed index",ylabel="Average logical-core equivalents",xticks=range(1,6),ylim=(0,21))
    axs[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(plots / "runtime_and_cpu_by_seed.png")
    plt.close(fig)

    result = dict(status="PILOT_DIAGNOSTICS_COMPLETE_NOT_ENSEMBLE_ACCEPTANCE", events_per_mode=5,
                  completed_events=15, integrity_gates_passed=True,
                  full_energy_ledger_verified=False, ensemble_physics_accepted=False,
                  stable_performance_gain_demonstrated=False,
                  scalar_1pct_intervals=int(tests.entire_interval_within_1pct.sum()),
                  scalar_holm_rejections=int((tests.holm_sign_flip_p<.05).sum()),
                  radio_holm_rejections=int((radiot.holm_sign_flip_p<.05).sum()),
                  high_energy_phase_at_snapshot=snapshot["phase"])
    write(out / "analysis_summary.json", result)
    lines = ["# 自适应双端：100TeV Fe56，5 vs 5 vs 5 试跑验收", "",
             "已检查单CUDA、原双端20线程、自适应双端20线程各5例，共15例。",
             "本报告为小样本诊断完成，不是大样本物理等价或稳定加速验收通过。", "",
             "## 完整性与物理设置", "",
             "15例均完整结束；初级、能量、方向、cut、薄化、磁场、天线/时窗、",
             "PROPOSAL及aux哈希一致；同一冻结二进制。CPU端线程数20；单CUDA保持1线程调度。",
             "队列/定点溢出、原生反解失败均为0；双端CUDA提交数等于回收数。",
             "逐例核对native_selection_replay数量等于CPU完成数。另有少量几何回退与Epair包络指定末态完成，",
             "均按原有明确允许的路径处理并单独计数，不允许未知原因通过。详细计数见integrity.json。",
             "双端立即执行的指定回放也计入旧字段cpu_generic_fallbacks，而单CUDA的延迟回放不计入它。",
             "因此该字段不能直接用于比较通用回退数量；本次不修改程序，只在分析中区分即时/延迟及回退原因。", "",
             "## 时间与资源", "", "| 模式 | 平均总耗时(s) | 中位总耗时(s) | 中位shower耗时(s) | 平均逻辑核数 | 最大RSS(GiB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        tm = times[times["mode"]==mode]
        lines.append(f"| {LABELS[mode]} | {tm.process_s.mean():.3f} | {tm.process_s.median():.3f} | "
                     f"{tm.shower_s.median():.3f} | {tm.average_logical_cores.mean():.2f} | {tm.peak_rss_gib.max():.2f} |")
    for rt in runtime:
        if rt["metric"] == "process_s":
            lines += ["", f"自适应相对{LABELS[rt['reference']]}：中位总耗时变化"
                       f"{rt['median_time_change_percent']:+.2f}%；按共同seed配对bootstrap的95%探索区间"
                       f"[{rt['median_change_bootstrap_low']:+.2f}%, {rt['median_change_bootstrap_high']:+.2f}%]。"]
    lines += ["", "首例单CUDA109.6s明显影响均值；不再采用前两个种子的25.7%降幅作为完整5例结论。",
              "动态调度会改变随机历史分配和shower树，相同seed不等于同一工作量。",
              "5例无法把调度收益与事件波动充分分离；当前没有证明达到稳定5%净加速目标。",
              "资源采样是进程CPU时间和设备级GPU采样，不是kernel时间线；本轮15个独立N=1进程也不能证明N>1无泄漏。", "",
              "## Profile、标量与波形", "",
              "全部纵向组分、muon parent profile及12项标量已提取；并保存完整数组供后续复核。",
              "标量比较使用同seed配对bootstrap；配对符号翻转为探索性检验，依赖零假设下差值对称。",
              "每组仅5例，两侧符号翻转检验最小p值为2/32=0.0625，本身不足以在5%水平拒绝零假设。",
              f"24项均值差bootstrap区间完全位于±1%的仅{result['scalar_1pct_intervals']}项；不因阴影重叠就宣布一致。", "",
              "CoREAS/ZHS逐事件调用原pulse_analysis投影为Ex′/Ey′/Ez′，然后求带符号平均。",
              "阴影为逐点Student-t 95%均值区间；不滤波、不峰值平移、不幅度归一化、不删除首尾点。",
              "横轴仅缩放显示；profile及波形统计保留完整记录，射电仅验证已有400ns时窗内信号。",
              "integrity.json给出首尾各5个采样点的平方场占比，这是边缘诊断，不能证明窗外绝无信号。", "",
              "## 能量账本边界", "",
              "三个模式的GPU能量账本均标记complete_coverage=false，accepted=false不能当成完整守恒验收。",
              "该账本不覆盖全部强子贡献；纯EM fixture的闭合不能代替Fe完整shower账本。",
              "本轮既不把这部分差额直接归因于自适应丢粒子，也不宣称完整能量守恒已验证。", "",
              "## 图片", "",
              "- [全部纵向组分](plots/longitudinal_all_components_mean.png)",
              "- [Muon parent profiles](plots/muon_production_parent_mean.png)",
              "- [标量分布](plots/shower_scalar_distributions.png)",
              "- [CoREAS平均时域波形](plots/CoREAS_overview_prime.png)",
              "- [ZHS平均时域波形](plots/ZHS_overview_prime.png)",
              "- [单事例时间分布](plots/runtime_distribution.png)",
              "- [逐种子耗时与CPU占用](plots/runtime_and_cpu_by_seed.png)", "",
              f"快照时高能队列阶段：{snapshot['phase']}。未结束的100PeV事件不纳入此报告；原测试队列不改动。", ""]
    (out / "VALIDATION_REPORT_CN.md").write_text("\n".join(lines))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
