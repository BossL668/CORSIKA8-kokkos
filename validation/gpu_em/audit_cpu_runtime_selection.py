#!/usr/bin/env python3
"""Audit runtime-dependent event selection in a scalar CPU shower ensemble."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from analyze_post_xmax_em_profiles import quadratic_peak_x  # noqa: E402


METRICS = (
    "xmax_ep_gcm2",
    "ep_peak",
    "em_peak",
    "ep_integral",
    "em_integral",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-block-size", type=int, default=50)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_801)
    return parser.parse_args()


def load_one_shower(root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame = pq.read_table(
        root / "profile" / "profile.parquet",
        columns=["shower", "X", "photon", "electron", "positron"],
    ).to_pandas().sort_values(["shower", "X"])
    shower_ids = frame["shower"].unique()
    if len(shower_ids) != 1:
        raise ValueError(f"expected exactly one shower in {root}")
    coordinate = frame["X"].to_numpy(dtype=np.float64)
    components = {
        name: frame[name].to_numpy(dtype=np.float64)
        for name in ("photon", "electron", "positron")
    }
    return coordinate, components


def load_runtime_seconds(root: Path) -> float:
    timing = yaml.safe_load(
        (root / "simulation_timing" / "summary.yaml").read_text()
    )
    shower = timing.get("shower_0")
    if not isinstance(shower, dict):
        raise ValueError(f"missing shower_0 timing in {root}")
    if shower.get("closed") is not True or shower.get("status") != "closed":
        raise ValueError(f"incomplete shower timing in {root}")
    runtime = float(shower["wall_time_ms"]) / 1000.0
    if not math.isfinite(runtime) or runtime <= 0.0:
        raise ValueError(f"invalid runtime in {root}")
    return runtime


def bootstrap_shift(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> list[float]:
    reference_indices = rng.integers(
        0, reference.size, size=(repetitions, reference.size)
    )
    candidate_indices = rng.integers(
        0, candidate.size, size=(repetitions, candidate.size)
    )
    shifts = (
        np.mean(candidate[candidate_indices], axis=1)
        / np.mean(reference[reference_indices], axis=1)
        - 1.0
    )
    return [float(value) for value in np.quantile(shifts, (0.025, 0.5, 0.975))]


def compare_groups(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    reference_mean = float(np.mean(reference))
    candidate_mean = float(np.mean(candidate))
    standard_error = math.sqrt(
        float(np.var(reference, ddof=1)) / reference.size
        + float(np.var(candidate, ddof=1)) / candidate.size
    )
    return {
        "reference_count": int(reference.size),
        "candidate_count": int(candidate.size),
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "candidate_over_reference_minus_one": (
            candidate_mean / reference_mean - 1.0
        ),
        "mean_difference_z_score": (
            (candidate_mean - reference_mean) / standard_error
            if standard_error > 0.0
            else 0.0
        ),
        "welch_ttest_pvalue": float(
            stats.ttest_ind(reference, candidate, equal_var=False).pvalue
        ),
        "KS_pvalue": float(stats.ks_2samp(reference, candidate).pvalue),
        "bootstrap_shift_95pct": bootstrap_shift(
            reference, candidate, repetitions, rng
        ),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(left, right)
    spearman = stats.spearmanr(left, right)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
    }


def write_readme(path: Path, report: dict[str, Any]) -> None:
    fixed = report["fixed_first_block_vs_remaining"]
    fastest = report["counterfactual_fastest_block_vs_remaining"]
    correlation_em = report["runtime_correlations"]["em_integral"]
    lines = [
        "# CPU 运行时间选择偏差审计",
        "",
        (
            f"共检查 {report['events']} 个闭合 CPU shower。正式子样本在运行前固定为"
            f"前 {report['fixed_block_size']} 个 shard/连续 seed；运行时间和 profile "
            "observable 不参与选择。"
        ),
        "",
        "## 关键结论",
        "",
        (
            "运行时间与总 EM profile 积分的 Pearson 相关系数为 "
            f"{correlation_em['pearson_r']:.3f}（p={correlation_em['pearson_pvalue']:.3g}）。"
            "因此如果事后选择最快完成的 shower，确实会系统性压低 EM 总量。"
        ),
        (
            f"但固定前 {report['fixed_block_size']} 例相对其余样本的总 EM 积分差为 "
            f"{100*fixed['em_integral']['candidate_over_reference_minus_one']:+.2f}%"
            "（这里 candidate=其余样本，reference=固定前块）；其 bootstrap 区间和 "
            "Welch 检验见下表。该比较直接检验预声明块是否异常。"
        ),
        (
            f"作为反事实，真正最快的 {report['fixed_block_size']} 例相对其余样本的"
            "总 EM 积分差为 "
            f"{100*fastest['em_integral']['candidate_over_reference_minus_one']:+.2f}%"
            "（candidate=其余样本，因此正值表示最快块偏低）。"
        ),
        "",
        "## 固定前块与其余样本",
        "",
        "| 指标 | 其余/固定前块−1 | bootstrap 95% CI | z | Welch p | KS p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = fixed[metric]
        low, _, high = item["bootstrap_shift_95pct"]
        lines.append(
            f"| {metric} | {100*item['candidate_over_reference_minus_one']:+.2f}% | "
            f"[{100*low:+.2f}%, {100*high:+.2f}%] | "
            f"{item['mean_difference_z_score']:+.2f} | "
            f"{item['welch_ttest_pvalue']:.3g} | {item['KS_pvalue']:.3g} |"
        )
    lines.extend(
        [
            "",
            "![选择偏差审计](cpu_runtime_selection_audit.png)",
            "",
            "逐事例数值见 `cpu_runtime_selection_events.csv`，完整统计见 "
            "`cpu_runtime_selection_summary.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.fixed_block_size < 2 or args.resamples < 100:
        raise ValueError("fixed block and resample counts are too small")
    root = args.ensemble_root.resolve()
    output = args.output_dir.resolve()
    shards = sorted(path for path in root.glob("proposal_shard_*") if path.is_dir())
    if len(shards) <= args.fixed_block_size:
        raise ValueError("ensemble is smaller than the requested fixed block")

    coordinate: np.ndarray | None = None
    profiles: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        x, components = load_one_shower(shard)
        if coordinate is None:
            coordinate = x
        elif not np.array_equal(coordinate, x):
            raise ValueError(f"profile grid differs in {shard}")
        ep = components["electron"] + components["positron"]
        em = ep + components["photon"]
        summary = yaml.safe_load((shard / "summary.yaml").read_text())
        rows.append(
            {
                "index": index,
                "shard": shard.name,
                "seed": int(summary["seed"]),
                "runtime_seconds": load_runtime_seconds(shard),
                "xmax_ep_gcm2": quadratic_peak_x(x, ep),
                "ep_peak": float(np.max(ep)),
                "em_peak": float(np.max(em)),
                "ep_integral": float(np.trapz(ep, x)),
                "em_integral": float(np.trapz(em, x)),
            }
        )
        profiles.append(em)
    assert coordinate is not None
    frame = pd.DataFrame(rows)
    seeds = frame["seed"].to_numpy(dtype=np.int64)
    if len(np.unique(seeds)) != len(seeds) or not np.all(np.diff(seeds) == 1):
        raise ValueError("CPU seeds are not unique and contiguous")

    fixed_mask = frame["index"].to_numpy() < args.fixed_block_size
    fastest_indices = np.argsort(frame["runtime_seconds"].to_numpy())[
        : args.fixed_block_size
    ]
    fastest_mask = np.zeros(len(frame), dtype=bool)
    fastest_mask[fastest_indices] = True
    frame["predeclared_fixed_block"] = fixed_mask
    frame["counterfactual_fastest_block"] = fastest_mask

    rng = np.random.default_rng(args.seed)
    fixed_comparisons = {
        metric: compare_groups(
            frame.loc[fixed_mask, metric].to_numpy(dtype=np.float64),
            frame.loc[~fixed_mask, metric].to_numpy(dtype=np.float64),
            args.resamples,
            rng,
        )
        for metric in METRICS
    }
    fastest_comparisons = {
        metric: compare_groups(
            frame.loc[fastest_mask, metric].to_numpy(dtype=np.float64),
            frame.loc[~fastest_mask, metric].to_numpy(dtype=np.float64),
            args.resamples,
            rng,
        )
        for metric in METRICS
    }
    runtime = frame["runtime_seconds"].to_numpy(dtype=np.float64)
    correlations = {
        metric: correlation(runtime, frame[metric].to_numpy(dtype=np.float64))
        for metric in METRICS
    }

    profile_matrix = np.stack(profiles)
    fixed_mean = np.mean(profile_matrix[fixed_mask], axis=0)
    remaining_mean = np.mean(profile_matrix[~fixed_mask], axis=0)
    ratio = np.divide(
        remaining_mean,
        fixed_mean,
        out=np.full_like(fixed_mean, np.nan),
        where=fixed_mean > 0.0,
    )
    active = (fixed_mean + remaining_mean) / 2.0
    active = active >= 1.0e-4 * float(np.max(active))

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    axes[0, 0].scatter(
        frame.loc[~fixed_mask, "runtime_seconds"] / 3600.0,
        frame.loc[~fixed_mask, "em_integral"] / 1.0e8,
        s=16,
        alpha=0.55,
        color="0.55",
        label=f"remaining {len(frame)-args.fixed_block_size}",
    )
    axes[0, 0].scatter(
        frame.loc[fixed_mask, "runtime_seconds"] / 3600.0,
        frame.loc[fixed_mask, "em_integral"] / 1.0e8,
        s=24,
        alpha=0.85,
        color="#1f77b4",
        label=f"predeclared first {args.fixed_block_size}",
    )
    axes[0, 0].set(
        xlabel="single-shower runtime [h]",
        ylabel=r"total EM profile integral [$10^8$ particles g cm$^{-2}$]",
        title=f"Runtime correlation: Pearson r={correlations['em_integral']['pearson_r']:.3f}",
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    bins = np.histogram_bin_edges(frame["em_integral"], bins="auto")
    axes[0, 1].hist(
        frame.loc[~fixed_mask, "em_integral"] / 1.0e8,
        bins=bins / 1.0e8,
        density=True,
        alpha=0.48,
        color="0.55",
        label="remaining",
    )
    axes[0, 1].hist(
        frame.loc[fixed_mask, "em_integral"] / 1.0e8,
        bins=bins / 1.0e8,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#1f77b4",
        label="predeclared first block",
    )
    axes[0, 1].set(
        xlabel=r"total EM profile integral [$10^8$ particles g cm$^{-2}$]",
        ylabel="density",
        title="Predeclared block is not selected by completion time",
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        coordinate[active], fixed_mean[active],
        color="#1f77b4", label="first fixed block",
    )
    axes[1, 0].plot(
        coordinate[active], remaining_mean[active],
        color="#d62728", linestyle="--", label="remaining",
    )
    axes[1, 0].set(
        xlabel=r"slant depth [g cm$^{-2}$]",
        ylabel="mean total EM particles",
        title="Mean longitudinal profile",
        yscale="log",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].axhline(1.0, color="0.35", linewidth=1.0)
    axes[1, 1].plot(coordinate[active], ratio[active], color="#7b2cbf")
    axes[1, 1].set(
        xlabel=r"slant depth [g cm$^{-2}$]",
        ylabel="remaining / first fixed block",
        title="Mean-profile ratio in active bins",
    )
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle("CPU ensemble runtime-selection audit")
    figure.tight_layout()

    report = {
        "source": str(root),
        "events": len(frame),
        "fixed_block_size": args.fixed_block_size,
        "seed_min": int(seeds.min()),
        "seed_max": int(seeds.max()),
        "selection_rule": "predeclared first contiguous shard/seed block",
        "runtime_correlations": correlations,
        "fixed_first_block_vs_remaining": fixed_comparisons,
        "counterfactual_fastest_block_vs_remaining": fastest_comparisons,
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cpu_runtime_selection_events.csv", index=False)
    (output / "cpu_runtime_selection_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure.savefig(output / "cpu_runtime_selection_audit.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    write_readme(output / "README.md", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
