#!/usr/bin/env python3
"""Attribute CPU/CUDA ground-EM and radio distribution shifts.

The ordinary ensemble plots compare independent proton showers.  Ground EM
and radio amplitude are both strongly correlated with shower age at the
observation level, so a raw mean ratio can be misleading when the two finite
samples have slightly different Xmax distributions.  This diagnostic reports
both the unadjusted distributions and a transparent log-linear adjustment for
Xmax and total EM profile integral.  Bootstrap resampling always treats a
whole shower as the statistical unit.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


GROUND_METRICS = (
    "ground_em_weighted_count",
    "ground_em_kinetic_energy_GeV",
)
GROUND_COMPONENT_METRICS = (
    "ground_photon_weighted_count",
    "ground_electron_weighted_count",
    "ground_positron_weighted_count",
)
RADIO_METRICS = (
    "geomagnetic_amplitude_geomean_V_per_m",
    "pulse_width_mean_ns",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observables", type=Path, required=True)
    parser.add_argument("--radio-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def finite_positive(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        selected &= np.isfinite(values)
        if column not in (
            "profile_xmax_charged_gcm2",
            "backend_indicator",
        ):
            selected &= values > 0.0
    return frame.loc[selected].copy()


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "geometric_mean": float(stats.gmean(values)),
        "trimmed_mean_10pct": float(stats.trim_mean(values, 0.1)),
        "quantiles": {
            f"q{int(round(q * 100)):02d}": float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.99)
        },
    }


def design_matrix(
    frame: pd.DataFrame,
    covariates: tuple[str, ...],
) -> np.ndarray:
    columns = [
        np.ones(len(frame), dtype=np.float64),
        frame["backend_indicator"].to_numpy(dtype=np.float64),
    ]
    for covariate in covariates:
        values = frame[covariate].to_numpy(dtype=np.float64)
        if covariate.startswith("log_"):
            values = np.log(values)
        values = (values - np.mean(values)) / np.std(values, ddof=0)
        columns.append(values)
    return np.column_stack(columns)


def fit_backend_ratio(
    frame: pd.DataFrame,
    metric: str,
    covariates: tuple[str, ...],
) -> float:
    matrix = design_matrix(frame, covariates)
    response = np.log(frame[metric].to_numpy(dtype=np.float64))
    coefficient = np.linalg.lstsq(matrix, response, rcond=None)[0][1]
    return float(math.exp(coefficient))


def bootstrap_backend_ratio(
    frame: pd.DataFrame,
    metric: str,
    covariates: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> tuple[float, list[float]]:
    ratio = fit_backend_ratio(frame, metric, covariates)
    groups = {
        backend: group.reset_index(drop=True)
        for backend, group in frame.groupby("backend_canonical", sort=True)
    }
    if set(groups) != {"cuda", "proposal"}:
        raise ValueError("expected canonical proposal and cuda samples")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        resampled = []
        for group in groups.values():
            indices = rng.integers(0, len(group), size=len(group))
            resampled.append(group.iloc[indices])
        samples[index] = fit_backend_ratio(
            pd.concat(resampled, ignore_index=True), metric, covariates
        )
    return ratio, [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def reference_band_report(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, Any]]:
    boundaries = np.quantile(reference, (0.05, 0.16, 0.5, 0.84))
    lower = np.concatenate(([-np.inf], boundaries))
    upper = np.concatenate((boundaries, [np.inf]))
    labels = ("<q05", "q05-q16", "q16-q50", "q50-q84", ">=q84")
    rows: list[dict[str, Any]] = []
    for label, lo, hi in zip(labels, lower, upper):
        row: dict[str, Any] = {
            "band": label,
            "reference_lower": None if not np.isfinite(lo) else float(lo),
            "reference_upper": None if not np.isfinite(hi) else float(hi),
        }
        for name, values in (("proposal", reference), ("cuda", candidate)):
            selected = (values >= lo) & (values < hi)
            row[f"{name}_event_fraction"] = float(np.mean(selected))
            row[f"{name}_sum_fraction"] = float(
                np.sum(values[selected]) / np.sum(values)
            )
        rows.append(row)
    return rows


def empirical_ks_location(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """Locate the largest signed separation of two empirical CDFs."""

    reference = np.sort(np.asarray(reference, dtype=np.float64))
    candidate = np.sort(np.asarray(candidate, dtype=np.float64))
    points = np.unique(np.concatenate((reference, candidate)))
    reference_cdf = np.searchsorted(
        reference, points, side="right"
    ) / reference.size
    candidate_cdf = np.searchsorted(
        candidate, points, side="right"
    ) / candidate.size
    signed = candidate_cdf - reference_cdf
    index = int(np.argmax(np.abs(signed)))
    location = float(points[index])
    reference_quantile = float(
        np.searchsorted(reference, location, side="right")
        / reference.size
    )
    return {
        "value": location,
        "statistic": float(abs(signed[index])),
        "signed_cuda_minus_proposal_cdf": float(signed[index]),
        "proposal_cdf": float(reference_cdf[index]),
        "cuda_cdf": float(candidate_cdf[index]),
        "proposal_quantile_at_location": reference_quantile,
        "inside_proposal_low_5pct": bool(reference_quantile <= 0.05),
        "inside_proposal_low_16pct": bool(reference_quantile <= 0.16),
    }


def low_tail_fraction_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    quantile: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Compare event fractions below a CPU-defined low-tail threshold.

    Every bootstrap repetition resamples complete showers independently in
    the two backends and re-estimates the threshold from the resampled CPU
    ensemble.  This propagates both the finite-sample threshold uncertainty
    and the CUDA low-tail counting uncertainty.
    """

    if not 0.0 < quantile < 0.5:
        raise ValueError("low-tail quantile must be in (0, 0.5)")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    def evaluate(cpu: np.ndarray, cuda: np.ndarray) -> tuple[float, float, float]:
        threshold = float(np.quantile(cpu, quantile))
        cpu_fraction = float(np.mean(cpu <= threshold))
        cuda_fraction = float(np.mean(cuda <= threshold))
        return threshold, cpu_fraction, cuda_fraction

    threshold, cpu_fraction, cuda_fraction = evaluate(reference, candidate)
    rng = np.random.default_rng(seed)
    fraction_differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        cpu = reference[
            rng.integers(0, reference.size, size=reference.size)
        ]
        cuda = candidate[
            rng.integers(0, candidate.size, size=candidate.size)
        ]
        _, bootstrap_cpu, bootstrap_cuda = evaluate(cpu, cuda)
        fraction_differences[index] = bootstrap_cuda - bootstrap_cpu
    return {
        "proposal_quantile": quantile,
        "proposal_threshold": threshold,
        "proposal_event_fraction": cpu_fraction,
        "cuda_event_fraction": cuda_fraction,
        "cuda_minus_proposal_fraction": cuda_fraction - cpu_fraction,
        "bootstrap_shower_95pct": [
            float(np.quantile(fraction_differences, 0.025)),
            float(np.quantile(fraction_differences, 0.975)),
        ],
    }


def bootstrap_quantile_ratio(
    reference: np.ndarray,
    candidate: np.ndarray,
    quantile: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a CUDA/CPU ratio at a fixed distribution quantile.

    This complements the fixed-threshold event-fraction comparison: the latter
    asks whether too many CUDA showers enter the CPU low tail, while this asks
    whether the physical scale of the CUDA low tail is displaced.
    """

    if not 0.0 < quantile < 0.5:
        raise ValueError("tail quantile must be in (0, 0.5)")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.size == 0
        or candidate.size == 0
        or np.any(~np.isfinite(reference))
        or np.any(~np.isfinite(candidate))
        or np.any(reference <= 0.0)
        or np.any(candidate <= 0.0)
    ):
        raise ValueError("quantile-ratio samples must be finite, positive, and non-empty")
    reference_quantile = float(np.quantile(reference, quantile))
    candidate_quantile = float(np.quantile(candidate, quantile))
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        cpu = reference[rng.integers(0, reference.size, size=reference.size)]
        cuda = candidate[rng.integers(0, candidate.size, size=candidate.size)]
        ratios[index] = float(np.quantile(cuda, quantile)) / float(
            np.quantile(cpu, quantile)
        )
    return {
        "quantile": quantile,
        "proposal_value": reference_quantile,
        "cuda_value": candidate_quantile,
        "cuda_over_proposal_ratio": candidate_quantile / reference_quantile,
        "bootstrap_shower_95pct": [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ],
    }


def analyze_metric(
    frame: pd.DataFrame,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    needed = [
        metric,
        "profile_xmax_charged_gcm2",
        "log_profile_em_integral",
        "backend_indicator",
    ]
    selected = finite_positive(frame, needed)
    reference = selected.loc[
        selected.backend_canonical == "proposal", metric
    ].to_numpy(dtype=np.float64)
    candidate = selected.loc[
        selected.backend_canonical == "cuda", metric
    ].to_numpy(dtype=np.float64)
    if reference.size == 0 or candidate.size == 0:
        raise ValueError(f"empty backend sample for {metric}")
    raw_reference = scalar_summary(reference)
    raw_candidate = scalar_summary(candidate)
    ks_result = stats.ks_2samp(reference, candidate)
    cvm_result = stats.cramervonmises_2samp(reference, candidate)
    # SciPy warns when the tabulated asymptotic significance is capped at
    # 0.25 or floored at 0.001.  The capped value is still useful here and is
    # explicitly labelled below; suppress only the presentation warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ad_result = stats.anderson_ksamp((reference, candidate))
    models: dict[str, Any] = {}
    for index, (label, covariates) in enumerate(
        (
            ("unadjusted_log", ()),
            ("adjusted_xmax", ("profile_xmax_charged_gcm2",)),
            (
                "adjusted_xmax_and_em_integral",
                (
                    "profile_xmax_charged_gcm2",
                    "log_profile_em_integral",
                ),
            ),
        )
    ):
        ratio, interval = bootstrap_backend_ratio(
            selected,
            metric,
            covariates,
            repetitions,
            seed + index * 1000,
        )
        models[label] = {
            "cuda_over_proposal_ratio": ratio,
            "bootstrap_shower_95pct": interval,
            "covariates": list(covariates),
        }
    return {
        "metric": metric,
        "proposal": raw_reference,
        "cuda": raw_candidate,
        "raw_ratios": {
            "arithmetic_mean": raw_candidate["mean"] / raw_reference["mean"],
            "median": raw_candidate["median"] / raw_reference["median"],
            "geometric_mean": (
                raw_candidate["geometric_mean"]
                / raw_reference["geometric_mean"]
            ),
            "trimmed_mean_10pct": (
                raw_candidate["trimmed_mean_10pct"]
                / raw_reference["trimmed_mean_10pct"]
            ),
        },
        "tests": {
            "ks": {
                "statistic": float(ks_result.statistic),
                "p_value": float(ks_result.pvalue),
                "location": empirical_ks_location(reference, candidate),
            },
            "welch_log": {
                "statistic": float(
                    stats.ttest_ind(
                        np.log(reference),
                        np.log(candidate),
                        equal_var=False,
                    ).statistic
                ),
                "p_value": float(
                    stats.ttest_ind(
                        np.log(reference),
                        np.log(candidate),
                        equal_var=False,
                    ).pvalue
                ),
            },
            "cramer_von_mises": {
                "statistic": float(cvm_result.statistic),
                "p_value": float(cvm_result.pvalue),
            },
            "anderson_darling_k_sample": {
                "statistic": float(ad_result.statistic),
                "p_value": float(ad_result.pvalue),
                "p_value_note": (
                    "SciPy asymptotic tabulation; values may be capped at "
                    "0.25 or floored at 0.001"
                ),
            },
        },
        "reference_quantile_bands": reference_band_report(
            reference, candidate
        ),
        "low_tail_event_fractions": {
            "q05": low_tail_fraction_comparison(
                reference,
                candidate,
                0.05,
                repetitions,
                seed + 3000,
            ),
            "q16": low_tail_fraction_comparison(
                reference,
                candidate,
                0.16,
                repetitions,
                seed + 4000,
            ),
        },
        "low_tail_quantile_ratios": {
            "q05": bootstrap_quantile_ratio(
                reference,
                candidate,
                0.05,
                repetitions,
                seed + 5000,
            ),
            "q16": bootstrap_quantile_ratio(
                reference,
                candidate,
                0.16,
                repetitions,
                seed + 6000,
            ),
        },
        "models": models,
    }


def canonical_observables(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["backend_canonical"] = frame.backend.replace(
        {"legacy_proposal": "proposal", "cpu": "proposal"}
    )
    frame["backend_indicator"] = (
        frame.backend_canonical == "cuda"
    ).astype(float)
    frame["log_profile_em_integral"] = frame["profile_em_integral"]
    return frame


def canonical_radio(path: Path, observables: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["backend_canonical"] = frame.backend.replace(
        {"legacy_proposal": "proposal", "cpu": "proposal"}
    )
    joined = frame.merge(
        observables[
            [
                "backend_canonical",
                "shower",
                "profile_xmax_charged_gcm2",
                "log_profile_em_integral",
            ]
        ],
        on=["backend_canonical", "shower"],
        how="inner",
        validate="many_to_one",
    )
    joined["backend_indicator"] = (
        joined.backend_canonical == "cuda"
    ).astype(float)
    return joined


def plot_attribution(
    observables: pd.DataFrame,
    radio: pd.DataFrame,
    output: Path,
) -> None:
    panels = [
        (
            observables,
            "ground_em_weighted_count",
            "Ground EM weighted count",
            None,
        ),
        (
            observables,
            "ground_em_kinetic_energy_GeV",
            "Ground EM kinetic energy [GeV]",
            None,
        ),
        (
            radio[radio.algorithm == "CoREAS"],
            "geomagnetic_amplitude_geomean_V_per_m",
            "CoREAS geomagnetic amplitude [V/m]",
            "CoREAS",
        ),
        (
            radio[radio.algorithm == "ZHS"],
            "geomagnetic_amplitude_geomean_V_per_m",
            "ZHS geomagnetic amplitude [V/m]",
            "ZHS",
        ),
    ]
    colors = {"proposal": "#2166ac", "cuda": "#b2182b"}
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), constrained_layout=True)
    for axis, (frame, metric, ylabel, _algorithm) in zip(axes.flat, panels):
        selected = finite_positive(
            frame,
            [metric, "profile_xmax_charged_gcm2", "backend_indicator"],
        )
        for backend, label in (("proposal", "CPU PROPOSAL"), ("cuda", "CUDA")):
            group = selected[selected.backend_canonical == backend]
            x = group.profile_xmax_charged_gcm2.to_numpy(dtype=np.float64)
            y = group[metric].to_numpy(dtype=np.float64)
            axis.scatter(
                x,
                y,
                s=10,
                alpha=0.18,
                color=colors[backend],
                linewidths=0,
            )
            coefficients = np.polyfit(x, np.log(y), 1)
            grid = np.linspace(
                selected.profile_xmax_charged_gcm2.quantile(0.01),
                selected.profile_xmax_charged_gcm2.quantile(0.99),
                200,
            )
            axis.plot(
                grid,
                np.exp(np.polyval(coefficients, grid)),
                color=colors[backend],
                linewidth=2,
                label=label,
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"Charged-profile $X_{\max}$ [g cm$^{-2}$]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22, which="both")
        axis.legend(frameon=False)
    figure.suptitle(
        "Ground and radio observables versus shower-development depth",
        fontsize=14,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_ground_tail_attribution(
    observables: pd.DataFrame,
    report: dict[str, Any],
    output: Path,
) -> None:
    """Show where the apparent low-bin difference contributes to the mean.

    The band boundaries are always fixed by the CPU sample.  This avoids the
    misleading comparison obtained when both backends define their own
    quantile edges and therefore have identical event fractions by design.
    """

    labels = ("<q05", "q05–q16", "q16–q50", "q50–q84", "≥q84")
    colors = {"proposal": "#2166ac", "cuda": "#b2182b"}
    metric_labels = {
        "ground_em_weighted_count": "Ground EM weighted count",
        "ground_em_kinetic_energy_GeV": "Ground EM kinetic energy [GeV]",
    }
    figure, axes = plt.subplots(
        2, 3, figsize=(15.0, 8.4), constrained_layout=True
    )
    for row_index, metric in enumerate(GROUND_METRICS):
        selected = finite_positive(
            observables,
            [metric, "profile_xmax_charged_gcm2", "backend_indicator"],
        )
        distribution_axis, event_axis, contribution_axis = axes[row_index]
        for backend, backend_label in (
            ("proposal", "CPU PROPOSAL"),
            ("cuda", "CUDA"),
        ):
            values = np.sort(
                selected.loc[
                    selected.backend_canonical == backend, metric
                ].to_numpy(dtype=np.float64)
            )
            probability = np.arange(1, values.size + 1, dtype=np.float64) / values.size
            distribution_axis.step(
                values,
                probability,
                where="post",
                color=colors[backend],
                linewidth=1.8,
                label=backend_label,
            )
        bands = report["ground"][metric]["reference_quantile_bands"]
        ks_location = report["ground"][metric]["tests"]["ks"]["location"]
        q05 = float(bands[0]["reference_upper"])
        q84 = float(bands[-1]["reference_lower"])
        distribution_axis.axvspan(
            selected[metric].min(), q05, color="#bdbdbd", alpha=0.22
        )
        distribution_axis.axvline(q84, color="#636363", linestyle="--", linewidth=1.0)
        distribution_axis.axvline(
            float(ks_location["value"]),
            color="#6a3d9a",
            linestyle=":",
            linewidth=1.4,
            label=(
                r"max $|\Delta\mathrm{CDF}|$ at CPU q"
                f"{100.0 * ks_location['proposal_quantile_at_location']:.1f}"
            ),
        )
        distribution_axis.set_xscale("log")
        distribution_axis.set_xlabel(metric_labels[metric])
        distribution_axis.set_ylabel("Empirical cumulative probability")
        distribution_axis.set_title("Distribution and CPU q05/q84")
        distribution_axis.grid(alpha=0.22, which="both")
        distribution_axis.legend(frameon=False, fontsize=8)

        positions = np.arange(len(bands), dtype=np.float64)
        width = 0.38
        for axis, key, title in (
            (event_axis, "event_fraction", "Events in CPU-defined bands"),
            (
                contribution_axis,
                "sum_fraction",
                "Contribution to ensemble total",
            ),
        ):
            cpu = np.asarray(
                [band[f"proposal_{key}"] for band in bands], dtype=np.float64
            )
            cuda = np.asarray(
                [band[f"cuda_{key}"] for band in bands], dtype=np.float64
            )
            axis.bar(
                positions - width / 2,
                cpu,
                width,
                color=colors["proposal"],
                label="CPU PROPOSAL",
            )
            axis.bar(
                positions + width / 2,
                cuda,
                width,
                color=colors["cuda"],
                label="CUDA",
            )
            axis.set_xticks(positions, labels, rotation=20)
            axis.set_ylabel("Fraction")
            axis.set_ylim(0.0, max(0.2, 1.12 * float(max(cpu.max(), cuda.max()))))
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.22)
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Ground-EM low-tail attribution using fixed CPU quantile bands",
        fontsize=14,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Ground EM 与射电差异归因诊断",
        "",
        "CPU 与 CUDA 样本是相互独立的质子 shower。本报告以完整 shower 为统计单位，",
        "同时给出原始分布，以及控制带电粒子 profile 的 $X_{\\max}$ 和总 EM profile",
        "积分后的 log-linear 比值。调整结果用于归因，不替代原始物理分布验收。",
        "",
        "## 主要比值",
        "",
        "| 量 | 原始算术均值 CUDA/CPU | 原始几何均值 CUDA/CPU | 控制 Xmax | 控制 Xmax 与 EM 积分 |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered: list[tuple[str, dict[str, Any]]] = []
    for metric, result in report["ground"].items():
        ordered.append((metric, result))
    for algorithm, algorithm_results in report["radio"].items():
        for metric, result in algorithm_results.items():
            ordered.append((f"{algorithm} {metric}", result))
    for label, result in ordered:
        raw = result["raw_ratios"]
        xmax = result["models"]["adjusted_xmax"]
        full = result["models"]["adjusted_xmax_and_em_integral"]
        lines.append(
            f"| `{label}` | {raw['arithmetic_mean']:.4f} | "
            f"{raw['geometric_mean']:.4f} | "
            f"{xmax['cuda_over_proposal_ratio']:.4f} "
            f"[{xmax['bootstrap_shower_95pct'][0]:.4f}, "
            f"{xmax['bootstrap_shower_95pct'][1]:.4f}] | "
            f"{full['cuda_over_proposal_ratio']:.4f} "
            f"[{full['bootstrap_shower_95pct'][0]:.4f}, "
            f"{full['bootstrap_shower_95pct'][1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- ground EM 图最左侧的少数离散 bin 不是均值偏移的主导来源；参考样本的固定分位区间显示，差异主要来自中上段和长尾的样本占比。",
            "- ground EM 数量在控制 $X_{\\max}$ 后接近 1，说明原始偏移主要来自两个有限质子样本在观测面处 shower age 不同。",
            "- photon、electron 和 positron 的ground计数分别进入同一归因模型，用于检查总EM低尾是否由某一种粒子转换或输出口径造成。",
            "- ground EM 动能控制后仍可能有几个百分点的点估计，但 shower-bootstrap 区间用于判断其是否可与统计涨落区分。",
            "- CoREAS 与 ZHS 振幅若在控制 $X_{\\max}$ 和 EM shower size 后同时接近 1，而宽度本来已接近 1，则更支持 shower population 差异，而不是两个独立射电公式同时出现同方向算法偏差。",
            "- 图中回归线仅用于显示相关性；最终结论仍应由新的 500 vs 500 独立验收和 same-track CPU/CUDA radio oracle 共同给出。",
            "",
            "## ground EM 左尾的定量贡献",
            "",
        ]
    )
    for metric in GROUND_METRICS:
        bands = report["ground"][metric]["reference_quantile_bands"]
        ks_location = report["ground"][metric]["tests"]["ks"]["location"]
        tail = report["ground"][metric]["low_tail_event_fractions"]
        tail_quantiles = report["ground"][metric]["low_tail_quantile_ratios"]
        tests = report["ground"][metric]["tests"]
        low = bands[0]
        high = bands[-1]
        lines.append(
            f"- `{metric}`：低于 CPU q05 的事件只贡献 CPU 总量的 "
            f"{100.0 * low['proposal_sum_fraction']:.3f}% 和 CUDA 总量的 "
            f"{100.0 * low['cuda_sum_fraction']:.3f}%；高于 CPU q84 的事件则贡献 "
            f"{100.0 * high['proposal_sum_fraction']:.1f}% 和 "
            f"{100.0 * high['cuda_sum_fraction']:.1f}%。因此图中最左侧 bin "
            "不能解释原始算术均值的主要差异。"
        )
        lines.append(
            f"- `{metric}` 的经验 CDF 最大分离为 {ks_location['statistic']:.4f}，"
            f"发生在 {ks_location['value']:.6g}；该位置对应 CPU 的"
            f"第 {100.0 * ks_location['proposal_quantile_at_location']:.1f} 百分位，"
            f"因此{'属于' if ks_location['inside_proposal_low_16pct'] else '不属于'}"
            "CPU 低16%区域。"
        )
        lines.append(
            f"  - 尾部敏感 Anderson–Darling 两样本统计量为 "
            f"{tests['anderson_darling_k_sample']['statistic']:.4f}，近似 "
            f"p={tests['anderson_darling_k_sample']['p_value']:.4g}；"
            f"Cramér–von Mises 统计量为 "
            f"{tests['cramer_von_mises']['statistic']:.4f}，"
            f"p={tests['cramer_von_mises']['p_value']:.4g}。"
        )
        for label in ("q05", "q16"):
            item = tail[label]
            quantile_item = tail_quantiles[label]
            lines.append(
                f"  - CPU `{label}` 阈值以下：CPU "
                f"{100.0 * item['proposal_event_fraction']:.2f}%，CUDA "
                f"{100.0 * item['cuda_event_fraction']:.2f}%；CUDA−CPU 为 "
                f"{100.0 * item['cuda_minus_proposal_fraction']:+.2f} 个百分点，"
                f"shower-bootstrap 95%区间 "
                f"[{100.0 * item['bootstrap_shower_95pct'][0]:+.2f}, "
                f"{100.0 * item['bootstrap_shower_95pct'][1]:+.2f}] 个百分点。"
            )
            lines.append(
                f"  - `{label}` 数值尺度 CUDA/CPU = "
                f"{quantile_item['cuda_over_proposal_ratio']:.4f}，"
                f"shower-bootstrap 95%区间 "
                f"[{quantile_item['bootstrap_shower_95pct'][0]:.4f}, "
                f"{quantile_item['bootstrap_shower_95pct'][1]:.4f}]。"
            )
    lines.extend(
        [
            "",
            "![Xmax attribution](ground_radio_xmax_attribution.png)",
            "",
            "![Ground EM tail attribution](ground_em_tail_attribution.png)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    observables = canonical_observables(args.observables)
    radio = canonical_radio(args.radio_features, observables)
    report: dict[str, Any] = {
        "inputs": {
            "observables": str(args.observables.resolve()),
            "radio_features": str(args.radio_features.resolve()),
        },
        "statistical_unit": "independent shower",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "xmax_summary": observables.groupby("backend_canonical")[
            "profile_xmax_charged_gcm2"
        ].agg(["count", "mean", "std", "median"]).to_dict(orient="index"),
        "ground": {},
        "radio": {},
    }
    available_ground_metrics = GROUND_METRICS + tuple(
        metric
        for metric in GROUND_COMPONENT_METRICS
        if metric in observables.columns
    )
    for index, metric in enumerate(available_ground_metrics):
        report["ground"][metric] = analyze_metric(
            observables,
            metric,
            args.bootstrap_repetitions,
            args.seed + index * 10000,
        )
    for algorithm_index, algorithm in enumerate(("CoREAS", "ZHS")):
        subset = radio[radio.algorithm == algorithm].copy()
        report["radio"][algorithm] = {}
        for metric_index, metric in enumerate(RADIO_METRICS):
            report["radio"][algorithm][metric] = analyze_metric(
                subset,
                metric,
                args.bootstrap_repetitions,
                args.seed + 100000 + algorithm_index * 20000 + metric_index * 10000,
            )
    with (args.output / "ground_radio_attribution.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_attribution(
        observables,
        radio,
        args.output / "ground_radio_xmax_attribution.png",
    )
    plot_ground_tail_attribution(
        observables,
        report,
        args.output / "ground_em_tail_attribution.png",
    )
    (args.output / "README.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
