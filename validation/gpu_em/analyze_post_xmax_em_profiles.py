#!/usr/bin/env python3
"""Compare CPU/CUDA electromagnetic profiles after each shower's own Xmax.

The two backends are independent Monte Carlo ensembles.  Each shower is first
shifted to ``Delta X = X - Xmax`` using the quadratic maximum of its e-/e+
profile.  Absolute profiles retain normalization differences, while profiles
divided by their own value at Delta X=0 isolate the post-maximum shape.
"""

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
from scipy import stats  # noqa: E402

from diagnose_longitudinal_mean_difference import (  # noqa: E402
    profile_roots,
    read_ensemble,
)


COMPONENTS = {
    "em": r"total EM ($\gamma+e^-+e^+$)",
    "electron_positron": r"$e^-+e^+$",
    "photon": r"$\gamma$",
    "electron": r"$e^-$",
    "positron": r"$e^+$",
}
COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {
    "proposal": "Original CPU (PROPOSAL)",
    "cuda": "CUDA EM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional analysis manifest containing additional_sources. "
            "Defaults to <ensemble-root>/run_manifest.json; using a separate "
            "file preserves an immutable simulation campaign manifest."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-max-gcm2", type=float, default=500.0)
    parser.add_argument("--tail-metric-max-gcm2", type=float, default=300.0)
    parser.add_argument("--grid-step-gcm2", type=float, default=5.0)
    parser.add_argument(
        "--minimum-coverage-fraction", type=float, default=0.90,
        help=(
            "Minimum fraction of showers that must still be above the "
            "observation surface for ratio plots and the common tail metric."
        ),
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_731)
    return parser.parse_args()


def quadratic_peak_x(x: np.ndarray, y: np.ndarray) -> float:
    """Return the grid maximum refined by a local equal-spacing parabola."""
    maximum = int(np.argmax(y))
    peak = float(x[maximum])
    if maximum == 0 or maximum + 1 == x.size:
        return peak
    left, center, right = map(
        float, (y[maximum - 1], y[maximum], y[maximum + 1])
    )
    denominator = left - 2.0 * center + right
    left_step = float(x[maximum] - x[maximum - 1])
    right_step = float(x[maximum + 1] - x[maximum])
    if (
        denominator >= 0.0
        or not math.isclose(left_step, right_step, rel_tol=1.0e-6)
    ):
        return peak
    offset = 0.5 * (left - right) / denominator
    if abs(offset) <= 1.0:
        peak += offset * left_step
    return peak


def align_matrix(
    coordinate: np.ndarray,
    matrix: np.ndarray,
    xmax: np.ndarray,
    delta_x: np.ndarray,
    last_observed_x: np.ndarray,
) -> np.ndarray:
    aligned = np.stack(
        [
            np.interp(
                delta_x,
                coordinate - xmax[index],
                matrix[index],
                left=np.nan,
                right=np.nan,
            )
            for index in range(matrix.shape[0])
        ]
    )
    # ProfileWriter keeps a common grid and zero-fills bins beyond the fixed
    # observation surface.  Those bins are outside the simulated atmosphere,
    # not measurements of a vanished shower, so exclude them after alignment.
    outside_observation = (
        xmax[:, None] + delta_x[None, :] > last_observed_x[:, None]
    )
    aligned[outside_observation] = np.nan
    return aligned


def mean_sem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.divide(
        np.nansum(matrix, axis=0),
        count,
        out=np.full(matrix.shape[1], np.nan),
        where=count > 0,
    )
    squared_residual = np.where(
        np.isfinite(matrix), (matrix - mean[None, :]) ** 2, 0.0
    )
    standard_deviation = np.sqrt(
        np.divide(
            np.sum(squared_residual, axis=0),
            count - 1,
            out=np.full(matrix.shape[1], np.nan),
            where=count > 1,
        )
    )
    sem = np.divide(
        standard_deviation,
        np.sqrt(count),
        out=np.full_like(mean, np.nan),
        where=count > 1,
    )
    return mean, sem, count


def normalize_at_xmax(matrix: np.ndarray) -> np.ndarray:
    scale = matrix[:, 0]
    valid = np.isfinite(scale) & (scale > 0.0)
    result = np.full_like(matrix, np.nan)
    result[valid] = matrix[valid] / scale[valid, None]
    return result


def bootstrap_curve_ratio(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent-shower bootstrap CI for candidate/reference mean curves."""
    draws = np.full((repetitions, reference.shape[1]), np.nan)
    valid_columns = (
        (np.sum(np.isfinite(reference), axis=0) >= reference.shape[0] // 2)
        & (np.sum(np.isfinite(candidate), axis=0) >= candidate.shape[0] // 2)
    )
    if not np.any(valid_columns):
        return tuple(np.full(reference.shape[1], np.nan) for _ in range(3))
    reference = reference[:, valid_columns]
    candidate = candidate[:, valid_columns]
    batch_size = 200
    for start in range(0, repetitions, batch_size):
        stop = min(repetitions, start + batch_size)
        batch = stop - start
        reference_indices = rng.integers(
            0, reference.shape[0], size=(batch, reference.shape[0])
        )
        candidate_indices = rng.integers(
            0, candidate.shape[0], size=(batch, candidate.shape[0])
        )
        reference_mean = np.nanmean(reference[reference_indices], axis=1)
        candidate_mean = np.nanmean(candidate[candidate_indices], axis=1)
        block = np.full_like(reference_mean, np.nan)
        np.divide(
            candidate_mean,
            reference_mean,
            out=block,
            where=reference_mean > np.finfo(np.float64).tiny,
        )
        draws[start:stop, valid_columns] = block
    result = []
    for quantile in (0.025, 0.5, 0.975):
        values = np.full(draws.shape[1], np.nan)
        values[valid_columns] = np.nanquantile(
            draws[:, valid_columns], quantile, axis=0
        )
        result.append(values)
    return tuple(result)


def bootstrap_relative_mean_shift(
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
    reference_mean = np.mean(reference[reference_indices], axis=1)
    candidate_mean = np.mean(candidate[candidate_indices], axis=1)
    shifts = candidate_mean / reference_mean - 1.0
    return [
        float(value)
        for value in np.quantile(shifts, (0.025, 0.5, 0.975))
    ]


def scalar_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    reference = reference[np.isfinite(reference)]
    candidate = candidate[np.isfinite(candidate)]
    reference_mean = float(np.mean(reference))
    candidate_mean = float(np.mean(candidate))
    standard_error = math.sqrt(
        float(np.var(reference, ddof=1)) / reference.size
        + float(np.var(candidate, ddof=1)) / candidate.size
    )
    return {
        "proposal_count": int(reference.size),
        "cuda_count": int(candidate.size),
        "proposal_mean": reference_mean,
        "cuda_mean": candidate_mean,
        "signed_relative_mean_shift": candidate_mean / reference_mean - 1.0,
        "mean_difference_z_score": (
            (candidate_mean - reference_mean) / standard_error
            if standard_error > 0.0
            else 0.0
        ),
        "welch_ttest_pvalue": float(
            stats.ttest_ind(reference, candidate, equal_var=False).pvalue
        ),
        "KS_pvalue": float(stats.ks_2samp(reference, candidate).pvalue),
        "bootstrap_signed_relative_mean_shift_95pct": (
            bootstrap_relative_mean_shift(
                reference, candidate, repetitions, rng
            )
        ),
    }


def plot_component_grid(
    delta_x: np.ndarray,
    matrices: dict[str, dict[str, np.ndarray]],
    output: Path,
    *,
    normalized: bool,
    coverage: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True)
    for axis, (component, title) in zip(axes.flat, COMPONENTS.items()):
        for backend in ("proposal", "cuda"):
            matrix = matrices[component][backend]
            if normalized:
                matrix = normalize_at_xmax(matrix)
            mean, sem, _ = mean_sem(matrix)
            axis.plot(
                delta_x,
                mean,
                color=COLORS[backend],
                linestyle="--" if backend == "cuda" else "-",
                linewidth=1.7,
                label=LABELS[backend],
            )
            lower = np.maximum(mean - sem, np.finfo(float).tiny)
            upper = np.maximum(mean + sem, np.finfo(float).tiny)
            axis.fill_between(
                delta_x,
                lower,
                upper,
                color=COLORS[backend],
                alpha=0.14,
                linewidth=0.0,
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.set_xlabel(r"$\Delta X=X-X_{\max}^{e^\pm}$ [g cm$^{-2}$]")
        axis.set_ylabel(
            r"$\langle N/N(0)\rangle$"
            if normalized
            else r"mean particle count $\langle N\rangle$"
        )
    coverage_axis = axes.flat[-1]
    for backend in ("proposal", "cuda"):
        coverage_axis.plot(
            delta_x,
            coverage[backend],
            color=COLORS[backend],
            linestyle="--" if backend == "cuda" else "-",
            label=LABELS[backend],
        )
    coverage_axis.set(
        title="Showers still above observation surface",
        xlabel=r"$\Delta X=X-X_{\max}^{e^\pm}$ [g cm$^{-2}$]",
        ylabel="valid showers",
    )
    coverage_axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle(
        "Post-Xmax electromagnetic profile: "
        + ("normalized shower age" if normalized else "absolute mean")
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_ratio_grid(
    delta_x: np.ndarray,
    ratios: dict[str, dict[str, np.ndarray]],
    output: Path,
    coverage: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True)
    active_depths: list[float] = []
    for axis, (component, title) in zip(axes.flat, COMPONENTS.items()):
        record = ratios[component]
        active = record["active"]
        if np.any(active):
            active_depths.append(float(np.max(delta_x[active])))
        axis.axhline(1.0, color="0.35", linewidth=1.0)
        axis.plot(
            delta_x[active],
            record["ratio"][active],
            color=COLORS["cuda"],
            linewidth=1.6,
        )
        axis.fill_between(
            delta_x[active],
            record["low"][active],
            record["high"][active],
            color=COLORS["cuda"],
            alpha=0.20,
            linewidth=0.0,
            label="independent-shower bootstrap 95% CI",
        )
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.set_xlabel(r"$\Delta X=X-X_{\max}^{e^\pm}$ [g cm$^{-2}$]")
        axis.set_ylabel("CUDA mean / CPU mean")
    coverage_axis = axes.flat[-1]
    for backend in ("proposal", "cuda"):
        coverage_axis.plot(
            delta_x, coverage[backend], color=COLORS[backend],
            linestyle="--" if backend == "cuda" else "-",
            label=LABELS[backend],
        )
    coverage_axis.set(
        title="Observation-surface coverage",
        xlabel=r"$\Delta X=X-X_{\max}^{e^\pm}$ [g cm$^{-2}$]",
        ylabel="valid showers",
    )
    coverage_axis.grid(alpha=0.2)
    coverage_axis.legend(frameon=False)
    if active_depths:
        axes.flat[0].set_xlim(0.0, max(active_depths))
    figure.suptitle("Post-Xmax electromagnetic profile mean ratios")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Xmax 后电磁组分 profile 平均比较",
        "",
        "## 方法",
        "",
        (
            f"CPU 与 CUDA 各使用 {report['events']['proposal']} 和 "
            f"{report['events']['cuda']} 个独立 shower。对每个事例用 "
            "$e^-+e^+$ profile 峰值附近三点抛物线求 "
            "$X_{\\max}^{e^\\pm}$，随后插值到共同的 "
            "$\\Delta X=X-X_{\\max}^{e^\\pm}$ 网格。"
        ),
        (
            "绝对均值保留 shower 归一化差异；归一化曲线先将每个事例"
            "除以其在 $\\Delta X=0$ 的数值，用于单独检查 Xmax 后的"
            "发展形状。误差带和 bootstrap 均以 shower 为独立单位，"
            "没有把纵向 bins 当作独立样本。"
        ),
        "",
        "## 主要数值",
        "",
        (
            f"请求的尾部积分上限为 {report['settings']['requested_tail_metric_max_gcm2']:.0f} "
            f"g cm⁻²；按至少 {100*report['settings']['minimum_coverage_fraction']:.0f}% "
            f"事例仍在观测面以上的要求，实际使用 0–"
            f"{report['settings']['effective_tail_metric_max_gcm2']:.0f} "
            "g cm⁻²。"
        ),
        "",
        "| 组分 | 尾部积分 CUDA/CPU−1 | bootstrap 95% CI | z | "
        "归一化尾部面积 CUDA/CPU−1 | shape bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for component, label in COMPONENTS.items():
        entry = report["components"][component]
        integral = entry["tail_integral"]
        shape = entry["normalized_tail_integral"]
        low, _, high = integral[
            "bootstrap_signed_relative_mean_shift_95pct"
        ]
        shape_low, _, shape_high = shape[
            "bootstrap_signed_relative_mean_shift_95pct"
        ]
        lines.append(
            f"| {label} | {100*integral['signed_relative_mean_shift']:+.2f}% | "
            f"[{100*low:+.2f}%, {100*high:+.2f}%] | "
            f"{integral['mean_difference_z_score']:+.2f} | "
            f"{100*shape['signed_relative_mean_shift']:+.2f}% | "
            f"[{100*shape_low:+.2f}%, {100*shape_high:+.2f}%] |"
        )
    lines.extend(
        [
            "",
            "## 如何解释",
            "",
            (
                "如果绝对尾部仍有偏移、而逐事例归一化后的尾部面积和"
                "形状相容，则主要差异来自每个 shower 的峰值归一化或"
                "有限样本涨落；如果归一化曲线也持续偏离 1，则才更像是"
                f"Xmax 后的输运形状差异。当前 "
                f"{report['events']['proposal']}+"
                f"{report['events']['cuda']} 样本的定量结论见上表。"
            ),
            (
                "原始 profile 在固定观测面以后是零填充。本分析把每个 "
                "shower 的最后有效深度之后改为缺失值；图中的 coverage "
                "面板给出每个 $\\Delta X$ 实际参与平均的事例数。覆盖率"
                "不足阈值的深尾部不画 CUDA/CPU 比值，避免把观测面截断"
                "误判为输运差异。"
            ),
            "",
            "## 输出文件",
            "",
            "- `post_xmax_em_absolute_mean_comparison.png`",
            "- `post_xmax_em_normalized_shape_comparison.png`",
            "- `post_xmax_em_mean_ratio.png`",
            "- `post_xmax_em_pointwise.csv`",
            "- `post_xmax_em_scalar_summary.csv`",
            "- `post_xmax_em_summary.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if (
        args.tail_max_gcm2 <= 0.0
        or args.grid_step_gcm2 <= 0.0
        or not 0.0 < args.tail_metric_max_gcm2 <= args.tail_max_gcm2
        or not 0.5 <= args.minimum_coverage_fraction <= 1.0
        or args.resamples < 100
    ):
        raise ValueError("invalid tail grid, fit interval, or resample count")
    root = args.ensemble_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else root / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    roots = {
        backend: profile_roots(root, manifest, backend)
        for backend in ("proposal", "cuda")
    }
    coordinate: np.ndarray | None = None
    profiles: dict[str, dict[str, np.ndarray]] = {}
    xmax: dict[str, np.ndarray] = {}
    last_observed_x: dict[str, np.ndarray] = {}
    for backend in ("proposal", "cuda"):
        backend_coordinate, backend_profiles = read_ensemble(roots[backend])
        if coordinate is None:
            coordinate = backend_coordinate
        elif not np.array_equal(coordinate, backend_coordinate):
            raise ValueError("CPU and CUDA longitudinal grids differ")
        profiles[backend] = backend_profiles
        xmax[backend] = np.asarray(
            [
                quadratic_peak_x(backend_coordinate, shower)
                for shower in backend_profiles["electron_positron"]
            ]
        )
        activity = backend_profiles["charged"] + backend_profiles["photon"]
        last_observed_x[backend] = np.asarray(
            [
                backend_coordinate[np.flatnonzero(shower > 0.0)[-1]]
                for shower in activity
            ]
        )
    assert coordinate is not None
    delta_x = np.arange(
        0.0,
        args.tail_max_gcm2 + 0.5 * args.grid_step_gcm2,
        args.grid_step_gcm2,
    )
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for component in COMPONENTS:
        aligned[component] = {
            backend: align_matrix(
                coordinate,
                profiles[backend][component],
                xmax[backend],
                delta_x,
                last_observed_x[backend],
            )
            for backend in ("proposal", "cuda")
        }

    rng = np.random.default_rng(args.seed)
    coverage = {
        backend: np.sum(
            xmax[backend][:, None] + delta_x[None, :]
            <= last_observed_x[backend][:, None],
            axis=0,
        )
        for backend in ("proposal", "cuda")
    }
    coverage_required = {
        backend: math.ceil(
            args.minimum_coverage_fraction * profiles[backend]["em"].shape[0]
        )
        for backend in ("proposal", "cuda")
    }
    metric_eligible = (
        (coverage["proposal"] >= coverage_required["proposal"])
        & (coverage["cuda"] >= coverage_required["cuda"])
        & (delta_x <= args.tail_metric_max_gcm2)
    )
    if not np.any(metric_eligible):
        raise ValueError("no post-Xmax bin satisfies the coverage requirement")
    effective_metric_max = float(np.max(delta_x[metric_eligible]))
    pointwise_rows: list[dict[str, float | int | str]] = []
    scalar_rows: list[dict[str, float | int | str]] = []
    report: dict[str, Any] = {
        "source": str(root),
        "manifest": str(manifest_path),
        "events": {
            backend: int(profiles[backend]["em"].shape[0])
            for backend in ("proposal", "cuda")
        },
        "settings": {
            "alignment": "quadratic e-/e+ Xmax per shower",
            "tail_max_gcm2": args.tail_max_gcm2,
            "requested_tail_metric_max_gcm2": args.tail_metric_max_gcm2,
            "effective_tail_metric_max_gcm2": effective_metric_max,
            "grid_step_gcm2": args.grid_step_gcm2,
            "minimum_coverage_fraction": args.minimum_coverage_fraction,
            "bootstrap_resamples": args.resamples,
            "seed": args.seed,
        },
        "xmax_electron_positron_gcm2": {
            backend: {
                "mean": float(np.mean(xmax[backend])),
                "standard_deviation": float(np.std(xmax[backend], ddof=1)),
                "minimum": float(np.min(xmax[backend])),
                "maximum": float(np.max(xmax[backend])),
            }
            for backend in ("proposal", "cuda")
        },
        "components": {},
    }
    curve_ratios: dict[str, dict[str, np.ndarray]] = {}
    metric_grid = delta_x <= effective_metric_max
    for component in COMPONENTS:
        reference = aligned[component]["proposal"]
        candidate = aligned[component]["cuda"]
        reference_normalized = normalize_at_xmax(reference)
        candidate_normalized = normalize_at_xmax(candidate)
        reference_mean, reference_sem, reference_count = mean_sem(reference)
        candidate_mean, candidate_sem, candidate_count = mean_sem(candidate)
        low, median, high = bootstrap_curve_ratio(
            reference, candidate, args.resamples, rng
        )
        ratio = np.divide(
            candidate_mean,
            reference_mean,
            out=np.full_like(reference_mean, np.nan),
            where=reference_mean > np.finfo(np.float64).tiny,
        )
        active = (
            (reference_mean >= 1.0e-4 * float(np.nanmax(reference_mean)))
            & (reference_count >= coverage_required["proposal"])
            & (candidate_count >= coverage_required["cuda"])
        )
        curve_ratios[component] = {
            "ratio": ratio,
            "low": low,
            "median": median,
            "high": high,
            "active": active,
        }
        normalized_reference_mean, normalized_reference_sem, _ = mean_sem(
            reference_normalized
        )
        normalized_candidate_mean, normalized_candidate_sem, _ = mean_sem(
            candidate_normalized
        )
        for index, depth in enumerate(delta_x):
            pointwise_rows.append(
                {
                    "component": component,
                    "delta_X_gcm2": depth,
                    "proposal_count": int(reference_count[index]),
                    "cuda_count": int(candidate_count[index]),
                    "proposal_mean": reference_mean[index],
                    "proposal_sem": reference_sem[index],
                    "cuda_mean": candidate_mean[index],
                    "cuda_sem": candidate_sem[index],
                    "cuda_over_proposal": ratio[index],
                    "bootstrap_ratio_95pct_low": low[index],
                    "bootstrap_ratio_median": median[index],
                    "bootstrap_ratio_95pct_high": high[index],
                    "proposal_normalized_mean": normalized_reference_mean[index],
                    "proposal_normalized_sem": normalized_reference_sem[index],
                    "cuda_normalized_mean": normalized_candidate_mean[index],
                    "cuda_normalized_sem": normalized_candidate_sem[index],
                }
            )
        tail_integral = {
            "proposal": np.trapz(reference[:, metric_grid],
                                  delta_x[metric_grid], axis=1),
            "cuda": np.trapz(candidate[:, metric_grid],
                              delta_x[metric_grid], axis=1),
        }
        normalized_tail_integral = {
            "proposal": np.trapz(reference_normalized[:, metric_grid],
                                  delta_x[metric_grid], axis=1),
            "cuda": np.trapz(candidate_normalized[:, metric_grid],
                              delta_x[metric_grid], axis=1),
        }
        component_report = {
            "tail_integral": scalar_comparison(
                tail_integral["proposal"], tail_integral["cuda"],
                args.resamples, rng
            ),
            "normalized_tail_integral": scalar_comparison(
                normalized_tail_integral["proposal"],
                normalized_tail_integral["cuda"], args.resamples, rng
            ),
            "selected_depth_ratios": {},
        }
        for depth in (0.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0):
            if depth > args.tail_max_gcm2:
                continue
            index = int(np.argmin(np.abs(delta_x - depth)))
            finite_ratio = bool(np.isfinite(ratio[index]))
            component_report["selected_depth_ratios"][f"{depth:g}"] = {
                "delta_X_gcm2": float(delta_x[index]),
                "proposal_count": int(reference_count[index]),
                "cuda_count": int(candidate_count[index]),
                "passes_coverage_requirement": bool(active[index]),
                "cuda_over_proposal": (
                    float(ratio[index]) if finite_ratio else None
                ),
                "bootstrap_ratio_95pct": (
                    [float(low[index]), float(high[index])]
                    if finite_ratio and np.isfinite(low[index])
                    and np.isfinite(high[index]) else None
                ),
            }
        report["components"][component] = component_report
        for metric_name in (
            "tail_integral",
            "normalized_tail_integral",
        ):
            scalar_rows.append(
                {
                    "component": component,
                    "metric": metric_name,
                    **component_report[metric_name],
                }
            )

    pd.DataFrame(pointwise_rows).to_csv(
        output / "post_xmax_em_pointwise.csv", index=False
    )
    scalar_frame = pd.DataFrame(scalar_rows)
    scalar_frame["bootstrap_signed_relative_mean_shift_95pct"] = (
        scalar_frame["bootstrap_signed_relative_mean_shift_95pct"].map(
            json.dumps
        )
    )
    scalar_frame.to_csv(output / "post_xmax_em_scalar_summary.csv", index=False)
    with (output / "post_xmax_em_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_component_grid(
        delta_x, aligned,
        output / "post_xmax_em_absolute_mean_comparison.png",
        normalized=False,
        coverage=coverage,
    )
    plot_component_grid(
        delta_x, aligned,
        output / "post_xmax_em_normalized_shape_comparison.png",
        normalized=True,
        coverage=coverage,
    )
    plot_ratio_grid(
        delta_x, curve_ratios, output / "post_xmax_em_mean_ratio.png",
        coverage,
    )
    write_markdown(output / "README.md", report)
    print(
        json.dumps(
            {
                "events": report["events"],
                "output": str(output),
                "total_em_tail_integral": report["components"]["em"][
                    "tail_integral"
                ],
                "total_em_normalized_tail_integral": report["components"][
                    "em"
                ]["normalized_tail_integral"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
