#!/usr/bin/env python3
"""Attribute CPU/CUDA radio differences along the true ``r_perp`` profile.

This script complements the ordinary radial comparison.  It separates three
questions which otherwise look similar in a plot:

* Is there an overall CPU/CUDA radio-amplitude normalization difference?
* Does a difference remain after controlling for shower ``Xmax`` and total EM
  profile size?
* Does the *shape* of the lateral radio profile differ after every shower is
  normalized to its value at a fixed reference radius?

The input radio table has already been aggregated over antennas at equal
``r_perp`` within one shower.  Every bootstrap therefore resamples complete
showers independently within the CPU and CUDA ensembles.  Pulse-width points
are marked acceptance-eligible only when both backends have the requested
minimum fraction of valid showers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CPU_ALIASES = {"legacy_proposal": "proposal", "cpu": "proposal"}
ALGORITHMS = ("CoREAS", "ZHS")
COLORS = {"CoREAS": "#2166ac", "ZHS": "#b2182b"}
AMPLITUDE = "geomagnetic_amplitude_geomean_V_per_m"
WIDTH = "pulse_width_mean_ns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observables", type=Path, required=True)
    parser.add_argument("--radial-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-radius-m", type=float, default=100.0)
    parser.add_argument("--minimum-width-valid-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def canonicalize_backend(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["backend_canonical"] = result.backend.replace(CPU_ALIASES)
    unexpected = set(result.backend_canonical.unique()) - {"proposal", "cuda"}
    if unexpected:
        raise ValueError(f"unexpected backends: {sorted(unexpected)}")
    return result


def canonicalize_radial_coordinate(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the physical shower-axis distance as ``r_perp_m``.

    Older radial-analysis products called this column ``radius_m`` even
    though the stored value was already the perpendicular distance to the
    shower axis.  Current products retain that compatibility alias and also
    write the explicit ``r_perp_m`` column.  Accept both schemas, but reject a
    product that contains inconsistent values under the two names.
    """

    result = frame.copy()
    has_explicit = "r_perp_m" in result.columns
    has_legacy = "radius_m" in result.columns
    if not has_explicit and not has_legacy:
        return result
    if has_explicit and has_legacy:
        explicit = result["r_perp_m"].to_numpy(dtype=np.float64)
        legacy = result["radius_m"].to_numpy(dtype=np.float64)
        if not np.allclose(explicit, legacy, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "radial feature table has inconsistent r_perp_m and "
                "legacy radius_m coordinates"
            )
    elif has_legacy:
        result["r_perp_m"] = result["radius_m"].to_numpy(dtype=np.float64)
    return result


def load_joined(observables_path: Path, radial_path: Path) -> pd.DataFrame:
    observables = canonicalize_backend(pd.read_csv(observables_path))
    radial = canonicalize_radial_coordinate(
        canonicalize_backend(pd.read_csv(radial_path))
    )
    required_observables = {
        "backend_canonical",
        "shower",
        "profile_xmax_charged_gcm2",
        "profile_em_integral",
    }
    required_radial = {
        "backend_canonical",
        "algorithm",
        "shower",
        "r_perp_m",
        AMPLITUDE,
        WIDTH,
        "pulse_width_filter_pass_count",
    }
    missing_observables = required_observables - set(observables.columns)
    missing_radial = required_radial - set(radial.columns)
    if missing_observables or missing_radial:
        raise ValueError(
            "missing columns: observables="
            f"{sorted(missing_observables)}, radial={sorted(missing_radial)}"
        )
    joined = radial.merge(
        observables[
            [
                "backend_canonical",
                "shower",
                "profile_xmax_charged_gcm2",
                "profile_em_integral",
            ]
        ],
        on=["backend_canonical", "shower"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(radial):
        raise ValueError(
            f"only {len(joined)} of {len(radial)} radial rows joined to showers"
        )
    joined["log_profile_em_integral"] = np.log(
        joined.profile_em_integral.to_numpy(dtype=np.float64)
    )
    return joined


def backend_arrays(
    frame: pd.DataFrame,
    response: str,
    covariates: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for backend in ("proposal", "cuda"):
        group = frame[frame.backend_canonical == backend]
        y = group[response].to_numpy(dtype=np.float64)
        if response.startswith("log_"):
            finite = np.isfinite(y)
        else:
            finite = np.isfinite(y) & (y > 0.0)
        x_columns = []
        for covariate in covariates:
            values = group[covariate].to_numpy(dtype=np.float64)
            finite &= np.isfinite(values)
            x_columns.append(values)
        x = (
            np.column_stack(x_columns)
            if x_columns
            else np.empty((len(group), 0), dtype=np.float64)
        )
        result[backend] = (y[finite], x[finite])
    return result


def fit_adjusted_ratio(
    cpu_y: np.ndarray,
    cpu_x: np.ndarray,
    cuda_y: np.ndarray,
    cuda_x: np.ndarray,
) -> float:
    y = np.concatenate((cpu_y, cuda_y))
    backend = np.concatenate((np.zeros(cpu_y.size), np.ones(cuda_y.size)))
    x = np.row_stack((cpu_x, cuda_x))
    columns = [np.ones(y.size), backend]
    if x.shape[1]:
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0, ddof=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("a radial attribution covariate has zero variance")
        columns.extend(((x - mean) / scale).T)
    design = np.column_stack(columns)
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0][1]
    return float(math.exp(coefficient))


def ratio_with_bootstrap(
    frame: pd.DataFrame,
    response: str,
    covariates: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    arrays = backend_arrays(frame, response, covariates)
    cpu_y, cpu_x = arrays["proposal"]
    cuda_y, cuda_x = arrays["cuda"]
    if min(cpu_y.size, cuda_y.size) < 2:
        return {
            "ratio": None,
            "bootstrap_95pct": [None, None],
            "proposal_count": int(cpu_y.size),
            "cuda_count": int(cuda_y.size),
        }
    # All modeled responses are logarithms.  Raw amplitudes and widths are
    # converted before this function; exp(beta_backend) is therefore a ratio.
    estimate = fit_adjusted_ratio(cpu_y, cpu_x, cuda_y, cuda_x)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        cpu_indices = rng.integers(0, cpu_y.size, size=cpu_y.size)
        cuda_indices = rng.integers(0, cuda_y.size, size=cuda_y.size)
        samples[index] = fit_adjusted_ratio(
            cpu_y[cpu_indices],
            cpu_x[cpu_indices],
            cuda_y[cuda_indices],
            cuda_x[cuda_indices],
        )
    return {
        "ratio": estimate,
        "bootstrap_95pct": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "proposal_count": int(cpu_y.size),
        "cuda_count": int(cuda_y.size),
    }


def add_log_response(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    result = frame.copy()
    values = result[source].to_numpy(dtype=np.float64)
    result[target] = np.where(values > 0.0, np.log(values), np.nan)
    return result


def add_reference_normalization(
    frame: pd.DataFrame,
    reference_radius: float,
) -> pd.DataFrame:
    reference = frame[np.isclose(frame.r_perp_m, reference_radius, atol=1.0e-6)][
        ["backend_canonical", "algorithm", "shower", AMPLITUDE]
    ].rename(columns={AMPLITUDE: "reference_amplitude"})
    if reference.empty:
        available = sorted(frame.r_perp_m.unique().tolist())
        raise ValueError(
            f"reference radius {reference_radius:g} m is unavailable; "
            f"available radii are {available}"
        )
    reference = reference.drop_duplicates(
        ["backend_canonical", "algorithm", "shower"]
    )
    result = frame.merge(
        reference,
        on=["backend_canonical", "algorithm", "shower"],
        how="left",
        validate="many_to_one",
    )
    result["log_amplitude_shape"] = np.log(
        result[AMPLITUDE] / result.reference_amplitude
    )
    return result


def analyze_radius(
    frame: pd.DataFrame,
    repetitions: int,
    seed: int,
    minimum_width_valid_fraction: float,
) -> dict[str, Any]:
    radial = add_log_response(frame, AMPLITUDE, "log_amplitude")
    radial = add_log_response(radial, WIDTH, "log_width")
    raw_amplitude = ratio_with_bootstrap(
        radial, "log_amplitude", (), repetitions, seed
    )
    adjusted_amplitude = ratio_with_bootstrap(
        radial,
        "log_amplitude",
        ("profile_xmax_charged_gcm2", "log_profile_em_integral"),
        repetitions,
        seed + 1,
    )
    shape = ratio_with_bootstrap(
        radial,
        "log_amplitude_shape",
        ("profile_xmax_charged_gcm2",),
        repetitions,
        seed + 2,
    )
    valid_width = (
        np.isfinite(radial[WIDTH].to_numpy(dtype=np.float64))
        & (radial[WIDTH].to_numpy(dtype=np.float64) > 0.0)
        & (radial.pulse_width_filter_pass_count.to_numpy(dtype=np.float64) > 0.0)
    )
    width_frame = radial.loc[valid_width].copy()
    width = ratio_with_bootstrap(
        width_frame, "log_width", (), repetitions, seed + 3
    )
    total = radial.groupby("backend_canonical").shower.nunique().to_dict()
    valid = width_frame.groupby("backend_canonical").shower.nunique().to_dict()
    fractions = {
        backend: float(valid.get(backend, 0) / total.get(backend, 1))
        for backend in ("proposal", "cuda")
    }
    width["valid_fraction"] = fractions
    width["acceptance_eligible"] = bool(
        min(fractions.values()) >= minimum_width_valid_fraction
    )
    return {
        "amplitude_raw": raw_amplitude,
        "amplitude_adjusted_xmax_em": adjusted_amplitude,
        "amplitude_shape_at_reference_adjusted_xmax": shape,
        "width_raw": width,
    }


def flatten_rows(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm, data in report["algorithms"].items():
        for item in data["radii"]:
            row: dict[str, Any] = {
                "algorithm": algorithm,
                "r_perp_m": item["r_perp_m"],
            }
            for name in (
                "amplitude_raw",
                "amplitude_adjusted_xmax_em",
                "amplitude_shape_at_reference_adjusted_xmax",
                "width_raw",
            ):
                result = item[name]
                row[f"{name}_ratio"] = result["ratio"]
                row[f"{name}_ci_low"] = result["bootstrap_95pct"][0]
                row[f"{name}_ci_high"] = result["bootstrap_95pct"][1]
                row[f"{name}_proposal_count"] = result["proposal_count"]
                row[f"{name}_cuda_count"] = result["cuda_count"]
            width = item["width_raw"]
            row["width_proposal_valid_fraction"] = width["valid_fraction"][
                "proposal"
            ]
            row["width_cuda_valid_fraction"] = width["valid_fraction"]["cuda"]
            row["width_acceptance_eligible"] = width["acceptance_eligible"]
            rows.append(row)
    return pd.DataFrame(rows)


def finite_interval_rows(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}_ratio", f"{prefix}_ci_low", f"{prefix}_ci_high"]
    return frame[np.isfinite(frame[columns]).all(axis=1)].copy()


def draw_ratio(
    axis: plt.Axes,
    frame: pd.DataFrame,
    prefix: str,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    selected = finite_interval_rows(frame, prefix)
    x = selected.r_perp_m.to_numpy(dtype=np.float64)
    ratio = selected[f"{prefix}_ratio"].to_numpy(dtype=np.float64)
    low = selected[f"{prefix}_ci_low"].to_numpy(dtype=np.float64)
    high = selected[f"{prefix}_ci_high"].to_numpy(dtype=np.float64)
    axis.plot(
        x,
        ratio,
        color=color,
        marker=marker,
        markersize=3.6,
        linewidth=1.6,
        linestyle=linestyle,
        label=label,
    )
    axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)


def plot_report(rows: pd.DataFrame, output: Path, reference_radius: float) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(14.6, 8.7), sharex=True, constrained_layout=True
    )
    for row_index, algorithm in enumerate(ALGORITHMS):
        data = rows[rows.algorithm == algorithm].sort_values("r_perp_m")
        color = COLORS[algorithm]
        raw_axis, shape_axis, width_axis = axes[row_index]
        draw_ratio(
            raw_axis,
            data,
            "amplitude_raw",
            label="Raw amplitude",
            color=color,
            marker="o",
        )
        draw_ratio(
            raw_axis,
            data,
            "amplitude_adjusted_xmax_em",
            label=r"Adjusted for $X_{\max}$ + EM size",
            color="#4d4d4d",
            marker="s",
            linestyle="--",
        )
        draw_ratio(
            shape_axis,
            data,
            "amplitude_shape_at_reference_adjusted_xmax",
            label=f"Shape / {reference_radius:g} m",
            color=color,
            marker="o",
        )
        eligible = data[data.width_acceptance_eligible.astype(bool)]
        diagnostic = data[~data.width_acceptance_eligible.astype(bool)]
        draw_ratio(
            width_axis,
            eligible,
            "width_raw",
            label=r"Width ($\geq 80\%$ valid)",
            color=color,
            marker="o",
        )
        finite_diagnostic = diagnostic[
            np.isfinite(diagnostic.width_raw_ratio)
        ]
        width_axis.scatter(
            finite_diagnostic.r_perp_m,
            finite_diagnostic.width_raw_ratio,
            facecolors="none",
            edgecolors="#8c8c8c",
            s=25,
            linewidths=1.0,
            label="Weak-signal diagnostic",
        )
        for axis in (raw_axis, shape_axis, width_axis):
            axis.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
            axis.grid(alpha=0.22, which="both")
            axis.set_xscale("log")
            axis.set_ylabel(f"{algorithm}: CUDA / CPU")
            axis.legend(frameon=False, fontsize=8)
        raw_axis.set_title("Geomagnetic amplitude")
        shape_axis.set_title("Per-shower lateral shape")
        width_axis.set_title("Pulse width and validity gate")
    for axis in axes[-1]:
        axis.set_xlabel(r"True shower-axis distance $r_\perp$ [m]")
    figure.suptitle(
        "CPU/CUDA radio radial attribution (whole-shower bootstrap)",
        fontsize=14,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def range_summary(rows: pd.DataFrame, column: str) -> list[float | None]:
    values = rows[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return [None, None]
    return [float(np.min(values)), float(np.max(values))]


def markdown_report(report: dict[str, Any], rows: pd.DataFrame) -> str:
    reference = report["reference_radius_m"]
    threshold = report["minimum_width_valid_fraction"]
    lines = [
        "# 射电径向 profile 差异归因",
        "",
        "本报告使用真实 shower 轴距离 $r_\\perp$。每个半径先在单个 shower 内合并天线，",
        "统计重采样单位始终是完整 shower，因此不会把同一 shower 的多个天线误当成独立事例。",
        "",
        "- `Raw amplitude`：两套独立 shower 样本的原始几何均值比。",
        "- `Adjusted`：在 log-linear 模型中控制带电粒子 $X_{\\max}$ 和总 EM profile 积分。",
        f"- `Shape / {reference:g} m`：每个 shower 先除以自身 {reference:g} m 振幅，再控制 $X_{{\\max}}$；它检验径向形状而非整体归一化。",
        f"- 宽度主比较要求 CPU 和 CUDA 都至少有 {threshold:.0%} shower 通过宽度筛选；空心灰点仅作弱信号诊断。",
        "",
        "## 数值摘要",
        "",
        "| 算法 | 原始振幅比范围 | 控制 Xmax+EM 后范围 | 100 m 归一化形状比范围 | 宽度主比较半径数/总数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        subset = rows[rows.algorithm == algorithm]
        raw = range_summary(subset, "amplitude_raw_ratio")
        adjusted = range_summary(subset, "amplitude_adjusted_xmax_em_ratio")
        shape = range_summary(
            subset, "amplitude_shape_at_reference_adjusted_xmax_ratio"
        )
        eligible = int(subset.width_acceptance_eligible.sum())
        lines.append(
            f"| {algorithm} | {raw[0]:.4f}–{raw[1]:.4f} | "
            f"{adjusted[0]:.4f}–{adjusted[1]:.4f} | "
            f"{shape[0]:.4f}–{shape[1]:.4f} | {eligible}/{len(subset)} |"
        )
    lines.extend(
        [
            "",
            "## 判读原则",
            "",
            "若原始振幅比随半径近似保持常数，而控制 shower development 或做单 shower 参考半径归一化后更接近 1，",
            "则主要差异是两个有限 shower 样本的整体射电规模，而不是 CUDA 射电公式改变了 lateral shape。",
            "远距离宽度在有效率下降、结果接近记录窗上限时不进入物理一致性的主判据。",
            "最终结论还需与 identical-track CPU/CUDA radio oracle 以及新的 100 TeV 500 vs 500 数据共同给出。",
            "",
            "![Radio radial attribution](radio_radial_attribution.png)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if not 0.0 < args.minimum_width_valid_fraction <= 1.0:
        raise ValueError("--minimum-width-valid-fraction must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    joined = load_joined(args.observables, args.radial_features)
    joined = add_reference_normalization(joined, args.reference_radius_m)
    report: dict[str, Any] = {
        "inputs": {
            "observables": str(args.observables.resolve()),
            "radial_features": str(args.radial_features.resolve()),
        },
        "statistical_unit": "independent shower",
        "reference_radius_m": args.reference_radius_m,
        "minimum_width_valid_fraction": args.minimum_width_valid_fraction,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "algorithms": {},
    }
    seed_index = 0
    for algorithm in ALGORITHMS:
        algorithm_frame = joined[joined.algorithm == algorithm]
        radii: list[dict[str, Any]] = []
        for radius, radius_frame in algorithm_frame.groupby("r_perp_m", sort=True):
            result = analyze_radius(
                radius_frame,
                args.bootstrap_repetitions,
                args.seed + seed_index * 100,
                args.minimum_width_valid_fraction,
            )
            result["r_perp_m"] = float(radius)
            radii.append(result)
            seed_index += 1
        report["algorithms"][algorithm] = {"radii": radii}
    rows = flatten_rows(report)
    rows.to_csv(args.output / "radio_radial_attribution.csv", index=False)
    with (args.output / "radio_radial_attribution.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_report(
        rows,
        args.output / "radio_radial_attribution.png",
        args.reference_radius_m,
    )
    (args.output / "radio_radial_attribution.md").write_text(
        markdown_report(report, rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
