#!/usr/bin/env python3
"""Summarize CPU/CUDA pure-EM longitudinal-profile attribution tests."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


@dataclass(frozen=True)
class Campaign:
    label: str
    root: Path


SCALAR_METRICS = (
    ("profile_em_integral", "Total EM profile integral", True),
    (
        "profile_electron_positron_integral",
        "Electron + positron profile integral",
        True,
    ),
    ("profile_photon_integral", "Photon profile integral", True),
    ("energy_deposit_sum_GeV", "Deposited energy", True),
    ("ground_em_weighted_count", "Ground EM weighted count", False),
    ("ground_em_total_energy_GeV", "Ground EM total energy", False),
    ("energy_closure_fraction", "Energy-closure fraction", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        metavar="LABEL=ROOT",
        help="Finalized comparison root; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_campaign(value: str) -> Campaign:
    label, separator, root = value.partition("=")
    if not separator or not label.strip() or not root.strip():
        raise ValueError("--campaign must have the form LABEL=ROOT")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"campaign root does not exist: {path}")
    return Campaign(label.strip(), path)


def unique_directory(root: Path, pattern: str) -> Path:
    candidates = sorted(path for path in root.glob(pattern) if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {pattern!r} directory below {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def hc3_backend_effect(
    frame: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    selected = frame.loc[
        frame["backend"].isin(("proposal", "cuda")),
        ["backend", "profile_xmax_charged_gcm2", target],
    ].dropna()
    if set(selected["backend"]) != {"proposal", "cuda"}:
        raise ValueError("per-shower table does not contain both backends")
    values = selected[target].to_numpy(dtype=np.float64)
    if np.any(values <= 0.0):
        raise ValueError(f"{target} must be strictly positive")
    y = np.log(values)
    backend = (selected["backend"] == "cuda").to_numpy(dtype=np.float64)
    xmax = selected["profile_xmax_charged_gcm2"].to_numpy(
        dtype=np.float64
    )
    scale = float(np.std(xmax, ddof=1))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid Xmax scale")
    z = (xmax - float(np.mean(xmax))) / scale
    design = np.column_stack(
        (np.ones(y.size), backend, z, z * z, z * z * z)
    )
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ y
    residual = y - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    corrected = residual / (1.0 - leverage)
    meat = (design.T * (corrected * corrected)) @ design
    covariance = inverse @ meat @ inverse
    standard_error = float(math.sqrt(covariance[1, 1]))
    coefficient = float(coefficients[1])
    degrees_of_freedom = y.size - design.shape[1]
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    low = coefficient - critical * standard_error
    high = coefficient + critical * standard_error
    pvalue = float(
        2.0
        * stats.t.sf(
            abs(coefficient / standard_error),
            degrees_of_freedom,
        )
    )
    residual_sum = float(np.sum(residual * residual))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "relative_effect": float(np.expm1(coefficient)),
        "relative_effect_95pct_low": float(np.expm1(low)),
        "relative_effect_95pct_high": float(np.expm1(high)),
        "pvalue": pvalue,
        "r_squared": 1.0 - residual_sum / total_sum,
    }


def mean_difference(
    proposal: np.ndarray,
    cuda: np.ndarray,
) -> dict[str, float]:
    proposal = proposal[np.isfinite(proposal)]
    cuda = cuda[np.isfinite(cuda)]
    if proposal.size < 2 or cuda.size < 2:
        raise ValueError("each backend needs at least two finite scalar values")
    proposal_mean = float(np.mean(proposal))
    cuda_mean = float(np.mean(cuda))
    if not math.isfinite(proposal_mean) or proposal_mean == 0.0:
        raise ValueError("proposal scalar mean must be finite and nonzero")
    relative = (cuda_mean - proposal_mean) / proposal_mean
    relative_se = (
        math.sqrt(
            float(np.var(proposal, ddof=1)) / proposal.size
            + float(np.var(cuda, ddof=1)) / cuda.size
        )
        / proposal_mean
    )
    return {
        "proposal_mean": proposal_mean,
        "cuda_mean": cuda_mean,
        "relative_difference": relative,
        "relative_standard_error": relative_se,
        "relative_95pct_low": relative - 1.96 * relative_se,
        "relative_95pct_high": relative + 1.96 * relative_se,
        "z_score": relative / relative_se,
    }


def events_for_half_width(
    proposal: np.ndarray,
    cuda: np.ndarray,
    half_width: float,
) -> int:
    proposal_cv = float(np.std(proposal, ddof=1) / np.mean(proposal))
    cuda_cv = float(np.std(cuda, ddof=1) / np.mean(cuda))
    required = (
        1.96
        * math.sqrt(proposal_cv * proposal_cv + cuda_cv * cuda_cv)
        / half_width
    ) ** 2
    return int(math.ceil(required))


def summarize_campaign(campaign: Campaign) -> dict[str, Any]:
    ensemble = unique_directory(campaign.root, "ensemble_comparison_*")
    fixed = unique_directory(campaign.root, "fixed_depth_profile_diagnosis_*")
    aligned = unique_directory(
        campaign.root, "post_xmax_em_profile_analysis_*"
    )
    per_shower = pd.read_csv(ensemble / "per_shower_observables.csv")
    diagnosis = read_json(fixed / "diagnosis.json")
    post_xmax = read_json(aligned / "post_xmax_em_summary.json")
    proposal = per_shower.loc[per_shower["backend"] == "proposal"]
    cuda = per_shower.loc[per_shower["backend"] == "cuda"]
    if proposal.empty or cuda.empty:
        raise ValueError(f"missing backend rows in {ensemble}")
    em_proposal = proposal["profile_em_integral"].to_numpy(
        dtype=np.float64
    )
    em_cuda = cuda["profile_em_integral"].to_numpy(dtype=np.float64)
    xmax_proposal = proposal["profile_xmax_charged_gcm2"].to_numpy(
        dtype=np.float64
    )
    xmax_cuda = cuda["profile_xmax_charged_gcm2"].to_numpy(
        dtype=np.float64
    )
    raw_integral = mean_difference(em_proposal, em_cuda)
    xmax = mean_difference(xmax_proposal, xmax_cuda)
    fixed_em = diagnosis["components"]["em"]
    aligned_em = post_xmax["components"]["em"]
    aligned_tail = aligned_em["tail_integral"]
    normalized_tail = aligned_em["normalized_tail_integral"]
    fixed_difference = float(fixed_em["fixed_depth_peak_relative_shift"])
    fixed_z = float(fixed_em["fixed_depth_peak_z_score"])
    if fixed_z == 0.0:
        if fixed_difference != 0.0:
            raise ValueError(
                "nonzero fixed-depth difference has a zero z score"
            )
        fixed_relative_se = 0.0
    else:
        fixed_relative_se = abs(fixed_difference / fixed_z)
    scalar_metrics: dict[str, Any] = {}
    for column, label, adjust_for_xmax in SCALAR_METRICS:
        if column not in per_shower.columns:
            raise ValueError(f"missing required pure-EM scalar column: {column}")
        raw = mean_difference(
            proposal[column].to_numpy(dtype=np.float64),
            cuda[column].to_numpy(dtype=np.float64),
        )
        scalar_metrics[column] = {
            "label": label,
            "raw": raw,
            "xmax_adjusted": (
                hc3_backend_effect(per_shower, column)
                if adjust_for_xmax
                else None
            ),
        }
    return {
        "label": campaign.label,
        "root": str(campaign.root),
        "events": {
            "proposal": int(proposal.shape[0]),
            "cuda": int(cuda.shape[0]),
        },
        "raw_em_integral": raw_integral,
        "xmax": {
            **xmax,
            "absolute_difference_gcm2": float(
                np.mean(xmax_cuda) - np.mean(xmax_proposal)
            ),
        },
        "fixed_depth_em_peak": {
            "relative_difference": fixed_difference,
            "relative_standard_error": fixed_relative_se,
            "z_score": fixed_z,
            "relative_95pct_low": (
                fixed_difference - 1.96 * fixed_relative_se
            ),
            "relative_95pct_high": (
                fixed_difference + 1.96 * fixed_relative_se
            ),
        },
        "per_shower_em_maximum": {
            "relative_difference": float(
                fixed_em["per_shower_maximum_relative_shift"]
            ),
            "z_score": float(fixed_em["per_shower_maximum_z_score"]),
        },
        "xmax_adjusted_em_integral": hc3_backend_effect(
            per_shower, "profile_em_integral"
        ),
        "aligned_em_tail": {
            "relative_difference": float(
                aligned_tail["signed_relative_mean_shift"]
            ),
            "relative_95pct_low": float(
                aligned_tail[
                    "bootstrap_signed_relative_mean_shift_95pct"
                ][0]
            ),
            "relative_95pct_high": float(
                aligned_tail[
                    "bootstrap_signed_relative_mean_shift_95pct"
                ][2]
            ),
            "pvalue": float(aligned_tail["welch_ttest_pvalue"]),
        },
        "aligned_normalized_shape": {
            "relative_difference": float(
                normalized_tail["signed_relative_mean_shift"]
            ),
            "relative_95pct_low": float(
                normalized_tail[
                    "bootstrap_signed_relative_mean_shift_95pct"
                ][0]
            ),
            "relative_95pct_high": float(
                normalized_tail[
                    "bootstrap_signed_relative_mean_shift_95pct"
                ][2]
            ),
        },
        "global_curve_tests": diagnosis["permutation"],
        "em_scalar_metrics": scalar_metrics,
        "events_per_backend_for_95pct_half_width": {
            "2_percent": events_for_half_width(
                em_proposal, em_cuda, 0.02
            ),
            "1_percent": events_for_half_width(
                em_proposal, em_cuda, 0.01
            ),
            "0.5_percent": events_for_half_width(
                em_proposal, em_cuda, 0.005
            ),
        },
    }


def interval_row(
    label: str,
    metric: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign": label,
        "metric": metric,
        "relative_difference": source["relative_difference"],
        "relative_95pct_low": source["relative_95pct_low"],
        "relative_95pct_high": source["relative_95pct_high"],
    }


def plot_summary(rows: pd.DataFrame, output: Path) -> None:
    campaigns = list(dict.fromkeys(rows["campaign"]))
    metrics = list(dict.fromkeys(rows["metric"]))
    colors = {
        "Raw fixed-depth peak": "#d62728",
        "Raw EM integral": "#ff7f0e",
        "Xmax-adjusted EM integral": "#2ca02c",
        "Xmax-aligned EM tail": "#1f77b4",
    }
    figure, axes = plt.subplots(
        1,
        len(campaigns),
        figsize=(6.0 * len(campaigns), 4.8),
        sharey=True,
        squeeze=False,
    )
    for axis, campaign in zip(axes[0], campaigns):
        selected = rows.loc[rows["campaign"] == campaign]
        positions = np.arange(selected.shape[0])
        centers = 100.0 * selected["relative_difference"].to_numpy(float)
        low = 100.0 * selected["relative_95pct_low"].to_numpy(float)
        high = 100.0 * selected["relative_95pct_high"].to_numpy(float)
        errors = np.vstack((centers - low, high - centers))
        for index, (_, record) in enumerate(selected.iterrows()):
            axis.errorbar(
                centers[index],
                positions[index],
                xerr=errors[:, index:index + 1],
                fmt="o",
                capsize=3,
                color=colors.get(record["metric"], "0.3"),
            )
        axis.axvline(0.0, color="0.25", linewidth=1.0)
        axis.axvspan(-1.0, 1.0, color="#2ca02c", alpha=0.08)
        axis.set_title(campaign)
        axis.set_xlabel("CUDA / CPU - 1 [%]")
        axis.set_yticks(positions, selected["metric"])
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Pure-EM longitudinal-profile attribution")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_scalar_summary(rows: pd.DataFrame, output: Path) -> None:
    campaigns = list(dict.fromkeys(rows["campaign"]))
    figure, axes = plt.subplots(
        1,
        len(campaigns),
        figsize=(7.2 * len(campaigns), 5.8),
        sharey=True,
        squeeze=False,
    )
    for axis, campaign in zip(axes[0], campaigns):
        selected = rows.loc[rows["campaign"] == campaign].reset_index(drop=True)
        positions = np.arange(selected.shape[0])
        centers = 100.0 * selected["raw_relative_difference"].to_numpy(float)
        low = 100.0 * selected["raw_relative_95pct_low"].to_numpy(float)
        high = 100.0 * selected["raw_relative_95pct_high"].to_numpy(float)
        axis.errorbar(
            centers,
            positions,
            xerr=np.vstack((centers - low, high - centers)),
            fmt="o",
            capsize=3,
            color="#d62728",
            label="Raw",
        )
        adjusted = selected["xmax_adjusted_relative_effect"].to_numpy(float)
        finite = np.isfinite(adjusted)
        if np.any(finite):
            adjusted_low = 100.0 * selected.loc[
                finite, "xmax_adjusted_95pct_low"
            ].to_numpy(float)
            adjusted_high = 100.0 * selected.loc[
                finite, "xmax_adjusted_95pct_high"
            ].to_numpy(float)
            adjusted_center = 100.0 * adjusted[finite]
            axis.errorbar(
                adjusted_center,
                positions[finite] + 0.14,
                xerr=np.vstack(
                    (
                        adjusted_center - adjusted_low,
                        adjusted_high - adjusted_center,
                    )
                ),
                fmt="s",
                capsize=3,
                color="#1f77b4",
                label="Xmax-adjusted",
            )
        axis.axvline(0.0, color="0.25", linewidth=1.0)
        axis.axvspan(-1.0, 1.0, color="#2ca02c", alpha=0.08)
        axis.set_title(campaign)
        axis.set_xlabel("CUDA / CPU - 1 [%]")
        axis.set_yticks(positions, selected["metric_label"])
        axis.grid(axis="x", alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Pure-EM scalar-observable attribution")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Pure-EM CPU/CUDA longitudinal-profile attribution",
        "",
        (
            "All uncertainty intervals use complete showers as the "
            "independent resampling unit. Bins and particles within one "
            "shower are not treated as independent observations."
        ),
        "",
        "| Campaign | Events CPU/CUDA | Fixed peak | Raw EM integral | "
        "Xmax-adjusted integral | Aligned EM tail | Normalized tail shape |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        events = item["events"]
        fixed = item["fixed_depth_em_peak"]
        raw = item["raw_em_integral"]
        adjusted = item["xmax_adjusted_em_integral"]
        aligned = item["aligned_em_tail"]
        shape = item["aligned_normalized_shape"]
        lines.append(
            "| {label} | {p}/{c} | {fixed:+.3f}% | {raw:+.3f}% | "
            "{adjusted:+.3f}% | {aligned:+.3f}% | {shape:+.3f}% |".format(
                label=item["label"],
                p=events["proposal"],
                c=events["cuda"],
                fixed=100.0 * fixed["relative_difference"],
                raw=100.0 * raw["relative_difference"],
                adjusted=100.0 * adjusted["relative_effect"],
                aligned=100.0 * aligned["relative_difference"],
                shape=100.0 * shape["relative_difference"],
            )
        )
    for item in summaries:
        lines.extend(
            [
                "",
                f"## Scalar EM observables: {item['label']}",
                "",
                "| Observable | Raw CUDA/CPU - 1 | Raw 95% interval | "
                "Xmax-adjusted CUDA/CPU - 1 | Adjusted 95% interval |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in item["em_scalar_metrics"].values():
            raw = metric["raw"]
            adjusted = metric["xmax_adjusted"]
            if adjusted is None:
                adjusted_effect = "--"
                adjusted_interval = "--"
            else:
                adjusted_effect = (
                    f"{100.0 * adjusted['relative_effect']:+.3f}%"
                )
                adjusted_interval = (
                    f"[{100.0 * adjusted['relative_effect_95pct_low']:+.3f}%, "
                    f"{100.0 * adjusted['relative_effect_95pct_high']:+.3f}%]"
                )
            lines.append(
                "| {label} | {effect:+.3f}% | [{low:+.3f}%, {high:+.3f}%] | "
                "{adjusted} | {interval} |".format(
                    label=metric["label"],
                    effect=100.0 * raw["relative_difference"],
                    low=100.0 * raw["relative_95pct_low"],
                    high=100.0 * raw["relative_95pct_high"],
                    adjusted=adjusted_effect,
                    interval=adjusted_interval,
                )
            )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- A raw fixed-depth difference that disappears after Xmax "
            "adjustment/alignment is attributed primarily to longitudinal "
            "location sampling.",
            "- A persistent aligned normalized-shape difference indicates "
            "a transport-shape residual and requires process-level review.",
            "- A persistent aligned absolute difference with matching "
            "normalized shape indicates a cascade-normalization residual.",
            "- Statistical compatibility does not prove strict 1% "
            "equivalence unless the corresponding interval is sufficiently "
            "narrow.",
            "",
            "See `pure_em_profile_attribution.json` for all confidence "
            "intervals, p-values, global curve tests, and sample-size "
            "estimates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    campaigns = [parse_campaign(value) for value in args.campaign]
    labels = [campaign.label for campaign in campaigns]
    if len(labels) != len(set(labels)):
        raise ValueError("campaign labels must be unique")
    summaries = [summarize_campaign(campaign) for campaign in campaigns]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaigns": summaries,
        "independent_unit": "complete shower",
    }
    (output / "pure_em_profile_attribution.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for item in summaries:
        label = item["label"]
        rows.extend(
            (
                interval_row(
                    label,
                    "Raw fixed-depth peak",
                    item["fixed_depth_em_peak"],
                ),
                interval_row(
                    label,
                    "Raw EM integral",
                    item["raw_em_integral"],
                ),
                {
                    "campaign": label,
                    "metric": "Xmax-adjusted EM integral",
                    "relative_difference": item[
                        "xmax_adjusted_em_integral"
                    ]["relative_effect"],
                    "relative_95pct_low": item[
                        "xmax_adjusted_em_integral"
                    ]["relative_effect_95pct_low"],
                    "relative_95pct_high": item[
                        "xmax_adjusted_em_integral"
                    ]["relative_effect_95pct_high"],
                },
                interval_row(
                    label,
                    "Xmax-aligned EM tail",
                    item["aligned_em_tail"],
                ),
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pure_em_profile_intervals.csv", index=False)
    plot_summary(frame, output / "pure_em_profile_attribution.png")
    scalar_rows: list[dict[str, Any]] = []
    for item in summaries:
        for column, metric in item["em_scalar_metrics"].items():
            raw = metric["raw"]
            adjusted = metric["xmax_adjusted"]
            scalar_rows.append(
                {
                    "campaign": item["label"],
                    "metric": column,
                    "metric_label": metric["label"],
                    "raw_relative_difference": raw["relative_difference"],
                    "raw_relative_standard_error": raw[
                        "relative_standard_error"
                    ],
                    "raw_relative_95pct_low": raw["relative_95pct_low"],
                    "raw_relative_95pct_high": raw["relative_95pct_high"],
                    "raw_z_score": raw["z_score"],
                    "xmax_adjusted_relative_effect": (
                        math.nan
                        if adjusted is None
                        else adjusted["relative_effect"]
                    ),
                    "xmax_adjusted_95pct_low": (
                        math.nan
                        if adjusted is None
                        else adjusted["relative_effect_95pct_low"]
                    ),
                    "xmax_adjusted_95pct_high": (
                        math.nan
                        if adjusted is None
                        else adjusted["relative_effect_95pct_high"]
                    ),
                    "xmax_adjusted_pvalue": (
                        math.nan if adjusted is None else adjusted["pvalue"]
                    ),
                }
            )
    scalar_frame = pd.DataFrame(scalar_rows)
    scalar_frame.to_csv(output / "pure_em_scalar_metrics.csv", index=False)
    plot_scalar_summary(
        scalar_frame,
        output / "pure_em_scalar_metrics.png",
    )
    write_markdown(output / "README.md", summaries)
    print(
        json.dumps(
            {
                "campaigns": labels,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
