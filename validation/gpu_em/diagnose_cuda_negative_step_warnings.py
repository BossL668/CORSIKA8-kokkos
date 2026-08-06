#!/usr/bin/env python3
"""Test whether scalar PROPOSAL negative-step warnings explain ground-EM tails."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402


EVENT_MARKER = "EM thinning is configured with threshold="
NEGATIVE_STEP = re.compile(
    r"negative step length of l=(-?[0-9.eE+\-]+) m"
)
GROUND_METRICS = (
    "ground_em_weighted_count",
    "ground_em_kinetic_energy_GeV",
)


def quadratic_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Match the three-bin peak interpolation used by compare_ensembles."""

    if x.size == 0:
        return 0.0, 0.0
    maximum = int(np.argmax(y))
    x_peak = float(x[maximum])
    y_peak = float(y[maximum])
    if maximum == 0 or maximum + 1 == x.size:
        return x_peak, y_peak
    left, center, right = (
        float(y[maximum - 1]),
        float(y[maximum]),
        float(y[maximum + 1]),
    )
    denominator = left - 2.0 * center + right
    spacing_left = float(x[maximum] - x[maximum - 1])
    spacing_right = float(x[maximum + 1] - x[maximum])
    if (
        denominator >= 0.0
        or not math.isclose(spacing_left, spacing_right, rel_tol=1.0e-6)
    ):
        return x_peak, y_peak
    offset = 0.5 * (left - right) / denominator
    if abs(offset) > 1.0:
        return x_peak, y_peak
    x_peak += offset * spacing_left
    y_peak = center - 0.25 * (left - right) * offset
    return x_peak, y_peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--wait-for-campaign-completion", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def parse_warning_sections(log_path: Path, events: int) -> list[dict[str, Any]]:
    sections: list[list[float]] = []
    current: list[float] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if EVENT_MARKER in line:
            current = []
            sections.append(current)
        match = NEGATIVE_STEP.search(line)
        if match is not None:
            if current is None:
                raise ValueError(
                    f"negative-step warning precedes the first event marker in {log_path}"
                )
            current.append(abs(float(match.group(1))))
    if len(sections) != events:
        raise ValueError(
            f"event-marker count differs in {log_path}: {len(sections)} versus {events}"
        )
    return [
        {
            "negative_step_warnings": len(values),
            "negative_step_abs_length_sum_m": float(np.sum(values)),
            "negative_step_abs_length_max_m": (
                float(np.max(values)) if values else 0.0
            ),
        }
        for values in sections
    ]


def batch_observables(cuda_root: Path, events: int) -> pd.DataFrame:
    gpu = read_mapping(cuda_root / "gpu_em/summary.yaml")
    scalar_rows = []
    for shower in range(events):
        record = gpu.get(f"shower_{shower}")
        if not isinstance(record, dict) or record.get("complete") is not True:
            raise ValueError(f"incomplete CUDA shower_{shower} in {cuda_root}")
        statistics = record.get("statistics")
        if not isinstance(statistics, dict):
            raise ValueError(f"missing CUDA statistics for shower_{shower}")
        ledger = statistics.get("energy_ledger")
        if not isinstance(ledger, dict):
            raise ValueError(f"missing energy ledger for shower_{shower}")
        scalar_rows.append(
            {
                "shower": shower,
                "scalar_em_steps": int(ledger.get("scalar_em_steps", -1)),
                "cpu_particle_steps": int(
                    statistics.get("cpu_particle_steps", -1)
                ),
            }
        )
    scalar = pd.DataFrame(scalar_rows).set_index("shower")
    if (scalar < 0).any().any():
        raise ValueError(f"negative or absent scalar-step counter in {cuda_root}")

    particles = pd.read_parquet(
        cuda_root / "particles/particles.parquet",
        columns=["shower", "pdg", "kinetic_energy", "weight"],
    )
    em = particles[particles.pdg.isin((-11, 11, 22))].copy()
    em["weighted_kinetic"] = em.kinetic_energy * em.weight
    ground = em.groupby("shower", sort=True).agg(
        ground_em_weighted_count=("weight", "sum"),
        ground_em_kinetic_energy_GeV=("weighted_kinetic", "sum"),
    )

    profiles = pd.read_parquet(cuda_root / "profile/profile.parquet")
    profile_rows = []
    for shower, group in profiles.groupby("shower", sort=True):
        coordinate = group.X.to_numpy(dtype=np.float64)
        charged = group.charged.to_numpy(dtype=np.float64)
        em_profile = (
            group.photon.to_numpy(dtype=np.float64)
            + group.electron.to_numpy(dtype=np.float64)
            + group.positron.to_numpy(dtype=np.float64)
        )
        xmax, _ = quadratic_peak(coordinate, charged)
        profile_rows.append(
            {
                "shower": int(shower),
                "profile_xmax_charged_gcm2": xmax,
                "profile_em_integral": float(np.trapz(em_profile, coordinate)),
            }
        )
    profile = pd.DataFrame(profile_rows).set_index("shower")
    result = scalar.join(ground, how="inner").join(profile, how="inner")
    expected_index = pd.Index(range(events), name="shower")
    if not result.index.equals(expected_index):
        raise ValueError(f"CUDA batch observables do not cover 0..{events - 1}")
    return result.reset_index()


def discover_complete_batches(campaign_root: Path) -> list[dict[str, Any]]:
    manifest = read_mapping(campaign_root / "campaign_manifest.json")
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ValueError("campaign manifest has no batch list")
    complete = [item for item in batches if item.get("status") == "complete"]
    if len(complete) != len(batches):
        raise ValueError("campaign manifest contains a non-complete recorded batch")
    return complete


def wait_for_campaign_completion(campaign_root: Path, poll_seconds: float) -> None:
    if poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    while True:
        manifest = read_mapping(campaign_root / "campaign_manifest.json")
        status = str(manifest.get("status", "missing"))
        if status == "complete":
            return
        if status == "failed":
            raise RuntimeError("CUDA campaign failed before diagnostic execution")
        time.sleep(poll_seconds)


def collect_campaign(campaign_root: Path) -> pd.DataFrame:
    frames = []
    for batch_index, record in enumerate(discover_complete_batches(campaign_root)):
        events = int(record["events"])
        seed = int(record["seed"])
        cuda_root = Path(record["cuda_output"])
        log_path = campaign_root / f"batch_{batch_index:03d}.runner.log"
        warnings = pd.DataFrame(parse_warning_sections(log_path, events))
        values = batch_observables(cuda_root, events)
        values = pd.concat((values, warnings), axis=1)
        values.insert(0, "batch", batch_index)
        values.insert(2, "seed", seed + values.shower)
        frames.append(values)
    if not frames:
        raise ValueError("campaign has no complete CUDA batches")
    result = pd.concat(frames, ignore_index=True)
    result["negative_step_warning_rate"] = (
        result.negative_step_warnings / result.scalar_em_steps
    )
    numeric = result.select_dtypes(include=(np.number,)).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("negative-step diagnostic contains non-finite values")
    if result.seed.duplicated().any() or (result.scalar_em_steps <= 0).any():
        raise ValueError("duplicate seeds or non-positive scalar EM step counts")
    return result


def standardized(values: np.ndarray) -> np.ndarray:
    spread = float(np.std(values, ddof=0))
    if not math.isfinite(spread) or spread <= 0.0:
        raise ValueError("cannot standardize a constant or non-finite predictor")
    return (values - float(np.mean(values))) / spread


def fit_models(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    response = np.log(frame[metric].to_numpy(dtype=np.float64))
    xmax = standardized(frame.profile_xmax_charged_gcm2.to_numpy(dtype=np.float64))
    em_size = standardized(np.log(frame.profile_em_integral.to_numpy(dtype=np.float64)))
    warning_rate = standardized(
        frame.negative_step_warning_rate.to_numpy(dtype=np.float64)
    )
    base = np.column_stack((np.ones(len(frame)), xmax, em_size))
    extended = np.column_stack((base, warning_rate))
    base_coefficients = np.linalg.lstsq(base, response, rcond=None)[0]
    extended_coefficients = np.linalg.lstsq(extended, response, rcond=None)[0]
    base_residual = response - base @ base_coefficients
    extended_residual = response - extended @ extended_coefficients
    warning_coefficient = float(extended_coefficients[-1])
    return {
        "warning_ratio_per_one_sd": math.exp(warning_coefficient),
        "warning_log_coefficient": warning_coefficient,
        "base_residual_sum_squares": float(np.sum(base_residual**2)),
        "extended_residual_sum_squares": float(np.sum(extended_residual**2)),
        "partial_r_squared": float(
            1.0
            - np.sum(extended_residual**2)
            / max(np.sum(base_residual**2), np.finfo(np.float64).tiny)
        ),
        "base_residual": base_residual,
        "warning_rate_z": warning_rate,
    }


def bootstrap_warning_ratio(
    frame: pd.DataFrame, metric: str, repetitions: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(frame), size=len(frame))
        try:
            samples[index] = fit_models(frame.iloc[selected], metric)[
                "warning_ratio_per_one_sd"
            ]
        except ValueError:
            samples[index] = np.nan
    samples = samples[np.isfinite(samples)]
    if samples.size < 0.95 * repetitions:
        raise ValueError("too many degenerate bootstrap warning-rate samples")
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def make_report(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "events": len(frame),
        "negative_step_warnings": int(frame.negative_step_warnings.sum()),
        "scalar_em_steps": int(frame.scalar_em_steps.sum()),
        "aggregate_warning_rate": float(
            frame.negative_step_warnings.sum() / frame.scalar_em_steps.sum()
        ),
        "warning_absolute_length_m": {
            "maximum_per_event": float(
                frame.negative_step_abs_length_max_m.max()
            ),
            "sum": float(frame.negative_step_abs_length_sum_m.sum()),
        },
        "models": {},
    }
    for index, metric in enumerate(GROUND_METRICS):
        model = fit_models(frame, metric)
        report["models"][metric] = {
            key: value
            for key, value in model.items()
            if key not in ("base_residual", "warning_rate_z")
        }
        report["models"][metric]["bootstrap_95pct"] = bootstrap_warning_ratio(
            frame, metric, repetitions, seed + index * 10000
        )
        correlation = stats.spearmanr(
            frame.negative_step_warning_rate, frame[metric]
        )
        report["models"][metric]["unadjusted_spearman"] = {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        }
    return report


def plot_diagnostics(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    axes[0, 0].scatter(
        frame.scalar_em_steps,
        frame.negative_step_warnings,
        s=14,
        alpha=0.45,
        color="#6a3d9a",
    )
    axes[0, 0].set_xlabel("Scalar EM steps")
    axes[0, 0].set_ylabel("Negative-step warnings")
    axes[0, 0].set_title("Raw warning count follows scalar workload")
    axes[0, 0].grid(alpha=0.22)

    axes[0, 1].scatter(
        frame.profile_xmax_charged_gcm2,
        frame.negative_step_warning_rate,
        s=14,
        alpha=0.45,
        color="#1b9e77",
    )
    axes[0, 1].set_xlabel(r"Charged $X_{\max}$ [g cm$^{-2}$]")
    axes[0, 1].set_ylabel("Warnings / scalar EM step")
    axes[0, 1].set_title("Normalized warning rate versus shower depth")
    axes[0, 1].grid(alpha=0.22)

    for axis, metric, label in zip(
        axes[1],
        GROUND_METRICS,
        ("Ground EM weighted count", "Ground EM kinetic energy"),
    ):
        model = fit_models(frame, metric)
        x = model["warning_rate_z"]
        y = model["base_residual"]
        axis.scatter(x, y, s=14, alpha=0.45, color="#d95f02")
        coefficient = np.polyfit(x, y, 1)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        axis.plot(grid, np.polyval(coefficient, grid), color="#252525")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_xlabel("Normalized warning rate [standard deviations]")
        axis.set_ylabel("Log residual after Xmax + EM-size adjustment")
        axis.set_title(label)
        axis.grid(alpha=0.22)
    figure.suptitle(
        "CUDA scalar-PROPOSAL negative-step diagnostic", fontsize=14
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CUDA negative-step warning diagnostic",
        "",
        f"Events: {report['events']}",
        "",
        f"Warnings / scalar EM step: {report['aggregate_warning_rate']:.6g}",
        "",
        "The warning-rate coefficient is evaluated only within CUDA showers after",
        "adjusting log ground observables for charged Xmax and total EM profile size.",
        "Association is not proof of causality; a persistent adjusted effect requires",
        "a controlled diagnostic rerun before changing transport physics.",
        "",
        "| Observable | Ratio per +1 SD warning rate | Shower bootstrap 95% | Partial R2 |",
        "|---|---:|---:|---:|",
    ]
    for metric, value in report["models"].items():
        interval = value["bootstrap_95pct"]
        lines.append(
            f"| `{metric}` | {value['warning_ratio_per_one_sd']:.5f} | "
            f"[{interval[0]:.5f}, {interval[1]:.5f}] | "
            f"{value['partial_r_squared']:.5f} |"
        )
    lines.extend(["", "![Diagnostic](negative_step_warning_diagnostics.png)", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    campaign_root = args.campaign_root.resolve()
    if args.wait_for_campaign_completion:
        wait_for_campaign_completion(campaign_root, args.poll_seconds)
    frame = collect_campaign(campaign_root)
    if args.expected_events is not None and len(frame) != args.expected_events:
        raise ValueError(
            f"expected {args.expected_events} events, observed {len(frame)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = make_report(frame, args.bootstrap_repetitions, args.seed)
    frame.to_csv(args.output / "negative_step_warning_per_shower.csv", index=False)
    (args.output / "negative_step_warning_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_diagnostics(frame, args.output / "negative_step_warning_diagnostics.png")
    (args.output / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "events": len(frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
