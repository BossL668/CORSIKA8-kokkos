#!/usr/bin/env python3
"""Fit energy scaling laws for per-shower CPU/CUDA observables.

The independent statistical unit is always a complete shower.  Positive
observables are fitted as ensemble-mean power laws, while Xmax is fitted with
the electromagnetic-shower elongation form

    Xmax(E) = Xmax(1 TeV) + D10 log10(E / 1 TeV).

Uncertainties are obtained by resampling showers independently inside every
energy/backend stratum.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_geomagnetic_scaling import read_dataset_configuration  # noqa: E402
from compare_ensembles import extract_ensemble  # noqa: E402


BACKENDS = ("legacy_proposal", "cuda")
LABELS = {"legacy_proposal": "Original CPU", "cuda": "CUDA"}
COLORS = {"legacy_proposal": "#1565c0", "cuda": "#d84315"}
POWER_METRICS = (
    (
        "profile_charged_max",
        r"$N_{\rm charged}(X_{\max})$",
        1.0,
    ),
    (
        "profile_charged_integral",
        r"$\int N_{\rm charged}\,dX$",
        1.0,
    ),
    (
        "profile_photon_integral",
        r"$\int N_\gamma\,dX$",
        1.0,
    ),
    (
        "energy_deposit_sum_GeV",
        "Deposited energy [GeV]",
        1.0,
    ),
    (
        "ground_em_weighted_count",
        "Ground EM weighted count",
        None,
    ),
    (
        "ground_em_kinetic_energy_GeV",
        "Ground EM kinetic energy [GeV]",
        None,
    ),
)
XMAX_METRIC = "profile_xmax_charged_gcm2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each complete CPU/CUDA ensemble.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=88421)
    return parser.parse_args()


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual = y - prediction
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - float(np.sum(residual**2)) / total
        if total > 0.0
        else 1.0,
    }


def collect_rows(datasets: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    configurations: list[dict[str, Any]] = []
    seen_energy: set[float] = set()
    for dataset in datasets:
        config = read_dataset_configuration(dataset)
        if config["primary_pdg"] != 11 or abs(config["theta_deg"]) > 1.0e-9:
            continue
        energy = float(config["energy_GeV"])
        if energy in seen_energy:
            raise ValueError(f"duplicate vertical energy point: {energy:g} GeV")
        seen_energy.add(energy)
        configurations.append(
            {
                "dataset": str(dataset.resolve()),
                "dataset_name": dataset.name,
                "energy_GeV": energy,
                "events_per_backend": int(config["events"]),
                "seed": int(config["seed"]),
            }
        )
        for backend in BACKENDS:
            ensemble = extract_ensemble(
                backend,
                dataset / backend,
                backend == "cuda",
                allow_legacy_provenance=True,
            )
            frame = ensemble.scalars.reset_index()
            frame["dataset"] = dataset.name
            frame["energy_GeV"] = energy
            frame["backend"] = backend
            rows.append(frame)
    if len(seen_energy) < 3:
        raise ValueError("at least three vertical energy points are required")
    return pd.concat(rows, ignore_index=True), configurations


def grouped_values(
    frame: pd.DataFrame, backend: str, metric: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    selected = frame.loc[frame["backend"] == backend]
    grouped: dict[float, list[float]] = defaultdict(list)
    for energy, value in zip(selected["energy_GeV"], selected[metric]):
        if np.isfinite(value):
            grouped[float(energy)].append(float(value))
    energies = np.asarray(sorted(grouped), dtype=np.float64)
    arrays = [np.asarray(grouped[energy], dtype=np.float64) for energy in energies]
    if len(arrays) < 3 or any(values.size < 2 for values in arrays):
        raise ValueError(f"insufficient values for {backend}/{metric}")
    return energies, arrays


def bootstrap_group_means(
    arrays: list[np.ndarray], repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    samples = np.empty((repetitions, len(arrays)), dtype=np.float64)
    for column, values in enumerate(arrays):
        indices = rng.integers(
            0, values.size, size=(repetitions, values.size)
        )
        samples[:, column] = np.mean(values[indices], axis=1)
    return samples


def power_fit(
    energies: np.ndarray,
    arrays: list[np.ndarray],
    repetitions: int,
    seed: int,
    expected_slope: float | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    means = np.asarray([np.mean(values) for values in arrays])
    if np.any(means <= 0.0):
        raise ValueError("power-law group means must be positive")
    fit = linear_fit(np.log(energies / 1000.0), np.log(means))
    bootstrap_means = bootstrap_group_means(
        arrays, repetitions, np.random.default_rng(seed)
    )
    slopes = np.asarray(
        [
            linear_fit(np.log(energies / 1000.0), np.log(sample))["slope"]
            for sample in bootstrap_means
        ]
    )
    local = np.diff(np.log(means)) / np.diff(np.log(energies))
    local_samples = np.diff(np.log(bootstrap_means), axis=1) / np.diff(
        np.log(energies)
    )
    result: dict[str, Any] = {
        "law": "mean(Y) = normalization * (E / 1 TeV)^slope",
        "slope": fit["slope"],
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
        "normalization_at_1TeV": math.exp(fit["intercept"]),
        "r_squared_group_means": fit["r_squared"],
        "groups": [
            {
                "energy_GeV": float(energy),
                "showers": int(values.size),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "mean_ci95": np.quantile(
                    bootstrap_means[:, index], [0.025, 0.975]
                ).tolist(),
            }
            for index, (energy, values) in enumerate(zip(energies, arrays))
        ],
        "adjacent_local_slopes": [
            {
                "energy_low_GeV": float(energies[index]),
                "energy_high_GeV": float(energies[index + 1]),
                "slope": float(local[index]),
                "ci95": np.quantile(
                    local_samples[:, index], [0.025, 0.975]
                ).tolist(),
            }
            for index in range(local.size)
        ],
        "scientific_status": (
            "accepted"
            if fit["r_squared"] >= 0.90
            else "diagnostic_only_due_to_single_power_law_misfit"
        ),
    }
    if expected_slope is not None:
        result["expected_slope"] = expected_slope
        result["expected_inside_ci95"] = bool(
            (
                result["slope_ci95"][0]
                <= expected_slope
                <= result["slope_ci95"][1]
            )
            or np.isclose(expected_slope, result["slope_ci95"][0])
            or np.isclose(expected_slope, result["slope_ci95"][1])
        )
    return result, slopes, bootstrap_means


def xmax_fit(
    energies: np.ndarray,
    arrays: list[np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    means = np.asarray([np.mean(values) for values in arrays])
    x = np.log10(energies / 1000.0)
    fit = linear_fit(x, means)
    bootstrap_means = bootstrap_group_means(
        arrays, repetitions, np.random.default_rng(seed)
    )
    slopes = np.asarray(
        [linear_fit(x, sample)["slope"] for sample in bootstrap_means]
    )
    return {
        "law": "mean(Xmax) = Xmax_1TeV + D10 * log10(E / 1 TeV)",
        "elongation_rate_gcm2_per_decade": fit["slope"],
        "elongation_rate_ci95": np.quantile(
            slopes, [0.025, 0.975]
        ).tolist(),
        "Xmax_at_1TeV_gcm2": fit["intercept"],
        "r_squared_group_means": fit["r_squared"],
        "groups": [
            {
                "energy_GeV": float(energy),
                "showers": int(values.size),
                "mean_gcm2": float(np.mean(values)),
                "standard_deviation_gcm2": float(np.std(values, ddof=1)),
                "mean_ci95": np.quantile(
                    bootstrap_means[:, index], [0.025, 0.975]
                ).tolist(),
            }
            for index, (energy, values) in enumerate(zip(energies, arrays))
        ],
        "scientific_status": (
            "accepted"
            if fit["r_squared"] >= 0.90
            else "diagnostic_only_due_to_single_log_linear_misfit"
        ),
    }, slopes, bootstrap_means


def compare_bootstraps(
    cpu_slopes: np.ndarray,
    cuda_slopes: np.ndarray,
    cpu_means: np.ndarray,
    cuda_means: np.ndarray,
    slope_key: str = "cuda_minus_cpu",
) -> dict[str, Any]:
    difference = cuda_slopes - cpu_slopes
    ratios = cuda_means / cpu_means
    return {
        slope_key: float(np.median(cuda_slopes) - np.median(cpu_slopes)),
        "slope_difference_ci95": np.quantile(
            difference, [0.025, 0.975]
        ).tolist(),
        "zero_inside_slope_difference_ci95": bool(
            np.quantile(difference, 0.025)
            <= 0.0
            <= np.quantile(difference, 0.975)
        ),
        "cuda_over_cpu_group_mean_ratios": [
            {
                "ratio": float(np.median(ratios[:, index])),
                "ci95": np.quantile(
                    ratios[:, index], [0.025, 0.975]
                ).tolist(),
            }
            for index in range(ratios.shape[1])
        ],
    }


def build_report(
    frame: pd.DataFrame,
    configurations: list[dict[str, Any]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "method": {
            "statistical_unit": "complete shower",
            "group_center": "arithmetic ensemble mean",
            "bootstrap": "independent shower resampling within energy/backend",
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
        },
        "datasets": sorted(configurations, key=lambda row: row["energy_GeV"]),
        "xmax": {},
        "power_laws": {},
    }
    fit_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for backend_index, backend in enumerate(BACKENDS):
        energies, arrays = grouped_values(frame, backend, XMAX_METRIC)
        result, slopes, means = xmax_fit(
            energies,
            arrays,
            repetitions,
            seed + 100 * backend_index,
        )
        report["xmax"][backend] = result
        fit_cache[backend] = (slopes, means)
    cpu_slopes, cpu_means = fit_cache["legacy_proposal"]
    cuda_slopes, cuda_means = fit_cache["cuda"]
    report["xmax"]["comparison"] = compare_bootstraps(
        cpu_slopes, cuda_slopes, cpu_means, cuda_means,
        slope_key="cuda_minus_cpu_elongation_rate",
    )

    for metric_index, (metric, label, expected) in enumerate(POWER_METRICS):
        report["power_laws"][metric] = {"label": label}
        metric_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for backend_index, backend in enumerate(BACKENDS):
            energies, arrays = grouped_values(frame, backend, metric)
            result, slopes, means = power_fit(
                energies,
                arrays,
                repetitions,
                seed + 1000 * (metric_index + 1) + 100 * backend_index,
                expected,
            )
            report["power_laws"][metric][backend] = result
            metric_cache[backend] = (slopes, means)
        cpu_slopes, cpu_means = metric_cache["legacy_proposal"]
        cuda_slopes, cuda_means = metric_cache["cuda"]
        comparison = compare_bootstraps(
            cpu_slopes, cuda_slopes, cpu_means, cuda_means
        )
        for index, item in enumerate(
            comparison["cuda_over_cpu_group_mean_ratios"]
        ):
            item["energy_GeV"] = float(energies[index])
        report["power_laws"][metric]["comparison"] = comparison
    return report


def plot_report(report: dict[str, Any], output: Path) -> None:
    panels = (
        ("xmax", r"$X_{\max}^{\rm charged}$ [g cm$^{-2}$]"),
        ("profile_charged_max", r"$N_{\rm charged}(X_{\max})$"),
        ("profile_charged_integral", r"$\int N_{\rm charged}\,dX$"),
        ("energy_deposit_sum_GeV", "Deposited energy [GeV]"),
        ("ground_em_weighted_count", "Ground EM weighted count"),
        ("ground_em_kinetic_energy_GeV", "Ground EM kinetic energy [GeV]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        block = report["xmax"] if metric == "xmax" else report["power_laws"][metric]
        for backend in BACKENDS:
            groups = block[backend]["groups"]
            energies = np.asarray([item["energy_GeV"] for item in groups])
            key = "mean_gcm2" if metric == "xmax" else "mean"
            values = np.asarray([item[key] for item in groups])
            ci = np.asarray([item["mean_ci95"] for item in groups])
            axis.errorbar(
                energies,
                values,
                yerr=np.vstack([values - ci[:, 0], ci[:, 1] - values]),
                fmt="o-",
                capsize=2,
                color=COLORS[backend],
                label=LABELS[backend],
            )
        axis.set_xscale("log")
        if metric != "xmax":
            axis.set_yscale("log")
        axis.set_xlabel("Primary energy [GeV]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, which="both")
        if metric == "xmax":
            cpu = block["legacy_proposal"]
            cuda = block["cuda"]
            text = (
                f"CPU $D_{{10}}$={cpu['elongation_rate_gcm2_per_decade']:.1f}\n"
                f"CUDA $D_{{10}}$={cuda['elongation_rate_gcm2_per_decade']:.1f}"
            )
        else:
            cpu = block["legacy_proposal"]
            cuda = block["cuda"]
            text = (
                f"CPU slope={cpu['slope']:.3f}\n"
                f"CUDA slope={cuda['slope']:.3f}"
            )
        axis.text(
            0.04,
            0.94,
            text,
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Electromagnetic-shower observable scaling\n"
        "vertical electron primaries; shower-bootstrap 95% intervals",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output / "shower_observable_scaling.png", dpi=200)
    plt.close(figure)


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Shower observable scaling",
        "",
        "每个统计单位是一场完整 shower；区间来自按能量和后端分层的 "
        f"{report['method']['bootstrap_repetitions']} 次 bootstrap。",
        "",
        "## Xmax elongation",
        "",
        "| 后端 | D10 [g/cm²/decade] | 95% CI | R² |",
        "|---|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        item = report["xmax"][backend]
        lines.append(
            f"| {LABELS[backend]} | "
            f"{item['elongation_rate_gcm2_per_decade']:.2f} | "
            f"[{item['elongation_rate_ci95'][0]:.2f}, "
            f"{item['elongation_rate_ci95'][1]:.2f}] | "
            f"{item['r_squared_group_means']:.3f} |"
        )
    comparison = report["xmax"]["comparison"]
    lines.extend(
        [
            "",
            "CUDA−CPU elongation-rate difference: "
            f"{comparison['cuda_minus_cpu_elongation_rate']:.2f} "
            f"[{comparison['slope_difference_ci95'][0]:.2f}, "
            f"{comparison['slope_difference_ci95'][1]:.2f}] g/cm²/decade.",
            "",
            "## Power-law exponents",
            "",
            "| Observable | CPU | CUDA | CUDA−CPU [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric, label, _ in POWER_METRICS:
        block = report["power_laws"][metric]
        cpu = block["legacy_proposal"]
        cuda = block["cuda"]
        diff = block["comparison"]
        lines.append(
            f"| {label} | {cpu['slope']:.3f} "
            f"[{cpu['slope_ci95'][0]:.3f}, {cpu['slope_ci95'][1]:.3f}] | "
            f"{cuda['slope']:.3f} "
            f"[{cuda['slope_ci95'][0]:.3f}, {cuda['slope_ci95'][1]:.3f}] | "
            f"{diff['cuda_minus_cpu']:+.3f} "
            f"[{diff['slope_difference_ci95'][0]:+.3f}, "
            f"{diff['slope_difference_ci95'][1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "![Shower observable scaling](shower_observable_scaling.png)",
            "",
            "完整的逐能量均值、局部指数、归一化比和 bootstrap 区间见 "
            "`shower_scaling_laws.json`。",
        ]
    )
    (output / "shower_scaling_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    args.output.mkdir(parents=True, exist_ok=True)
    frame, configurations = collect_rows(args.dataset)
    report = build_report(
        frame,
        configurations,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    frame.to_csv(args.output / "per_shower_observables.csv", index=False)
    (args.output / "shower_scaling_laws.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_report(report, args.output)
    write_markdown(report, args.output)
    print(f"Wrote shower scaling analysis to {args.output.resolve()}")


if __name__ == "__main__":
    main()
