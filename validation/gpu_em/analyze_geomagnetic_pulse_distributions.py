#!/usr/bin/env python3
"""Compare CPU/CUDA geomagnetic-pulse amplitude and width distributions.

The pulse definition is deliberately imported from the user's existing
``pulse_analysis_modular`` project:

* geomagnetic component: EyPrime = E dot unit(v x B);
* amplitude: absolute value of the selected signed main-pulse peak;
* width: the valid max-Q square-window fit after 0.1 ns interpolation.

Radio observers at a fixed radius are correlated measurements of one shower.
The statistical sample is therefore one azimuth-aggregated value per shower,
not one value per antenna.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ks_2samp, ttest_ind


BACKENDS = ("legacy_proposal", "cuda")
BACKEND_LABELS = {"legacy_proposal": "Original CPU", "cuda": "CUDA"}
BACKEND_COLORS = {"legacy_proposal": "#1565c0", "cuda": "#d84315"}
ALGORITHMS = ("CoREAS", "ZHS")


def finite(values: Iterable[float], *, positive: bool = False) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    mask = np.isfinite(result)
    if positive:
        mask &= result > 0.0
    return result[mask]


def geometric_mean(values: Iterable[float]) -> float:
    array = finite(values, positive=True)
    if array.size == 0:
        return math.nan
    return float(np.exp(np.mean(np.log(array))))


def scalar_summary(values: Iterable[float]) -> dict[str, Any]:
    array = finite(values)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": (
            float(np.std(array, ddof=1)) if array.size > 1 else math.nan
        ),
        "q16": float(np.quantile(array, 0.16)),
        "q84": float(np.quantile(array, 0.84)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def load_reference_apis(
    pulse_root: Path,
) -> tuple[
    Callable[..., dict[str, np.ndarray]],
    Callable[..., dict[str, float]],
    Callable[..., tuple[list[dict[str, Any]], np.ndarray]],
    Callable[..., np.ndarray],
    Callable[..., np.ndarray],
    type,
]:
    if not (pulse_root / "pulse_analysis" / "core" / "analysis.py").is_file():
        raise FileNotFoundError(
            f"pulse_analysis_modular was not found under {pulse_root}"
        )
    validation_dir = Path(__file__).resolve().parent
    for path in (pulse_root, validation_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    geometry = importlib.import_module("pulse_analysis.core.geometry")
    analysis = importlib.import_module("pulse_analysis.core.analysis")
    plotting_config = importlib.import_module(
        "pulse_analysis.plotting.config"
    )
    plotting_stats = importlib.import_module("pulse_analysis.plotting.stats")
    radio = importlib.import_module("analyze_cpu_cuda_radio")
    return (
        geometry.build_polarization_basis,
        analysis.analyze_pulse_parameters,
        radio.read_radio_records,
        radio.band_limited_waveform,
        plotting_stats.robust_pulse_width_mask,
        plotting_config.PulseWidthFilterConfig,
    )


def load_run_configuration(
    dataset: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else dataset / "run_manifest.json"
    )
    gpu_config_path = dataset / "cuda" / "gpu_em" / "config.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gpu_config_path.is_file():
        cuda_sources = manifest.get("additional_sources", {}).get("cuda", [])
        if not isinstance(cuda_sources, list):
            raise ValueError(
                f"additional_sources.cuda must be a list in {manifest_path}"
            )
        for source in cuda_sources:
            candidate = Path(str(source)).resolve() / "gpu_em" / "config.yaml"
            if candidate.is_file():
                gpu_config_path = candidate
                break
    if not gpu_config_path.is_file():
        raise FileNotFoundError(gpu_config_path)
    gpu_config = yaml.safe_load(gpu_config_path.read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    environment = gpu_config["environment"]
    magnetic = environment["magnetic_field_T"]
    igrf = configuration.get("IGRF", {})
    return {
        "manifest": manifest,
        "theta_deg": float(configuration["zenith_deg"]),
        "phi_deg": float(configuration["azimuth_deg"]),
        "magnetic_field_T": np.asarray(
            [magnetic["x"], magnetic["y"], magnetic["z"]], dtype=np.float64
        ),
        "igrf_model": str(
            igrf.get(
                "model",
                environment.get("geomagnetic_model", "unknown"),
            )
        ),
        "igrf_year": float(
            igrf.get(
                "year",
                environment.get("geomagnetic_year", math.nan),
            )
        ),
    }


def backend_output_directory(dataset: Path, backend: str) -> Path:
    direct = dataset / backend
    if direct.is_dir():
        return direct
    if backend == "legacy_proposal":
        proposal = dataset / "proposal"
        if proposal.is_dir():
            return proposal
    raise FileNotFoundError(
        f"radio backend output is absent for {backend}: {direct}"
    )


def backend_output_directories(
    dataset: Path,
    backend: str,
    manifest_path: Path | None = None,
) -> list[Path]:
    directories: list[Path] = []
    try:
        directories.append(backend_output_directory(dataset, backend))
    except FileNotFoundError:
        if backend == "legacy_proposal":
            directories.extend(
                sorted(
                    path
                    for path in dataset.glob("proposal_shard_*")
                    if path.is_dir()
                )
            )
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else dataset / "run_manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        source_key = "proposal" if backend == "legacy_proposal" else "cuda"
        additional = (
            manifest.get("additional_sources", {}).get(source_key, [])
        )
        if not isinstance(additional, list):
            raise ValueError(
                f"run_manifest additional_sources.{source_key} is not a list"
            )
        for requested in additional:
            path = Path(str(requested)).resolve()
            if not path.is_dir():
                raise FileNotFoundError(
                    f"additional radio source is absent: {path}"
                )
            directories.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError(
            f"no radio outputs for {backend} under {dataset}"
        )
    return unique


def nearest_available_radius(
    records: list[dict[str, Any]], requested_radius_m: float
) -> float:
    radii = np.unique(
        np.asarray([round(record["radius_m"], 6) for record in records])
    )
    nearest = float(radii[np.argmin(np.abs(radii - requested_radius_m))])
    tolerance = max(1.e-6, 1.e-6 * max(abs(requested_radius_m), 1.0))
    if abs(nearest - requested_radius_m) > tolerance:
        available = ", ".join(f"{value:g}" for value in radii)
        raise ValueError(
            f"requested radius {requested_radius_m:g} m is unavailable; "
            f"available radii are {available} m"
        )
    return nearest


def extract_antenna_rows(
    *,
    records: list[dict[str, Any]],
    backend: str,
    algorithm: str,
    radius_m: float,
    yprime: np.ndarray,
    analyze_pulse_parameters: Callable[..., dict[str, float]],
    band_limited_waveform: Callable[..., np.ndarray],
    band_MHz: tuple[float, float] | None,
    analysis_sampling_rate_GHz: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not np.isclose(
            record["radius_m"], radius_m, rtol=0.0, atol=1.e-5
        ):
            continue
        time_ns = np.asarray(
            record["time_ns"], dtype=np.float64
        )
        field = np.asarray(record["field"], dtype=np.float64)
        if analysis_sampling_rate_GHz is not None:
            if time_ns.size < 2:
                raise ValueError("radio waveform has fewer than two samples")
            source_dt_ns = float(np.median(np.diff(time_ns)))
            target_dt_ns = 1.0 / analysis_sampling_rate_GHz
            factor = int(round(target_dt_ns / source_dt_ns))
            if (
                factor < 1
                or not np.isclose(
                    factor * source_dt_ns,
                    target_dt_ns,
                    rtol=1.e-6,
                    atol=1.e-9,
                )
            ):
                raise ValueError(
                    "analysis sampling rate must be an integer "
                    "downsampling of the waveform rate"
                )
            retained = (time_ns.size // factor) * factor
            if retained < factor:
                raise ValueError("radio waveform is too short to downsample")
            time_ns = time_ns[:retained].reshape(-1, factor).mean(axis=1)
            field = field[:retained].reshape(
                -1, factor, field.shape[1]
            ).mean(axis=1)
        geomagnetic_waveform = np.asarray(
            field @ yprime, dtype=np.float64
        )
        if band_MHz is not None:
            if time_ns.size < 2:
                raise ValueError("radio waveform has fewer than two samples")
            dt_s = float(np.median(np.diff(time_ns))) * 1.e-9
            geomagnetic_waveform = np.asarray(
                band_limited_waveform(
                    geomagnetic_waveform[:, np.newaxis],
                    dt_s,
                    band_MHz[0],
                    band_MHz[1],
                )[:, 0],
                dtype=np.float64,
            )
        pulse = analyze_pulse_parameters(
            time_ns,
            geomagnetic_waveform,
            ro_m=float(radius_m),
        )
        peak_signed = float(pulse["peak_amplitude"])
        width = float(pulse["pulse_width"])
        valid = bool(pulse["valid"]) and np.isfinite(width) and width > 0.0
        rows.append(
            {
                "backend": backend,
                "algorithm": algorithm,
                "shower": int(record["shower"]),
                "observer": int(record["observer"]),
                # Preserve the requested nominal radius.  Diagonal antenna
                # coordinates are rounded in text files, so their reconstructed
                # Euclidean radii may differ from an axial antenna at the same
                # nominal radius by O(1e-6 m).
                "radius_m": float(radius_m),
                "geomagnetic_peak_signed_V_per_m": peak_signed,
                "geomagnetic_peak_abs_V_per_m": abs(peak_signed),
                "pulse_width_ns": width,
                "pulse_width_valid": valid,
                "pulse_method": str(pulse["method"]),
            }
        )
    return rows


def aggregate_by_shower(antenna_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in antenna_rows:
        grouped[
            (
                row["backend"],
                row["algorithm"],
                row["shower"],
                round(float(row["radius_m"]), 6),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (
        backend,
        algorithm,
        shower,
        nominal_radius_m,
    ), rows in sorted(grouped.items()):
        amplitudes = finite(
            (row["geomagnetic_peak_abs_V_per_m"] for row in rows),
            positive=True,
        )
        raw_widths = finite(
            (
                row["pulse_width_ns"]
                for row in rows
                if row["pulse_width_valid"]
            ),
            positive=True,
        )
        widths = finite(
            (
                row["pulse_width_ns"]
                for row in rows
                if row["pulse_width_filter_pass"]
            ),
            positive=True,
        )
        result.append(
            {
                "backend": backend,
                "algorithm": algorithm,
                "shower": shower,
                "radius_m": nominal_radius_m,
                "antenna_count": len(rows),
                "amplitude_valid_count": int(amplitudes.size),
                "geomagnetic_amplitude_geomean_V_per_m": geometric_mean(
                    amplitudes
                ),
                "geomagnetic_amplitude_median_V_per_m": (
                    float(np.median(amplitudes))
                    if amplitudes.size
                    else math.nan
                ),
                "pulse_width_raw_valid_count": int(raw_widths.size),
                "pulse_width_filter_pass_count": int(widths.size),
                "pulse_width_filter_pass_fraction": (
                    float(widths.size / len(rows)) if rows else math.nan
                ),
                "pulse_width_mean_ns": (
                    float(np.mean(widths)) if widths.size else math.nan
                ),
                "pulse_width_median_ns": (
                    float(np.median(widths)) if widths.size else math.nan
                ),
            }
        )
    return result


def apply_reference_width_filter(
    antenna_rows: list[dict[str, Any]],
    *,
    robust_pulse_width_mask: Callable[..., np.ndarray],
    filter_config: Any,
) -> None:
    """Apply pulse_analysis_modular's default plot-time width filter in place."""
    grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(antenna_rows):
        grouped[
            (
                row["backend"],
                row["algorithm"],
                round(float(row["radius_m"]), 6),
            )
        ].append(index)
    for indices in grouped.values():
        widths = np.asarray(
            [
                (
                    antenna_rows[index]["pulse_width_ns"]
                    if antenna_rows[index]["pulse_width_valid"]
                    else math.nan
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        keep = np.asarray(
            robust_pulse_width_mask(widths, filter_config), dtype=bool
        )
        if keep.shape != widths.shape:
            raise ValueError("reference pulse-width filter returned wrong shape")
        for index, accepted in zip(indices, keep):
            antenna_rows[index]["pulse_width_filter_pass"] = bool(accepted)


def bootstrap_ratio(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float],
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        a = reference[
            rng.integers(0, reference.size, size=reference.size)
        ]
        b = candidate[rng.integers(0, candidate.size, size=candidate.size)]
        denominator = statistic(a)
        ratios[index] = (
            statistic(b) / denominator if denominator > 0.0 else math.nan
        )
    ratios = ratios[np.isfinite(ratios)]
    return {
        "ratio": float(statistic(candidate) / statistic(reference)),
        "ci95_low": float(np.quantile(ratios, 0.025)),
        "ci95_high": float(np.quantile(ratios, 0.975)),
    }


def distribution_comparison(
    reference: Iterable[float],
    candidate: Iterable[float],
    *,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    a = finite(reference, positive=True)
    b = finite(candidate, positive=True)
    if a.size == 0 or b.size == 0:
        return {
            "available": False,
            "cpu_count": int(a.size),
            "cuda_count": int(b.size),
        }
    if metric == "amplitude":
        statistic = geometric_mean
        statistic_name = "geometric_mean"
    elif metric == "width":
        statistic = lambda values: float(np.mean(values))
        statistic_name = "arithmetic_mean"
    else:
        raise ValueError(metric)
    ks = ks_2samp(a, b, method="exact")
    welch = ttest_ind(np.log10(a), np.log10(b), equal_var=False)
    return {
        "available": True,
        "metric": metric,
        "cpu": scalar_summary(a),
        "cuda": scalar_summary(b),
        "reported_statistic": statistic_name,
        "cpu_reported_statistic": float(statistic(a)),
        "cuda_reported_statistic": float(statistic(b)),
        "cuda_over_cpu": bootstrap_ratio(
            a,
            b,
            statistic=statistic,
            repetitions=repetitions,
            seed=seed,
        ),
        "ks_two_sample": {
            "statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "method": "exact",
        },
        "welch_log10": {
            "statistic": float(welch.statistic),
            "p_value": float(welch.pvalue),
        },
    }


def metric_equivalence(
    result: dict[str, Any],
    *,
    relative_tolerance: float,
    minimum_ks_p_value: float,
    minimum_count: int,
) -> dict[str, Any]:
    """Evaluate a two-sided equivalence interval plus a shape diagnostic."""
    if not result.get("available", False):
        return {
            "passed": False,
            "reason": "metric is unavailable",
        }
    ratio = result["cuda_over_cpu"]
    lower = 1.0 - relative_tolerance
    upper = 1.0 + relative_tolerance
    cpu_count = int(result["cpu"]["count"])
    cuda_count = int(result["cuda"]["count"])
    confidence_interval_contained = (
        float(ratio["ci95_low"]) >= lower
        and float(ratio["ci95_high"]) <= upper
    )
    ks_p_value = float(result["ks_two_sample"]["p_value"])
    count_pass = cpu_count >= minimum_count and cuda_count >= minimum_count
    shape_pass = ks_p_value >= minimum_ks_p_value
    return {
        "passed": bool(
            confidence_interval_contained and shape_pass and count_pass
        ),
        "equivalence_interval": [lower, upper],
        "bootstrap_ratio_interval": [
            float(ratio["ci95_low"]),
            float(ratio["ci95_high"]),
        ],
        "confidence_interval_contained": confidence_interval_contained,
        "minimum_ks_p_value": minimum_ks_p_value,
        "ks_p_value": ks_p_value,
        "shape_pass": shape_pass,
        "minimum_count_per_backend": minimum_count,
        "cpu_count": cpu_count,
        "cuda_count": cuda_count,
        "count_pass": count_pass,
    }


def holm_bonferroni_rejections(
    p_values: list[float], familywise_alpha: float
) -> list[bool]:
    """Return Holm-Bonferroni reject decisions in input order."""
    if not p_values:
        return []
    if not (
        math.isfinite(familywise_alpha)
        and 0.0 <= familywise_alpha <= 1.0
    ):
        raise ValueError("familywise alpha must be in [0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    rejected = [False] * len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        if not (math.isfinite(p_value) and 0.0 <= p_value <= 1.0):
            raise ValueError("KS p values must be finite and in [0, 1]")
        threshold = familywise_alpha / (len(p_values) - rank)
        if p_value > threshold:
            break
        rejected[original_index] = True
    return rejected


def build_acceptance(
    *,
    comparisons: dict[str, Any],
    shower_rows: list[dict[str, Any]],
    band_MHz: tuple[float, float] | None,
    relative_tolerance: float,
    minimum_ks_p_value: float,
    minimum_count: int,
    minimum_width_shower_fraction: float,
) -> dict[str, Any]:
    """Build a fail-closed pulse-distribution acceptance result.

    The user's max-Q square-window width estimator is calibrated for the raw
    monopolar pulse.  An ideal rectangular FFT band-pass creates long ringing,
    so band-passed widths remain useful diagnostics but cannot satisfy this
    formal gate.
    """
    algorithms: dict[str, Any] = {}
    minimum_width_count = max(
        2,
        math.ceil(minimum_count * minimum_width_shower_fraction),
    )
    for algorithm in ALGORITHMS:
        backend_shower_counts = {
            backend: sum(
                row["algorithm"] == algorithm
                and row["backend"] == backend
                for row in shower_rows
            )
            for backend in BACKENDS
        }
        width_counts = {
            "legacy_proposal": int(
                comparisons[algorithm]["width"]
                .get("cpu", {})
                .get("count", 0)
            ),
            "cuda": int(
                comparisons[algorithm]["width"]
                .get("cuda", {})
                .get("count", 0)
            ),
        }
        width_fractions = {
            backend: (
                width_counts[backend] / backend_shower_counts[backend]
                if backend_shower_counts[backend]
                else 0.0
            )
            for backend in BACKENDS
        }
        width_coverage_pass = all(
            fraction >= minimum_width_shower_fraction
            for fraction in width_fractions.values()
        )
        amplitude = metric_equivalence(
            comparisons[algorithm]["amplitude"],
            relative_tolerance=relative_tolerance,
            minimum_ks_p_value=minimum_ks_p_value,
            minimum_count=minimum_count,
        )
        width = metric_equivalence(
            comparisons[algorithm]["width"],
            relative_tolerance=relative_tolerance,
            minimum_ks_p_value=minimum_ks_p_value,
            minimum_count=minimum_width_count,
        )
        width.update(
            {
                "minimum_valid_shower_fraction": (
                    minimum_width_shower_fraction
                ),
                "valid_shower_fraction": width_fractions,
                "coverage_pass": width_coverage_pass,
            }
        )
        algorithms[algorithm] = {
            "amplitude": amplitude,
            "width": width,
            "passed": False,
        }

    metric_paths = [
        (algorithm, metric)
        for algorithm in ALGORITHMS
        for metric in ("amplitude", "width")
    ]
    p_values = [
        float(algorithms[algorithm][metric]["ks_p_value"])
        for algorithm, metric in metric_paths
    ]
    rejected = holm_bonferroni_rejections(
        p_values, minimum_ks_p_value
    )
    for (algorithm, metric), familywise_rejected in zip(
        metric_paths, rejected
    ):
        result = algorithms[algorithm][metric]
        result["uncorrected_shape_pass"] = result["shape_pass"]
        result["holm_bonferroni_rejected"] = familywise_rejected
        result["shape_pass"] = not familywise_rejected
        result["passed"] = bool(
            result["confidence_interval_contained"]
            and result["shape_pass"]
            and result["count_pass"]
            and (
                result.get("coverage_pass", True)
                if metric == "width"
                else True
            )
        )
    for algorithm in ALGORITHMS:
        algorithms[algorithm]["passed"] = bool(
            algorithms[algorithm]["amplitude"]["passed"]
            and algorithms[algorithm]["width"]["passed"]
        )

    raw_waveform_gate = band_MHz is None
    passed = raw_waveform_gate and all(
        result["passed"] for result in algorithms.values()
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scientific_scope": (
            "raw monopolar pulse amplitude and max-Q square-window width"
            if raw_waveform_gate
            else (
                "diagnostic only: the ideal rectangular FFT band-pass "
                "introduces ringing for which the max-Q square-window width "
                "estimator is not calibrated"
            )
        ),
        "raw_waveform_gate": raw_waveform_gate,
        "relative_equivalence_tolerance": relative_tolerance,
        "minimum_ks_p_value": minimum_ks_p_value,
        "shape_test_correction": "Holm-Bonferroni",
        "shape_test_family_size": len(metric_paths),
        "shape_test_familywise_alpha": minimum_ks_p_value,
        "minimum_count_per_backend": minimum_count,
        "minimum_width_valid_count_per_backend": minimum_width_count,
        "minimum_width_valid_shower_fraction": (
            minimum_width_shower_fraction
        ),
        "algorithms": algorithms,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_values(
    shower_rows: list[dict[str, Any]],
    algorithm: str,
    backend: str,
    column: str,
) -> np.ndarray:
    return finite(
        (
            row[column]
            for row in shower_rows
            if row["algorithm"] == algorithm and row["backend"] == backend
        ),
        positive=True,
    )


def expanded_range(values: np.ndarray, *, logarithmic: bool) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if logarithmic:
        return low / 1.12, high * 1.12
    margin = 0.06 * max(high - low, abs(high), 1.e-12)
    return max(0.0, low - margin), high + margin


def plot_distributions(
    *,
    shower_rows: list[dict[str, Any]],
    comparisons: dict[str, Any],
    output: Path,
    radius_m: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    metric_specs = (
        (
            "geomagnetic_amplitude_geomean_V_per_m",
            "amplitude",
            r"$|E_{\mathbf{v}\times\mathbf{B}}|$ peak [V m$^{-1}$]",
            True,
        ),
        (
            "pulse_width_mean_ns",
            "width",
            "Pulse width [ns] (reference robust filter)",
            False,
        ),
    )
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (
            column,
            metric,
            xlabel,
            logarithmic,
        ) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            arrays = {
                backend: comparison_values(
                    shower_rows, algorithm, backend, column
                )
                for backend in BACKENDS
            }
            pooled = np.concatenate([arrays[backend] for backend in BACKENDS])
            low, high = expanded_range(pooled, logarithmic=logarithmic)
            bins = (
                np.geomspace(low, high, 15)
                if logarithmic
                else np.linspace(low, high, 15)
            )
            for backend in BACKENDS:
                axis.hist(
                    arrays[backend],
                    bins=bins,
                    weights=np.full(
                        arrays[backend].shape,
                        1.0 / arrays[backend].size,
                    ),
                    histtype="step",
                    linewidth=2.0,
                    color=BACKEND_COLORS[backend],
                    label=(
                        f"{BACKEND_LABELS[backend]} "
                        f"(N={arrays[backend].size})"
                    ),
                )
            if logarithmic:
                axis.set_xscale("log")
            result = comparisons[algorithm][metric]
            ratio = result["cuda_over_cpu"]
            axis.text(
                0.98,
                0.96,
                (
                    f"CUDA/CPU = {ratio['ratio']:.3f}\n"
                    f"95% CI [{ratio['ci95_low']:.3f}, "
                    f"{ratio['ci95_high']:.3f}]\n"
                    f"KS $D$ = {result['ks_two_sample']['statistic']:.3f}, "
                    f"$p$ = {result['ks_two_sample']['p_value']:.3g}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.88,
                },
            )
            axis.set_title(f"{algorithm}: {metric.capitalize()}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Fraction of showers")
            axis.grid(alpha=0.23, which="both")
            axis.legend(loc="upper left", frameon=False)
    figure.suptitle(
        (
            "Original CPU vs CUDA geomagnetic pulse distributions\n"
            f"{radius_m:g} m radius; each shower aggregates eight azimuth antennas"
        ),
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def format_ratio(result: dict[str, Any]) -> str:
    ratio = result["cuda_over_cpu"]
    return (
        f"{ratio['ratio']:.4f} "
        f"[{ratio['ci95_low']:.4f}, {ratio['ci95_high']:.4f}]"
    )


def write_markdown(
    *,
    path: Path,
    dataset: Path,
    radius_m: float,
    run_configuration: dict[str, Any],
    yprime: np.ndarray,
    comparisons: dict[str, Any],
    antenna_rows: list[dict[str, Any]],
    band_MHz: tuple[float, float] | None,
    analysis_sampling_rate_GHz: float | None,
    acceptance: dict[str, Any],
) -> None:
    shower_counts = {
        backend: len(
            {
                row["shower"]
                for row in antenna_rows
                if row["backend"] == backend
                and row["algorithm"] == ALGORITHMS[0]
            }
        )
        for backend in BACKENDS
    }
    lines = [
        "# CPU/CUDA 地磁脉冲振幅与宽度分布比较",
        "",
        "## 口径",
        "",
        f"- 数据集：`{dataset}`。",
        f"- 天线半径：{radius_m:g} m；每场 shower 有 8 个不同方位的天线。",
        (
            f"- 在提取振幅和宽度前施加理想 FFT "
            f"{band_MHz[0]:g}--{band_MHz[1]:g} MHz 带通；"
            "该频带对应 21CMA 的科研工作带。"
            if band_MHz is not None
            else "- 未施加带通滤波；结果包含直至 Nyquist 频率的离散轨迹结构。"
        ),
        (
            f"- 分析前将波形按非重叠 bin 守恒平均到 "
            f"{analysis_sampling_rate_GHz:g} GHz。"
            if analysis_sampling_rate_GHz is not None
            else "- 使用模拟文件的原始采样率进行分析。"
        ),
        (
            "- 地磁分量严格复用 `pulse_analysis_modular`："
            r"$E_{\mathbf{v}\times\mathbf{B}}="
            r"\mathbf{E}\cdot\widehat{(\mathbf{v}\times\mathbf{B})}$。"
        ),
        (
            "- 振幅为参考程序 `left-trend` 主脉冲的绝对峰值；"
            "先在每个天线提取，再对同一 shower 的 8 个方位取几何平均。"
        ),
        (
            "- 宽度为参考程序默认的 0.1 ns 插值加 max-Q 方波拟合；"
            "先保留 `valid=True` 的天线，再应用参考绘图程序默认的 "
            "log-MAD/3 倍中位数稳健筛选，最后对每场 shower 取算术平均。"
        ),
        (
            "- 统计单位是 shower，而不是天线；"
            f"CPU 为 {shower_counts['legacy_proposal']} 个、"
            f"CUDA 为 {shower_counts['cuda']} 个；"
            "相同 seed 在两套输运中不会建立严格逐事例配对，故使用两样本检验。"
        ),
        (
            f"- 磁场：{run_configuration['igrf_model']} / "
            f"{run_configuration['igrf_year']:g}，"
            f"`B_NWU = {run_configuration['magnetic_field_T'].tolist()} T`。"
        ),
        f"- 投影单位向量：`e_vxB = {yprime.tolist()}`。",
        (
            f"- 预注册统计门槛：CUDA/CPU 的 bootstrap 95% 区间必须完整落在 "
            f"`[1-{acceptance['relative_equivalence_tolerance']:.3g}, "
            f"1+{acceptance['relative_equivalence_tolerance']:.3g}]` 内，"
            f"四个 KS 形状检验以 "
            f"{acceptance['shape_test_correction']} 控制 family-wise "
            f"$\\alpha={acceptance['shape_test_familywise_alpha']:.3g}$，"
            f"振幅要求每个后端至少有 "
            f"{acceptance['minimum_count_per_backend']} 场有效 shower；"
            f"宽度要求至少有 "
            f"{acceptance['minimum_width_valid_count_per_backend']} 场，"
            f"并覆盖不少于 "
            f"{acceptance['minimum_width_valid_shower_fraction']:.0%} "
            "的输入 shower。"
        ),
        "",
        "## 结果",
        "",
        (
            "| 射电算法 | 指标 | CPU | CUDA | CUDA/CPU "
            "(bootstrap 95% CI) | KS D | KS p |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        amplitude = comparisons[algorithm]["amplitude"]
        width = comparisons[algorithm]["width"]
        lines.extend(
            [
                (
                    f"| {algorithm} | 振幅几何均值 [V/m] | "
                    f"{amplitude['cpu_reported_statistic']:.6e} | "
                    f"{amplitude['cuda_reported_statistic']:.6e} | "
                    f"{format_ratio(amplitude)} | "
                    f"{amplitude['ks_two_sample']['statistic']:.4f} | "
                    f"{amplitude['ks_two_sample']['p_value']:.4g} |"
                ),
                (
                    f"| {algorithm} | 宽度算术均值 [ns] | "
                    f"{width['cpu_reported_statistic']:.6g} | "
                    f"{width['cuda_reported_statistic']:.6g} | "
                    f"{format_ratio(width)} | "
                    f"{width['ks_two_sample']['statistic']:.4f} | "
                    f"{width['ks_two_sample']['p_value']:.4g} |"
                ),
            ]
        )
    raw_valid_rows = [
        row for row in antenna_rows if row["pulse_width_valid"]
    ]
    filtered_rows = [
        row for row in antenna_rows if row["pulse_width_filter_pass"]
    ]
    lines.extend(
        [
            "",
            (
                f"形式验收状态：`{acceptance['status']}`；"
                f"口径：{acceptance['scientific_scope']}。"
            ),
            "",
            (
                f"原始脉冲宽度有效天线数为 "
                f"{len(raw_valid_rows)}/{len(antenna_rows)} "
                f"({len(raw_valid_rows) / len(antenna_rows):.1%})；"
                f"参考绘图筛选后保留 {len(filtered_rows)}/{len(antenna_rows)} "
                f"({len(filtered_rows) / len(antenna_rows):.1%})。"
            ),
            "",
            "![地磁脉冲分布](geomagnetic_pulse_distributions.png)",
            "",
            "## 文件",
            "",
            "- `geomagnetic_pulse_distributions.png`：四面板主比较图。",
            "- `per_antenna_pulse_features.csv`：逐天线提取结果。",
            "- `per_shower_pulse_features.csv`：用于检验的逐 shower 统计量。",
            "- `comparison.json`：完整统计量和运行元数据。",
            "",
            "## 解释限制",
            "",
            (
                "KS 的较大 p 值只能表示当前 "
                f"{shower_counts['legacy_proposal']}+"
                f"{shower_counts['cuda']} 场样本未发现显著分布差异，"
                "不能单独证明两种分布等价。CUDA 与原版 CPU 的 shower history "
                "会因调度、随机数映射及物理采样实现不同而分叉，因此这里检验的是"
                "科研上更关键的 ensemble 射电统计一致性，而非逐 bin 完全相同。"
            ),
            (
                "理想矩形 FFT 带通会产生长时间振铃，而参考 max-Q 方波宽度"
                "估计器是为原始单极脉冲定义的；因此一旦设置带通，宽度只作诊断，"
                "不能通过本脚本的形式验收。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional external run manifest. This keeps pooled source lists "
            "separate from immutable simulation outputs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pulse-analysis-root", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20250730)
    parser.add_argument(
        "--band-low-mhz",
        type=float,
        default=None,
        help="Optional ideal FFT band-pass lower edge before pulse fitting.",
    )
    parser.add_argument(
        "--band-high-mhz",
        type=float,
        default=None,
        help="Optional ideal FFT band-pass upper edge before pulse fitting.",
    )
    parser.add_argument(
        "--analysis-sampling-rate-ghz",
        type=float,
        default=None,
        help=(
            "Optionally average non-overlapping waveform bins to this "
            "lower analysis sampling rate before pulse fitting."
        ),
    )
    parser.add_argument(
        "--equivalence-relative-tolerance",
        type=float,
        default=0.10,
        help=(
            "Require the full bootstrap 95%% ratio interval inside "
            "[1-tolerance, 1+tolerance]."
        ),
    )
    parser.add_argument(
        "--minimum-ks-p-value",
        type=float,
        default=0.05,
        help="Minimum two-sample KS p-value for the shape diagnostic.",
    )
    parser.add_argument(
        "--minimum-count-per-backend",
        type=int,
        default=20,
        help="Minimum number of valid shower-level values per backend.",
    )
    parser.add_argument(
        "--minimum-width-valid-shower-fraction",
        type=float,
        default=0.80,
        help="Minimum fraction of showers retaining a fitted pulse width.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=(
            "Return exit code 2 unless the raw-waveform amplitude and width "
            "equivalence gates pass for both CoREAS and ZHS."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    if (args.band_low_mhz is None) != (args.band_high_mhz is None):
        raise ValueError("both band-pass edges must be specified together")
    band_MHz = (
        None
        if args.band_low_mhz is None
        else (float(args.band_low_mhz), float(args.band_high_mhz))
    )
    if band_MHz is not None and not (
        math.isfinite(band_MHz[0])
        and math.isfinite(band_MHz[1])
        and 0.0 < band_MHz[0] < band_MHz[1]
    ):
        raise ValueError("band-pass edges must be finite, positive and ordered")
    analysis_sampling_rate_GHz = (
        None
        if args.analysis_sampling_rate_ghz is None
        else float(args.analysis_sampling_rate_ghz)
    )
    if analysis_sampling_rate_GHz is not None and not (
        math.isfinite(analysis_sampling_rate_GHz)
        and analysis_sampling_rate_GHz > 0.0
    ):
        raise ValueError("analysis sampling rate must be finite and positive")
    if not (
        math.isfinite(args.equivalence_relative_tolerance)
        and 0.0 < args.equivalence_relative_tolerance < 1.0
    ):
        raise ValueError(
            "--equivalence-relative-tolerance must be between zero and one"
        )
    if not (
        math.isfinite(args.minimum_ks_p_value)
        and 0.0 <= args.minimum_ks_p_value <= 1.0
    ):
        raise ValueError("--minimum-ks-p-value must be in [0, 1]")
    if args.minimum_count_per_backend < 2:
        raise ValueError("--minimum-count-per-backend must be at least two")
    if not (
        math.isfinite(args.minimum_width_valid_shower_fraction)
        and 0.0 < args.minimum_width_valid_shower_fraction <= 1.0
    ):
        raise ValueError(
            "--minimum-width-valid-shower-fraction must be in (0, 1]"
        )
    if args.require_pass and band_MHz is not None:
        raise ValueError(
            "--require-pass cannot be combined with an ideal FFT band-pass: "
            "the reference square-window pulse-width estimator is not "
            "calibrated for the resulting ringing"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (
        build_polarization_basis,
        analyze_pulse_parameters,
        read_radio_records,
        band_limited_waveform,
        robust_pulse_width_mask,
        PulseWidthFilterConfig,
    ) = load_reference_apis(args.pulse_analysis_root.resolve())
    manifest_path = (
        args.manifest.resolve() if args.manifest is not None else None
    )
    run_configuration = load_run_configuration(
        args.dataset.resolve(), manifest_path
    )
    basis = build_polarization_basis(
        run_configuration["theta_deg"],
        run_configuration["phi_deg"],
        run_configuration["magnetic_field_T"],
    )
    antenna_rows: list[dict[str, Any]] = []
    selected_radius = math.nan
    source_directories = {
        backend: backend_output_directories(
            args.dataset, backend, manifest_path
        )
        for backend in BACKENDS
    }
    for backend in BACKENDS:
        for algorithm in ALGORITHMS:
            records: list[dict[str, Any]] = []
            shower_offset = 0
            for output_directory in source_directories[backend]:
                shard_records, _ = read_radio_records(
                    output_directory, algorithm
                )
                local_showers = sorted(
                    {int(record["shower"]) for record in shard_records}
                )
                if local_showers and local_showers != list(
                    range(local_showers[-1] + 1)
                ):
                    raise ValueError(
                        "radio shard shower IDs are not contiguous: "
                        f"{output_directory}"
                    )
                for record in shard_records:
                    item = dict(record)
                    item["shower"] = (
                        int(record["shower"]) + shower_offset
                    )
                    records.append(item)
                if local_showers:
                    shower_offset += local_showers[-1] + 1
            radius = nearest_available_radius(records, args.radius_m)
            if np.isfinite(selected_radius) and not np.isclose(
                selected_radius, radius, atol=1.e-6, rtol=0.0
            ):
                raise ValueError("CPU/CUDA observer radii differ")
            selected_radius = radius
            extracted = extract_antenna_rows(
                records=records,
                backend=backend,
                algorithm=algorithm,
                radius_m=radius,
                yprime=np.asarray(basis["yprime"], dtype=np.float64),
                analyze_pulse_parameters=analyze_pulse_parameters,
                band_limited_waveform=band_limited_waveform,
                band_MHz=band_MHz,
                analysis_sampling_rate_GHz=analysis_sampling_rate_GHz,
            )
            if not extracted:
                raise ValueError(
                    f"no {backend}/{algorithm} observers at {radius:g} m"
                )
            antenna_rows.extend(extracted)
            print(
                f"{backend}/{algorithm}: extracted {len(extracted)} "
                f"antenna pulses at {radius:g} m",
                flush=True,
            )
    width_filter_config = PulseWidthFilterConfig()
    apply_reference_width_filter(
        antenna_rows,
        robust_pulse_width_mask=robust_pulse_width_mask,
        filter_config=width_filter_config,
    )
    shower_rows = aggregate_by_shower(antenna_rows)
    comparisons: dict[str, Any] = {}
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        comparisons[algorithm] = {}
        for metric_index, (metric, column) in enumerate(
            (
                ("amplitude", "geomagnetic_amplitude_geomean_V_per_m"),
                ("width", "pulse_width_mean_ns"),
            )
        ):
            comparisons[algorithm][metric] = distribution_comparison(
                comparison_values(
                    shower_rows, algorithm, "legacy_proposal", column
                ),
                comparison_values(shower_rows, algorithm, "cuda", column),
                metric=metric,
                repetitions=args.bootstrap_repetitions,
                seed=(
                    args.bootstrap_seed
                    + 100 * algorithm_index
                    + metric_index
                ),
            )
    acceptance = build_acceptance(
        comparisons=comparisons,
        shower_rows=shower_rows,
        band_MHz=band_MHz,
        relative_tolerance=args.equivalence_relative_tolerance,
        minimum_ks_p_value=args.minimum_ks_p_value,
        minimum_count=args.minimum_count_per_backend,
        minimum_width_shower_fraction=(
            args.minimum_width_valid_shower_fraction
        ),
    )
    write_csv(args.output / "per_antenna_pulse_features.csv", antenna_rows)
    write_csv(args.output / "per_shower_pulse_features.csv", shower_rows)
    report = {
        "dataset": str(args.dataset.resolve()),
        "radius_m": selected_radius,
        "statistical_unit": "one shower after azimuth aggregation",
        "pulse_analysis_root": str(args.pulse_analysis_root.resolve()),
        "source_directories": {
            backend: [
                str(path) for path in source_directories[backend]
            ]
            for backend in BACKENDS
        },
        "geomagnetic_definition": "EyPrime = E dot unit(v x B)",
        "magnetic_field_T_NWU": run_configuration[
            "magnetic_field_T"
        ].tolist(),
        "geomagnetic_unit_vector_NWU": np.asarray(
            basis["yprime"], dtype=np.float64
        ).tolist(),
        "amplitude_definition": (
            "geometric mean across azimuth antennas of absolute signed "
            "left-trend main-pulse peak"
        ),
        "width_definition": (
            "arithmetic mean across azimuth antennas of the 0.1 ns "
            "interpolated max-Q square-window width after the reference "
            "plotter's default robust filter"
        ),
        "bandpass_MHz": (
            list(band_MHz) if band_MHz is not None else None
        ),
        "analysis_sampling_rate_GHz": analysis_sampling_rate_GHz,
        "width_filter": {
            "source": "pulse_analysis.plotting.stats.robust_pulse_width_mask",
            "enabled": bool(width_filter_config.enabled),
            "group_min_count": int(width_filter_config.group_min_count),
            "group_robust_z": float(width_filter_config.group_robust_z),
            "group_factor": float(width_filter_config.group_factor),
        },
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "comparisons": comparisons,
        "acceptance": acceptance,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_distributions(
        shower_rows=shower_rows,
        comparisons=comparisons,
        output=args.output / "geomagnetic_pulse_distributions.png",
        radius_m=selected_radius,
    )
    write_markdown(
        path=args.output / "README.md",
        dataset=args.dataset.resolve(),
        radius_m=selected_radius,
        run_configuration=run_configuration,
        yprime=np.asarray(basis["yprime"], dtype=np.float64),
        comparisons=comparisons,
        antenna_rows=antenna_rows,
        band_MHz=band_MHz,
        analysis_sampling_rate_GHz=analysis_sampling_rate_GHz,
        acceptance=acceptance,
    )
    print(f"Wrote results to {args.output.resolve()}", flush=True)
    if args.require_pass and not acceptance["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
