#!/usr/bin/env python3
"""Fit geomagnetic radio-pulse scaling laws across CORSIKA ensembles.

This analysis reuses ``pulse_analysis_modular`` for the polarization
projection, main-pulse selection, pulse-width fit, and robust width filter.
It treats one shower as the independent statistical unit.  Antennas at the
same nominal ground radius are first aggregated within each shower.

The fitted laws are:

* coherent energy scaling: A(E, r) proportional to E**alpha;
* pulse-width energy scaling: W(E, r) proportional to E**gamma;
* geomagnetic-angle scaling: A proportional to sin(alpha_B)**eta;
* piecewise radial laws: A(r), W(r) proportional to r**beta.

All exponent intervals use stratified or shower-cluster bootstrap resampling.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


WORKSPACE = Path("/home/yuhanglu/21CMA")
DEFAULT_PULSE_ROOT = (
    WORKSPACE / "python" / "MCMCTidyUp" / "pulse_analysis_modular"
)
BACKENDS = ("legacy_proposal", "cuda")
BACKEND_LABELS = {"legacy_proposal": "Original CPU", "cuda": "CUDA"}
BACKEND_COLORS = {"legacy_proposal": "#1565c0", "cuda": "#d84315"}
ALGORITHMS = ("CoREAS", "ZHS")
RAW_AMPLITUDE_COLUMN = "geomagnetic_amplitude_geomean_V_per_m"
AMPLITUDE_COLUMN = "geomagnetic_coherent_amplitude_V_per_m"
WIDTH_COLUMN = "pulse_width_mean_ns"
ROUND_RADIUS_DIGITS = 5


def finite(values: Iterable[float], positive: bool = False) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    mask = np.isfinite(array)
    if positive:
        mask &= array > 0.0
    return array[mask]


def geometric_mean(values: Iterable[float]) -> float:
    array = finite(values, positive=True)
    if array.size == 0:
        return math.nan
    return float(np.exp(np.mean(np.log(array))))


def load_apis(
    pulse_root: Path,
) -> tuple[
    Callable[..., dict[str, np.ndarray]],
    Callable[..., dict[str, float]],
    Callable[..., tuple[list[dict[str, Any]], np.ndarray]],
    Callable[..., np.ndarray],
    type,
]:
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
        plotting_stats.robust_pulse_width_mask,
        plotting_config.PulseWidthFilterConfig,
    )


def read_dataset_configuration(dataset: Path) -> dict[str, Any]:
    manifest_path = dataset / "run_manifest.json"
    gpu_config_path = dataset / "cuda" / "gpu_em" / "config.yaml"
    if not gpu_config_path.is_file():
        raise FileNotFoundError(gpu_config_path)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = manifest["configuration"]
        configuration_source = str(manifest_path)
    else:
        output_config_path = dataset / "cuda" / "config.yaml"
        if not output_config_path.is_file():
            raise FileNotFoundError(manifest_path)
        output_config = yaml.safe_load(
            output_config_path.read_text(encoding="utf-8")
        )
        tokens = shlex.split(str(output_config["args"]))

        def option_value(*names: str) -> str:
            for name in names:
                if name in tokens:
                    index = tokens.index(name)
                    if index + 1 < len(tokens):
                        return tokens[index + 1]
            raise ValueError(
                f"none of {names} occurs in {output_config_path}"
            )

        configuration = {
            "energy_GeV": float(option_value("-E")),
            "events": int(option_value("-N")),
            "primary_pdg": int(option_value("-p")),
            "zenith_deg": float(option_value("--zenith")),
            "azimuth_deg": float(option_value("--azimuth")),
            "seed": int(option_value("--seed")),
            "IGRF": {"model": "IGRF13", "year": 2025.0},
        }
        configuration_source = (
            f"{output_config_path} command-line fallback"
        )
    gpu_config = yaml.safe_load(gpu_config_path.read_text(encoding="utf-8"))
    magnetic = gpu_config["environment"]["magnetic_field_T"]
    return {
        "dataset": str(dataset.resolve()),
        "dataset_name": dataset.name,
        "configuration_source": configuration_source,
        "energy_GeV": float(configuration["energy_GeV"]),
        "events": int(configuration["events"]),
        "primary_pdg": int(configuration["primary_pdg"]),
        "theta_deg": float(configuration["zenith_deg"]),
        "phi_deg": float(configuration["azimuth_deg"]),
        "seed": int(configuration["seed"]),
        "igrf_model": str(configuration["IGRF"]["model"]),
        "igrf_year": float(configuration["IGRF"]["year"]),
        "magnetic_field_T": np.asarray(
            [magnetic["x"], magnetic["y"], magnetic["z"]],
            dtype=np.float64,
        ),
    }


def extract_dataset(
    *,
    dataset: Path,
    build_polarization_basis: Callable[..., dict[str, np.ndarray]],
    analyze_pulse_parameters: Callable[..., dict[str, float]],
    read_radio_records: Callable[
        ..., tuple[list[dict[str, Any]], np.ndarray]
    ],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = read_dataset_configuration(dataset)
    basis = build_polarization_basis(
        config["theta_deg"],
        config["phi_deg"],
        config["magnetic_field_T"],
    )
    direction = np.asarray(basis["n"], dtype=np.float64)
    yprime = np.asarray(basis["yprime"], dtype=np.float64)
    magnetic = config["magnetic_field_T"]
    config["sin_geomagnetic_angle"] = float(
        np.linalg.norm(np.cross(direction, magnetic))
        / np.linalg.norm(magnetic)
    )
    config["geomagnetic_angle_deg"] = float(
        np.degrees(np.arcsin(config["sin_geomagnetic_angle"]))
    )
    config["geomagnetic_unit_vector_NWU"] = yprime.tolist()
    rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for algorithm in ALGORITHMS:
            records, locations = read_radio_records(
                dataset / backend, algorithm
            )
            relative_locations = np.asarray(locations, dtype=np.float64)
            relative_locations -= relative_locations[0]
            if len(records) == 0:
                raise ValueError(f"{dataset}/{backend}/{algorithm} is empty")
            algorithm_rows: list[dict[str, Any]] = []
            waveform_groups: dict[
                tuple[int, float],
                list[tuple[int, np.ndarray, np.ndarray, float, float]],
            ] = defaultdict(list)
            for record in records:
                observer = int(record["observer"])
                displacement = relative_locations[observer]
                parallel = float(np.dot(displacement, direction))
                shower_plane_vector = displacement - parallel * direction
                shower_plane_radius = float(
                    np.linalg.norm(shower_plane_vector)
                )
                waveform = np.asarray(
                    record["field"] @ yprime, dtype=np.float64
                )
                pulse = analyze_pulse_parameters(
                    np.asarray(record["time_ns"], dtype=np.float64),
                    waveform,
                    ro_m=shower_plane_radius,
                )
                signed_peak = float(pulse["peak_amplitude"])
                width = float(pulse["pulse_width"])
                width_valid = (
                    bool(pulse["valid"])
                    and np.isfinite(width)
                    and width > 0.0
                )
                row_index = len(algorithm_rows)
                algorithm_rows.append(
                    {
                        "dataset": config["dataset_name"],
                        "backend": backend,
                        "algorithm": algorithm,
                        "energy_GeV": config["energy_GeV"],
                        "primary_pdg": config["primary_pdg"],
                        "theta_deg": config["theta_deg"],
                        "phi_deg": config["phi_deg"],
                        "sin_geomagnetic_angle": config[
                            "sin_geomagnetic_angle"
                        ],
                        "geomagnetic_angle_deg": config[
                            "geomagnetic_angle_deg"
                        ],
                        "shower": int(record["shower"]),
                        "observer": observer,
                        "ground_radius_m": float(record["radius_m"]),
                        "shower_plane_radius_m": shower_plane_radius,
                        "geomagnetic_peak_signed_V_per_m": signed_peak,
                        "geomagnetic_peak_abs_V_per_m": abs(signed_peak),
                        "geomagnetic_coherent_peak_signed_V_per_m": math.nan,
                        "geomagnetic_coherent_peak_abs_V_per_m": math.nan,
                        "pulse_width_ns": width,
                        "pulse_width_valid": width_valid,
                        "pulse_width_filter_pass": False,
                        "pulse_method": str(pulse["method"]),
                    }
                )
                waveform_groups[
                    (
                        int(record["shower"]),
                        round(
                            float(record["radius_m"]),
                            ROUND_RADIUS_DIGITS,
                        ),
                    )
                ].append(
                    (
                        row_index,
                        np.asarray(record["time_ns"], dtype=np.float64),
                        waveform,
                        shower_plane_radius,
                        float(pulse["peak_time"]),
                    )
                )
            for group in waveform_groups.values():
                sample_count = group[0][1].size
                if any(item[1].size != sample_count for item in group):
                    raise ValueError("observer waveforms have unequal lengths")
                target_index = sample_count // 2
                aligned_waveforms = []
                plane_radii = []
                for _, times, waveform, plane_radius, peak_time in group:
                    peak_index = int(
                        np.argmin(np.abs(times - peak_time))
                    )
                    shift = target_index - peak_index
                    aligned = np.zeros_like(waveform)
                    source_start = max(0, -shift)
                    destination_start = max(0, shift)
                    count = min(
                        sample_count - source_start,
                        sample_count - destination_start,
                    )
                    if count > 0:
                        aligned[
                            destination_start : destination_start + count
                        ] = waveform[
                            source_start : source_start + count
                        ]
                    aligned_waveforms.append(aligned)
                    if plane_radius > 0.0:
                        plane_radii.append(plane_radius)
                coherent_waveform = np.mean(
                    np.stack(aligned_waveforms), axis=0
                )
                dt_ns = float(np.median(np.diff(group[0][1])))
                centered_times = (
                    np.arange(sample_count, dtype=np.float64) - target_index
                ) * dt_ns
                effective_radius = (
                    geometric_mean(plane_radii)
                    if plane_radii
                    else 0.0
                )
                coherent_pulse = analyze_pulse_parameters(
                    centered_times,
                    coherent_waveform,
                    ro_m=effective_radius,
                )
                coherent_signed = float(coherent_pulse["peak_amplitude"])
                for row_index, _, _, _, _ in group:
                    algorithm_rows[row_index][
                        "geomagnetic_coherent_peak_signed_V_per_m"
                    ] = coherent_signed
                    algorithm_rows[row_index][
                        "geomagnetic_coherent_peak_abs_V_per_m"
                    ] = abs(coherent_signed)
            rows.extend(algorithm_rows)
            print(
                f"{dataset.name}/{backend}/{algorithm}: "
                f"{len(records)} waveforms",
                flush=True,
            )
    return config, rows


def apply_width_filter(
    rows: list[dict[str, Any]],
    *,
    robust_pulse_width_mask: Callable[..., np.ndarray],
    filter_config: Any,
) -> None:
    grouped: dict[tuple[str, str, str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (
            row["dataset"],
            row["backend"],
            row["algorithm"],
            round(row["ground_radius_m"], ROUND_RADIUS_DIGITS),
        )
        grouped[key].append(index)
    for indices in grouped.values():
        widths = np.asarray(
            [
                (
                    rows[index]["pulse_width_ns"]
                    if rows[index]["pulse_width_valid"]
                    else math.nan
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        keep = np.asarray(
            robust_pulse_width_mask(widths, filter_config), dtype=bool
        )
        for index, accepted in zip(indices, keep):
            rows[index]["pulse_width_filter_pass"] = bool(accepted)


def aggregate_showers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int, float], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["backend"],
                row["algorithm"],
                row["shower"],
                round(row["ground_radius_m"], ROUND_RADIUS_DIGITS),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        dataset, backend, algorithm, shower, ground_radius = key
        amplitudes = finite(
            (row["geomagnetic_peak_abs_V_per_m"] for row in group),
            positive=True,
        )
        coherent_amplitudes = finite(
            (
                row["geomagnetic_coherent_peak_abs_V_per_m"]
                for row in group
            ),
            positive=True,
        )
        raw_widths = finite(
            (
                row["pulse_width_ns"]
                for row in group
                if row["pulse_width_valid"]
            ),
            positive=True,
        )
        widths = finite(
            (
                row["pulse_width_ns"]
                for row in group
                if row["pulse_width_filter_pass"]
            ),
            positive=True,
        )
        plane_radii = finite(
            (row["shower_plane_radius_m"] for row in group),
            positive=True,
        )
        first = group[0]
        output.append(
            {
                "dataset": dataset,
                "backend": backend,
                "algorithm": algorithm,
                "energy_GeV": first["energy_GeV"],
                "primary_pdg": first["primary_pdg"],
                "theta_deg": first["theta_deg"],
                "phi_deg": first["phi_deg"],
                "sin_geomagnetic_angle": first[
                    "sin_geomagnetic_angle"
                ],
                "geomagnetic_angle_deg": first["geomagnetic_angle_deg"],
                "shower": shower,
                "ground_radius_m": ground_radius,
                "effective_shower_plane_radius_m": (
                    geometric_mean(plane_radii)
                    if plane_radii.size
                    else 0.0
                ),
                "antenna_count": len(group),
                "amplitude_valid_count": int(amplitudes.size),
                RAW_AMPLITUDE_COLUMN: geometric_mean(amplitudes),
                AMPLITUDE_COLUMN: geometric_mean(coherent_amplitudes),
                "geomagnetic_amplitude_median_V_per_m": (
                    float(np.median(amplitudes))
                    if amplitudes.size
                    else math.nan
                ),
                "pulse_width_raw_valid_count": int(raw_widths.size),
                "pulse_width_filter_pass_count": int(widths.size),
                WIDTH_COLUMN: (
                    float(np.mean(widths)) if widths.size else math.nan
                ),
                "pulse_width_median_ns": (
                    float(np.median(widths)) if widths.size else math.nan
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linear_fit(x: np.ndarray, log_y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(x.size), x])
    coefficients, _, _, _ = np.linalg.lstsq(
        design, log_y, rcond=None
    )
    prediction = design @ coefficients
    residual = log_y - prediction
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    squared_error = float(np.sum(residual**2))
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "r_squared": (
            1.0 - squared_error / total if total > 0.0 else math.nan
        ),
        "rmse_log": float(np.sqrt(np.mean(residual**2))),
    }


def summarize_group(x: float, values: np.ndarray) -> dict[str, Any]:
    return {
        "x": float(x),
        "count": int(values.size),
        "geometric_mean": geometric_mean(values),
        "median": float(np.median(values)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
    }


def group_power_fit(
    rows: list[dict[str, Any]],
    *,
    x_column: str,
    y_column: str,
    pivot: float,
    repetitions: int,
    seed: int,
    expected_slope: Optional[float] = None,
) -> tuple[dict[str, Any], np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    total_by_x: dict[float, int] = defaultdict(int)
    width_pass_fractions: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = float(row[x_column])
        y = float(row[y_column])
        if not np.isfinite(x) or x <= 0.0:
            continue
        rounded_x = round(x, 12)
        total_by_x[rounded_x] += 1
        if y_column == WIDTH_COLUMN:
            antenna_count = max(int(row.get("antenna_count", 0)), 1)
            width_pass_fractions[rounded_x].append(
                float(row.get("pulse_width_filter_pass_count", 0))
                / antenna_count
            )
        if np.isfinite(y) and y > 0.0:
            grouped[rounded_x].append(y)
    if len(grouped) < 3:
        return {
            "available": False,
            "reason": "fewer than three distinct parameter values",
        }, np.asarray([], dtype=np.float64)
    xs = np.asarray(sorted(grouped), dtype=np.float64)
    value_arrays = [
        np.asarray(grouped[x], dtype=np.float64) for x in xs
    ]
    log_centers = np.asarray(
        [np.mean(np.log(values)) for values in value_arrays]
    )
    central_local_slopes = np.diff(log_centers) / np.diff(np.log(xs))
    central = linear_fit(np.log(xs / pivot), log_centers)
    rng = np.random.default_rng(seed)
    slopes = np.empty(repetitions, dtype=np.float64)
    normalizations = np.empty(repetitions, dtype=np.float64)
    local_slope_samples = np.empty(
        (repetitions, xs.size - 1), dtype=np.float64
    )
    for index in range(repetitions):
        sample_centers = np.asarray(
            [
                np.mean(
                    np.log(
                        values[
                            rng.integers(
                                0, values.size, size=values.size
                            )
                        ]
                    )
                )
                for values in value_arrays
            ]
        )
        fit = linear_fit(np.log(xs / pivot), sample_centers)
        slopes[index] = fit["slope"]
        normalizations[index] = math.exp(fit["intercept"])
        local_slope_samples[index] = np.diff(sample_centers) / np.diff(
            np.log(xs)
        )
    group_summaries = []
    for x, values in zip(xs, value_arrays):
        summary = summarize_group(x, values)
        summary["total_shower_count"] = int(total_by_x[x])
        summary["usable_shower_fraction"] = float(
            values.size / total_by_x[x]
        )
        if y_column == WIDTH_COLUMN:
            fractions = np.asarray(
                width_pass_fractions[x], dtype=np.float64
            )
            summary["median_antenna_width_pass_fraction"] = float(
                np.median(fractions)
            )
        group_summaries.append(summary)
    width_quality_gate = True
    if y_column == WIDTH_COLUMN:
        width_quality_gate = (
            not (set(total_by_x) - set(grouped))
            and all(
                group["usable_shower_fraction"] >= 0.80
                and group["median_antenna_width_pass_fraction"] >= 0.50
                for group in group_summaries
            )
        )
    result: dict[str, Any] = {
        "available": True,
        "law": f"{y_column} = normalization * ({x_column}/{pivot:g})^slope",
        "x_column": x_column,
        "y_column": y_column,
        "pivot": pivot,
        "slope": central["slope"],
        "slope_ci95": [
            float(np.quantile(slopes, 0.025)),
            float(np.quantile(slopes, 0.975)),
        ],
        "normalization_at_pivot": math.exp(central["intercept"]),
        "normalization_ci95": [
            float(np.quantile(normalizations, 0.025)),
            float(np.quantile(normalizations, 0.975)),
        ],
        "r_squared_group_centers": central["r_squared"],
        "rmse_log_group_centers": central["rmse_log"],
        "single_power_law_model_quality": (
            "adequate"
            if central["r_squared"] >= 0.90
            else "inadequate_r_squared_below_0.90"
        ),
        "groups": group_summaries,
        "parameter_values_without_usable_measurements": sorted(
            set(total_by_x) - set(grouped)
        ),
        "adjacent_local_slopes": [
            {
                "x_low": float(xs[index]),
                "x_high": float(xs[index + 1]),
                "slope": float(
                    central_local_slopes[index]
                ),
                "ci95": [
                    float(
                        np.quantile(
                            local_slope_samples[:, index], 0.025
                        )
                    ),
                    float(
                        np.quantile(
                            local_slope_samples[:, index], 0.975
                        )
                    ),
                ],
            }
            for index in range(xs.size - 1)
        ],
        "low_x_minus_high_x_local_slope": (
            {
                "difference": float(
                    central_local_slopes[0] - central_local_slopes[-1]
                ),
                "ci95": [
                    float(
                        np.quantile(
                            local_slope_samples[:, 0]
                            - local_slope_samples[:, -1],
                            0.025,
                        )
                    ),
                    float(
                        np.quantile(
                            local_slope_samples[:, 0]
                            - local_slope_samples[:, -1],
                            0.975,
                        )
                    ),
                ],
            }
            if xs.size >= 3
            else None
        ),
        "scientific_status": (
            (
                "accepted"
                if central["r_squared"] >= 0.90
                else "diagnostic_only_due_to_single_power_law_misfit"
            )
            if width_quality_gate
            else "diagnostic_only_due_to_width_coverage"
        ),
    }
    if expected_slope is not None:
        lower, upper = result["slope_ci95"]
        result["expected_slope"] = expected_slope
        result["expected_inside_ci95"] = (
            lower <= expected_slope <= upper
        )
        tail = min(
            float(np.mean(slopes <= expected_slope)),
            float(np.mean(slopes >= expected_slope)),
        )
        result["bootstrap_two_sided_p_for_expected"] = min(1.0, 2.0 * tail)
    return result, slopes


def radial_power_fit(
    rows: list[dict[str, Any]],
    *,
    y_column: str,
    radius_low_m: float,
    radius_high_m: float,
    pivot_m: float,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    candidates = [
        row
        for row in rows
        if radius_low_m - 1.e-3
        <= float(row["effective_shower_plane_radius_m"])
        <= radius_high_m + 1.e-3
    ]
    selected = [
        row
        for row in candidates
        if np.isfinite(float(row[y_column]))
        and float(row[y_column]) > 0.0
    ]
    by_shower: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_shower[int(row["shower"])].append(row)
    radii = sorted(
        {
            round(float(row["effective_shower_plane_radius_m"]), 8)
            for row in selected
        }
    )
    if len(by_shower) < 3 or len(radii) < 3:
        return {
            "available": False,
            "reason": "insufficient showers or radii",
        }, np.asarray([], dtype=np.float64)

    def fit_from_rows(sample: list[dict[str, Any]]) -> dict[str, float]:
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in sample:
            grouped[
                round(
                    float(row["effective_shower_plane_radius_m"]), 8
                )
            ].append(float(row[y_column]))
        xs = np.asarray(sorted(grouped), dtype=np.float64)
        log_y = np.asarray(
            [np.mean(np.log(grouped[x])) for x in xs],
            dtype=np.float64,
        )
        return linear_fit(np.log(xs / pivot_m), log_y)

    central = fit_from_rows(selected)
    shower_ids = np.asarray(sorted(by_shower), dtype=int)
    rng = np.random.default_rng(seed)
    slopes = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_ids = shower_ids[
            rng.integers(0, shower_ids.size, size=shower_ids.size)
        ]
        sample: list[dict[str, Any]] = []
        for shower in sampled_ids:
            sample.extend(by_shower[int(shower)])
        slopes[index] = fit_from_rows(sample)["slope"]
    grouped_values: dict[float, list[float]] = defaultdict(list)
    for row in selected:
        grouped_values[
            round(float(row["effective_shower_plane_radius_m"]), 8)
        ].append(float(row[y_column]))
    candidate_counts: dict[float, int] = defaultdict(int)
    candidate_width_pass_fractions: dict[float, list[float]] = defaultdict(
        list
    )
    for row in candidates:
        radius = round(
            float(row["effective_shower_plane_radius_m"]), 8
        )
        candidate_counts[radius] += 1
        if y_column == WIDTH_COLUMN:
            candidate_width_pass_fractions[radius].append(
                float(row.get("pulse_width_filter_pass_count", 0))
                / max(int(row.get("antenna_count", 0)), 1)
            )
    group_summaries = []
    for radius in sorted(grouped_values):
        summary = summarize_group(
            radius, np.asarray(grouped_values[radius], dtype=np.float64)
        )
        summary["total_shower_count"] = candidate_counts[radius]
        summary["usable_shower_fraction"] = (
            summary["count"] / candidate_counts[radius]
        )
        if y_column == WIDTH_COLUMN:
            summary["median_antenna_width_pass_fraction"] = float(
                np.median(candidate_width_pass_fractions[radius])
            )
        group_summaries.append(summary)
    width_quality_gate = True
    if y_column == WIDTH_COLUMN:
        width_quality_gate = all(
            group["usable_shower_fraction"] >= 0.80
            and group["median_antenna_width_pass_fraction"] >= 0.50
            for group in group_summaries
        )
    result = {
        "available": True,
        "law": f"{y_column} proportional to r^slope",
        "y_column": y_column,
        "radius_range_m": [radius_low_m, radius_high_m],
        "pivot_m": pivot_m,
        "slope": central["slope"],
        "slope_ci95": [
            float(np.quantile(slopes, 0.025)),
            float(np.quantile(slopes, 0.975)),
        ],
        "r_squared_group_centers": central["r_squared"],
        "rmse_log_group_centers": central["rmse_log"],
        "single_power_law_model_quality": (
            "adequate"
            if central["r_squared"] >= 0.90
            else "inadequate_r_squared_below_0.90"
        ),
        "scientific_status": (
            (
                "accepted"
                if central["r_squared"] >= 0.90
                else "diagnostic_only_due_to_single_power_law_misfit"
            )
            if width_quality_gate
            else "diagnostic_only_due_to_width_coverage"
        ),
        "shower_count": int(shower_ids.size),
        "groups": group_summaries,
    }
    return result, slopes


def slope_difference(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    if reference.size == 0 or candidate.size == 0:
        return {"available": False}
    size = min(reference.size, candidate.size)
    difference = candidate[:size] - reference[:size]
    return {
        "available": True,
        "cuda_minus_cpu": float(
            np.median(candidate) - np.median(reference)
        ),
        "ci95": [
            float(np.quantile(difference, 0.025)),
            float(np.quantile(difference, 0.975)),
        ],
        "zero_inside_ci95": bool(
            np.quantile(difference, 0.025)
            <= 0.0
            <= np.quantile(difference, 0.975)
        ),
    }


def select_rows(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    algorithm: str,
    primary_pdg: int = 11,
    energy_GeV: Optional[float] = None,
    theta_deg: Optional[float] = None,
    phi_values: Optional[set[float]] = None,
    ground_radius_m: Optional[float] = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["backend"] != backend or row["algorithm"] != algorithm:
            continue
        if int(row["primary_pdg"]) != primary_pdg:
            continue
        if energy_GeV is not None and not np.isclose(
            row["energy_GeV"], energy_GeV
        ):
            continue
        if theta_deg is not None and not np.isclose(
            row["theta_deg"], theta_deg
        ):
            continue
        if phi_values is not None and not any(
            np.isclose(row["phi_deg"], phi) for phi in phi_values
        ):
            continue
        if ground_radius_m is not None and not np.isclose(
            row["ground_radius_m"], ground_radius_m, atol=1.e-4
        ):
            continue
        selected.append(row)
    return selected


def fit_all_laws(
    rows: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "energy": {},
        "geomagnetic_angle": {},
        "radial": {},
    }
    metric_specs = (
        ("amplitude", AMPLITUDE_COLUMN, 1.0),
        ("width", WIDTH_COLUMN, 0.0),
    )
    fit_counter = 0
    for algorithm in ALGORITHMS:
        report["energy"][algorithm] = {}
        report["geomagnetic_angle"][algorithm] = {}
        report["radial"][algorithm] = {}
        for metric, column, expected in metric_specs:
            report["energy"][algorithm][metric] = {}
            report["geomagnetic_angle"][algorithm][metric] = {}
            energy_bootstrap: dict[str, np.ndarray] = {}
            angle_bootstrap: dict[str, np.ndarray] = {}
            for backend in BACKENDS:
                energy_rows = select_rows(
                    rows,
                    backend=backend,
                    algorithm=algorithm,
                    theta_deg=0.0,
                    ground_radius_m=100.0,
                )
                result, boot = group_power_fit(
                    energy_rows,
                    x_column="energy_GeV",
                    y_column=column,
                    pivot=1000.0,
                    repetitions=repetitions,
                    seed=seed + fit_counter,
                    expected_slope=expected,
                )
                fit_counter += 1
                report["energy"][algorithm][metric][backend] = result
                energy_bootstrap[backend] = boot

                angle_rows = select_rows(
                    rows,
                    backend=backend,
                    algorithm=algorithm,
                    energy_GeV=1000.0,
                    theta_deg=45.0,
                    phi_values={0.0, 90.0, 180.0},
                    ground_radius_m=100.0,
                )
                result, boot = group_power_fit(
                    angle_rows,
                    x_column="sin_geomagnetic_angle",
                    y_column=column,
                    pivot=1.0,
                    repetitions=repetitions,
                    seed=seed + fit_counter,
                    expected_slope=expected,
                )
                fit_counter += 1
                report["geomagnetic_angle"][algorithm][metric][
                    backend
                ] = result
                angle_bootstrap[backend] = boot
            report["energy"][algorithm][metric][
                "cuda_minus_cpu_slope"
            ] = slope_difference(
                energy_bootstrap["legacy_proposal"],
                energy_bootstrap["cuda"],
            )
            report["geomagnetic_angle"][algorithm][metric][
                "cuda_minus_cpu_slope"
            ] = slope_difference(
                angle_bootstrap["legacy_proposal"],
                angle_bootstrap["cuda"],
            )

        anchor_rows = [
            row
            for row in rows
            if int(row["primary_pdg"]) == 11
            and np.isclose(row["energy_GeV"], 1000.0)
            and np.isclose(row["theta_deg"], 0.0)
        ]
        for metric, column, _ in metric_specs:
            report["radial"][algorithm][metric] = {}
            for range_name, low, high, pivot in (
                ("inner_50_200m", 50.0, 200.0, 100.0),
                ("outer_200_600m", 200.0, 600.0, 300.0),
            ):
                report["radial"][algorithm][metric][range_name] = {}
                bootstraps: dict[str, np.ndarray] = {}
                for backend in BACKENDS:
                    selected = select_rows(
                        anchor_rows,
                        backend=backend,
                        algorithm=algorithm,
                    )
                    result, boot = radial_power_fit(
                        selected,
                        y_column=column,
                        radius_low_m=low,
                        radius_high_m=high,
                        pivot_m=pivot,
                        repetitions=repetitions,
                        seed=seed + fit_counter,
                    )
                    fit_counter += 1
                    report["radial"][algorithm][metric][range_name][
                        backend
                    ] = result
                    bootstraps[backend] = boot
                report["radial"][algorithm][metric][range_name][
                    "cuda_minus_cpu_slope"
                ] = slope_difference(
                    bootstraps["legacy_proposal"],
                    bootstraps["cuda"],
                )
    return report


def plot_power_panel(
    axis: Any,
    fits: dict[str, Any],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for backend in BACKENDS:
        fit = fits[backend]
        if not fit.get("available", False):
            continue
        groups = fit["groups"]
        x = np.asarray([group["x"] for group in groups])
        y = np.asarray([group["geometric_mean"] for group in groups])
        low = np.asarray([group["q16"] for group in groups])
        high = np.asarray([group["q84"] for group in groups])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="o",
            capsize=3,
            color=BACKEND_COLORS[backend],
            label=(
                f"{BACKEND_LABELS[backend]}: "
                f"{fit['slope']:.3f} "
                f"[{fit['slope_ci95'][0]:.3f}, "
                f"{fit['slope_ci95'][1]:.3f}]"
                + (
                    ""
                    if fit.get("scientific_status") == "accepted"
                    else " (diagnostic)"
                )
            ),
        )
        grid = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
        prediction = fit["normalization_at_pivot"] * (
            grid / fit["pivot"]
        ) ** fit["slope"]
        axis.plot(
            grid,
            prediction,
            color=BACKEND_COLORS[backend],
            lw=1.8,
            ls=(
                "-"
                if fit.get("scientific_status") == "accepted"
                else "--"
            ),
        )
    reference_fit = fits.get("legacy_proposal", {})
    if reference_fit.get("available", False) and (
        "expected_slope" in reference_fit
    ):
        groups = reference_fit["groups"]
        expected = float(reference_fit["expected_slope"])
        x = np.asarray([group["x"] for group in groups])
        log_y = np.log(
            np.asarray([group["geometric_mean"] for group in groups])
        )
        pivot = float(reference_fit["pivot"])
        fixed_intercept = float(
            np.mean(log_y - expected * np.log(x / pivot))
        )
        grid = np.geomspace(float(np.min(x)), float(np.max(x)), 200)
        axis.plot(
            grid,
            np.exp(fixed_intercept) * (grid / pivot) ** expected,
            color="0.25",
            lw=1.2,
            ls=":",
            label=f"reference slope = {expected:g}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8.5, frameon=False)


def plot_energy_and_angle(report: dict[str, Any], output: Path) -> None:
    for law_name, xlabel, filename in (
        (
            "energy",
            "Primary energy [GeV]",
            "energy_scaling.png",
        ),
        (
            "geomagnetic_angle",
            r"$\sin\alpha_B$",
            "geomagnetic_angle_scaling.png",
        ),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
        for row_index, algorithm in enumerate(ALGORITHMS):
            for column_index, (metric, ylabel) in enumerate(
                (
                    (
                        "amplitude",
                        r"$|E_{\mathbf{v}\times\mathbf{B}}|$ [V/m]",
                    ),
                    ("width", "Pulse width [ns]"),
                )
            ):
                plot_power_panel(
                    axes[row_index, column_index],
                    report[law_name][algorithm][metric],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=f"{algorithm}: {metric}",
                )
        subtitle = (
            "100 m, vertical electron showers"
            if law_name == "energy"
            else "1 TeV electrons, zenith 45°, 100 m"
        )
        figure.suptitle(
            f"Geomagnetic pulse {law_name.replace('_', ' ')} law\n{subtitle}",
            fontsize=14,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(output / filename, dpi=200, bbox_inches="tight")
        plt.close(figure)


def plot_radial(
    rows: list[dict[str, Any]], report: dict[str, Any], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (metric, column, ylabel) in enumerate(
            (
                (
                    "amplitude",
                    AMPLITUDE_COLUMN,
                    r"$|E_{\mathbf{v}\times\mathbf{B}}|$ [V/m]",
                ),
                ("width", WIDTH_COLUMN, "Pulse width [ns]"),
            )
        ):
            axis = axes[row_index, column_index]
            for backend in BACKENDS:
                selected = select_rows(
                    rows,
                    backend=backend,
                    algorithm=algorithm,
                    energy_GeV=1000.0,
                    theta_deg=0.0,
                )
                grouped: dict[float, list[float]] = defaultdict(list)
                for item in selected:
                    radius = item["effective_shower_plane_radius_m"]
                    value = item[column]
                    if (
                        radius > 0.0
                        and np.isfinite(value)
                        and value > 0.0
                    ):
                        grouped[round(radius, 6)].append(value)
                radii = np.asarray(sorted(grouped))
                centers = np.asarray(
                    [geometric_mean(grouped[radius]) for radius in radii]
                )
                q16 = np.asarray(
                    [np.quantile(grouped[radius], 0.16) for radius in radii]
                )
                q84 = np.asarray(
                    [np.quantile(grouped[radius], 0.84) for radius in radii]
                )
                axis.errorbar(
                    radii,
                    centers,
                    yerr=np.vstack([centers - q16, q84 - centers]),
                    fmt="o-",
                    ms=4,
                    capsize=2,
                    color=BACKEND_COLORS[backend],
                    label=BACKEND_LABELS[backend],
                )
            axis.axvline(200.0, color="0.5", ls=":", lw=1.0)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Shower-plane radius [m]")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{algorithm}: {metric}")
            axis.grid(alpha=0.25, which="both")
            axis.legend(frameon=False)
            outer = report["radial"][algorithm][metric][
                "outer_200_600m"
            ]
            text_lines = []
            for backend in BACKENDS:
                fit = outer[backend]
                if fit.get("available", False):
                    status = (
                        ""
                        if fit.get("scientific_status") == "accepted"
                        else " (diagnostic)"
                    )
                    text_lines.append(
                        f"{BACKEND_LABELS[backend]} "
                        f"$\\beta_{{200-600}}$={fit['slope']:.2f}"
                        f"{status}"
                    )
            axis.text(
                0.98,
                0.04,
                "\n".join(text_lines),
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
            )
    figure.suptitle(
        "Radial geomagnetic-pulse scaling\n1 TeV vertical electron showers",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output / "radial_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def exponent_table_lines(
    report: dict[str, Any], law_name: str
) -> list[str]:
    lines = [
        "| 算法 | 指标 | CPU 指数 [95% CI] | CUDA 指数 [95% CI] | "
        "CUDA−CPU 指数 [95% CI] | 解释状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for algorithm in ALGORITHMS:
        for metric in ("amplitude", "width"):
            fits = report[law_name][algorithm][metric]
            cpu = fits["legacy_proposal"]
            cuda = fits["cuda"]
            difference = fits["cuda_minus_cpu_slope"]
            if not cpu.get("available") or not cuda.get("available"):
                lines.append(
                    f"| {algorithm} | {metric} | unavailable | "
                    "unavailable | unavailable | unavailable |"
                )
                continue
            lines.append(
                f"| {algorithm} | {metric} | "
                f"{cpu['slope']:.4f} "
                f"[{cpu['slope_ci95'][0]:.4f}, "
                f"{cpu['slope_ci95'][1]:.4f}] | "
                f"{cuda['slope']:.4f} "
                f"[{cuda['slope_ci95'][0]:.4f}, "
                f"{cuda['slope_ci95'][1]:.4f}] | "
                f"{difference['cuda_minus_cpu']:.4f} "
                f"[{difference['ci95'][0]:.4f}, "
                f"{difference['ci95'][1]:.4f}] | "
                f"{cpu.get('scientific_status', 'unknown')} / "
                f"{cuda.get('scientific_status', 'unknown')} |"
            )
    return lines


def radial_table_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "| 算法 | 指标 | 区间 | CPU 指数 [95% CI] | "
        "CUDA 指数 [95% CI] | CUDA−CPU [95% CI] | 解释状态 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for algorithm in ALGORITHMS:
        for metric in ("amplitude", "width"):
            for range_name in ("inner_50_200m", "outer_200_600m"):
                fits = report["radial"][algorithm][metric][range_name]
                cpu = fits["legacy_proposal"]
                cuda = fits["cuda"]
                difference = fits["cuda_minus_cpu_slope"]
                label = range_name.replace("inner_", "").replace(
                    "outer_", ""
                )
                if not cpu.get("available") or not cuda.get("available"):
                    lines.append(
                        f"| {algorithm} | {metric} | {label} | "
                        "unavailable | unavailable | unavailable | unavailable |"
                    )
                    continue
                lines.append(
                    f"| {algorithm} | {metric} | {label} | "
                    f"{cpu['slope']:.4f} "
                    f"[{cpu['slope_ci95'][0]:.4f}, "
                    f"{cpu['slope_ci95'][1]:.4f}] | "
                    f"{cuda['slope']:.4f} "
                    f"[{cuda['slope_ci95'][0]:.4f}, "
                    f"{cuda['slope_ci95'][1]:.4f}] | "
                    f"{difference['cuda_minus_cpu']:.4f} "
                    f"[{difference['ci95'][0]:.4f}, "
                    f"{difference['ci95'][1]:.4f}] | "
                    f"{cpu.get('scientific_status', 'unknown')} / "
                    f"{cuda.get('scientific_status', 'unknown')} |"
                )
    return lines


def write_markdown(
    *,
    path: Path,
    configs: list[dict[str, Any]],
    report: dict[str, Any],
    antenna_rows: list[dict[str, Any]],
    shower_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# CPU/CUDA 地磁射电脉冲 scaling-law 分析",
        "",
        "## 数据与方法",
        "",
        "| 数据集 | E [GeV] | N | zenith | azimuth | sin(alpha_B) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in sorted(
        configs,
        key=lambda item: (
            item["theta_deg"],
            item["energy_GeV"],
            item["phi_deg"],
        ),
    ):
        lines.append(
            f"| {config['dataset_name']} | {config['energy_GeV']:g} | "
            f"{config['events']} | {config['theta_deg']:g}° | "
            f"{config['phi_deg']:g}° | "
            f"{config['sin_geomagnetic_angle']:.5f} |"
        )
    raw_width_count = sum(row["pulse_width_valid"] for row in antenna_rows)
    filtered_width_count = sum(
        row["pulse_width_filter_pass"] for row in antenna_rows
    )
    lines.extend(
        [
            "",
            (
                "- 地磁分量、主峰与宽度完全复用 "
                "`pulse_analysis_modular`；宽度还使用其默认稳健筛选。"
            ),
            (
                "- 用于 scaling 的地磁振幅不是八个绝对峰值的平均：先将"
                "同一 shower、同一半径的八条 "
                r"$E_{\mathbf v\times\mathbf B}$ 波形按各自主峰无回卷对齐，"
                "再作有符号相干平均并提取峰值。这使径向电荷过剩分量在"
                "相反方位相消，避免绝对值平均制造不随 "
                r"$\sin\alpha_B$ 缩放的底噪。"
            ),
            (
                "- 先在每场 shower、每个名义地面半径内聚合八个方位天线；"
                "bootstrap 以 shower 为重采样单位。"
            ),
            (
                f"- 逐天线记录 {len(antenna_rows)} 条，逐 shower-半径记录 "
                f"{len(shower_rows)} 条；原始有效宽度 {raw_width_count} 条，"
                f"筛选后 {filtered_width_count} 条。"
            ),
            (
                "- 能量律在垂直电子 shower 的 100 m 处拟合；地磁角律在 "
                "1 TeV、45°、100 m 处拟合；半径律使用 1 TeV 垂直样本。"
            ),
            "",
            "## 能量 scaling",
            "",
            (
                "拟合形式为振幅 $A\\propto E^\\alpha$、"
                "宽度 $W\\propto E^\\gamma$。完全相干辐射预期 "
                "$\\alpha\\simeq1$，而宽度的一阶预期为 $\\gamma\\simeq0$。"
            ),
            "",
        ]
    )
    lines.extend(exponent_table_lines(report, "energy"))
    lines.extend(
        [
            "",
            "![能量 scaling](energy_scaling.png)",
            "",
            "## 地磁角 scaling",
            "",
            (
                "拟合形式为 $A\\propto(\\sin\\alpha_B)^\\eta$。"
                "纯地磁辐射的一阶预期是 $\\eta\\simeq1$；"
                "宽度不应具有同样的线性振幅因子。"
            ),
            "",
        ]
    )
    lines.extend(exponent_table_lines(report, "geomagnetic_angle"))
    lines.extend(
        [
            "",
            "![地磁角 scaling](geomagnetic_angle_scaling.png)",
            "",
            "## 半径 scaling",
            "",
            (
                "射电 footprint 受 Cherenkov 压缩影响，不应先验假定全半径"
                "只有一个幂律。因此分别报告 50–200 m 与 200–600 m 的"
                "局部幂律指数，并在图中保留全部半径点。"
            ),
            "",
        ]
    )
    lines.extend(radial_table_lines(report))
    lines.extend(
        [
            "",
            "![半径 scaling](radial_scaling.png)",
            "",
            "## 解释原则",
            "",
            (
                "指数的 CPU/CUDA 差异只有在其 bootstrap 95% 区间不含零时"
                "才视为有证据；单个后端的指数置信区间包含理论值只表示"
                "与理论相容，不构成理论已被证明。三个地磁角点能够检验"
                "一阶趋势，但不足以排除高阶角度项；半径分段结果也不能"
                "外推到阵列范围之外。"
            ),
            "",
            "## 可复现文件",
            "",
            "- `per_antenna_features.csv`：逐天线原始脉冲参数。",
            "- `per_shower_radius_features.csv`：拟合实际使用的独立样本。",
            "- `scaling_laws.json`：指数、置信区间、残差和组统计量。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        required=True,
        help="Repeat for each complete legacy_proposal/cuda radio ensemble.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pulse-analysis-root", type=Path, default=DEFAULT_PULSE_ROOT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 500:
        raise ValueError("at least 500 bootstrap repetitions are required")
    args.output.mkdir(parents=True, exist_ok=True)
    (
        build_polarization_basis,
        analyze_pulse_parameters,
        read_radio_records,
        robust_pulse_width_mask,
        PulseWidthFilterConfig,
    ) = load_apis(args.pulse_analysis_root.resolve())
    configs: list[dict[str, Any]] = []
    antenna_rows: list[dict[str, Any]] = []
    for dataset in args.dataset:
        config, rows = extract_dataset(
            dataset=dataset.resolve(),
            build_polarization_basis=build_polarization_basis,
            analyze_pulse_parameters=analyze_pulse_parameters,
            read_radio_records=read_radio_records,
        )
        configs.append(config)
        antenna_rows.extend(rows)
    filter_config = PulseWidthFilterConfig()
    apply_width_filter(
        antenna_rows,
        robust_pulse_width_mask=robust_pulse_width_mask,
        filter_config=filter_config,
    )
    shower_rows = aggregate_showers(antenna_rows)
    report = fit_all_laws(
        shower_rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    output_report = {
        "datasets": [
            {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in config.items()
            }
            for config in configs
        ],
        "statistical_unit": "shower after azimuth aggregation",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "width_filter": {
            "source": "pulse_analysis.plotting.stats.robust_pulse_width_mask",
            "group_robust_z": filter_config.group_robust_z,
            "group_factor": filter_config.group_factor,
        },
        "laws": report,
    }
    write_csv(args.output / "per_antenna_features.csv", antenna_rows)
    write_csv(
        args.output / "per_shower_radius_features.csv", shower_rows
    )
    (args.output / "scaling_laws.json").write_text(
        json.dumps(output_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_energy_and_angle(report, args.output)
    plot_radial(shower_rows, report, args.output)
    write_markdown(
        path=args.output / "README.md",
        configs=configs,
        report=report,
        antenna_rows=antenna_rows,
        shower_rows=shower_rows,
    )
    print(f"Wrote scaling analysis to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
