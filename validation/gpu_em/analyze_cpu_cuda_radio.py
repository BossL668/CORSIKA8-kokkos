#!/usr/bin/env python3
"""Diagnose CPU-PROPOSAL versus CUDA-EM radio ensembles.

Unlike a same-track radio-backend test, CPU and CUDA transport generally
produce different showers even when the integer seed is the same.  This tool
therefore compares distributions and ensemble pulse templates, not raw
event-paired waveform bins.  It reports:

* band-limited radiation-energy proxies and bootstrap confidence intervals;
* pulse amplitude, time, width, spectral centroid and window-edge leakage;
* peak-aligned, unit-energy time-domain envelope similarity by antenna radius;
* band-integrated polarization fractions and Stokes parameters;
* normalized frequency-spectrum and radial-footprint shape similarity;
* correlations with deposited energy and Xmax;
* CoREAS/ZHS track diagnostics when RadioProcess summary files are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import yaml

from compare_cuda_replay import (
    EPSILON_0_F_PER_M,
    LIGHT_SPEED_M_PER_S,
    RADIAL_GROUP_DECIMALS,
    energy_observables,
    ks_distance,
    radiation_energy_proxy,
    radio_fluence_maps,
    radio_observer_layout,
    shower_key_index,
)


ALGORITHMS = ("CoREAS", "ZHS")
BANDS = ((30.0, 80.0), (50.0, 350.0))
FEATURES = (
    "peak_field_per_GeV",
    "peak_offset_ns",
    "centroid_offset_ns",
    "rms_width_ns",
    "spectral_centroid_MHz",
    "edge_energy_fraction",
    "fluence_J_per_m2_per_GeV2",
    "fluence_x_fraction",
    "fluence_y_fraction",
    "fluence_z_fraction",
    "stokes_q_over_i",
    "stokes_u_over_i",
    "stokes_v_over_i",
)
NORMALIZATION_ENERGY_KEY = "electromagnetic_deposited_energy_GeV"


def json_compatible(value: Any) -> Any:
    """Return a strict-JSON representation of a diagnostic value.

    Ensemble statistics such as the sample standard deviation are undefined
    for a single shower.  Keep that distinction explicit as JSON ``null``
    instead of failing during report serialization or emitting non-standard
    NaN tokens.
    """

    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_array(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    return result[np.isfinite(result)]


def scalar_summary(values: Iterable[float]) -> dict[str, Any]:
    array = finite_array(values)
    if array.size == 0:
        return {"count": 0}
    standard_deviation = (
        float(np.std(array, ddof=1)) if array.size > 1 else math.nan
    )
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": standard_deviation,
        "standard_error": (
            standard_deviation / math.sqrt(array.size)
            if array.size > 1
            else math.nan
        ),
        "q05": float(np.quantile(array, 0.05)),
        "q16": float(np.quantile(array, 0.16)),
        "q84": float(np.quantile(array, 0.84)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def quantile_wasserstein(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.size == 0 or candidate.size == 0:
        return math.nan
    quantiles = np.linspace(0.0, 1.0, 1001)
    distance = float(
        np.mean(
            np.abs(
                np.quantile(reference, quantiles)
                - np.quantile(candidate, quantiles)
            )
        )
    )
    scale = max(
        float(np.mean(np.abs(reference))),
        float(np.mean(np.abs(candidate))),
        np.finfo(np.float64).tiny,
    )
    return distance / scale


def compare_distributions(
    reference: Iterable[float],
    candidate: Iterable[float],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    a = finite_array(reference)
    b = finite_array(candidate)
    if a.size == 0 or b.size == 0:
        return {
            "available": False,
            "reference_count": int(a.size),
            "candidate_count": int(b.size),
        }
    a_summary = scalar_summary(a)
    b_summary = scalar_summary(b)
    mean_difference = b_summary["mean"] - a_summary["mean"]
    combined_standard_error = math.sqrt(
        a_summary["standard_error"] ** 2 + b_summary["standard_error"] ** 2
    )
    reference_scale = float(np.mean(np.abs(a)))
    candidate_scale = float(np.mean(np.abs(b)))
    scale = max(
        reference_scale,
        candidate_scale,
        np.finfo(np.float64).tiny,
    )
    rng = np.random.default_rng(seed)
    reference_indices = rng.integers(
        0, a.size, size=(bootstrap_repetitions, a.size)
    )
    candidate_indices = rng.integers(
        0, b.size, size=(bootstrap_repetitions, b.size)
    )
    boot_reference = np.mean(a[reference_indices], axis=1)
    boot_candidate = np.mean(b[candidate_indices], axis=1)
    # Use a fixed absolute reference scale.  A bootstrap mean can cross zero
    # for signed cancellation observables; dividing by that resample-specific
    # mean produces meaningless infinities even though all input samples are
    # finite.
    bootstrap_scale = max(
        reference_scale,
        np.finfo(np.float64).tiny,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        boot_relative_shift = (
            boot_candidate - boot_reference
        ) / bootstrap_scale
    boot_relative_shift = boot_relative_shift[
        np.isfinite(boot_relative_shift)
    ]
    bootstrap_interval = (
        [
            float(np.quantile(boot_relative_shift, 0.025)),
            float(np.quantile(boot_relative_shift, 0.975)),
        ]
        if boot_relative_shift.size
        else None
    )
    signed_relative_shift = (
        mean_difference / reference_scale
        if reference_scale > np.finfo(np.float64).tiny
        else (0.0 if mean_difference == 0.0 else None)
    )
    return {
        "available": True,
        "reference": a_summary,
        "candidate": b_summary,
        "mean_difference": mean_difference,
        "relative_mean_difference": abs(mean_difference) / scale,
        "signed_relative_shift_from_reference": signed_relative_shift,
        "median_difference": b_summary["median"] - a_summary["median"],
        "combined_standard_error": combined_standard_error,
        "z_score": (
            abs(mean_difference) / combined_standard_error
            if combined_standard_error > 0.0
            else (0.0 if mean_difference == 0.0 else None)
        ),
        "KS_distance": ks_distance(a, b),
        "relative_quantile_wasserstein": quantile_wasserstein(a, b),
        "bootstrap_signed_relative_mean_shift_95pct": bootstrap_interval,
        "bootstrap_repetitions": bootstrap_repetitions,
    }


def band_limited_waveform(
    field: np.ndarray, dt_s: float, low_MHz: float, high_MHz: float
) -> np.ndarray:
    spectrum = np.fft.rfft(field, axis=0)
    frequencies = np.fft.rfftfreq(field.shape[0], d=dt_s)
    keep = (frequencies >= low_MHz * 1.0e6) & (
        frequencies <= high_MHz * 1.0e6
    )
    spectrum[~keep, :] = 0.0
    return np.fft.irfft(spectrum, n=field.shape[0], axis=0)


def peak_aligned_unit_energy(power: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float64)
    total = float(np.sum(power))
    if power.size == 0 or total <= 0.0:
        return np.zeros_like(power)
    normalized = power / total
    source_peak = int(np.argmax(normalized))
    target_peak = normalized.size // 2
    shift = target_peak - source_peak
    result = np.zeros_like(normalized)
    if shift >= 0:
        result[shift:] = normalized[: normalized.size - shift]
    else:
        result[:shift] = normalized[-shift:]
    retained = float(np.sum(result))
    return result / retained if retained > 0.0 else result


def unit_sum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    return values / total if total > 0.0 else np.zeros_like(values)


def waveform_feature(
    field: np.ndarray,
    time_ns: np.ndarray,
    energy_GeV: float,
    band: tuple[float, float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dt_s = float(np.median(np.diff(time_ns))) * 1.0e-9
    filtered = band_limited_waveform(field, dt_s, *band)
    power = np.sum(filtered * filtered, axis=1)
    total_power = float(np.sum(power))
    energy_scale = max(energy_GeV, np.finfo(np.float64).tiny)
    center_time = 0.5 * (float(time_ns[0]) + float(time_ns[-1]))
    if total_power <= 0.0:
        return (
            {
                "peak_field_per_GeV": 0.0,
                "peak_offset_ns": math.nan,
                "centroid_offset_ns": math.nan,
                "rms_width_ns": math.nan,
                "spectral_centroid_MHz": math.nan,
                "edge_energy_fraction": 0.0,
                "fluence_J_per_m2_per_GeV2": 0.0,
                "fluence_x_fraction": 0.0,
                "fluence_y_fraction": 0.0,
                "fluence_z_fraction": 0.0,
                "stokes_q_over_i": 0.0,
                "stokes_u_over_i": 0.0,
                "stokes_v_over_i": 0.0,
            },
            np.zeros_like(power),
            np.zeros(0, dtype=np.float64),
        )
    peak_index = int(np.argmax(power))
    centroid = float(np.sum(time_ns * power) / total_power)
    width = math.sqrt(
        max(float(np.sum((time_ns - centroid) ** 2 * power) / total_power), 0.0)
    )
    edge_count = max(1, int(math.ceil(0.10 * power.size)))
    edge_fraction = float(
        (np.sum(power[:edge_count]) + np.sum(power[-edge_count:]))
        / total_power
    )
    spectrum = np.fft.rfft(field, axis=0)
    frequencies_Hz = np.fft.rfftfreq(field.shape[0], d=dt_s)
    keep = (frequencies_Hz >= band[0] * 1.0e6) & (
        frequencies_Hz <= band[1] * 1.0e6
    )
    spectral_power = np.sum(np.abs(spectrum[keep]) ** 2, axis=1)
    component_power = np.sum(np.abs(spectrum[keep]) ** 2, axis=0)
    component_total = max(
        float(np.sum(component_power)), np.finfo(np.float64).tiny
    )
    ex = spectrum[keep, 0]
    ey = spectrum[keep, 1]
    transverse_i = max(
        float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2)),
        np.finfo(np.float64).tiny,
    )
    stokes_q = float(np.sum(np.abs(ex) ** 2 - np.abs(ey) ** 2))
    stokes_u = float(2.0 * np.real(np.sum(ex * np.conj(ey))))
    stokes_v = float(-2.0 * np.imag(np.sum(ex * np.conj(ey))))
    spectral_centroid = (
        float(
            np.sum(frequencies_Hz[keep] * spectral_power)
            / np.sum(spectral_power)
            / 1.0e6
        )
        if float(np.sum(spectral_power)) > 0.0
        else math.nan
    )
    return (
        {
            "peak_field_per_GeV": math.sqrt(float(power[peak_index]))
            / energy_scale,
            "peak_offset_ns": float(time_ns[peak_index]) - center_time,
            "centroid_offset_ns": centroid - center_time,
            "rms_width_ns": width,
            "spectral_centroid_MHz": spectral_centroid,
            "edge_energy_fraction": edge_fraction,
            "fluence_J_per_m2_per_GeV2": (
                EPSILON_0_F_PER_M
                * LIGHT_SPEED_M_PER_S
                * total_power
                * dt_s
                / (energy_scale * energy_scale)
            ),
            "fluence_x_fraction": float(component_power[0] / component_total),
            "fluence_y_fraction": float(component_power[1] / component_total),
            "fluence_z_fraction": float(component_power[2] / component_total),
            "stokes_q_over_i": stokes_q / transverse_i,
            "stokes_u_over_i": stokes_u / transverse_i,
            "stokes_v_over_i": stokes_v / transverse_i,
        },
        peak_aligned_unit_energy(power),
        unit_sum(spectral_power),
    )


def read_radio_records(
    output: Path, algorithm: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    layout = radio_observer_layout(output, algorithm)
    table = pq.read_table(
        output / algorithm / "observers.parquet",
        columns=["shower", "Time", "Ex", "Ey", "Ez"],
    )
    values = {
        name: np.asarray(table[name].to_numpy(zero_copy_only=False))
        for name in table.column_names
    }
    expected_rows = sum(observer["bins"] for observer in layout)
    locations = np.asarray([observer["location_m"] for observer in layout])
    center_xy = locations[0, :2]
    radii = np.linalg.norm(locations[:, :2] - center_xy, axis=1)
    records: list[dict[str, Any]] = []
    for shower in np.unique(values["shower"]):
        indices = np.flatnonzero(values["shower"] == shower)
        if indices.size != expected_rows:
            raise ValueError(
                f"{algorithm} shower {shower} has {indices.size} rows; "
                f"expected {expected_rows}"
            )
        offset = 0
        for observer_index, observer in enumerate(layout):
            count = observer["bins"]
            selected = indices[offset : offset + count]
            offset += count
            records.append(
                {
                    "shower": int(shower),
                    "observer": observer_index,
                    "radius_m": float(radii[observer_index]),
                    "time_ns": np.asarray(values["Time"][selected], dtype=np.float64),
                    "field": np.column_stack(
                        [
                            np.asarray(values["Ex"][selected], dtype=np.float64),
                            np.asarray(values["Ey"][selected], dtype=np.float64),
                            np.asarray(values["Ez"][selected], dtype=np.float64),
                        ]
                    ),
                }
            )
    return records, locations


def aggregate_radial_features(
    records: list[dict[str, Any]],
    energies: dict[int, dict[str, float]],
    band: tuple[float, float],
    normalization_energy_GeV: float | None = None,
) -> tuple[
    dict[tuple[int, float], dict[str, float]],
    dict[float, list[np.ndarray]],
    dict[float, list[np.ndarray]],
]:
    raw_features: dict[tuple[int, float], list[dict[str, float]]] = defaultdict(list)
    templates: dict[float, list[np.ndarray]] = defaultdict(list)
    spectra: dict[float, list[np.ndarray]] = defaultdict(list)
    for record in records:
        shower = record["shower"]
        energy = (
            normalization_energy_GeV
            if normalization_energy_GeV is not None
            else energies[shower][NORMALIZATION_ENERGY_KEY]
        )
        feature, template, spectrum = waveform_feature(
            record["field"], record["time_ns"], energy, band
        )
        radius = round(record["radius_m"], RADIAL_GROUP_DECIMALS)
        raw_features[(shower, radius)].append(feature)
        templates[radius].append(template)
        spectra[radius].append(spectrum)
    aggregated: dict[tuple[int, float], dict[str, float]] = {}
    for key, rows in raw_features.items():
        aggregated[key] = {
            feature: float(np.mean([row[feature] for row in rows]))
            for feature in FEATURES
        }
    return aggregated, templates, spectra


def template_comparison(
    reference: list[np.ndarray], candidate: list[np.ndarray]
) -> dict[str, float]:
    a = np.mean(np.stack(reference), axis=0)
    b = np.mean(np.stack(candidate), axis=0)
    difference = b - a
    scale_l1 = max(float(np.sum(np.abs(a))), float(np.sum(np.abs(b))), 1.e-300)
    scale_l2 = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.e-300)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "reference_waveforms": len(reference),
        "candidate_waveforms": len(candidate),
        "relative_L1": float(np.sum(np.abs(difference)) / scale_l1),
        "relative_L2": float(np.linalg.norm(difference) / scale_l2),
        "cosine_similarity": (
            float(np.dot(a, b) / denominator) if denominator > 0.0 else 1.0
        ),
    }


def vector_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("vector shapes differ")
    difference = b - a
    scale_l1 = max(float(np.sum(np.abs(a))), float(np.sum(np.abs(b))), 1.e-300)
    scale_l2 = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.e-300)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "bins": int(a.size),
        "relative_L1": float(np.sum(np.abs(difference)) / scale_l1),
        "relative_L2": float(np.linalg.norm(difference) / scale_l2),
        "cosine_similarity": (
            float(np.dot(a, b) / denominator) if denominator > 0.0 else 1.0
        ),
    }


def radial_integral_contributions(
    radii_m: np.ndarray, fluence: np.ndarray
) -> np.ndarray:
    radii = np.asarray(radii_m, dtype=np.float64)
    values = np.asarray(fluence, dtype=np.float64)
    if radii.ndim != 1 or values.ndim != 1 or radii.size != values.size:
        raise ValueError("radii and fluence must be equal-length vectors")
    if radii.size < 2 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("at least two strictly increasing radii are required")
    integrand = values * radii
    return math.pi * (integrand[:-1] + integrand[1:]) * np.diff(radii)


def pearson(values_x: Iterable[float], values_y: Iterable[float]) -> float:
    x = finite_array(values_x)
    y = finite_array(values_y)
    if x.size != y.size or x.size < 2:
        return math.nan
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def add_derived_track_diagnostics(
    showers: dict[str, Any],
) -> dict[str, Any]:
    """Add intensive track moments after scalar/GPU records are merged."""

    def ratio(
        record: dict[str, Any], numerator: str, denominator: str
    ) -> float:
        value = float(record.get(numerator, math.nan))
        scale = float(record.get(denominator, math.nan))
        if (
            not math.isfinite(value)
            or not math.isfinite(scale)
            or scale == 0.0
        ):
            return math.nan
        return value / scale

    for record in showers.values():
        if not isinstance(record, dict):
            continue
        record["weighted_mean_segment_length_m"] = ratio(
            record,
            "weighted_track_length_m",
            "weighted_segment_count",
        )
        record["weighted_mean_direction_change_rad"] = ratio(
            record,
            "weighted_direction_change_rad",
            "weighted_segment_count",
        )
        squared_mean = ratio(
            record,
            "weighted_direction_change_squared_rad2",
            "weighted_segment_count",
        )
        record["weighted_rms_direction_change_rad"] = (
            math.sqrt(max(squared_mean, 0.0))
            if math.isfinite(squared_mean)
            else math.nan
        )
        record[
            "weighted_direction_change_per_track_length_rad_per_m"
        ] = ratio(
            record,
            "weighted_direction_change_rad",
            "weighted_track_length_m",
        )
        record["weighted_beta_deficit_fraction"] = ratio(
            record,
            "weighted_beta_deficit_track_length_m",
            "weighted_track_length_m",
        )
        record["weighted_mean_time_residual_s"] = ratio(
            record,
            "weighted_time_residual_s",
            "weighted_segment_count",
        )
        record[
            "energy_track_weighted_mean_kinetic_energy_GeV"
        ] = ratio(
            record,
            "energy_weighted_track_length_GeV_m",
            "weighted_track_length_m",
        )
        signed_direction = record.get(
            "signed_charge_weighted_direction_change", {}
        )
        if isinstance(signed_direction, dict):
            for axis in ("x", "y", "z"):
                record[
                    f"signed_charge_weighted_direction_change_{axis}"
                ] = float(signed_direction.get(axis, 0.0))
    return showers


def read_track_diagnostics(output: Path, algorithm: str) -> dict[str, Any]:
    path = output / algorithm / "summary.yaml"
    value: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as source:
            loaded = yaml.safe_load(source)
        if isinstance(loaded, dict):
            value = loaded

    # In CUDA-radio mode, RadioProcess sees only scalar fallback tracks.
    # Resident e+/e- tracks are projected without leaving the device and their
    # scalar-compatible diagnostics live in gpu_em/summary.yaml. Merge both
    # sources so candidate/reference comparisons describe the complete radio
    # input rather than the small CPU fallback tail.
    gpu_path = output / "gpu_em" / "summary.yaml"
    if not gpu_path.is_file():
        return add_derived_track_diagnostics(value)
    with gpu_path.open("r", encoding="utf-8") as source:
        gpu_summary = yaml.safe_load(source)
    if not isinstance(gpu_summary, dict):
        return add_derived_track_diagnostics(value)

    additive_fields = (
        "segment_count",
        "weighted_segment_count",
        "track_length_m",
        "weighted_track_length_m",
        "electron_weighted_track_length_m",
        "positron_weighted_track_length_m",
        "signed_charge_weighted_track_length_m",
        "energy_weighted_track_length_GeV_m",
        "weighted_direction_change_rad",
        "weighted_direction_change_squared_rad2",
        "weighted_beta_deficit_track_length_m",
        "weighted_time_residual_s",
    )
    for shower, record in gpu_summary.items():
        if not isinstance(record, dict):
            continue
        radio = record.get("statistics", {}).get("radio", {})
        if (
            not isinstance(radio, dict)
            or radio.get("track_diagnostics_enabled") is not True
        ):
            continue
        combined = value.setdefault(str(shower), {})
        for field in additive_fields:
            combined[field] = float(combined.get(field, 0.0)) + float(
                radio.get(field, 0.0)
            )
        combined["maximum_segment_length_m"] = max(
            float(combined.get("maximum_segment_length_m", 0.0)),
            float(radio.get("maximum_segment_length_m", 0.0)),
        )
        combined["maximum_direction_change_rad"] = max(
            float(combined.get("maximum_direction_change_rad", 0.0)),
            float(radio.get("maximum_direction_change_rad", 0.0)),
        )
        scalar_direction = combined.get(
            "signed_charge_weighted_direction_change", {}
        )
        gpu_direction = radio.get(
            "signed_charge_weighted_direction_change", {}
        )
        if not isinstance(scalar_direction, dict):
            scalar_direction = {}
        if not isinstance(gpu_direction, dict):
            gpu_direction = {}
        combined["signed_charge_weighted_direction_change"] = {
            axis: float(scalar_direction.get(axis, 0.0))
            + float(gpu_direction.get(axis, 0.0))
            for axis in ("x", "y", "z")
        }

        gpu_bins = radio.get(
            "weighted_track_length_by_kinetic_energy", {}
        )
        if not isinstance(gpu_bins, dict):
            continue
        gpu_edges = gpu_bins.get("upper_edge_GeV", [])
        gpu_lengths = gpu_bins.get("weighted_track_length_m", [])
        scalar_bins = combined.get(
            "weighted_track_length_by_kinetic_energy"
        )
        if not isinstance(scalar_bins, dict):
            combined["weighted_track_length_by_kinetic_energy"] = {
                "units": dict(gpu_bins.get("units", {})),
                "upper_edge_GeV": list(gpu_edges),
                "weighted_track_length_m": [
                    float(item) for item in gpu_lengths
                ],
            }
            continue
        scalar_edges = scalar_bins.get("upper_edge_GeV", [])
        scalar_lengths = scalar_bins.get(
            "weighted_track_length_m", []
        )
        if scalar_edges != gpu_edges:
            raise ValueError(
                f"{algorithm} scalar/GPU radio track energy-bin edges differ"
            )
        if len(scalar_lengths) != len(gpu_lengths):
            raise ValueError(
                f"{algorithm} scalar/GPU radio track bin counts differ"
            )
        scalar_bins["weighted_track_length_m"] = [
            float(left) + float(right)
            for left, right in zip(scalar_lengths, gpu_lengths)
        ]
    return add_derived_track_diagnostics(value)


def compare_track_diagnostics(
    reference: Path,
    candidate: Path,
    algorithm: str,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    ref = read_track_diagnostics(reference, algorithm)
    cand = read_track_diagnostics(candidate, algorithm)
    if not ref or not cand:
        return {
            "available": False,
            "reference_present": bool(ref),
            "candidate_present": bool(cand),
        }
    scalar_fields = (
        "segment_count",
        "weighted_segment_count",
        "track_length_m",
        "weighted_track_length_m",
        "electron_weighted_track_length_m",
        "positron_weighted_track_length_m",
        "signed_charge_weighted_track_length_m",
        "energy_weighted_track_length_GeV_m",
        "maximum_segment_length_m",
        "weighted_direction_change_rad",
        "weighted_direction_change_squared_rad2",
        "weighted_beta_deficit_track_length_m",
        "weighted_time_residual_s",
        "maximum_direction_change_rad",
        "weighted_mean_segment_length_m",
        "weighted_mean_direction_change_rad",
        "weighted_rms_direction_change_rad",
        "weighted_direction_change_per_track_length_rad_per_m",
        "weighted_beta_deficit_fraction",
        "weighted_mean_time_residual_s",
        "energy_track_weighted_mean_kinetic_energy_GeV",
        "signed_charge_weighted_direction_change_x",
        "signed_charge_weighted_direction_change_y",
        "signed_charge_weighted_direction_change_z",
    )
    common = sorted(set(ref) & set(cand))
    ref_energy = energy_observables(reference)
    cand_energy = energy_observables(candidate)
    result: dict[str, Any] = {
        "available": True,
        "showers": len(common),
        "fields": {},
        "fields_per_electromagnetic_deposited_GeV": {},
        "weighted_track_length_by_kinetic_energy": {},
    }
    available_scalar_fields = [
        field
        for field in scalar_fields
        if all(
            field in ref[key] and field in cand[key]
            for key in common
        )
    ]
    for index, field in enumerate(available_scalar_fields):
        result["fields"][field] = compare_distributions(
            [float(ref[key][field]) for key in common],
            [float(cand[key][field]) for key in common],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + index,
        )
        result["fields_per_electromagnetic_deposited_GeV"][
            field
        ] = compare_distributions(
            [
                float(ref[key][field])
                / max(ref_energy[shower_key_index(key)][NORMALIZATION_ENERGY_KEY], 1.e-30)
                for key in common
            ],
            [
                float(cand[key][field])
                / max(cand_energy[shower_key_index(key)][NORMALIZATION_ENERGY_KEY], 1.e-30)
                for key in common
            ],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + 100 + index,
        )

    reference_edges = ref[common[0]][
        "weighted_track_length_by_kinetic_energy"
    ]["upper_edge_GeV"]
    candidate_edges = cand[common[0]][
        "weighted_track_length_by_kinetic_energy"
    ]["upper_edge_GeV"]
    if reference_edges != candidate_edges:
        raise ValueError(f"{algorithm} radio track energy-bin edges differ")
    for index, edge in enumerate(reference_edges):
        label = f"le_{edge}_GeV" if str(edge) != "inf" else "overflow"
        ref_values = [
            float(
                ref[key]["weighted_track_length_by_kinetic_energy"][
                    "weighted_track_length_m"
                ][index]
            )
            for key in common
        ]
        cand_values = [
            float(
                cand[key]["weighted_track_length_by_kinetic_energy"][
                    "weighted_track_length_m"
                ][index]
            )
            for key in common
        ]
        result["weighted_track_length_by_kinetic_energy"][label] = {
            "upper_edge_GeV": edge,
            "raw": compare_distributions(
                ref_values,
                cand_values,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed + 1000 + index,
            ),
            "per_electromagnetic_deposited_GeV": compare_distributions(
                [
                    value
                    / max(
                        ref_energy[shower_key_index(key)][
                            NORMALIZATION_ENERGY_KEY
                        ],
                        1.e-30,
                    )
                    for key, value in zip(common, ref_values)
                ],
                [
                    value
                    / max(
                        cand_energy[shower_key_index(key)][
                            NORMALIZATION_ENERGY_KEY
                        ],
                        1.e-30,
                    )
                    for key, value in zip(common, cand_values)
                ],
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed + 2000 + index,
            ),
        }
    return result


def radiation_proxy_distributions(
    output: Path,
    algorithm: str,
    band: tuple[float, float],
    locations: np.ndarray,
    *,
    normalize_by_em_deposit: bool = True,
) -> dict[int, float]:
    maps = radio_fluence_maps(output, algorithm, [band])[band]
    energies = energy_observables(output)
    return {
        shower: float(
            radiation_energy_proxy(
                (
                    fluence
                    / max(
                        energies[shower][NORMALIZATION_ENERGY_KEY], 1.e-30
                    )
                    ** 2
                    if normalize_by_em_deposit
                    else fluence
                ),
                locations,
            )[3]
        )
        for shower, fluence in maps.items()
    }


def analyze_algorithm(
    reference: Path,
    candidate: Path,
    algorithm: str,
    bootstrap_repetitions: int,
    seed: int,
    normalization_energy_GeV: float | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, dict[int, float]]],
]:
    ref_energy = energy_observables(reference)
    cand_energy = energy_observables(candidate)
    ref_records, ref_locations = read_radio_records(reference, algorithm)
    cand_records, cand_locations = read_radio_records(candidate, algorithm)
    if not np.allclose(ref_locations, cand_locations, rtol=0.0, atol=1.e-9):
        raise ValueError(f"{algorithm} observer layouts differ")
    ref_fluence_maps = radio_fluence_maps(
        reference, algorithm, list(BANDS)
    )
    cand_fluence_maps = radio_fluence_maps(
        candidate, algorithm, list(BANDS)
    )
    report: dict[str, Any] = {
        "observers": len(ref_locations),
        "waveform_amplitude_normalization": (
            {
                "kind": "fixed_primary_energy",
                "energy_GeV": normalization_energy_GeV,
            }
            if normalization_energy_GeV is not None
            else {"kind": NORMALIZATION_ENERGY_KEY}
        ),
        "radiation_energy_normalization": NORMALIZATION_ENERGY_KEY,
        "track_diagnostics": compare_track_diagnostics(
            reference,
            candidate,
            algorithm,
            bootstrap_repetitions,
            seed + 5000,
        ),
        "bands": {},
    }
    csv_rows: list[dict[str, Any]] = []
    raw_proxies: dict[str, dict[str, dict[int, float]]] = {
        "reference": {},
        "candidate": {},
    }
    for band_index, band in enumerate(BANDS):
        ref_aggregated, ref_templates, ref_spectra = aggregate_radial_features(
            ref_records,
            ref_energy,
            band,
            normalization_energy_GeV,
        )
        cand_aggregated, cand_templates, cand_spectra = aggregate_radial_features(
            cand_records,
            cand_energy,
            band,
            normalization_energy_GeV,
        )
        radii = sorted(
            set(radius for _, radius in ref_aggregated)
            & set(radius for _, radius in cand_aggregated)
        )
        radial_report: dict[str, Any] = {}
        for radius_index, radius in enumerate(radii):
            ref_keys = sorted(key for key in ref_aggregated if key[1] == radius)
            cand_keys = sorted(key for key in cand_aggregated if key[1] == radius)
            feature_report: dict[str, Any] = {}
            for feature_index, feature in enumerate(FEATURES):
                comparison = compare_distributions(
                    [ref_aggregated[key][feature] for key in ref_keys],
                    [cand_aggregated[key][feature] for key in cand_keys],
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=(
                        seed
                        + band_index * 10000
                        + radius_index * 100
                        + feature_index
                    ),
                )
                feature_report[feature] = comparison
                csv_rows.append(
                    {
                        "algorithm": algorithm,
                        "low_MHz": band[0],
                        "high_MHz": band[1],
                        "radius_m": radius,
                        "feature": feature,
                        "reference_mean": comparison["reference"]["mean"],
                        "candidate_mean": comparison["candidate"]["mean"],
                        "signed_relative_shift": comparison[
                            "signed_relative_shift_from_reference"
                        ],
                        "z_score": comparison["z_score"],
                        "KS_distance": comparison["KS_distance"],
                        "bootstrap_low": comparison[
                            "bootstrap_signed_relative_mean_shift_95pct"
                        ][0],
                        "bootstrap_high": comparison[
                            "bootstrap_signed_relative_mean_shift_95pct"
                        ][1],
                    }
                )
            radial_report[f"{radius:g}_m"] = {
                "features": feature_report,
                "normalized_aligned_power_template": template_comparison(
                    ref_templates[radius], cand_templates[radius]
                ),
                "normalized_frequency_power_template": template_comparison(
                    ref_spectra[radius], cand_spectra[radius]
                ),
            }
        ref_radial_fluence = np.asarray(
            [
                np.mean(
                    [
                        ref_aggregated[key]["fluence_J_per_m2_per_GeV2"]
                        for key in ref_aggregated
                        if key[1] == radius
                    ]
                )
                for radius in radii
            ]
        )
        cand_radial_fluence = np.asarray(
            [
                np.mean(
                    [
                        cand_aggregated[key]["fluence_J_per_m2_per_GeV2"]
                        for key in cand_aggregated
                        if key[1] == radius
                    ]
                )
                for radius in radii
            ]
        )
        ref_raw_proxy = {
            shower: float(
                radiation_energy_proxy(fluence, ref_locations)[3]
            )
            for shower, fluence in ref_fluence_maps[band].items()
        }
        cand_raw_proxy = {
            shower: float(
                radiation_energy_proxy(fluence, cand_locations)[3]
            )
            for shower, fluence in cand_fluence_maps[band].items()
        }
        ref_proxy = {
            shower: value
            / max(
                ref_energy[shower][NORMALIZATION_ENERGY_KEY], 1.e-30
            )
            ** 2
            for shower, value in ref_raw_proxy.items()
        }
        cand_proxy = {
            shower: value
            / max(
                cand_energy[shower][NORMALIZATION_ENERGY_KEY], 1.e-30
            )
            ** 2
            for shower, value in cand_raw_proxy.items()
        }
        band_label = f"{band[0]:g}-{band[1]:g}_MHz"
        raw_proxies["reference"][band_label] = ref_raw_proxy
        raw_proxies["candidate"][band_label] = cand_raw_proxy
        common_ref = sorted(ref_proxy)
        common_cand = sorted(cand_proxy)
        proxy_comparison = compare_distributions(
            [ref_proxy[key] for key in common_ref],
            [cand_proxy[key] for key in common_cand],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + band_index + 30000,
        )
        raw_proxy_comparison = compare_distributions(
            [ref_raw_proxy[key] for key in sorted(ref_raw_proxy)],
            [cand_raw_proxy[key] for key in sorted(cand_raw_proxy)],
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + band_index + 40000,
        )
        report["bands"][band_label] = {
            "radiation_energy_proxy": proxy_comparison,
            "raw_radiation_energy_proxy": raw_proxy_comparison,
            "correlations": {
                "reference_with_electromagnetic_deposited_energy": pearson(
                    [ref_proxy[key] for key in common_ref],
                    [
                        ref_energy[key][NORMALIZATION_ENERGY_KEY]
                        for key in common_ref
                    ],
                ),
                "candidate_with_electromagnetic_deposited_energy": pearson(
                    [cand_proxy[key] for key in common_cand],
                    [
                        cand_energy[key][NORMALIZATION_ENERGY_KEY]
                        for key in common_cand
                    ],
                ),
                "reference_with_Xmax": pearson(
                    [ref_proxy[key] for key in common_ref],
                    [ref_energy[key]["Xmax_g_per_cm2"] for key in common_ref],
                ),
                "candidate_with_Xmax": pearson(
                    [cand_proxy[key] for key in common_cand],
                    [cand_energy[key]["Xmax_g_per_cm2"] for key in common_cand],
                ),
            },
            "normalized_radial_radiation_energy_contribution_shape": (
                vector_comparison(
                    unit_sum(
                        radial_integral_contributions(
                            np.asarray(radii), ref_radial_fluence
                        )
                    ),
                    unit_sum(
                        radial_integral_contributions(
                            np.asarray(radii), cand_radial_fluence
                        )
                    ),
                )
            ),
            "by_radius": radial_report,
        }
    feature_rows: list[dict[str, Any]] = []
    time_templates: list[dict[str, Any]] = []
    spectrum_templates: list[dict[str, Any]] = []
    footprints: list[dict[str, Any]] = []
    for band_label, band_report in report["bands"].items():
        footprint = band_report[
            "normalized_radial_radiation_energy_contribution_shape"
        ]
        footprints.append({"band": band_label, **footprint})
        for radius_label, radius_report in band_report["by_radius"].items():
            for feature, comparison in radius_report["features"].items():
                feature_rows.append(
                    {
                        "band": band_label,
                        "radius": radius_label,
                        "feature": feature,
                        "z_score": comparison["z_score"],
                        "signed_relative_shift": comparison[
                            "signed_relative_shift_from_reference"
                        ],
                    }
                )
            time_templates.append(
                {
                    "band": band_label,
                    "radius": radius_label,
                    **radius_report["normalized_aligned_power_template"],
                }
            )
            spectrum_templates.append(
                {
                    "band": band_label,
                    "radius": radius_label,
                    **radius_report["normalized_frequency_power_template"],
                }
            )
    report["multiple_observable_summary"] = {
        "feature_comparisons": len(feature_rows),
        "feature_z_score_at_least_2": sum(
            row["z_score"] >= 2.0 for row in feature_rows
        ),
        "feature_z_score_at_least_3": sum(
            row["z_score"] >= 3.0 for row in feature_rows
        ),
        "largest_feature_z_scores": sorted(
            feature_rows, key=lambda row: row["z_score"], reverse=True
        )[:10],
        "minimum_time_template_cosine_similarity": min(
            row["cosine_similarity"] for row in time_templates
        ),
        "minimum_spectrum_template_cosine_similarity": min(
            row["cosine_similarity"] for row in spectrum_templates
        ),
        "minimum_radial_contribution_cosine_similarity": min(
            row["cosine_similarity"] for row in footprints
        ),
        "note": (
            "Feature z-scores are correlated and are summarized without "
            "treating them as independent hypothesis tests."
        ),
    }
    return report, csv_rows, raw_proxies


def paired_formalism_comparison(
    coreas: dict[int, float],
    zhs: dict[int, float],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    common = sorted(set(coreas) & set(zhs))
    if not common:
        return {"available": False, "showers": 0}
    c = np.asarray([coreas[key] for key in common], dtype=np.float64)
    z = np.asarray([zhs[key] for key in common], dtype=np.float64)
    denominator = max(abs(float(np.mean(z))), np.finfo(np.float64).tiny)
    aggregate_shift = (float(np.mean(z)) - float(np.mean(c))) / denominator
    per_shower = (z - c) / np.maximum(
        np.abs(z), np.finfo(np.float64).tiny
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(common), size=(bootstrap_repetitions, len(common))
    )
    c_means = np.mean(c[indices], axis=1)
    z_means = np.mean(z[indices], axis=1)
    bootstrap_shift = (z_means - c_means) / np.maximum(
        np.abs(z_means), np.finfo(np.float64).tiny
    )
    return {
        "available": True,
        "showers": len(common),
        "definition": "(ZHS - CoREAS) / ZHS",
        "aggregate_radiation_energy_shift": aggregate_shift,
        "mean_per_shower_shift": float(np.mean(per_shower)),
        "median_per_shower_shift": float(np.median(per_shower)),
        "standard_error_per_shower_shift": (
            float(np.std(per_shower, ddof=1) / math.sqrt(per_shower.size))
            if per_shower.size > 1
            else math.nan
        ),
        "bootstrap_aggregate_shift_95pct": [
            float(np.quantile(bootstrap_shift, 0.025)),
            float(np.quantile(bootstrap_shift, 0.975)),
        ],
        "pearson_radiation_energy": pearson(c, z),
        "bootstrap_repetitions": bootstrap_repetitions,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--normalization-energy-gev",
        type=float,
        help=(
            "Use a fixed energy to normalize waveform amplitudes/fluences; "
            "recommended for fixed-energy electron, positron or photon "
            "primaries. The integrated report still includes both raw and "
            "per-shower EM-deposit-normalized radiation energy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if args.normalization_energy_gev is not None:
        if (
            not math.isfinite(args.normalization_energy_gev)
            or args.normalization_energy_gev <= 0.0
        ):
            raise ValueError("normalization energy must be finite and positive")
    report: dict[str, Any] = {
        "comparison_semantics": (
            "independent CPU-PROPOSAL and CUDA-EM shower ensembles; "
            "waveforms are compared as distributions and aligned templates"
        ),
        "reference": str(args.reference_output.resolve()),
        "candidate": str(args.candidate_output.resolve()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "waveform_normalization_energy_GeV": args.normalization_energy_gev,
        "algorithms": {},
    }
    rows: list[dict[str, Any]] = []
    raw_proxies_by_algorithm: dict[
        str, dict[str, dict[str, dict[int, float]]]
    ] = {}
    for index, algorithm in enumerate(ALGORITHMS):
        algorithm_report, algorithm_rows, raw_proxies = analyze_algorithm(
            args.reference_output.resolve(),
            args.candidate_output.resolve(),
            algorithm,
            args.bootstrap_repetitions,
            args.seed + index * 100000,
            args.normalization_energy_gev,
        )
        report["algorithms"][algorithm] = algorithm_report
        raw_proxies_by_algorithm[algorithm] = raw_proxies
        rows.extend(algorithm_rows)
    report["within_backend_coreas_zhs"] = {}
    for backend_index, backend in enumerate(("reference", "candidate")):
        report["within_backend_coreas_zhs"][backend] = {
            band_label: paired_formalism_comparison(
                raw_proxies_by_algorithm["CoREAS"][backend][band_label],
                raw_proxies_by_algorithm["ZHS"][backend][band_label],
                bootstrap_repetitions=args.bootstrap_repetitions,
                seed=args.seed + 500000 + backend_index * 1000 + band_index,
            )
            for band_index, band_label in enumerate(
                raw_proxies_by_algorithm["CoREAS"][backend]
            )
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as destination:
        json.dump(
            json_compatible(report), destination, indent=2, allow_nan=False
        )
        destination.write("\n")
    if args.csv is not None:
        write_csv(args.csv, rows)
    print(json.dumps({"status": "diagnostic_complete", "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
