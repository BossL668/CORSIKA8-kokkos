#!/usr/bin/env python3
"""Compare legacy scalar-PROPOSAL and CUDA-EM CORSIKA 8 showers.

There are intentionally two different notions of agreement:

* exact/numerical agreement for the legacy versus refactored scalar control;
* ensemble statistical agreement for independently sampled CUDA showers.

Raw radio waveforms are not compared bin by bin unless the particle tracks are
identical.  This tool follows the CORSIKA 8 radio-validation paper and compares
band-limited energy fluence maps and integrated radiation-energy proxies.
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


PROCESS_NAMES = {
    0: "none",
    1_000_000_001: "particle",
    1_000_000_002: "bremsstrahlung",
    1_000_000_003: "ionization",
    1_000_000_004: "electron_pair",
    1_000_000_005: "photonuclear",
    1_000_000_006: "muon_pair",
    1_000_000_007: "hadrons",
    1_000_000_008: "continuous_energy_loss",
    1_000_000_009: "weak_interaction",
    1_000_000_010: "compton",
    1_000_000_011: "decay",
    1_000_000_012: "annihilation",
    1_000_000_013: "photon_pair",
    1_000_000_014: "photoproduction",
    1_000_000_015: "photon_muon_pair",
    1_000_000_016: "photoelectric",
}

EPSILON_0_F_PER_M = 8.8541878128e-12
LIGHT_SPEED_M_PER_S = 299_792_458.0
RADIAL_GROUP_DECIMALS = 3


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    return value if isinstance(value, dict) else {}


def shower_key_index(key: str) -> int:
    if not key.startswith("shower_"):
        raise ValueError(f"invalid shower key: {key}")
    return int(key.removeprefix("shower_"))


def energy_observables(output: Path) -> dict[int, dict[str, float]]:
    summary = read_yaml(output / "energyloss" / "summary.yaml")
    result: dict[int, dict[str, float]] = {}
    for key, values in summary.items():
        if not isinstance(values, dict):
            continue
        deposited_energy = float(values["sum_dEdX"])
        result[shower_key_index(str(key))] = {
            "deposited_energy_GeV": deposited_energy,
            "electromagnetic_deposited_energy_GeV": float(
                values.get("sum_dEdX_em", deposited_energy)
            ),
            "Xmax_g_per_cm2": float(values["Xmax"]),
            "dEdXmax_GeV_per_g_cm2": float(values["dEdXmax"]),
        }
    return result


def mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, math.nan
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def relative_difference(reference: float, candidate: float) -> float:
    scale = max(abs(reference), abs(candidate), np.finfo(np.float64).tiny)
    return abs(candidate - reference) / scale


def radio_acceptance_decision(
    reference_mean: float,
    candidate_mean: float,
    reference_standard_error: float,
    candidate_standard_error: float,
    relative_tolerance: float,
    z_limit: float,
) -> dict[str, Any]:
    """Apply independent physics-accuracy and statistical-significance gates.

    A small z-score alone does not demonstrate the requested accuracy: it can
    merely mean that the ensemble is too small.  Conversely, a small relative
    difference with an extremely precise ensemble should not hide a
    statistically significant discrepancy.  Production acceptance therefore
    requires both conditions.
    """
    mean_difference = abs(candidate_mean - reference_mean)
    combined_standard_error = math.sqrt(
        reference_standard_error * reference_standard_error
        + candidate_standard_error * candidate_standard_error
    )
    absolute_tolerance = relative_tolerance * max(
        abs(reference_mean), np.finfo(np.float64).tiny
    )
    relative_accuracy_pass = mean_difference <= absolute_tolerance
    statistically_consistent = (
        mean_difference <= z_limit * combined_standard_error
        if math.isfinite(combined_standard_error)
        else False
    )
    z_score = (
        mean_difference / combined_standard_error
        if math.isfinite(combined_standard_error)
        and combined_standard_error > 0.0
        else math.inf
    )
    return {
        "relative_accuracy_pass": relative_accuracy_pass,
        "z_score": z_score,
        "statistically_consistent": statistically_consistent,
        "accepted": relative_accuracy_pass and statistically_consistent,
    }


def compare_scalar_observables(
    reference: Path, candidate: Path
) -> dict[str, Any]:
    ref = energy_observables(reference)
    cand = energy_observables(candidate)
    common = sorted(set(ref) & set(cand))
    if not common:
        raise ValueError("the outputs have no common shower IDs")
    result: dict[str, Any] = {"showers": len(common), "observables": {}}
    for name in (
        "deposited_energy_GeV",
        "electromagnetic_deposited_energy_GeV",
        "Xmax_g_per_cm2",
        "dEdXmax_GeV_per_g_cm2",
    ):
        a = np.asarray([ref[index][name] for index in common])
        b = np.asarray([cand[index][name] for index in common])
        ref_mean, ref_se = mean_and_standard_error(a)
        cand_mean, cand_se = mean_and_standard_error(b)
        result["observables"][name] = {
            "reference_mean": ref_mean,
            "candidate_mean": cand_mean,
            "reference_standard_error": ref_se,
            "candidate_standard_error": cand_se,
            "relative_mean_difference": relative_difference(ref_mean, cand_mean),
            "paired_rms_difference": float(np.sqrt(np.mean((b - a) ** 2))),
            "maximum_paired_absolute_difference": float(np.max(np.abs(b - a))),
        }
    return result


def table_to_numpy(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        name: np.asarray(table[name].to_numpy(zero_copy_only=False))
        for name in table.column_names
    }


def compare_longitudinal_profiles(
    reference: Path, candidate: Path
) -> dict[str, Any]:
    def ensemble_mean_curve(
        table: dict[str, np.ndarray], column: str
    ) -> tuple[np.ndarray, np.ndarray]:
        curves = []
        axis: np.ndarray | None = None
        for shower in np.unique(table["shower"]):
            selected = table["shower"] == shower
            shower_axis = np.asarray(table["X"][selected], dtype=np.float64)
            shower_curve = np.asarray(table[column][selected], dtype=np.float64)
            if axis is None:
                axis = shower_axis
            elif not np.array_equal(axis, shower_axis):
                raise ValueError("profile depth axes differ between showers")
            curves.append(shower_curve)
        if axis is None or not curves:
            raise ValueError("profile contains no showers")
        return axis, np.mean(np.stack(curves), axis=0)

    result: dict[str, Any] = {}
    for relative in (
        Path("energyloss/dEdX.parquet"),
        Path("profile/profile.parquet"),
        Path("production_profile/profile.parquet"),
    ):
        ref_path = reference / relative
        cand_path = candidate / relative
        if not ref_path.is_file() or not cand_path.is_file():
            result[str(relative)] = {
                "available": False,
                "reference_present": ref_path.is_file(),
                "candidate_present": cand_path.is_file(),
            }
            continue
        ref = table_to_numpy(ref_path)
        cand = table_to_numpy(cand_path)
        if set(ref) != set(cand):
            raise ValueError(f"profile schemas differ for {relative}")
        columns: dict[str, Any] = {}
        for name in ref:
            if name in {"shower", "X"}:
                continue
            ref_axis, a = ensemble_mean_curve(ref, name)
            cand_axis, b = ensemble_mean_curve(cand, name)
            if not np.array_equal(ref_axis, cand_axis):
                raise ValueError(f"profile depth axes differ for {relative}")
            l1_scale = max(
                float(np.sum(np.abs(a))),
                float(np.sum(np.abs(b))),
                np.finfo(np.float64).tiny,
            )
            l2_scale = max(
                float(np.linalg.norm(a)),
                float(np.linalg.norm(b)),
                np.finfo(np.float64).tiny,
            )
            columns[name] = {
                "relative_L1": float(np.sum(np.abs(b - a)) / l1_scale),
                "relative_L2": float(np.linalg.norm(b - a) / l2_scale),
                "reference_peak": float(np.max(a)),
                "candidate_peak": float(np.max(b)),
            }
        result[str(relative)] = {
            "available": True,
            "reference_showers": len(np.unique(ref["shower"])),
            "candidate_showers": len(np.unique(cand["shower"])),
            "bins": len(np.unique(ref["X"])),
            "columns": columns,
        }
    return result


def load_process_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            process_id = int(row["process_id"])
            if process_id == 0:
                continue
            records.append(
                {
                    "backend": row["backend"],
                    "shower": int(row["shower"]),
                    "ordinal": int(row["ordinal"]),
                    "record_kind": row["record_kind"],
                    "history_id": int(row["history_id"]),
                    "step_id": int(row["step_id"]),
                    "pdg": int(row["pdg"]),
                    "process_id": process_id,
                    "component_hash": int(row["component_hash"]),
                    "process": PROCESS_NAMES.get(
                        process_id, f"unknown_{process_id}"
                    ),
                    "start_energy_GeV": float(row["start_energy_GeV"]),
                    "loss_fraction": float(row["loss_fraction"]),
                    "altitude_m": float(row["z_m"]) - 6_371_000.0,
                }
            )
    return records


def ks_distance(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.sort(np.asarray(list(a), dtype=np.float64))
    y = np.sort(np.asarray(list(b), dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return math.nan
    support = np.sort(np.concatenate((x, y)))
    cdf_x = np.searchsorted(x, support, side="right") / x.size
    cdf_y = np.searchsorted(y, support, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def compare_process_traces(reference_trace: Path, candidate_trace: Path) -> dict[str, Any]:
    ref = load_process_trace(reference_trace)
    cand = load_process_trace(candidate_trace)
    if not ref or not cand:
        return {
            "available": False,
            "reference_records": len(ref),
            "candidate_records": len(cand),
        }
    ref_by_process: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cand_by_process: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ref:
        ref_by_process[record["process"]].append(record)
    for record in cand:
        cand_by_process[record["process"]].append(record)

    scalar_lockstep = (
        len(ref) == len(cand)
        and all(row["record_kind"] == "proposal_selection" for row in ref)
        and all(row["record_kind"] == "proposal_selection" for row in cand)
    )
    lockstep_report: dict[str, Any] = {
        "applicable": scalar_lockstep,
        "selection_identity": False,
    }
    if scalar_lockstep:
        identity_fields = (
            "shower",
            "ordinal",
            "pdg",
            "process_id",
            "component_hash",
        )
        lockstep_report["selection_identity"] = all(
            all(a[field] == b[field] for field in identity_fields)
            for a, b in zip(ref, cand)
        )
        for field in ("start_energy_GeV", "loss_fraction", "altitude_m"):
            a = np.asarray([row[field] for row in ref], dtype=np.float64)
            b = np.asarray([row[field] for row in cand], dtype=np.float64)
            lockstep_report[f"maximum_absolute_{field}_difference"] = float(
                np.max(np.abs(b - a))
            )

    processes: dict[str, Any] = {}
    for process in sorted(set(ref_by_process) | set(cand_by_process)):
        a = ref_by_process[process]
        b = cand_by_process[process]
        a_energy = [row["start_energy_GeV"] for row in a]
        b_energy = [row["start_energy_GeV"] for row in b]
        a_v = [row["loss_fraction"] for row in a if math.isfinite(row["loss_fraction"])]
        b_v = [row["loss_fraction"] for row in b if math.isfinite(row["loss_fraction"])]
        a_altitude = [row["altitude_m"] for row in a]
        b_altitude = [row["altitude_m"] for row in b]
        processes[process] = {
            "reference_count": len(a),
            "candidate_count": len(b),
            "relative_count_difference": relative_difference(len(a), len(b)),
            "start_energy_GeV": {
                "reference_median": float(np.median(a_energy)) if a_energy else math.nan,
                "candidate_median": float(np.median(b_energy)) if b_energy else math.nan,
                "KS_distance": ks_distance(a_energy, b_energy),
            },
            "loss_fraction": {
                "reference_median": float(np.median(a_v)) if a_v else math.nan,
                "candidate_median": float(np.median(b_v)) if b_v else math.nan,
                "KS_distance": ks_distance(a_v, b_v),
            },
            "altitude_m": {
                "reference_mean": float(np.mean(a_altitude))
                if a_altitude
                else math.nan,
                "candidate_mean": float(np.mean(b_altitude))
                if b_altitude
                else math.nan,
                "KS_distance": ks_distance(a_altitude, b_altitude),
            },
        }
    return {
        "available": True,
        "reference_records": len(ref),
        "candidate_records": len(cand),
        "scalar_lockstep": lockstep_report,
        "note": (
            "Counts from one shower are correlated branching observables. "
            "Use an ensemble before interpreting KS/count differences as a gate."
        ),
        "processes": processes,
    }


def radio_observer_layout(
    output: Path, algorithm: str
) -> list[dict[str, Any]]:
    config = read_yaml(output / algorithm / "config.yaml")
    observers = config.get("observers", {})
    if not isinstance(observers, dict):
        raise ValueError(f"invalid {algorithm} observer configuration")
    layout: list[dict[str, Any]] = []
    for name, values in observers.items():
        location = values["location"]
        layout.append(
            {
                "name": name,
                "bins": int(values["number of bins"]),
                "sampling_frequency_GHz": float(values["sampling frequency"]),
                "location_m": np.asarray(location, dtype=np.float64),
            }
        )
    return layout


def band_limited_fluence(
    field: np.ndarray, dt_s: float, low_MHz: float, high_MHz: float
) -> np.ndarray:
    spectrum = np.fft.rfft(field, axis=0)
    frequencies = np.fft.rfftfreq(field.shape[0], d=dt_s)
    keep = (frequencies >= low_MHz * 1.0e6) & (
        frequencies <= high_MHz * 1.0e6
    )
    spectrum[~keep, :] = 0.0
    filtered = np.fft.irfft(spectrum, n=field.shape[0], axis=0)
    return (
        EPSILON_0_F_PER_M
        * LIGHT_SPEED_M_PER_S
        * np.sum(filtered * filtered, axis=0)
        * dt_s
    )


def radio_fluence_maps(
    output: Path,
    algorithm: str,
    bands: list[tuple[float, float]],
) -> dict[tuple[float, float], dict[int, np.ndarray]]:
    layout = radio_observer_layout(output, algorithm)
    table = table_to_numpy(output / algorithm / "observers.parquet")
    shower_ids = np.unique(table["shower"])
    expected_rows = sum(observer["bins"] for observer in layout)
    result: dict[tuple[float, float], dict[int, np.ndarray]] = {
        band: {} for band in bands
    }
    for shower in shower_ids:
        indices = np.flatnonzero(table["shower"] == shower)
        if indices.size != expected_rows:
            raise ValueError(
                f"{algorithm} shower {shower} has {indices.size} rows; "
                f"expected {expected_rows}"
            )
        offset = 0
        band_values: dict[tuple[float, float], list[np.ndarray]] = {
            band: [] for band in bands
        }
        for observer in layout:
            count = observer["bins"]
            block = indices[offset : offset + count]
            offset += count
            times_s = np.asarray(table["Time"][block], dtype=np.float64) * 1.0e-9
            if times_s.size < 2:
                raise ValueError("radio observer has fewer than two samples")
            dt_s = float(np.median(np.diff(times_s)))
            field = np.column_stack(
                [
                    np.asarray(table["Ex"][block], dtype=np.float64),
                    np.asarray(table["Ey"][block], dtype=np.float64),
                    np.asarray(table["Ez"][block], dtype=np.float64),
                ]
            )
            for band in bands:
                components = band_limited_fluence(field, dt_s, *band)
                band_values[band].append(
                    np.append(components, np.sum(components))
                )
        for band in bands:
            result[band][int(shower)] = np.asarray(band_values[band])
    return result


def radiation_energy_proxy(
    fluence: np.ndarray, locations: np.ndarray
) -> np.ndarray:
    center_xy = locations[0, :2]
    radii = np.linalg.norm(locations[:, :2] - center_xy, axis=1)
    # Text antenna coordinates can put points from the same nominal ring a
    # few micrometres apart. Group at millimetre precision so one physical
    # ring is not interpreted as two adjacent integration nodes.
    rounded = np.round(radii, decimals=RADIAL_GROUP_DECIMALS)
    unique = np.unique(rounded)
    radial_means = np.asarray(
        [np.mean(fluence[rounded == radius], axis=0) for radius in unique]
    )
    return 2.0 * math.pi * np.trapz(
        radial_means * unique[:, np.newaxis], unique, axis=0
    )


def compare_radio(
    reference: Path,
    candidate: Path,
    relative_tolerances: dict[tuple[float, float], float],
    minimum_showers: int,
    z_limit: float,
    primary_normalization_energy_GeV: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bands = list(relative_tolerances)
    ref_energy = energy_observables(reference)
    cand_energy = energy_observables(candidate)
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    all_formal_gates_pass = True
    any_formal_gate = False

    for algorithm in ("CoREAS", "ZHS"):
        ref_layout = radio_observer_layout(reference, algorithm)
        cand_layout = radio_observer_layout(candidate, algorithm)
        if len(ref_layout) != len(cand_layout):
            raise ValueError(f"{algorithm} observer counts differ")
        ref_locations = np.asarray([row["location_m"] for row in ref_layout])
        cand_locations = np.asarray([row["location_m"] for row in cand_layout])
        if not np.allclose(ref_locations, cand_locations, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{algorithm} observer locations differ")

        ref_maps = radio_fluence_maps(reference, algorithm, bands)
        cand_maps = radio_fluence_maps(candidate, algorithm, bands)
        algorithm_report: dict[str, Any] = {
            "observers": len(ref_layout),
            "bands": {},
        }
        for band in bands:
            common = sorted(set(ref_maps[band]) & set(cand_maps[band]))
            if not common:
                raise ValueError(f"{algorithm} has no common radio showers")
            ref_proxy = []
            cand_proxy = []
            ref_normalized_maps = []
            cand_normalized_maps = []
            for shower in common:
                a = ref_maps[band][shower]
                b = cand_maps[band][shower]
                if primary_normalization_energy_GeV is None:
                    a_norm = (
                        max(
                            ref_energy[shower][
                                "electromagnetic_deposited_energy_GeV"
                            ],
                            1.0e-30,
                        )
                        ** 2
                    )
                    b_norm = (
                        max(
                            cand_energy[shower][
                                "electromagnetic_deposited_energy_GeV"
                            ],
                            1.0e-30,
                        )
                        ** 2
                    )
                    normalization_label = "deposited_EM_energy_GeV_squared"
                else:
                    a_norm = b_norm = primary_normalization_energy_GeV**2
                    normalization_label = "fixed_primary_energy_GeV_squared"
                a_normalized = a / a_norm
                b_normalized = b / b_norm
                ref_normalized_maps.append(a_normalized)
                cand_normalized_maps.append(b_normalized)
                ref_proxy.append(
                    radiation_energy_proxy(a_normalized, ref_locations)[:,]
                )
                cand_proxy.append(
                    radiation_energy_proxy(b_normalized, cand_locations)[:,]
                )

            ref_proxy_array = np.asarray(ref_proxy)
            cand_proxy_array = np.asarray(cand_proxy)
            ref_mean_map = np.mean(np.stack(ref_normalized_maps), axis=0)
            cand_mean_map = np.mean(np.stack(cand_normalized_maps), axis=0)
            map_scale = np.maximum(
                np.maximum(np.abs(ref_mean_map[:, 3]), np.abs(cand_mean_map[:, 3])),
                np.finfo(np.float64).tiny,
            )
            map_relative_differences = (
                np.abs(cand_mean_map[:, 3] - ref_mean_map[:, 3]) / map_scale
            )
            component_names = ("Ex", "Ey", "Ez", "total")
            band_report: dict[str, Any] = {
                "showers": len(common),
                "frequency_MHz": list(band),
                "normalization": normalization_label,
                "normalization_energy_GeV": (
                    primary_normalization_energy_GeV
                    if primary_normalization_energy_GeV is not None
                    else "per_shower_sum_dEdX_em"
                ),
                "components": {},
                "fluence_map_relative_difference": {
                    "median": float(np.median(map_relative_differences)),
                    "p90": float(np.quantile(map_relative_differences, 0.90)),
                    "maximum": float(np.max(map_relative_differences)),
                },
            }
            for component_index, component in enumerate(component_names):
                a = ref_proxy_array[:, component_index]
                b = cand_proxy_array[:, component_index]
                ref_mean, ref_se = mean_and_standard_error(a)
                cand_mean, cand_se = mean_and_standard_error(b)
                tolerance = relative_tolerances[band]
                acceptance = radio_acceptance_decision(
                    ref_mean,
                    cand_mean,
                    ref_se,
                    cand_se,
                    tolerance,
                    z_limit,
                )
                sufficient = len(common) >= minimum_showers
                if sufficient and component == "total":
                    any_formal_gate = True
                    all_formal_gates_pass &= acceptance["accepted"]
                values = {
                    "reference_mean_J_per_GeV2": ref_mean,
                    "candidate_mean_J_per_GeV2": cand_mean,
                    "reference_standard_error": ref_se,
                    "candidate_standard_error": cand_se,
                    "relative_mean_difference": relative_difference(
                        ref_mean, cand_mean
                    ),
                    "relative_tolerance": tolerance,
                    "z_limit": z_limit,
                    **acceptance,
                    "sufficient_showers_for_gate": sufficient,
                }
                band_report["components"][component] = values
                csv_rows.append(
                    {
                        "algorithm": algorithm,
                        "low_MHz": band[0],
                        "high_MHz": band[1],
                        "component": component,
                        **values,
                    }
                )
            algorithm_report["bands"][f"{band[0]:g}-{band[1]:g}_MHz"] = band_report
        report[algorithm] = algorithm_report

    report["formal_gate"] = {
        "minimum_showers": minimum_showers,
        "enough_showers": any_formal_gate,
        "passed": all_formal_gates_pass if any_formal_gate else None,
    }
    return report, csv_rows


def compare_outputs(
    reference: Path,
    candidate: Path,
    reference_trace: Path,
    candidate_trace: Path,
    minimum_showers: int = 20,
    z_limit: float = 3.0,
    tolerance_30_80: float = 0.10,
    tolerance_50_350: float = 0.02,
    primary_normalization_energy_GeV: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref_summary = read_yaml(reference / "summary.yaml")
    cand_summary = read_yaml(candidate / "summary.yaml")
    if int(ref_summary["showers"]) != int(cand_summary["showers"]):
        raise ValueError("reference and candidate shower counts differ")
    if int(ref_summary["seed"]) != int(cand_summary["seed"]):
        raise ValueError("reference and candidate seeds differ")

    radio, radio_rows = compare_radio(
        reference,
        candidate,
        {(30.0, 80.0): tolerance_30_80, (50.0, 350.0): tolerance_50_350},
        minimum_showers,
        z_limit,
        primary_normalization_energy_GeV,
    )
    formal = radio["formal_gate"]
    status = (
        "diagnostic_insufficient_statistics"
        if not formal["enough_showers"]
        else ("passed" if formal["passed"] else "failed")
    )
    report = {
        "status": status,
        "comparison_contract": {
            "same_seed": True,
            "event_level_bitwise_identity_required": False,
            "process_trace": "diagnostic per-process distributions",
            "radio": (
                "ensemble mean energy fluence and radiation-energy proxy; "
                "raw waveform bin comparisons require identical tracks"
            ),
        },
        "configuration": {
            "showers": int(ref_summary["showers"]),
            "seed": int(ref_summary["seed"]),
            "minimum_showers_for_formal_gate": minimum_showers,
        },
        "shower_observables": compare_scalar_observables(reference, candidate),
        "longitudinal_profiles": compare_longitudinal_profiles(reference, candidate),
        "process_trace": compare_process_traces(reference_trace, candidate_trace),
        "radio": radio,
    }
    return report, radio_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--radio-csv", type=Path)
    parser.add_argument("--minimum-showers", type=int, default=20)
    parser.add_argument("--z-limit", type=float, default=3.0)
    parser.add_argument("--tolerance-30-80", type=float, default=0.10)
    parser.add_argument("--tolerance-50-350", type=float, default=0.02)
    parser.add_argument(
        "--primary-normalization-energy-gev",
        type=float,
        help=(
            "Use one fixed primary energy for radio normalization instead "
            "of each shower's electromagnetic deposited energy."
        ),
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, rows = compare_outputs(
        args.reference_output.resolve(),
        args.candidate_output.resolve(),
        args.reference_trace.resolve(),
        args.candidate_trace.resolve(),
        args.minimum_showers,
        args.z_limit,
        args.tolerance_30_80,
        args.tolerance_50_350,
        args.primary_normalization_energy_gev,
    )
    with args.report.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=True)
        destination.write("\n")
    if args.radio_csv:
        write_csv(args.radio_csv, rows)
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 2 if args.require_pass and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
