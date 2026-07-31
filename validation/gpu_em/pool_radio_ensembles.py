#!/usr/bin/env python3
"""Pool independent radio-validation shards and compare CPU/CUDA ensembles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from analyze_cpu_cuda_radio import (
    compare_distributions,
    read_track_diagnostics,
)
from compare_cuda_replay import (
    energy_observables,
    radio_fluence_maps,
    radio_observer_layout,
    radiation_energy_proxy,
    read_yaml,
)


ALGORITHMS = ("CoREAS", "ZHS")
BANDS = ((30.0, 80.0), (50.0, 350.0))
TRACK_FIELDS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-output",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--equivalence-relative-tolerance", type=float, default=0.10
    )
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help=(
            "Return exit code 2 unless the robust deposited-energy-normalized "
            "radio comparison passes."
        ),
    )
    parser.add_argument(
        "--skip-track-diagnostics",
        action="store_true",
        help=(
            "Skip the expensive scalar/CUDA track-ledger diagnostics. They "
            "are not part of radio-energy acceptance because the two backends "
            "record different ledger populations."
        ),
    )
    return parser.parse_args()


def collect(
    roots: list[Path],
    algorithm: str,
    band: tuple[float, float],
) -> dict[str, Any]:
    raw: list[float] = []
    normalized: list[float] = []
    track_normalized: list[float] = []
    records: list[dict[str, Any]] = []
    reference_locations: np.ndarray | None = None
    for root in roots:
        resolved = root.resolve()
        layout = radio_observer_layout(resolved, algorithm)
        locations = np.asarray(
            [observer["location_m"] for observer in layout],
            dtype=np.float64,
        )
        if reference_locations is None:
            reference_locations = locations
        elif not np.allclose(
            locations,
            reference_locations,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                f"{algorithm} observer locations differ in {resolved}"
            )
        maps = radio_fluence_maps(resolved, algorithm, [band])[band]
        energies = energy_observables(resolved)
        track_summary = read_track_diagnostics(
            resolved, algorithm
        )
        for shower in sorted(maps):
            if shower not in energies:
                raise ValueError(
                    f"missing energy summary for shower {shower}: {resolved}"
                )
            proxy = float(
                radiation_energy_proxy(
                    maps[shower],
                    locations,
                )[-1]
            )
            electromagnetic_energy = float(
                energies[shower][
                    "electromagnetic_deposited_energy_GeV"
                ]
            )
            if (
                not math.isfinite(electromagnetic_energy)
                or electromagnetic_energy <= 0.0
            ):
                raise ValueError(
                    "invalid electromagnetic deposited energy for "
                    f"shower {shower}: {resolved}"
                )
            value = proxy / (electromagnetic_energy * electromagnetic_energy)
            track_key = f"shower_{shower}"
            if track_key not in track_summary:
                raise ValueError(
                    f"missing {algorithm} track summary for shower "
                    f"{shower}: {resolved}"
                )
            weighted_track_length = float(
                track_summary[track_key]["weighted_track_length_m"]
            )
            if (
                not math.isfinite(weighted_track_length)
                or weighted_track_length <= 0.0
            ):
                raise ValueError(
                    "invalid weighted track length for "
                    f"shower {shower}: {resolved}"
                )
            track_value = proxy / (
                weighted_track_length * weighted_track_length
            )
            raw.append(proxy)
            normalized.append(value)
            track_normalized.append(track_value)
            records.append(
                {
                    "output": str(resolved),
                    "shower": int(shower),
                    "raw_radiation_energy_proxy_J": proxy,
                    "electromagnetic_deposited_energy_GeV":
                        electromagnetic_energy,
                    "normalized_proxy_J_per_GeV2": value,
                    "weighted_track_length_m": weighted_track_length,
                    "track_length_normalized_proxy_J_per_m2": track_value,
                }
            )
    return {
        "raw": np.asarray(raw, dtype=np.float64),
        "normalized": np.asarray(normalized, dtype=np.float64),
        "track_length_normalized": np.asarray(
            track_normalized,
            dtype=np.float64,
        ),
        "records": records,
    }


def collect_track_fields(
    roots: list[Path], algorithm: str
) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    record_count = 0
    for root in roots:
        summary = read_track_diagnostics(
            root.resolve(), algorithm
        )
        for shower in sorted(
            summary,
            key=lambda item: int(str(item).removeprefix("shower_")),
        ):
            record = summary[shower]
            record_count += 1
            for field in TRACK_FIELDS:
                if field in record:
                    values.setdefault(field, []).append(
                        float(record[field])
                    )
    return {
        field: np.asarray(field_values, dtype=np.float64)
        for field, field_values in values.items()
        if len(field_values) == record_count
    }


def leave_one_out_shift(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    if reference.size < 2 or candidate.size < 2:
        return {}
    shifts: list[float] = []
    for index in range(reference.size):
        a = np.delete(reference, index)
        shifts.append(
            (float(np.mean(candidate)) - float(np.mean(a)))
            / max(abs(float(np.mean(a))), np.finfo(np.float64).tiny)
        )
    for index in range(candidate.size):
        b = np.delete(candidate, index)
        shifts.append(
            (float(np.mean(b)) - float(np.mean(reference)))
            / max(
                abs(float(np.mean(reference))),
                np.finfo(np.float64).tiny,
            )
        )
    return {
        "minimum": float(np.min(shifts)),
        "maximum": float(np.max(shifts)),
    }


def trimmed_shift(
    reference: np.ndarray,
    candidate: np.ndarray,
    fraction: float,
) -> float:
    def trimmed(values: np.ndarray) -> float:
        count = int(math.floor(fraction * values.size))
        ordered = np.sort(values)
        core = ordered[count : values.size - count] if count else ordered
        return float(np.mean(core))

    a = trimmed(reference)
    b = trimmed(candidate)
    return (b - a) / max(abs(a), np.finfo(np.float64).tiny)


def paired_ratio_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or a.size == 0
        or np.any(~np.isfinite(a))
        or np.any(~np.isfinite(b))
        or np.any(a <= 0.0)
        or np.any(b <= 0.0)
    ):
        return {"available": False}
    ratios = b / a
    log_ratios = np.log(ratios)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, a.size, size=(repetitions, a.size)
    )
    bootstrap_geometric = np.exp(
        np.mean(log_ratios[indices], axis=1)
    )
    return {
        "available": True,
        "pairs": int(a.size),
        "aggregate_candidate_over_reference": (
            float(np.sum(b) / np.sum(a))
        ),
        "geometric_mean_candidate_over_reference": float(
            np.exp(np.mean(log_ratios))
        ),
        "geometric_mean_ratio_bootstrap_95pct": [
            float(np.quantile(bootstrap_geometric, 0.025)),
            float(np.quantile(bootstrap_geometric, 0.975)),
        ],
        "median_candidate_over_reference": float(
            np.median(ratios)
        ),
        "mean_paired_fractional_difference": float(
            np.mean(ratios - 1.0)
        ),
        "pearson": (
            float(np.corrcoef(a, b)[0, 1])
            if a.size > 1
            and float(np.std(a)) > 0.0
            and float(np.std(b)) > 0.0
            else math.nan
        ),
    }


def geometric_mean_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Compare positive heavy-tailed values on their logarithmic scale."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if (
        a.ndim != 1
        or b.ndim != 1
        or a.size < 2
        or b.size < 2
        or np.any(~np.isfinite(a))
        or np.any(~np.isfinite(b))
        or np.any(a <= 0.0)
        or np.any(b <= 0.0)
    ):
        return {"available": False}
    log_a = np.log(a)
    log_b = np.log(b)
    rng = np.random.default_rng(seed)
    bootstrap_ratios = np.empty(repetitions, dtype=np.float64)
    batch_size = 256
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        count = stop - start
        sample_a = rng.integers(
            0, log_a.size, size=(count, log_a.size)
        )
        sample_b = rng.integers(
            0, log_b.size, size=(count, log_b.size)
        )
        bootstrap_ratios[start:stop] = np.exp(
            np.mean(log_b[sample_b], axis=1)
            - np.mean(log_a[sample_a], axis=1)
        )
    ks = stats.ks_2samp(a, b, method="exact")
    ratio = float(np.exp(np.mean(log_b) - np.mean(log_a)))
    return {
        "available": True,
        "reference_count": int(a.size),
        "candidate_count": int(b.size),
        "reference_geometric_mean": float(np.exp(np.mean(log_a))),
        "candidate_geometric_mean": float(np.exp(np.mean(log_b))),
        "candidate_over_reference": ratio,
        "bootstrap_candidate_over_reference_95pct": [
            float(np.quantile(bootstrap_ratios, 0.025)),
            float(np.quantile(bootstrap_ratios, 0.975)),
        ],
        "KS_distance": float(ks.statistic),
        "KS_p_value": float(ks.pvalue),
        "bootstrap_repetitions": repetitions,
    }


def holm_bonferroni_rejections(
    p_values: list[float], familywise_alpha: float
) -> list[bool]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    rejected = [False] * len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        threshold = familywise_alpha / (len(p_values) - rank)
        if p_value > threshold:
            break
        rejected[original_index] = True
    return rejected


def robust_radio_acceptance(
    report: dict[str, Any],
    *,
    relative_tolerance: float,
    familywise_alpha: float,
) -> dict[str, Any]:
    metric_paths = [
        (algorithm, f"{band[0]:g}-{band[1]:g}_MHz")
        for algorithm in ALGORITHMS
        for band in BANDS
    ]
    p_values = [
        float(
            report["algorithms"][algorithm][band][
                "deposited_energy_normalized_robust"
            ]["KS_p_value"]
        )
        for algorithm, band in metric_paths
    ]
    rejected = holm_bonferroni_rejections(
        p_values, familywise_alpha
    )
    lower = 1.0 - relative_tolerance
    upper = 1.0 + relative_tolerance
    results: dict[str, Any] = {}
    for (algorithm, band), shape_rejected in zip(
        metric_paths, rejected
    ):
        result = report["algorithms"][algorithm][band][
            "deposited_energy_normalized_robust"
        ]
        interval = result[
            "bootstrap_candidate_over_reference_95pct"
        ]
        interval_contained = (
            float(interval[0]) >= lower and float(interval[1]) <= upper
        )
        key = f"{algorithm}_{band}"
        results[key] = {
            "passed": bool(interval_contained and not shape_rejected),
            "equivalence_interval": [lower, upper],
            "bootstrap_ratio_interval": interval,
            "confidence_interval_contained": interval_contained,
            "KS_p_value": result["KS_p_value"],
            "holm_bonferroni_rejected": shape_rejected,
        }
    passed = all(result["passed"] for result in results.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scientific_scope": (
            "per-shower radio-energy proxy normalized by deposited EM energy "
            "squared; geometric means are used for the positive heavy-tailed "
            "distribution"
        ),
        "relative_equivalence_tolerance": relative_tolerance,
        "shape_test_correction": "Holm-Bonferroni",
        "shape_test_familywise_alpha": familywise_alpha,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if not (0.0 < args.equivalence_relative_tolerance < 1.0):
        raise ValueError("equivalence tolerance must be in (0, 1)")
    if not (0.0 < args.familywise_alpha < 1.0):
        raise ValueError("familywise alpha must be in (0, 1)")
    report: dict[str, Any] = {
        "comparison_semantics": (
            "pooled independent shower ensembles; repeated shower IDs in "
            "different output roots remain independent records"
        ),
        "reference_outputs": [
            str(path.resolve()) for path in args.reference_output
        ],
        "candidate_outputs": [
            str(path.resolve()) for path in args.candidate_output
        ],
        "algorithms": {},
    }
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        report["algorithms"][algorithm] = {}
        if args.skip_track_diagnostics:
            report["algorithms"][algorithm]["track_diagnostics"] = {
                "skipped": True,
                "reason": (
                    "excluded from formal acceptance because backend track "
                    "ledger populations are not comparable"
                ),
            }
        else:
            reference_tracks = collect_track_fields(
                args.reference_output, algorithm
            )
            candidate_tracks = collect_track_fields(
                args.candidate_output, algorithm
            )
            common_track_fields = [
                field
                for field in TRACK_FIELDS
                if field in reference_tracks
                and field in candidate_tracks
            ]
            report["algorithms"][algorithm][
                "track_diagnostics"
            ] = {
                field: compare_distributions(
                    reference_tracks[field],
                    candidate_tracks[field],
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    seed=(
                        args.seed
                        + algorithm_index * 100000
                        + index
                    ),
                )
                for index, field in enumerate(common_track_fields)
            }
        for band_index, band in enumerate(BANDS):
            label = f"{band[0]:g}-{band[1]:g}_MHz"
            reference = collect(args.reference_output, algorithm, band)
            candidate = collect(args.candidate_output, algorithm, band)
            band_report: dict[str, Any] = {}
            for kind_index, kind in enumerate(
                ("raw", "normalized", "track_length_normalized")
            ):
                a = reference[kind]
                b = candidate[kind]
                comparison = compare_distributions(
                    a,
                    b,
                    bootstrap_repetitions=args.bootstrap_repetitions,
                    seed=(
                        args.seed
                        + algorithm_index * 100000
                        + band_index * 10000
                        + kind_index * 1000
                    ),
                )
                comparison["trimmed_5pct_signed_shift"] = trimmed_shift(
                    a, b, 0.05
                )
                comparison["leave_one_out_signed_shift_range"] = (
                    leave_one_out_shift(a, b)
                )
                if (
                    len(args.reference_output) == 1
                    and len(args.candidate_output) == 1
                ):
                    comparison["paired_same_index_diagnostic"] = (
                        paired_ratio_summary(
                            a,
                            b,
                            args.bootstrap_repetitions,
                            args.seed
                            + algorithm_index * 100000
                            + band_index * 10000
                            + kind_index * 1000
                            + 500,
                        )
                    )
                band_report[kind] = comparison
            band_report["deposited_energy_normalized_robust"] = (
                geometric_mean_comparison(
                    reference["normalized"],
                    candidate["normalized"],
                    repetitions=args.bootstrap_repetitions,
                    seed=(
                        args.seed
                        + algorithm_index * 100000
                        + band_index * 10000
                        + 9000
                    ),
                )
            )
            band_report["track_length_normalization_comparability"] = {
                "comparable": False,
                "reason": (
                    "the scalar RadioProcess and CUDA resident-radio path "
                    "record different track-ledger populations; this "
                    "normalization remains a diagnostic and is excluded from "
                    "acceptance"
                ),
            }
            band_report["reference_records"] = reference["records"]
            band_report["candidate_records"] = candidate["records"]
            report["algorithms"][algorithm][label] = band_report
    report["acceptance"] = robust_radio_acceptance(
        report,
        relative_tolerance=args.equivalence_relative_tolerance,
        familywise_alpha=args.familywise_alpha,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(
        json.dumps(
            {
                algorithm: {
                    band: {
                        kind: {
                            "reference_count": values[kind]["reference"][
                                "count"
                            ],
                            "candidate_count": values[kind]["candidate"][
                                "count"
                            ],
                            "signed_shift": values[kind][
                                "signed_relative_shift_from_reference"
                            ],
                            "z_score": values[kind]["z_score"],
                            "bootstrap_95pct": values[kind][
                                "bootstrap_signed_relative_mean_shift_95pct"
                            ],
                        }
                        for kind in (
                            "raw",
                            "normalized",
                            "track_length_normalized",
                        )
                    }
                    for band, values in report["algorithms"][
                        algorithm
                    ].items()
                    if band != "track_diagnostics"
                }
                for algorithm in ALGORITHMS
            },
            indent=2,
        )
    )
    if args.require_pass and not report["acceptance"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
