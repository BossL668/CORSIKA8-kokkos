#!/usr/bin/env python3
"""Attribute an apparent CPU/CUDA radiation-energy mean difference.

This diagnostic is intended for the common situation in which a small
validation sample shows the same signed radiation-energy shift in CoREAS and
ZHS.  Those four algorithm/band numbers are strongly correlated because they
are produced by the same shower tracks; they are not four independent tests.

The tool compares an earlier ("signal") ensemble with a larger independent
("validation") ensemble and reports:

* raw mean, median, trimmed-mean and tail-sensitive shifts;
* how much the largest reference event moves the observed mean;
* the probability of obtaining an equally negative small-sample shift when
  resampling the independent validation ensembles;
* cross-channel correlations between CoREAS/ZHS and the two frequency bands;
* radiation yield per squared weighted track length in the validation sample.

The last quantity is a diagnostic, not a physical calibration: coherent radio
energy approximately scales quadratically with the amount of charged-particle
track contributing to the pulse.  Agreement in this ratio helps distinguish a
radio-calculation bias from a difference in shower track content.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analyze_cpu_cuda_radio import (
    compare_distributions,
    radiation_proxy_distributions,
)
from compare_cuda_replay import radio_observer_layout, shower_key_index


ALGORITHMS = ("CoREAS", "ZHS")
BANDS = ((30.0, 80.0), (50.0, 350.0))
TRACK_FIELD = "weighted_track_length_m"


def signed_shift(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_mean = float(np.mean(reference))
    return (float(np.mean(candidate)) - reference_mean) / max(
        abs(reference_mean), np.finfo(np.float64).tiny
    )


def trimmed(values: np.ndarray, count: int) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if count < 0 or 2 * count >= ordered.size:
        raise ValueError("invalid symmetric trim count")
    return ordered[count : ordered.size - count] if count else ordered


def upper_tail_share(values: np.ndarray, count: int) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if count <= 0 or count > ordered.size:
        raise ValueError("invalid upper-tail count")
    total = float(np.sum(ordered))
    if total == 0.0:
        return 0.0
    return float(np.sum(ordered[-count:]) / total)


def resampled_shift_probability(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_size: int,
    threshold: float,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if sample_size <= 0 or repetitions <= 0:
        raise ValueError("sample size and repetitions must be positive")
    rng = np.random.default_rng(seed)
    reference_indices = rng.integers(
        0, reference.size, size=(repetitions, sample_size)
    )
    candidate_indices = rng.integers(
        0, candidate.size, size=(repetitions, sample_size)
    )
    reference_means = np.mean(reference[reference_indices], axis=1)
    candidate_means = np.mean(candidate[candidate_indices], axis=1)
    shifts = (candidate_means - reference_means) / np.maximum(
        np.abs(reference_means), np.finfo(np.float64).tiny
    )
    return {
        "sample_size_per_backend": sample_size,
        "repetitions": repetitions,
        "threshold": threshold,
        "probability_shift_at_or_below_threshold": float(
            np.mean(shifts <= threshold)
        ),
        "shift_quantiles": {
            "q025": float(np.quantile(shifts, 0.025)),
            "q05": float(np.quantile(shifts, 0.05)),
            "median": float(np.quantile(shifts, 0.5)),
            "q95": float(np.quantile(shifts, 0.95)),
            "q975": float(np.quantile(shifts, 0.975)),
        },
    }


def load_raw_proxies(output: Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for algorithm in ALGORITHMS:
        locations = np.asarray(
            [
                observer["location_m"]
                for observer in radio_observer_layout(output, algorithm)
            ]
        )
        for band in BANDS:
            label = f"{algorithm}_{band[0]:g}-{band[1]:g}_MHz"
            result[label] = radiation_proxy_distributions(
                output,
                algorithm,
                band,
                locations,
                normalize_by_em_deposit=False,
            )
    return result


def cross_channel_correlation(
    proxies: dict[str, dict[int, float]]
) -> dict[str, Any]:
    labels = sorted(proxies)
    common_showers = sorted(
        set.intersection(*(set(proxies[label]) for label in labels))
    )
    values = np.asarray(
        [
            [proxies[label][shower] for label in labels]
            for shower in common_showers
        ],
        dtype=np.float64,
    )
    correlation = (
        np.corrcoef(values, rowvar=False)
        if values.shape[0] > 1
        else np.full((len(labels), len(labels)), math.nan)
    )
    off_diagonal = correlation[np.triu_indices(len(labels), k=1)]
    return {
        "showers": len(common_showers),
        "labels": labels,
        "matrix": correlation.tolist(),
        "minimum_off_diagonal": (
            float(np.nanmin(off_diagonal))
            if off_diagonal.size
            else math.nan
        ),
        "median_off_diagonal": (
            float(np.nanmedian(off_diagonal))
            if off_diagonal.size
            else math.nan
        ),
    }


def load_track_field(output: Path, algorithm: str) -> dict[int, float]:
    path = output / algorithm / "summary.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as source:
        rows = yaml.safe_load(source)
    if not isinstance(rows, dict):
        return {}
    result: dict[int, float] = {}
    for key, values in rows.items():
        if isinstance(values, dict) and TRACK_FIELD in values:
            result[shower_key_index(str(key))] = float(values[TRACK_FIELD])
    return result


def log_correlation(x: np.ndarray, y: np.ndarray) -> float:
    selected = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(selected)) < 2:
        return math.nan
    return float(np.corrcoef(np.log(x[selected]), np.log(y[selected]))[0, 1])


def track_yield_report(
    reference_output: Path,
    candidate_output: Path,
    reference_proxies: dict[str, dict[int, float]],
    candidate_proxies: dict[str, dict[int, float]],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        reference_track = load_track_field(reference_output, algorithm)
        candidate_track = load_track_field(candidate_output, algorithm)
        if not reference_track or not candidate_track:
            result[algorithm] = {"available": False}
            continue
        algorithm_result: dict[str, Any] = {
            "available": True,
            "track_field": TRACK_FIELD,
            "bands": {},
        }
        for band_index, band in enumerate(BANDS):
            label = f"{algorithm}_{band[0]:g}-{band[1]:g}_MHz"
            reference_showers = sorted(
                set(reference_proxies[label]) & set(reference_track)
            )
            candidate_showers = sorted(
                set(candidate_proxies[label]) & set(candidate_track)
            )
            reference_radio = np.asarray(
                [reference_proxies[label][key] for key in reference_showers]
            )
            candidate_radio = np.asarray(
                [candidate_proxies[label][key] for key in candidate_showers]
            )
            reference_length = np.asarray(
                [reference_track[key] for key in reference_showers]
            )
            candidate_length = np.asarray(
                [candidate_track[key] for key in candidate_showers]
            )
            reference_yield = reference_radio / np.maximum(
                reference_length * reference_length,
                np.finfo(np.float64).tiny,
            )
            candidate_yield = candidate_radio / np.maximum(
                candidate_length * candidate_length,
                np.finfo(np.float64).tiny,
            )
            algorithm_result["bands"][f"{band[0]:g}-{band[1]:g}_MHz"] = {
                "reference_log_radio_track_correlation": log_correlation(
                    reference_radio, reference_length
                ),
                "candidate_log_radio_track_correlation": log_correlation(
                    candidate_radio, candidate_length
                ),
                "radio_per_weighted_track_length_squared": compare_distributions(
                    reference_yield,
                    candidate_yield,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=seed + algorithm_index * 100 + band_index,
                ),
            }
        result[algorithm] = algorithm_result
    return result


def analyze(
    signal_reference: Path,
    signal_candidate: Path,
    validation_reference: Path,
    validation_candidate: Path,
    *,
    bootstrap_repetitions: int,
    resample_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    signal_ref = load_raw_proxies(signal_reference)
    signal_cand = load_raw_proxies(signal_candidate)
    validation_ref = load_raw_proxies(validation_reference)
    validation_cand = load_raw_proxies(validation_candidate)
    channel_results: dict[str, Any] = {}
    for index, label in enumerate(sorted(signal_ref)):
        old_reference = np.asarray(list(signal_ref[label].values()))
        old_candidate = np.asarray(list(signal_cand[label].values()))
        new_reference = np.asarray(list(validation_ref[label].values()))
        new_candidate = np.asarray(list(validation_cand[label].values()))
        observed_shift = signed_shift(old_reference, old_candidate)
        trim_count = max(1, int(math.floor(0.05 * old_reference.size)))
        reference_without_maximum = np.delete(
            old_reference, int(np.argmax(old_reference))
        )
        channel_results[label] = {
            "signal_ensemble": {
                "reference_events": int(old_reference.size),
                "candidate_events": int(old_candidate.size),
                "raw_mean_shift": observed_shift,
                "raw_median_shift": (
                    float(np.median(old_candidate))
                    - float(np.median(old_reference))
                )
                / max(
                    abs(float(np.median(old_reference))),
                    np.finfo(np.float64).tiny,
                ),
                "symmetric_trim_count": trim_count,
                "trimmed_mean_shift": signed_shift(
                    trimmed(old_reference, trim_count),
                    trimmed(old_candidate, trim_count),
                ),
                "shift_after_removing_reference_maximum": signed_shift(
                    reference_without_maximum, old_candidate
                ),
                "reference_maximum_to_mean": float(
                    np.max(old_reference) / np.mean(old_reference)
                ),
                "candidate_maximum_to_mean": float(
                    np.max(old_candidate) / np.mean(old_candidate)
                ),
                "reference_upper_10pct_share": upper_tail_share(
                    old_reference, max(1, math.ceil(old_reference.size * 0.10))
                ),
                "candidate_upper_10pct_share": upper_tail_share(
                    old_candidate, max(1, math.ceil(old_candidate.size * 0.10))
                ),
            },
            "independent_validation_ensemble": compare_distributions(
                new_reference,
                new_candidate,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed + index,
            ),
            "small_sample_reproduction_from_validation": (
                resampled_shift_probability(
                    new_reference,
                    new_candidate,
                    sample_size=old_reference.size,
                    threshold=observed_shift,
                    repetitions=resample_repetitions,
                    seed=seed + 1000 + index,
                )
            ),
        }
    return {
        "interpretation_contract": {
            "signal_ensemble": (
                "the earlier small sample in which CUDA appeared low"
            ),
            "validation_ensemble": (
                "a larger independent sample; resampling probability is a "
                "finite-ensemble diagnostic, not an asymptotic p-value"
            ),
            "channels_are_independent_tests": False,
        },
        "inputs": {
            "signal_reference": str(signal_reference.resolve()),
            "signal_candidate": str(signal_candidate.resolve()),
            "validation_reference": str(validation_reference.resolve()),
            "validation_candidate": str(validation_candidate.resolve()),
        },
        "channels": channel_results,
        "cross_channel_correlation": {
            "signal_reference": cross_channel_correlation(signal_ref),
            "signal_candidate": cross_channel_correlation(signal_cand),
            "validation_reference": cross_channel_correlation(validation_ref),
            "validation_candidate": cross_channel_correlation(validation_cand),
        },
        "validation_track_yield": track_yield_report(
            validation_reference,
            validation_candidate,
            validation_ref,
            validation_cand,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=seed + 10000,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-reference", type=Path, required=True)
    parser.add_argument("--signal-candidate", type=Path, required=True)
    parser.add_argument("--validation-reference", type=Path, required=True)
    parser.add_argument("--validation-candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--resample-repetitions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if args.resample_repetitions < 1000:
        raise ValueError("at least 1000 resampling repetitions are required")
    report = analyze(
        args.signal_reference,
        args.signal_candidate,
        args.validation_reference,
        args.validation_candidate,
        bootstrap_repetitions=args.bootstrap_repetitions,
        resample_repetitions=args.resample_repetitions,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=True)
        destination.write("\n")
    print(
        json.dumps(
            {"status": "diagnostic_complete", "report": str(args.report)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
