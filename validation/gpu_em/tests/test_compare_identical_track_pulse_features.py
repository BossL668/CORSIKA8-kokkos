#!/usr/bin/env python3

from validation.gpu_em.compare_identical_track_pulse_features import (
    compare_features,
)


def feature_row(
    backend: str,
    *,
    amplitude: float = 1.0,
    width: float = 5.0,
    valid: bool = True,
    filtered: bool = True,
    method: str = "fit",
) -> dict:
    return {
        "backend": backend,
        "algorithm": "CoREAS",
        "shower": 0,
        "observer": 1,
        "radius_m": 100.0,
        "geomagnetic_peak_abs_V_per_m": amplitude,
        "pulse_width_ns": width,
        "pulse_width_valid": valid,
        "pulse_width_filter_pass": filtered,
        "pulse_method": method,
    }


def with_zhs(rows: list[dict]) -> list[dict]:
    result = list(rows)
    for row in rows:
        duplicate = dict(row)
        duplicate["algorithm"] = "ZHS"
        result.append(duplicate)
    return result


def test_identical_features_pass() -> None:
    report, rows = compare_features(
        with_zhs([feature_row("cpu_radio"), feature_row("cuda_radio")]),
        amplitude_relative_tolerance=1.0e-4,
        amplitude_absolute_tolerance=0.0,
        width_absolute_tolerance_ns=0.11,
    )
    assert report["status"] == "passed"
    assert len(rows) == 2
    assert report["algorithms"]["CoREAS"]["width_pairs"] == 1


def test_feature_disagreement_fails_with_explicit_counters() -> None:
    report, _ = compare_features(
        with_zhs(
            [
                feature_row("cpu_radio"),
                feature_row(
                    "cuda_radio",
                    amplitude=1.1,
                    width=5.2,
                    filtered=False,
                    method="other",
                ),
            ]
        ),
        amplitude_relative_tolerance=1.0e-4,
        amplitude_absolute_tolerance=0.0,
        width_absolute_tolerance_ns=0.11,
    )
    assert report["status"] == "failed"
    coreas = report["algorithms"]["CoREAS"]
    assert coreas["amplitude_failures"] == 1
    assert coreas["width_failures"] == 1
    assert coreas["width_filter_mismatches"] == 1
    assert coreas["pulse_method_mismatches"] == 1


def test_width_validity_mismatch_is_a_failure() -> None:
    report, _ = compare_features(
        with_zhs(
            [
                feature_row("cpu_radio", valid=True),
                feature_row("cuda_radio", valid=False, filtered=False),
            ]
        ),
        amplitude_relative_tolerance=1.0e-4,
        amplitude_absolute_tolerance=0.0,
        width_absolute_tolerance_ns=0.11,
    )
    assert report["algorithms"]["CoREAS"]["width_validity_mismatches"] == 1
    assert report["status"] == "failed"
