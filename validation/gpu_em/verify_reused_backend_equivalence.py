#!/usr/bin/env python3
"""Verify that backend reuse preserves paired CUDA shower observables bitwise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_ensembles import Ensemble, extract_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired CUDA events from a fresh-per-shower backend and "
            "the reusable resident backend."
        )
    )
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--reused", type=Path, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help=(
            "Permit diagnostic use of outputs without validation provenance; "
            "not valid for production evidence."
        ),
    )
    return parser.parse_args()


def array_result(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    if left.shape != right.shape:
        return {
            "passed": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "different_values": None,
            "maximum_absolute_difference": None,
        }
    equal = np.equal(left, right)
    equal |= np.isnan(left) & np.isnan(right)
    different = int(np.size(equal) - np.count_nonzero(equal))
    finite = np.isfinite(left) & np.isfinite(right)
    finite_difference = np.zeros(left.shape, dtype=np.float64)
    finite_difference[finite] = np.abs(
        left[finite] - right[finite]
    )
    finite_difference[~finite & ~equal] = np.inf
    maximum = (
        float(np.nanmax(finite_difference))
        if finite_difference.size
        else 0.0
    )
    return {
        "passed": different == 0,
        "shape": list(left.shape),
        "different_values": different,
        "maximum_absolute_difference": maximum,
    }


def compare_paired(
    fresh: Ensemble, reused: Ensemble, events: int
) -> dict[str, object]:
    if events <= 0:
        raise ValueError("events must be positive")
    if len(fresh.scalars) < events or len(reused.scalars) < events:
        raise ValueError(
            "requested paired event count exceeds an input ensemble"
        )
    if list(fresh.scalars.columns) != list(reused.scalars.columns):
        raise ValueError("scalar observable columns differ")
    if set(fresh.curves) != set(reused.curves):
        raise ValueError("curve observable sets differ")
    if set(fresh.histograms) != set(reused.histograms):
        raise ValueError("histogram observable sets differ")
    if (
        fresh.metadata.get("provenance_fingerprint")
        != reused.metadata.get("provenance_fingerprint")
    ):
        raise ValueError(
            "fresh and reused build/table provenance differs"
        )

    scalar_results: dict[str, object] = {}
    for column in fresh.scalars.columns:
        scalar_results[column] = array_result(
            fresh.scalars[column]
            .iloc[:events]
            .to_numpy(dtype=np.float64),
            reused.scalars[column]
            .iloc[:events]
            .to_numpy(dtype=np.float64),
        )

    curve_results: dict[str, object] = {}
    for name in sorted(fresh.curves):
        fresh_axis, fresh_values = fresh.curves[name]
        reused_axis, reused_values = reused.curves[name]
        curve_results[name] = {
            "axis": array_result(fresh_axis, reused_axis),
            "values": array_result(
                fresh_values[:events], reused_values[:events]
            ),
        }

    histogram_results: dict[str, object] = {}
    for name in sorted(fresh.histograms):
        fresh_edges, fresh_values = fresh.histograms[name]
        reused_edges, reused_values = reused.histograms[name]
        histogram_results[name] = {
            "edges": array_result(fresh_edges, reused_edges),
            "values": array_result(
                fresh_values[:events], reused_values[:events]
            ),
        }

    scalar_pass = all(
        bool(result["passed"])
        for result in scalar_results.values()
    )
    curve_pass = all(
        bool(result["axis"]["passed"])
        and bool(result["values"]["passed"])
        for result in curve_results.values()
    )
    histogram_pass = all(
        bool(result["edges"]["passed"])
        and bool(result["values"]["passed"])
        for result in histogram_results.values()
    )
    return {
        "events": events,
        "scalar_columns": len(scalar_results),
        "curve_observables": len(curve_results),
        "histogram_observables": len(histogram_results),
        "scalar_pass": scalar_pass,
        "curve_pass": curve_pass,
        "histogram_pass": histogram_pass,
        "passed": scalar_pass and curve_pass and histogram_pass,
        "scalars": scalar_results,
        "curves": curve_results,
        "histograms": histogram_results,
    }


def main() -> int:
    args = parse_args()
    fresh = extract_ensemble(
        "fresh",
        args.fresh.resolve(),
        expect_gpu=True,
        allow_legacy_provenance=
            args.allow_legacy_provenance,
    )
    reused = extract_ensemble(
        "reused",
        args.reused.resolve(),
        expect_gpu=True,
        allow_legacy_provenance=
            args.allow_legacy_provenance,
    )
    result = compare_paired(fresh, reused, args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "events",
                    "scalar_columns",
                    "curve_observables",
                    "histogram_observables",
                    "scalar_pass",
                    "curve_pass",
                    "histogram_pass",
                    "passed",
                )
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
