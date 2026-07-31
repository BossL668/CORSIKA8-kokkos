#!/usr/bin/env python3
"""Measure event-wise identity and correlation in a paired-seed control run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTITY_ATOL = 0.0


def analyse_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.size < 2:
        raise ValueError("paired vectors must have the same shape and at least two rows")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("paired vectors contain NaN or infinity")
    difference = candidate - reference
    reference_scale = max(
        abs(float(np.mean(reference))), np.finfo(np.float64).tiny
    )
    reference_sd = float(np.std(reference, ddof=1))
    correlation = (
        float(np.corrcoef(reference, candidate)[0, 1])
        if float(np.std(reference)) > 0.0 and float(np.std(candidate)) > 0.0
        else math.nan
    )
    return {
        "pairs": int(reference.size),
        "exact_pairs": int(np.count_nonzero(np.abs(difference) <= IDENTITY_ATOL)),
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
        "signed_relative_mean_shift": float(np.mean(difference)) / reference_scale,
        "pearson_correlation": correlation,
        "paired_rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "paired_rms_in_reference_standard_deviations": (
            float(np.sqrt(np.mean(difference * difference))) / reference_sd
            if reference_sd > 0.0
            else (0.0 if np.all(difference == 0.0) else math.inf)
        ),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
    }


def analyse_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"shower", "backend"}
    if not required.issubset(frame.columns):
        raise ValueError("per-shower table must contain shower and backend columns")
    reference = (
        frame.loc[frame["backend"] == "proposal"]
        .set_index("shower")
        .sort_index()
    )
    candidate = (
        frame.loc[frame["backend"] == "cuda"].set_index("shower").sort_index()
    )
    if not reference.index.equals(candidate.index):
        raise ValueError("proposal and CUDA shower IDs are not aligned")
    metrics = [
        column
        for column in frame.columns
        if column not in required
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return {
        "interpretation": (
            "Equal integer seeds are a diagnostic control. Exact-pair counts "
            "test event-wise reproduction; correlations show how much shower "
            "history remains paired after CPU/CUDA random streams diverge."
        ),
        "showers": int(len(reference)),
        "metrics": {
            metric: analyse_metric(
                reference[metric].to_numpy(dtype=np.float64),
                candidate[metric].to_numpy(dtype=np.float64),
            )
            for metric in metrics
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-shower", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyse_frame(pd.read_csv(args.per_shower))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=True)
        destination.write("\n")
    print(json.dumps({"showers": report["showers"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
