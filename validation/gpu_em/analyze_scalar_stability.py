#!/usr/bin/env python3
"""Quantify whether CPU/CUDA scalar-mean differences are statistically resolved."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap independent per-shower CPU/CUDA scalar observables and "
            "estimate the ensemble size needed for a requested precision."
        )
    )
    parser.add_argument("--per-shower", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        action="append",
        help="Observable column to analyse; repeat to select several.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=8052026)
    return parser.parse_args()


def _finite_vector(frame: pd.DataFrame, backend: str, metric: str) -> np.ndarray:
    selected = frame.loc[frame["backend"] == backend, metric].to_numpy(
        dtype=np.float64
    )
    if selected.size < 2:
        raise ValueError(f"{metric}: backend {backend} has fewer than two events")
    if not np.isfinite(selected).all():
        raise ValueError(f"{metric}: backend {backend} contains NaN or infinity")
    return selected


def _signed_relative(difference: np.ndarray | float, reference: np.ndarray | float):
    scale = np.abs(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.asarray(difference) / scale
    if np.ndim(result) == 0:
        if float(scale) == 0.0:
            return 0.0 if float(difference) == 0.0 else None
        return float(result)
    result = np.where(
        scale > 0.0,
        result,
        np.where(np.asarray(difference) == 0.0, 0.0, np.nan),
    )
    return result


def analyse_metric(
    cpu: np.ndarray,
    gpu: np.ndarray,
    *,
    relative_tolerance: float,
    sigma_limit: float,
    confidence: float,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if relative_tolerance <= 0.0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative tolerance must be finite and positive")
    if sigma_limit <= 0.0 or not math.isfinite(sigma_limit):
        raise ValueError("sigma limit must be finite and positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")

    cpu_mean = float(np.mean(cpu))
    gpu_mean = float(np.mean(gpu))
    cpu_sd = float(np.std(cpu, ddof=1))
    gpu_sd = float(np.std(gpu, ddof=1))
    difference = gpu_mean - cpu_mean
    relative_difference = _signed_relative(difference, cpu_mean)
    standard_error = math.hypot(
        cpu_sd / math.sqrt(cpu.size),
        gpu_sd / math.sqrt(gpu.size),
    )
    z_score = difference / standard_error if standard_error > 0.0 else (
        0.0 if difference == 0.0 else math.copysign(math.inf, difference)
    )

    # Keep bootstrap memory bounded for the UHE matrices.  A direct
    # (repetitions, events) index matrix becomes hundreds of gigabytes when
    # the high-variance ground-tail gate needs O(1e5) showers.
    maximum_indices_per_chunk = 2_000_000
    chunk_size = max(
        1,
        min(
            bootstrap_repetitions,
            maximum_indices_per_chunk // max(cpu.size, gpu.size),
        ),
    )
    relative_boot = np.empty(bootstrap_repetitions, dtype=np.float64)
    for begin in range(0, bootstrap_repetitions, chunk_size):
        end = min(begin + chunk_size, bootstrap_repetitions)
        count = end - begin
        cpu_indices = rng.integers(0, cpu.size, size=(count, cpu.size))
        gpu_indices = rng.integers(0, gpu.size, size=(count, gpu.size))
        cpu_boot = np.mean(cpu[cpu_indices], axis=1)
        gpu_boot = np.mean(gpu[gpu_indices], axis=1)
        relative_boot[begin:end] = _signed_relative(
            gpu_boot - cpu_boot, cpu_boot
        )
    # Sparse ground observables legitimately contain many exactly-zero
    # showers. A bootstrap resample can therefore have a zero CPU reference
    # and a non-zero CUDA mean even though the full ensemble reference is
    # positive. The CPU-relative ratio is undefined for that resample; it is
    # not a corrupt input and must not invalidate an otherwise complete
    # physics run. Keep the undefined count explicit and classify only when
    # at least 100 finite resamples remain.
    finite_boot = relative_boot[np.isfinite(relative_boot)]
    undefined_bootstrap = bootstrap_repetitions - finite_boot.size
    minimum_finite_bootstrap = 100
    if finite_boot.size >= minimum_finite_bootstrap:
        tail = 0.5 * (1.0 - confidence)
        lower, upper = np.quantile(
            finite_boot, [tail, 1.0 - tail]
        )
        tolerance = relative_tolerance
        if lower >= -tolerance and upper <= tolerance:
            classification = "equivalent"
        elif lower > tolerance or upper < -tolerance:
            classification = "different"
        else:
            classification = "inconclusive"
        bootstrap_interval: list[float] | None = [
            float(lower),
            float(upper),
        ]
    else:
        classification = "inconclusive_zero_reference"
        bootstrap_interval = None

    scale = abs(cpu_mean)
    if scale == 0.0:
        required_events = None
        precision = None
    else:
        precision = sigma_limit * standard_error / scale
        variance_sum = cpu_sd * cpu_sd + gpu_sd * gpu_sd
        required_events = max(
            2,
            int(
                math.ceil(
                    sigma_limit
                    * sigma_limit
                    * variance_sum
                    / ((relative_tolerance * scale) ** 2)
                )
            ),
        )

    return {
        "cpu": {
            "events": int(cpu.size),
            "mean": cpu_mean,
            "standard_deviation": cpu_sd,
        },
        "cuda": {
            "events": int(gpu.size),
            "mean": gpu_mean,
            "standard_deviation": gpu_sd,
        },
        "difference": difference,
        "signed_relative_difference": relative_difference,
        "z_score": z_score,
        "current_sigma_precision": precision,
        "bootstrap": {
            "confidence": confidence,
            "repetitions": bootstrap_repetitions,
            "finite_relative_repetitions": int(finite_boot.size),
            "undefined_zero_reference_repetitions": int(
                undefined_bootstrap
            ),
            "signed_relative_interval": bootstrap_interval,
        },
        "required_events_per_backend_for_sigma_precision": required_events,
        "relative_tolerance": relative_tolerance,
        "sigma_limit": sigma_limit,
        "classification": classification,
    }


def analyse_frame(
    frame: pd.DataFrame,
    metrics: list[str] | None,
    *,
    relative_tolerance: float,
    sigma_limit: float,
    confidence: float,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if "backend" not in frame:
        raise ValueError("per-shower table has no backend column")
    observed_backends = set(frame["backend"].astype(str))
    required_backends = {"proposal", "cuda"}
    if not required_backends.issubset(observed_backends):
        raise ValueError(
            f"per-shower table must contain {sorted(required_backends)}"
        )
    if metrics is None:
        metrics = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column != "shower"
        ]
    if not metrics:
        raise ValueError("no scalar metrics were selected")
    unknown = [metric for metric in metrics if metric not in frame]
    if unknown:
        raise ValueError(f"unknown scalar metrics: {unknown}")

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for metric in metrics:
        results[metric] = analyse_metric(
            _finite_vector(frame, "proposal", metric),
            _finite_vector(frame, "cuda", metric),
            relative_tolerance=relative_tolerance,
            sigma_limit=sigma_limit,
            confidence=confidence,
            bootstrap_repetitions=bootstrap_repetitions,
            rng=rng,
        )
    counts = frame.groupby("backend").size()
    return {
        "source_events": {
            "proposal": int(counts.get("proposal", 0)),
            "cuda": int(counts.get("cuda", 0)),
        },
        "seed": seed,
        "metrics": results,
    }


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.per_shower)
    result = analyse_frame(
        frame,
        args.metric,
        relative_tolerance=args.relative_tolerance,
        sigma_limit=args.sigma_limit,
        confidence=args.confidence,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
