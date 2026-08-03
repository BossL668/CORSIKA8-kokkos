#!/usr/bin/env python3
"""Compare one legacy CPU shower with its decision-injected CUDA replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml


ALGORITHMS = ("CoREAS", "ZHS")
COMPONENTS = ("Ex", "Ey", "Ez")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*.parquet"))
    }


def verify_capture_is_noninvasive(
    untaped: Path, taped: Path
) -> dict[str, Any]:
    a = parquet_files(untaped)
    b = parquet_files(taped)
    if set(a) != set(b):
        raise ValueError("tape-on/off Parquet file sets differ")
    rows = []
    accepted = True
    for relative in sorted(a):
        hash_a = sha256_file(a[relative])
        hash_b = sha256_file(b[relative])
        equal = hash_a == hash_b
        accepted &= equal
        rows.append(
            {
                "path": str(relative),
                "untaped_sha256": hash_a,
                "taped_sha256": hash_b,
                "bitwise_equal": equal,
            }
        )
    return {"accepted": accepted, "files": rows}


def read_layout(root: Path, algorithm: str) -> list[dict[str, Any]]:
    with (root / algorithm / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    observers = config.get("observers", {})
    if not isinstance(observers, dict):
        raise ValueError(f"{algorithm} observer config is invalid")
    result = []
    for name, values in observers.items():
        result.append(
            {
                "name": name,
                "bins": int(values["number of bins"]),
                "sampling_frequency_GHz": float(
                    values["sampling frequency"]
                ),
                "location_m": np.asarray(
                    values["location"], dtype=np.float64
                ),
            }
        )
    return result


def read_table(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        name: np.asarray(table[name].to_numpy(zero_copy_only=False))
        for name in table.column_names
    }


def waveform_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    difference = candidate - reference
    maximum_absolute = float(np.max(np.abs(difference), initial=0.0))
    peak = float(
        max(
            np.max(np.abs(reference), initial=0.0),
            np.max(np.abs(candidate), initial=0.0),
            np.finfo(np.float64).tiny,
        )
    )
    reference_l2 = float(np.linalg.norm(reference))
    candidate_l2 = float(np.linalg.norm(candidate))
    l2_scale = max(
        reference_l2, candidate_l2, np.finfo(np.float64).tiny
    )
    reference_fluence = float(np.dot(reference, reference))
    candidate_fluence = float(np.dot(candidate, candidate))
    fluence_scale = max(
        reference_fluence,
        candidate_fluence,
        np.finfo(np.float64).tiny,
    )
    return {
        "maximum_absolute_V_per_m": maximum_absolute,
        "peak_V_per_m": peak,
        "peak_normalized_maximum": maximum_absolute / peak,
        "relative_l2": float(np.linalg.norm(difference)) / l2_scale,
        "relative_fluence": abs(
            candidate_fluence - reference_fluence
        )
        / fluence_scale,
    }


def compare_radio(
    reference_root: Path,
    replay_root: Path,
    *,
    peak_tolerance: float,
    l2_tolerance: float,
    fluence_tolerance: float,
    absolute_tolerance: float,
    time_tolerance_ns: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"algorithms": {}}
    accepted = True
    for algorithm in ALGORITHMS:
        reference_layout = read_layout(reference_root, algorithm)
        replay_layout = read_layout(replay_root, algorithm)
        if len(reference_layout) != len(replay_layout):
            raise ValueError(f"{algorithm} observer counts differ")
        for a, b in zip(reference_layout, replay_layout):
            if a["bins"] != b["bins"]:
                raise ValueError(f"{algorithm} observer bins differ")
            if not np.allclose(
                a["location_m"], b["location_m"], rtol=0.0, atol=1e-9
            ):
                raise ValueError(f"{algorithm} observer locations differ")
            if not math.isclose(
                a["sampling_frequency_GHz"],
                b["sampling_frequency_GHz"],
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"{algorithm} observer sampling frequencies differ"
                )

        reference = read_table(
            reference_root / algorithm / "observers.parquet"
        )
        replay = read_table(replay_root / algorithm / "observers.parquet")
        if set(reference) != set(replay):
            raise ValueError(f"{algorithm} columns differ")
        reference_showers = np.unique(reference["shower"])
        replay_showers = np.unique(replay["shower"])
        if not np.array_equal(reference_showers, replay_showers):
            raise ValueError(f"{algorithm} shower IDs differ")

        algorithm_report: dict[str, Any] = {
            "observers": len(reference_layout),
            "showers": int(reference_showers.size),
            "comparisons": [],
        }
        for shower in reference_showers:
            ref_indices = np.flatnonzero(reference["shower"] == shower)
            replay_indices = np.flatnonzero(replay["shower"] == shower)
            expected_rows = sum(row["bins"] for row in reference_layout)
            if (
                ref_indices.size != expected_rows
                or replay_indices.size != expected_rows
            ):
                raise ValueError(
                    f"{algorithm} shower {shower} row count differs"
                )
            offset = 0
            for observer_index, observer in enumerate(reference_layout):
                count = observer["bins"]
                ref_block = ref_indices[offset : offset + count]
                replay_block = replay_indices[offset : offset + count]
                offset += count
                maximum_time_difference = float(
                    np.max(
                        np.abs(
                            reference["Time"][ref_block]
                            - replay["Time"][replay_block]
                        ),
                        initial=0.0,
                    )
                )
                time_passed = (
                    maximum_time_difference <= time_tolerance_ns
                )
                accepted &= time_passed
                for component in COMPONENTS:
                    metrics = waveform_metrics(
                        np.asarray(
                            reference[component][ref_block],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            replay[component][replay_block],
                            dtype=np.float64,
                        ),
                    )
                    relative_passed = (
                        metrics["peak_normalized_maximum"]
                        <= peak_tolerance
                        and metrics["relative_l2"] <= l2_tolerance
                        and metrics["relative_fluence"]
                        <= fluence_tolerance
                    )
                    absolute_passed = (
                        metrics["maximum_absolute_V_per_m"]
                        <= absolute_tolerance
                    )
                    passed = (
                        time_passed
                        and (relative_passed or absolute_passed)
                    )
                    accepted &= passed
                    algorithm_report["comparisons"].append(
                        {
                            "shower": int(shower),
                            "observer_index": observer_index,
                            "reference_observer": observer["name"],
                            "component": component,
                            "maximum_time_difference_ns": (
                                maximum_time_difference
                            ),
                            **metrics,
                            "relative_gate": relative_passed,
                            "absolute_weak_signal_gate": absolute_passed,
                            "accepted": passed,
                        }
                    )

        comparisons = algorithm_report["comparisons"]
        algorithm_report["worst"] = {
            key: max(row[key] for row in comparisons)
            for key in (
                "maximum_time_difference_ns",
                "maximum_absolute_V_per_m",
                "peak_normalized_maximum",
                "relative_l2",
                "relative_fluence",
            )
        }
        algorithm_report["accepted"] = all(
            row["accepted"] for row in comparisons
        )
        report["algorithms"][algorithm] = algorithm_report
    report["accepted"] = accepted
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--untaped-reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--peak-tolerance", type=float, default=1e-4)
    parser.add_argument("--l2-tolerance", type=float, default=1e-4)
    parser.add_argument("--fluence-tolerance", type=float, default=5e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-18)
    parser.add_argument("--time-tolerance-ns", type=float, default=1e-12)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    for path in (args.reference, args.replay):
        if not path.is_dir():
            raise ValueError(f"not a directory: {path}")
    for name in (
        "peak_tolerance",
        "l2_tolerance",
        "fluence_tolerance",
        "absolute_tolerance",
        "time_tolerance_ns",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    with (args.replay / "replay_summary.json").open() as source:
        replay_summary = json.load(source)
    transport_accepted = bool(replay_summary.get("accepted", False))
    transport = replay_summary.get("transport_replay", {})
    transport_accepted &= (
        transport.get("byte_hash_mismatches") == 0
        and transport.get("invalid_records") == 0
        and transport.get("ordered_host_hash")
        == transport.get("ordered_device_hash")
    )

    noninvasive = None
    if args.untaped_reference is not None:
        noninvasive = verify_capture_is_noninvasive(
            args.untaped_reference, args.reference
        )
    radio = compare_radio(
        args.reference,
        args.replay,
        peak_tolerance=args.peak_tolerance,
        l2_tolerance=args.l2_tolerance,
        fluence_tolerance=args.fluence_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        time_tolerance_ns=args.time_tolerance_ns,
    )
    accepted = (
        transport_accepted
        and radio["accepted"]
        and (noninvasive is None or noninvasive["accepted"])
    )
    report = {
        "mode": "legacy_cpu_to_decision_injected_cuda_replay",
        "interpretation": (
            "The GPU consumed the exact scalar transport tape; this is not "
            "an independently sampled production CUDA shower."
        ),
        "transport_replay": {
            "accepted": transport_accepted,
            "summary": replay_summary,
        },
        "capture_noninvasive": noninvasive,
        "radio": radio,
        "tolerances": {
            "peak_normalized_maximum": args.peak_tolerance,
            "relative_l2": args.l2_tolerance,
            "relative_fluence": args.fluence_tolerance,
            "absolute_weak_signal_V_per_m": args.absolute_tolerance,
            "time_ns": args.time_tolerance_ns,
        },
        "accepted": accepted,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({"accepted": accepted}, indent=2))
    return 0 if accepted or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
