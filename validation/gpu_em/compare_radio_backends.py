#!/usr/bin/env python3
"""
Compare CPU and resident-CUDA radio projection for identical CUDA EM tracks.

Both input directories must come from c8_air_shower with the same CUDA
transport configuration and seed.  Only --radio-backend is allowed to differ.
The comparison therefore tests CoREAS/ZHS projection, not shower-to-shower
fluctuations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml


ALGORITHMS = ("CoREAS", "ZHS")
COMPONENTS = ("Ex", "Ey", "Ez")
TRANSPORT_TABLES = (
    "energyloss/dEdX.parquet",
    "profile/profile.parquet",
    "particles/particles.parquet",
    "production_profile/profile.parquet",
    "interactions/interactions.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU and CUDA CoREAS/ZHS waveforms generated from "
            "identical CUDA EM transport."
        )
    )
    parser.add_argument("--cpu-radio-output", type=Path, required=True)
    parser.add_argument("--cuda-radio-output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--l2-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--fluence-tolerance", type=float, default=5.0e-4)
    parser.add_argument(
        "--absolute-tolerance-v-per-m",
        type=float,
        default=1.0e-18,
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def validate_tolerances(args: argparse.Namespace) -> None:
    for name in (
        "relative_tolerance",
        "l2_tolerance",
        "fluence_tolerance",
        "absolute_tolerance_v_per_m",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")


def load_observer_layout(output: Path, algorithm: str) -> list[dict[str, Any]]:
    config_path = output / algorithm / "config.yaml"
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or config.get("algorithm") != algorithm:
        raise RuntimeError(f"invalid {algorithm} configuration: {config_path}")
    observers = config.get("observers")
    if not isinstance(observers, dict) or not observers:
        raise RuntimeError(f"{algorithm} has no configured observers: {config_path}")
    result: list[dict[str, Any]] = []
    for name, observer in observers.items():
        if not isinstance(observer, dict):
            raise RuntimeError(f"invalid observer {name!r}: {config_path}")
        bins = int(observer.get("number of bins", 0))
        if bins <= 0:
            raise RuntimeError(
                f"observer {name!r} has an invalid bin count: {config_path}"
            )
        result.append(
            {
                "name": str(name),
                "bins": bins,
                "start_time_ns": float(observer["start time"]),
                "duration_ns": float(observer["duration"]),
                "sampling_frequency_GHz": float(
                    observer["sampling frequency"]
                ),
                "location_m": [float(value) for value in observer["location"]],
            }
        )
    return result


def read_waveform_table(output: Path, algorithm: str) -> dict[str, np.ndarray]:
    path = output / algorithm / "observers.parquet"
    table = pq.read_table(path, columns=["shower", "Time", *COMPONENTS])
    result: dict[str, np.ndarray] = {}
    for column in ("shower", "Time", *COMPONENTS):
        result[column] = table[column].to_numpy(zero_copy_only=False)
    if any(len(values) != table.num_rows for values in result.values()):
        raise RuntimeError(f"inconsistent waveform columns: {path}")
    for component in COMPONENTS:
        if not np.all(np.isfinite(result[component])):
            raise RuntimeError(f"non-finite {algorithm} {component} waveform: {path}")
    return result


def compare_layouts(
    cpu_layout: list[dict[str, Any]],
    cuda_layout: list[dict[str, Any]],
    algorithm: str,
) -> None:
    if cpu_layout != cuda_layout:
        raise RuntimeError(
            f"{algorithm} observer configurations differ between radio backends"
        )


def split_waveforms(
    table: dict[str, np.ndarray],
    layout: list[dict[str, Any]],
    algorithm: str,
) -> list[dict[str, Any]]:
    expected_per_shower = sum(observer["bins"] for observer in layout)
    shower_ids = list(dict.fromkeys(int(value) for value in table["shower"]))
    records: list[dict[str, Any]] = []
    for shower_id in shower_ids:
        indices = np.flatnonzero(table["shower"] == shower_id)
        if len(indices) != expected_per_shower:
            raise RuntimeError(
                f"{algorithm} shower {shower_id} has {len(indices)} rows; "
                f"expected {expected_per_shower}"
            )
        # The writer emits every observer contiguously in ObserverCollection
        # order. Configuration is emitted by the same loop and preserves that
        # order, so the otherwise absent observer column is reconstructed here.
        offset = 0
        for observer_index, observer in enumerate(layout):
            count = observer["bins"]
            selected = indices[offset : offset + count]
            record: dict[str, Any] = {
                "algorithm": algorithm,
                "shower": shower_id,
                "observer_index": observer_index,
                "observer": observer["name"],
                "time": table["Time"][selected],
            }
            for component in COMPONENTS:
                record[component] = table[component][selected]
            records.append(record)
            offset += count
    return records


def component_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    relative_tolerance: float,
    l2_tolerance: float,
    fluence_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError("radio component shapes differ")
    difference = candidate - reference
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    reference_peak = float(np.max(np.abs(reference), initial=0.0))
    candidate_peak = float(np.max(np.abs(candidate), initial=0.0))
    peak_scale = max(reference_peak, candidate_peak)
    normalized_maximum = (
        max_absolute / peak_scale if peak_scale > 0.0 else 0.0
    )
    difference_l2 = float(np.linalg.norm(difference))
    reference_l2 = float(np.linalg.norm(reference))
    candidate_l2 = float(np.linalg.norm(candidate))
    l2_scale = max(reference_l2, candidate_l2)
    relative_l2 = difference_l2 / l2_scale if l2_scale > 0.0 else 0.0
    reference_fluence = float(np.dot(reference, reference))
    candidate_fluence = float(np.dot(candidate, candidate))
    fluence_scale = max(reference_fluence, candidate_fluence)
    relative_fluence = (
        abs(candidate_fluence - reference_fluence) / fluence_scale
        if fluence_scale > 0.0
        else 0.0
    )

    pointwise_accepted = (
        max_absolute <= absolute_tolerance
        or normalized_maximum <= relative_tolerance
    )
    l2_accepted = (
        difference_l2 <= absolute_tolerance * math.sqrt(reference.size)
        or relative_l2 <= l2_tolerance
    )
    fluence_absolute_tolerance = (
        absolute_tolerance * absolute_tolerance * reference.size
    )
    fluence_accepted = (
        abs(candidate_fluence - reference_fluence)
        <= fluence_absolute_tolerance
        or relative_fluence <= fluence_tolerance
    )
    return {
        "samples": int(reference.size),
        "reference_peak_V_per_m": reference_peak,
        "candidate_peak_V_per_m": candidate_peak,
        "max_absolute_difference_V_per_m": max_absolute,
        "normalized_maximum_difference": normalized_maximum,
        "relative_l2_difference": relative_l2,
        "reference_sum_squared_field": reference_fluence,
        "candidate_sum_squared_field": candidate_fluence,
        "relative_fluence_difference": relative_fluence,
        "pointwise_accepted": pointwise_accepted,
        "l2_accepted": l2_accepted,
        "fluence_accepted": fluence_accepted,
        "accepted": pointwise_accepted and l2_accepted and fluence_accepted,
    }


def compare_transport_tables(
    cpu_output: Path, cuda_output: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in TRANSPORT_TABLES:
        cpu_path = cpu_output / relative
        cuda_path = cuda_output / relative
        if not cpu_path.is_file() or not cuda_path.is_file():
            records.append(
                {
                    "path": relative,
                    "present": False,
                    "equal": False,
                }
            )
            continue
        cpu_table = pq.read_table(cpu_path)
        cuda_table = pq.read_table(cuda_path)
        records.append(
            {
                "path": relative,
                "present": True,
                "rows": cpu_table.num_rows,
                "candidate_rows": cuda_table.num_rows,
                "equal": cpu_table.equals(cuda_table, check_metadata=False),
            }
        )
    return {
        "required_tables": list(TRANSPORT_TABLES),
        "tables": records,
        "accepted": all(record["equal"] for record in records),
    }


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a YAML mapping: {path}")
    return value


def compare_run_provenance(
    cpu_output: Path, cuda_output: Path
) -> dict[str, Any]:
    cpu_summary = load_yaml_mapping(cpu_output / "summary.yaml")
    cuda_summary = load_yaml_mapping(cuda_output / "summary.yaml")
    summary_identity = {
        key: cpu_summary.get(key) == cuda_summary.get(key)
        for key in ("showers", "seed")
    }

    cpu_gpu_config = load_yaml_mapping(cpu_output / "gpu_em" / "config.yaml")
    cuda_gpu_config = load_yaml_mapping(cuda_output / "gpu_em" / "config.yaml")
    cpu_radio_backend = cpu_gpu_config.get("radio_backend")
    cuda_radio_backend = cuda_gpu_config.get("radio_backend")
    cpu_transport_config = dict(cpu_gpu_config)
    cuda_transport_config = dict(cuda_gpu_config)
    cpu_transport_config.pop("radio_backend", None)
    cuda_transport_config.pop("radio_backend", None)
    return {
        "summary_identity": summary_identity,
        "cpu_radio_backend": cpu_radio_backend,
        "cuda_radio_backend": cuda_radio_backend,
        "deterministic": (
            cpu_gpu_config.get("deterministic") is True
            and cuda_gpu_config.get("deterministic") is True
        ),
        "transport_configuration_equal": (
            cpu_transport_config == cuda_transport_config
        ),
        "accepted": (
            all(summary_identity.values())
            and cpu_radio_backend == "cpu"
            and cuda_radio_backend == "cuda"
            and cpu_gpu_config.get("deterministic") is True
            and cuda_gpu_config.get("deterministic") is True
            and cpu_transport_config == cuda_transport_config
        ),
    }


def compare_outputs(
    cpu_output: Path,
    cuda_output: Path,
    relative_tolerance: float,
    l2_tolerance: float,
    fluence_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not cpu_output.is_dir() or not cuda_output.is_dir():
        raise ValueError("both radio output paths must be directories")

    metric_rows: list[dict[str, Any]] = []
    algorithm_summaries: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        cpu_layout = load_observer_layout(cpu_output, algorithm)
        cuda_layout = load_observer_layout(cuda_output, algorithm)
        compare_layouts(cpu_layout, cuda_layout, algorithm)
        cpu_table = read_waveform_table(cpu_output, algorithm)
        cuda_table = read_waveform_table(cuda_output, algorithm)
        if not np.array_equal(cpu_table["shower"], cuda_table["shower"]):
            raise RuntimeError(f"{algorithm} shower row identities differ")
        if not np.array_equal(cpu_table["Time"], cuda_table["Time"]):
            raise RuntimeError(f"{algorithm} waveform time axes differ")
        cpu_records = split_waveforms(cpu_table, cpu_layout, algorithm)
        cuda_records = split_waveforms(cuda_table, cuda_layout, algorithm)
        if len(cpu_records) != len(cuda_records):
            raise RuntimeError(f"{algorithm} observer record counts differ")

        for reference, candidate in zip(cpu_records, cuda_records):
            identity = (
                reference["shower"],
                reference["observer_index"],
                reference["observer"],
            )
            candidate_identity = (
                candidate["shower"],
                candidate["observer_index"],
                candidate["observer"],
            )
            if identity != candidate_identity or not np.array_equal(
                reference["time"], candidate["time"]
            ):
                raise RuntimeError(f"{algorithm} observer record identities differ")
            for component in COMPONENTS:
                metrics = component_metrics(
                    reference[component],
                    candidate[component],
                    relative_tolerance,
                    l2_tolerance,
                    fluence_tolerance,
                    absolute_tolerance,
                )
                metric_rows.append(
                    {
                        "algorithm": algorithm,
                        "shower": reference["shower"],
                        "observer_index": reference["observer_index"],
                        "observer": reference["observer"],
                        "component": component,
                        **metrics,
                    }
                )
        rows = [row for row in metric_rows if row["algorithm"] == algorithm]
        algorithm_summaries[algorithm] = {
            "showers": len(set(row["shower"] for row in rows)),
            "observers": len(cpu_layout),
            "component_comparisons": len(rows),
            "maximum_normalized_difference": max(
                row["normalized_maximum_difference"] for row in rows
            ),
            "maximum_relative_l2_difference": max(
                row["relative_l2_difference"] for row in rows
            ),
            "maximum_relative_fluence_difference": max(
                row["relative_fluence_difference"] for row in rows
            ),
            "maximum_absolute_difference_V_per_m": max(
                row["max_absolute_difference_V_per_m"] for row in rows
            ),
            "accepted": all(row["accepted"] for row in rows),
        }

    provenance = compare_run_provenance(cpu_output, cuda_output)
    transport = compare_transport_tables(cpu_output, cuda_output)
    accepted = (
        provenance["accepted"]
        and
        transport["accepted"]
        and all(summary["accepted"] for summary in algorithm_summaries.values())
    )
    report = {
        "status": "passed" if accepted else "failed",
        "comparison_semantics": (
            "same CUDA EM transport seed; only radio projection backend differs"
        ),
        "inputs": {
            "cpu_radio_output": str(cpu_output.resolve()),
            "cuda_radio_output": str(cuda_output.resolve()),
        },
        "tolerances": {
            "normalized_maximum": relative_tolerance,
            "relative_l2": l2_tolerance,
            "relative_fluence": fluence_tolerance,
            "absolute_V_per_m": absolute_tolerance,
        },
        "provenance": provenance,
        "transport_identity": transport,
        "algorithms": algorithm_summaries,
        "component_comparisons": len(metric_rows),
    }
    return report, metric_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty radio comparison CSV")
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    validate_tolerances(args)
    if args.output_json.exists():
        raise ValueError(f"refusing to overwrite: {args.output_json}")
    if args.output_csv is not None and args.output_csv.exists():
        raise ValueError(f"refusing to overwrite: {args.output_csv}")
    report, rows = compare_outputs(
        args.cpu_radio_output,
        args.cuda_radio_output,
        args.relative_tolerance,
        args.l2_tolerance,
        args.fluence_tolerance,
        args.absolute_tolerance_v_per_m,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    print(json.dumps({"status": report["status"], **report["algorithms"]}, indent=2))
    return 0 if report["status"] == "passed" or not args.require_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"radio backend comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1)
