#!/usr/bin/env python3
"""Plot per-shower CPU and CUDA runtime distributions from CORSIKA timing output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SHOWER_PATTERN = re.compile(r"^shower_(\d+)$")
CPU_COLOR = "#4472C4"
GPU_COLOR = "#ED7D31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scalar-CPU and CUDA per-shower runtime histograms."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional external run manifest containing pooled source lists "
            "and expected event counts."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help=(
            "Allow archived timing sources without validation_provenance.json. "
            "Their source stratum is recorded explicitly as legacy/unrecorded."
        ),
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def extract_closed_showers(path: Path) -> list[tuple[int, float]]:
    timing = read_yaml(path)
    records: list[tuple[int, float]] = []
    for key, value in timing.items():
        match = SHOWER_PATTERN.fullmatch(str(key))
        if match is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid shower timing record {key} in {path}")
        if value.get("closed") is not True or value.get("status") != "closed":
            raise ValueError(f"incomplete shower timing record {key} in {path}")
        wall_time_ms = float(value.get("wall_time_ms", math.nan))
        if not math.isfinite(wall_time_ms) or wall_time_ms <= 0.0:
            raise ValueError(f"invalid wall_time_ms for {key} in {path}")
        records.append((int(match.group(1)), wall_time_ms / 1000.0))
    if not records:
        raise ValueError(f"no closed shower timing records in {path}")
    return sorted(records)


def pooled_sources(
    dataset: Path,
    backend: str,
    manifest_path: Path | None = None,
) -> list[Path]:
    if backend not in {"proposal", "cuda"}:
        raise ValueError(backend)
    roots: list[Path] = []
    if backend == "proposal":
        direct = dataset / "proposal"
        if direct.is_dir():
            roots.append(direct)
        roots.extend(
            path
            for path in sorted(dataset.glob("proposal_shard_*"))
            if path.is_dir()
        )
    else:
        direct = dataset / "cuda"
        if direct.is_dir():
            roots.append(direct)
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else dataset / "run_manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        requested = manifest.get("additional_sources", {}).get(backend, [])
        if not isinstance(requested, list):
            raise ValueError(
                f"additional_sources.{backend} must be a list in {manifest_path}"
            )
        roots.extend(Path(str(path)).resolve() for path in requested)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise ValueError(f"no {backend} timing sources below {dataset}")
    return unique


def provenance_stratum(root: Path, *, allow_legacy: bool = False) -> str:
    provenance_path = root / "validation_provenance.json"
    if not provenance_path.is_file():
        if allow_legacy:
            return "legacy:unrecorded"
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    backend = str(provenance.get("backend", "unknown"))
    executable_sha256 = str(
        provenance.get("executable", {}).get("sha256", "missing")
    )
    table = provenance.get("table")
    table_sha256 = (
        str(table.get("sha256", "missing"))
        if isinstance(table, dict)
        else "none"
    )
    return f"{backend}:exe={executable_sha256}:table={table_sha256}"


def load_cpu(
    dataset: Path,
    manifest_path: Path | None = None,
    *,
    allow_legacy_provenance: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_offset = 0
    for shard in pooled_sources(dataset, "proposal", manifest_path):
        timing_path = shard / "simulation_timing" / "summary.yaml"
        source_stratum = provenance_stratum(
            shard, allow_legacy=allow_legacy_provenance
        )
        records = extract_closed_showers(timing_path)
        local_indices = [index for index, _ in records]
        if local_indices != list(range(len(records))):
            raise ValueError(
                f"non-contiguous shower timing records in CPU source {shard}"
            )
        for local_index, seconds in records:
            rows.append(
                {
                    "backend": "CPU scalar PROPOSAL",
                    "event_index": event_offset + local_index,
                    "runtime_seconds": seconds,
                    "source": str(timing_path),
                    "source_stratum": source_stratum,
                }
            )
        event_offset += len(records)
    if not rows:
        raise ValueError(f"no proposal_shard_* outputs below {dataset}")
    return rows


def load_gpu(
    dataset: Path,
    manifest_path: Path | None = None,
    *,
    allow_legacy_provenance: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_offset = 0
    for root in pooled_sources(dataset, "cuda", manifest_path):
        timing_path = root / "simulation_timing" / "summary.yaml"
        source_stratum = provenance_stratum(
            root, allow_legacy=allow_legacy_provenance
        )
        records = extract_closed_showers(timing_path)
        local_indices = [index for index, _ in records]
        if local_indices != list(range(len(records))):
            raise ValueError(
                f"non-contiguous shower timing records in CUDA source {root}"
            )
        for local_index, seconds in records:
            rows.append(
                {
                    "backend": "CUDA",
                    "event_index": event_offset + local_index,
                    "runtime_seconds": seconds,
                    "source": str(timing_path),
                    "source_stratum": source_stratum,
                }
            )
        event_offset += len(records)
    return rows


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    sample_std = (
        float(np.std(values, ddof=1)) if values.size > 1 else None
    )
    return {
        "count": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "std_seconds": sample_std,
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "p16_seconds": float(np.percentile(values, 16)),
        "p84_seconds": float(np.percentile(values, 84)),
        "total_seconds": float(np.sum(values)),
        "coefficient_of_variation": (
            sample_std / float(np.mean(values))
            if sample_std is not None
            else None
        ),
    }


def describe_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain timing strata so pooled cross-machine data cannot look homogeneous."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("source_stratum", row["source"]))
        group = grouped.setdefault(key, {"values": [], "sources": set()})
        group["values"].append(float(row["runtime_seconds"]))
        group["sources"].add(str(row["source"]))
    return [
        {
            "stratum": stratum,
            "source_count": len(group["sources"]),
            "sources": sorted(group["sources"]),
            **describe(np.asarray(group["values"], dtype=np.float64)),
        }
        for stratum, group in sorted(grouped.items())
    ]


def histogram_edges(values: np.ndarray) -> np.ndarray:
    edges = np.histogram_bin_edges(values, bins="fd")
    number_of_bins = min(15, max(6, len(edges) - 1))
    return np.linspace(float(np.min(values)), float(np.max(values)), number_of_bins + 1)


def annotation(statistics: dict[str, float | int], unit_scale: float, unit: str) -> str:
    return "\n".join(
        [
            f"N = {statistics['count']}",
            f"Mean = {statistics['mean_seconds'] / unit_scale:.2f} {unit}",
            f"Median = {statistics['median_seconds'] / unit_scale:.2f} {unit}",
            f"Std. dev. = {statistics['std_seconds'] / unit_scale:.2f} {unit}",
            (
                f"Range = {statistics['minimum_seconds'] / unit_scale:.2f}"
                f"–{statistics['maximum_seconds'] / unit_scale:.2f} {unit}"
            ),
        ]
    )


def draw_histogram(
    values_seconds: np.ndarray,
    statistics: dict[str, float | int],
    output: Path,
    *,
    title: str,
    color: str,
    unit_scale: float,
    unit_label: str,
) -> None:
    values = values_seconds / unit_scale
    figure, axis = plt.subplots(figsize=(8.4, 5.7), constrained_layout=True)
    axis.hist(
        values,
        bins=histogram_edges(values),
        color=color,
        edgecolor="white",
        linewidth=1.0,
        alpha=0.88,
    )
    axis.axvline(
        statistics["mean_seconds"] / unit_scale,
        color="#B22222",
        linewidth=2.0,
        label="Mean",
    )
    axis.axvline(
        statistics["median_seconds"] / unit_scale,
        color="#202020",
        linewidth=2.0,
        linestyle="--",
        label="Median",
    )
    axis.set_title(title)
    axis.set_xlabel(f"Per-shower wall time ({unit_label})")
    axis.set_ylabel("Number of showers")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.98,
        0.96,
        annotation(statistics, unit_scale, unit_label),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#B0B0B0",
            "alpha": 0.92,
        },
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def draw_two_panel(
    cpu_seconds: np.ndarray,
    gpu_seconds: np.ndarray,
    cpu_statistics: dict[str, float | int],
    gpu_statistics: dict[str, float | int],
    output: Path,
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), constrained_layout=True)
    panels = [
        (
            axes[0],
            cpu_seconds / 3600.0,
            cpu_statistics,
            CPU_COLOR,
            3600.0,
            "h",
            "CPU scalar PROPOSAL",
        ),
        (
            axes[1],
            gpu_seconds,
            gpu_statistics,
            GPU_COLOR,
            1.0,
            "s",
            "CUDA",
        ),
    ]
    for axis, values, statistics, color, scale, unit, panel_title in panels:
        axis.hist(
            values,
            bins=histogram_edges(values),
            color=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.88,
        )
        axis.axvline(
            statistics["mean_seconds"] / scale,
            color="#B22222",
            linewidth=1.8,
            label="Mean",
        )
        axis.axvline(
            statistics["median_seconds"] / scale,
            color="#202020",
            linewidth=1.8,
            linestyle="--",
            label="Median",
        )
        axis.set_title(panel_title)
        axis.set_xlabel(f"Per-shower wall time ({unit})")
        axis.set_ylabel("Number of showers")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, loc="upper left")
        axis.text(
            0.98,
            0.96,
            annotation(statistics, scale, unit),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#B0B0B0",
                "alpha": 0.92,
            },
        )
    figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"dataset is not a directory: {dataset}")
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else dataset / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cpu = int(
        manifest["configuration"].get(
            "combined_proposal_events",
            manifest["configuration"]["events_per_backend"],
        )
    )
    expected_gpu = int(
        manifest["configuration"].get(
            "combined_cuda_events",
            manifest["configuration"]["events_per_backend"],
        )
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset /
            f"runtime_distribution_analysis_{expected_cpu}_{expected_gpu}"
    )
    output.mkdir(parents=True, exist_ok=False)

    cpu_rows = load_cpu(
        dataset,
        manifest_path,
        allow_legacy_provenance=args.allow_legacy_provenance,
    )
    gpu_rows = load_gpu(
        dataset,
        manifest_path,
        allow_legacy_provenance=args.allow_legacy_provenance,
    )
    if len(cpu_rows) != expected_cpu or len(gpu_rows) != expected_gpu:
        raise ValueError(
            f"expected {expected_cpu}+{expected_gpu} events, observed "
            f"{len(cpu_rows)}+{len(gpu_rows)}"
        )

    cpu_seconds = np.asarray(
        [row["runtime_seconds"] for row in cpu_rows], dtype=np.float64
    )
    gpu_seconds = np.asarray(
        [row["runtime_seconds"] for row in gpu_rows], dtype=np.float64
    )
    cpu_statistics = describe(cpu_seconds)
    gpu_statistics = describe(gpu_seconds)

    with (output / "single_event_runtimes.csv").open(
        "x", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "backend",
                "event_index",
                "runtime_seconds",
                "runtime_minutes",
                "runtime_hours",
                "source",
                "source_stratum",
            ],
        )
        writer.writeheader()
        for row in [*cpu_rows, *gpu_rows]:
            seconds = float(row["runtime_seconds"])
            writer.writerow(
                {
                    **row,
                    "runtime_minutes": seconds / 60.0,
                    "runtime_hours": seconds / 3600.0,
                }
            )

    summary = {
        "dataset": str(dataset),
        "timing_definition": (
            "simulation_timing wall_time_ms; OutputManager startOfShower "
            "through endOfShower"
        ),
        "cpu": cpu_statistics,
        "gpu": gpu_statistics,
        "cpu_source_strata": describe_sources(cpu_rows),
        "gpu_source_strata": describe_sources(gpu_rows),
        "pooled_speedup_interpretation": (
            "Descriptive only when sources span different machines, CPUs, "
            "concurrency levels, or executable builds; use an isolated "
            "same-machine benchmark for a hardware speedup claim."
        ),
        "speedup_from_means": (
            cpu_statistics["mean_seconds"] / gpu_statistics["mean_seconds"]
        ),
        "speedup_from_medians": (
            cpu_statistics["median_seconds"] / gpu_statistics["median_seconds"]
        ),
    }
    with (output / "runtime_summary.json").open(
        "x", encoding="utf-8"
    ) as destination:
        json.dump(summary, destination, indent=2, allow_nan=False)
        destination.write("\n")

    draw_histogram(
        cpu_seconds,
        cpu_statistics,
        output / "cpu_single_event_runtime_histogram.png",
        title="CPU scalar PROPOSAL: single-shower runtime distribution",
        color=CPU_COLOR,
        unit_scale=3600.0,
        unit_label="h",
    )
    draw_histogram(
        gpu_seconds,
        gpu_statistics,
        output / "gpu_single_event_runtime_histogram.png",
        title="CUDA: single-shower runtime distribution",
        color=GPU_COLOR,
        unit_scale=1.0,
        unit_label="s",
    )
    draw_two_panel(
        cpu_seconds,
        gpu_seconds,
        cpu_statistics,
        gpu_statistics,
        output / "cpu_gpu_single_event_runtime_histograms.png",
        title=(
            "Single-shower runtime distributions: "
            f"{manifest.get('label', dataset.name)}"
        ),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
