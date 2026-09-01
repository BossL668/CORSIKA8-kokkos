#!/usr/bin/env python3
"""Run resumable CUDA shower ensembles for one or both physics sources.

By default both c8emrt and proposal-native formal arms are run with disjoint
seed ranges.  A single source may be selected when an existing external
reference ensemble will be used.  An optional, separately labelled paired-seed
diagnostic is available only for the two-source campaign, and paired showers
are never counted as part of the independent formal ensembles.  Every
simulation is first written below a private attempt directory and is promoted
to its canonical batch directory only after all output and provenance checks
pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
SOURCES = ("c8emrt", "proposal-native")
SOURCE_DIRECTORY = {
    "c8emrt": "c8emrt",
    "proposal-native": "proposal-native",
}
HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class BatchTask:
    phase: str
    source: str
    index: int
    events: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=SOURCES,
        help=(
            "Physics-source arms to run. The default runs both sources; "
            "use '--sources proposal-native' for a native-only campaign."
        ),
    )
    parser.add_argument(
        "--c8emrt-table",
        type=Path,
        help="Required only when the c8emrt source is selected.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--antenna-file",
        type=Path,
        required=True,
        help="Observer file used by the full CUDA CoREAS and ZHS calculation.",
    )
    parser.add_argument(
        "--flupro",
        type=Path,
        default=(Path(os.environ["FLUPRO"]) if os.environ.get("FLUPRO") else None),
        help="FLUKA installation directory; defaults to the FLUPRO environment.",
    )
    parser.add_argument(
        "--native-aux-cache-dir",
        type=Path,
        help=(
            "Optional proposal-native .c8emaux cache directory. When omitted, "
            "the application's XDG cache default is used."
        ),
    )
    parser.add_argument("--events-per-source", type=int, default=500)
    parser.add_argument("--batch-events", type=int, default=25)
    parser.add_argument("--c8emrt-seed-start", type=int, default=2026301001)
    parser.add_argument("--native-seed-start", type=int, default=2026302001)
    parser.add_argument(
        "--paired-events",
        type=int,
        default=0,
        help=(
            "Optional same-seed cross-source diagnostic events, excluded from "
            "formal counts; requires both sources."
        ),
    )
    parser.add_argument("--paired-seed-start", type=int, default=2026303001)
    parser.add_argument(
        "--order",
        choices=("alternating", "c8emrt-first", "proposal-native-first"),
        default="alternating",
        help=(
            "alternating runs AB for even batches and BA for odd batches; "
            "the other modes complete one source before the other."
        ),
    )
    parser.add_argument("--energy-gev", type=float, default=1.0e3)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument(
        "--geomagnetic-model", choices=("IGRF13", "IGRF14"), default="IGRF14"
    )
    parser.add_argument("--geomagnetic-year", type=float, default=2027.0)
    parser.add_argument("--shower-core-x-m", type=float, default=0.0)
    parser.add_argument("--shower-core-y-m", type=float, default=0.0)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--radio-sampling-rate-ghz", type=float, default=1.0)
    parser.add_argument("--radio-window-duration-ns", type=float, default=400.0)
    parser.add_argument("--radio-pretrigger-ns", type=float, default=10.0)
    parser.add_argument("--em-cut-gev", type=float, default=0.5e-3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-6)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.0,
        help=(
            "Explicit EM thinning maximum weight. Zero preserves the "
            "application's automatic value by omitting --max-weight."
        ),
    )
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--gpu-radio-field-limit", type=float, default=1.0)
    parser.add_argument("--hadronic-workers", type=int, default=4)
    parser.add_argument("--hadronic-min-batch", type=int, default=64)
    parser.add_argument("--hadronic-target-batch-ms", type=float, default=5.0)
    parser.add_argument("--hadronic-max-batch", type=int, default=256)
    parser.add_argument(
        "--detailed-stage-timing",
        action="store_true",
        help="Enable profiling counters; leave disabled for primary timing evidence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the immutable plan without creating output.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return yaml.load(source, Loader=loader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_DIGITS for character in value)
    )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_arguments(args: argparse.Namespace) -> None:
    sources = tuple(args.sources)
    if not sources:
        raise ValueError("at least one GPU physics source must be selected")
    if len(set(sources)) != len(sources):
        raise ValueError(f"GPU physics sources must be unique: {sources}")
    if not args.executable.is_file():
        raise ValueError(f"executable is not a regular file: {args.executable}")
    if not os.access(args.executable, os.X_OK):
        raise ValueError(f"executable is not executable: {args.executable}")
    if "c8emrt" in sources:
        if args.c8emrt_table is None:
            raise ValueError("--c8emrt-table is required when c8emrt is selected")
        if not args.c8emrt_table.is_file():
            raise ValueError(
                f"c8emrt table is not a regular file: {args.c8emrt_table}"
            )
        if args.c8emrt_table.suffix != ".c8emrt":
            raise ValueError("--c8emrt-table must name a .c8emrt file")
    if not args.antenna_file.is_file():
        raise ValueError(f"antenna file is not a regular file: {args.antenna_file}")
    if args.flupro is None or not args.flupro.is_dir():
        raise ValueError("--flupro or the FLUPRO environment must name a directory")
    fluka_library = args.flupro / "libflukahp.a"
    if not fluka_library.is_file():
        raise ValueError(f"FLUKA library is missing: {fluka_library}")
    if "proposal-native" in sources and args.native_aux_cache_dir is not None:
        ancestor = args.native_aux_cache_dir.expanduser().resolve()
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
            raise ValueError(
                f"native aux cache has no writable existing ancestor: {ancestor}"
            )
    if args.events_per_source <= 0 or args.batch_events <= 0:
        raise ValueError("event and batch counts must be positive")
    if args.paired_events < 0:
        raise ValueError("--paired-events must be non-negative")
    if args.paired_events and set(sources) != set(SOURCES):
        raise ValueError(
            "--paired-events requires both c8emrt and proposal-native sources"
        )
    seed_starts = {
        "c8emrt": args.c8emrt_seed_start,
        "proposal-native": args.native_seed_start,
    }
    selected_seed_starts = [seed_starts[source] for source in sources]
    if args.paired_events:
        selected_seed_starts.append(args.paired_seed_start)
    if min(selected_seed_starts) < 0:
        raise ValueError("seeds must be non-negative")
    formal_ranges = {
        source: range(seed_starts[source], seed_starts[source] + args.events_per_source)
        for source in sources
    }
    for left_index, left_source in enumerate(sources):
        for right_source in sources[left_index + 1 :]:
            if ranges_overlap(
                formal_ranges[left_source], formal_ranges[right_source]
            ):
                raise ValueError(
                    f"formal {left_source} and {right_source} seed ranges overlap"
                )
    if args.paired_events:
        paired = range(
            args.paired_seed_start, args.paired_seed_start + args.paired_events
        )
        if any(ranges_overlap(formal, paired) for formal in formal_ranges.values()):
            raise ValueError("paired diagnostic seed range overlaps a formal range")
    finite_positive = (
        args.energy_gev,
        args.geomagnetic_year,
        args.radio_sampling_rate_ghz,
        args.radio_window_duration_ns,
        args.em_cut_gev,
        args.had_cut_gev,
        args.mu_cut_gev,
        args.tau_cut_gev,
        args.gpu_table_tolerance,
        args.gpu_radio_field_limit,
        args.hadronic_target_batch_ms,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
        raise ValueError("energy, cuts, tolerances, and timing values must be positive")
    if not math.isfinite(args.radio_pretrigger_ns) or args.radio_pretrigger_ns < 0.0:
        raise ValueError("radio pretrigger must be finite and non-negative")
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("EM thinning must lie in [0, 1]")
    if args.maximum_weight < 0.0 or not math.isfinite(args.maximum_weight):
        raise ValueError("maximum weight must be finite and non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must lie in (0, 1]")
    if min(
        args.gpu_min_batch,
        args.hadronic_workers,
        args.hadronic_min_batch,
        args.hadronic_max_batch,
    ) <= 0:
        raise ValueError("GPU and hadronic scheduler sizes must be positive")


def ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def immutable_configuration(args: argparse.Namespace) -> dict[str, Any]:
    fluka_library = args.flupro.resolve() / "libflukahp.a"
    sources = tuple(args.sources)
    artifacts = {
        "executable": artifact_identity(args.executable),
        "antenna_file": artifact_identity(args.antenna_file),
        "fluka_library": artifact_identity(fluka_library),
        "runner": artifact_identity(Path(__file__)),
    }
    if "c8emrt" in sources:
        if args.c8emrt_table is None:
            raise ValueError("c8emrt table identity requested without a table path")
        artifacts["c8emrt_table"] = artifact_identity(args.c8emrt_table)
    seeds: dict[str, int] = {}
    if "c8emrt" in sources:
        seeds["c8emrt_start"] = args.c8emrt_seed_start
    if "proposal-native" in sources:
        seeds["proposal_native_start"] = args.native_seed_start
    # Preserve the historical two-source immutable shape.  For a single-source
    # campaign the paired seed is intentionally absent because paired
    # diagnostics are not meaningful and are rejected by validate_arguments().
    if set(sources) == set(SOURCES):
        seeds["paired_start"] = args.paired_seed_start
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "sources": list(sources),
        "events_per_source": args.events_per_source,
        "batch_events": args.batch_events,
        "paired_events": args.paired_events,
        "order": args.order,
        "seeds": seeds,
        "physics": {
            "primary_pdg": args.primary_pdg,
            "energy_GeV": args.energy_gev,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "geomagnetic_model": args.geomagnetic_model,
            "geomagnetic_year": args.geomagnetic_year,
            "shower_core_x_m": args.shower_core_x_m,
            "shower_core_y_m": args.shower_core_y_m,
            "ring": args.ring,
            "radio_sampling_rate_GHz": args.radio_sampling_rate_ghz,
            "radio_window_duration_ns": args.radio_window_duration_ns,
            "radio_pretrigger_ns": args.radio_pretrigger_ns,
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "had_cut_GeV": args.had_cut_gev,
            "mu_cut_GeV": args.mu_cut_gev,
            "tau_cut_GeV": args.tau_cut_gev,
            "em_backend": "cuda",
            "radio_backend": "cuda",
            "hadronic_backend": "fluka-process",
        },
        "scheduler": {
            "gpu_device": args.gpu_device,
            "gpu_min_batch": args.gpu_min_batch,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "gpu_table_tolerance": args.gpu_table_tolerance,
            "gpu_deterministic": True,
            "gpu_resident_cross_species": True,
            "gpu_radio_field_limit_V_per_m": args.gpu_radio_field_limit,
            "hadronic_workers": args.hadronic_workers,
            "hadronic_min_batch": args.hadronic_min_batch,
            "hadronic_target_batch_ms": args.hadronic_target_batch_ms,
            "hadronic_max_batch": args.hadronic_max_batch,
            "detailed_stage_timing": args.detailed_stage_timing,
        },
        "artifacts": artifacts,
    }
    if "proposal-native" in sources:
        configuration["native_aux_cache_dir"] = (
            str(args.native_aux_cache_dir.expanduser().resolve())
            if args.native_aux_cache_dir is not None
            else None
        )
    return configuration


def batch_specs(total: int, batch_size: int, seed_start: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    completed = 0
    while completed < total:
        events = min(batch_size, total - completed)
        specs.append((events, seed_start + completed))
        completed += events
    return specs


def interleaved_tasks(
    phase: str,
    left_specs: list[tuple[int, int]],
    right_specs: list[tuple[int, int]],
    order: str,
) -> list[BatchTask]:
    left = [
        BatchTask(phase, "c8emrt", index, events, seed)
        for index, (events, seed) in enumerate(left_specs)
    ]
    right = [
        BatchTask(phase, "proposal-native", index, events, seed)
        for index, (events, seed) in enumerate(right_specs)
    ]
    if order == "c8emrt-first":
        return left + right
    if order == "proposal-native-first":
        return right + left
    tasks: list[BatchTask] = []
    for index in range(max(len(left), len(right))):
        pair: list[BatchTask] = []
        if index < len(left):
            pair.append(left[index])
        if index < len(right):
            pair.append(right[index])
        if index % 2:
            pair.reverse()
        tasks.extend(pair)
    return tasks


def campaign_tasks(args: argparse.Namespace) -> list[BatchTask]:
    sources = tuple(args.sources)
    seed_starts = {
        "c8emrt": args.c8emrt_seed_start,
        "proposal-native": args.native_seed_start,
    }
    formal_specs = {
        source: batch_specs(
            args.events_per_source, args.batch_events, seed_starts[source]
        )
        for source in sources
    }
    if set(sources) == set(SOURCES):
        tasks = interleaved_tasks(
            "formal",
            formal_specs["c8emrt"],
            formal_specs["proposal-native"],
            args.order,
        )
    else:
        source = sources[0]
        tasks = [
            BatchTask("formal", source, index, events, seed)
            for index, (events, seed) in enumerate(formal_specs[source])
        ]
    if args.paired_events:
        paired = batch_specs(
            args.paired_events, args.batch_events, args.paired_seed_start
        )
        tasks.extend(interleaved_tasks("paired", paired, paired, args.order))
    return tasks


def canonical_output(root: Path, task: BatchTask) -> Path:
    return (
        root
        / task.phase
        / f"batch_{task.index:03d}"
        / SOURCE_DIRECTORY[task.source]
    )


def build_command(args: argparse.Namespace, task: BatchTask, output: Path) -> list[str]:
    command = [
        str(args.executable.resolve()),
        "-p",
        str(args.primary_pdg),
        "-E",
        f"{args.energy_gev:.17g}",
        "-N",
        str(task.events),
        "-f",
        str(output.resolve()),
        "--seed",
        str(task.seed),
        "--zenith",
        f"{args.zenith_deg:.17g}",
        "--azimuth",
        f"{args.azimuth_deg:.17g}",
        "--geomagnetic-model",
        args.geomagnetic_model,
        "--geomagnetic-year",
        f"{args.geomagnetic_year:.17g}",
        "--shower-core-x",
        f"{args.shower_core_x_m:.17g}",
        "--shower-core-y",
        f"{args.shower_core_y_m:.17g}",
        "--ring",
        str(args.ring),
        "--antenna-file",
        str(args.antenna_file.resolve()),
        "--radio-sampling-rate-ghz",
        f"{args.radio_sampling_rate_ghz:.17g}",
        "--radio-window-duration-ns",
        f"{args.radio_window_duration_ns:.17g}",
        "--radio-pretrigger-ns",
        f"{args.radio_pretrigger_ns:.17g}",
        "--emcut",
        f"{args.em_cut_gev:.17g}",
        "--emthin",
        f"{args.em_thinning:.17g}",
        "--hadcut",
        f"{args.had_cut_gev:.17g}",
        "--mucut",
        f"{args.mu_cut_gev:.17g}",
        "--taucut",
        f"{args.tau_cut_gev:.17g}",
        "--verbosity",
        "warn",
        "--em-backend",
        "cuda",
        "--radio-backend",
        "cuda",
        "--gpu-device",
        str(args.gpu_device),
        "--gpu-min-batch",
        str(args.gpu_min_batch),
        "--gpu-memory-fraction",
        f"{args.gpu_memory_fraction:.17g}",
        "--gpu-physics-source",
        task.source,
        "--gpu-table-tolerance",
        f"{args.gpu_table_tolerance:.17g}",
        "--gpu-deterministic",
        "true",
        "--gpu-resident-cross-species",
        "true",
        "--gpu-radio-field-limit",
        f"{args.gpu_radio_field_limit:.17g}",
        "--hadronic-backend",
        "fluka-process",
        "--hadronic-workers",
        str(args.hadronic_workers),
        "--hadronic-min-batch",
        str(args.hadronic_min_batch),
        "--hadronic-target-batch-ms",
        f"{args.hadronic_target_batch_ms:.17g}",
        "--hadronic-max-batch",
        str(args.hadronic_max_batch),
    ]
    if task.source == "c8emrt":
        if args.c8emrt_table is None:
            raise ValueError("cannot build a c8emrt command without a table path")
        source_index = command.index("--gpu-table-tolerance")
        command[source_index:source_index] = [
            "--gpu-table-cache",
            str(args.c8emrt_table.resolve()),
        ]
    elif args.native_aux_cache_dir is not None:
        source_index = command.index("--gpu-table-tolerance")
        command[source_index:source_index] = [
            "--gpu-aux-cache-dir",
            str(args.native_aux_cache_dir.expanduser().resolve()),
        ]
    if args.maximum_weight > 0.0:
        command.extend(("--max-weight", f"{args.maximum_weight:.17g}"))
    if args.detailed_stage_timing:
        command.append("--gpu-detailed-stage-timing")
    return command


def expected_shower_keys(events: int) -> set[str]:
    return {f"shower_{index}" for index in range(events)}


def require_shower_mapping(
    path: Path,
    events: int,
    *,
    label: str,
) -> dict[str, Any]:
    value = safe_load_yaml(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a YAML mapping: {path}")
    observed = {str(key) for key in value if str(key).startswith("shower_")}
    expected = expected_shower_keys(events)
    if observed != expected:
        raise ValueError(
            f"{label} shower keys differ in {path}: "
            f"expected {len(expected)}, observed {len(observed)}"
        )
    return {key: value[key] for key in sorted(expected)}


def validate_parquet_shower_column(
    path: Path,
    events: int,
    *,
    require_every_shower: bool,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if "shower" not in parquet.schema.names:
        raise ValueError(f"Parquet file has no shower column: {path}")
    observed: set[int] = set()
    for batch in parquet.iter_batches(batch_size=65536, columns=["shower"]):
        observed.update(int(value) for value in batch.column(0).to_pylist())
    invalid = sorted(value for value in observed if value < 0 or value >= events)
    if invalid:
        raise ValueError(f"Parquet shower IDs are out of range in {path}: {invalid[:8]}")
    expected = set(range(events))
    if require_every_shower and observed != expected:
        raise ValueError(
            f"Parquet shower coverage differs in {path}: "
            f"expected {len(expected)}, observed {len(observed)}"
        )
    return {
        "rows": parquet.metadata.num_rows,
        "showers_observed": len(observed),
    }


def parse_runtime_seconds(summary: dict[str, Any]) -> float:
    for key in ("runtime_raw", "runtime"):
        value = summary.get(key)
        if isinstance(value, (float, int)):
            seconds = float(value)
            if math.isfinite(seconds) and seconds > 0.0:
                return seconds
        if not isinstance(value, str):
            continue
        text = value.strip()
        days = 0.0
        if " day" in text:
            day_text, separator, text = text.partition(",")
            if not separator:
                continue
            days = float(day_text.split()[0])
            text = text.strip()
        fields = text.split(":")
        if len(fields) != 3:
            continue
        hours, minutes, seconds_text = fields
        seconds = (
            days * 86400.0
            + float(hours) * 3600.0
            + float(minutes) * 60.0
            + float(seconds_text)
        )
        if math.isfinite(seconds) and seconds > 0.0:
            return seconds
    raise ValueError("top-level summary has no positive finite runtime")


def source_identity_from_output(
    args: argparse.Namespace,
    task: BatchTask,
    output: Path,
    gpu_summary: dict[str, Any],
) -> dict[str, Any]:
    config_path = output / "gpu_em" / "config.yaml"
    config = safe_load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"invalid GPU configuration: {config_path}")
    if config.get("gpu_physics_source") != task.source:
        raise ValueError(
            f"GPU config source mismatch: expected {task.source}, "
            f"observed {config.get('gpu_physics_source')}"
        )
    per_shower_sources: set[str] = set()
    for shower in gpu_summary.values():
        statistics = shower.get("statistics") if isinstance(shower, dict) else None
        if not isinstance(statistics, dict):
            raise ValueError("GPU summary shower has no statistics mapping")
        per_shower_sources.add(str(statistics.get("gpu_physics_source")))
    if per_shower_sources != {task.source}:
        raise ValueError(
            f"GPU summary physics sources differ: {sorted(per_shower_sources)}"
        )

    if task.source == "c8emrt":
        table_config = config.get("table")
        if not isinstance(table_config, dict):
            raise ValueError("c8emrt GPU config has no table mapping")
        configured_path = table_config.get("path")
        if not isinstance(configured_path, str) or Path(configured_path).resolve() != (
            args.c8emrt_table.resolve()
        ):
            raise ValueError("c8emrt GPU config table path differs from requested table")
        internal_hash = table_config.get("sha256")
        if not is_sha256(internal_hash):
            raise ValueError("c8emrt GPU config has an invalid serialized table hash")
        identity = artifact_identity(args.c8emrt_table)
        identity.update(
            {
                "identity_kind": "c8emrt-file",
                "serialized_payload_sha256": internal_hash,
                "proposal_version": table_config.get("proposal_version"),
                "format_version": table_config.get("format_version"),
            }
        )
        return identity

    table_hashes: set[str] = set()
    aux_hashes: set[str] = set()
    node_counts: set[int] = set()
    device_bytes: set[int] = set()
    proposal_versions: set[str] = set()
    cubic_versions: set[str] = set()
    for shower in gpu_summary.values():
        statistics = shower["statistics"]
        native = statistics.get("proposal_native")
        if not isinstance(native, dict):
            raise ValueError("proposal-native summary is missing native metadata")
        table_hashes.add(str(native.get("table_sha256")))
        aux_hashes.add(str(native.get("aux_sha256")))
        node_counts.add(int(native.get("node_count", -1)))
        device_bytes.add(int(native.get("device_bytes", -1)))
        proposal_versions.add(str(native.get("proposal_version")))
        cubic_versions.add(str(native.get("cubic_interpolation_version")))
    if len(table_hashes) != 1 or not is_sha256(next(iter(table_hashes))):
        raise ValueError(f"proposal-native table hash differs or is invalid: {table_hashes}")
    if len(aux_hashes) != 1 or not is_sha256(next(iter(aux_hashes))):
        raise ValueError(f"proposal-native aux hash differs or is invalid: {aux_hashes}")
    singleton_sets: Iterable[tuple[str, set[Any]]] = (
        ("node count", node_counts),
        ("device bytes", device_bytes),
        ("PROPOSAL version", proposal_versions),
        ("CubicInterpolation version", cubic_versions),
    )
    for label, values in singleton_sets:
        if len(values) != 1:
            raise ValueError(f"proposal-native {label} differs across showers: {values}")
    if next(iter(node_counts)) <= 0 or next(iter(device_bytes)) <= 0:
        raise ValueError("proposal-native table sizes must be positive")
    table_config = config.get("table")
    if not isinstance(table_config, dict):
        raise ValueError("proposal-native GPU config has no table mapping")
    configured_aux = table_config.get("aux_cache_directory")
    if not isinstance(configured_aux, str) or not configured_aux:
        raise ValueError("proposal-native GPU config has no aux cache directory")
    if args.native_aux_cache_dir is not None and Path(configured_aux).resolve() != (
        args.native_aux_cache_dir.expanduser().resolve()
    ):
        raise ValueError("proposal-native aux cache directory differs from request")
    return {
        "identity_kind": "proposal-native-canonical",
        "sha256": next(iter(table_hashes)),
        "aux_sha256": next(iter(aux_hashes)),
        "node_count": next(iter(node_counts)),
        "device_bytes": next(iter(device_bytes)),
        "proposal_version": next(iter(proposal_versions)),
        "cubic_interpolation_version": next(iter(cubic_versions)),
        "aux_cache_directory": str(Path(configured_aux).resolve()),
    }


def nested_counter(
    mapping: dict[str, Any], path: tuple[str, ...], *, label: str
) -> int:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"GPU statistics counter is missing: {label}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"GPU statistics counter is invalid: {label}={value!r}")
    return value


PERMITTED_GENERIC_FALLBACK_REASONS = (
    "unsupported_particle",
    "unsupported_medium",
    "unsupported_geometry",
)

SELECTED_LOSS_FALLBACK_REASONS = (
    "inverse_cdf_unavailable",
    "loss_energy_out_of_range",
    "loss_quantile_out_of_range",
    "native_selection_replay",
)

PERMITTED_SPECIFIED_FALLBACK_REASONS = (
    *SELECTED_LOSS_FALLBACK_REASONS,
    "cpu_only_process",
    "epair_rejection_envelope_exceeded",
)


def validate_permitted_generic_fallbacks(
    statistics: dict[str, Any], *, shower_label: str
) -> dict[str, int]:
    generic = nested_counter(
        statistics, ("cpu_generic_fallbacks",), label="cpu_generic_fallbacks"
    )
    specified = nested_counter(
        statistics,
        ("cpu_specified_final_states",),
        label="cpu_specified_final_states",
    )
    raw_reasons = statistics.get("cpu_fallbacks_by_reason_name")
    # yaml-cpp serializes a default-constructed empty YAML::Node as null.  The
    # application therefore emits ``null`` (rather than ``{}``) when no
    # fallback reason was observed.  Accept that representation only when the
    # independently recorded generic-fallback total is also zero; otherwise a
    # missing reason map remains a provenance/accounting failure.
    if raw_reasons is None and generic == 0 and specified == 0:
        raw_reasons = {}
    if not isinstance(raw_reasons, dict):
        raise ValueError(f"CPU fallback reason mapping is missing in {shower_label}")
    permitted: dict[str, int] = {}
    for reason in PERMITTED_GENERIC_FALLBACK_REASONS:
        value = raw_reasons.get(reason, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"invalid CPU fallback reason counter in {shower_label}: "
                f"{reason}={value!r}"
            )
        permitted[reason] = value
    permitted_total = sum(permitted.values())
    if generic != permitted_total:
        raise ValueError(
            f"generic CPU fallbacks in {shower_label} are not exactly the "
            "declared scalar-routing cases: "
            f"generic={generic}, permitted={permitted}"
        )
    return permitted


def validate_counter_mapping(
    value: Any, *, label: str, allow_null: bool
) -> dict[Any, int]:
    if value is None and allow_null:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"GPU statistics counter mapping is missing: {label}")
    result: dict[Any, int] = {}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"GPU statistics counter mapping is invalid: "
                f"{label}.{key}={count!r}"
            )
        result[key] = count
    return result


def validate_fallback_accounting(
    statistics: dict[str, Any], *, source: str, shower_label: str
) -> dict[str, Any]:
    generic = nested_counter(
        statistics, ("cpu_generic_fallbacks",), label="cpu_generic_fallbacks"
    )
    specified = nested_counter(
        statistics,
        ("cpu_specified_final_states",),
        label="cpu_specified_final_states",
    )
    queued = nested_counter(
        statistics,
        ("deferred_cpu_fallbacks_queued",),
        label="deferred_cpu_fallbacks_queued",
    )
    flushed = nested_counter(
        statistics,
        ("deferred_cpu_fallbacks_flushed",),
        label="deferred_cpu_fallbacks_flushed",
    )
    if queued != specified or flushed != specified:
        raise ValueError(
            f"specified/deferred CPU fallback accounting differs in {shower_label}: "
            f"specified={specified}, queued={queued}, flushed={flushed}"
        )

    total_fallbacks = generic + specified
    reason_names = validate_counter_mapping(
        statistics.get("cpu_fallbacks_by_reason_name"),
        label="cpu_fallbacks_by_reason_name",
        allow_null=total_fallbacks == 0,
    )
    reason_ids = validate_counter_mapping(
        statistics.get("cpu_fallbacks_by_reason"),
        label="cpu_fallbacks_by_reason",
        allow_null=total_fallbacks == 0,
    )
    process_names = validate_counter_mapping(
        statistics.get("cpu_fallbacks_by_process_name"),
        label="cpu_fallbacks_by_process_name",
        allow_null=total_fallbacks == 0,
    )
    process_ids = validate_counter_mapping(
        statistics.get("cpu_fallbacks_by_process"),
        label="cpu_fallbacks_by_process",
        allow_null=total_fallbacks == 0,
    )
    for label, mapping in (
        ("reason-name", reason_names),
        ("reason-id", reason_ids),
        ("process-name", process_names),
        ("process-id", process_ids),
    ):
        if sum(mapping.values()) != total_fallbacks:
            raise ValueError(
                f"{label} fallback map does not close in {shower_label}: "
                f"sum={sum(mapping.values())}, expected={total_fallbacks}"
            )

    known_reasons = set(PERMITTED_GENERIC_FALLBACK_REASONS) | set(
        PERMITTED_SPECIFIED_FALLBACK_REASONS
    )
    unknown_reasons = sorted(set(reason_names) - known_reasons)
    if unknown_reasons:
        raise ValueError(
            f"unpermitted CPU fallback reasons in {shower_label}: "
            f"{unknown_reasons}"
        )
    permitted_generic = validate_permitted_generic_fallbacks(
        statistics, shower_label=shower_label
    )
    specified_from_reasons = sum(
        reason_names.get(reason, 0)
        for reason in PERMITTED_SPECIFIED_FALLBACK_REASONS
    )
    if specified_from_reasons != specified:
        raise ValueError(
            f"specified CPU fallback reasons do not close in {shower_label}: "
            f"reason_total={specified_from_reasons}, specified={specified}"
        )

    completed_losses = nested_counter(
        statistics,
        ("cpu_completed_selected_losses",),
        label="cpu_completed_selected_losses",
    )
    selected_from_reasons = sum(
        reason_names.get(reason, 0) for reason in SELECTED_LOSS_FALLBACK_REASONS
    )
    if completed_losses != selected_from_reasons:
        raise ValueError(
            f"selected-loss fallback accounting differs in {shower_label}: "
            f"completed={completed_losses}, reasons={selected_from_reasons}"
        )
    native_replays = nested_counter(
        statistics,
        ("cpu_completed_native_selection_replays",),
        label="cpu_completed_native_selection_replays",
    )
    replay_reasons = reason_names.get("native_selection_replay", 0)
    if native_replays != replay_reasons:
        raise ValueError(
            f"native-selection replay accounting differs in {shower_label}: "
            f"completed={native_replays}, reasons={replay_reasons}"
        )
    if source != "proposal-native" and native_replays != 0:
        raise ValueError(
            f"{source} unexpectedly reports native selection replay in "
            f"{shower_label}"
        )
    return {
        "permitted_generic": permitted_generic,
        "specified": specified,
        "queued": queued,
        "flushed": flushed,
        "completed_selected_losses": completed_losses,
        "completed_native_selection_replays": native_replays,
        "reason_names": reason_names,
        "reason_ids": reason_ids,
        "process_names": process_names,
        "process_ids": process_ids,
    }


def validate_gpu_integrity(
    task: BatchTask, gpu_summary: dict[str, Any]
) -> dict[str, Any]:
    totals = {
        "cpu_generic_fallbacks": 0,
        "cpu_unsupported_particle_fallbacks": 0,
        "cpu_unsupported_medium_fallbacks": 0,
        "cpu_unsupported_geometry_fallbacks": 0,
        "queue_overflows": 0,
        "native_inverse_failures": 0,
        "cpu_completed_selected_losses": 0,
        "cpu_completed_native_selection_replays": 0,
        "cpu_memory_spill_particles": 0,
        "cross_species_host_spills": 0,
        "particles_spilled_to_cpu": 0,
    }
    initialization_times: set[float] = set()
    static_upload_sizes: set[int] = set()
    for key, shower in gpu_summary.items():
        shower_index = int(key.removeprefix("shower_"))
        statistics = shower.get("statistics") if isinstance(shower, dict) else None
        if not isinstance(statistics, dict):
            raise ValueError(f"GPU shower has no statistics mapping: {key}")
        if statistics.get("gpu_physics_source") != task.source:
            raise ValueError(
                f"GPU physics source differs for {key}: expected {task.source}, "
                f"observed {statistics.get('gpu_physics_source')!r}"
            )
        lifecycle = statistics.get("backend_lifecycle")
        if not isinstance(lifecycle, dict):
            raise ValueError(f"GPU backend lifecycle is missing: {key}")
        expected_reused = shower_index > 0
        if lifecycle.get("reused") is not expected_reused:
            raise ValueError(
                f"GPU backend reuse flag differs for {key}: expected "
                f"{expected_reused}, observed {lifecycle.get('reused')!r}"
            )
        if lifecycle.get("shower_ordinal") != shower_index + 1:
            raise ValueError(
                f"GPU backend shower ordinal differs for {key}: expected "
                f"{shower_index + 1}, observed {lifecycle.get('shower_ordinal')!r}"
            )
        initialization = lifecycle.get("one_time_initialization_ms")
        static_bytes = lifecycle.get("static_host_to_device_bytes")
        if (
            isinstance(initialization, bool)
            or not isinstance(initialization, (int, float))
            or not math.isfinite(float(initialization))
            or float(initialization) < 0.0
        ):
            raise ValueError(f"invalid one-time initialization metadata for {key}")
        if (
            isinstance(static_bytes, bool)
            or not isinstance(static_bytes, int)
            or static_bytes < 0
        ):
            raise ValueError(f"invalid static upload metadata for {key}")
        initialization_times.add(float(initialization))
        static_upload_sizes.add(static_bytes)

        for total_key, path in (
            ("cpu_generic_fallbacks", ("cpu_generic_fallbacks",)),
            ("queue_overflows", ("queue_overflows",)),
            ("cpu_completed_selected_losses", ("cpu_completed_selected_losses",)),
            (
                "cpu_completed_native_selection_replays",
                ("cpu_completed_native_selection_replays",),
            ),
            ("cpu_memory_spill_particles", ("cpu_memory_spill_particles",)),
            ("cross_species_host_spills", ("cross_species", "host_spills")),
            (
                "particles_spilled_to_cpu",
                ("cross_species", "particles_spilled_to_cpu"),
            ),
        ):
            totals[total_key] += nested_counter(
                statistics, path, label=".".join(path)
            )
        fallback_accounting = validate_fallback_accounting(
            statistics, source=task.source, shower_label=key
        )
        permitted_generic = fallback_accounting["permitted_generic"]
        totals["cpu_unsupported_particle_fallbacks"] += permitted_generic[
            "unsupported_particle"
        ]
        totals["cpu_unsupported_medium_fallbacks"] += permitted_generic[
            "unsupported_medium"
        ]
        totals["cpu_unsupported_geometry_fallbacks"] += permitted_generic[
            "unsupported_geometry"
        ]
        if nested_counter(
            statistics, ("queue_overflows",), label="queue_overflows"
        ) != 0:
            raise ValueError(f"GPU queue overflow occurred in {key}")

        native_failures = 0
        if task.source == "proposal-native":
            native_failures = nested_counter(
                statistics,
                ("proposal_native", "inverse_failures"),
                label="proposal_native.inverse_failures",
            )
            completed_losses = nested_counter(
                statistics,
                ("cpu_completed_selected_losses",),
                label="cpu_completed_selected_losses",
            )
            native_replays = nested_counter(
                statistics,
                ("cpu_completed_native_selection_replays",),
                label="cpu_completed_native_selection_replays",
            )
            if native_failures > completed_losses:
                raise ValueError(
                    f"native inverse failures exceed completed selected losses in "
                    f"{key}: {native_failures} > {completed_losses}"
                )
            if native_replays > completed_losses:
                raise ValueError(
                    f"native selection replays exceed completed selected losses in "
                    f"{key}: {native_replays} > {completed_losses}"
                )
        elif nested_counter(
            statistics,
            ("cpu_completed_native_selection_replays",),
            label="cpu_completed_native_selection_replays",
        ) != 0:
            raise ValueError(
                f"c8emrt unexpectedly reports native selection replay in {key}"
            )
        totals["native_inverse_failures"] += native_failures

    if len(initialization_times) != 1 or len(static_upload_sizes) != 1:
        raise ValueError(
            "GPU one-time initialization metadata changed across showers: "
            f"times={initialization_times}, bytes={static_upload_sizes}"
        )
    totals.update(
        {
            "events_checked": len(gpu_summary),
            "backend_reused_events": max(len(gpu_summary) - 1, 0),
            "one_time_initialization_ms": next(iter(initialization_times)),
            "static_host_to_device_bytes": next(iter(static_upload_sizes)),
            "explicit_spills_are_diagnostic_only": True,
        }
    )
    return totals


def validate_simulation_output(
    args: argparse.Namespace,
    task: BatchTask,
    output: Path,
) -> dict[str, Any]:
    required = (
        output / "summary.yaml",
        output / "config.yaml",
        output / "gpu_em" / "config.yaml",
        output / "gpu_em" / "summary.yaml",
        output / "CoREAS" / "config.yaml",
        output / "CoREAS" / "summary.yaml",
        output / "ZHS" / "config.yaml",
        output / "ZHS" / "summary.yaml",
        output / "profile" / "profile.parquet",
        output / "particles" / "particles.parquet",
        output / "simulation_timing" / "summary.yaml",
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(
            "simulation output is incomplete: " + ", ".join(str(path) for path in missing)
        )
    summary = safe_load_yaml(output / "summary.yaml")
    if not isinstance(summary, dict):
        raise ValueError("top-level summary is not a mapping")
    if int(summary.get("showers", -1)) != task.events:
        raise ValueError("top-level shower count differs from requested batch")
    if int(summary.get("seed", -1)) != task.seed:
        raise ValueError("top-level seed differs from requested batch")
    gpu_summary = require_shower_mapping(
        output / "gpu_em" / "summary.yaml", task.events, label="GPU summary"
    )
    for key, shower in gpu_summary.items():
        if not isinstance(shower, dict):
            raise ValueError(f"invalid GPU shower record: {key}")
        if shower.get("complete") is not True or shower.get("status") != "complete":
            raise ValueError(f"GPU shower is incomplete: {key}")
    gpu_integrity = validate_gpu_integrity(task, gpu_summary)
    for algorithm in ("CoREAS", "ZHS"):
        require_shower_mapping(
            output / algorithm / "summary.yaml",
            task.events,
            label=f"{algorithm} summary",
        )
    timing = require_shower_mapping(
        output / "simulation_timing" / "summary.yaml",
        task.events,
        label="simulation timing summary",
    )
    for key, record in timing.items():
        if record.get("closed") is not True or record.get("status") != "closed":
            raise ValueError(f"simulation timing is not closed: {key}")
        wall_time = float(record.get("wall_time_ms", math.nan))
        if not math.isfinite(wall_time) or wall_time <= 0.0:
            raise ValueError(f"simulation timing is invalid: {key}")
    profile = validate_parquet_shower_column(
        output / "profile" / "profile.parquet",
        task.events,
        require_every_shower=True,
    )
    particles = validate_parquet_shower_column(
        output / "particles" / "particles.parquet",
        task.events,
        require_every_shower=False,
    )
    source_identity = source_identity_from_output(
        args, task, output, gpu_summary
    )
    return {
        "runtime_seconds": parse_runtime_seconds(summary),
        "source_identity": source_identity,
        "gpu_integrity": gpu_integrity,
        "profile": profile,
        "particles": particles,
    }


def provenance_payload(
    immutable: dict[str, Any],
    task: BatchTask,
    command: list[str],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    encoded_command = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "backend": "cuda",
        "gpu_physics_source": task.source,
        "ensemble_role": (
            "formal_independent" if task.phase == "formal" else "paired_diagnostic"
        ),
        "batch_index": task.index,
        "events": task.events,
        "seed": task.seed,
        "executable": immutable["artifacts"]["executable"],
        "table": source_identity,
        "antenna_file": immutable["artifacts"]["antenna_file"],
        "flupro": immutable["artifacts"]["fluka_library"],
        "runner": immutable["artifacts"]["runner"],
        "command": command,
        "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
    }


def validate_provenance(
    args: argparse.Namespace,
    path: Path,
    immutable: dict[str, Any],
    task: BatchTask,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"validation provenance is missing: {path}")
    with path.open("r", encoding="utf-8") as source:
        provenance = json.load(source)
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"provenance schema differs: {path}")
    if provenance.get("backend") != "cuda":
        raise ValueError(f"provenance backend differs: {path}")
    if provenance.get("gpu_physics_source") != task.source:
        raise ValueError(f"provenance physics source differs: {path}")
    expected_role = (
        "formal_independent" if task.phase == "formal" else "paired_diagnostic"
    )
    if provenance.get("ensemble_role") != expected_role:
        raise ValueError(f"provenance ensemble role differs: {path}")
    if int(provenance.get("batch_index", -1)) != task.index:
        raise ValueError(f"provenance batch index differs: {path}")
    if int(provenance.get("events", -1)) != task.events:
        raise ValueError(f"provenance event count differs: {path}")
    if int(provenance.get("seed", -1)) != task.seed:
        raise ValueError(f"provenance seed differs: {path}")
    if provenance.get("executable", {}).get("sha256") != immutable["artifacts"][
        "executable"
    ]["sha256"]:
        raise ValueError(f"provenance executable hash differs: {path}")
    for label in ("antenna_file", "flupro", "runner"):
        immutable_label = "fluka_library" if label == "flupro" else label
        if provenance.get(label, {}).get("sha256") != immutable["artifacts"][
            immutable_label
        ]["sha256"]:
            raise ValueError(f"provenance {label} hash differs: {path}")
    if provenance.get("table", {}).get("sha256") != source_identity.get("sha256"):
        raise ValueError(f"provenance table hash differs: {path}")
    if task.source == "proposal-native" and provenance.get("table", {}).get(
        "aux_sha256"
    ) != source_identity.get("aux_sha256"):
        raise ValueError(f"provenance aux hash differs: {path}")
    command = provenance.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError(f"provenance command is invalid: {path}")
    encoded = json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if provenance.get("command_sha256") != hashlib.sha256(encoded).hexdigest():
        raise ValueError(f"provenance command hash differs: {path}")
    if "-f" not in command or command.index("-f") + 1 >= len(command):
        raise ValueError(f"provenance command has no output option: {path}")
    recorded_output = Path(command[command.index("-f") + 1])
    expected_command = build_command(args, task, recorded_output)
    if command != expected_command:
        raise ValueError(f"provenance command differs from requested configuration: {path}")
    config = safe_load_yaml(path.parent / "config.yaml")
    if not isinstance(config, dict) or not isinstance(config.get("args"), str):
        raise ValueError(f"simulation config has no command string: {path.parent}")
    if shlex.split(config["args"]) != command:
        raise ValueError(f"simulation config and provenance commands differ: {path}")
    return provenance


def validate_completed_batch(
    args: argparse.Namespace,
    immutable: dict[str, Any],
    task: BatchTask,
    output: Path,
) -> dict[str, Any]:
    result = validate_simulation_output(args, task, output)
    validate_provenance(
        args,
        output / "validation_provenance.json",
        immutable,
        task,
        result["source_identity"],
    )
    return result


def source_identity_contract(
    manifest: dict[str, Any],
    source: str,
    identity: dict[str, Any],
) -> None:
    recorded = manifest.setdefault("source_identities", {}).get(source)
    stable = {
        key: identity.get(key)
        for key in (
            "identity_kind",
            "sha256",
            "aux_sha256",
            "serialized_payload_sha256",
            "proposal_version",
            "cubic_interpolation_version",
            "format_version",
            "node_count",
            "device_bytes",
        )
        if key in identity
    }
    if recorded is None:
        manifest["source_identities"][source] = stable
    elif recorded != stable:
        raise ValueError(
            f"{source} identity changed during campaign: {recorded} vs {stable}"
        )


def task_key(task: BatchTask) -> str:
    return f"{task.phase}:{task.index:03d}:{task.source}"


def next_attempt_number(manifest: dict[str, Any], task: BatchTask) -> int:
    key = task_key(task)
    attempt_numbers: list[int] = []
    for item in manifest.get("attempts", []):
        if item.get("task") != key:
            continue
        attempt = item.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"invalid recorded attempt number for {key}: {attempt!r}")
        attempt_numbers.append(attempt)
    return max(attempt_numbers, default=-1) + 1


def archive_failed_attempt(
    output_root: Path,
    attempt_output: Path,
    task: BatchTask,
    attempt_number: int,
) -> str | None:
    if not attempt_output.exists():
        return None
    destination_root = output_root / "failed_attempts"
    destination_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    destination = destination_root / (
        f"{task.phase}_batch{task.index:03d}_{SOURCE_DIRECTORY[task.source]}_"
        f"attempt{attempt_number:03d}_{timestamp}"
    )
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{destination.name}_{suffix}")
    shutil.move(str(attempt_output), str(destination))
    return str(destination.resolve())


def reconcile_interrupted_attempts(
    output_root: Path,
    manifest: dict[str, Any],
    task: BatchTask,
    batch_parent: Path,
) -> bool:
    """Archive stale private outputs and close interrupted manifest records.

    A SIGTERM can leave attempt 000 marked ``running`` and its private output
    directory on disk.  Counting records would then allocate attempt 001 and
    silently skip the old directory.  Reconciliation is deliberately performed
    before either canonical reuse or a new attempt: every numeric private
    directory is accounted for, and every running record becomes terminal.
    """

    key = task_key(task)
    records: dict[int, dict[str, Any]] = {}
    for record in manifest.setdefault("attempts", []):
        if record.get("task") != key:
            continue
        attempt = record.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"invalid recorded attempt number for {key}: {attempt!r}")
        if attempt in records:
            raise ValueError(f"duplicate manifest attempt number for {key}: {attempt}")
        records[attempt] = record

    prefix = f".attempt-{SOURCE_DIRECTORY[task.source]}-"
    directories: dict[int, Path] = {}
    if batch_parent.exists():
        for path in batch_parent.iterdir():
            if not path.name.startswith(prefix):
                continue
            suffix = path.name.removeprefix(prefix)
            if not suffix.isdigit():
                continue
            if not path.is_dir():
                raise ValueError(f"attempt output is not a directory: {path}")
            attempt = int(suffix)
            if attempt in directories:
                raise ValueError(f"duplicate private attempt directory: {path}")
            directories[attempt] = path

    changed = False
    reconciled_utc = utc_now()
    for attempt, record in records.items():
        if record.get("status") != "running":
            continue
        attempt_output = directories.pop(attempt, None)
        archived = (
            archive_failed_attempt(output_root, attempt_output, task, attempt)
            if attempt_output is not None
            else None
        )
        record.update(
            {
                "status": "interrupted_orphaned",
                "interrupted_status": "running",
                "finished_utc": reconciled_utc,
                "reconciled_utc": reconciled_utc,
                "archived_output": archived,
                "private_output_was_present": attempt_output is not None,
            }
        )
        changed = True

    for attempt, attempt_output in sorted(directories.items()):
        record = records.get(attempt)
        if record is not None and record.get("status") == "complete":
            raise ValueError(
                f"completed attempt unexpectedly retains private output: "
                f"{attempt_output}"
            )
        archived = archive_failed_attempt(
            output_root, attempt_output, task, attempt
        )
        if record is None:
            record = {
                "task": key,
                "attempt": attempt,
                "status": "orphaned_untracked_before_restart",
                "reconciled_utc": reconciled_utc,
                "archived_output": archived,
            }
            manifest["attempts"].append(record)
            records[attempt] = record
        else:
            record.update(
                {
                    "reconciled_utc": reconciled_utc,
                    "reconciled_archived_output": archived,
                    "orphaned_private_output_recovered": True,
                }
            )
        changed = True
    return changed


def subprocess_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "FLUPRO": str(args.flupro.resolve()),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def initial_manifest(
    immutable: dict[str, Any], expected_event_counts: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "host": platform.node(),
        "immutable_configuration": immutable,
        "expected_event_counts": expected_event_counts,
        "source_identities": {},
        "completed": {},
        "attempts": [],
    }


def summarize_completion(
    manifest: dict[str, Any], tasks: list[BatchTask], sources: Iterable[str]
) -> dict[str, Any]:
    selected = tuple(sources)
    summary: dict[str, Any] = {
        "formal": {source: 0 for source in selected},
        "paired": {source: 0 for source in selected},
    }
    completed = manifest.get("completed", {})
    for task in tasks:
        if task_key(task) in completed:
            summary[task.phase][task.source] += task.events
    return summary


def expected_completion(args: argparse.Namespace) -> dict[str, Any]:
    sources = tuple(args.sources)
    return {
        "formal": {source: args.events_per_source for source in sources},
        "paired": {source: args.paired_events for source in sources},
    }


def summarize_gpu_integrity(manifest: dict[str, Any]) -> dict[str, int]:
    keys = (
        "events_checked",
        "cpu_generic_fallbacks",
        "cpu_unsupported_particle_fallbacks",
        "cpu_unsupported_medium_fallbacks",
        "cpu_unsupported_geometry_fallbacks",
        "queue_overflows",
        "native_inverse_failures",
        "cpu_completed_selected_losses",
        "cpu_completed_native_selection_replays",
        "cpu_memory_spill_particles",
        "cross_species_host_spills",
        "particles_spilled_to_cpu",
        "backend_reused_events",
    )
    totals = {key: 0 for key in keys}
    for completed in manifest.get("completed", {}).values():
        integrity = completed.get("gpu_integrity", {})
        for key in keys:
            totals[key] += int(integrity.get(key, 0))
    return totals


def print_dry_run(
    args: argparse.Namespace,
    immutable: dict[str, Any],
    tasks: list[BatchTask],
) -> None:
    root = args.output_root.expanduser().resolve()
    payload = {
        "immutable_configuration": immutable,
        "expected_event_counts": expected_completion(args),
        "task_count": len(tasks),
        "tasks": [
            {
                "phase": task.phase,
                "source": task.source,
                "batch": task.index,
                "events": task.events,
                "seed": task.seed,
                "canonical_output": str(canonical_output(root, task)),
                "command": build_command(
                    args,
                    task,
                    canonical_output(root, task).with_name(
                        f".dry-run-{SOURCE_DIRECTORY[task.source]}"
                    ),
                ),
            }
            for task in tasks
        ],
    }
    print(json.dumps(payload, indent=2, allow_nan=False))


def main() -> int:
    args = parse_args()
    args.sources = tuple(args.sources)
    if args.native_aux_cache_dir is not None:
        args.native_aux_cache_dir = args.native_aux_cache_dir.expanduser()
    args.output_root = args.output_root.expanduser()
    validate_arguments(args)
    immutable = immutable_configuration(args)
    tasks = campaign_tasks(args)
    expected_counts = expected_completion(args)
    if args.dry_run:
        print_dry_run(args, immutable, tasks)
        return 0

    output_root = args.output_root.resolve()
    manifest_path = output_root / "campaign_manifest.json"
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output root is not a directory: {output_root}")
    if output_root.exists() and not manifest_path.is_file():
        unexpected = [
            path for path in output_root.iterdir() if path.name != ".runner.lock"
        ]
        if unexpected:
            raise ValueError(
                "existing output root has no campaign manifest; refusing to overwrite: "
                + str(output_root)
            )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign is already running: {output_root}") from error
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "utc": utc_now()}))
        lock.flush()

        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as source:
                manifest = json.load(source)
            if manifest.get("immutable_configuration") != immutable:
                raise ValueError(
                    "campaign immutable configuration changed; refusing unsafe resume"
                )
            if manifest.get("status") == "complete":
                for task in tasks:
                    result = validate_completed_batch(
                        args, immutable, task, canonical_output(output_root, task)
                    )
                    source_identity_contract(
                        manifest, task.source, result["source_identity"]
                    )
                    completed_record = manifest.get("completed", {}).get(
                        task_key(task)
                    )
                    if not isinstance(completed_record, dict):
                        raise ValueError(
                            f"completed manifest is missing {task_key(task)}"
                        )
                    completed_record["gpu_integrity"] = result["gpu_integrity"]
                observed_counts = summarize_completion(
                    manifest, tasks, args.sources
                )
                if observed_counts != expected_counts:
                    raise ValueError(
                        f"completed campaign counts differ: "
                        f"{observed_counts} vs {expected_counts}"
                    )
                manifest["gpu_integrity_totals"] = summarize_gpu_integrity(manifest)
                manifest["updated_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                print("campaign already complete and all batches revalidated")
                return 0
            manifest["status"] = "running"
        else:
            manifest = initial_manifest(immutable, expected_counts)
            atomic_write_json(manifest_path, manifest)

        for task_position, task in enumerate(tasks, start=1):
            key = task_key(task)
            final_output = canonical_output(output_root, task)
            batch_parent = final_output.parent
            batch_parent.mkdir(parents=True, exist_ok=True)
            if reconcile_interrupted_attempts(
                output_root, manifest, task, batch_parent
            ):
                manifest["updated_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
            if final_output.exists():
                result = validate_completed_batch(
                    args, immutable, task, final_output
                )
                source_identity_contract(
                    manifest, task.source, result["source_identity"]
                )
                manifest.setdefault("completed", {})[key] = {
                    "status": "complete",
                    "phase": task.phase,
                    "source": task.source,
                    "batch_index": task.index,
                    "events": task.events,
                    "seed": task.seed,
                    "output": str(final_output),
                    "runtime_seconds": result["runtime_seconds"],
                    "gpu_integrity": result["gpu_integrity"],
                    "revalidated_utc": utc_now(),
                }
                manifest["updated_utc"] = utc_now()
                manifest["event_counts"] = summarize_completion(
                    manifest, tasks, args.sources
                )
                manifest["gpu_integrity_totals"] = summarize_gpu_integrity(manifest)
                atomic_write_json(manifest_path, manifest)
                print(f"reuse {key}: {task.events} events", flush=True)
                continue

            attempt_number = next_attempt_number(manifest, task)
            attempt_output = batch_parent / (
                f".attempt-{SOURCE_DIRECTORY[task.source]}-{attempt_number:03d}"
            )
            if attempt_output.exists():
                raise RuntimeError(
                    "attempt reconciliation failed to clear private output: "
                    f"{attempt_output}"
                )
            command = build_command(args, task, attempt_output)
            log_path = batch_parent / (
                f"{SOURCE_DIRECTORY[task.source]}.attempt-{attempt_number:03d}.log"
            )
            print(
                f"[{task_position}/{len(tasks)}] start {key}: "
                f"events={task.events} seed={task.seed}",
                flush=True,
            )
            attempt_record = {
                "task": key,
                "attempt": attempt_number,
                "status": "running",
                "started_utc": utc_now(),
                "command": command,
                "log": str(log_path.resolve()),
            }
            manifest.setdefault("attempts", []).append(attempt_record)
            manifest["updated_utc"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            try:
                with log_path.open("xb") as log:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=subprocess_environment(args),
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"simulation exited with status {completed.returncode}; "
                        f"see {log_path}"
                    )
                result = validate_simulation_output(
                    args, task, attempt_output
                )
                provenance = provenance_payload(
                    immutable, task, command, result["source_identity"]
                )
                atomic_write_json(
                    attempt_output / "validation_provenance.json", provenance
                )
                validate_provenance(
                    args,
                    attempt_output / "validation_provenance.json",
                    immutable,
                    task,
                    result["source_identity"],
                )
                source_identity_contract(
                    manifest, task.source, result["source_identity"]
                )
                if final_output.exists():
                    raise RuntimeError(
                        f"canonical output appeared during attempt: {final_output}"
                    )
                os.replace(attempt_output, final_output)
                attempt_record.update(
                    {
                        "status": "complete",
                        "finished_utc": utc_now(),
                        "output": str(final_output.resolve()),
                        "runtime_seconds": result["runtime_seconds"],
                    }
                )
                manifest.setdefault("completed", {})[key] = {
                    "status": "complete",
                    "phase": task.phase,
                    "source": task.source,
                    "batch_index": task.index,
                    "events": task.events,
                    "seed": task.seed,
                    "output": str(final_output.resolve()),
                    "runtime_seconds": result["runtime_seconds"],
                    "gpu_integrity": result["gpu_integrity"],
                    "completed_utc": utc_now(),
                }
                manifest["event_counts"] = summarize_completion(
                    manifest, tasks, args.sources
                )
                manifest["gpu_integrity_totals"] = summarize_gpu_integrity(manifest)
                manifest["updated_utc"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                print(f"complete {key}: {result['runtime_seconds']:.3f} s", flush=True)
            except BaseException as error:
                archived = archive_failed_attempt(
                    output_root, attempt_output, task, attempt_number
                )
                attempt_record.update(
                    {
                        "status": "failed",
                        "finished_utc": utc_now(),
                        "error": f"{type(error).__name__}: {error}",
                        "archived_output": archived,
                    }
                )
                manifest["status"] = "failed_retriable"
                manifest["updated_utc"] = utc_now()
                manifest["event_counts"] = summarize_completion(
                    manifest, tasks, args.sources
                )
                atomic_write_json(manifest_path, manifest)
                raise

        manifest["event_counts"] = summarize_completion(
            manifest, tasks, args.sources
        )
        manifest["gpu_integrity_totals"] = summarize_gpu_integrity(manifest)
        if manifest["event_counts"] != expected_counts:
            raise RuntimeError(
                f"campaign counts differ: {manifest['event_counts']} vs {expected_counts}"
            )
        manifest["status"] = "complete"
        manifest["updated_utc"] = utc_now()
        manifest["completed_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(manifest["event_counts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
