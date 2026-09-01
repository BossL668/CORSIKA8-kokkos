#!/usr/bin/env python3
"""Benchmark C8EMRT and native-PROPOSAL CUDA physics sources fairly.

The measured schedule is an ABBA block.  Every block contains two paired
processes: A1/B1 use one seed and B2/A2 use the next seed.  Each process runs
several showers so the first shower can be treated as a warm-up while the
remaining showers measure the resident-backend steady state.

Headline timings deliberately leave ``--gpu-detailed-stage-timing`` disabled.
CUDA event totals are retained as diagnostics but are never added to wall time.

Every subprocess writes to an immutable attempt directory.  A locked,
hash-pinned campaign can be resumed with ``--resume``; completed attempts are
revalidated before their timings are reused.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


SCHEMA_VERSION = 2
SOURCES = ("c8emrt", "proposal-native")
SOURCE_LABELS = {
    "c8emrt": "C8EMRT",
    "proposal-native": "PROPOSAL native",
}
SOURCE_COLORS = {
    "c8emrt": "#1f77b4",
    "proposal-native": "#d62728",
}
SHOWER_KEY = re.compile(r"^shower_(\d+)$")


# ``--extra-arg`` is intentionally available for options outside the benchmark
# contract (for example a hadronic-model choice).  It must never be usable to
# append a second value for an option already owned by this runner: CLI11's
# duplicate-option semantics are not a suitable physics-configuration gate.
CONTROLLED_LONG_OPTIONS = frozenset(
    {
        "--pdg",
        "--energy",
        "--energy_range",
        "--eslope",
        "--zenith",
        "--azimuth",
        "--nevent",
        "--filename",
        "--seed",
        "--geomagnetic-model",
        "--geomagnetic-year",
        "--antenna-file",
        "--ring",
        "--shower-core-x",
        "--shower-core-y",
        "--emcut",
        "--hadcut",
        "--mucut",
        "--taucut",
        "--emthin",
        "--max-weight",
        "--max-deflection-angle",
        "--em-backend",
        "--radio-backend",
        "--gpu-device",
        "--gpu-min-batch",
        "--gpu-memory-fraction",
        "--gpu-table-cache",
        "--gpu-physics-source",
        "--gpu-aux-cache-dir",
        "--gpu-table-tolerance",
        "--gpu-deterministic",
        "--gpu-detailed-stage-timing",
        "--gpu-radio-field-limit",
        "--radio-sampling-rate-ghz",
        "--radio-window-duration-ns",
        "--radio-pretrigger-ns",
        "--verbosity",
    }
)
CONTROLLED_SHORT_OPTIONS = frozenset(
    {"-Z", "-A", "-p", "-E", "-N", "-z", "-a", "-s", "-f", "-v"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--native-aux-cache", type=Path, required=True)
    parser.add_argument("--antenna-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--flupro", type=Path)

    parser.add_argument("--energy-gev", type=float, default=1.0e5)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--events-per-process", type=int, default=6)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of complete A-B-B-A blocks; at least five are required.",
    )
    parser.add_argument("--seed", type=int, default=2026083001)
    parser.add_argument("--seed-stride", type=int, default=1000)

    parser.add_argument("--em-cut-gev", type=float, default=0.0005)
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-6)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.0,
        help=(
            "Explicit EM thinning maximum weight. Zero omits --max-weight "
            "and preserves c8_air_shower's automatic value."
        ),
    )
    parser.add_argument("--max-deflection-angle", type=float, default=0.2)
    parser.add_argument("--shower-core-x-m", type=float, default=0.0)
    parser.add_argument("--shower-core-y-m", type=float, default=0.0)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument(
        "--geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        default="IGRF14",
    )
    parser.add_argument("--geomagnetic-year", type=float, default=2027.0)

    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--gpu-radio-field-limit", type=float, default=1.0)
    parser.add_argument("--radio-sampling-rate-ghz", type=float, default=1.0)
    parser.add_argument("--radio-window-duration-ns", type=float, default=400.0)
    parser.add_argument("--radio-pretrigger-ns", type=float, default=10.0)

    parser.add_argument(
        "--warmup-energy-gev",
        type=float,
        default=10.0,
        help=(
            "Energy of the one-shower cache warm-up. The warm-up uses a "
            "photon primary but the same medium, cuts and backend settings."
        ),
    )
    parser.add_argument("--warmup-seed", type=int, default=2026083999)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    parser.add_argument(
        "--maximum-native-regression-fraction",
        type=float,
        default=0.03,
        help=(
            "Maximum accepted native/C8EMRT steady-state median slowdown. "
            "The default implements the three-percent production gate."
        ),
    )
    parser.add_argument(
        "--allow-native-inverse-fallbacks",
        action="store_true",
        help=(
            "Allow explicitly completed native inverse-CDF CPU fallbacks. "
            "They remain counted and reported; all generic fallbacks remain fatal."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Per-process timeout; zero disables the timeout.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional common c8_air_shower argument; repeat as needed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a matching interrupted benchmark. Completed attempts are "
            "strictly revalidated and reused; an incomplete task receives a new "
            "attempt directory."
        ),
    )
    parser.add_argument("--verbosity", default="warn")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def finite_positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")


def controlled_extra_option(argument: str) -> str | None:
    """Return the script-owned option hidden in an extra argument, if any."""

    stripped = argument.strip()
    if not stripped:
        raise ValueError("--extra-arg may not be empty")
    # A whitespace-containing value would be forwarded as one argv element and
    # is invalid for c8_air_shower. Splitting here still gives a useful error if
    # it was an attempted ``--extra-arg='--max-weight 100'`` bypass.
    first = stripped.split(None, 1)[0]
    option = first.split("=", 1)[0]
    if option in CONTROLLED_LONG_OPTIONS or option in CONTROLLED_SHORT_OPTIONS:
        return option
    if first.startswith("--"):
        return None
    for short in CONTROLLED_SHORT_OPTIONS:
        if first.startswith(short) and len(first) > len(short):
            return short
    return None


def validate_extra_args(extra_args: Iterable[str]) -> None:
    for argument in extra_args:
        option = controlled_extra_option(argument)
        if option is not None:
            raise ValueError(
                f"--extra-arg may not override controlled option {option}"
            )


def validate_args(args: argparse.Namespace) -> None:
    if not args.executable.is_file() or not os.access(args.executable, os.X_OK):
        raise ValueError(f"executable is not an executable file: {args.executable}")
    if not args.table.is_file():
        raise ValueError(f"C8EMRT table is not a regular file: {args.table}")
    if not args.antenna_file.is_file():
        raise ValueError(f"antenna file is not a regular file: {args.antenna_file}")
    if args.flupro is not None and not args.flupro.is_dir():
        raise ValueError(f"FLUPRO is not a directory: {args.flupro}")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise ValueError(f"output root is not a directory: {args.output_root}")
    if args.output_root.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite output root: {args.output_root}")
    if args.events_per_process < 6:
        raise ValueError("--events-per-process must be at least 6")
    if args.rounds < 5:
        raise ValueError("--rounds must be at least 5 complete ABBA blocks")
    if args.seed <= 0 or args.seed_stride <= 0 or args.warmup_seed <= 0:
        raise ValueError("seeds and seed stride must be positive")
    if args.primary_pdg == 0:
        raise ValueError("primary PDG must be non-zero")
    for value, label in (
        (args.energy_gev, "energy"),
        (args.warmup_energy_gev, "warm-up energy"),
        (args.em_cut_gev, "EM cut"),
        (args.had_cut_gev, "hadron cut"),
        (args.mu_cut_gev, "muon cut"),
        (args.tau_cut_gev, "tau cut"),
        (args.max_deflection_angle, "maximum deflection angle"),
        (args.gpu_memory_fraction, "GPU memory fraction"),
        (args.gpu_table_tolerance, "GPU table tolerance"),
        (args.gpu_radio_field_limit, "GPU radio field limit"),
        (args.radio_sampling_rate_ghz, "radio sampling rate"),
        (args.radio_window_duration_ns, "radio window duration"),
        (args.radio_pretrigger_ns, "radio pretrigger"),
    ):
        finite_positive(value, label)
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("EM thinning must be in [0, 1]")
    if args.maximum_weight < 0.0 or not math.isfinite(args.maximum_weight):
        raise ValueError("maximum weight must be zero or positive and finite")
    if not 0.0 <= args.zenith_deg <= 90.0:
        raise ValueError("zenith must be in [0, 90] degrees")
    if not 0.0 <= args.azimuth_deg <= 360.0:
        raise ValueError("azimuth must be in [0, 360] degrees")
    if not 1900.0 <= args.geomagnetic_year <= 2030.0:
        raise ValueError("geomagnetic year must be in [1900, 2030]")
    if args.gpu_device < 0 or args.gpu_min_batch <= 0:
        raise ValueError("GPU device and minimum batch are invalid")
    if not 0.01 <= args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must be in [0.01, 1]")
    if args.ring < 0:
        raise ValueError("ring count must be non-negative")
    if args.bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if args.maximum_native_regression_fraction < 0.0:
        raise ValueError("maximum native regression fraction must be non-negative")
    if args.timeout_seconds < 0.0:
        raise ValueError("timeout must be zero or positive")

    validate_extra_args(args.extra_arg)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def verify_artifact_identity(expected: dict[str, Any], label: str) -> None:
    observed = artifact_identity(Path(expected["path"]))
    if (
        observed["size_bytes"] != expected["size_bytes"]
        or observed["sha256"] != expected["sha256"]
    ):
        raise RuntimeError(f"{label} changed while the benchmark was running")


def environment_for_run(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if args.flupro is not None:
        environment["FLUPRO"] = str(args.flupro.resolve())
    return environment


def common_command(
    args: argparse.Namespace,
    output: Path,
    seed: int,
    events: int,
    primary_pdg: int | None = None,
    energy_gev: float | None = None,
) -> list[str]:
    command = [
        str(args.executable.resolve()),
        "-p",
        str(args.primary_pdg if primary_pdg is None else primary_pdg),
        "-E",
        f"{args.energy_gev if energy_gev is None else energy_gev:.17g}",
        "-N",
        str(events),
        "-z",
        f"{args.zenith_deg:.17g}",
        "-a",
        f"{args.azimuth_deg:.17g}",
        "-s",
        str(seed),
        "-f",
        str(output.resolve()),
        "--geomagnetic-model",
        args.geomagnetic_model,
        "--geomagnetic-year",
        f"{args.geomagnetic_year:.17g}",
        "--antenna-file",
        str(args.antenna_file.resolve()),
        "--ring",
        str(args.ring),
        "--shower-core-x",
        f"{args.shower_core_x_m:.17g}",
        "--shower-core-y",
        f"{args.shower_core_y_m:.17g}",
        "--emcut",
        f"{args.em_cut_gev:.17g}",
        "--hadcut",
        f"{args.had_cut_gev:.17g}",
        "--mucut",
        f"{args.mu_cut_gev:.17g}",
        "--taucut",
        f"{args.tau_cut_gev:.17g}",
        "--emthin",
        f"{args.em_thinning:.17g}",
        "--max-deflection-angle",
        f"{args.max_deflection_angle:.17g}",
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
        "--gpu-table-tolerance",
        f"{args.gpu_table_tolerance:.17g}",
        "--gpu-radio-field-limit",
        f"{args.gpu_radio_field_limit:.17g}",
        "--radio-sampling-rate-ghz",
        f"{args.radio_sampling_rate_ghz:.17g}",
        "--radio-window-duration-ns",
        f"{args.radio_window_duration_ns:.17g}",
        "--radio-pretrigger-ns",
        f"{args.radio_pretrigger_ns:.17g}",
        "--verbosity",
        args.verbosity,
    ]
    if args.maximum_weight > 0.0:
        command.extend(("--max-weight", f"{args.maximum_weight:.17g}"))
    command.extend(args.extra_arg)
    return command


def source_command(
    args: argparse.Namespace,
    source: str,
    output: Path,
    seed: int,
    events: int,
    primary_pdg: int | None = None,
    energy_gev: float | None = None,
) -> list[str]:
    command = common_command(
        args, output, seed, events, primary_pdg=primary_pdg,
        energy_gev=energy_gev,
    )
    command.extend(("--gpu-physics-source", source))
    if source == "c8emrt":
        command.extend(("--gpu-table-cache", str(args.table.resolve())))
    elif source == "proposal-native":
        command.extend(
            ("--gpu-aux-cache-dir", str(args.native_aux_cache.resolve()))
        )
    else:
        raise ValueError(f"unsupported source: {source}")
    return command


def run_command(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
                timeout=None if timeout_seconds == 0.0 else timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"process exceeded {timeout_seconds:g} s; see {log_path}"
            ) from error
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}: {shlex.join(command)}; "
            f"see {log_path}"
        )
    return elapsed


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required output is missing: {path}")
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a YAML mapping: {path}")
    return value


def ordered_shower_records(
    value: dict[str, Any], expected_events: int, path: Path
) -> list[dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for key, record in value.items():
        match = SHOWER_KEY.fullmatch(str(key))
        if match is None or not isinstance(record, dict):
            raise RuntimeError(f"invalid shower record {key!r} in {path}")
        index = int(match.group(1))
        if index in indexed:
            raise RuntimeError(f"duplicate shower index {index} in {path}")
        indexed[index] = record
    expected = set(range(expected_events))
    if set(indexed) != expected:
        raise RuntimeError(
            f"shower indices in {path} are {sorted(indexed)}, expected "
            f"0..{expected_events - 1}"
        )
    return [indexed[index] for index in range(expected_events)]


def nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def require_zero(
    errors: list[str], statistics_node: dict[str, Any], path: tuple[str, ...]
) -> None:
    value = nested(statistics_node, *path)
    if value != 0:
        errors.append(f"{'.'.join(path)}={value!r}, expected 0")


def nonnegative_counter(
    errors: list[str],
    statistics_node: dict[str, Any],
    path: tuple[str, ...],
    *,
    default: int | None = None,
) -> int:
    value = nested(statistics_node, *path, default=default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(
            f"{'.'.join(path)}={value!r}, expected a non-negative integer"
        )
        return 0
    return value


def normalized_counter_map(
    errors: list[str], value: Any, label: str, *, allow_empty_null: bool
) -> dict[str, int]:
    if value is None and allow_empty_null:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}={value!r}, expected a counter mapping")
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(
                f"{label}.{key}={count!r}, expected a non-negative integer"
            )
            continue
        result[str(key)] = count
    return result


def validate_requested_thinning(
    errors: list[str],
    statistics_node: dict[str, Any],
    requested_em_thinning: float,
    configured_maximum_weight: float,
    primary_energy_gev: float,
) -> dict[str, Any]:
    thinning = statistics_node.get("thinning")
    if not isinstance(thinning, dict):
        errors.append("thinning mapping is missing")
        return {}

    automatic = configured_maximum_weight <= 0.0
    expected_maximum_weight = (
        0.5 * requested_em_thinning * primary_energy_gev
        if automatic
        else configured_maximum_weight
    )
    expected_activation = (
        requested_em_thinning > 0.0 and expected_maximum_weight > 1.0
    )
    for key, expected in (
        ("em_fraction", requested_em_thinning),
        ("maximum_weight", expected_maximum_weight),
    ):
        observed = thinning.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), float(expected), rel_tol=5.0e-14, abs_tol=1.0e-15
            )
        ):
            errors.append(f"thinning.{key}={observed!r}, expected {expected:.17g}")
    if thinning.get("automatic_maximum_weight") is not automatic:
        errors.append(
            "thinning.automatic_maximum_weight="
            f"{thinning.get('automatic_maximum_weight')!r}, expected {automatic}"
        )
    if thinning.get("can_activate_from_unit_weight") is not expected_activation:
        errors.append(
            "thinning.can_activate_from_unit_weight="
            f"{thinning.get('can_activate_from_unit_weight')!r}, "
            f"expected {expected_activation}"
        )
    return {
        "em_fraction": thinning.get("em_fraction"),
        "maximum_weight": thinning.get("maximum_weight"),
        "automatic_maximum_weight": thinning.get("automatic_maximum_weight"),
        "can_activate_from_unit_weight": thinning.get(
            "can_activate_from_unit_weight"
        ),
    }


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
    errors: list[str], statistics_node: dict[str, Any]
) -> dict[str, int]:
    generic = nonnegative_counter(
        errors, statistics_node, ("cpu_generic_fallbacks",)
    )
    reasons = statistics_node.get("cpu_fallbacks_by_reason_name")
    reasons = normalized_counter_map(
        errors,
        reasons,
        "cpu_fallbacks_by_reason_name",
        allow_empty_null=generic == 0,
    )
    permitted: dict[str, int] = {}
    for reason in PERMITTED_GENERIC_FALLBACK_REASONS:
        value = reasons.get(reason, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"cpu_fallbacks_by_reason_name.{reason}={value!r}, "
                "expected non-negative int"
            )
            return permitted
        permitted[reason] = value
    if generic != sum(permitted.values()):
        errors.append(
            "generic CPU fallbacks are not exactly the declared scalar-routing "
            f"cases: generic={generic}, permitted={permitted}"
        )
    return permitted


def validate_gpu_record(
    record: dict[str, Any],
    source: str,
    shower_index: int,
    allow_native_inverse_fallbacks: bool,
    requested_em_thinning: float,
    configured_maximum_weight: float,
    primary_energy_gev: float,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    if record.get("complete") is not True or record.get("status") != "complete":
        errors.append(
            f"completion state is complete={record.get('complete')!r}, "
            f"status={record.get('status')!r}"
        )
    statistics_node = record.get("statistics")
    if not isinstance(statistics_node, dict):
        return {}, {}, errors + ["statistics mapping is missing"]
    if statistics_node.get("gpu_physics_source") != source:
        errors.append(
            f"gpu_physics_source={statistics_node.get('gpu_physics_source')!r}"
        )
    if statistics_node.get("gpu_muon_transport_enabled") is not True:
        errors.append(
            "gpu_muon_transport_enabled is not true; the two physics sources "
            "would not have equal particle coverage"
        )
    lifecycle = statistics_node.get("backend_lifecycle")
    if not isinstance(lifecycle, dict):
        errors.append("backend_lifecycle mapping is missing")
    else:
        expected_reused = shower_index > 0
        if lifecycle.get("reused") is not expected_reused:
            errors.append(
                f"backend_lifecycle.reused={lifecycle.get('reused')!r}, "
                f"expected {expected_reused}"
            )
        if lifecycle.get("shower_ordinal") != shower_index + 1:
            errors.append(
                "backend_lifecycle.shower_ordinal="
                f"{lifecycle.get('shower_ordinal')!r}, expected {shower_index + 1}"
            )

    registry = statistics_node.get("process_registry")
    if not isinstance(registry, dict) or registry.get("accepted") is not True:
        errors.append("process_registry was not accepted")
    if isinstance(registry, dict):
        for key, value in registry.items():
            if key.startswith("unregistered_") and value != 0:
                errors.append(f"process_registry.{key}={value!r}, expected 0")

    for path in (
        ("queue_overflows",),
        ("cpu_memory_spill_particles",),
        ("cross_species", "host_spills"),
        ("cross_species", "particles_spilled_to_cpu"),
        ("cross_species", "final_pending_photons"),
        ("cross_species", "final_pending_leptons"),
        ("profile", "fixed_point_overflows"),
        ("profile", "invalid_records"),
        ("radio", "fixed_point_overflows"),
        ("epair_sampler", "cpu_fallbacks"),
        ("epair_sampler", "envelope_violations"),
    ):
        require_zero(errors, statistics_node, path)
    thinning_diagnostics = validate_requested_thinning(
        errors,
        statistics_node,
        requested_em_thinning,
        configured_maximum_weight,
        primary_energy_gev,
    )
    permitted_generic = validate_permitted_generic_fallbacks(
        errors, statistics_node
    )

    queued = nonnegative_counter(
        errors, statistics_node, ("deferred_cpu_fallbacks_queued",)
    )
    flushed = nonnegative_counter(
        errors, statistics_node, ("deferred_cpu_fallbacks_flushed",)
    )
    if queued != flushed:
        errors.append(
            f"deferred fallbacks queued/flushed mismatch: {queued!r}/{flushed!r}"
        )
    fallback_flushes = nonnegative_counter(
        errors, statistics_node, ("deferred_cpu_fallback_flushes",)
    )
    fallback_scalar_rounds = nonnegative_counter(
        errors, statistics_node, ("deferred_fallback_scalar_expansion_rounds",)
    )
    maximum_fallback_batch = nonnegative_counter(
        errors, statistics_node, ("maximum_deferred_cpu_fallback_batch",)
    )
    fallback_steps = nonnegative_counter(
        errors, statistics_node, ("cpu_fallback_steps_executed",)
    )
    fallback_time = statistics_node.get("cpu_specified_fallback_time_ms")
    if (
        isinstance(fallback_time, bool)
        or not isinstance(fallback_time, (int, float))
        or not math.isfinite(float(fallback_time))
        or float(fallback_time) < 0.0
    ):
        errors.append(
            "cpu_specified_fallback_time_ms="
            f"{fallback_time!r}, expected a finite non-negative number"
        )
        fallback_time = 0.0

    inverse_failures = nonnegative_counter(
        errors,
        statistics_node,
        ("proposal_native", "inverse_failures"),
        default=0,
    )
    completed_losses = nonnegative_counter(
        errors, statistics_node, ("cpu_completed_selected_losses",)
    )
    native_replays = nonnegative_counter(
        errors, statistics_node, ("cpu_completed_native_selection_replays",)
    )
    generic_fallbacks = nonnegative_counter(
        errors, statistics_node, ("cpu_generic_fallbacks",)
    )
    specified_final_states = nonnegative_counter(
        errors, statistics_node, ("cpu_specified_final_states",)
    )
    if queued != specified_final_states or flushed != specified_final_states:
        errors.append(
            "specified/deferred CPU fallback accounting differs: "
            f"specified={specified_final_states}, queued={queued}, flushed={flushed}"
        )
    total_fallbacks = generic_fallbacks + specified_final_states
    reason_names = normalized_counter_map(
        errors,
        statistics_node.get("cpu_fallbacks_by_reason_name"),
        "cpu_fallbacks_by_reason_name",
        allow_empty_null=total_fallbacks == 0,
    )
    reason_ids = normalized_counter_map(
        errors,
        statistics_node.get("cpu_fallbacks_by_reason"),
        "cpu_fallbacks_by_reason",
        allow_empty_null=total_fallbacks == 0,
    )
    process_names = normalized_counter_map(
        errors,
        statistics_node.get("cpu_fallbacks_by_process_name"),
        "cpu_fallbacks_by_process_name",
        allow_empty_null=total_fallbacks == 0,
    )
    process_ids = normalized_counter_map(
        errors,
        statistics_node.get("cpu_fallbacks_by_process"),
        "cpu_fallbacks_by_process",
        allow_empty_null=total_fallbacks == 0,
    )
    for label, mapping in (
        ("reason-name", reason_names),
        ("reason-id", reason_ids),
        ("process-name", process_names),
        ("process-id", process_ids),
    ):
        if sum(mapping.values()) != total_fallbacks:
            errors.append(
                f"{label} fallback map does not close: "
                f"sum={sum(mapping.values())}, expected={total_fallbacks}"
            )
    known_reasons = set(PERMITTED_GENERIC_FALLBACK_REASONS) | set(
        PERMITTED_SPECIFIED_FALLBACK_REASONS
    )
    unknown_reasons = sorted(set(reason_names) - known_reasons)
    if unknown_reasons:
        errors.append(f"unpermitted CPU fallback reasons: {unknown_reasons}")
    specified_from_reasons = sum(
        reason_names.get(reason, 0)
        for reason in PERMITTED_SPECIFIED_FALLBACK_REASONS
    )
    if specified_from_reasons != specified_final_states:
        errors.append(
            "specified CPU fallback reasons do not close: "
            f"reason_total={specified_from_reasons}, "
            f"specified={specified_final_states}"
        )
    selected_from_reasons = sum(
        reason_names.get(reason, 0) for reason in SELECTED_LOSS_FALLBACK_REASONS
    )
    if completed_losses != selected_from_reasons:
        errors.append(
            "selected-loss fallback accounting differs: "
            f"completed={completed_losses}, reasons={selected_from_reasons}"
        )
    replay_reasons = reason_names.get("native_selection_replay", 0)
    if replay_reasons != native_replays:
        errors.append(
            "native-selection replay counter/reason mismatch: "
            f"counter={native_replays}, reason={replay_reasons}"
        )
    if source == "proposal-native":
        if not bool(
            nested(
                statistics_node,
                "proposal_native",
                "proposal_cache_all_hit",
                default=False,
            )
        ):
            errors.append("proposal_native.proposal_cache_all_hit is not true")
        if not bool(
            nested(
                statistics_node,
                "proposal_native",
                "aux_cache_hit",
                default=False,
            )
        ):
            errors.append("proposal_native.aux_cache_hit is not true")
        if inverse_failures and not allow_native_inverse_fallbacks:
            errors.append(
                f"proposal_native.inverse_failures={inverse_failures}; "
                "pass --allow-native-inverse-fallbacks to accept completed fallbacks"
            )
        if inverse_failures > completed_losses:
            errors.append(
                f"native inverse failures ({inverse_failures}) exceed completed "
                f"selected losses ({completed_losses})"
            )
        if native_replays > completed_losses:
            errors.append(
                f"native selection replays ({native_replays}) exceed completed "
                f"selected losses ({completed_losses})"
            )
    else:
        if inverse_failures != 0:
            errors.append(
                f"C8EMRT unexpectedly reports {inverse_failures} native failures"
            )
        if native_replays != 0:
            errors.append(
                f"C8EMRT unexpectedly reports {native_replays} native replays"
            )

    if nested(
        statistics_node,
        "transfer_timing",
        "device_event_timing_enabled",
        default=False,
    ) is not False:
        errors.append("headline run unexpectedly enabled detailed CUDA event timing")

    if nested(statistics_node, "energy_ledger", "complete_coverage") is True and nested(
        statistics_node, "energy_ledger", "accepted"
    ) is not True:
        errors.append("complete energy ledger did not pass acceptance")
    diagnostics = {
        "thinning": thinning_diagnostics,
        "cpu_generic_fallbacks": generic_fallbacks,
        "cpu_specified_final_states": specified_final_states,
        "cpu_permitted_generic_fallbacks": permitted_generic,
        "deferred_cpu_fallbacks_queued": queued,
        "deferred_cpu_fallbacks_flushed": flushed,
        "deferred_cpu_fallback_flushes": fallback_flushes,
        "deferred_fallback_scalar_expansion_rounds": fallback_scalar_rounds,
        "maximum_deferred_cpu_fallback_batch": maximum_fallback_batch,
        "cpu_fallback_steps_executed": fallback_steps,
        "cpu_specified_fallback_time_ms": float(fallback_time),
        "cpu_completed_selected_losses": completed_losses,
        "cpu_completed_native_selection_replays": native_replays,
        "native_inverse_failures": inverse_failures,
        "cpu_fallbacks_by_reason": reason_ids,
        "cpu_fallbacks_by_reason_name": reason_names,
        "cpu_fallbacks_by_process": process_ids,
        "cpu_fallbacks_by_process_name": process_names,
    }
    return statistics_node, diagnostics, errors


def read_and_validate_output(
    output: Path,
    source: str,
    expected_events: int,
    allow_native_inverse_fallbacks: bool,
    require_warm_cache: bool,
    requested_em_thinning: float,
    configured_maximum_weight: float,
    primary_energy_gev: float,
) -> dict[str, Any]:
    timing_path = output / "simulation_timing" / "summary.yaml"
    gpu_path = output / "gpu_em" / "summary.yaml"
    timing_records = ordered_shower_records(
        load_yaml_mapping(timing_path), expected_events, timing_path
    )
    gpu_records = ordered_shower_records(
        load_yaml_mapping(gpu_path), expected_events, gpu_path
    )
    config = load_yaml_mapping(output / "gpu_em" / "config.yaml")

    shower_rows: list[dict[str, Any]] = []
    gate_errors: list[str] = []
    for index, (timing, gpu) in enumerate(zip(timing_records, gpu_records)):
        if timing.get("closed") is not True or timing.get("status") != "closed":
            gate_errors.append(
                f"shower_{index}: timing is closed={timing.get('closed')!r}, "
                f"status={timing.get('status')!r}"
            )
        wall_time_ms = timing.get("wall_time_ms")
        if not isinstance(wall_time_ms, (int, float)) or not math.isfinite(
            float(wall_time_ms)
        ) or float(wall_time_ms) <= 0.0:
            gate_errors.append(f"shower_{index}: invalid wall_time_ms={wall_time_ms!r}")
            wall_time_ms = math.nan

        statistics_node, diagnostics, errors = validate_gpu_record(
            gpu,
            source,
            index,
            allow_native_inverse_fallbacks,
            requested_em_thinning,
            configured_maximum_weight,
            primary_energy_gev,
        )
        if not require_warm_cache and source == "proposal-native":
            errors = [
                error
                for error in errors
                if error
                not in {
                    "proposal_native.proposal_cache_all_hit is not true",
                    "proposal_native.aux_cache_hit is not true",
                }
            ]
        gate_errors.extend(f"shower_{index}: {error}" for error in errors)
        lifecycle = statistics_node.get("backend_lifecycle", {})
        shower_rows.append(
            {
                "shower_index": index,
                "wall_time_ms": float(wall_time_ms),
                "reused": lifecycle.get("reused"),
                "shower_ordinal": lifecycle.get("shower_ordinal"),
                "one_time_initialization_ms": lifecycle.get(
                    "one_time_initialization_ms"
                ),
                "static_host_to_device_bytes": lifecycle.get(
                    "static_host_to_device_bytes"
                ),
                "gpu_particles": statistics_node.get("gpu_particles"),
                "cpu_particle_steps": statistics_node.get("cpu_particle_steps"),
                "wavefronts": statistics_node.get("wavefronts"),
                "resident_photon_wavefronts": statistics_node.get(
                    "resident_photon_wavefronts"
                ),
                "resident_lepton_wavefronts": statistics_node.get(
                    "resident_lepton_wavefronts"
                ),
                "maximum_input_batch": statistics_node.get("maximum_input_batch"),
                "kernel_time_ms": statistics_node.get("kernel_time_ms"),
                "photon_backend_wall_time_ms": statistics_node.get(
                    "photon_backend_wall_time_ms"
                ),
                "lepton_backend_wall_time_ms": statistics_node.get(
                    "lepton_backend_wall_time_ms"
                ),
                "router_advance_time_ms": nested(
                    statistics_node, "hybrid_timing_ms", "router_advance"
                ),
                "output_end_time_ms": nested(
                    statistics_node, "hybrid_timing_ms", "output_end"
                ),
                "transfer_host_api_time_ms": nested(
                    statistics_node, "transfer_timing", "host_api_time_ms"
                ),
                "physical_pipeline_wait_time_ms": nested(
                    statistics_node,
                    "synchronization_timing",
                    "physical_pipeline_wait_time_ms",
                ),
                "radio_device_time_ms": nested(
                    statistics_node, "radio", "device_time_ms"
                ),
                "radio_direct_projection_batches": nested(
                    statistics_node, "radio", "direct_projection_batches"
                ),
                "profile_kernel_time_ms": nested(
                    statistics_node, "profile", "kernel_time_ms"
                ),
                "table_device_bytes": statistics_node.get("table_device_bytes"),
                "peak_device_bytes": statistics_node.get("peak_device_bytes"),
                "workspace_bytes": statistics_node.get("workspace_bytes"),
                "native_newton_iterations": nested(
                    statistics_node, "proposal_native", "newton_iterations", default=0
                ),
                "native_bisection_iterations": nested(
                    statistics_node,
                    "proposal_native",
                    "bisection_iterations",
                    default=0,
                ),
                "native_inverse_failures": diagnostics.get(
                    "native_inverse_failures", 0
                ),
                "native_table_sha256": nested(
                    statistics_node, "proposal_native", "table_sha256"
                ),
                "native_aux_sha256": nested(
                    statistics_node, "proposal_native", "aux_sha256"
                ),
                "cpu_completed_selected_losses": diagnostics.get(
                    "cpu_completed_selected_losses", 0
                ),
                "cpu_completed_native_selection_replays": diagnostics.get(
                    "cpu_completed_native_selection_replays", 0
                ),
                "cpu_generic_fallbacks": diagnostics.get(
                    "cpu_generic_fallbacks", 0
                ),
                "cpu_specified_final_states": diagnostics.get(
                    "cpu_specified_final_states", 0
                ),
                "cpu_unsupported_particle_fallbacks": diagnostics.get(
                    "cpu_permitted_generic_fallbacks", {}
                ).get("unsupported_particle", 0),
                "cpu_unsupported_medium_fallbacks": diagnostics.get(
                    "cpu_permitted_generic_fallbacks", {}
                ).get("unsupported_medium", 0),
                "cpu_unsupported_geometry_fallbacks": diagnostics.get(
                    "cpu_permitted_generic_fallbacks", {}
                ).get("unsupported_geometry", 0),
                "deferred_cpu_fallbacks_queued": diagnostics.get(
                    "deferred_cpu_fallbacks_queued", 0
                ),
                "deferred_cpu_fallbacks_flushed": diagnostics.get(
                    "deferred_cpu_fallbacks_flushed", 0
                ),
                "deferred_cpu_fallback_flushes": diagnostics.get(
                    "deferred_cpu_fallback_flushes", 0
                ),
                "deferred_fallback_scalar_expansion_rounds": diagnostics.get(
                    "deferred_fallback_scalar_expansion_rounds", 0
                ),
                "maximum_deferred_cpu_fallback_batch": diagnostics.get(
                    "maximum_deferred_cpu_fallback_batch", 0
                ),
                "cpu_fallback_steps_executed": diagnostics.get(
                    "cpu_fallback_steps_executed", 0
                ),
                "cpu_specified_fallback_time_ms": diagnostics.get(
                    "cpu_specified_fallback_time_ms", 0.0
                ),
                "cpu_fallbacks_by_reason_json": json.dumps(
                    diagnostics.get("cpu_fallbacks_by_reason", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "cpu_fallbacks_by_reason_name_json": json.dumps(
                    diagnostics.get("cpu_fallbacks_by_reason_name", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "cpu_fallbacks_by_process_json": json.dumps(
                    diagnostics.get("cpu_fallbacks_by_process", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "cpu_fallbacks_by_process_name_json": json.dumps(
                    diagnostics.get("cpu_fallbacks_by_process_name", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "thinning_em_fraction": diagnostics.get("thinning", {}).get(
                    "em_fraction"
                ),
                "thinning_maximum_weight": diagnostics.get("thinning", {}).get(
                    "maximum_weight"
                ),
                "thinning_automatic_maximum_weight": diagnostics.get(
                    "thinning", {}
                ).get("automatic_maximum_weight"),
                "radio_tracks": statistics_node.get("radio_tracks"),
                "profile_steps": nested(statistics_node, "profile", "steps"),
            }
        )
    if source == "proposal-native":
        table_hashes = {row["native_table_sha256"] for row in shower_rows}
        aux_hashes = {row["native_aux_sha256"] for row in shower_rows}
        if len(table_hashes) != 1 or None in table_hashes:
            gate_errors.append(
                f"native table hash changed within one process: {table_hashes}"
            )
        if len(aux_hashes) != 1 or None in aux_hashes:
            gate_errors.append(
                f"native auxiliary hash changed within one process: {aux_hashes}"
            )
    else:
        c8_hash = nested(config, "table", "sha256")
        if not isinstance(c8_hash, str) or len(c8_hash) != 64:
            gate_errors.append(f"invalid C8EMRT semantic table hash: {c8_hash!r}")
    if gate_errors:
        raise RuntimeError(
            f"strict output gate failed for {output}:\n  - "
            + "\n  - ".join(gate_errors)
        )
    return {
        "config": config,
        "showers": shower_rows,
    }


def normalized_gpu_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    normalized.pop("gpu_physics_source", None)
    normalized.pop("table", None)
    return normalized


def describe(values: Iterable[float]) -> dict[str, Any]:
    sample = [float(value) for value in values]
    if not sample or any(not math.isfinite(value) for value in sample):
        raise ValueError("cannot summarize empty or non-finite sample")
    return {
        "count": len(sample),
        "mean": statistics.fmean(sample),
        "median": statistics.median(sample),
        "minimum": min(sample),
        "maximum": max(sample),
        "standard_deviation": statistics.stdev(sample) if len(sample) > 1 else 0.0,
        "q05": float(np.quantile(sample, 0.05)),
        "q95": float(np.quantile(sample, 0.95)),
    }


def bootstrap_median_interval(
    values: list[float], repetitions: int, seed: int
) -> list[float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=np.float64)
    chunk = max(1, min(repetitions, 1000))
    offset = 0
    while offset < repetitions:
        count = min(chunk, repetitions - offset)
        indices = rng.integers(0, sample.size, size=(count, sample.size))
        medians[offset : offset + count] = np.median(sample[indices], axis=1)
        offset += count
    return [
        float(np.quantile(medians, 0.025)),
        float(np.median(sample)),
        float(np.quantile(medians, 0.975)),
    ]


def build_schedule(args: argparse.Namespace) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    run_index = 0
    for round_index in range(args.rounds):
        first_pair = 2 * round_index
        entries = (
            ("c8emrt", first_pair, "A1"),
            ("proposal-native", first_pair, "B1"),
            ("proposal-native", first_pair + 1, "B2"),
            ("c8emrt", first_pair + 1, "A2"),
        )
        for position, (source, pair_id, abba_position) in enumerate(entries, 1):
            schedule.append(
                {
                    "run_index": run_index,
                    "round": round_index + 1,
                    "position": position,
                    "abba_position": abba_position,
                    "source": source,
                    "pair_id": pair_id,
                    "seed": args.seed + pair_id * args.seed_stride,
                }
            )
            run_index += 1
    return schedule


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_counter_json(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        raw = row.get(key, "{}")
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            raise ValueError(f"cannot aggregate non-mapping {key}={value!r}")
        for name, count in value.items():
            totals[str(name)] = totals.get(str(name), 0) + int(count)
    return dict(sorted(totals.items()))


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON manifest without exposing a partial document."""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"cannot read JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping: {path}")
    return value


def acquire_campaign_lock(output_root: Path):
    """Acquire a non-blocking process lock held for the runner's lifetime."""

    path = output_root / ".performance_runner.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"another performance runner holds the campaign lock: {path}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\nstarted_utc={utc_now()}\n")
    handle.flush()
    return handle


def campaign_contract(
    args: argparse.Namespace,
    schedule: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Return every immutable input needed to safely resume this benchmark."""

    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "execution_host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "flupro": str(args.flupro.resolve()) if args.flupro is not None else None,
        },
        "configuration": {
            "primary_pdg": args.primary_pdg,
            "energy_gev": args.energy_gev,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "events_per_process": args.events_per_process,
            "rounds": args.rounds,
            "base_seed": args.seed,
            "seed_stride": args.seed_stride,
            "em_cut_gev": args.em_cut_gev,
            "had_cut_gev": args.had_cut_gev,
            "mu_cut_gev": args.mu_cut_gev,
            "tau_cut_gev": args.tau_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "max_deflection_angle": args.max_deflection_angle,
            "shower_core_x_m": args.shower_core_x_m,
            "shower_core_y_m": args.shower_core_y_m,
            "ring": args.ring,
            "geomagnetic_model": args.geomagnetic_model,
            "geomagnetic_year": args.geomagnetic_year,
            "gpu_device": args.gpu_device,
            "gpu_min_batch": args.gpu_min_batch,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "gpu_table_tolerance": args.gpu_table_tolerance,
            "gpu_radio_field_limit": args.gpu_radio_field_limit,
            "radio_sampling_rate_ghz": args.radio_sampling_rate_ghz,
            "radio_window_duration_ns": args.radio_window_duration_ns,
            "radio_pretrigger_ns": args.radio_pretrigger_ns,
            "warmup_energy_gev": args.warmup_energy_gev,
            "warmup_seed": args.warmup_seed,
            "allow_native_inverse_fallbacks": args.allow_native_inverse_fallbacks,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "maximum_native_regression_fraction": (
                args.maximum_native_regression_fraction
            ),
            "native_aux_cache": str(args.native_aux_cache.resolve()),
            "extra_args": list(args.extra_arg),
            "verbosity": args.verbosity,
            "schedule": schedule,
        },
    }


ATTEMPT_KEY = re.compile(r"^attempt_(\d+)$")


def run_or_reuse_attempt(
    task_root: Path,
    command_for_output,
    environment: dict[str, str],
    timeout_seconds: float,
    validate_output,
    *,
    resume: bool,
) -> tuple[float, dict[str, Any], Path, bool, list[str]]:
    """Run one task in an immutable attempt directory or reuse a valid result."""

    task_root.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[int, Path]] = []
    for child in task_root.iterdir():
        match = ATTEMPT_KEY.fullmatch(child.name)
        if child.is_dir() and match is not None:
            attempts.append((int(match.group(1)), child))
    attempts.sort()

    if resume:
        for _, attempt in reversed(attempts):
            state_path = attempt / "attempt.json"
            if not state_path.is_file():
                continue
            state = load_json_mapping(state_path)
            if state.get("status") != "complete":
                continue
            command = command_for_output(attempt / "output")
            if state.get("command") != command:
                raise RuntimeError(
                    f"completed attempt command differs from campaign contract: "
                    f"{state_path}"
                )
            elapsed = state.get("external_wall_seconds")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) <= 0.0
            ):
                raise RuntimeError(f"invalid completed attempt timing: {state_path}")
            parsed = validate_output(attempt / "output")
            return float(elapsed), parsed, attempt, True, command

    next_index = attempts[-1][0] + 1 if attempts else 1
    attempt = task_root / f"attempt_{next_index:03d}"
    attempt.mkdir()
    command = command_for_output(attempt / "output")
    state_path = attempt / "attempt.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": utc_now(),
        "command": command,
    }
    atomic_write_json(state_path, state)
    try:
        elapsed = run_command(
            command,
            attempt / "run.log",
            environment,
            timeout_seconds,
        )
        parsed = validate_output(attempt / "output")
    except BaseException as error:
        state.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_write_json(state_path, state)
        raise
    state.update(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "external_wall_seconds": elapsed,
        }
    )
    atomic_write_json(state_path, state)
    return elapsed, parsed, attempt, False, command


def write_progress(
    output_root: Path, completed_tasks: list[dict[str, Any]], total_tasks: int
) -> None:
    atomic_write_json(
        output_root / "progress.json",
        {
            "schema_version": 1,
            "updated_utc": utc_now(),
            "completed_count": len(completed_tasks),
            "total_count": total_tasks,
            "completed_tasks": completed_tasks,
        },
    )


def make_plot(
    process_rows: list[dict[str, Any]],
    shower_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    output: Path,
    regression_limit: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    steady_by_source = {
        source: [
            row["wall_time_ms"] / 1000.0
            for row in shower_rows
            if row["source"] == source and row["shower_index"] > 0
        ]
        for source in SOURCES
    }
    common_bins = np.histogram_bin_edges(
        steady_by_source["c8emrt"] + steady_by_source["proposal-native"],
        bins="auto",
    )
    for source in SOURCES:
        axes[0].hist(
            steady_by_source[source],
            bins=common_bins,
            alpha=0.50,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
        )
    axes[0].set_xlabel("steady-state shower wall time [s]")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False)

    positions = np.arange(len(SOURCES), dtype=np.float64)
    samples = [
        [
            row["external_wall_seconds"]
            for row in process_rows
            if row["source"] == source
        ]
        for source in SOURCES
    ]
    violin = axes[1].violinplot(samples, positions=positions, showmedians=True)
    for body, source in zip(violin["bodies"], SOURCES):
        body.set_facecolor(SOURCE_COLORS[source])
        body.set_alpha(0.55)
    axes[1].set_xticks(positions, [SOURCE_LABELS[source] for source in SOURCES])
    axes[1].set_ylabel("whole-process wall time [s]")

    pair_ids = [row["pair_id"] for row in paired_rows]
    ratios = [row["steady_sum_native_over_c8emrt"] for row in paired_rows]
    axes[2].plot(pair_ids, ratios, "o-", color="#6a3d9a", linewidth=1.2)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(
        1.0 + regression_limit,
        color="#d62728",
        linestyle=":",
        linewidth=1.2,
        label=f"gate = {1.0 + regression_limit:.3f}",
    )
    axes[2].set_xlabel("paired seed cohort")
    axes[2].set_ylabel("native / C8EMRT steady sum")
    axes[2].legend(frameon=False)
    fig.suptitle("CUDA physics-source runtime comparison (ABBA schedule)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_report(summary: dict[str, Any]) -> str:
    source_stats = summary["timing_statistics"]
    ratio = summary["relative_performance"]
    acceptance = summary["acceptance"]
    config = summary["configuration"]
    fallback = summary["fallback_statistics"]
    lines = [
        "# PROPOSAL-native 与 C8EMRT GPU 性能验收",
        "",
        f"生成时间：`{summary['created_utc']}`。",
        "",
        "## 配置与计时口径",
        "",
        f"- 初级粒子 PDG：`{config['primary_pdg']}`；能量："
        f"`{config['energy_gev']:.8g} GeV`；方向："
        f"`theta={config['zenith_deg']:.8g} deg, phi={config['azimuth_deg']:.8g} deg`。",
        f"- 每个进程 `{config['events_per_process']}` 个 shower，共 "
        f"`{config['rounds']}` 个完整 `A-B-B-A` 轮次。",
        f"- `gpu-min-batch={config['gpu_min_batch']}`，"
        f"`gpu-memory-fraction={config['gpu_memory_fraction']:.3f}`，"
        "CUDA EM 和 CUDA radio 均开启。",
        "- 每个进程的第 0 个 shower 只用于进程内预热，不计入 headline "
        "steady-state 分布。`wall_time_ms` 来自共同的 OutputManager "
        "start/end 边界。",
        "- 完整进程时间由 Python `perf_counter()` 测量，包含进程启动、"
        "物理表初始化、设备上传和输出收尾。详细 CUDA stage timing 在 headline "
        "运行中关闭。",
        "",
        "## 运行时间",
        "",
        "| 物理源 | 稳态 shower 数 | 稳态中位数 [s] | 稳态均值 [s] | "
        "完整进程中位数 [s] |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        steady = source_stats[source]["steady_shower_wall_seconds"]
        external = source_stats[source]["external_process_wall_seconds"]
        lines.append(
            f"| {SOURCE_LABELS[source]} | {steady['count']} | "
            f"{steady['median']:.6g} | {steady['mean']:.6g} | "
            f"{external['median']:.6g} |"
        )
    interval = ratio["paired_steady_sum_ratio_bootstrap_95pct"]
    lines.extend(
        [
            "",
            "## 相对性能",
            "",
            f"- 两个来源全部稳态 shower 的中位时间比 "
            f"`native/C8EMRT = {ratio['ratio_of_steady_shower_medians']:.6f}`。",
            f"- 按配对进程 steady-state 总时间计算的中位比为 "
            f"`{interval[1]:.6f}`，paired bootstrap 95% 区间为 "
            f"`[{interval[0]:.6f}, {interval[2]:.6f}]`。",
            f"- 完整进程中位时间比为 "
            f"`{ratio['ratio_of_external_process_medians']:.6f}`。该指标包含"
            "初始化，但不等同于新介质的离线制表成本。",
            "",
            "## 验收结论",
            "",
            f"- 完整性门禁：`{'PASS' if acceptance['completeness_passed'] else 'FAIL'}`。",
            f"- 稳态性能门禁（native 不得比 C8EMRT 慢超过 "
            f"`{100.0 * acceptance['maximum_native_regression_fraction']:.1f}%`）："
            f"`{'PASS' if acceptance['performance_passed'] else 'FAIL'}`。",
            f"- 总结论：`{'PASS' if acceptance['passed'] else 'FAIL'}`。",
            "",
            "## CPU fallback 审计",
            "",
            "| 物理源 | generic | specified | selected-loss | native replay | "
            "native inverse failure | fallback time [ms] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in SOURCES:
        item = fallback[source]
        lines.append(
            f"| {SOURCE_LABELS[source]} | {item['cpu_generic_fallbacks']} | "
            f"{item['cpu_specified_final_states']} | "
            f"{item['cpu_completed_selected_losses']} | "
            f"{item['cpu_completed_native_selection_replays']} | "
            f"{item['native_inverse_failures']} | "
            f"{item['cpu_specified_fallback_time_ms']:.6g} |"
        )
    lines.extend(
        [
            "",
            "逐 shower 的 fallback process/reason 映射保存在 "
            "`shower_timings.csv`；汇总映射保存在 `benchmark_summary.json`。",
            "",
            "## 输出文件",
            "",
            "- `benchmark_summary.json`：完整配置、provenance、逐源统计和验收结果；",
            "- `process_runs.csv`：每个 ABBA 进程的外部及 shower 汇总时间；",
            "- `shower_timings.csv`：逐 shower 时间、复用标志和 GPU 工作量；",
            "- `paired_ratios.csv`：每个成对 seed cohort 的 native/C8EMRT 比值；",
            "- `runtime_distributions.png`：稳态 shower、进程时间和配对比值图。",
            "- `campaign_manifest.json`、`progress.json` 与各 `attempt_NNN/attempt.json`："
            "可恢复执行的不可变配置、原子进度和尝试状态。",
            "",
            "`kernel_time_ms` 是可能跨 stream 重叠的 CUDA event 时长之和，"
            "只用于诊断，没有与 wall time 相加。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    # Keep this handle alive until main returns. The OS also releases the lock
    # on every exception/termination path handled by this process.
    campaign_lock = acquire_campaign_lock(args.output_root)
    args.native_aux_cache.mkdir(parents=True, exist_ok=True)
    environment = environment_for_run(args)
    schedule = build_schedule(args)

    artifacts = {
        "executable": artifact_identity(args.executable),
        "c8emrt_table": artifact_identity(args.table),
        "antenna_file": artifact_identity(args.antenna_file),
    }
    contract = campaign_contract(args, schedule, artifacts)
    contract_path = args.output_root / "campaign_manifest.json"
    if contract_path.exists():
        if not args.resume:
            raise RuntimeError(f"campaign manifest already exists: {contract_path}")
        observed_contract = load_json_mapping(contract_path)
        if observed_contract != contract:
            raise RuntimeError(
                "resume contract differs from the existing benchmark; use a new "
                f"output root instead of mixing configurations: {contract_path}"
            )
    else:
        unexpected = [
            path.name
            for path in args.output_root.iterdir()
            if path.name != ".performance_runner.lock"
        ]
        if args.resume and unexpected:
            raise RuntimeError(
                "cannot resume an output root without campaign_manifest.json; "
                f"unexpected entries: {sorted(unexpected)}"
            )
        atomic_write_json(contract_path, contract)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "artifacts": artifacts,
        "configuration": {
            "primary_pdg": args.primary_pdg,
            "energy_gev": args.energy_gev,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "events_per_process": args.events_per_process,
            "rounds": args.rounds,
            "schedule": "A-B-B-A; A=c8emrt, B=proposal-native",
            "base_seed": args.seed,
            "seed_stride": args.seed_stride,
            "em_cut_gev": args.em_cut_gev,
            "had_cut_gev": args.had_cut_gev,
            "mu_cut_gev": args.mu_cut_gev,
            "tau_cut_gev": args.tau_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "gpu_device": args.gpu_device,
            "gpu_min_batch": args.gpu_min_batch,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "gpu_table_tolerance": args.gpu_table_tolerance,
            "radio_backend": "cuda",
            "detailed_stage_timing": False,
            "native_aux_cache": str(args.native_aux_cache.resolve()),
        },
        "warmups": {},
        "schedule": schedule,
    }
    completed_tasks: list[dict[str, Any]] = []
    if not args.resume or not (args.output_root / "progress.json").is_file():
        write_progress(args.output_root, completed_tasks, 2 + len(schedule))

    warmup_root = args.output_root / "warmup"
    warmup_root.mkdir(exist_ok=args.resume)
    normalized_reference: dict[str, Any] | None = None
    for source_index, source in enumerate(SOURCES):
        command_for_output = (
            lambda output, source=source, source_index=source_index: source_command(
                args,
                source,
                output,
                args.warmup_seed + source_index,
                1,
                primary_pdg=22,
                energy_gev=args.warmup_energy_gev,
            )
        )
        validate_output = lambda output, source=source: read_and_validate_output(
            output,
            source,
            1,
            args.allow_native_inverse_fallbacks,
            require_warm_cache=False,
            requested_em_thinning=args.em_thinning,
            configured_maximum_weight=args.maximum_weight,
            primary_energy_gev=args.warmup_energy_gev,
        )
        elapsed, parsed, attempt, reused, command = run_or_reuse_attempt(
            warmup_root / source,
            command_for_output,
            environment,
            args.timeout_seconds,
            validate_output,
            resume=args.resume,
        )
        print(
            f"[warmup] {source}: {'reused' if reused else 'completed'} "
            f"{attempt.name}: {shlex.join(command)}",
            flush=True,
        )
        manifest["warmups"][source] = {
            "external_wall_seconds": elapsed,
            "command": command,
            "attempt": str(attempt),
            "reused": reused,
            "gpu_config": parsed["config"],
        }
        normalized = normalized_gpu_config(parsed["config"])
        if normalized_reference is None:
            normalized_reference = normalized
        elif normalized != normalized_reference:
            raise RuntimeError(
                "common gpu_em configuration differs between warm-up sources; "
                "only the physics-source table block may differ"
            )
        completed_tasks.append(
            {
                "task": f"warmup:{source}",
                "attempt": str(attempt),
                "reused": reused,
            }
        )
        write_progress(args.output_root, completed_tasks, 2 + len(schedule))

    process_rows: list[dict[str, Any]] = []
    shower_rows: list[dict[str, Any]] = []
    runs_root = args.output_root / "runs"
    runs_root.mkdir(exist_ok=args.resume)
    for item in schedule:
        run_id = (
            f"run_{item['run_index']:03d}_round_{item['round']:02d}_"
            f"{item['abba_position']}_{item['source']}_seed_{item['seed']}"
        )
        run_dir = runs_root / run_id
        print(
            f"[{item['run_index'] + 1:02d}/{len(schedule):02d}] "
            f"round={item['round']} {item['abba_position']} "
            f"source={item['source']} seed={item['seed']}",
            flush=True,
        )
        command_for_output = lambda output, item=item: source_command(
            args,
            item["source"],
            output,
            item["seed"],
            args.events_per_process,
        )
        validate_output = lambda output, item=item: read_and_validate_output(
            output,
            item["source"],
            args.events_per_process,
            args.allow_native_inverse_fallbacks,
            require_warm_cache=True,
            requested_em_thinning=args.em_thinning,
            configured_maximum_weight=args.maximum_weight,
            primary_energy_gev=args.energy_gev,
        )
        elapsed, parsed, attempt, reused, command = run_or_reuse_attempt(
            run_dir,
            command_for_output,
            environment,
            args.timeout_seconds,
            validate_output,
            resume=args.resume,
        )
        output = attempt / "output"
        normalized = normalized_gpu_config(parsed["config"])
        if normalized_reference is None:
            normalized_reference = normalized
        elif normalized != normalized_reference:
            raise RuntimeError(
                f"common gpu_em configuration changed in {run_id}; only the "
                "physics-source table block may differ"
            )

        parsed_showers = parsed["showers"]
        walls = [row["wall_time_ms"] for row in parsed_showers]
        steady = walls[1:]
        process_row = {
            **item,
            "run_id": run_id,
            "output": str(output),
            "attempt": str(attempt),
            "reused": reused,
            "command": shlex.join(command),
            "external_wall_seconds": elapsed,
            "all_shower_sum_ms": sum(walls),
            "first_shower_wall_ms": walls[0],
            "steady_shower_count": len(steady),
            "steady_shower_sum_ms": sum(steady),
            "steady_shower_mean_ms": statistics.fmean(steady),
            "steady_shower_median_ms": statistics.median(steady),
            "outside_shower_wall_ms": elapsed * 1000.0 - sum(walls),
            "one_time_initialization_ms": parsed["showers"][0][
                "one_time_initialization_ms"
            ],
            "static_host_to_device_bytes": parsed["showers"][0][
                "static_host_to_device_bytes"
            ],
            "table_device_bytes": parsed["showers"][0]["table_device_bytes"],
            "cpu_generic_fallbacks": sum(
                row["cpu_generic_fallbacks"] for row in parsed_showers
            ),
            "cpu_specified_final_states": sum(
                row["cpu_specified_final_states"] for row in parsed_showers
            ),
            "cpu_completed_selected_losses": sum(
                row["cpu_completed_selected_losses"] for row in parsed_showers
            ),
            "cpu_completed_native_selection_replays": sum(
                row["cpu_completed_native_selection_replays"]
                for row in parsed_showers
            ),
            "native_inverse_failures": sum(
                row["native_inverse_failures"] for row in parsed_showers
            ),
            "deferred_cpu_fallbacks_queued": sum(
                row["deferred_cpu_fallbacks_queued"] for row in parsed_showers
            ),
            "cpu_specified_fallback_time_ms": sum(
                row["cpu_specified_fallback_time_ms"] for row in parsed_showers
            ),
            "cpu_fallbacks_by_reason_name_json": json.dumps(
                aggregate_counter_json(
                    parsed_showers, "cpu_fallbacks_by_reason_name_json"
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "cpu_fallbacks_by_process_name_json": json.dumps(
                aggregate_counter_json(
                    parsed_showers, "cpu_fallbacks_by_process_name_json"
                ),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        process_rows.append(process_row)
        for row in parsed_showers:
            shower_rows.append(
                {
                    "run_id": run_id,
                    "run_index": item["run_index"],
                    "round": item["round"],
                    "position": item["position"],
                    "pair_id": item["pair_id"],
                    "source": item["source"],
                    "process_seed": item["seed"],
                    **row,
                }
            )
        completed_tasks.append(
            {"task": run_id, "attempt": str(attempt), "reused": reused}
        )
        write_progress(args.output_root, completed_tasks, 2 + len(schedule))

    verify_artifact_identity(manifest["artifacts"]["executable"], "executable")
    verify_artifact_identity(manifest["artifacts"]["c8emrt_table"], "C8EMRT table")
    verify_artifact_identity(manifest["artifacts"]["antenna_file"], "antenna file")
    native_table_hashes = {
        row["native_table_sha256"]
        for row in shower_rows
        if row["source"] == "proposal-native"
    }
    native_aux_hashes = {
        row["native_aux_sha256"]
        for row in shower_rows
        if row["source"] == "proposal-native"
    }
    if len(native_table_hashes) != 1 or len(native_aux_hashes) != 1:
        raise RuntimeError(
            "native table or auxiliary hash changed between measured processes"
        )
    manifest["artifacts"]["proposal_native_table_sha256"] = next(
        iter(native_table_hashes)
    )
    manifest["artifacts"]["proposal_native_aux_sha256"] = next(
        iter(native_aux_hashes)
    )

    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in process_rows:
        pair = pairs.setdefault(int(row["pair_id"]), {})
        source = str(row["source"])
        if source in pair:
            raise RuntimeError(f"duplicate {source} run for pair {row['pair_id']}")
        pair[source] = row
    paired_rows: list[dict[str, Any]] = []
    for pair_id in sorted(pairs):
        pair = pairs[pair_id]
        if set(pair) != set(SOURCES):
            raise RuntimeError(f"incomplete source pairing for pair {pair_id}: {pair}")
        c8emrt = pair["c8emrt"]
        native = pair["proposal-native"]
        if c8emrt["seed"] != native["seed"]:
            raise RuntimeError(f"seed mismatch in pair {pair_id}")
        paired_rows.append(
            {
                "pair_id": pair_id,
                "seed": c8emrt["seed"],
                "c8emrt_run_id": c8emrt["run_id"],
                "proposal_native_run_id": native["run_id"],
                "external_native_over_c8emrt": native["external_wall_seconds"]
                / c8emrt["external_wall_seconds"],
                "all_shower_sum_native_over_c8emrt": native["all_shower_sum_ms"]
                / c8emrt["all_shower_sum_ms"],
                "steady_sum_native_over_c8emrt": native["steady_shower_sum_ms"]
                / c8emrt["steady_shower_sum_ms"],
                "steady_median_native_over_c8emrt": native[
                    "steady_shower_median_ms"
                ]
                / c8emrt["steady_shower_median_ms"],
            }
        )

    timing_statistics: dict[str, Any] = {}
    fallback_statistics: dict[str, Any] = {}
    for source in SOURCES:
        source_processes = [row for row in process_rows if row["source"] == source]
        source_showers = [row for row in shower_rows if row["source"] == source]
        source_steady_showers = [
            row["wall_time_ms"] / 1000.0
            for row in source_showers
            if row["shower_index"] > 0
        ]
        timing_statistics[source] = {
            "external_process_wall_seconds": describe(
                row["external_wall_seconds"] for row in source_processes
            ),
            "first_shower_wall_seconds": describe(
                row["first_shower_wall_ms"] / 1000.0 for row in source_processes
            ),
            "steady_shower_wall_seconds": describe(source_steady_showers),
            "outside_shower_wall_seconds": describe(
                row["outside_shower_wall_ms"] / 1000.0 for row in source_processes
            ),
            "one_time_initialization_seconds": describe(
                row["one_time_initialization_ms"] / 1000.0
                for row in source_processes
            ),
            "table_device_bytes": describe(
                row["table_device_bytes"] for row in source_processes
            ),
        }
        fallback_statistics[source] = {
            "showers": len(source_showers),
            "cpu_generic_fallbacks": sum(
                row["cpu_generic_fallbacks"] for row in source_showers
            ),
            "cpu_specified_final_states": sum(
                row["cpu_specified_final_states"] for row in source_showers
            ),
            "cpu_completed_selected_losses": sum(
                row["cpu_completed_selected_losses"] for row in source_showers
            ),
            "cpu_completed_native_selection_replays": sum(
                row["cpu_completed_native_selection_replays"]
                for row in source_showers
            ),
            "native_inverse_failures": sum(
                row["native_inverse_failures"] for row in source_showers
            ),
            "deferred_cpu_fallbacks_queued": sum(
                row["deferred_cpu_fallbacks_queued"] for row in source_showers
            ),
            "deferred_cpu_fallbacks_flushed": sum(
                row["deferred_cpu_fallbacks_flushed"] for row in source_showers
            ),
            "cpu_specified_fallback_time_ms": sum(
                row["cpu_specified_fallback_time_ms"] for row in source_showers
            ),
            "by_reason_name": aggregate_counter_json(
                source_showers, "cpu_fallbacks_by_reason_name_json"
            ),
            "by_process_name": aggregate_counter_json(
                source_showers, "cpu_fallbacks_by_process_name_json"
            ),
        }

    paired_steady_ratios = [
        row["steady_sum_native_over_c8emrt"] for row in paired_rows
    ]
    c8_steady_median = timing_statistics["c8emrt"][
        "steady_shower_wall_seconds"
    ]["median"]
    native_steady_median = timing_statistics["proposal-native"][
        "steady_shower_wall_seconds"
    ]["median"]
    c8_external_median = timing_statistics["c8emrt"][
        "external_process_wall_seconds"
    ]["median"]
    native_external_median = timing_statistics["proposal-native"][
        "external_process_wall_seconds"
    ]["median"]
    relative_performance = {
        "ratio_definition": "proposal-native / c8emrt; smaller is faster",
        "ratio_of_steady_shower_medians": native_steady_median / c8_steady_median,
        "ratio_of_external_process_medians": native_external_median
        / c8_external_median,
        "paired_steady_sum_ratio": describe(paired_steady_ratios),
        "paired_steady_sum_ratio_bootstrap_95pct": bootstrap_median_interval(
            paired_steady_ratios,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        ),
    }
    performance_passed = (
        relative_performance["ratio_of_steady_shower_medians"]
        <= 1.0 + args.maximum_native_regression_fraction
    )
    acceptance = {
        "completeness_passed": True,
        "maximum_native_regression_fraction": args.maximum_native_regression_fraction,
        "performance_metric": "ratio_of_steady_shower_medians",
        "performance_passed": performance_passed,
        "passed": performance_passed,
    }
    manifest.update(
        {
            "completed_utc": utc_now(),
            "timing_statistics": timing_statistics,
            "fallback_statistics": fallback_statistics,
            "relative_performance": relative_performance,
            "acceptance": acceptance,
            "process_runs": process_rows,
            "paired_runs": paired_rows,
        }
    )

    write_csv(args.output_root / "process_runs.csv", process_rows)
    write_csv(args.output_root / "shower_timings.csv", shower_rows)
    write_csv(args.output_root / "paired_ratios.csv", paired_rows)
    make_plot(
        process_rows,
        shower_rows,
        paired_rows,
        args.output_root / "runtime_distributions.png",
        args.maximum_native_regression_fraction,
    )
    atomic_write_json(args.output_root / "benchmark_summary.json", manifest)
    (args.output_root / "PERFORMANCE_REPORT_CN.md").write_text(
        markdown_report(manifest), encoding="utf-8"
    )
    print(
        f"acceptance={'PASS' if acceptance['passed'] else 'FAIL'}; "
        "native/C8EMRT steady median="
        f"{relative_performance['ratio_of_steady_shower_medians']:.6f}; "
        f"output={args.output_root}",
        flush=True,
    )
    campaign_lock.close()
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
