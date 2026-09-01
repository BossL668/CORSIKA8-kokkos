#!/usr/bin/env python3
"""Compare per-shower wall times for C8EMRT and PROPOSAL-native CUDA.

The tool accepts repeated, explicit shard directories.  It deliberately does
not infer shards by recursively walking a campaign: doing so can silently count
an archived or interrupted attempt twice.  The first shower in every process
is reported separately from the steady-state sample because it can contain
first-shower allocator, kernel and output warm-up.  CORSIKA initializes and
uploads the physics source before this OutputManager timing interval; process
startup must therefore be compared from an optional external campaign manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SOURCES = ("c8emrt", "proposal-native")
SOURCE_LABELS = {
    "c8emrt": "C8EMRT CUDA",
    "proposal-native": "PROPOSAL-native CUDA",
}
SOURCE_COLORS = {"c8emrt": "#4472C4", "proposal-native": "#ED7D31"}
SHOWER_PATTERN = re.compile(r"^shower_(\d+)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# CLI11 accepts aliases, but a timing audit must compare semantic options rather
# than the spelling selected by an older runner.  Unknown options are rejected:
# silently dropping a new physics or performance switch would make the timing
# comparison unsafe.
VALUE_OPTION_CANONICAL = {
    "-Z": "primary_atomic_number",
    "-A": "primary_mass_number",
    "-p": "primary_pdg",
    "--pdg": "primary_pdg",
    "-E": "primary_energy_gev",
    "--energy": "primary_energy_gev",
    "--energy_range": "primary_energy_range",
    "--eslope": "energy_spectral_index",
    "-z": "zenith_deg",
    "--zenith": "zenith_deg",
    "-a": "azimuth_deg",
    "--azimuth": "azimuth_deg",
    "--emcut": "em_cut_gev",
    "--hadcut": "hadron_cut_gev",
    "--mucut": "muon_cut_gev",
    "--taucut": "tau_cut_gev",
    "--max-deflection-angle": "maximum_deflection_angle_rad",
    "--geomagnetic-model": "geomagnetic_model",
    "--geomagnetic-year": "geomagnetic_year",
    "--em-backend": "em_backend",
    "--gpu-device": "gpu_device",
    "--gpu-min-batch": "gpu_min_batch",
    "--gpu-memory-fraction": "gpu_memory_fraction",
    "--gpu-table-cache": "gpu_table_cache",
    "--gpu-physics-source": "gpu_physics_source",
    "--gpu-aux-cache-dir": "gpu_aux_cache_dir",
    "--gpu-table-tolerance": "gpu_table_tolerance",
    "--gpu-deterministic": "gpu_deterministic",
    "--gpu-resident-cross-species": "gpu_resident_cross_species",
    "--radio-backend": "radio_backend",
    "--gpu-radio-field-limit": "gpu_radio_field_limit",
    "--radio-sampling-rate-ghz": "radio_sampling_rate_ghz",
    "--radio-window-duration-ns": "radio_window_duration_ns",
    "--radio-pretrigger-ns": "radio_pretrigger_ns",
    "--cuda-replay-trace": "cuda_replay_trace",
    "--cuda-replay-tape-out": "cuda_replay_tape_out",
    "--hadronic-plan-workers": "hadronic_plan_workers",
    "--hadronic-plan-target-ms": "hadronic_plan_target_ms",
    "--hadronic-plan-max-batch": "hadronic_plan_max_batch",
    "--hadronic-backend": "hadronic_backend",
    "--hadronic-workers": "hadronic_workers",
    "--hadronic-min-batch": "hadronic_min_batch",
    "--hadronic-target-batch-ms": "hadronic_target_batch_ms",
    "--hadronic-max-batch": "hadronic_max_batch",
    "--hadronic-initial-cost-ms": "hadronic_initial_cost_ms",
    "--hadronic-worker-executable": "hadronic_worker_executable",
    "--neutrino-interaction-type": "neutrino_interaction_type",
    "--observation-level": "observation_level_m",
    "--injection-height": "injection_height_m",
    "--shower-core-x": "shower_core_x_m",
    "--shower-core-y": "shower_core_y_m",
    "-N": "event_count",
    "--nevent": "event_count",
    "-f": "output_path",
    "--filename": "output_path",
    "-s": "seed",
    "--seed": "seed",
    "-v": "verbosity",
    "--verbosity": "verbosity",
    "-M": "hadronic_model",
    "--hadronModel": "hadronic_model",
    "-T": "hadronic_transition_energy_gev",
    "--hadronModelTransitionEnergy": "hadronic_transition_energy_gev",
    "--emthin": "em_thinning",
    "--max-weight": "maximum_weight",
    "--ring": "observer_ring",
    "--antenna-file": "antenna_file",
}
FLAG_OPTION_CANONICAL = {
    "--track-neutrinos": "track_neutrinos",
    "--gpu-detailed-stage-timing": "gpu_detailed_stage_timing",
    "--gpu-full-step-records": "gpu_full_step_records",
    "--gpu-radio-track-diagnostics": "gpu_radio_track_diagnostics",
    "--cpu-detailed-step-timing": "cpu_detailed_step_timing",
    "--compress": "compress_output",
    "--force-interaction": "force_interaction",
    "--force-decay": "force_decay",
    "--disable-interaction-histograms": "disable_interaction_histograms",
    "--multithin": "multithin",
}
INTEGER_COMMAND_OPTIONS = {
    "primary_atomic_number",
    "primary_mass_number",
    "primary_pdg",
    "gpu_device",
    "gpu_min_batch",
    "hadronic_plan_workers",
    "hadronic_plan_max_batch",
    "hadronic_workers",
    "hadronic_min_batch",
    "hadronic_max_batch",
    "event_count",
    "seed",
    "observer_ring",
}
FLOAT_COMMAND_OPTIONS = {
    "primary_energy_gev",
    "energy_spectral_index",
    "zenith_deg",
    "azimuth_deg",
    "em_cut_gev",
    "hadron_cut_gev",
    "muon_cut_gev",
    "tau_cut_gev",
    "maximum_deflection_angle_rad",
    "geomagnetic_year",
    "gpu_memory_fraction",
    "gpu_table_tolerance",
    "gpu_radio_field_limit",
    "radio_sampling_rate_ghz",
    "radio_window_duration_ns",
    "radio_pretrigger_ns",
    "hadronic_plan_target_ms",
    "hadronic_target_batch_ms",
    "hadronic_initial_cost_ms",
    "observation_level_m",
    "injection_height_m",
    "shower_core_x_m",
    "shower_core_y_m",
    "hadronic_transition_energy_gev",
    "em_thinning",
    "maximum_weight",
}
BOOLEAN_VALUE_COMMAND_OPTIONS = {
    "gpu_deterministic",
    "gpu_resident_cross_species",
}
PERMITTED_COMMAND_DIFFERENCES = {
    "output_path",
    "seed",
    "gpu_physics_source",
    "gpu_table_cache",
    "gpu_aux_cache_dir",
    "antenna_file",  # compared by immutable content SHA-256 instead
}
REQUIRED_COMPARISON_OPTIONS = {
    "primary_pdg",
    "primary_energy_gev",
    "event_count",
    "zenith_deg",
    "azimuth_deg",
    "geomagnetic_model",
    "geomagnetic_year",
    "shower_core_x_m",
    "shower_core_y_m",
    "observer_ring",
    "antenna_file",
    "radio_sampling_rate_ghz",
    "radio_window_duration_ns",
    "radio_pretrigger_ns",
    "em_cut_gev",
    "em_thinning",
    "hadron_cut_gev",
    "muon_cut_gev",
    "tau_cut_gev",
    "em_backend",
    "radio_backend",
    "gpu_device",
    "gpu_min_batch",
    "gpu_memory_fraction",
    "gpu_table_tolerance",
    "gpu_deterministic",
    "gpu_resident_cross_species",
    "gpu_radio_field_limit",
    "hadronic_backend",
    "hadronic_workers",
    "hadronic_min_batch",
    "hadronic_target_batch_ms",
    "hadronic_max_batch",
}


def parse_prefix_map(value: str) -> dict[str, str]:
    old_text, separator, new_text = value.partition("=")
    if not separator or not old_text or not new_text:
        raise argparse.ArgumentTypeError("path map must use absolute OLD=NEW syntax")
    old_path = Path(old_text)
    new_path = Path(new_text)
    if not old_path.is_absolute() or not new_path.is_absolute():
        raise argparse.ArgumentTypeError("both path-map prefixes must be absolute")
    old_resolved = old_path.resolve(strict=False)
    new_resolved = new_path.resolve(strict=False)
    if old_resolved == new_resolved:
        raise argparse.ArgumentTypeError("OLD and NEW path-map prefixes must differ")
    return {
        "old_prefix": str(old_resolved),
        "new_prefix": str(new_resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c8emrt", type=Path, action="append", required=True,
        help="Completed C8EMRT CUDA output; repeat once per shard.",
    )
    parser.add_argument(
        "--proposal-native", type=Path, action="append", required=True,
        help="Completed PROPOSAL-native CUDA output; repeat once per shard.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-c8emrt-events", type=int)
    parser.add_argument("--expected-proposal-native-events", type=int)
    parser.add_argument(
        "--c8emrt-campaign-manifest",
        type=Path,
        action="append",
        help="Optional completed runner manifest for external C8EMRT batch time.",
    )
    parser.add_argument(
        "--proposal-native-campaign-manifest",
        type=Path,
        action="append",
        help="Optional completed runner manifest for external native batch time.",
    )
    parser.add_argument(
        "--c8emrt-manifest-path-map",
        action="append",
        metavar="OLD=NEW",
        help="Explicit absolute prefix remap for paths recorded by C8EMRT manifests.",
    )
    parser.add_argument(
        "--proposal-native-manifest-path-map",
        action="append",
        metavar="OLD=NEW",
        help="Explicit absolute prefix remap for paths recorded by native manifests.",
    )
    parser.add_argument(
        "--allow-legacy-c8emrt-source-inference",
        action="store_true",
        help=(
            "Accept an archived C8EMRT shard without gpu_physics_source only "
            "when config and provenance independently identify a .c8emrt table."
        ),
    )
    parser.add_argument(
        "--allow-performance-config-mismatch",
        action="store_true",
        help=(
            "Report rather than reject GPU/device/timing configuration "
            "mismatches.  Such a result is descriptive, not controlled."
        ),
    )
    parser.add_argument(
        "--maximum-regression-percent", type=float, default=3.0,
        help="Gate on the steady-state median native/C8EMRT ratio.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument(
        "--fail-on-regression", action="store_true",
        help="Return status 2 when the steady-state median gate fails.",
    )
    args = parser.parse_args()
    if args.expected_c8emrt_events is not None and args.expected_c8emrt_events < 1:
        parser.error("--expected-c8emrt-events must be positive")
    if (
        args.expected_proposal_native_events is not None
        and args.expected_proposal_native_events < 1
    ):
        parser.error("--expected-proposal-native-events must be positive")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    if args.maximum_regression_percent < 0.0:
        parser.error("--maximum-regression-percent must be non-negative")
    if (args.c8emrt_campaign_manifest is None) != (
        args.proposal_native_campaign_manifest is None
    ):
        parser.error("both campaign manifests must be supplied together")
    for option in (
        "c8emrt_manifest_path_map",
        "proposal_native_manifest_path_map",
    ):
        requested = getattr(args, option)
        if requested is not None and len(requested) != 1:
            parser.error(f"--{option.replace('_', '-')} may be supplied only once")
        if requested is not None:
            try:
                mapping = parse_prefix_map(requested[0])
            except argparse.ArgumentTypeError as error:
                parser.error(str(error))
            setattr(args, option, mapping)
    if args.c8emrt_campaign_manifest is None and (
        args.c8emrt_manifest_path_map is not None
        or args.proposal_native_manifest_path_map is not None
    ):
        parser.error("manifest path maps require both campaign-manifest options")
    return args


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"invalid {label} SHA-256: {value!r}")
    return text


def command_has_option(command: Any, option: str) -> bool:
    return isinstance(command, list) and any(str(token) == option for token in command)


def normalize_command_value(name: str, raw_value: str) -> Any:
    if name in INTEGER_COMMAND_OPTIONS:
        try:
            return int(raw_value)
        except ValueError as error:
            raise ValueError(f"invalid integer value for {name}: {raw_value!r}") from error
    if name in FLOAT_COMMAND_OPTIONS:
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"invalid floating value for {name}: {raw_value!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"non-finite floating value for {name}: {raw_value!r}")
        return value
    if name in BOOLEAN_VALUE_COMMAND_OPTIONS:
        lowered = raw_value.strip().lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"invalid Boolean value for {name}: {raw_value!r}")
        return lowered == "true"
    return raw_value


def normalized_command_signature(command: Any) -> dict[str, Any]:
    """Normalize a recorded CLI command while retaining every known option."""

    if (
        not isinstance(command, list)
        or len(command) < 2
        or not all(isinstance(token, (str, int, float)) for token in command)
    ):
        raise ValueError("recorded command must be a non-empty token list")
    options: dict[str, Any] = {}
    index = 1  # executable path is an explicitly permitted build difference
    while index < len(command):
        raw_token = str(command[index])
        inline_value: str | None = None
        token = raw_token
        if raw_token.startswith("--") and "=" in raw_token:
            token, inline_value = raw_token.split("=", 1)
        if token in FLAG_OPTION_CANONICAL:
            if inline_value is not None:
                raise ValueError(f"flag {token} unexpectedly has a value")
            name = FLAG_OPTION_CANONICAL[token]
            value: Any = True
            index += 1
        elif token in VALUE_OPTION_CANONICAL:
            name = VALUE_OPTION_CANONICAL[token]
            if inline_value is None:
                if index + 1 >= len(command):
                    raise ValueError(f"option {token} has no value")
                inline_value = str(command[index + 1])
                index += 2
            else:
                index += 1
            value = normalize_command_value(name, inline_value)
        else:
            raise ValueError(f"unknown or positional command token: {raw_token!r}")
        if name in options:
            raise ValueError(f"duplicate command option after alias normalization: {name}")
        options[name] = value

    missing = sorted(REQUIRED_COMPARISON_OPTIONS - set(options))
    if missing:
        raise ValueError(f"recorded command lacks required comparison options: {missing}")
    immutable = {
        key: value
        for key, value in sorted(options.items())
        if key not in PERMITTED_COMMAND_DIFFERENCES
    }
    return {
        "immutable_options": immutable,
        "maximum_weight": {
            "present": "maximum_weight" in options,
            "value": options.get("maximum_weight"),
        },
        "permitted_differences": {
            key: options.get(key)
            for key in sorted(PERMITTED_COMMAND_DIFFERENCES)
        },
    }


def validate_recorded_commands(
    root: Path, provenance: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = provenance.get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise ValueError(f"invalid provenance command in {root}")
    encoded = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    if provenance.get("command_sha256") != expected_sha256:
        raise ValueError(f"provenance command SHA-256 differs in {root}")
    provenance_signature = normalized_command_signature(command)

    output_config_path = root / "config.yaml"
    output_config = read_yaml(output_config_path)
    args = output_config.get("args")
    if not isinstance(args, str) or not args.strip():
        raise ValueError(f"OutputManager config has no command string in {root}")
    try:
        output_command = shlex.split(args)
    except ValueError as error:
        raise ValueError(f"invalid OutputManager command string in {root}") from error
    output_signature = normalized_command_signature(output_command)

    # The executable path is intentionally omitted by normalization.  The
    # output path can refer to a temporary attempt directory that was renamed
    # atomically after completion.  Every other allowed-difference field must
    # still agree within one shard.
    provenance_crosscheck = {
        "immutable_options": provenance_signature["immutable_options"],
        "maximum_weight": provenance_signature["maximum_weight"],
        "permitted_differences": {
            key: value
            for key, value in provenance_signature["permitted_differences"].items()
            if key != "output_path"
        },
    }
    output_crosscheck = {
        "immutable_options": output_signature["immutable_options"],
        "maximum_weight": output_signature["maximum_weight"],
        "permitted_differences": {
            key: value
            for key, value in output_signature["permitted_differences"].items()
            if key != "output_path"
        },
    }
    if provenance_crosscheck != output_crosscheck:
        raise ValueError(
            f"provenance and OutputManager normalized commands differ in {root}"
        )
    return provenance_signature, {
        "path": str(output_config_path),
        "sha256": file_sha256(output_config_path),
        "normalized_signature": output_signature,
    }


def nested_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping for {label}")
    return value


def normalized_config_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Select immutable non-table GPU/environment configuration fields."""

    device = nested_mapping(config.get("device"), "device")
    cuda = nested_mapping(config.get("cuda"), "cuda")
    environment = nested_mapping(config.get("environment"), "environment")
    magnetic_field = nested_mapping(
        environment.get("magnetic_field_T"), "environment.magnetic_field_T"
    )
    plane_point = nested_mapping(
        environment.get("observation_plane_point_m"),
        "environment.observation_plane_point_m",
    )
    plane_normal = nested_mapping(
        environment.get("observation_plane_normal"),
        "environment.observation_plane_normal",
    )
    return {
        "backend": config.get("backend"),
        "backend_version": config.get("backend_version"),
        "device": {
            "index": device.get("index"),
            "name": device.get("name"),
            "compute_capability": device.get("compute_capability"),
            "total_memory_bytes": device.get("total_memory_bytes"),
        },
        "cuda": {
            "driver_version": cuda.get("driver_version"),
            "runtime_version": cuda.get("runtime_version"),
        },
        "deterministic": config.get("deterministic"),
        "detailed_stage_timing": config.get("detailed_stage_timing"),
        "radio_backend": config.get("radio_backend"),
        "radio_fixed_point_field_limit_V_per_m": config.get(
            "radio_fixed_point_field_limit_V_per_m"
        ),
        "radio_track_diagnostics": config.get("radio_track_diagnostics"),
        "minimum_batch_size": config.get("minimum_batch_size"),
        "memory_fraction": config.get("memory_fraction"),
        "accepted_table_tolerance": config.get("accepted_table_tolerance"),
        "environment": {
            "geomagnetic_model": environment.get("geomagnetic_model"),
            "geomagnetic_year": environment.get("geomagnetic_year"),
            "latitude_deg": environment.get("latitude_deg"),
            "longitude_deg": environment.get("longitude_deg"),
            "altitude_m": environment.get("altitude_m"),
            "magnetic_field_T": {
                "x": magnetic_field.get("x"),
                "y": magnetic_field.get("y"),
                "z": magnetic_field.get("z"),
            },
            "maximum_magnetic_deflection_rad": environment.get(
                "maximum_magnetic_deflection_rad"
            ),
            "observation_geometry": environment.get("observation_geometry"),
            "observation_plane_point_m": {
                "x": plane_point.get("x"),
                "y": plane_point.get("y"),
                "z": plane_point.get("z"),
            },
            "observation_plane_normal": {
                "x": plane_normal.get("x"),
                "y": plane_normal.get("y"),
                "z": plane_normal.get("z"),
            },
        },
    }


def require_provenance_content_hash(
    provenance: dict[str, Any], key: str, label: str
) -> str:
    record = provenance.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"missing {label} provenance")
    return require_sha256(record.get("sha256"), label)


def float_equal(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(
            float(first), float(second), rel_tol=2.0e-15, abs_tol=1.0e-15
        )
    return first == second


def require_command_config_consistency(
    command_signature: dict[str, Any],
    config_signature: dict[str, Any],
    root: Path,
) -> None:
    options = command_signature["immutable_options"]
    config_checks = {
        "gpu_device": config_signature["device"]["index"],
        "gpu_min_batch": config_signature["minimum_batch_size"],
        "gpu_memory_fraction": config_signature["memory_fraction"],
        "gpu_table_tolerance": config_signature["accepted_table_tolerance"],
        "gpu_deterministic": config_signature["deterministic"],
        "radio_backend": config_signature["radio_backend"],
        "gpu_radio_field_limit": config_signature[
            "radio_fixed_point_field_limit_V_per_m"
        ],
        "geomagnetic_model": config_signature["environment"][
            "geomagnetic_model"
        ],
        "geomagnetic_year": config_signature["environment"]["geomagnetic_year"],
    }
    for option, configured in config_checks.items():
        if not float_equal(options.get(option), configured):
            raise ValueError(
                f"command/config contradiction in {root}: {option}="
                f"{options.get(option)!r}, configured={configured!r}"
            )
    command_detailed = bool(options.get("gpu_detailed_stage_timing", False))
    if command_detailed is not bool(config_signature["detailed_stage_timing"]):
        raise ValueError(
            f"command/config contradiction in {root}: detailed stage timing"
        )
    command_radio_diagnostics = bool(
        options.get("gpu_radio_track_diagnostics", False)
    )
    if command_radio_diagnostics is not bool(
        config_signature["radio_track_diagnostics"]
    ):
        raise ValueError(
            f"command/config contradiction in {root}: radio track diagnostics"
        )


def validate_source_identity(
    root: Path,
    expected_source: str,
    *,
    allow_legacy_c8emrt: bool,
) -> dict[str, Any]:
    """Fail closed unless a shard has an unambiguous GPU physics source."""

    config_path = root / "gpu_em" / "config.yaml"
    provenance_path = root / "validation_provenance.json"
    config = read_yaml(config_path)
    provenance = read_json(provenance_path)
    if provenance.get("backend") != "cuda":
        raise ValueError(f"non-CUDA provenance in {provenance_path}")

    explicit_values = {
        str(value)
        for value in (
            config.get("gpu_physics_source"),
            provenance.get("gpu_physics_source"),
        )
        if value is not None
    }
    if explicit_values:
        if explicit_values != {expected_source}:
            raise ValueError(
                f"GPU physics source mismatch in {root}: expected "
                f"{expected_source!r}, observed {sorted(explicit_values)!r}"
            )
        inference = "explicit"
        evidence = ["gpu_em/config.yaml or validation_provenance.json"]
    else:
        if expected_source != "c8emrt" or not allow_legacy_c8emrt:
            raise ValueError(
                f"GPU physics source is not recorded in {root}; legacy "
                "C8EMRT inference requires explicit opt-in"
            )
        table_config = config.get("table")
        table_provenance = provenance.get("table")
        command = provenance.get("command")
        if not isinstance(table_config, dict) or not isinstance(table_provenance, dict):
            raise ValueError(f"legacy C8EMRT table evidence is missing in {root}")
        config_table_path = Path(str(table_config.get("path", "")))
        provenance_table_path = Path(str(table_provenance.get("path", "")))
        format_version = table_config.get("format_version")
        if (
            config_table_path.suffix != ".c8emrt"
            or provenance_table_path.suffix != ".c8emrt"
            or not isinstance(format_version, int)
            or format_version < 1
            or not command_has_option(command, "--gpu-table-cache")
            or command_has_option(command, "--gpu-physics-source")
            or command_has_option(command, "--gpu-aux-cache-dir")
        ):
            raise ValueError(f"legacy C8EMRT evidence is contradictory in {root}")
        require_sha256(table_config.get("sha256"), "serialized table payload")
        require_sha256(table_provenance.get("sha256"), "C8EMRT table file")
        inference = "legacy-c8emrt-inferred"
        evidence = [
            "config table.path has .c8emrt suffix and positive format_version",
            "provenance table.path has .c8emrt suffix and valid SHA-256",
            "recorded command uses --gpu-table-cache and no native-source option",
        ]

    command_signature, output_manager_command = validate_recorded_commands(
        root, provenance
    )
    permitted_command = command_signature["permitted_differences"]
    command_source = permitted_command["gpu_physics_source"]
    command_table = permitted_command["gpu_table_cache"]
    command_aux = permitted_command["gpu_aux_cache_dir"]
    if inference == "legacy-c8emrt-inferred":
        if command_source is not None:
            raise ValueError(f"legacy C8EMRT command unexpectedly names a source: {root}")
    elif command_source != expected_source:
        raise ValueError(
            f"command/config contradiction in {root}: command physics source "
            f"is {command_source!r}, expected {expected_source!r}"
        )
    if expected_source == "c8emrt":
        if command_table is None or command_aux is not None:
            raise ValueError(f"C8EMRT command has invalid table/aux options in {root}")
    elif command_table is not None:
        raise ValueError(f"PROPOSAL-native command unexpectedly uses C8EMRT in {root}")

    device = config.get("device")
    cuda = config.get("cuda")
    if not isinstance(device, dict) or not isinstance(cuda, dict):
        raise ValueError(f"GPU device/runtime metadata is missing in {config_path}")
    signature = {
        "device_name": str(device.get("name")),
        "compute_capability": str(device.get("compute_capability")),
        "total_memory_bytes": int(device.get("total_memory_bytes", -1)),
        "cuda_driver_version": str(cuda.get("driver_version")),
        "cuda_runtime_version": str(cuda.get("runtime_version")),
        "minimum_batch_size": int(config.get("minimum_batch_size", -1)),
        "memory_fraction": float(config.get("memory_fraction", math.nan)),
        "detailed_stage_timing": config.get("detailed_stage_timing"),
        "radio_backend": str(config.get("radio_backend")),
        "deterministic": config.get("deterministic"),
    }
    if (
        signature["device_name"] in {"", "None"}
        or signature["total_memory_bytes"] <= 0
        or signature["minimum_batch_size"] <= 0
        or not math.isfinite(signature["memory_fraction"])
        or not 0.0 < signature["memory_fraction"] <= 1.0
        or signature["detailed_stage_timing"] is not False
        or signature["deterministic"] is not True
    ):
        raise ValueError(f"invalid performance configuration in {config_path}")
    config_signature = normalized_config_signature(config)
    require_command_config_consistency(command_signature, config_signature, root)
    antenna_sha256 = require_provenance_content_hash(
        provenance, "antenna_file", "antenna file"
    )
    flupro_sha256 = require_provenance_content_hash(provenance, "flupro", "FLUKA")
    immutable_run_signature = {
        "command_options": command_signature["immutable_options"],
        "maximum_weight": command_signature["maximum_weight"],
        "gpu_environment_config": config_signature,
        "antenna_sha256": antenna_sha256,
        "flupro_sha256": flupro_sha256,
    }
    return {
        "identity_method": inference,
        "identity_evidence": evidence,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": file_sha256(provenance_path),
        "performance_signature": signature,
        "immutable_run_signature": immutable_run_signature,
        "permitted_command_differences": permitted_command,
        "output_manager_command": output_manager_command,
    }


def extract_timing_records(path: Path) -> list[tuple[int, float]]:
    timing = read_yaml(path)
    records: list[tuple[int, float]] = []
    for key, value in timing.items():
        match = SHOWER_PATTERN.fullmatch(str(key))
        if match is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid timing record {key} in {path}")
        if value.get("closed") is not True or value.get("status") != "closed":
            raise ValueError(f"incomplete timing record {key} in {path}")
        wall_time_ms = value.get("wall_time_ms")
        if isinstance(wall_time_ms, bool) or not isinstance(wall_time_ms, (int, float)):
            raise ValueError(f"invalid wall_time_ms for {key} in {path}")
        seconds = float(wall_time_ms) / 1000.0
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"non-positive wall time for {key} in {path}")
        records.append((int(match.group(1)), seconds))
    records.sort()
    if [index for index, _ in records] != list(range(len(records))):
        raise ValueError(f"non-contiguous shower timing records in {path}")
    if not records:
        raise ValueError(f"no closed shower timing records in {path}")
    return records


def load_source(
    roots: Iterable[Path],
    source: str,
    *,
    allow_legacy_c8emrt: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    seen: set[Path] = set()
    event_offset = 0
    for shard_index, raw_root in enumerate(roots):
        root = raw_root.resolve()
        if root in seen:
            raise ValueError(f"duplicate {source} timing shard: {root}")
        if not root.is_dir():
            raise FileNotFoundError(root)
        seen.add(root)
        identity = validate_source_identity(
            root, source, allow_legacy_c8emrt=allow_legacy_c8emrt
        )
        timing_path = root / "simulation_timing" / "summary.yaml"
        records = extract_timing_records(timing_path)
        command_events = identity["immutable_run_signature"]["command_options"][
            "event_count"
        ]
        if command_events != len(records):
            raise ValueError(
                f"recorded -N/--nevent differs from timing coverage in {root}: "
                f"command={command_events}, timing={len(records)}"
            )
        provenance_events = read_json(root / "validation_provenance.json").get("events")
        if provenance_events is not None and provenance_events != len(records):
            raise ValueError(
                f"provenance event count differs from timing coverage in {root}: "
                f"provenance={provenance_events!r}, timing={len(records)}"
            )
        for local_index, seconds in records:
            rows.append(
                {
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "event_index": event_offset + local_index,
                    "shard_index": shard_index,
                    "local_event_index": local_index,
                    "is_warm": local_index > 0,
                    "runtime_seconds": seconds,
                    "root": str(root),
                    "timing_path": str(timing_path),
                }
            )
        shards.append(
            {
                "source": source,
                "shard_index": shard_index,
                "root": str(root),
                "events": len(records),
                "warm_events": max(0, len(records) - 1),
                "timing_path": str(timing_path),
                "timing_sha256": file_sha256(timing_path),
                **identity,
            }
        )
        event_offset += len(records)
    if not rows:
        raise ValueError(f"no {source} timing shards were supplied")
    return rows, shards


def remap_manifest_output(
    recorded_output: str,
    path_map: dict[str, str] | None,
    *,
    label: str,
) -> Path:
    recorded = Path(recorded_output)
    if not recorded.is_absolute():
        raise ValueError(f"manifest output is not absolute in {label}: {recorded}")
    recorded_resolved = recorded.resolve(strict=False)
    if path_map is None:
        return recorded_resolved
    old_prefix = Path(path_map["old_prefix"])
    new_prefix = Path(path_map["new_prefix"])
    try:
        relative = recorded_resolved.relative_to(old_prefix)
    except ValueError as error:
        raise ValueError(
            f"manifest output is outside OLD path-map prefix in {label}: "
            f"output={recorded_resolved}, OLD={old_prefix}"
        ) from error
    remapped = (new_prefix / relative).resolve(strict=False)
    try:
        remapped.relative_to(new_prefix)
    except ValueError as error:
        raise ValueError(
            f"remapped manifest output escapes NEW prefix in {label}: {remapped}"
        ) from error
    return remapped


def load_campaign_batch_timings(
    manifest_paths: Iterable[Path],
    source: str,
    shards: list[dict[str, Any]],
    *,
    path_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load external child-process times and prove exact shard coverage."""

    expected = {Path(shard["root"]).resolve(): shard for shard in shards}
    observed: dict[Path, dict[str, Any]] = {}
    manifest_metadata: list[dict[str, Any]] = []
    seen_manifests: set[Path] = set()
    for raw_path in manifest_paths:
        path = raw_path.resolve()
        if path in seen_manifests:
            raise ValueError(f"duplicate campaign manifest: {path}")
        seen_manifests.add(path)
        manifest = read_json(path)
        if manifest.get("status") != "complete":
            raise ValueError(f"campaign manifest is not complete: {path}")
        has_batches = isinstance(manifest.get("batches"), list)
        has_attempts = isinstance(manifest.get("attempts"), list)
        if has_batches == has_attempts:
            raise ValueError(
                f"ambiguous or unsupported campaign manifest schema: {path}"
            )

        records: list[dict[str, Any]] = []
        if has_batches:
            for index, batch in enumerate(manifest["batches"]):
                if not isinstance(batch, dict) or batch.get("status") != "complete":
                    raise ValueError(f"incomplete batch {index} in {path}")
                records.append(
                    {
                        "manifest_record": f"batches[{index}]",
                        "recorded_output": batch.get("cuda_output"),
                        "events": batch.get("events"),
                        "runtime_seconds": batch.get("runtime_seconds"),
                    }
                )
            schema = "legacy-batches"
        else:
            for index, attempt in enumerate(manifest["attempts"]):
                if not isinstance(attempt, dict) or attempt.get("status") != "complete":
                    continue
                task = str(attempt.get("task", ""))
                task_fields = task.split(":")
                if len(task_fields) != 3 or task_fields[0] != "formal":
                    continue
                if task_fields[2] != source:
                    continue
                command = normalized_command_signature(attempt.get("command"))
                records.append(
                    {
                        "manifest_record": f"attempts[{index}]",
                        "task": task,
                        "recorded_output": attempt.get("output"),
                        "events": command["immutable_options"]["event_count"],
                        "runtime_seconds": attempt.get("runtime_seconds"),
                    }
                )
            schema = "attempts"

        for record in records:
            recorded_output = record.get("recorded_output")
            events = record.get("events")
            runtime = record.get("runtime_seconds")
            label = f"{path}:{record['manifest_record']}"
            if not isinstance(recorded_output, str) or not recorded_output:
                raise ValueError(f"invalid campaign output in {label}")
            root = remap_manifest_output(recorded_output, path_map, label=label)
            if root in observed:
                raise ValueError(
                    f"duplicate completed campaign output across manifests: {root}"
                )
            if isinstance(events, bool) or not isinstance(events, int) or events < 1:
                raise ValueError(f"invalid campaign event count for {root}: {events!r}")
            if (
                isinstance(runtime, bool)
                or not isinstance(runtime, (int, float))
                or not math.isfinite(float(runtime))
                or float(runtime) <= 0.0
            ):
                raise ValueError(f"invalid campaign runtime for {root}: {runtime!r}")
            observed[root] = {
                **record,
                "manifest_path": str(path),
                "output": str(root),
                "runtime_seconds": float(runtime),
                "per_event_turnaround_seconds": float(runtime) / events,
            }
        manifest_metadata.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "schema": schema,
                "completed_records": len(records),
            }
        )

    if set(observed) != set(expected):
        missing = sorted(str(root) for root in set(expected) - set(observed))
        extra = sorted(str(root) for root in set(observed) - set(expected))
        raise ValueError(
            f"campaign manifests do not exactly cover supplied {source} shards: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    for root, record in observed.items():
        if record["events"] != expected[root]["events"]:
            raise ValueError(
                f"campaign/timing event count differs for {root}: "
                f"manifest={record['events']}, timing={expected[root]['events']}"
            )
    return {
        "manifests": manifest_metadata,
        "path_map": path_map,
        "batches": [observed[root] for root in sorted(observed, key=str)],
    }


def values(rows: list[dict[str, Any]], *, warm_only: bool) -> np.ndarray:
    selected = [
        float(row["runtime_seconds"])
        for row in rows
        if not warm_only or bool(row["is_warm"])
    ]
    if not selected:
        raise ValueError("steady-state comparison requires a multi-event shard")
    return np.asarray(selected, dtype=np.float64)


def describe(sample: np.ndarray) -> dict[str, Any]:
    if sample.size == 0 or not np.all(np.isfinite(sample)) or np.any(sample <= 0.0):
        raise ValueError("runtime samples must be positive finite values")
    return {
        "count": int(sample.size),
        "mean_seconds": float(np.mean(sample)),
        "median_seconds": float(np.median(sample)),
        "sample_std_seconds": (
            float(np.std(sample, ddof=1)) if sample.size > 1 else None
        ),
        "minimum_seconds": float(np.min(sample)),
        "p05_seconds": float(np.percentile(sample, 5)),
        "p16_seconds": float(np.percentile(sample, 16)),
        "p84_seconds": float(np.percentile(sample, 84)),
        "p95_seconds": float(np.percentile(sample, 95)),
        "maximum_seconds": float(np.max(sample)),
        "total_seconds": float(np.sum(sample)),
    }


def ratio_summary(c8emrt: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    mean_ratio = float(np.mean(native) / np.mean(c8emrt))
    median_ratio = float(np.median(native) / np.median(c8emrt))
    return {
        "ratio_definition": "proposal-native / c8emrt; smaller is faster",
        "ratio_of_means": mean_ratio,
        "mean_regression_percent": 100.0 * (mean_ratio - 1.0),
        "mean_c8emrt_over_native_speed_factor": 1.0 / mean_ratio,
        "ratio_of_medians": median_ratio,
        "median_regression_percent": 100.0 * (median_ratio - 1.0),
        "median_c8emrt_over_native_speed_factor": 1.0 / median_ratio,
        "ratio_of_totals": (
            float(np.sum(native) / np.sum(c8emrt))
            if native.size == c8emrt.size
            else None
        ),
    }


def bootstrap_ratio_intervals(
    c8emrt: np.ndarray,
    native: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    mean_ratios = np.empty(repetitions, dtype=np.float64)
    median_ratios = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        c8_draw = rng.choice(c8emrt, size=c8emrt.size, replace=True)
        native_draw = rng.choice(native, size=native.size, replace=True)
        mean_ratios[index] = np.mean(native_draw) / np.mean(c8_draw)
        median_ratios[index] = np.median(native_draw) / np.median(c8_draw)
    return {
        "independent_mean_ratio_95pct": [
            float(value) for value in np.percentile(mean_ratios, [2.5, 97.5])
        ],
        "independent_median_ratio_95pct": [
            float(value) for value in np.percentile(median_ratios, [2.5, 97.5])
        ],
    }


def flattened_signature(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in sorted(value.items()):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(flattened_signature(child, child_prefix))
    return flattened


def signature_mismatches(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_root = shards[0]["root"]
    baseline = flattened_signature(
        {
            "performance_signature": shards[0]["performance_signature"],
            "immutable_run_signature": shards[0]["immutable_run_signature"],
        }
    )
    mismatches: list[dict[str, Any]] = []
    for shard in shards[1:]:
        observed = flattened_signature(
            {
                "performance_signature": shard["performance_signature"],
                "immutable_run_signature": shard["immutable_run_signature"],
            }
        )
        for key in sorted(set(baseline) | set(observed)):
            expected_value = baseline.get(key, "<missing>")
            observed_value = observed.get(key, "<missing>")
            if not float_equal(expected_value, observed_value):
                mismatches.append(
                    {
                        "baseline_root": baseline_root,
                        "root": shard["root"],
                        "field": key,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )
    return mismatches


def common_edges(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    combined = np.concatenate((first, second))
    edges = np.histogram_bin_edges(combined, bins="fd")
    bin_count = min(40, max(8, len(edges) - 1))
    low = float(np.min(combined))
    high = float(np.max(combined))
    if low == high:
        return np.asarray([low * 0.95, high * 1.05], dtype=np.float64)
    return np.linspace(low, high, bin_count + 1)


def draw_histograms(
    samples: dict[str, dict[str, np.ndarray]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    for axis, sample_name, title in (
        (axes[0], "all_events", "All recorded shower intervals"),
        (axes[1], "steady_state", "Steady state (shower index > 0 per process)"),
    ):
        c8 = samples[sample_name]["c8emrt"]
        native = samples[sample_name]["proposal-native"]
        edges = common_edges(c8, native)
        for source, values_ in (("c8emrt", c8), ("proposal-native", native)):
            weights = np.full(values_.size, 1.0 / values_.size)
            axis.hist(
                values_, bins=edges, weights=weights, alpha=0.52,
                color=SOURCE_COLORS[source], edgecolor="white", linewidth=0.6,
                label=SOURCE_LABELS[source],
            )
            axis.axvline(
                np.median(values_), color=SOURCE_COLORS[source], linewidth=1.8,
                linestyle="--",
            )
        ratio = summary["comparisons"][sample_name]["ratio_of_medians"]
        axis.set_title(title)
        axis.set_xlabel("Per-shower wall time [s]")
        axis.set_ylabel("Fraction of showers per bin")
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, loc="upper left")
        axis.text(
            0.98, 0.96,
            f"native / C8EMRT median = {ratio:.4f}\n"
            f"regression = {100.0 * (ratio - 1.0):+.2f}%",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )
    figure.suptitle("CUDA physics-source per-shower runtime comparison")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def markdown_report(summary: dict[str, Any]) -> str:
    all_stats = summary["statistics"]["all_events"]
    warm_stats = summary["statistics"]["steady_state"]
    all_ratio = summary["comparisons"]["all_events"]
    warm_ratio = summary["comparisons"]["steady_state"]
    acceptance = summary["acceptance"]
    lines = [
        "# C8EMRT 与 PROPOSAL-native CUDA 逐事例计时比较",
        "",
        f"生成时间：`{summary['created_utc']}`。",
        "",
        "## 计时口径",
        "",
        "- 时间来自每个输出的 `simulation_timing/summary.yaml` 中闭合的 "
        "`wall_time_ms`，对应单个 shower 的 OutputManager 起止边界。",
        "- `全部事例` 包含每个进程的第 0 个 shower；`稳态` 从每个 shard "
        "中剔除第 0 个 shower，以隔离首次 allocator、kernel 与输出路径预热。"
        "物理表导出和设备上传发生在该计时区间之前，不能由 `wall_time_ms` "
        "衡量；如需生产周转时间，必须另外使用 runner 记录的完整批次时间。",
        "- 两个系综是独立随机样本，未计算逐事例配对比值。",
        "- 比值定义为 `PROPOSAL-native / C8EMRT`；小于 1 表示原生表更快。",
        "",
        "## 结果",
        "",
        "| 样本 | C8EMRT N | C8EMRT 中位数 [s] | Native N | Native 中位数 [s] | Native/C8EMRT | 相对变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 全部事例 | {all_stats['c8emrt']['count']} | "
        f"{all_stats['c8emrt']['median_seconds']:.6g} | "
        f"{all_stats['proposal-native']['count']} | "
        f"{all_stats['proposal-native']['median_seconds']:.6g} | "
        f"{all_ratio['ratio_of_medians']:.6f} | "
        f"{all_ratio['median_regression_percent']:+.3f}% |",
        f"| 稳态 | {warm_stats['c8emrt']['count']} | "
        f"{warm_stats['c8emrt']['median_seconds']:.6g} | "
        f"{warm_stats['proposal-native']['count']} | "
        f"{warm_stats['proposal-native']['median_seconds']:.6g} | "
        f"{warm_ratio['ratio_of_medians']:.6f} | "
        f"{warm_ratio['median_regression_percent']:+.3f}% |",
        "",
        f"稳态中位数门禁：原生表回退不得超过 "
        f"`{acceptance['maximum_regression_percent']:.3f}%`；结果："
        f"`{'PASS' if acceptance['shower_timing_passed'] else 'FAIL'}`。",
    ]
    external = summary.get("external_batch_turnaround")
    if external is not None:
        c8 = external["per_event_turnaround_seconds"]["c8emrt"]
        native = external["per_event_turnaround_seconds"]["proposal-native"]
        ratio = external["per_event_turnaround_comparison"]
        lines.extend(
            [
                "",
                "## 完整批次周转时间",
                "",
                "该时间来自 runner 对子进程的外部计时，包含进程启动、物理表"
                "导出或加载、设备上传、全部 shower 与输出关闭。",
            ]
        )
        for source in SOURCES:
            mapping = external["campaigns"][source].get("path_map")
            if mapping is not None:
                lines.append(
                    f"- {SOURCE_LABELS[source]} manifest 路径前缀映射："
                    f"`{mapping['old_prefix']}` → `{mapping['new_prefix']}`。"
                )
        lines.extend(
            [
                "",
                "| 物理源 | 批次数 | 每事例周转中位数 [s] | 每事例周转均值 [s] |",
                "|---|---:|---:|---:|",
                f"| C8EMRT CUDA | {c8['count']} | {c8['median_seconds']:.6g} | "
                f"{c8['mean_seconds']:.6g} |",
                f"| PROPOSAL-native CUDA | {native['count']} | "
                f"{native['median_seconds']:.6g} | {native['mean_seconds']:.6g} |",
                "",
                f"完整批次每事例中位数比为 `{ratio['ratio_of_medians']:.6f}` "
                f"（相对变化 `{ratio['median_regression_percent']:+.3f}%`）；门禁 "
                f"`{'PASS' if acceptance['batch_turnaround_passed'] else 'FAIL'}`。",
                "逐批次数据见 `batch_turnaround_timings.csv`。",
            ]
        )
    lines.extend(
        [
            "",
            "![逐事例运行时间分布](gpu_physics_source_runtime_histograms.png)",
            "",
            "逐事例数据见 `per_shower_timings.csv`，完整统计、bootstrap 区间、"
            "输入文件哈希和性能配置审计见 `timing_comparison.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_shards: dict[str, list[dict[str, Any]]] = {}
    for source, roots in (
        ("c8emrt", args.c8emrt),
        ("proposal-native", args.proposal_native),
    ):
        all_rows[source], all_shards[source] = load_source(
            roots,
            source,
            allow_legacy_c8emrt=args.allow_legacy_c8emrt_source_inference,
        )

    expected = {
        "c8emrt": args.expected_c8emrt_events,
        "proposal-native": args.expected_proposal_native_events,
    }
    for source in SOURCES:
        if expected[source] is not None and len(all_rows[source]) != expected[source]:
            raise ValueError(
                f"expected {expected[source]} {source} events, observed "
                f"{len(all_rows[source])}"
            )

    flattened_shards = [*all_shards["c8emrt"], *all_shards["proposal-native"]]
    mismatches = signature_mismatches(flattened_shards)
    if mismatches and not args.allow_performance_config_mismatch:
        first = mismatches[0]
        raise ValueError(
            "performance configuration mismatch: "
            f"{first['field']} in {first['root']}: expected "
            f"{first['expected']!r}, observed {first['observed']!r}"
        )

    samples: dict[str, dict[str, np.ndarray]] = {
        "all_events": {
            source: values(all_rows[source], warm_only=False) for source in SOURCES
        },
        "steady_state": {
            source: values(all_rows[source], warm_only=True) for source in SOURCES
        },
    }
    statistics: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for sample_name, by_source in samples.items():
        statistics[sample_name] = {
            source: describe(by_source[source]) for source in SOURCES
        }
        comparison = ratio_summary(by_source["c8emrt"], by_source["proposal-native"])
        comparison.update(
            bootstrap_ratio_intervals(
                by_source["c8emrt"],
                by_source["proposal-native"],
                args.bootstrap_repetitions,
                args.bootstrap_seed + (0 if sample_name == "all_events" else 1),
            )
        )
        comparisons[sample_name] = comparison

    external_batch_turnaround: dict[str, Any] | None = None
    if args.c8emrt_campaign_manifest is not None:
        campaign_records = {
            "c8emrt": load_campaign_batch_timings(
                args.c8emrt_campaign_manifest,
                "c8emrt",
                all_shards["c8emrt"],
                path_map=args.c8emrt_manifest_path_map,
            ),
            "proposal-native": load_campaign_batch_timings(
                args.proposal_native_campaign_manifest,
                "proposal-native",
                all_shards["proposal-native"],
                path_map=args.proposal_native_manifest_path_map,
            ),
        }
        batch_runtime_samples = {
            source: np.asarray(
                [
                    record["runtime_seconds"]
                    for record in campaign_records[source]["batches"]
                ],
                dtype=np.float64,
            )
            for source in SOURCES
        }
        batch_per_event_samples = {
            source: np.asarray(
                [
                    record["per_event_turnaround_seconds"]
                    for record in campaign_records[source]["batches"]
                ],
                dtype=np.float64,
            )
            for source in SOURCES
        }
        external_batch_turnaround = {
            "definition": (
                "runner-observed child-process wall time; includes process startup, "
                "physics-source export/load, device upload, showers, and output close"
            ),
            "campaigns": campaign_records,
            "batch_runtime_seconds": {
                source: describe(batch_runtime_samples[source]) for source in SOURCES
            },
            "per_event_turnaround_seconds": {
                source: describe(batch_per_event_samples[source]) for source in SOURCES
            },
            "batch_runtime_comparison": ratio_summary(
                batch_runtime_samples["c8emrt"],
                batch_runtime_samples["proposal-native"],
            ),
            "per_event_turnaround_comparison": ratio_summary(
                batch_per_event_samples["c8emrt"],
                batch_per_event_samples["proposal-native"],
            ),
        }

    warm_ratio = comparisons["steady_state"]["ratio_of_medians"]
    shower_timing_passed = (
        warm_ratio <= 1.0 + args.maximum_regression_percent / 100.0
    )
    batch_turnaround_passed: bool | None = None
    if external_batch_turnaround is not None:
        external_ratio = external_batch_turnaround[
            "per_event_turnaround_comparison"
        ]["ratio_of_medians"]
        batch_turnaround_passed = (
            external_ratio <= 1.0 + args.maximum_regression_percent / 100.0
        )
    acceptance = {
        "metric": "steady_state.ratio_of_medians",
        "maximum_regression_percent": args.maximum_regression_percent,
        "threshold_ratio": 1.0 + args.maximum_regression_percent / 100.0,
        "observed_ratio": warm_ratio,
        "shower_timing_passed": shower_timing_passed,
        "batch_turnaround_passed": batch_turnaround_passed,
        "passed": shower_timing_passed and batch_turnaround_passed is not False,
        "controlled_same_performance_configuration": not mismatches,
    }
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timing_definition": (
            "simulation_timing wall_time_ms; OutputManager startOfShower "
            "through endOfShower; excludes pre-run physics-source export/upload"
        ),
        "source_labels": SOURCE_LABELS,
        "statistics": statistics,
        "comparisons": comparisons,
        "bootstrap": {
            "sampling": "independent within each physics-source ensemble",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "performance_configuration_mismatches": mismatches,
        "shards": all_shards,
        "external_batch_turnaround": external_batch_turnaround,
        "acceptance": acceptance,
    }

    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_shower_timings.csv").open(
        "x", encoding="utf-8", newline=""
    ) as destination:
        fieldnames = [
            "source", "source_label", "event_index", "shard_index",
            "local_event_index", "is_warm", "runtime_seconds", "root",
            "timing_path",
        ]
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*all_rows["c8emrt"], *all_rows["proposal-native"]])
    if external_batch_turnaround is not None:
        with (output / "batch_turnaround_timings.csv").open(
            "x", encoding="utf-8", newline=""
        ) as destination:
            fieldnames = [
                "source",
                "manifest_record",
                "output",
                "events",
                "runtime_seconds",
                "per_event_turnaround_seconds",
            ]
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for source in SOURCES:
                for record in external_batch_turnaround["campaigns"][source][
                    "batches"
                ]:
                    writer.writerow(
                        {key: source if key == "source" else record.get(key) for key in fieldnames}
                    )
    (output / "timing_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    draw_histograms(
        samples, summary, output / "gpu_physics_source_runtime_histograms.png"
    )
    (output / "README_CN.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"output": str(output), "acceptance": acceptance}, indent=2))
    return 0 if acceptance["passed"] or not args.fail_on_regression else 2


if __name__ == "__main__":
    raise SystemExit(main())
