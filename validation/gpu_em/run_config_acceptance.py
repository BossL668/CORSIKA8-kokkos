#!/usr/bin/env python3
"""Translate one existing CORSIKA task YAML into a CPU/CUDA acceptance run.

The adapter deliberately supports only the scalar-valued c8_air_shower options
that run_physics_acceptance.py can reproduce exactly.  Unknown, duplicated, or
backend-specific task options are rejected instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ANTENNA_FILE = Path("/home/yuhanglu/21CMA/data/antennas.txt")
DEFAULT_PROPOSAL_SEED = 41001
DEFAULT_CUDA_SEED = 51001


@dataclass
class TaskPhysics:
    task_name: str
    primary_pdg: int
    energy_GeV: float
    configured_events: int
    configured_seed: int
    zenith_deg: float
    azimuth_deg: float
    shower_core_x_m: float
    shower_core_y_m: float
    ring: int
    antenna_file: str
    em_cut_GeV: float
    em_thinning: float
    maximum_weight: float
    had_cut_GeV: float
    mu_cut_GeV: float
    tau_cut_GeV: float
    configured_output: str


VALUE_OPTIONS = {
    "-p": "primary_pdg",
    "--pdg": "primary_pdg",
    "-E": "energy_GeV",
    "--energy": "energy_GeV",
    "-N": "configured_events",
    "--nevent": "configured_events",
    "-s": "configured_seed",
    "--seed": "configured_seed",
    "-z": "zenith_deg",
    "--zenith": "zenith_deg",
    "-a": "azimuth_deg",
    "--azimuth": "azimuth_deg",
    "--shower-core-x": "shower_core_x_m",
    "--shower-core-y": "shower_core_y_m",
    "--ring": "ring",
    "--antenna-file": "antenna_file",
    "--emcut": "em_cut_GeV",
    "--emthin": "em_thinning",
    "--max-weight": "maximum_weight",
    "--hadcut": "had_cut_GeV",
    "--mucut": "mu_cut_GeV",
    "--taucut": "tau_cut_GeV",
    "-f": "configured_output",
    "--filename": "configured_output",
}

BACKEND_OPTIONS = {
    "--em-backend",
    "--radio-backend",
    "--gpu-device",
    "--gpu-min-batch",
    "--gpu-memory-fraction",
    "--gpu-table-cache",
    "--gpu-table-tolerance",
    "--gpu-deterministic",
    "--hadronic-backend",
    "--hadronic-workers",
    "--hadronic-min-batch",
    "--hadronic-target-batch-ms",
    "--hadronic-max-batch",
}

CONVERTERS = {
    "primary_pdg": int,
    "energy_GeV": float,
    "configured_events": int,
    "configured_seed": int,
    "zenith_deg": float,
    "azimuth_deg": float,
    "shower_core_x_m": float,
    "shower_core_y_m": float,
    "ring": int,
    "antenna_file": str,
    "em_cut_GeV": float,
    "em_thinning": float,
    "maximum_weight": float,
    "had_cut_GeV": float,
    "mu_cut_GeV": float,
    "tau_cut_GeV": float,
    "configured_output": str,
}

DEFAULTS: dict[str, Any] = {
    "configured_events": 1,
    "configured_seed": 0,
    "zenith_deg": 0.0,
    "azimuth_deg": 0.0,
    "shower_core_x_m": 0.0,
    "shower_core_y_m": 0.0,
    "ring": 0,
    "antenna_file": str(DEFAULT_ANTENNA_FILE),
    "em_cut_GeV": 0.5e-3,
    "em_thinning": 1.0e-6,
    # Zero means c8_air_shower selects 0.5 times the Kobal optimum.
    "maximum_weight": 0.0,
    "had_cut_GeV": 0.3,
    "mu_cut_GeV": 0.3,
    "tau_cut_GeV": 0.3,
    "configured_output": "",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def split_option(token: str) -> tuple[str, str | None]:
    if token.startswith("--") and "=" in token:
        option, value = token.split("=", 1)
        return option, value
    return token, None


def parse_task_physics(
    config_path: Path, task_index: int
) -> tuple[TaskPhysics, dict[str, Any], list[str]]:
    with config_path.open("r", encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError("task configuration must be a YAML mapping")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task configuration contains no tasks")
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(
            f"task index {task_index} is outside [0, {len(tasks) - 1}]"
        )
    task = tasks[task_index]
    if not isinstance(task, dict):
        raise ValueError(f"task {task_index} is not a mapping")
    raw_args = task.get("args")
    if not isinstance(raw_args, list):
        raise ValueError(f"task {task_index} args must be a list")
    tokens = [str(value) for value in raw_args]

    values = dict(DEFAULTS)
    seen: set[str] = set()
    unsupported: list[str] = []
    index = 0
    while index < len(tokens):
        option, attached = split_option(tokens[index])
        if option in BACKEND_OPTIONS:
            raise ValueError(
                f"task already contains implementation option {option}; "
                "the acceptance adapter owns backend selection"
            )
        field = VALUE_OPTIONS.get(option)
        if field is None:
            unsupported.append(tokens[index])
            index += 1
            continue
        if field in seen:
            raise ValueError(
                f"task specifies {field} more than once through {option}"
            )
        if attached is None:
            if index + 1 >= len(tokens):
                raise ValueError(f"task option {option} has no value")
            attached = tokens[index + 1]
            index += 2
        else:
            index += 1
        try:
            values[field] = CONVERTERS[field](attached)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid value for task option {option}: {attached}"
            ) from error
        seen.add(field)

    if unsupported:
        raise ValueError(
            "task contains options that cannot yet be reproduced exactly: "
            + " ".join(unsupported)
        )
    for required in ("primary_pdg", "energy_GeV"):
        if required not in seen:
            raise ValueError(f"task is missing required option {required}")
    if values["energy_GeV"] <= 0.0:
        raise ValueError("task energy must be positive")
    if values["configured_events"] <= 0:
        raise ValueError("task event count must be positive")
    if values["configured_seed"] < 0:
        raise ValueError("task seed must be non-negative")
    if not 0.0 <= values["em_thinning"] <= 1.0:
        raise ValueError("task emthin must be in [0, 1]")
    if values["maximum_weight"] < 0.0:
        raise ValueError("task maximum weight must be non-negative")
    for cut in (
        "em_cut_GeV",
        "had_cut_GeV",
        "mu_cut_GeV",
        "tau_cut_GeV",
    ):
        if values[cut] <= 0.0:
            raise ValueError(f"task {cut} must be positive")
    if values["ring"] < 0:
        raise ValueError("task ring must be non-negative")

    physics = TaskPhysics(
        task_name=str(task.get("name", f"task_{task_index}")),
        **values,
    )
    return physics, document, tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--cuda-executable", type=Path, required=True)
    parser.add_argument("--proposal-executable", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--events",
        type=int,
        required=True,
        help="Independent showers per backend; intentionally has no implicit UHE default.",
    )
    parser.add_argument("--proposal-seed", type=int)
    parser.add_argument("--cuda-seed", type=int)
    parser.add_argument(
        "--development-emthin",
        type=float,
        help=(
            "Explicitly replace the task emthin for a faster development run. "
            "The manifest marks the result non-production-equivalent."
        ),
    )
    parser.add_argument(
        "--shower-only",
        action="store_true",
        help=(
            "Disable observers in both arms to isolate shower transport. "
            "The manifest marks this as a controlled non-radio benchmark."
        ),
    )
    parser.add_argument("--antenna-file", type=Path)
    parser.add_argument("--proposal-shards", type=int, default=1)
    parser.add_argument("--proposal-parallelism", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--hadronic-workers", type=int, default=4)
    parser.add_argument("--hadronic-min-batch", type=int, default=64)
    parser.add_argument("--hadronic-target-batch-ms", type=float, default=5.0)
    parser.add_argument("--hadronic-max-batch", type=int, default=256)
    parser.add_argument(
        "--cuda-radio-backend",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--stability-bootstrap-repetitions",
        type=int,
        default=5000,
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Propagate the strict physics acceptance failure code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the translated command without executing it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label in (
        "config",
        "cuda_executable",
        "proposal_executable",
        "table",
    ):
        path = getattr(args, label)
        if not path.is_file():
            raise ValueError(f"{label} is not a regular file: {path}")
    if args.output_root.exists():
        raise ValueError(
            f"output root already exists; refusing to overwrite: {args.output_root}"
        )
    if args.events < 2:
        raise ValueError("at least two events per backend are required")
    if args.proposal_seed is not None and args.proposal_seed <= 0:
        raise ValueError("proposal seed must be positive for reproducible validation")
    if args.cuda_seed is not None and args.cuda_seed <= 0:
        raise ValueError("CUDA seed must be positive for reproducible validation")
    if (
        args.development_emthin is not None
        and not 0.0 <= args.development_emthin <= 1.0
    ):
        raise ValueError("development emthin must be in [0, 1]")
    positive = (
        args.proposal_shards,
        args.proposal_parallelism,
        args.gpu_min_batch,
        args.hadronic_workers,
        args.hadronic_min_batch,
        args.hadronic_max_batch,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("shard, batch and worker counts must be positive")
    if args.proposal_parallelism > args.proposal_shards:
        raise ValueError("proposal parallelism cannot exceed shard count")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    if args.hadronic_target_batch_ms <= 0.0:
        raise ValueError("hadronic target batch time must be positive")


def top_level_consistency_warnings(
    document: dict[str, Any], physics: TaskPhysics
) -> list[str]:
    warnings: list[str] = []
    energy_range = document.get("energy_range")
    if (
        isinstance(energy_range, list)
        and len(energy_range) == 2
        and all(isinstance(value, (int, float)) for value in energy_range)
        and not (
            math.isclose(float(energy_range[0]), physics.energy_GeV)
            and math.isclose(float(energy_range[1]), physics.energy_GeV)
        )
    ):
        warnings.append(
            "top-level energy_range differs from task -E; task args are "
            "authoritative"
        )
    emthin_values = document.get("emthin_values")
    if (
        isinstance(emthin_values, list)
        and emthin_values
        and not any(
            isinstance(value, (int, float))
            and math.isclose(float(value), physics.em_thinning)
            for value in emthin_values
        )
    ):
        warnings.append(
            "top-level emthin_values does not contain the task --emthin; "
            "task args are authoritative"
        )
    return warnings


def translated_command(
    args: argparse.Namespace,
    physics: TaskPhysics,
) -> tuple[list[str], dict[str, Any]]:
    proposal_seed = (
        args.proposal_seed
        if args.proposal_seed is not None
        else (
            physics.configured_seed
            if physics.configured_seed > 0
            else DEFAULT_PROPOSAL_SEED
        )
    )
    cuda_seed = (
        args.cuda_seed
        if args.cuda_seed is not None
        else (
            physics.configured_seed + 10000
            if physics.configured_seed > 0
            else DEFAULT_CUDA_SEED
        )
    )
    if proposal_seed == cuda_seed:
        raise ValueError(
            "independent ensemble seeds must differ; use the paired-control "
            "runner for an equal-seed diagnostic"
        )
    emthin = (
        physics.em_thinning
        if args.development_emthin is None
        else args.development_emthin
    )
    effective_maximum_weight = (
        physics.maximum_weight
        if physics.maximum_weight > 0.0
        else 0.5 * emthin * physics.energy_GeV
    )
    thinning_can_activate_from_unit_weight = (
        emthin > 0.0 and effective_maximum_weight > 1.0
    )
    antenna = (
        args.antenna_file
        if args.antenna_file is not None
        else Path(physics.antenna_file)
    )
    ring = physics.ring
    radio_backend = args.cuda_radio_backend
    if args.shower_only:
        antenna = Path("/dev/null")
        ring = 0
        radio_backend = "cpu"
    if not antenna.exists():
        raise ValueError(f"antenna file does not exist: {antenna}")

    runner = Path(__file__).with_name("run_physics_acceptance.py")
    command = [
        sys.executable,
        str(runner),
        "--executable",
        str(args.cuda_executable.resolve()),
        "--proposal-executable",
        str(args.proposal_executable.resolve()),
        "--table",
        str(args.table.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--label",
        f"config:{physics.task_name}",
        "--energy-gev",
        f"{physics.energy_GeV:.17g}",
        "--events",
        str(args.events),
        "--proposal-seed",
        str(proposal_seed),
        "--cuda-seed",
        str(cuda_seed),
        "--proposal-shards",
        str(args.proposal_shards),
        "--proposal-parallelism",
        str(args.proposal_parallelism),
        "--primary-pdg",
        str(physics.primary_pdg),
        "--zenith-deg",
        f"{physics.zenith_deg:.17g}",
        "--azimuth-deg",
        f"{physics.azimuth_deg:.17g}",
        "--shower-core-x-m",
        f"{physics.shower_core_x_m:.17g}",
        "--shower-core-y-m",
        f"{physics.shower_core_y_m:.17g}",
        "--ring",
        str(ring),
        "--antenna-file",
        str(antenna.resolve()),
        "--em-cut-gev",
        f"{physics.em_cut_GeV:.17g}",
        "--em-thinning",
        f"{emthin:.17g}",
        "--maximum-weight",
        f"{physics.maximum_weight:.17g}",
        "--had-cut-gev",
        f"{physics.had_cut_GeV:.17g}",
        "--mu-cut-gev",
        f"{physics.mu_cut_GeV:.17g}",
        "--tau-cut-gev",
        f"{physics.tau_cut_GeV:.17g}",
        "--cuda-hadronic-backend",
        "fluka-process",
        "--cuda-hadronic-workers",
        str(args.hadronic_workers),
        "--cuda-hadronic-min-batch",
        str(args.hadronic_min_batch),
        "--cuda-hadronic-target-batch-ms",
        f"{args.hadronic_target_batch_ms:.17g}",
        "--cuda-hadronic-max-batch",
        str(args.hadronic_max_batch),
        "--cuda-radio-backend",
        radio_backend,
        "--gpu-device",
        str(args.gpu_device),
        "--gpu-min-batch",
        str(args.gpu_min_batch),
        "--gpu-memory-fraction",
        f"{args.gpu_memory_fraction:.17g}",
        "--gpu-table-tolerance",
        f"{args.gpu_table_tolerance:.17g}",
        "--stability-bootstrap-repetitions",
        str(args.stability_bootstrap_repetitions),
        "--relative-tolerance",
        f"{args.relative_tolerance:.17g}",
        "--sigma-limit",
        f"{args.sigma_limit:.17g}",
    ]
    if args.require_pass:
        command.append("--require-pass")
    mapping = {
        "proposal_seed": proposal_seed,
        "cuda_seed": cuda_seed,
        "events_per_backend": args.events,
        "effective_em_thinning": emthin,
        "effective_maximum_weight": effective_maximum_weight,
        "automatic_maximum_weight": physics.maximum_weight <= 0.0,
        "thinning_can_activate_from_unit_weight": (
            thinning_can_activate_from_unit_weight
        ),
        "em_thinning_overridden": args.development_emthin is not None,
        "shower_only": args.shower_only,
        "effective_ring": ring,
        "effective_antenna_file": str(antenna.resolve()),
        "effective_cuda_radio_backend": radio_backend,
        "production_equivalent": (
            args.development_emthin is None and not args.shower_only
        ),
    }
    return command, mapping


def summarize_hadronic_pool(output_root: Path) -> dict[str, Any]:
    path = output_root / "cuda" / "gpu_em" / "summary.yaml"
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}
    with path.open("r", encoding="utf-8") as source:
        showers = yaml.safe_load(source)
    if not isinstance(showers, dict):
        return {"available": False, "reason": "invalid GPU summary"}

    total_run_ms = 0.0
    execute_ms = 0.0
    final_state_cpu_ms = 0.0
    prepared = 0
    worker_times: dict[int, float] = {}
    worker_requests: dict[int, int] = {}
    cost_flush_predicted_ratios: list[float] = []
    cost_flush_actual_ratios: list[float] = []
    cost_flush_requests: list[int] = []
    complete_showers = 0
    for record in showers.values():
        if not isinstance(record, dict) or record.get("complete") is not True:
            continue
        statistics = record.get("statistics", {})
        if not isinstance(statistics, dict):
            continue
        pool = statistics.get("hadronic_process_pool", {})
        timing = statistics.get("hybrid_timing_ms", {})
        models = statistics.get("hadronic_models", {})
        if not isinstance(pool, dict) or not isinstance(timing, dict):
            continue
        complete_showers += 1
        total_run_ms += float(timing.get("total_run", 0.0))
        execute_ms += float(pool.get("execute_time_ms", 0.0))
        prepared += int(pool.get("prepared_interactions", 0))
        if isinstance(models, dict):
            for label in ("high_energy", "low_energy"):
                model = models.get(label, {})
                if isinstance(model, dict):
                    final_state_cpu_ms += float(
                        model.get("final_state_time_ms", 0.0)
                    )
        loads = pool.get("actual_worker_loads", [])
        if isinstance(loads, list):
            for load in loads:
                if not isinstance(load, dict):
                    continue
                worker = int(load.get("worker_id", -1))
                if worker < 0:
                    continue
                worker_times[worker] = worker_times.get(worker, 0.0) + float(
                    load.get("final_state_time_ms", 0.0)
                )
                worker_requests[worker] = worker_requests.get(worker, 0) + int(
                    load.get("requests", 0)
                )
        flush_loads = pool.get("flush_loads", [])
        if isinstance(flush_loads, list):
            for flush in flush_loads:
                if (
                    not isinstance(flush, dict)
                    or flush.get("trigger") != "estimated_cost"
                ):
                    continue
                predicted = flush.get("predicted_cost_by_worker", [])
                actual = flush.get(
                    "actual_final_state_time_by_worker_ms", []
                )
                if (
                    isinstance(predicted, list)
                    and predicted
                    and all(float(value) > 0.0 for value in predicted)
                ):
                    values = [float(value) for value in predicted]
                    cost_flush_predicted_ratios.append(
                        max(values) / min(values)
                    )
                if (
                    isinstance(actual, list)
                    and actual
                    and all(float(value) > 0.0 for value in actual)
                ):
                    values = [float(value) for value in actual]
                    cost_flush_actual_ratios.append(max(values) / min(values))
                cost_flush_requests.append(int(flush.get("requests", 0)))
    positive_times = [value for value in worker_times.values() if value > 0.0]
    balance_ratio = (
        max(positive_times) / min(positive_times)
        if positive_times
        else None
    )
    share = execute_ms / total_run_ms if total_run_ms > 0.0 else None
    def mean_or_none(values: list[float] | list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "available": complete_showers > 0,
        "complete_showers": complete_showers,
        "hybrid_total_run_ms": total_run_ms,
        "process_pool_execute_wall_ms": execute_ms,
        "process_pool_wall_fraction": share,
        "hadronic_final_state_cpu_ms": final_state_cpu_ms,
        "prepared_interactions": prepared,
        "worker_final_state_time_ms": {
            str(key): worker_times[key] for key in sorted(worker_times)
        },
        "worker_requests": {
            str(key): worker_requests[key] for key in sorted(worker_requests)
        },
        "worker_max_over_min_final_state_time": balance_ratio,
        "cost_triggered_flushes_with_loads": len(cost_flush_requests),
        "cost_flush_mean_requests": mean_or_none(cost_flush_requests),
        "cost_flush_mean_predicted_worker_max_over_min": mean_or_none(
            cost_flush_predicted_ratios
        ),
        "cost_flush_max_predicted_worker_max_over_min": (
            max(cost_flush_predicted_ratios)
            if cost_flush_predicted_ratios
            else None
        ),
        "cost_flush_mean_actual_worker_max_over_min": mean_or_none(
            cost_flush_actual_ratios
        ),
        "cost_flush_max_actual_worker_max_over_min": (
            max(cost_flush_actual_ratios)
            if cost_flush_actual_ratios
            else None
        ),
        "gpu_hadronic_transport_recommended": (
            bool(share is not None and math.isfinite(share) and share >= 0.10)
        ),
        "gpu_hadronic_decision_rule": (
            "recommend only when measured process-pool execute wall is at "
            "least 10% of hybrid shower wall"
        ),
    }


def write_adapter_manifest(
    args: argparse.Namespace,
    physics: TaskPhysics,
    raw_tokens: list[str],
    warnings: list[str],
    command: list[str],
    mapping: dict[str, Any],
    return_code: int | None,
) -> None:
    if not args.output_root.exists():
        return
    run_manifest_path = args.output_root / "run_manifest.json"
    run_manifest = None
    if run_manifest_path.is_file():
        with run_manifest_path.open("r", encoding="utf-8") as source:
            run_manifest = json.load(source)
    manifest = {
        "schema_version": 2,
        "status": (
            "dry_run"
            if return_code is None
            else ("completed" if return_code == 0 else "failed")
        ),
        "execution_status": (
            "dry_run"
            if return_code is None
            else ("completed" if return_code == 0 else "failed")
        ),
        "physics_status": (
            run_manifest.get("physics_status")
            if isinstance(run_manifest, dict)
            else None
        ),
        "source": {
            "config": artifact(args.config),
            "task_index": args.task_index,
            "task_name": physics.task_name,
            "raw_task_args": raw_tokens,
            "parsed_physics": asdict(physics),
            "consistency_warnings": warnings,
        },
        "mapping": mapping,
        "acceptance_runner": artifact(
            Path(__file__).with_name("run_physics_acceptance.py")
        ),
        "adapter": artifact(Path(__file__)),
        "translated_command": command,
        "translated_command_shell": shlex.join(command),
        "return_code": return_code,
        "run_manifest": run_manifest,
        "hadronic_bottleneck": (
            summarize_hadronic_pool(args.output_root)
            if return_code == 0
            else {"available": False}
        ),
    }
    path = args.output_root / "config_acceptance_manifest.json"
    with path.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, allow_nan=False)
        destination.write("\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    physics, document, raw_tokens = parse_task_physics(
        args.config, args.task_index
    )
    warnings = top_level_consistency_warnings(document, physics)
    command, mapping = translated_command(args, physics)
    if (
        mapping["effective_em_thinning"] > 0.0
        and not mapping[
            "thinning_can_activate_from_unit_weight"
        ]
    ):
        warnings.append(
            "effective automatic/explicit maximum weight is not above the "
            "unit initial particle weight; EMThinning cannot activate"
        )
    preview = {
        "parsed_physics": asdict(physics),
        "mapping": mapping,
        "warnings": warnings,
        "command": command,
        "command_shell": shlex.join(command),
    }
    print(json.dumps(preview, indent=2, allow_nan=False))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    write_adapter_manifest(
        args,
        physics,
        raw_tokens,
        warnings,
        command,
        mapping,
        completed.returncode,
    )
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"config acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
