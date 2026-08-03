#!/usr/bin/env python3
"""
Run and compare independent scalar-PROPOSAL and CUDA-EM shower ensembles.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from compare_ensembles import (
    AVAILABLE_KEY_SCALAR_METRICS,
    DEFAULT_KEY_SCALAR_METRICS,
    compare_ensembles,
    concatenate_ensembles,
    extract_ensemble,
    provenance_fingerprint,
    read_validation_provenance,
    VALIDATION_PROVENANCE_FILENAME,
    VALIDATION_PROVENANCE_SCHEMA_VERSION,
)
from analyze_scalar_stability import analyse_frame
from run_performance_acceptance import MOLIERE_CACHE_SUFFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent CPU/CUDA ensembles and compare physics observables."
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument(
        "--proposal-executable",
        type=Path,
        help=(
            "Original scalar-PROPOSAL c8_air_shower executable. When omitted, "
            "--executable is used for both backends."
        ),
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--additional-proposal",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing independent scalar output to pool with the new run; "
            "repeat for shards. Physics configuration is checked exactly."
        ),
    )
    parser.add_argument(
        "--additional-cuda",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing independent CUDA output to pool with the new run; "
            "repeat for shards. Physics configuration is checked exactly."
        ),
    )
    parser.add_argument("--label", default="custom")
    parser.add_argument("--energy-gev", type=float, default=1.0e3)
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--proposal-seed", type=int, default=41001)
    parser.add_argument("--cuda-seed", type=int, default=51001)
    parser.add_argument(
        "--paired-seed-control",
        action="store_true",
        help=(
            "Use the same seed in one unsharded CPU and CUDA run as a "
            "diagnostic control. The resulting showers are not expected to "
            "be event-wise identical once the EM random streams diverge."
        ),
    )
    parser.add_argument(
        "--proposal-shards",
        type=int,
        default=1,
        help="Split the scalar ensemble into this many independent processes.",
    )
    parser.add_argument(
        "--proposal-parallelism",
        type=int,
        default=1,
        help="Maximum simultaneously running scalar shards.",
    )
    parser.add_argument(
        "--resume-completed-proposal",
        action="store_true",
        help=(
            "Reuse already completed scalar shard directories in an existing "
            "output root after verifying their seed, event count and exact "
            "command. CUDA output and final manifests must not yet exist."
        ),
    )
    parser.add_argument(
        "--resume-completed-cuda",
        action="store_true",
        help=(
            "Reuse an already closed CUDA directory after exact seed, event "
            "count and command verification. This requires "
            "--resume-completed-proposal and is intended to recover an "
            "interrupted post-processing stage without rerunning showers."
        ),
    )
    parser.add_argument(
        "--skip-proposal-run",
        action="store_true",
        help=(
            "Do not launch a new scalar ensemble. At least one "
            "--additional-proposal source is required; this is intended for "
            "pooling completed CPU shards while producing a CUDA supplement."
        ),
    )
    parser.add_argument(
        "--allow-mixed-proposal-builds",
        action="store_true",
        help=(
            "Permit additional scalar shards from separately built original "
            "CORSIKA executables. Provenance is retained per source and the "
            "canonical physics configuration must still match exactly."
        ),
    )
    parser.add_argument(
        "--allow-mixed-cuda-builds",
        action="store_true",
        help=(
            "Permit CUDA shards from separately built executables while "
            "requiring the same rate-table SHA-256 and exact canonical "
            "physics configuration. All executable fingerprints are retained."
        ),
    )
    parser.add_argument(
        "--overlap-backends",
        action="store_true",
        help=(
            "Run the CUDA ensemble concurrently with scalar shards. "
            "Use only for physics, never for performance timing."
        ),
    )
    parser.add_argument("--primary-pdg", type=int, default=11)
    parser.add_argument(
        "--primary-z",
        type=int,
        default=None,
        help=(
            "Nuclear-primary charge Z. When set, --primary-a is required and "
            "the c8_air_shower -Z/-A interface replaces --primary-pdg."
        ),
    )
    parser.add_argument(
        "--primary-a",
        type=int,
        default=None,
        help="Nuclear-primary mass number A; requires --primary-z.",
    )
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--em-cut-gev", type=float, default=0.5e-3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-4)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=100.0,
        help=(
            "Explicit EM thinning maximum weight; zero omits --max-weight "
            "and preserves c8_air_shower's automatic Kobal value."
        ),
    )
    parser.add_argument(
        "--non-em-cut-gev",
        type=float,
        default=None,
        help=(
            "Legacy shorthand setting hadron, muon and tau cuts together. "
            "When omitted, the three dedicated cut options are used."
        ),
    )
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--shower-core-x-m", type=float, default=0.0)
    parser.add_argument("--shower-core-y-m", type=float, default=0.0)
    parser.add_argument(
        "--cuda-hadronic-backend",
        choices=("scalar", "fluka-process"),
        default="scalar",
        help=(
            "Low-energy hadronic implementation used only by the CUDA arm; "
            "both choices use the same FLUKA physics."
        ),
    )
    parser.add_argument("--cuda-hadronic-workers", type=int, default=4)
    parser.add_argument("--cuda-hadronic-min-batch", type=int, default=64)
    parser.add_argument(
        "--cuda-hadronic-target-batch-ms", type=float, default=5.0
    )
    parser.add_argument("--cuda-hadronic-max-batch", type=int, default=256)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument(
        "--antenna-file",
        type=Path,
        default=Path("/dev/null"),
        help=(
            "Observer file for shower-only acceptance. /dev/null disables "
            "radio when --ring is zero."
        ),
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--cuda-detailed-stage-timing",
        action="store_true",
        help=(
            "Enable CUDA-event stage profiling. It is disabled by default so "
            "production performance runs do not pay profiling overhead."
        ),
    )
    parser.add_argument(
        "--cuda-radio-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Radio projection used by the CUDA ensemble. Keep cpu for "
            "shower-only acceptance; select cuda for independent "
            "original/CUDA radio ensembles."
        ),
    )
    parser.add_argument(
        "--gpu-radio-field-limit",
        type=float,
        default=1.0,
        help=(
            "Deterministic CUDA radio fixed-point field limit in V/m."
        ),
    )
    parser.add_argument(
        "--gpu-radio-track-diagnostics",
        action="store_true",
        help=(
            "Collect scalar-compatible CUDA electron/positron track "
            "diagnostics. This is a validation mode with extra GPU "
            "reduction overhead."
        ),
    )
    parser.add_argument(
        "--radio-sampling-rate-ghz",
        type=float,
        default=1.0,
        help="CoREAS/ZHS time-domain sampling rate for both ensembles.",
    )
    parser.add_argument(
        "--radio-window-duration-ns",
        type=float,
        default=400.0,
        help="CoREAS/ZHS observer window duration for both ensembles.",
    )
    parser.add_argument(
        "--radio-pretrigger-ns",
        type=float,
        default=10.0,
        help="Radio window pretrigger before geometrical arrival.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument("--active-fraction", type=float, default=1.0e-4)
    parser.add_argument("--minimum-bin-pass-fraction", type=float, default=0.95)
    parser.add_argument(
        "--stability-bootstrap-repetitions",
        type=int,
        default=20000,
        help=(
            "Bootstrap repetitions for the automatic scalar-stability "
            "diagnostic; use zero to disable it."
        ),
    )
    parser.add_argument(
        "--stability-seed",
        type=int,
        default=8052026,
        help="Deterministic seed for the scalar-stability bootstrap.",
    )
    parser.add_argument(
        "--key-scalar",
        action="append",
        choices=AVAILABLE_KEY_SCALAR_METRICS,
        help=(
            "Scalar mean receiving both 1%% and statistical gates. "
            "Repeat to replace the default core list."
        ),
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 when the physics acceptance gates fail.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.executable.is_file():
        raise ValueError(f"executable is not a regular file: {args.executable}")
    proposal_executable = getattr(args, "proposal_executable", None)
    if proposal_executable is not None and not proposal_executable.is_file():
        raise ValueError(
            "proposal executable is not a regular file: "
            f"{proposal_executable}"
        )
    if not args.table.is_file():
        raise ValueError(f"rate table is not a regular file: {args.table}")
    if not args.antenna_file.exists():
        raise ValueError(f"antenna file does not exist: {args.antenna_file}")
    resume_proposal = getattr(args, "resume_completed_proposal", False)
    resume_cuda = getattr(args, "resume_completed_cuda", False)
    skip_proposal = getattr(args, "skip_proposal_run", False)
    if args.output_root.exists():
        if not (resume_proposal or resume_cuda):
            raise ValueError(
                "output root already exists; refusing to overwrite: "
                f"{args.output_root}"
            )
        if not args.output_root.is_dir():
            raise ValueError(
                f"resume output root is not a directory: {args.output_root}"
            )
        forbidden = [
            path
            for path in (
                args.output_root / "run_manifest.json",
                args.output_root / "comparison.json",
            )
            if path.exists()
        ]
        if not resume_cuda and (args.output_root / "cuda").exists():
            forbidden.append(args.output_root / "cuda")
        if forbidden:
            raise ValueError(
                "resume requires an unfinished CPU-only output root; found: "
                + ", ".join(str(path) for path in forbidden)
            )
        if resume_cuda and not (args.output_root / "cuda").is_dir():
            raise ValueError(
                "--resume-completed-cuda requires an existing cuda directory"
            )
    for label, roots in (
        ("additional PROPOSAL output", args.additional_proposal),
        ("additional CUDA output", args.additional_cuda),
    ):
        for root in roots:
            if not root.is_dir():
                raise ValueError(f"{label} is not a directory: {root}")
    if skip_proposal and not args.additional_proposal:
        raise ValueError(
            "--skip-proposal-run requires at least one "
            "--additional-proposal source"
        )
    if skip_proposal and resume_proposal:
        raise ValueError(
            "--skip-proposal-run cannot be combined with "
            "--resume-completed-proposal"
        )
    if resume_cuda and not resume_proposal:
        raise ValueError(
            "--resume-completed-cuda requires --resume-completed-proposal"
        )
    if resume_cuda and skip_proposal:
        raise ValueError(
            "--resume-completed-cuda cannot be combined with "
            "--skip-proposal-run"
        )
    if args.energy_gev <= 0.0 or args.events < 2:
        raise ValueError("energy must be positive and at least two events are required")
    primary_z = getattr(args, "primary_z", None)
    primary_a = getattr(args, "primary_a", None)
    if (primary_z is None) != (primary_a is None):
        raise ValueError("nuclear primary requires both --primary-z and --primary-a")
    if primary_z is not None and (
        primary_z < 0
        or primary_z > 26
        or primary_a < 1
        or primary_a > 58
        or primary_z > primary_a
    ):
        raise ValueError("nuclear primary requires 0 <= Z <= A, Z <= 26, A <= 58")
    if (
        args.proposal_shards <= 0
        or args.proposal_shards > args.events
        or args.proposal_parallelism <= 0
        or args.proposal_parallelism >
            args.proposal_shards
    ):
        raise ValueError(
            "proposal shards must be in [1, events] and parallelism in [1, shards]"
        )
    if (resume_proposal or resume_cuda or skip_proposal) and getattr(
        args, "overlap_backends", False
    ):
        raise ValueError(
            "a skipped or resumed proposal run cannot overlap CPU and CUDA backends"
        )
    paired_seed_control = getattr(args, "paired_seed_control", False)
    overlapping_seed = (
        args.proposal_seed <= args.cuda_seed <
        args.proposal_seed + args.proposal_shards
    )
    if overlapping_seed and not paired_seed_control:
        raise ValueError(
            "proposal shard seeds and CUDA seed must differ for independent ensembles"
        )
    if paired_seed_control and (
        args.proposal_seed != args.cuda_seed
        or args.proposal_shards != 1
        or args.additional_proposal
        or args.additional_cuda
    ):
        raise ValueError(
            "paired seed control requires equal seeds, one proposal shard, "
            "and no additional ensembles"
        )
    non_em_cut = getattr(args, "non_em_cut_gev", None)
    dedicated_cuts = (
        getattr(args, "had_cut_gev", 0.3),
        getattr(args, "mu_cut_gev", 0.3),
        getattr(args, "tau_cut_gev", 0.3),
    )
    if (
        args.em_cut_gev <= 0.0
        or (non_em_cut is not None and non_em_cut <= 0.0)
        or any(value <= 0.0 for value in dedicated_cuts)
    ):
        raise ValueError("particle cuts must be positive")
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("EM thinning fraction must be in [0, 1]")
    if args.maximum_weight < 0.0:
        raise ValueError("maximum weight must be non-negative")
    if (
        getattr(args, "cuda_hadronic_workers", 4) <= 0
        or getattr(args, "cuda_hadronic_min_batch", 64) <= 0
        or getattr(args, "cuda_hadronic_target_batch_ms", 5.0) <= 0.0
        or getattr(args, "cuda_hadronic_max_batch", 256) <= 0
    ):
        raise ValueError("CUDA hadronic scheduling controls must be positive")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    if (
        not math.isfinite(
            getattr(args, "gpu_radio_field_limit", 1.0)
        )
        or getattr(args, "gpu_radio_field_limit", 1.0) <= 0.0
    ):
        raise ValueError(
            "GPU radio fixed-point field limit must be finite and positive"
        )
    if (
        not math.isfinite(
            getattr(args, "radio_sampling_rate_ghz", 1.0)
        )
        or not 0.1
        <= getattr(args, "radio_sampling_rate_ghz", 1.0)
        <= 1000.0
        or not math.isfinite(
            getattr(args, "radio_window_duration_ns", 400.0)
        )
        or getattr(args, "radio_window_duration_ns", 400.0) <= 0.0
        or not math.isfinite(
            getattr(args, "radio_pretrigger_ns", 10.0)
        )
        or getattr(args, "radio_pretrigger_ns", 10.0) < 0.0
    ):
        raise ValueError("radio sampling/window controls are invalid")
    legacy_proposal = (
        getattr(args, "proposal_executable", None) is not None
        and args.proposal_executable != args.executable
    )
    if legacy_proposal and (
        getattr(args, "radio_sampling_rate_ghz", 1.0) != 1.0
        or getattr(args, "radio_window_duration_ns", 400.0) != 400.0
        or getattr(args, "radio_pretrigger_ns", 10.0) != 10.0
    ):
        raise ValueError(
            "the original proposal executable has fixed 1 GHz, 400 ns, "
            "10 ns radio controls; use a configurable scalar reference "
            "binary for other sampling settings"
        )
    if not 0.0 < args.relative_tolerance < 1.0:
        raise ValueError("relative tolerance must be in (0, 1)")
    if args.sigma_limit <= 0.0:
        raise ValueError("sigma limit must be positive")
    if not 0.0 <= args.active_fraction < 1.0:
        raise ValueError("active fraction must be in [0, 1)")
    if not 0.0 < args.minimum_bin_pass_fraction <= 1.0:
        raise ValueError("minimum bin pass fraction must be in (0, 1]")
    if (
        args.stability_bootstrap_repetitions != 0
        and args.stability_bootstrap_repetitions < 100
    ):
        raise ValueError(
            "stability bootstrap repetitions must be zero or at least 100"
        )


def single_thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


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


def validation_identity(
    args: argparse.Namespace,
    executable: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_PROVENANCE_SCHEMA_VERSION,
        "executable": artifact_identity(
            args.executable if executable is None else executable
        ),
        "table": artifact_identity(args.table),
        "runner": artifact_identity(Path(__file__)),
    }


def output_provenance(
    identity: dict[str, Any],
    backend: str,
    command: list[str],
) -> dict[str, Any]:
    if backend not in {"proposal", "cuda"}:
        raise ValueError(f"unsupported validation backend: {backend}")
    encoded_command = json.dumps(
        command,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": identity["schema_version"],
        "backend": backend,
        "executable": dict(identity["executable"]),
        "table": (
            dict(identity["table"])
            if backend == "cuda"
            else None
        ),
        "runner": dict(identity["runner"]),
        "command": list(command),
        "command_sha256": hashlib.sha256(
            encoded_command
        ).hexdigest(),
    }


def write_output_provenance(
    output: Path,
    identity: dict[str, Any],
    backend: str,
    command: list[str],
) -> None:
    if not output.is_dir():
        raise RuntimeError(
            f"completed {backend} output directory is missing: {output}"
        )
    path = output / VALIDATION_PROVENANCE_FILENAME
    if path.exists():
        raise RuntimeError(
            f"refusing to overwrite validation provenance: {path}"
        )
    with path.open("x", encoding="utf-8") as destination:
        json.dump(
            output_provenance(identity, backend, command),
            destination,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        destination.write("\n")


def verify_artifact_identity(
    expected: dict[str, Any],
    label: str,
) -> None:
    observed = artifact_identity(Path(expected["path"]))
    if (
        observed["size_bytes"] != expected["size_bytes"]
        or observed["sha256"] != expected["sha256"]
    ):
        raise RuntimeError(
            f"{label} changed while validation showers were running; "
            "outputs are not eligible for pooling"
        )


def validate_additional_provenance(
    args: argparse.Namespace,
    proposal_identity: dict[str, Any],
    cuda_identity: dict[str, Any] | None = None,
) -> None:
    if cuda_identity is None:
        cuda_identity = proposal_identity
    for backend, roots in (
        ("proposal", args.additional_proposal),
        ("cuda", args.additional_cuda),
    ):
        identity = proposal_identity if backend == "proposal" else cuda_identity
        expected = provenance_fingerprint(
            output_provenance(identity, backend, [])
        )
        for root in roots:
            provenance = read_validation_provenance(
                root,
                backend,
            )
            observed = provenance_fingerprint(provenance)
            if (
                backend == "proposal"
                and getattr(args, "allow_mixed_proposal_builds", False)
            ):
                continue
            if (
                backend == "cuda"
                and getattr(args, "allow_mixed_cuda_builds", False)
            ):
                if observed[:2] != expected[:2] or observed[3] != expected[3]:
                    raise ValueError(
                        "additional CUDA output schema/backend/table "
                        f"provenance differs from the requested run: {root}"
                    )
                continue
            if observed != expected:
                raise ValueError(
                    f"additional {backend} output build/table provenance "
                    f"differs from the requested run: {root}"
                )


def common_command(
    args: argparse.Namespace,
    executable: Path,
    output: Path,
    seed: int,
    events: int | None = None,
    *,
    include_radio_controls: bool = True,
) -> list[str]:
    non_em_cut = getattr(args, "non_em_cut_gev", None)
    had_cut = (
        non_em_cut
        if non_em_cut is not None
        else getattr(args, "had_cut_gev", 0.3)
    )
    mu_cut = (
        non_em_cut
        if non_em_cut is not None
        else getattr(args, "mu_cut_gev", 0.3)
    )
    tau_cut = (
        non_em_cut
        if non_em_cut is not None
        else getattr(args, "tau_cut_gev", 0.3)
    )
    primary_z = getattr(args, "primary_z", None)
    primary_a = getattr(args, "primary_a", None)
    primary_arguments = (
        ["-p", str(args.primary_pdg)]
        if primary_z is None
        else ["-Z", str(primary_z), "-A", str(primary_a)]
    )
    command = [
        str(executable),
        *primary_arguments,
        "-E",
        f"{args.energy_gev:.17g}",
        "-N",
        str(args.events if events is None else events),
        "-f",
        str(output),
        "--seed",
        str(seed),
        "--zenith",
        f"{args.zenith_deg:.17g}",
        "--azimuth",
        f"{args.azimuth_deg:.17g}",
        "--shower-core-x",
        f"{getattr(args, 'shower_core_x_m', 0.0):.17g}",
        "--shower-core-y",
        f"{getattr(args, 'shower_core_y_m', 0.0):.17g}",
        "--ring",
        str(args.ring),
        "--antenna-file",
        str(args.antenna_file),
        "--emcut",
        f"{args.em_cut_gev:.17g}",
        "--emthin",
        f"{args.em_thinning:.17g}",
        "--hadcut",
        f"{had_cut:.17g}",
        "--mucut",
        f"{mu_cut:.17g}",
        "--taucut",
        f"{tau_cut:.17g}",
        "--verbosity",
        "warn",
    ]
    if include_radio_controls:
        insertion = command.index("--emcut")
        command[insertion:insertion] = [
            "--radio-sampling-rate-ghz",
            f"{getattr(args, 'radio_sampling_rate_ghz', 1.0):.17g}",
            "--radio-window-duration-ns",
            f"{getattr(args, 'radio_window_duration_ns', 400.0):.17g}",
            "--radio-pretrigger-ns",
            f"{getattr(args, 'radio_pretrigger_ns', 10.0):.17g}",
        ]
    if args.maximum_weight > 0.0:
        insertion = command.index("--hadcut")
        command[insertion:insertion] = [
            "--max-weight",
            f"{args.maximum_weight:.17g}",
        ]
    return command


def proposal_command(
    args: argparse.Namespace,
    output: Path,
    seed: int | None = None,
    events: int | None = None,
) -> list[str]:
    has_refactor_cli = (
        getattr(args, "proposal_executable", None) is None
        or args.proposal_executable == args.executable
    )
    return common_command(
        args,
        getattr(args, "proposal_executable", None) or args.executable,
        output,
        args.proposal_seed if seed is None else seed,
        events,
        include_radio_controls=has_refactor_cli,
    ) + (
        [
            "--em-backend",
            "proposal",
            "--radio-backend",
            "cpu",
        ]
        if has_refactor_cli
        else []
    )


def cuda_command(
    args: argparse.Namespace,
    output: Path,
    *,
    warmup: bool = False,
) -> list[str]:
    radio_backend = getattr(
        args, "cuda_radio_backend", "cpu"
    )
    legacy_proposal = (
        getattr(args, "proposal_executable", None) is not None
        and args.proposal_executable != args.executable
    )
    command = common_command(
        args,
        args.executable,
        output,
        args.cuda_seed,
        include_radio_controls=not legacy_proposal,
    ) + [
        "--em-backend",
        "cuda",
        "--radio-backend",
        radio_backend,
        "--gpu-device",
        str(args.gpu_device),
        "--gpu-min-batch",
        str(args.gpu_min_batch),
        "--gpu-memory-fraction",
        f"{args.gpu_memory_fraction:.17g}",
        "--gpu-table-cache",
        str(args.table),
        "--gpu-table-tolerance",
        f"{args.gpu_table_tolerance:.17g}",
    ]
    if getattr(args, "cuda_detailed_stage_timing", False):
        command.append("--gpu-detailed-stage-timing")
    if radio_backend == "cuda":
        command.extend(
            [
                "--gpu-radio-field-limit",
                f"{getattr(args, 'gpu_radio_field_limit', 1.0):.17g}",
            ]
        )
        if getattr(args, "gpu_radio_track_diagnostics", False):
            command.append("--gpu-radio-track-diagnostics")
    if getattr(args, "cuda_hadronic_backend", "scalar") == "fluka-process":
        command.extend(
            [
                "--hadronic-backend",
                "fluka-process",
                "--hadronic-workers",
                str(getattr(args, "cuda_hadronic_workers", 4)),
                "--hadronic-min-batch",
                str(getattr(args, "cuda_hadronic_min_batch", 64)),
                "--hadronic-target-batch-ms",
                (
                    f"{getattr(args, 'cuda_hadronic_target_batch_ms', 5.0):.17g}"
                ),
                "--hadronic-max-batch",
                str(getattr(args, "cuda_hadronic_max_batch", 256)),
            ]
        )
    if warmup:
        command[command.index("-E") + 1] = "10"
        command[command.index("-N") + 1] = "1"
    return command


def run_command(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{shlex.join(command)}; see {log_path}"
        )
    return elapsed


def ensure_hot_cache(
    args: argparse.Namespace,
    root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    cache_path = Path(str(args.table) + MOLIERE_CACHE_SUFFIX)
    if cache_path.is_file():
        return {
            "performed": False,
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
        }
    output = root / "cache_warmup"
    command = cuda_command(args, output, warmup=True)
    elapsed = run_command(command, root / "cache_warmup.log", environment)
    if not cache_path.is_file():
        raise RuntimeError(f"warmup did not create Moliere cache: {cache_path}")
    return {
        "performed": True,
        "path": str(cache_path),
        "bytes": cache_path.stat().st_size,
        "wall_seconds": elapsed,
        "command": command,
    }


def split_event_count(total: int, shards: int) -> list[int]:
    base, remainder = divmod(total, shards)
    return [
        base + (1 if index < remainder else 0)
        for index in range(shards)
    ]


def completed_proposal_shard(
    output: Path,
    command: list[str],
    expected_seed: int,
    expected_events: int,
) -> Optional[dict[str, Any]]:
    """Validate a completed scalar shard and recover its timing metadata."""
    if not output.exists():
        return None
    summary_path = output / "summary.yaml"
    config_path = output / "config.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        raise ValueError(
            "existing proposal shard is incomplete; move it aside before "
            f"resuming: {output}"
        )
    with summary_path.open("r", encoding="utf-8") as source:
        summary = yaml.safe_load(source)
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise ValueError(f"invalid YAML metadata in completed shard: {output}")
    observed_events = int(summary.get("showers", -1))
    observed_seed = int(summary.get("seed", -1))
    observed_command = shlex.split(str(config.get("args", "")))
    if observed_events != expected_events or observed_seed != expected_seed:
        raise ValueError(
            "completed shard seed/event count differs from requested resume "
            f"configuration: {output}"
        )
    if observed_command != command:
        raise ValueError(
            "completed shard command differs from requested resume "
            f"configuration: {output}"
        )
    try:
        elapsed = float(summary["runtime_raw"])
        started = datetime.strptime(
            str(summary["start time"]),
            "%Y-%m-%dT%H:%M:%S%z",
        )
        ended = datetime.strptime(
            str(summary["end time"]),
            "%Y-%m-%dT%H:%M:%S%z",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"completed shard has invalid timing metadata: {output}"
        ) from error
    if elapsed <= 0.0 or ended < started:
        raise ValueError(
            f"completed shard has non-positive timing metadata: {output}"
        )
    return {
        "elapsed_seconds": elapsed,
        "start": started,
        "end": ended,
    }


def run_proposal_shards(
    args: argparse.Namespace,
    root: Path,
    environment: dict[str, str],
) -> tuple[list[Path], list[list[str]], list[float], float]:
    counts = split_event_count(
        args.events,
        args.proposal_shards,
    )
    outputs = [
        (
            root / "proposal"
            if args.proposal_shards == 1
            else root /
                f"proposal_shard_{index:03d}"
        )
        for index in range(args.proposal_shards)
    ]
    commands = [
        proposal_command(
            args,
            output,
            seed=args.proposal_seed + index,
            events=counts[index],
        )
        for index, output in enumerate(outputs)
    ]
    logs = [
        (
            root / "proposal.log"
            if args.proposal_shards == 1
            else root /
                f"proposal_shard_{index:03d}.log"
        )
        for index in range(args.proposal_shards)
    ]
    elapsed = [0.0] * args.proposal_shards
    reused: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    resume = getattr(args, "resume_completed_proposal", False)
    for index in range(args.proposal_shards):
        metadata = (
            completed_proposal_shard(
                outputs[index],
                commands[index],
                args.proposal_seed + index,
                counts[index],
            )
            if resume
            else None
        )
        if metadata is None:
            pending.append(index)
        else:
            reused[index] = metadata
            elapsed[index] = metadata["elapsed_seconds"]

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.proposal_parallelism
    ) as executor:
        futures = {
            executor.submit(
                run_command,
                commands[index],
                logs[index],
                environment,
            ): index
            for index in pending
        }
        for future in concurrent.futures.as_completed(
            futures
        ):
            index = futures[future]
            elapsed[index] = future.result()
    pending_wall = time.perf_counter() - started
    if reused and not pending:
        group_wall = (
            max(item["end"] for item in reused.values())
            - min(item["start"] for item in reused.values())
        ).total_seconds()
        group_wall_source = "summary timestamp envelope"
    else:
        group_wall = pending_wall
        group_wall_source = "current invocation wall clock"
    args._proposal_resume_metadata = {
        "enabled": resume,
        "reused_shards": sorted(reused),
        "executed_shards": pending,
        "reused_elapsed_source": (
            "summary.yaml runtime_raw"
            if reused
            else None
        ),
        "group_wall_source": group_wall_source,
    }
    return outputs, commands, elapsed, group_wall


def write_results(
    root: Path,
    proposal: Any,
    cuda: Any,
    report: dict[str, Any],
    curve_rows: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    combined_scalars = pd.concat(
        [
            proposal.scalars.assign(backend="proposal"),
            cuda.scalars.assign(backend="cuda"),
        ]
    ).reset_index()
    combined_scalars.to_csv(root / "per_shower_observables.csv", index=False)
    curve_rows.to_csv(root / "curve_comparison.csv", index=False)
    with (root / "comparison.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    with (root / "run_manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, allow_nan=False)
        destination.write("\n")


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    args.executable = args.executable.resolve()
    args.proposal_executable = (
        args.executable
        if args.proposal_executable is None
        else args.proposal_executable.resolve()
    )
    args.table = args.table.resolve()
    args.antenna_file = args.antenna_file.resolve()
    args.output_root = args.output_root.resolve()
    args.additional_proposal = [
        path.resolve()
        for path in args.additional_proposal
    ]
    args.additional_cuda = [
        path.resolve()
        for path in args.additional_cuda
    ]
    proposal_identity = validation_identity(
        args, args.proposal_executable
    )
    cuda_identity = validation_identity(args, args.executable)
    validate_additional_provenance(
        args,
        proposal_identity,
        cuda_identity,
    )
    args.output_root.mkdir(
        parents=True,
        exist_ok=getattr(args, "resume_completed_proposal", False),
    )
    environment = single_thread_environment()
    flupro = environment.get("FLUPRO")

    cache = ensure_hot_cache(args, args.output_root, environment)
    cuda_output = args.output_root / "cuda"
    cuda = cuda_command(args, cuda_output)
    skip_proposal = getattr(args, "skip_proposal_run", False)
    resume_cuda = getattr(args, "resume_completed_cuda", False)
    if args.overlap_backends:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            proposal_future = executor.submit(
                run_proposal_shards,
                args,
                args.output_root,
                environment,
            )
            cuda_future = executor.submit(
                run_command,
                cuda,
                args.output_root / "cuda.log",
                environment,
            )
            (
                proposal_outputs,
                proposal_commands,
                proposal_walls,
                proposal_group_wall,
            ) = proposal_future.result()
            cuda_wall = cuda_future.result()
    elif skip_proposal:
        proposal_outputs = []
        proposal_commands = []
        proposal_walls = []
        proposal_group_wall = 0.0
        args._proposal_resume_metadata = {
            "enabled": False,
            "skipped": True,
            "reused_shards": [],
            "executed_shards": [],
            "reused_elapsed_source": None,
            "group_wall_source": "skipped; additional sources only",
        }
        cuda_wall = run_command(
            cuda,
            args.output_root / "cuda.log",
            environment,
        )
    else:
        (
            proposal_outputs,
            proposal_commands,
            proposal_walls,
            proposal_group_wall,
        ) = run_proposal_shards(
            args,
            args.output_root,
            environment,
        )
        if resume_cuda:
            cuda_metadata = completed_proposal_shard(
                cuda_output,
                cuda,
                args.cuda_seed,
                args.events,
            )
            if cuda_metadata is None:
                raise ValueError(
                    "existing CUDA output is incomplete and cannot be resumed"
                )
            cuda_wall = float(cuda_metadata["elapsed_seconds"])
            args._cuda_resume_metadata = {
                "enabled": True,
                "elapsed_source": "summary.yaml runtime_raw",
                "start": cuda_metadata["start"].isoformat(),
                "end": cuda_metadata["end"].isoformat(),
            }
        else:
            cuda_wall = run_command(
                cuda,
                args.output_root / "cuda.log",
                environment,
            )

    verify_artifact_identity(
        proposal_identity["executable"],
        "scalar-PROPOSAL c8_air_shower executable",
    )
    verify_artifact_identity(
        cuda_identity["executable"],
        "CUDA c8_air_shower executable",
    )
    verify_artifact_identity(
        cuda_identity["table"],
        "CUDA EM table",
    )
    for output, command in zip(
        proposal_outputs,
        proposal_commands,
    ):
        write_output_provenance(
            output,
            proposal_identity,
            "proposal",
            command,
        )
    write_output_provenance(
        cuda_output,
        cuda_identity,
        "cuda",
        cuda,
    )

    proposal_ensemble = concatenate_ensembles(
        "proposal",
        [
            extract_ensemble(
                "proposal",
                output,
                expect_gpu=False,
            )
            for output in proposal_outputs
        ] + [
            extract_ensemble(
                "proposal",
                output,
                expect_gpu=False,
            )
            for output in args.additional_proposal
        ],
        allow_mixed_provenance=getattr(
            args, "allow_mixed_proposal_builds", False
        ),
    )
    cuda_ensemble = concatenate_ensembles(
        "cuda",
        [
            extract_ensemble(
                "cuda",
                cuda_output,
                expect_gpu=True,
            )
        ] + [
            extract_ensemble(
                "cuda",
                output,
                expect_gpu=True,
            )
            for output in args.additional_cuda
        ],
        allow_mixed_provenance=getattr(
            args, "allow_mixed_cuda_builds", False
        ),
    )
    report, curve_rows = compare_ensembles(
        proposal_ensemble,
        cuda_ensemble,
        args.relative_tolerance,
        args.sigma_limit,
        args.active_fraction,
        args.minimum_bin_pass_fraction,
        tuple(args.key_scalar)
        if args.key_scalar
        else DEFAULT_KEY_SCALAR_METRICS,
        allow_cross_build_reference=(
            proposal_identity["executable"]["sha256"]
            != cuda_identity["executable"]["sha256"]
        ),
    )
    manifest = {
        "label": args.label,
        "configuration": {
            "energy_GeV": args.energy_gev,
            "events_per_backend": args.events,
            "combined_proposal_events":
                len(proposal_ensemble.showers),
            "combined_cuda_events":
                len(cuda_ensemble.showers),
            "proposal_seed": args.proposal_seed,
            "cuda_seed": args.cuda_seed,
            "paired_seed_control": getattr(
                args, "paired_seed_control", False
            ),
            "proposal_shards":
                args.proposal_shards,
            "proposal_parallelism":
                args.proposal_parallelism,
            "overlap_backends":
                args.overlap_backends,
            "skip_proposal_run": skip_proposal,
            "allow_mixed_proposal_builds": getattr(
                args, "allow_mixed_proposal_builds", False
            ),
            "allow_mixed_cuda_builds": getattr(
                args, "allow_mixed_cuda_builds", False
            ),
            "resume_completed_cuda": resume_cuda,
            "primary_pdg": (
                args.primary_pdg
                if getattr(args, "primary_z", None) is None
                else None
            ),
            "primary_Z": getattr(args, "primary_z", None),
            "primary_A": getattr(args, "primary_a", None),
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "non_em_cut_GeV": args.non_em_cut_gev,
            "ring": args.ring,
            "antenna_file": str(args.antenna_file),
            "cuda_radio_backend": getattr(
                args, "cuda_radio_backend", "cpu"
            ),
            "cuda_detailed_stage_timing": getattr(
                args, "cuda_detailed_stage_timing", False
            ),
            "gpu_radio_field_limit_V_per_m": getattr(
                args, "gpu_radio_field_limit", 1.0
            ),
            "gpu_radio_track_diagnostics": getattr(
                args, "gpu_radio_track_diagnostics", False
            ),
            "radio_sampling_rate_GHz": getattr(
                args, "radio_sampling_rate_ghz", 1.0
            ),
            "radio_window_duration_ns": getattr(
                args, "radio_window_duration_ns", 400.0
            ),
            "radio_pretrigger_ns": getattr(
                args, "radio_pretrigger_ns", 10.0
            ),
            "stability_bootstrap_repetitions":
                args.stability_bootstrap_repetitions,
            "stability_seed": args.stability_seed,
            "key_scalar_metrics": (
                args.key_scalar
                if args.key_scalar
                else list(
                    DEFAULT_KEY_SCALAR_METRICS
                )
            ),
            "single_thread_environment": {
                key: environment[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "fluka_environment": {
                "FLUPRO": flupro,
                "FLUPRO_is_directory": bool(
                    flupro and Path(flupro).is_dir()
                ),
            },
        },
        "cache": cache,
        "artifact_identity": {
            "proposal": proposal_identity,
            "cuda": cuda_identity,
        },
        "commands": {
            "proposal": proposal_commands,
            "cuda": cuda,
        },
        "additional_sources": {
            "proposal": [
                str(path)
                for path in args.additional_proposal
            ],
            "cuda": [
                str(path)
                for path in args.additional_cuda
            ],
        },
        "external_wall_seconds": {
            "proposal_shards": proposal_walls,
            "proposal_group": proposal_group_wall,
            "cuda": cuda_wall,
        },
        "proposal_resume": getattr(
            args,
            "_proposal_resume_metadata",
            {
                "enabled": False,
                "reused_shards": [],
                "executed_shards": list(
                    range(args.proposal_shards)
                ),
                "reused_elapsed_source": None,
                "group_wall_source": "current invocation wall clock",
            },
        ),
        "cuda_resume": getattr(
            args,
            "_cuda_resume_metadata",
            {
                "enabled": False,
                "elapsed_source": "current invocation wall clock",
            },
        ),
        "physics_status": report["status"],
    }
    write_results(
        args.output_root,
        proposal_ensemble,
        cuda_ensemble,
        report,
        curve_rows,
        manifest,
    )
    if args.stability_bootstrap_repetitions != 0:
        selected_key_scalars = (
            list(args.key_scalar)
            if args.key_scalar
            else list(DEFAULT_KEY_SCALAR_METRICS)
        )
        stability_metrics = list(
            dict.fromkeys(
                selected_key_scalars
                + [
                    "ground_radius_mean_m",
                    "ground_radius_rms_m",
                    "ground_time_rms_s",
                ]
            )
        )
        per_shower = pd.read_csv(
            args.output_root / "per_shower_observables.csv"
        )
        stability = analyse_frame(
            per_shower,
            stability_metrics,
            relative_tolerance=args.relative_tolerance,
            sigma_limit=args.sigma_limit,
            confidence=0.95,
            bootstrap_repetitions=
                args.stability_bootstrap_repetitions,
            seed=args.stability_seed,
        )
        with (
            args.output_root / "scalar_stability.json"
        ).open("w", encoding="utf-8") as destination:
            json.dump(
                stability,
                destination,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            destination.write("\n")
    print(json.dumps(report["acceptance"], indent=2))
    print(f"summary: {args.output_root / 'comparison.json'}")
    return 2 if args.require_pass and report["status"] != "passed" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"physics acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
