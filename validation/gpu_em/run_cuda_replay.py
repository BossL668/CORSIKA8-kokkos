#!/usr/bin/env python3
"""Run legacy, refactored-scalar, and CUDA CORSIKA 8 replay comparisons."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from compare_cuda_replay import compare_outputs, write_csv
from run_performance_acceptance import run_command, single_thread_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-executable", type=Path, required=True)
    parser.add_argument("--cuda-executable", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--antenna-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--energy-gev", type=float, default=1.0e3)
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260729)
    parser.add_argument("--primary-pdg", type=int, default=11)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--observation-level-m", type=float, default=2680.444195)
    parser.add_argument("--injection-height-m", type=float, default=112750.0)
    parser.add_argument("--em-cut-gev", type=float, default=0.5e-3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-3)
    parser.add_argument("--maximum-weight", type=float, default=1.0e6)
    parser.add_argument("--non-em-cut-gev", type=float, default=1.0e13)
    parser.add_argument(
        "--max-deflection-angle",
        type=float,
        default=0.2,
        help=(
            "Maximum magnetic deflection per tracking step in radians. "
            "Use 0.001 for the small-track radio-convergence setting studied "
            "in arXiv:2409.15999."
        ),
    )
    parser.add_argument(
        "--radio-sampling-rate-ghz",
        type=float,
        default=1.0,
        help=(
            "Time-domain observer sampling rate. Use 10 GHz (0.1 ns) "
            "for a converged 50--350 MHz CoREAS/ZHS comparison."
        ),
    )
    parser.add_argument(
        "--radio-window-duration-ns",
        type=float,
        default=400.0,
    )
    parser.add_argument(
        "--radio-pretrigger-ns",
        type=float,
        default=10.0,
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--minimum-showers", type=int, default=20)
    parser.add_argument(
        "--radio-normalization",
        choices=("auto", "primary-energy", "em-deposit"),
        default="auto",
        help=(
            "Radiation-energy normalization. auto uses fixed primary energy "
            "for electron/positron/photon primaries and per-shower EM "
            "deposited energy for hadronic primaries."
        ),
    )
    parser.add_argument("--skip-scalar-control", action="store_true")
    parser.add_argument(
        "--overlap-backends",
        action="store_true",
        help=(
            "Run independent legacy/scalar/CUDA arms concurrently. Use only "
            "for physics validation, not for performance measurements."
        ),
    )
    parser.add_argument(
        "--disable-process-trace",
        action="store_true",
        help="omit large per-step traces for radio/profile ensemble runs",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    for name in ("legacy_executable", "cuda_executable", "table", "antenna_file"):
        path = getattr(args, name)
        if not path.is_file():
            raise ValueError(f"{name} is not a regular file: {path}")
    if args.output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {args.output_root}")
    if args.events <= 0 or args.energy_gev <= 0.0:
        raise ValueError("events and energy must be positive")
    if args.minimum_showers <= 0:
        raise ValueError("minimum shower count must be positive")
    for name in ("gpu_memory_fraction", "gpu_table_tolerance"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not math.isfinite(args.max_deflection_angle)
        or args.max_deflection_angle <= 0.0
    ):
        raise ValueError(
            "max_deflection_angle must be finite and positive"
        )
    for name in (
        "radio_sampling_rate_ghz",
        "radio_window_duration_ns",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not math.isfinite(args.radio_pretrigger_ns)
        or args.radio_pretrigger_ns < 0.0
    ):
        raise ValueError(
            "radio_pretrigger_ns must be finite and non-negative"
        )
    configurable_radio_requested = (
        args.radio_sampling_rate_ghz != 1.0
        or args.radio_window_duration_ns != 400.0
        or args.radio_pretrigger_ns != 10.0
    )
    if (
        configurable_radio_requested
        and args.legacy_executable.resolve()
        != args.cuda_executable.resolve()
    ):
        raise ValueError(
            "the separate legacy executable has fixed radio sampling/window "
            "settings; use the refactored executable for both arms when "
            "requesting non-default radio observer configuration"
        )


def common_command(args: argparse.Namespace, executable: Path, output: Path) -> list[str]:
    command = [
        str(executable),
        "-p",
        str(args.primary_pdg),
        "-E",
        f"{args.energy_gev:.17g}",
        "-N",
        str(args.events),
        "-f",
        str(output),
        "--seed",
        str(args.seed),
        "--zenith",
        f"{args.zenith_deg:.17g}",
        "--azimuth",
        f"{args.azimuth_deg:.17g}",
        "--observation-level",
        f"{args.observation_level_m:.17g}",
        "--injection-height",
        f"{args.injection_height_m:.17g}",
        "--ring",
        "0",
        "--antenna-file",
        str(args.antenna_file),
        "--emcut",
        f"{args.em_cut_gev:.17g}",
        "--emthin",
        f"{args.em_thinning:.17g}",
        "--max-weight",
        f"{args.maximum_weight:.17g}",
        "--hadcut",
        f"{args.non_em_cut_gev:.17g}",
        "--mucut",
        f"{args.non_em_cut_gev:.17g}",
        "--taucut",
        f"{args.non_em_cut_gev:.17g}",
        "--max-deflection-angle",
        f"{args.max_deflection_angle:.17g}",
        "--verbosity",
        "warn",
    ]
    if executable.resolve() == args.cuda_executable.resolve():
        command += [
            "--radio-sampling-rate-ghz",
            f"{args.radio_sampling_rate_ghz:.17g}",
            "--radio-window-duration-ns",
            f"{args.radio_window_duration_ns:.17g}",
            "--radio-pretrigger-ns",
            f"{args.radio_pretrigger_ns:.17g}",
        ]
    return command


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    args.legacy_executable = args.legacy_executable.resolve()
    args.cuda_executable = args.cuda_executable.resolve()
    args.table = args.table.resolve()
    args.antenna_file = args.antenna_file.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True)

    legacy_output = args.output_root / "legacy_proposal"
    scalar_output = args.output_root / "refactor_proposal"
    cuda_output = args.output_root / "cuda"
    legacy_trace = args.output_root / "legacy_process_trace.csv"
    scalar_trace = args.output_root / "refactor_process_trace.csv"
    cuda_trace = args.output_root / "cuda_process_trace.csv"

    legacy_command = common_command(
        args, args.legacy_executable, legacy_output
    )
    if not args.disable_process_trace:
        legacy_command += ["--cuda-replay-trace", str(legacy_trace)]
    scalar_command = common_command(
        args, args.cuda_executable, scalar_output
    ) + [
        "--em-backend",
        "proposal",
        "--radio-backend",
        "cpu",
    ]
    if not args.disable_process_trace:
        scalar_command += ["--cuda-replay-trace", str(scalar_trace)]
    cuda_command = common_command(
        args, args.cuda_executable, cuda_output
    ) + [
        "--em-backend",
        "cuda",
        "--radio-backend",
        "cpu",
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
        "--gpu-detailed-stage-timing",
    ]
    if not args.disable_process_trace:
        cuda_command += ["--cuda-replay-trace", str(cuda_trace)]

    environment = single_thread_environment()
    wall_seconds: dict[str, float] = {}
    commands = {
        "legacy_proposal": (
            legacy_command,
            args.output_root / "legacy.log",
        ),
    }
    if not args.skip_scalar_control:
        commands["refactor_proposal"] = (
            scalar_command,
            args.output_root / "refactor_proposal.log",
        )
    commands["cuda"] = (cuda_command, args.output_root / "cuda.log")
    if args.overlap_backends:
        with ThreadPoolExecutor(max_workers=len(commands)) as pool:
            futures = {
                name: pool.submit(run_command, command, log, environment)
                for name, (command, log) in commands.items()
            }
            for name, future in futures.items():
                wall_seconds[name] = future.result()
    else:
        for name, (command, log) in commands.items():
            wall_seconds[name] = run_command(command, log, environment)

    reports: dict[str, Any] = {}
    if args.radio_normalization == "primary-energy" or (
        args.radio_normalization == "auto"
        and args.primary_pdg in (11, -11, 22)
    ):
        primary_normalization_energy_GeV: float | None = args.energy_gev
        resolved_radio_normalization = "primary-energy"
    else:
        primary_normalization_energy_GeV = None
        resolved_radio_normalization = "em-deposit"
    if not args.skip_scalar_control:
        scalar_report, scalar_rows = compare_outputs(
            legacy_output,
            scalar_output,
            legacy_trace,
            scalar_trace,
            args.minimum_showers,
            primary_normalization_energy_GeV=
                primary_normalization_energy_GeV,
        )
        reports["legacy_vs_refactor_proposal"] = scalar_report
        with (args.output_root / "legacy_vs_refactor_proposal.json").open(
            "w", encoding="utf-8"
        ) as destination:
            json.dump(scalar_report, destination, indent=2, allow_nan=True)
            destination.write("\n")
        write_csv(args.output_root / "scalar_control_radio.csv", scalar_rows)

    cuda_report, cuda_rows = compare_outputs(
        legacy_output,
        cuda_output,
        legacy_trace,
        cuda_trace,
        args.minimum_showers,
        primary_normalization_energy_GeV=
            primary_normalization_energy_GeV,
    )
    reports["legacy_vs_cuda"] = cuda_report
    with (args.output_root / "legacy_vs_cuda.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(cuda_report, destination, indent=2, allow_nan=True)
        destination.write("\n")
    write_csv(args.output_root / "cuda_radio_statistics.csv", cuda_rows)

    manifest = {
        "status": cuda_report["status"],
        "configuration": {
            "energy_GeV": args.energy_gev,
            "events": args.events,
            "seed": args.seed,
            "primary_pdg": args.primary_pdg,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "observation_level_m": args.observation_level_m,
            "injection_height_m": args.injection_height_m,
            "IGRF": {"model": "IGRF13", "year": 2025.0},
            "max_deflection_angle_rad": args.max_deflection_angle,
            "antenna_file": str(args.antenna_file),
            "radio_sampling_rate_GHz": args.radio_sampling_rate_ghz,
            "radio_window_duration_ns": args.radio_window_duration_ns,
            "radio_pretrigger_ns": args.radio_pretrigger_ns,
            "process_trace_enabled": not args.disable_process_trace,
            "overlap_backends": args.overlap_backends,
            "radio_normalization_requested": args.radio_normalization,
            "radio_normalization_resolved": resolved_radio_normalization,
        },
        "commands": {
            "legacy_proposal": legacy_command,
            "refactor_proposal": None
            if args.skip_scalar_control
            else scalar_command,
            "cuda": cuda_command,
        },
        "external_wall_seconds": wall_seconds,
        "reports": {
            "legacy_vs_refactor_proposal": None
            if args.skip_scalar_control
            else str(args.output_root / "legacy_vs_refactor_proposal.json"),
            "legacy_vs_cuda": str(args.output_root / "legacy_vs_cuda.json"),
        },
    }
    with (args.output_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")

    print(json.dumps({"status": cuda_report["status"], "output": str(args.output_root)}, indent=2))
    return 2 if args.require_pass and cuda_report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
