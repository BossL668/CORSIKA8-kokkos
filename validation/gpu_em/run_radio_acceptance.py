#!/usr/bin/env python3
"""
Run identical CUDA EM showers with CPU and resident-CUDA radio projection.

The same deterministic transport seed is intentionally used for both runs.
This isolates the CoREAS/ZHS backend while compare_radio_backends.py verifies
that non-radio transport tables are identical.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from compare_radio_backends import compare_outputs, write_csv
from run_performance_acceptance import (
    MOLIERE_CACHE_SUFFIX,
    run_command,
    single_thread_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate resident CUDA CoREAS/ZHS against the scalar radio "
            "implementation using identical CUDA EM tracks."
        )
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--energy-gev", type=float, default=1.0e3)
    parser.add_argument("--events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=25025)
    parser.add_argument("--primary-pdg", type=int, default=11)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--ring", type=int, default=1)
    parser.add_argument("--em-cut-gev", type=float, default=0.5e-3)
    parser.add_argument("--em-thinning", type=float, default=1.0e-4)
    parser.add_argument("--maximum-weight", type=float, default=100.0)
    parser.add_argument(
        "--non-em-cut-gev",
        type=float,
        default=0.3,
        help=(
            "Common hadron/muon/tau cut in GeV. The muon value must match "
            "the CUDA table metadata; the production EM+muon table uses "
            "0.3 GeV."
        ),
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=64)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--gpu-radio-field-limit", type=float, default=1.0)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--l2-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--fluence-tolerance", type=float, default=5.0e-4)
    parser.add_argument(
        "--absolute-tolerance-v-per-m",
        type=float,
        default=1.0e-18,
    )
    parser.add_argument("--skip-cache-check", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.executable.is_file():
        raise ValueError(f"executable is not a regular file: {args.executable}")
    if not args.table.is_file():
        raise ValueError(f"rate table is not a regular file: {args.table}")
    if args.output_root.exists():
        raise ValueError(
            f"output root already exists; refusing to overwrite: {args.output_root}"
        )
    if args.energy_gev <= 0.0 or args.events <= 0:
        raise ValueError("energy and event count must be positive")
    if args.ring == 0:
        raise ValueError("radio acceptance requires a non-zero observer ring")
    if args.em_cut_gev <= 0.0 or args.non_em_cut_gev <= 0.0:
        raise ValueError("particle cuts must be positive")
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("EM thinning fraction must be in [0, 1]")
    if args.maximum_weight <= 0.0:
        raise ValueError("maximum weight must be positive")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must lie in (0, 1]")
    if args.gpu_radio_field_limit <= 0.0:
        raise ValueError("GPU radio field limit must be positive")
    for name in (
        "relative_tolerance",
        "l2_tolerance",
        "fluence_tolerance",
        "absolute_tolerance_v_per_m",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def common_cuda_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        str(args.executable),
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
        "--ring",
        str(args.ring),
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
        "--verbosity",
        "warn",
        "--em-backend",
        "cuda",
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


def radio_command(
    args: argparse.Namespace, output: Path, backend: str
) -> list[str]:
    command = common_cuda_command(args, output) + [
        "--radio-backend",
        backend,
    ]
    if backend == "cuda":
        command += [
            "--gpu-radio-field-limit",
            f"{args.gpu_radio_field_limit:.17g}",
        ]
    return command


def main() -> int:
    args = parse_args()
    validate_arguments(args)
    args.executable = args.executable.resolve()
    args.table = args.table.resolve()
    args.output_root = args.output_root.resolve()
    cache_path = Path(str(args.table) + MOLIERE_CACHE_SUFFIX)
    if not args.skip_cache_check and not cache_path.is_file():
        raise RuntimeError(
            f"required hot Moliere cache is absent: {cache_path}"
        )
    args.output_root.mkdir(parents=True)

    cpu_output = args.output_root / "cpu_radio"
    cuda_output = args.output_root / "cuda_radio"
    cpu_command = radio_command(args, cpu_output, "cpu")
    cuda_command = radio_command(args, cuda_output, "cuda")
    environment = single_thread_environment()
    flupro = environment.get("FLUPRO")

    cpu_wall = run_command(
        cpu_command, args.output_root / "cpu_radio.log", environment
    )
    cuda_wall = run_command(
        cuda_command, args.output_root / "cuda_radio.log", environment
    )
    report, rows = compare_outputs(
        cpu_output,
        cuda_output,
        args.relative_tolerance,
        args.l2_tolerance,
        args.fluence_tolerance,
        args.absolute_tolerance_v_per_m,
    )
    comparison_path = args.output_root / "radio_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    write_csv(args.output_root / "radio_component_comparison.csv", rows)

    manifest: dict[str, Any] = {
        "status": report["status"],
        "configuration": {
            "energy_GeV": args.energy_gev,
            "events": args.events,
            "seed": args.seed,
            "primary_pdg": args.primary_pdg,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "ring": args.ring,
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "non_em_cut_GeV": args.non_em_cut_gev,
            "gpu_device": args.gpu_device,
            "gpu_min_batch": args.gpu_min_batch,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "gpu_table_tolerance": args.gpu_table_tolerance,
            "gpu_radio_field_limit_V_per_m": args.gpu_radio_field_limit,
        },
        "cache": {
            "path": str(cache_path),
            "required": not args.skip_cache_check,
            "present": cache_path.is_file(),
        },
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
        "commands": {
            "cpu_radio": cpu_command,
            "cuda_radio": cuda_command,
        },
        "external_wall_seconds": {
            "cpu_radio": cpu_wall,
            "cuda_radio": cuda_wall,
        },
        "outputs": {
            "comparison": str(comparison_path),
            "component_csv": str(
                args.output_root / "radio_component_comparison.csv"
            ),
        },
    }
    with (args.output_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")
    print(json.dumps(report["algorithms"], indent=2))
    print(f"comparison: {comparison_path}")
    return 0 if report["status"] == "passed" or not args.require_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"radio acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
