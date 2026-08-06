#!/usr/bin/env python3
"""Run the same-track GPU radio oracle after a CUDA campaign is complete."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def campaign_status(campaign_root: Path) -> str:
    manifest = campaign_root / "campaign_manifest.json"
    if not manifest.is_file():
        return "missing"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"campaign manifest is not a mapping: {manifest}")
    return str(value.get("status", "missing"))


def run_oracle(
    campaign_root: Path,
    build_dir: Path,
    ctest_executable: str,
    test_name: str,
) -> int:
    log_path = campaign_root / "radio_projection_oracle_ctest.log"
    status_path = campaign_root / "radio_projection_oracle_status.json"
    command = [
        ctest_executable,
        "--output-on-failure",
        "-R",
        f"^{test_name}$",
    ]
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=build_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete" if completed.returncode == 0 else "failed",
            "test": test_name,
            "returncode": completed.returncode,
            "command": command,
            "build_dir": str(build_dir.resolve()),
            "log": str(log_path.resolve()),
            "started_utc": started,
            "finished_utc": utc_now(),
        },
    )
    return completed.returncode


def run_full_acceptance(args: argparse.Namespace) -> int:
    if args.acceptance_script is None:
        return 0
    required = (args.executable, args.table, args.acceptance_output)
    if any(value is None for value in required):
        raise ValueError(
            "--acceptance-script requires --executable, --table, and "
            "--acceptance-output"
        )
    if not args.acceptance_script.is_file() or not args.executable.is_file():
        raise ValueError("radio acceptance script or executable is absent")
    if not args.table.is_file():
        raise ValueError("radio acceptance table is absent")
    if args.acceptance_output.exists():
        raise ValueError(
            f"refusing to overwrite radio acceptance output: {args.acceptance_output}"
        )
    command = [
        sys.executable,
        str(args.acceptance_script.resolve()),
        "--executable",
        str(args.executable.resolve()),
        "--table",
        str(args.table.resolve()),
        "--output-root",
        str(args.acceptance_output.resolve()),
        "--energy-gev",
        f"{args.acceptance_energy_gev:.17g}",
        "--events",
        str(args.acceptance_events),
        "--seed",
        str(args.acceptance_seed),
        "--zenith-deg",
        f"{args.acceptance_zenith_deg:.17g}",
        "--azimuth-deg",
        f"{args.acceptance_azimuth_deg:.17g}",
        "--geomagnetic-model",
        args.acceptance_geomagnetic_model,
        "--geomagnetic-year",
        f"{args.acceptance_geomagnetic_year:.17g}",
        "--ring",
        str(args.acceptance_ring),
        "--gpu-device",
        str(args.gpu_device),
        "--gpu-memory-fraction",
        f"{args.gpu_memory_fraction:.17g}",
        "--require-pass",
    ]
    if args.acceptance_antenna_file is not None:
        if not args.acceptance_antenna_file.is_file():
            raise ValueError("radio acceptance antenna file is absent")
        command.extend(
            ("--antenna-file", str(args.acceptance_antenna_file.resolve()))
        )
    log_path = args.campaign_root / "radio_identical_track_acceptance.runner.log"
    status_path = args.campaign_root / "radio_identical_track_acceptance_status.json"
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "output": str(args.acceptance_output.resolve()),
            "log": str(log_path.resolve()),
            "started_utc": started,
            "finished_utc": utc_now(),
        },
    )
    return completed.returncode


def run_pulse_feature_acceptance(args: argparse.Namespace) -> int:
    if args.pulse_feature_script is None:
        return 0
    if (
        args.acceptance_output is None
        or args.pulse_analysis_root is None
        or args.pulse_feature_output is None
    ):
        raise ValueError(
            "--pulse-feature-script requires --acceptance-output, "
            "--pulse-analysis-root, and --pulse-feature-output"
        )
    if not args.pulse_feature_script.is_file():
        raise ValueError("identical-track pulse-feature script is absent")
    if not args.pulse_analysis_root.is_dir():
        raise ValueError("pulse_analysis_modular root is absent")
    cpu_output = args.acceptance_output / "cpu_radio"
    cuda_output = args.acceptance_output / "cuda_radio"
    if not cpu_output.is_dir() or not cuda_output.is_dir():
        raise ValueError("identical-track CPU/CUDA radio products are absent")
    if args.pulse_feature_output.exists():
        raise ValueError(
            "refusing to overwrite identical-track pulse-feature output: "
            f"{args.pulse_feature_output}"
        )
    command = [
        sys.executable,
        str(args.pulse_feature_script.resolve()),
        "--cpu-radio-output",
        str(cpu_output.resolve()),
        "--cuda-radio-output",
        str(cuda_output.resolve()),
        "--pulse-analysis-root",
        str(args.pulse_analysis_root.resolve()),
        "--output",
        str(args.pulse_feature_output.resolve()),
        "--zenith-deg",
        f"{args.acceptance_zenith_deg:.17g}",
        "--azimuth-deg",
        f"{args.acceptance_azimuth_deg:.17g}",
        "--amplitude-relative-tolerance",
        f"{args.pulse_amplitude_relative_tolerance:.17g}",
        "--width-absolute-tolerance-ns",
        f"{args.pulse_width_absolute_tolerance_ns:.17g}",
        "--require-pass",
    ]
    log_path = args.campaign_root / "radio_identical_track_pulse_features.runner.log"
    status_path = (
        args.campaign_root
        / "radio_identical_track_pulse_features_status.json"
    )
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "output": str(args.pulse_feature_output.resolve()),
            "log": str(log_path.resolve()),
            "started_utc": started,
            "finished_utc": utc_now(),
        },
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ctest-executable", default="ctest")
    parser.add_argument("--test-name", default="testGpuRadioProjection")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--acceptance-script", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument("--acceptance-energy-gev", type=float, default=1000.0)
    parser.add_argument("--acceptance-events", type=int, default=10)
    parser.add_argument("--acceptance-seed", type=int, default=2026087001)
    parser.add_argument("--acceptance-zenith-deg", type=float, default=0.0)
    parser.add_argument("--acceptance-azimuth-deg", type=float, default=0.0)
    parser.add_argument(
        "--acceptance-geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        default="IGRF14",
    )
    parser.add_argument(
        "--acceptance-geomagnetic-year", type=float, default=2027.0
    )
    parser.add_argument("--acceptance-ring", type=int, default=1)
    parser.add_argument("--acceptance-antenna-file", type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--pulse-feature-script", type=Path)
    parser.add_argument("--pulse-analysis-root", type=Path)
    parser.add_argument("--pulse-feature-output", type=Path)
    parser.add_argument(
        "--pulse-amplitude-relative-tolerance",
        type=float,
        default=2.0e-4,
    )
    parser.add_argument(
        "--pulse-width-absolute-tolerance-ns",
        type=float,
        default=0.11,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Inspect once and return 75 while the campaign is not complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.campaign_root.is_dir() or not args.build_dir.is_dir():
        raise ValueError("campaign root and build directory must exist")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if (
        args.acceptance_energy_gev <= 0.0
        or args.acceptance_events <= 0
        or args.acceptance_seed < 0
        or not math.isfinite(args.acceptance_geomagnetic_year)
        or not 0.0 < args.gpu_memory_fraction <= 1.0
        or not math.isfinite(args.pulse_amplitude_relative_tolerance)
        or args.pulse_amplitude_relative_tolerance < 0.0
        or not math.isfinite(args.pulse_width_absolute_tolerance_ns)
        or args.pulse_width_absolute_tolerance_ns < 0.0
    ):
        raise ValueError("invalid full radio acceptance configuration")
    while True:
        status = campaign_status(args.campaign_root)
        if status == "complete":
            oracle_code = run_oracle(
                args.campaign_root,
                args.build_dir,
                args.ctest_executable,
                args.test_name,
            )
            if oracle_code != 0:
                return oracle_code
            acceptance_code = run_full_acceptance(args)
            if acceptance_code != 0:
                return acceptance_code
            return run_pulse_feature_acceptance(args)
        if status == "failed":
            atomic_json(
                args.campaign_root / "radio_projection_oracle_status.json",
                {
                    "schema_version": 1,
                    "status": "not_run_campaign_failed",
                    "test": args.test_name,
                    "returncode": None,
                    "finished_utc": utc_now(),
                },
            )
            return 2
        if args.once:
            return 75
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
