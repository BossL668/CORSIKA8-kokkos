#!/usr/bin/env python3
"""Run a resumable local CUDA radio campaign paired to existing CPU seeds."""

from __future__ import annotations

import argparse
import json
import os
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--antenna-file", required=True, type=Path)
    parser.add_argument("--flupro", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--energy-gev", type=float, default=1.0e8)
    parser.add_argument("--zenith-deg", type=float, default=47.0)
    parser.add_argument("--azimuth-deg", type=float, default=180.0)
    parser.add_argument("--em-thinning", type=float, default=1.0e-6)
    parser.add_argument(
        "--geomagnetic-model", choices=("IGRF13", "IGRF14"), default="IGRF14"
    )
    parser.add_argument("--geomagnetic-year", type=float, default=2027.0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--gpu-radio-field-limit", type=float, default=1.0)
    parser.add_argument("--hadronic-workers", type=int, default=4)
    parser.add_argument("--hadronic-min-batch", type=int, default=64)
    parser.add_argument("--hadronic-target-batch-ms", type=float, default=5.0)
    parser.add_argument("--hadronic-max-batch", type=int, default=256)
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def scientific_tag(value: float) -> str:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"value must be positive and finite: {value}")
    mantissa, exponent = f"{value:.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def thinning_tag(value: float) -> str:
    mantissa, exponent = scientific_tag(value).split("e")
    exponent_value = int(exponent)
    sign = "m" if exponent_value < 0 else "p"
    return f"{mantissa}e{sign}{abs(exponent_value):02d}"


def number_tag(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"value must be finite: {value}")
    return f"{value:.12g}"


def reference_pattern(
    energy_GeV: float,
    zenith_deg: float,
    azimuth_deg: float,
    em_thinning: float,
) -> re.Pattern[str]:
    prefix = (
        f"proton_E{scientific_tag(energy_GeV)}_"
        f"TH{number_tag(zenith_deg)}_PH{number_tag(azimuth_deg)}_"
        f"CX0_CY0_SEED0_EM{thinning_tag(em_thinning)}_N"
    )
    return re.compile(rf"^{re.escape(prefix)}(?P<index>\d+)$")


def reference_events(
    root: Path,
    count: int,
    energy_GeV: float,
    zenith_deg: float,
    azimuth_deg: float,
    em_thinning: float,
) -> list[dict]:
    pattern = reference_pattern(
        energy_GeV, zenith_deg, azimuth_deg, em_thinning
    )
    events = []
    for candidate in root.iterdir():
        match = pattern.match(candidate.name)
        if match is None or not candidate.is_dir():
            continue
        index = int(match.group("index"))
        library = candidate / candidate.name
        summary = read_yaml(library / "summary.yaml")
        primary = read_yaml(library / "primary" / "summary.yaml")["shower_0"]
        if int(summary.get("showers", -1)) != 1:
            raise ValueError(f"CPU reference is not a complete one-shower library: {library}")
        if int(primary.get("pdg", -1)) != 2212 or not math.isclose(
            float(primary.get("total_energy", 0.0)), energy_GeV, rel_tol=1.0e-12
        ):
            raise ValueError(f"CPU reference has unexpected primary: {library}")
        waveform = library / "CoREAS" / "observers.parquet"
        if not waveform.is_file() or waveform.stat().st_size == 0:
            raise ValueError(f"CPU reference has no CoREAS waveform: {library}")
        events.append(
            {
                "index": index,
                "seed": int(summary["seed"]),
                "cpu_library": str(library.resolve()),
            }
        )
    events.sort(key=lambda event: event["index"])
    selected = events[:count]
    if len(selected) != count or [event["index"] for event in selected] != list(range(count)):
        raise ValueError(f"expected CPU references N0..N{count - 1}, found {len(selected)}")
    return selected


def cuda_name(
    index: int,
    seed: int,
    energy_GeV: float,
    zenith_deg: float,
    azimuth_deg: float,
    em_thinning: float,
) -> str:
    return (
        f"proton_E{scientific_tag(energy_GeV)}_"
        f"TH{number_tag(zenith_deg)}_PH{number_tag(azimuth_deg)}_"
        f"CX0_CY0_SEED{seed}_EM{thinning_tag(em_thinning)}_N{index}"
    )


def complete_output(path: Path, seed: int) -> bool:
    try:
        summary = read_yaml(path / "summary.yaml")
        gpu = read_yaml(path / "gpu_em" / "summary.yaml")["shower_0"]
        timing = read_yaml(path / "simulation_timing" / "summary.yaml")["shower_0"]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError):
        return False
    return (
        int(summary.get("showers", -1)) == 1
        and int(summary.get("seed", -1)) == seed
        and gpu.get("status") == "complete"
        and gpu.get("complete") is True
        and timing.get("closed") is True
        and (path / "CoREAS" / "observers.parquet").is_file()
    )


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    for path in (args.executable, args.table, args.antenna_file):
        if not path.is_file():
            raise ValueError(f"required file does not exist: {path}")
    if not (args.flupro / "libflukahp.a").is_file():
        raise ValueError(f"FLUPRO has no libflukahp.a: {args.flupro}")
    positive_values = (
        args.count,
        args.energy_gev,
        args.em_thinning,
        args.gpu_min_batch,
        args.gpu_memory_fraction,
        args.gpu_table_tolerance,
        args.gpu_radio_field_limit,
        args.hadronic_workers,
        args.hadronic_min_batch,
        args.hadronic_target_batch_ms,
        args.hadronic_max_batch,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive_values):
        raise ValueError("counts, energy, thinning, scheduler values, and tolerances must be positive")
    if args.gpu_device < 0 or not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("invalid CUDA device or memory fraction")

    references = reference_events(
        args.reference_root.resolve(),
        args.count,
        args.energy_gev,
        args.zenith_deg,
        args.azimuth_deg,
        args.em_thinning,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shower_root = output_root / (
        f"proton_cuda_{args.geomagnetic_model}_{number_tag(args.geomagnetic_year)}_"
        f"E{scientific_tag(args.energy_gev)}_TH{number_tag(args.zenith_deg)}_"
        f"PH{number_tag(args.azimuth_deg)}_EM{thinning_tag(args.em_thinning)}"
    )
    log_root = output_root / "logs"
    shower_root.mkdir(exist_ok=True)
    log_root.mkdir(exist_ok=True)
    manifest_path = output_root / "campaign_manifest.json"
    manifest = {
        "configuration": {
            "primary_pdg": 2212,
            "event_count": args.count,
            "energy_GeV": args.energy_gev,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "em_thinning": args.em_thinning,
            "geomagnetic_model": args.geomagnetic_model,
            "geomagnetic_year": args.geomagnetic_year,
            "executable": str(args.executable.resolve()),
            "table": str(args.table.resolve()),
            "antenna_file": str(args.antenna_file.resolve()),
            "flupro": str(args.flupro.resolve()),
            "reference_root": str(args.reference_root.resolve()),
            "acceleration": {
                "em_backend": "cuda",
                "gpu_device": args.gpu_device,
                "gpu_min_batch": args.gpu_min_batch,
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "gpu_resident_cross_species": True,
                "radio_backend": "cuda",
                "gpu_radio_field_limit_V_per_m": args.gpu_radio_field_limit,
                "hadronic_backend": "fluka-process",
                "hadronic_workers": args.hadronic_workers,
                "hadronic_min_batch": args.hadronic_min_batch,
                "hadronic_target_batch_ms": args.hadronic_target_batch_ms,
                "hadronic_max_batch": args.hadronic_max_batch,
            },
        },
        "events": [],
    }

    environment = os.environ.copy()
    environment["FLUPRO"] = str(args.flupro.resolve())
    environment["FLUFOR"] = "gfortran"

    for reference in references:
        index = reference["index"]
        seed = reference["seed"]
        name = cuda_name(
            index,
            seed,
            args.energy_gev,
            args.zenith_deg,
            args.azimuth_deg,
            args.em_thinning,
        )
        output = shower_root / name
        log_path = log_root / f"{name}.log"
        event = dict(reference)
        event.update(
            {
                "cuda_library": str(output),
                "log": str(log_path),
            }
        )
        if complete_output(output, seed):
            event["status"] = "complete"
            event["resumed"] = True
            event["runtime_s"] = float(read_yaml(output / "summary.yaml")["runtime_raw"])
            manifest["events"].append(event)
            write_manifest(manifest_path, manifest)
            print(f"[{index + 1}/{args.count}] already complete: {name}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"incomplete output already exists; refusing to overwrite: {output}")

        command = [
            str(args.executable.resolve()),
            "-p", "2212",
            "-E", f"{args.energy_gev:.17g}",
            "-z", f"{args.zenith_deg:.17g}",
            "-a", f"{args.azimuth_deg:.17g}",
            "-s", str(seed),
            "-f", str(output),
            "--emthin", f"{args.em_thinning:.17g}",
            "--antenna-file", str(args.antenna_file.resolve()),
            "--geomagnetic-model", args.geomagnetic_model,
            "--geomagnetic-year", f"{args.geomagnetic_year:.17g}",
            "--em-backend", "cuda",
            "--gpu-device", str(args.gpu_device),
            "--gpu-min-batch", str(args.gpu_min_batch),
            "--gpu-memory-fraction", f"{args.gpu_memory_fraction:.17g}",
            "--gpu-table-cache", str(args.table.resolve()),
            "--gpu-table-tolerance", f"{args.gpu_table_tolerance:.17g}",
            "--gpu-resident-cross-species", "true",
            "--radio-backend", "cuda",
            "--gpu-radio-field-limit", f"{args.gpu_radio_field_limit:.17g}",
            "--hadronic-backend", "fluka-process",
            "--hadronic-workers", str(args.hadronic_workers),
            "--hadronic-min-batch", str(args.hadronic_min_batch),
            "--hadronic-target-batch-ms", f"{args.hadronic_target_batch_ms:.17g}",
            "--hadronic-max-batch", str(args.hadronic_max_batch),
        ]
        event["command"] = command
        event["start_time"] = datetime.now().astimezone().isoformat()
        event["status"] = "running"
        manifest["events"].append(event)
        write_manifest(manifest_path, manifest)
        print(f"[{index + 1}/{args.count}] start seed={seed}: {name}", flush=True)
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=str(output_root),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        event["end_time"] = datetime.now().astimezone().isoformat()
        event["runner_wall_time_s"] = time.monotonic() - start
        event["return_code"] = result.returncode
        if result.returncode != 0 or not complete_output(output, seed):
            event["status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise RuntimeError(f"CUDA event failed: {name}; see {log_path}")
        event["status"] = "complete"
        event["runtime_s"] = float(read_yaml(output / "summary.yaml")["runtime_raw"])
        write_manifest(manifest_path, manifest)
        print(
            f"[{index + 1}/{args.count}] complete in {event['runtime_s']:.3f} s: {name}",
            flush=True,
        )

    manifest["status"] = "complete"
    manifest["completed_time"] = datetime.now().astimezone().isoformat()
    write_manifest(manifest_path, manifest)
    print(f"campaign complete: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
