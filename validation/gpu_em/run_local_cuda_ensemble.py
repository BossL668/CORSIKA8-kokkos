#!/usr/bin/env python3
"""Run a resumable, sequential CUDA shower supplement in fixed-size batches."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--proposal-executable", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--physics-runner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--existing-cuda", type=Path, action="append", default=[],
        help="Completed CUDA source already counted toward --target-events.",
    )
    parser.add_argument(
        "--reference-proposal-root", type=Path,
        help="Directory containing complete one-shower proposal_shard_* outputs.",
    )
    parser.add_argument(
        "--allow-mixed-proposal-builds",
        action="store_true",
        help=(
            "Accept CPU references produced by a separately linked original "
            "CORSIKA executable while retaining strict canonical physics checks."
        ),
    )
    parser.add_argument(
        "--defer-reference-comparison",
        action="store_true",
        help=(
            "Run resumable CUDA production batches immediately without CPU "
            "references. Preserve provenance for a separate final comparison."
        ),
    )
    parser.add_argument("--antenna-file", type=Path, required=True)
    parser.add_argument("--flupro", type=Path, required=True)
    parser.add_argument("--target-events", type=int, default=500)
    parser.add_argument("--batch-events", type=int, default=25)
    parser.add_argument("--cuda-seed-start", type=int, required=True)
    parser.add_argument("--energy-gev", type=float, default=100_000.0)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument(
        "--geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        default="IGRF14",
    )
    parser.add_argument("--geomagnetic-year", type=float, default=2027.0)
    parser.add_argument("--em-cut-gev", type=float, default=0.0005)
    parser.add_argument("--em-thinning", type=float, default=1.0e-6)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.0,
        help=(
            "Explicit EM thinning maximum weight; zero omits --max-weight "
            "and preserves c8_air_shower's automatic Kobal value."
        ),
    )
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--shower-core-x-m", type=float, default=0.0)
    parser.add_argument("--shower-core-y-m", type=float, default=0.0)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-min-batch", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.70)
    parser.add_argument("--gpu-table-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--gpu-radio-field-limit", type=float, default=1.0)
    parser.add_argument("--cuda-hadronic-workers", type=int, default=4)
    parser.add_argument("--cuda-hadronic-min-batch", type=int, default=64)
    parser.add_argument(
        "--cuda-hadronic-target-batch-ms", type=float, default=5.0
    )
    parser.add_argument("--cuda-hadronic-max-batch", type=int, default=256)
    parser.add_argument(
        "--label-prefix",
        default="",
        help="Stable output label prefix; an unambiguous physics label is generated when empty.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "sha256": sha256(resolved),
    }


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def completed_event_count(root: Path) -> int:
    summary_path = root / "summary.yaml"
    provenance_path = root / "validation_provenance.json"
    required = (
        summary_path,
        provenance_path,
        root / "profile" / "profile.parquet",
        root / "particles" / "particles.parquet",
        root / "gpu_em" / "summary.yaml",
        root / "CoREAS" / "summary.yaml",
        root / "ZHS" / "summary.yaml",
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(
            f"incomplete existing CUDA source {root}: "
            + ", ".join(str(path) for path in missing)
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("backend") != "cuda":
        raise ValueError(f"not a CUDA provenance record: {provenance_path}")
    showers = int(read_yaml(summary_path).get("showers", -1))
    if showers <= 0:
        raise ValueError(f"invalid completed shower count in {summary_path}")
    return showers


def reference_sources(root: Path, count: int) -> list[Path]:
    sources = sorted(
        path.resolve()
        for path in root.glob("proposal_shard_*")
        if path.is_dir()
    )
    valid: list[Path] = []
    for source in sources:
        required = (
            source / "summary.yaml",
            source / "validation_provenance.json",
            source / "profile" / "profile.parquet",
            source / "particles" / "particles.parquet",
        )
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            summary = read_yaml(source / "summary.yaml")
            if int(summary.get("showers", -1)) == 1:
                valid.append(source)
        if len(valid) == count:
            break
    if len(valid) < count:
        raise ValueError(
            f"need {count} complete one-shower CPU references below {root}; "
            f"found {len(valid)}"
        )
    return valid


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def immutable_configuration(args: argparse.Namespace) -> dict[str, Any]:
    fluka_library = args.flupro / "libflukahp.a"
    return {
        "schema_version": SCHEMA_VERSION,
        "target_events": args.target_events,
        "batch_events": args.batch_events,
        "cuda_seed_start": args.cuda_seed_start,
        "energy_GeV": args.energy_gev,
        "primary_pdg": args.primary_pdg,
        "gpu_device": args.gpu_device,
        "physics": {
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "geomagnetic_model": args.geomagnetic_model,
            "geomagnetic_year": args.geomagnetic_year,
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "had_cut_GeV": args.had_cut_gev,
            "mu_cut_GeV": args.mu_cut_gev,
            "tau_cut_GeV": args.tau_cut_gev,
            "shower_core_x_m": args.shower_core_x_m,
            "shower_core_y_m": args.shower_core_y_m,
            "ring": args.ring,
            "cuda_radio_backend": "cuda",
            "cuda_hadronic_backend": "fluka-process",
        },
        "scheduler": {
            "gpu_min_batch": args.gpu_min_batch,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "gpu_table_tolerance": args.gpu_table_tolerance,
            "gpu_radio_field_limit_V_per_m": args.gpu_radio_field_limit,
            "cuda_hadronic_workers": args.cuda_hadronic_workers,
            "cuda_hadronic_min_batch": args.cuda_hadronic_min_batch,
            "cuda_hadronic_target_batch_ms": args.cuda_hadronic_target_batch_ms,
            "cuda_hadronic_max_batch": args.cuda_hadronic_max_batch,
        },
        "artifacts": {
            "executable": fingerprint(args.executable),
            "proposal_executable": fingerprint(args.proposal_executable),
            "table": fingerprint(args.table),
            "physics_runner": fingerprint(args.physics_runner),
            "antenna_file": fingerprint(args.antenna_file),
            "fluka_library": fingerprint(fluka_library),
        },
        "existing_cuda": [
            {"root": str(path.resolve()), "events": completed_event_count(path)}
            for path in args.existing_cuda
        ],
        "reference_proposal_root": (
            str(args.reference_proposal_root.resolve())
            if args.reference_proposal_root is not None
            else None
        ),
        "allow_mixed_proposal_builds": args.allow_mixed_proposal_builds,
        "defer_reference_comparison": args.defer_reference_comparison,
    }


def batch_command(
    args: argparse.Namespace,
    output: Path,
    events: int,
    seed: int,
    references: list[Path],
) -> list[str]:
    label_prefix = args.label_prefix.strip()
    if not label_prefix:
        energy_pev = args.energy_gev / 1.0e6
        label_prefix = (
            f"pdg{args.primary_pdg}_{energy_pev:.8g}PeV_"
            f"theta{args.zenith_deg:.8g}_phi{args.azimuth_deg:.8g}_"
            f"emthin{args.em_thinning:.8g}"
        )
    command = [
        sys.executable,
        str(args.physics_runner.resolve()),
        "--executable", str(args.executable.resolve()),
        "--proposal-executable", str(args.proposal_executable.resolve()),
        "--table", str(args.table.resolve()),
        "--output-root", str(output.resolve()),
        "--label", f"{label_prefix}_cuda_batch_seed{seed}",
        "--energy-gev", f"{args.energy_gev:.17g}",
        "--events", str(events),
        "--proposal-seed", "10100001",
        "--cuda-seed", str(seed),
        "--skip-proposal-run",
        "--primary-pdg", str(args.primary_pdg),
        "--zenith-deg", f"{args.zenith_deg:.17g}",
        "--azimuth-deg", f"{args.azimuth_deg:.17g}",
        "--em-cut-gev", f"{args.em_cut_gev:.17g}",
        "--em-thinning", f"{args.em_thinning:.17g}",
        "--maximum-weight", f"{args.maximum_weight:.17g}",
        "--had-cut-gev", f"{args.had_cut_gev:.17g}",
        "--mu-cut-gev", f"{args.mu_cut_gev:.17g}",
        "--tau-cut-gev", f"{args.tau_cut_gev:.17g}",
        "--shower-core-x-m", f"{args.shower_core_x_m:.17g}",
        "--shower-core-y-m", f"{args.shower_core_y_m:.17g}",
        "--ring", str(args.ring),
        "--antenna-file", str(args.antenna_file.resolve()),
        "--cuda-radio-backend", "cuda",
        "--gpu-device", str(args.gpu_device),
        "--gpu-min-batch", str(args.gpu_min_batch),
        "--gpu-memory-fraction", f"{args.gpu_memory_fraction:.17g}",
        "--gpu-table-tolerance", f"{args.gpu_table_tolerance:.17g}",
        "--gpu-radio-field-limit", f"{args.gpu_radio_field_limit:.17g}",
        "--cuda-hadronic-backend", "fluka-process",
        "--cuda-hadronic-workers", str(args.cuda_hadronic_workers),
        "--cuda-hadronic-min-batch", str(args.cuda_hadronic_min_batch),
        "--cuda-hadronic-target-batch-ms",
        f"{args.cuda_hadronic_target_batch_ms:.17g}",
        "--cuda-hadronic-max-batch", str(args.cuda_hadronic_max_batch),
        "--stability-bootstrap-repetitions", "1000",
    ]
    for reference in references[:events]:
        command.extend(("--additional-proposal", str(reference)))
    if args.allow_mixed_proposal_builds:
        command.append("--allow-mixed-proposal-builds")
    return command


def direct_batch_command(
    args: argparse.Namespace,
    output: Path,
    events: int,
    seed: int,
) -> list[str]:
    command = [
        str(args.executable.resolve()),
        "-p", str(args.primary_pdg),
        "-E", f"{args.energy_gev:.17g}",
        "-N", str(events),
        "-f", str((output / "cuda").resolve()),
        "--seed", str(seed),
        "--zenith", f"{args.zenith_deg:.17g}",
        "--azimuth", f"{args.azimuth_deg:.17g}",
        "--geomagnetic-model", args.geomagnetic_model,
        "--geomagnetic-year", f"{args.geomagnetic_year:.17g}",
        "--shower-core-x", f"{args.shower_core_x_m:.17g}",
        "--shower-core-y", f"{args.shower_core_y_m:.17g}",
        "--ring", str(args.ring),
        "--antenna-file", str(args.antenna_file.resolve()),
        "--radio-sampling-rate-ghz", "1",
        "--radio-window-duration-ns", "400",
        "--radio-pretrigger-ns", "10",
        "--emcut", f"{args.em_cut_gev:.17g}",
        "--emthin", f"{args.em_thinning:.17g}",
        "--hadcut", f"{args.had_cut_gev:.17g}",
        "--mucut", f"{args.mu_cut_gev:.17g}",
        "--taucut", f"{args.tau_cut_gev:.17g}",
        "--verbosity", "warn",
        "--em-backend", "cuda",
        "--radio-backend", "cuda",
        "--gpu-device", str(args.gpu_device),
        "--gpu-min-batch", str(args.gpu_min_batch),
        "--gpu-memory-fraction", f"{args.gpu_memory_fraction:.17g}",
        "--gpu-table-cache", str(args.table.resolve()),
        "--gpu-table-tolerance", f"{args.gpu_table_tolerance:.17g}",
        "--gpu-deterministic", "true",
        "--gpu-resident-cross-species", "true",
        "--gpu-radio-field-limit", f"{args.gpu_radio_field_limit:.17g}",
        "--hadronic-backend", "fluka-process",
        "--hadronic-workers", str(args.cuda_hadronic_workers),
        "--hadronic-min-batch", str(args.cuda_hadronic_min_batch),
        "--hadronic-target-batch-ms",
        f"{args.cuda_hadronic_target_batch_ms:.17g}",
        "--hadronic-max-batch", str(args.cuda_hadronic_max_batch),
    ]
    if args.maximum_weight > 0.0:
        insertion = command.index("--hadcut")
        command[insertion:insertion] = [
            "--max-weight", f"{args.maximum_weight:.17g}"
        ]
    return command


def write_direct_cuda_provenance(
    args: argparse.Namespace,
    output: Path,
    command: list[str],
) -> None:
    encoded_command = json.dumps(
        command,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "schema_version": 1,
        "backend": "cuda",
        "executable": fingerprint(args.executable),
        "table": fingerprint(args.table),
        "antenna_file": fingerprint(args.antenna_file),
        "flupro": fingerprint(args.flupro / "libflukahp.a"),
        "runner": fingerprint(args.physics_runner),
        "command": command,
        "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
        "production_mode": "deferred_reference_comparison",
    }
    destination = output / "cuda" / "validation_provenance.json"
    if destination.exists():
        raise ValueError(f"refusing to overwrite CUDA provenance: {destination}")
    write_manifest(destination, payload)


def parse_summary_runtime_seconds(summary: dict[str, Any]) -> float:
    """Read both the legacy numeric and current duration-string schemas."""

    for key in ("runtime_raw", "runtime"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
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
                raise ValueError(f"invalid summary runtime duration: {value!r}")
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
    raise ValueError("summary does not contain a positive finite runtime")


def validate_completed_batch(
    output: Path,
    events: int,
    seed: int,
    immutable: dict[str, Any],
    *,
    deferred: bool = False,
) -> dict[str, Any]:
    cuda_required = (
        output / "cuda" / "summary.yaml",
        output / "cuda" / "validation_provenance.json",
        output / "cuda" / "profile" / "profile.parquet",
        output / "cuda" / "particles" / "particles.parquet",
        output / "cuda" / "gpu_em" / "summary.yaml",
        output / "cuda" / "CoREAS" / "summary.yaml",
        output / "cuda" / "ZHS" / "summary.yaml",
    )
    required = cuda_required if deferred else (
        output / "run_manifest.json",
        output / "comparison.json",
        output / "per_shower_observables.csv",
        *cuda_required,
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(
            f"incomplete batch {output}: " + ", ".join(str(path) for path in missing)
        )
    summary = read_yaml(output / "cuda" / "summary.yaml")
    if int(summary.get("showers", -1)) != events or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"batch count/seed mismatch in {output}")
    provenance = json.loads(
        (output / "cuda" / "validation_provenance.json").read_text(encoding="utf-8")
    )
    if (
        provenance.get("backend") != "cuda"
        or provenance.get("executable", {}).get("sha256")
        != immutable["artifacts"]["executable"]["sha256"]
        or provenance.get("table", {}).get("sha256")
        != immutable["artifacts"]["table"]["sha256"]
    ):
        raise ValueError(f"batch provenance mismatch in {output}")
    return {
        "output": str(output.resolve()),
        "cuda_output": str((output / "cuda").resolve()),
        "events": events,
        "seed": seed,
        "runtime_seconds": parse_summary_runtime_seconds(summary),
        "status": "complete",
        "comparison_status": "deferred" if deferred else "complete",
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.executable,
        args.proposal_executable,
        args.table,
        args.physics_runner,
        args.antenna_file,
        args.flupro / "libflukahp.a",
    ):
        if not path.is_file():
            raise ValueError(f"required file is missing: {path}")
    if args.target_events <= 0 or args.batch_events < 2 or args.cuda_seed_start < 0:
        raise ValueError("invalid event target, batch size, or CUDA seed")
    finite_positive = (
        args.energy_gev,
        args.geomagnetic_year,
        args.em_cut_gev,
        args.had_cut_gev,
        args.mu_cut_gev,
        args.tau_cut_gev,
        args.gpu_table_tolerance,
        args.gpu_radio_field_limit,
        args.cuda_hadronic_target_batch_ms,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
        raise ValueError("energy, cuts, tolerances, and timing targets must be positive")
    if not 0.0 <= args.em_thinning <= 1.0:
        raise ValueError("--em-thinning must be in [0, 1]")
    if args.maximum_weight < 0.0:
        raise ValueError("--maximum-weight must be non-negative")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be in (0, 1]")
    if min(
        args.gpu_min_batch,
        args.cuda_hadronic_workers,
        args.cuda_hadronic_min_batch,
        args.cuda_hadronic_max_batch,
    ) <= 0:
        raise ValueError("GPU and hadronic scheduler sizes must be positive")
    if not args.defer_reference_comparison and args.reference_proposal_root is None:
        raise ValueError(
            "--reference-proposal-root is required unless "
            "--defer-reference-comparison is selected"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".runner.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign is already running: {output_root}") from error
        lock.write(json.dumps({"pid": os.getpid(), "host": platform.node(), "utc": utc_now()}))
        lock.flush()

        immutable = immutable_configuration(args)
        initial_events = sum(
            int(record["events"]) for record in immutable["existing_cuda"]
        )
        if initial_events > args.target_events:
            raise ValueError("existing CUDA sources exceed the requested target")
        supplement_events = args.target_events - initial_events
        references = (
            []
            if args.defer_reference_comparison
            else reference_sources(
                args.reference_proposal_root,
                min(args.batch_events, supplement_events),
            )
        )
        batch_count = math.ceil(supplement_events / args.batch_events)
        manifest_path = output_root / "campaign_manifest.json"
        if manifest_path.is_file():
            campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
            if campaign.get("immutable_configuration") != immutable:
                raise ValueError("campaign configuration/artifact hash changed; refusing resume")
        else:
            campaign = {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "initial_events": initial_events,
                "supplement_events_requested": supplement_events,
                "immutable_configuration": immutable,
                "batches": [],
            }
            write_manifest(manifest_path, campaign)

        completed: list[dict[str, Any]] = []
        remaining = supplement_events
        seed = args.cuda_seed_start
        for index in range(batch_count):
            events = min(args.batch_events, remaining)
            batch_root = output_root / f"batch_{index:03d}"
            if batch_root.exists():
                record = validate_completed_batch(
                    batch_root,
                    events,
                    seed,
                    immutable,
                    deferred=args.defer_reference_comparison,
                )
                print(f"reuse complete batch {index:03d}: {events} events", flush=True)
            else:
                current = immutable_configuration(args)
                if current != immutable:
                    raise RuntimeError("immutable executable/table/configuration changed mid-campaign")
                command = (
                    direct_batch_command(args, batch_root, events, seed)
                    if args.defer_reference_comparison
                    else batch_command(
                        args, batch_root, events, seed, references
                    )
                )
                print(
                    f"start batch {index:03d}/{batch_count-1:03d}: "
                    f"events={events} seed={seed}",
                    flush=True,
                )
                log_path = output_root / f"batch_{index:03d}.runner.log"
                if args.defer_reference_comparison:
                    batch_root.mkdir()
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
                with log_path.open("w", encoding="utf-8") as log:
                    completed_process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                    )
                if completed_process.returncode != 0:
                    campaign["status"] = "failed"
                    campaign["failure"] = {
                        "batch": index,
                        "return_code": completed_process.returncode,
                        "log": str(log_path),
                    }
                    campaign["updated_utc"] = utc_now()
                    write_manifest(manifest_path, campaign)
                    raise RuntimeError(
                        f"batch {index:03d} failed with exit code "
                        f"{completed_process.returncode}; see {log_path}"
                    )
                if args.defer_reference_comparison:
                    write_direct_cuda_provenance(args, batch_root, command)
                record = validate_completed_batch(
                    batch_root,
                    events,
                    seed,
                    immutable,
                    deferred=args.defer_reference_comparison,
                )
                print(
                    f"complete batch {index:03d}: "
                    f"runtime={record['runtime_seconds']:.1f}s",
                    flush=True,
                )
            completed.append(record)
            remaining -= events
            seed += events
            campaign["batches"] = completed
            campaign["completed_supplement_events"] = sum(
                int(item["events"]) for item in completed
            )
            campaign["completed_total_events"] = (
                initial_events + campaign["completed_supplement_events"]
            )
            campaign["status"] = (
                "complete" if remaining == 0 else "running"
            )
            campaign["updated_utc"] = utc_now()
            write_manifest(manifest_path, campaign)

        print(
            json.dumps(
                {
                    "status": campaign["status"],
                    "completed_total_events": campaign["completed_total_events"],
                    "batches": len(completed),
                    "manifest": str(manifest_path),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
