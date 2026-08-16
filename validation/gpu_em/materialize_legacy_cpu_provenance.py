#!/usr/bin/env python3
"""Reconstruct auditable provenance sidecars for one legacy CPU campaign.

This tool is deliberately narrow and fail closed.  It accepts only the legacy
one-shower-per-directory layout whose campaign status, per-shower CORSIKA
configuration, timing summary and wrapper log all agree.  It never changes a
physics output.  Without ``--execute`` it performs the complete audit and only
prints the report that would be written.

The reconstructed sidecar is explicitly marked as post-hoc provenance.  It is
therefore not equivalent to provenance captured before execution, but it makes
the surviving evidence and its limitations machine-readable instead of
silently treating an archived sample as unprovenanced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
SIDECAR = "validation_provenance.json"
REPORT = "legacy_provenance_materialization_report.json"
SHARD_PATTERN = re.compile(r"proposal_shard_(\d+)_seed(\d+)$")
REQUIRED_OUTPUTS = (
    "config.yaml",
    "summary.yaml",
    "simulation_timing/summary.yaml",
    "profile/profile.parquet",
    "production_profile/profile.parquet",
    "energyloss/dEdX.parquet",
    "particles/particles.parquet",
    "CoREAS/config.yaml",
    "CoREAS/summary.yaml",
    "CoREAS/observers.parquet",
    "ZHS/config.yaml",
    "ZHS/summary.yaml",
    "ZHS/observers.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--antenna-file", type=Path, required=True)
    parser.add_argument("--flupro", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write missing sidecars after every shard has passed the audit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required immutable artifact is missing: {resolved}")
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required YAML is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def command_option(command: list[str], option: str) -> str | None:
    if option not in command:
        return None
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"option has no value: {option}")
    return command[index + 1]


def require_number(
    command: list[str], option: str, expected: float, *, label: str
) -> None:
    value = command_option(command, option)
    if value is None:
        raise ValueError(f"required option is absent ({label}): {option}")
    observed = float(value)
    if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            f"{label} differs: expected {expected:.17g}, observed {observed:.17g}"
        )


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, value: Any) -> None:
    with path.open("xb") as destination:
        destination.write(json_bytes(value))
        destination.flush()
        os.fsync(destination.fileno())


def write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as destination:
        destination.write(json_bytes(value))
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def physics_payload(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_pdg": int(parameters["pdg"]),
        "energy_GeV": float(parameters["energy_GeV"]),
        "zenith_deg": float(parameters["zenith_deg"]),
        "azimuth_deg": float(parameters["azimuth_deg"]),
        "geomagnetic_model": str(parameters["geomagnetic_model"]),
        "geomagnetic_year": float(parameters["geomagnetic_year"]),
        "shower_core_x_m": 0.0,
        "shower_core_y_m": 0.0,
        "ring": 0,
        "em_cut_GeV": float(parameters["emcut_GeV"]),
        "em_thinning": float(parameters["emthin"]),
        "had_cut_GeV": float(parameters["hadcut_GeV"]),
        "mu_cut_GeV": float(parameters["mucut_GeV"]),
        "tau_cut_GeV": float(parameters["taucut_GeV"]),
        "maximum_weight": 0.0,
        "maximum_weight_cli_omitted": True,
        "em_backend": "legacy_scalar_proposal",
        "radio_backend": "cpu_coreas_and_zhs",
    }


def audit_shard(
    *,
    root: Path,
    index: int,
    seed: int,
    status: dict[str, Any],
    executable: dict[str, Any],
    antenna: dict[str, Any],
    flupro: dict[str, Any],
    physics: dict[str, Any],
) -> dict[str, Any]:
    match = SHARD_PATTERN.fullmatch(root.name)
    if match is None or int(match.group(1)) != index or int(match.group(2)) != seed:
        raise ValueError(f"shard directory identity differs: {root}")
    for relative in REQUIRED_OUTPUTS:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 4:
            raise ValueError(f"missing or empty physics output: {path}")

    summary = read_yaml(root / "summary.yaml")
    if int(summary.get("showers", -1)) != 1 or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"summary seed/count differs: {root}")
    if float(summary.get("runtime", 0.0)) <= 0.0:
        raise ValueError(f"non-positive summary runtime: {root}")
    timing = read_yaml(root / "simulation_timing/summary.yaml")
    shower = timing.get("shower_0")
    if not isinstance(shower, dict) or shower.get("closed") is not True:
        raise ValueError(f"unclosed shower timing: {root}")
    if float(shower.get("wall_time_ms", 0.0)) <= 0.0:
        raise ValueError(f"non-positive shower wall time: {root}")

    config = read_yaml(root / "config.yaml")
    command_text = config.get("args")
    if not isinstance(command_text, str):
        raise ValueError(f"missing exact CORSIKA command: {root / 'config.yaml'}")
    command = shlex.split(command_text)
    if not command or Path(command[0]).resolve() != Path(executable["path"]):
        raise ValueError(f"configured executable differs: {root}")
    require_number(command, "-p", physics["primary_pdg"], label="primary PDG")
    require_number(command, "-E", physics["energy_GeV"], label="primary energy")
    require_number(command, "-N", 1.0, label="event count")
    require_number(command, "--seed", seed, label="seed")
    require_number(command, "--zenith", physics["zenith_deg"], label="zenith")
    require_number(command, "--azimuth", physics["azimuth_deg"], label="azimuth")
    require_number(command, "--geomagnetic-year", physics["geomagnetic_year"], label="year")
    require_number(command, "--emcut", physics["em_cut_GeV"], label="EM cut")
    require_number(command, "--emthin", physics["em_thinning"], label="EM thinning")
    require_number(command, "--hadcut", physics["had_cut_GeV"], label="hadron cut")
    require_number(command, "--mucut", physics["mu_cut_GeV"], label="muon cut")
    require_number(command, "--taucut", physics["tau_cut_GeV"], label="tau cut")
    require_number(command, "--shower-core-x", 0.0, label="core x")
    require_number(command, "--shower-core-y", 0.0, label="core y")
    require_number(command, "--ring", 0.0, label="ring")
    required_tokens = {
        "--geomagnetic-model": physics["geomagnetic_model"],
        "--antenna-file": antenna["path"],
        "--em-backend": "proposal",
        "--radio-backend": "cpu",
        "--hadronic-backend": "scalar",
    }
    for option, expected in required_tokens.items():
        observed = command_option(command, option)
        if observed != str(expected):
            raise ValueError(
                f"{option} differs in {root}: expected {expected!r}, observed {observed!r}"
            )
    if command_option(command, "-f") != str(root):
        raise ValueError(f"output path differs in command: {root}")
    if "--max-weight" in command:
        raise ValueError(f"legacy campaign unexpectedly sets --max-weight: {root}")

    log_path = Path(str(status["log"]))
    if not log_path.is_file():
        raise ValueError(f"wrapper log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="strict")
    lines = log_text.splitlines()
    command_lines = [line[len("command=") :] for line in lines if line.startswith("command=")]
    if len(command_lines) != 1:
        raise ValueError(f"wrapper log has no unique command: {log_path}")
    wrapped = shlex.split(command_lines[0])
    expected_prefix = ["taskset", "-c", str(status["core"]), "/usr/bin/time", "-v"]
    if wrapped[: len(expected_prefix)] != expected_prefix or wrapped[len(expected_prefix) :] != command:
        raise ValueError(f"wrapper/config command mismatch: {log_path}")
    if not lines or lines[-1].split()[:2] != ["end", "status=0"]:
        raise ValueError(f"wrapper log does not end successfully: {log_path}")
    if "\tExit status: 0" not in log_text:
        raise ValueError(f"/usr/bin/time did not record exit status 0: {log_path}")

    encoded_command = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "proposal",
        "seed": seed,
        "events": 1,
        "cpu": int(status["core"]),
        "executable": executable,
        "table": None,
        "antenna_file": antenna,
        "flupro": flupro,
        "physics": physics,
        "command": command,
        "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
        "provenance_capture": {
            "mode": "post_hoc_archive_reconstruction",
            "limitation": (
                "Artifact hashes were captured after simulation; identity is supported "
                "by the archived absolute command path and pre-run artifact mtime, not "
                "by a pre-execution hash sidecar."
            ),
            "campaign_status_sha256": status["campaign_status_sha256"],
            "campaign_parameters_sha256": status["campaign_parameters_sha256"],
            "wrapper_log": {
                "path": str(log_path.resolve()),
                "sha256": sha256_file(log_path),
            },
        },
    }


def main() -> int:
    args = parse_args()
    if args.expected_events <= 0 or args.seed_start < 0:
        raise ValueError("expected event count must be positive and seed non-negative")
    root = args.campaign_root.resolve()
    parameters_path = root / "campaign_parameters.json"
    status_path = root / "campaign_status.json"
    parameters = read_json(parameters_path)
    campaign_status = read_json(status_path)
    if int(parameters.get("events", -1)) != args.expected_events:
        raise ValueError("campaign_parameters event count differs")
    if int(parameters.get("seed_start", -1)) != args.seed_start:
        raise ValueError("campaign_parameters seed start differs")
    if int(campaign_status.get("requested", -1)) != args.expected_events:
        raise ValueError("campaign_status requested count differs")
    if int(campaign_status.get("failed", -1)) != 0:
        raise ValueError("campaign_status contains failed showers")
    if int(campaign_status.get("complete_or_reused", -1)) != args.expected_events:
        raise ValueError("campaign_status is incomplete")
    statuses = campaign_status.get("events")
    if not isinstance(statuses, list) or len(statuses) != args.expected_events:
        raise ValueError("campaign_status event records are incomplete")

    executable = artifact_identity(args.executable)
    antenna = artifact_identity(args.antenna_file)
    flupro = artifact_identity(args.flupro.resolve() / "libflukahp.a")
    if Path(str(parameters.get("executable", ""))).resolve() != Path(executable["path"]):
        raise ValueError("campaign executable path differs from --executable")
    if Path(str(parameters.get("antenna_file", ""))).resolve() != Path(antenna["path"]):
        raise ValueError("campaign antenna path differs from --antenna-file")

    physics = physics_payload(parameters)
    common_evidence = {
        "campaign_status_sha256": sha256_file(status_path),
        "campaign_parameters_sha256": sha256_file(parameters_path),
    }
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for index, status_record in enumerate(statuses):
        if not isinstance(status_record, dict):
            raise ValueError(f"invalid campaign status event record: {index}")
        seed = args.seed_start + index
        if (
            int(status_record.get("index", -1)) != index
            or int(status_record.get("seed", -1)) != seed
            or status_record.get("status") != "complete"
            or int(status_record.get("returncode", -1)) != 0
        ):
            raise ValueError(f"campaign status differs for event index {index}")
        enriched = dict(status_record)
        enriched.update(common_evidence)
        shard = root / f"proposal_shard_{index:04d}_seed{seed}"
        payload = audit_shard(
            root=shard,
            index=index,
            seed=seed,
            status=enriched,
            executable=executable,
            antenna=antenna,
            flupro=flupro,
            physics=physics,
        )
        sidecar = shard / SIDECAR
        if sidecar.exists():
            if sidecar.read_bytes() != json_bytes(payload):
                raise ValueError(f"existing reconstructed provenance differs: {sidecar}")
        payloads.append((sidecar, payload))

    missing = sum(not sidecar.exists() for sidecar, _ in payloads)
    report = {
        "schema_version": 1,
        "status": "pass",
        "mode": "execute" if args.execute else "dry_run",
        "generated_utc": utc_now(),
        "campaign_root": str(root),
        "expected_events": args.expected_events,
        "seed_start": args.seed_start,
        "audited_shards": len(payloads),
        "missing_sidecars_before": missing,
        "written_sidecars": missing if args.execute else 0,
        "executable": executable,
        "antenna_file": antenna,
        "flupro": flupro,
        "campaign_parameters_sha256": common_evidence["campaign_parameters_sha256"],
        "campaign_status_sha256": common_evidence["campaign_status_sha256"],
        "provenance_quality": "post_hoc_archive_reconstruction",
    }
    if args.execute:
        for sidecar, payload in payloads:
            if not sidecar.exists():
                write_exclusive(sidecar, payload)
        write_atomic(root / REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
