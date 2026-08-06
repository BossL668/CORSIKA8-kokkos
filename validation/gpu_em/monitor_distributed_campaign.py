#!/usr/bin/env python3
"""Incrementally stage completed remote CPU shards and monitor local CUDA runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--ssh-control-path", required=True, type=Path)
    parser.add_argument(
        "--remote-campaign",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Remote runner root; repeat for main/preflight campaigns.",
    )
    parser.add_argument(
        "--local-campaign",
        action="append",
        default=[],
        type=Path,
        help="Local CUDA campaign root; repeat for independent shards.",
    )
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--status-json", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def parse_labeled_path(specification: str) -> tuple[str, str]:
    if "=" not in specification:
        raise ValueError(
            f"remote campaign must use LABEL=PATH: {specification}"
        )
    label, path = specification.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path or "/" in label or label in {".", ".."}:
        raise ValueError(f"invalid remote campaign: {specification}")
    if not path.startswith("/"):
        raise ValueError(f"remote campaign path must be absolute: {path}")
    return label, path.rstrip("/")


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        value = yaml.load(source, Loader=loader)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def remote_json(
    host: str,
    control_path: Path,
    path: str,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ssh",
            "-S",
            str(control_path),
            "-o",
            "ConnectTimeout=15",
            host,
            f"cat {shlex.quote(path)}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"remote JSON is not a mapping: {path}")
    return value


def expected_seed(manifest: dict[str, Any], index: int) -> int:
    configuration = manifest.get("immutable_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("remote manifest lacks immutable_configuration")
    schedule = configuration.get("seed_schedule")
    if isinstance(schedule, list):
        if index < 0 or index >= len(schedule):
            raise ValueError(f"shard index outside seed schedule: {index}")
        return int(schedule[index])
    start = configuration.get("seed_start")
    if start is None:
        raise ValueError("remote manifest lacks seed schedule/start")
    return int(start) + index


def validate_staged_shard(
    root: Path,
    expected_seed_value: int,
    expected_executable_sha256: str,
) -> dict[str, Any]:
    required = (
        root / "summary.yaml",
        root / "config.yaml",
        root / "validation_provenance.json",
        root / "profile" / "profile.parquet",
        root / "production_profile" / "profile.parquet",
        root / "particles" / "particles.parquet",
        root / "energyloss" / "dEdX.parquet",
        root / "simulation_timing" / "summary.yaml",
        root / "CoREAS" / "summary.yaml",
        root / "ZHS" / "summary.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("staged shard is incomplete: " + ", ".join(missing))
    summary = read_yaml_mapping(root / "summary.yaml")
    timing = read_yaml_mapping(root / "simulation_timing" / "summary.yaml")
    shower = timing.get("shower_0")
    provenance = json.loads(
        (root / "validation_provenance.json").read_text(encoding="utf-8")
    )
    if int(summary.get("showers", -1)) != 1:
        raise ValueError(f"unexpected shower count in {root}")
    if int(summary.get("seed", -1)) != expected_seed_value:
        raise ValueError(f"summary seed differs in {root}")
    if (
        not isinstance(shower, dict)
        or shower.get("closed") is not True
        or shower.get("status") != "closed"
        or not math.isfinite(float(shower.get("wall_time_ms", math.nan)))
        or float(shower["wall_time_ms"]) <= 0.0
    ):
        raise ValueError(f"simulation timing is not closed in {root}")
    if (
        not isinstance(provenance, dict)
        or provenance.get("backend") != "proposal"
        or int(provenance.get("seed", -1)) != expected_seed_value
        or provenance.get("executable", {}).get("sha256")
        != expected_executable_sha256
    ):
        raise ValueError(f"validation provenance differs in {root}")
    return {
        "path": str(root.resolve()),
        "seed": expected_seed_value,
        # Older campaign summaries used ``runtime_raw`` while current
        # c8_air_shower writes ``runtime``.  Preserve both schemas so the
        # staging status never turns a valid measured runtime into zero.
        "runtime_seconds": float(
            summary.get("runtime_raw", summary.get("runtime", 0.0))
        ),
        "wall_time_seconds": float(shower["wall_time_ms"]) / 1000.0,
        "executable_sha256": expected_executable_sha256,
    }


def stage_completed_campaign(
    *,
    host: str,
    control_path: Path,
    label: str,
    remote_root: str,
    staging_root: Path,
) -> dict[str, Any]:
    manifest = remote_json(
        host,
        control_path,
        f"{remote_root}/run_manifest.json",
    )
    configuration = manifest["immutable_configuration"]
    executable_sha256 = str(configuration["executable"]["sha256"])
    completed_indices = sorted(
        {int(value) for value in manifest.get("completed_indices", [])}
    )
    destination_root = staging_root / label
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in completed_indices:
        seed = expected_seed(manifest, index)
        destination = destination_root / f"proposal_shard_{index:03d}"
        try:
            record = validate_staged_shard(
                destination,
                seed,
                executable_sha256,
            )
            record["transferred_this_poll"] = False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    f"ssh -S {control_path}",
                    f"{host}:{remote_root}/proposal_shard_{index:03d}/",
                    f"{destination}/",
                ],
                check=True,
            )
            record = validate_staged_shard(
                destination,
                seed,
                executable_sha256,
            )
            record["transferred_this_poll"] = True
        record["index"] = index
        record["source_label"] = label
        records.append(record)
    return {
        "label": label,
        "remote_root": remote_root,
        "remote_status": manifest.get("status"),
        "remote_completed": int(manifest.get("completed", 0)),
        "remote_failed": int(manifest.get("failed", 0)),
        "remote_failures": manifest.get("failures", []),
        "executable_sha256": executable_sha256,
        "staged_events": len(records),
        "records": records,
    }


def local_campaign_status(path: Path) -> dict[str, Any]:
    manifest_path = path.resolve() / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable = manifest.get("immutable_configuration", {})
    return {
        "path": str(path.resolve()),
        "status": manifest.get("status"),
        "completed_events": int(manifest.get("completed_total_events", 0)),
        "completed_batches": len(manifest.get("batches", [])),
        "target_events": int(immutable.get("target_events", 0)),
        "seed_start": immutable.get("cuda_seed_start"),
        "failure": manifest.get("failure"),
        "updated_utc": manifest.get("updated_utc"),
    }


def poll(args: argparse.Namespace) -> dict[str, Any]:
    remote = [
        stage_completed_campaign(
            host=args.remote_host,
            control_path=args.ssh_control_path,
            label=label,
            remote_root=path,
            staging_root=args.staging_root,
        )
        for label, path in args.parsed_remote_campaigns
    ]
    local = [local_campaign_status(path) for path in args.local_campaign]
    return {
        "schema_version": 1,
        "status": "monitoring",
        "updated_utc": utc_now(),
        "remote_host": args.remote_host,
        "remote": remote,
        "remote_staged_events": sum(item["staged_events"] for item in remote),
        "local": local,
        "local_completed_events": sum(item["completed_events"] for item in local),
        "local_target_events": sum(item["target_events"] for item in local),
    }


def compact_status(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("status") != "monitoring":
        return value
    return {
        "status": value["status"],
        "updated_utc": value["updated_utc"],
        "remote": [
            {
                "label": item["label"],
                "status": item["remote_status"],
                "completed": item["remote_completed"],
                "failed": item["remote_failed"],
                "staged": item["staged_events"],
                "transferred_this_poll": sum(
                    1
                    for record in item["records"]
                    if record["transferred_this_poll"]
                ),
            }
            for item in value["remote"]
        ],
        "local": [
            {
                "path": item["path"],
                "status": item["status"],
                "completed": item["completed_events"],
                "target": item["target_events"],
            }
            for item in value["local"]
        ],
    }


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [1, 60]")
    args.ssh_control_path = args.ssh_control_path.resolve()
    args.staging_root = args.staging_root.resolve()
    args.status_json = args.status_json.resolve()
    args.parsed_remote_campaigns = [
        parse_labeled_path(value) for value in args.remote_campaign
    ]
    labels = [label for label, _ in args.parsed_remote_campaigns]
    if len(labels) != len(set(labels)):
        raise ValueError("remote campaign labels must be unique")
    if not args.ssh_control_path.exists():
        raise ValueError(
            f"SSH control socket is missing: {args.ssh_control_path}"
        )
    while True:
        try:
            status = poll(args)
        except Exception as error:  # noqa: BLE001 - retain transient monitor state
            status = {
                "schema_version": 1,
                "status": "poll_failed",
                "updated_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        atomic_json(args.status_json, status)
        print(json.dumps(compact_status(status), sort_keys=True), flush=True)
        if args.once:
            return 0 if status["status"] == "monitoring" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
