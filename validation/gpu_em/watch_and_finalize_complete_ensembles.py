#!/usr/bin/env python3
"""Wait for exact distributed ensembles and invoke the fail-closed finalizer.

This watcher is intentionally small and generic.  It consumes the status
written by ``monitor_distributed_campaign.py`` and the authoritative local
CUDA campaign manifests.  It never analyzes completion-conditioned partial
samples.  Finalization starts only when all requested CPU shards have closed
and been staged, all CUDA batches have closed, both event totals are exact,
and neither runner reports a failure.
"""

from __future__ import annotations

import argparse
import hashlib
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed-status", type=Path, required=True)
    parser.add_argument("--local-campaign", type=Path, action="append", required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--finalizer-script", type=Path, required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, action="append", required=True)
    parser.add_argument("--cuda-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--proposal-seed-start", type=int, required=True)
    parser.add_argument("--cuda-seed-start", type=int, required=True)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--energy-gev", type=float, required=True)
    parser.add_argument("--zenith-deg", type=float, required=True)
    parser.add_argument("--azimuth-deg", type=float, required=True)
    parser.add_argument("--geomagnetic-model", choices=("IGRF13", "IGRF14"), required=True)
    parser.add_argument("--geomagnetic-year", type=float, required=True)
    parser.add_argument("--em-cut-gev", type=float, default=5.0e-4)
    parser.add_argument("--em-thinning", type=float, required=True)
    parser.add_argument("--maximum-weight", type=float, default=0.0)
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--antenna-sha256", required=True)
    parser.add_argument("--pulse-analysis-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def local_cuda_state(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        manifest = read_json(path / "campaign_manifest.json")
        immutable = manifest.get("immutable_configuration")
        if not isinstance(immutable, dict):
            raise ValueError(f"CUDA manifest lacks immutable configuration: {path}")
        records.append(
            {
                "path": str(path),
                "status": manifest.get("status"),
                "completed_events": int(manifest.get("completed_total_events", 0)),
                "target_events": int(immutable.get("target_events", 0)),
                "failed": manifest.get("failure") is not None,
                "failure": manifest.get("failure"),
            }
        )
    return records


def completion_state(
    distributed: dict[str, Any],
    local: list[dict[str, Any]],
    expected_events: int,
) -> dict[str, Any]:
    if distributed.get("status") != "monitoring":
        return {
            "ready": False,
            "terminal_failure": False,
            "reason": f"distributed monitor status={distributed.get('status')}",
        }
    remote = distributed.get("remote")
    if not isinstance(remote, list) or not remote:
        raise ValueError("distributed status contains no remote campaigns")
    remote_completed = sum(int(item.get("remote_completed", 0)) for item in remote)
    remote_staged = sum(int(item.get("staged_events", 0)) for item in remote)
    remote_failed = sum(int(item.get("remote_failed", 0)) for item in remote)
    remote_terminal = any(item.get("remote_status") == "failed" for item in remote)
    local_completed = sum(int(item["completed_events"]) for item in local)
    local_targets = sum(int(item["target_events"]) for item in local)
    local_failed = any(bool(item["failed"]) or item["status"] == "failed" for item in local)
    counts = {
        "remote_completed": remote_completed,
        "remote_staged": remote_staged,
        "remote_failed": remote_failed,
        "local_completed": local_completed,
        "local_targets": local_targets,
    }
    if max(remote_completed, remote_staged, local_completed) > expected_events:
        return {
            "ready": False,
            "terminal_failure": True,
            "reason": "an ensemble exceeds the exact requested event count",
            **counts,
        }
    if local_targets != expected_events:
        return {
            "ready": False,
            "terminal_failure": True,
            "reason": "local CUDA target differs from expected event count",
            **counts,
        }
    if remote_failed or remote_terminal or local_failed:
        return {
            "ready": False,
            "terminal_failure": True,
            "reason": "a CPU or CUDA runner reports failure",
            **counts,
        }
    ready = bool(
        remote_completed == expected_events
        and remote_staged == expected_events
        and all(item.get("remote_status") == "complete" for item in remote)
        and local_completed == expected_events
        and all(item["status"] == "complete" for item in local)
    )
    return {
        "ready": ready,
        "terminal_failure": False,
        "reason": "complete ensembles are ready" if ready else "waiting for exact ensembles",
        **counts,
    }


def finalizer_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(args.finalizer_script),
        "--final-root",
        str(args.final_root),
    ]
    for root in args.proposal_root:
        command.extend(("--proposal-root", str(root)))
    for root in args.cuda_root:
        command.extend(("--cuda-root", str(root)))
    command.extend(
        (
            "--expected-events", str(args.expected_events),
            "--proposal-seed-start", str(args.proposal_seed_start),
            "--cuda-seed-start", str(args.cuda_seed_start),
            "--primary-pdg", str(args.primary_pdg),
            "--energy-gev", f"{args.energy_gev:.17g}",
            "--zenith-deg", f"{args.zenith_deg:.17g}",
            "--azimuth-deg", f"{args.azimuth_deg:.17g}",
            "--geomagnetic-model", args.geomagnetic_model,
            "--geomagnetic-year", f"{args.geomagnetic_year:.17g}",
            "--em-cut-gev", f"{args.em_cut_gev:.17g}",
            "--em-thinning", f"{args.em_thinning:.17g}",
            "--maximum-weight", f"{args.maximum_weight:.17g}",
            "--had-cut-gev", f"{args.had_cut_gev:.17g}",
            "--mu-cut-gev", f"{args.mu_cut_gev:.17g}",
            "--tau-cut-gev", f"{args.tau_cut_gev:.17g}",
            "--ring", str(args.ring),
            "--antenna-sha256", args.antenna_sha256,
            "--pulse-analysis-root", str(args.pulse_analysis_root),
            "--bootstrap-repetitions", str(args.bootstrap_repetitions),
            "--execute",
        )
    )
    return command


def main() -> int:
    args = parse_args()
    if args.expected_events <= 1 or args.bootstrap_repetitions <= 0:
        raise ValueError("event and bootstrap counts must be positive")
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [1, 60]")
    for name in (
        "distributed_status", "status_json", "finalizer_script", "final_root",
        "pulse_analysis_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.local_campaign = [path.resolve() for path in args.local_campaign]
    args.proposal_root = [path.resolve() for path in args.proposal_root]
    args.cuda_root = [path.resolve() for path in args.cuda_root]
    observed_hash = sha256_file(args.finalizer_script)
    if observed_hash != args.expected_finalizer_sha256:
        raise ValueError(
            "finalizer script hash differs: "
            f"expected {args.expected_finalizer_sha256}, observed {observed_hash}"
        )
    base_status: dict[str, Any] = {
        "schema_version": 1,
        "finalizer_script": str(args.finalizer_script),
        "finalizer_sha256": observed_hash,
        "final_root": str(args.final_root),
        "expected_events_per_backend": args.expected_events,
    }
    while True:
        try:
            distributed = read_json(args.distributed_status)
            local = local_cuda_state(args.local_campaign)
            state = completion_state(distributed, local, args.expected_events)
            status = {
                **base_status,
                "status": "ready" if state["ready"] else "waiting",
                "updated_utc": utc_now(),
                "completion": state,
                "local": local,
                "distributed_status_updated_utc": distributed.get("updated_utc"),
            }
            if state["terminal_failure"]:
                status["status"] = "failed"
                atomic_json(args.status_json, status)
                return 1
            if state["ready"]:
                command = finalizer_command(args)
                log_path = args.status_json.with_suffix(".finalizer.log")
                status.update(
                    {
                        "status": "finalizing",
                        "finalizer_command": command,
                        "finalizer_log": str(log_path),
                    }
                )
                atomic_json(args.status_json, status)
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                status["updated_utc"] = utc_now()
                status["finalizer_returncode"] = result.returncode
                status["status"] = "complete" if result.returncode == 0 else "failed"
                atomic_json(args.status_json, status)
                return result.returncode
            atomic_json(args.status_json, status)
            print(json.dumps(status["completion"], sort_keys=True), flush=True)
        except Exception as error:  # retain transient monitor failures
            status = {
                **base_status,
                "status": "poll_failed",
                "updated_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            atomic_json(args.status_json, status)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
