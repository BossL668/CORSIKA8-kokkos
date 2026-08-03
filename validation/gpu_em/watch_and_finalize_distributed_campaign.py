#!/usr/bin/env python3
"""Stage exact remote reruns and finalize a distributed CPU/CUDA campaign.

The original campaign monitor continuously stages successful scalar shards.
This second watcher covers the deliberately separate writer-fix rerun campaign.
It waits for the failed-seed launcher, incrementally stages those exact seeds,
and invokes the fail-closed finalizer only after the original successes, reruns,
and both local CUDA shards form the requested complete ensembles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .finalize_distributed_campaign import audit_source
except ImportError:  # Direct execution from validation/gpu_em.
    from finalize_distributed_campaign import audit_source  # type: ignore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"invalid {label} SHA-256")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--ssh-control-path", type=Path, required=True)
    parser.add_argument("--rerun-watcher-status", type=Path, required=True)
    parser.add_argument("--remote-rerun-root", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--original-monitor-status", type=Path, required=True)
    parser.add_argument(
        "--precompleted-monitor-status",
        type=Path,
        action="append",
        default=[],
        help=(
            "Status from a monitor staging writer-fix seeds repaired before "
            "the original campaigns became terminal; repeatable."
        ),
    )
    parser.add_argument("--local-campaign", type=Path, action="append", required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--rerun-monitor-status", type=Path, required=True)
    parser.add_argument("--monitor-script", type=Path, required=True)
    parser.add_argument("--finalizer-script", type=Path, required=True)
    parser.add_argument("--expected-monitor-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--proposal-seed-start", type=int, required=True)
    parser.add_argument("--cuda-seed-start", type=int, required=True)
    parser.add_argument("--antenna-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--pulse-analysis-root", type=Path, required=True)
    return parser.parse_args()


def validate_local_artifacts(args: argparse.Namespace) -> dict[str, str]:
    expected_monitor = require_sha256(
        args.expected_monitor_sha256, "monitor script"
    )
    expected_finalizer = require_sha256(
        args.expected_finalizer_sha256, "finalizer script"
    )
    antenna = require_sha256(args.antenna_sha256, "antenna")
    observed = {
        "completion_watcher_script": sha256_file(Path(__file__).resolve()),
        "monitor_script": sha256_file(args.monitor_script),
        "finalizer_script": sha256_file(args.finalizer_script),
    }
    if observed["monitor_script"] != expected_monitor:
        raise ValueError("remote staging monitor script hash changed")
    if observed["finalizer_script"] != expected_finalizer:
        raise ValueError("distributed finalizer script hash changed")
    observed["antenna"] = antenna
    return observed


def local_cuda_state(paths: list[Path]) -> dict[str, Any]:
    records = []
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
                "seed_start": int(immutable.get("cuda_seed_start", -1)),
                "failure": manifest.get("failure"),
            }
        )
    return {
        "records": records,
        "completed_events": sum(record["completed_events"] for record in records),
        "target_events": sum(record["target_events"] for record in records),
    }


def precompleted_repair_state(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    campaigns: list[dict[str, Any]] = []
    for path in paths:
        status = read_json(path)
        if status.get("status") not in {"monitoring", "complete"}:
            raise ValueError(f"precompleted repair monitor is unhealthy: {path}")
        remote = status.get("remote")
        if not isinstance(remote, list):
            raise ValueError(f"precompleted repair monitor lacks campaigns: {path}")
        for campaign in remote:
            if not isinstance(campaign, dict):
                raise ValueError(f"invalid precompleted repair campaign: {path}")
            observed = campaign.get("records")
            if not isinstance(observed, list):
                raise ValueError(f"precompleted repair campaign lacks records: {path}")
            for record in observed:
                if not isinstance(record, dict) or "seed" not in record:
                    raise ValueError(f"invalid precompleted repair record: {path}")
                records.append(record)
            campaigns.append(
                {
                    "monitor_status": str(path),
                    "label": campaign.get("label"),
                    "remote_status": campaign.get("remote_status"),
                    "remote_failed": int(campaign.get("remote_failed", 0)),
                    "staged_events": int(campaign.get("staged_events", 0)),
                }
            )
    seeds = [int(record["seed"]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("precompleted repair monitors contain duplicate staged seeds")
    return {
        "campaigns": campaigns,
        "staged_events": len(seeds),
        "staged_seeds": sorted(seeds),
    }


def audit_new_cuda_batches(
    paths: list[Path],
    cache: dict[str, dict[str, Any]],
    *,
    antenna_sha256: str,
) -> dict[str, Any]:
    expected = SimpleNamespace(
        antenna_sha256=antenna_sha256,
        primary_pdg=2212,
        energy_gev=1.0e8,
        zenith_deg=47.0,
        azimuth_deg=180.0,
        em_cut_gev=5.0e-4,
        em_thinning=1.0e-4,
        had_cut_gev=0.3,
        mu_cut_gev=0.3,
        tau_cut_gev=0.3,
        ring=0,
        geomagnetic_model="IGRF13",
        geomagnetic_year=2025.0,
    )
    observed_roots: set[str] = set()
    for campaign_path in paths:
        manifest = read_json(campaign_path / "campaign_manifest.json")
        immutable = manifest.get("immutable_configuration")
        batches = manifest.get("batches", [])
        if not isinstance(immutable, dict) or not isinstance(batches, list):
            raise ValueError(f"invalid completed CUDA batch manifest: {campaign_path}")
        expected_seed = int(immutable.get("cuda_seed_start", -1))
        completed_events = 0
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError(f"invalid CUDA batch record: {campaign_path}")
            root_text = batch.get("cuda_output")
            if not isinstance(root_text, str):
                raise ValueError(f"CUDA batch lacks output path: {campaign_path}")
            root = Path(root_text).resolve()
            key = str(root)
            if key in observed_roots:
                raise ValueError(f"duplicate CUDA batch output in manifests: {root}")
            observed_roots.add(key)
            events = int(batch.get("events", -1))
            seed = int(batch.get("seed", -1))
            if events <= 0 or seed != expected_seed or batch.get("status") != "complete":
                raise ValueError(f"non-contiguous or incomplete CUDA batch: {root}")
            if key not in cache:
                record = audit_source(root, "cuda", expected=expected)
                if record.events != events or record.seed != seed:
                    raise ValueError(f"strict CUDA audit differs from batch record: {root}")
                cache[key] = asdict(record)
            completed_events += events
            expected_seed += events
        if completed_events != int(manifest.get("completed_total_events", 0)):
            raise ValueError(
                f"CUDA manifest total differs from strictly audited batches: {campaign_path}"
            )
    disappeared = sorted(set(cache).difference(observed_roots))
    if disappeared:
        raise ValueError(f"previously audited CUDA batches disappeared: {disappeared[:5]}")
    records = list(cache.values())
    return {
        "strictly_audited_batches": len(records),
        "strictly_audited_events": sum(int(record["events"]) for record in records),
        "executable_sha256": sorted(
            {str(record["executable_sha256"]) for record in records}
        ),
        "table_sha256": sorted({str(record["table_sha256"]) for record in records}),
        "observer_layout_sha256": sorted(
            {str(record["observer_layout_sha256"]) for record in records}
        ),
    }


def campaign_ready(
    *,
    original_staged: int,
    original_completed: int,
    rerun_staged: int,
    failed_events: int,
    rerun_remote_status: str,
    cuda: dict[str, Any],
    expected_events: int,
) -> bool:
    records = cuda.get("records")
    return bool(
        isinstance(records, list)
        and original_staged == original_completed
        and rerun_staged == failed_events
        and (failed_events == 0 or rerun_remote_status == "complete")
        and int(cuda.get("target_events", -1)) == expected_events
        and int(cuda.get("completed_events", -1)) == expected_events
        and all(
            isinstance(record, dict) and record.get("status") == "complete"
            for record in records
        )
    )


def mark_partition_complete(path: Path, readiness_path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("status") != "running":
        raise ValueError(
            f"partition manifest is not in the running state: {manifest.get('status')}"
        )
    completed_utc = utc_now()
    manifest.update(
        {
            "status": "complete",
            "updated_utc": completed_utc,
            "completed_utc": completed_utc,
            "finalization_readiness": str(readiness_path.resolve()),
        }
    )
    atomic_json(path, manifest)
    return manifest


def run_rerun_staging_once(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(args.monitor_script),
        "--remote-host",
        args.remote_host,
        "--ssh-control-path",
        str(args.ssh_control_path),
        "--remote-campaign",
        f"rerun={args.remote_rerun_root}",
        "--staging-root",
        str(args.staging_root),
        "--status-json",
        str(args.rerun_monitor_status),
        "--poll-seconds",
        "60",
        "--once",
    ]
    completed = subprocess.run(command, check=False)
    status = read_json(args.rerun_monitor_status)
    # The exact-seed launcher records ``reruns_launched`` immediately after
    # creating the remote tmux session.  The runner may need a fraction of a
    # second to create run_manifest.json.  The one-shot staging monitor
    # deliberately represents that race (and transient SSH failures) as
    # poll_failed, so retry it on the next bounded polling interval instead of
    # turning a healthy campaign into a permanent watcher failure.
    if completed.returncode != 0 and status.get("status") != "poll_failed":
        raise RuntimeError("one-shot remote rerun staging failed without status")
    return status


def finalizer_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(args.finalizer_script),
        "--final-root",
        str(args.final_root),
        "--proposal-root",
        str(args.staging_root),
    ]
    for root in args.local_campaign:
        command.extend(("--cuda-root", str(root)))
    command.extend(
        (
            "--expected-events",
            str(args.expected_events),
            "--proposal-seed-start",
            str(args.proposal_seed_start),
            "--cuda-seed-start",
            str(args.cuda_seed_start),
            "--primary-pdg",
            "2212",
            "--energy-gev",
            "1e8",
            "--zenith-deg",
            "47",
            "--azimuth-deg",
            "180",
            "--geomagnetic-model",
            "IGRF13",
            "--geomagnetic-year",
            "2025",
            "--em-cut-gev",
            "0.0005",
            "--em-thinning",
            "1e-4",
            "--had-cut-gev",
            "0.3",
            "--mu-cut-gev",
            "0.3",
            "--tau-cut-gev",
            "0.3",
            "--ring",
            "0",
            "--antenna-sha256",
            args.antenna_sha256,
            "--pulse-analysis-root",
            str(args.pulse_analysis_root),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
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
    args.ssh_control_path = args.ssh_control_path.resolve()
    args.rerun_watcher_status = args.rerun_watcher_status.resolve()
    args.staging_root = args.staging_root.resolve()
    args.original_monitor_status = args.original_monitor_status.resolve()
    args.precompleted_monitor_status = [
        path.resolve() for path in args.precompleted_monitor_status
    ]
    args.local_campaign = [path.resolve() for path in args.local_campaign]
    args.final_root = args.final_root.resolve()
    args.partition_manifest = args.partition_manifest.resolve()
    args.status_json = args.status_json.resolve()
    args.rerun_monitor_status = args.rerun_monitor_status.resolve()
    args.monitor_script = args.monitor_script.resolve()
    args.finalizer_script = args.finalizer_script.resolve()
    args.pulse_analysis_root = args.pulse_analysis_root.resolve()
    if not args.ssh_control_path.exists():
        raise ValueError(f"SSH control socket is missing: {args.ssh_control_path}")
    artifacts = validate_local_artifacts(args)
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "watching",
        "updated_utc": utc_now(),
        "artifacts": artifacts,
    }
    atomic_json(args.status_json, status)
    audited_cuda_batches: dict[str, dict[str, Any]] = {}

    while True:
        try:
            launcher = read_json(args.rerun_watcher_status)
            original = read_json(args.original_monitor_status)
            precompleted = precompleted_repair_state(
                args.precompleted_monitor_status
            )
            cuda = local_cuda_state(args.local_campaign)
            cuda_batch_audit = audit_new_cuda_batches(
                args.local_campaign,
                audited_cuda_batches,
                antenna_sha256=args.antenna_sha256,
            )
            launcher_status = str(launcher.get("status"))
            status.update(
                {
                    "status": "waiting_for_simulations",
                    "updated_utc": utc_now(),
                    "rerun_launcher_status": launcher_status,
                    "original_remote_staged_events": int(
                        original.get("remote_staged_events", 0)
                    ),
                    "precompleted_repairs": precompleted,
                    "cuda": cuda,
                    "cuda_batch_audit": cuda_batch_audit,
                }
            )
            if any(record["status"] == "failed" for record in cuda["records"]):
                raise RuntimeError("a local CUDA campaign failed")
            if launcher_status == "error":
                raise RuntimeError("the exact failed-seed launcher failed")
            if launcher_status not in {
                "reruns_launched",
                "complete_no_reruns_required",
                "complete_precompleted_repairs",
            }:
                atomic_json(args.status_json, status)
                time.sleep(args.poll_seconds)
                continue

            terminal_audit = launcher.get("terminal_audit")
            if not isinstance(terminal_audit, dict):
                raise ValueError("rerun launcher lacks terminal audit")
            failed_events = int(terminal_audit.get("failed_events", -1))
            completed_events = int(terminal_audit.get("completed_events", -1))
            if failed_events < 0 or completed_events + failed_events != args.expected_events:
                raise ValueError("terminal original CPU partition is invalid")

            precompleted_audit = terminal_audit.get("precompleted_repairs", {})
            if not isinstance(precompleted_audit, dict):
                raise ValueError("terminal audit has invalid precompleted repairs")
            expected_precompleted = {
                int(seed) for seed in precompleted_audit.get("completed_seeds", [])
            }
            staged_precompleted = {
                int(seed) for seed in precompleted.get("staged_seeds", [])
            }
            unexpected_precompleted = sorted(
                staged_precompleted.difference(expected_precompleted)
            )
            if unexpected_precompleted:
                raise ValueError(
                    "staged precompleted repairs are not accepted by the terminal audit: "
                    f"{unexpected_precompleted[:20]}"
                )
            if staged_precompleted != expected_precompleted:
                status["rerun"] = {
                    "expected_precompleted_seeds": sorted(expected_precompleted),
                    "staged_precompleted_seeds": sorted(staged_precompleted),
                    "status": "waiting_for_precompleted_staging",
                }
                atomic_json(args.status_json, status)
                time.sleep(args.poll_seconds)
                continue

            remaining_failed_events = int(
                terminal_audit.get(
                    "remaining_failed_events",
                    failed_events - len(expected_precompleted),
                )
            )
            if (
                remaining_failed_events < 0
                or remaining_failed_events + len(expected_precompleted)
                != failed_events
            ):
                raise ValueError("terminal repair partition is invalid")

            canonical_rerun_staged = 0
            canonical_remote_status = "not_required"
            rerun_remote_failed = 0
            if remaining_failed_events:
                rerun = run_rerun_staging_once(args)
                if rerun.get("status") == "poll_failed":
                    status["rerun"] = {
                        "expected_events": failed_events,
                        "precompleted_events": len(expected_precompleted),
                        "remaining_events": remaining_failed_events,
                        "status": "waiting_for_remote_manifest_or_ssh",
                        "poll_error": rerun.get("error"),
                        "poll_error_type": rerun.get("error_type"),
                    }
                    atomic_json(args.status_json, status)
                    time.sleep(args.poll_seconds)
                    continue
                if rerun.get("status") != "monitoring":
                    raise RuntimeError("remote rerun staging monitor is not healthy")
                campaigns = rerun.get("remote")
                if not isinstance(campaigns, list) or len(campaigns) != 1:
                    raise ValueError("unexpected rerun staging status")
                campaign = campaigns[0]
                canonical_rerun_staged = int(campaign.get("staged_events", 0))
                canonical_remote_status = str(campaign.get("remote_status"))
                rerun_remote_failed = int(campaign.get("remote_failed", 0))
                if canonical_remote_status == "failed" or rerun_remote_failed:
                    raise RuntimeError("the writer-fix CPU rerun campaign failed")

            rerun_staged = len(expected_precompleted) + canonical_rerun_staged
            rerun_remote_status = (
                "complete"
                if remaining_failed_events == 0
                or canonical_remote_status == "complete"
                else canonical_remote_status
            )
            status["rerun"] = {
                "expected_events": failed_events,
                "precompleted_events": len(expected_precompleted),
                "precompleted_seeds": sorted(expected_precompleted),
                "remaining_events": remaining_failed_events,
                "canonical_staged_events": canonical_rerun_staged,
                "staged_events": rerun_staged,
                "remote_status": rerun_remote_status,
                "remote_failed": rerun_remote_failed,
            }

            original_staged = int(original.get("remote_staged_events", 0))
            ready = campaign_ready(
                original_staged=original_staged,
                original_completed=completed_events,
                rerun_staged=rerun_staged,
                failed_events=failed_events,
                rerun_remote_status=rerun_remote_status,
                cuda=cuda,
                expected_events=args.expected_events,
            )
            if not ready:
                atomic_json(args.status_json, status)
                time.sleep(args.poll_seconds)
                continue

            validate_local_artifacts(args)
            status.update({"status": "finalizing", "updated_utc": utc_now()})
            atomic_json(args.status_json, status)
            command = finalizer_command(args)
            log_path = args.final_root / "automatic_finalizer.log"
            with log_path.open("w", encoding="utf-8") as log:
                log.write("command: " + shlex.join(command) + "\n")
                log.flush()
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"distributed finalizer failed with exit code {result.returncode}"
                )
            readiness = read_json(args.final_root / "finalization_readiness.json")
            if readiness.get("status") != "complete":
                raise RuntimeError("finalizer exited without a complete readiness record")
            partition = mark_partition_complete(
                args.partition_manifest,
                args.final_root / "finalization_readiness.json",
            )
            status.update(
                {
                    "status": "complete",
                    "updated_utc": utc_now(),
                    "finalizer_log": str(log_path),
                    "finalization_readiness": str(
                        args.final_root / "finalization_readiness.json"
                    ),
                    "partition_manifest": str(args.partition_manifest),
                    "partition_manifest_status": partition["status"],
                }
            )
            atomic_json(args.status_json, status)
            return 0
        except Exception as error:
            status.update(
                {
                    "status": "error",
                    "updated_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            atomic_json(args.status_json, status)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
