#!/usr/bin/env python3
"""Wait for exact CPU/CUDA ensembles, then audit radio-setting equivalence."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .watch_and_finalize_complete_ensembles import (
        atomic_json,
        completion_state,
        local_cuda_state,
        read_json,
        sha256_file,
        utc_now,
    )
except ImportError:  # Direct execution from validation/gpu_em.
    from watch_and_finalize_complete_ensembles import (  # type: ignore
        atomic_json,
        completion_state,
        local_cuda_state,
        read_json,
        sha256_file,
        utc_now,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed-status", type=Path, required=True)
    parser.add_argument("--local-campaign", type=Path, action="append", required=True)
    parser.add_argument("--proposal-root", type=Path, action="append", required=True)
    parser.add_argument("--cuda-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def audit_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(args.audit_script)]
    for root in args.proposal_root:
        command.extend(("--proposal-root", str(root)))
    for root in args.cuda_root:
        command.extend(("--cuda-root", str(root)))
    command.extend(
        (
            "--expected-events",
            str(args.expected_events),
            "--output",
            str(args.output),
        )
    )
    return command


def main() -> int:
    args = parse_args()
    if args.expected_events <= 0:
        raise ValueError("--expected-events must be positive")
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [1, 60]")
    for name in (
        "distributed_status",
        "audit_script",
        "output",
        "status_json",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.local_campaign = [path.resolve() for path in args.local_campaign]
    args.proposal_root = [path.resolve() for path in args.proposal_root]
    args.cuda_root = [path.resolve() for path in args.cuda_root]
    observed_hash = sha256_file(args.audit_script)
    if observed_hash != args.expected_audit_sha256:
        raise ValueError(
            "radio configuration audit script hash differs: "
            f"expected {args.expected_audit_sha256}, observed {observed_hash}"
        )
    base_status: dict[str, Any] = {
        "schema_version": 1,
        "audit_script": str(args.audit_script),
        "audit_script_sha256": observed_hash,
        "output": str(args.output),
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
                command = audit_command(args)
                log_path = args.status_json.with_suffix(".audit.log")
                status.update(
                    {
                        "status": "auditing",
                        "audit_command": command,
                        "audit_log": str(log_path),
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
                status["audit_returncode"] = result.returncode
                status["status"] = "complete" if result.returncode == 0 else "failed"
                atomic_json(args.status_json, status)
                return result.returncode
            atomic_json(args.status_json, status)
            print(json.dumps(state, sort_keys=True), flush=True)
        except Exception as error:
            atomic_json(
                args.status_json,
                {
                    **base_status,
                    "status": "poll_failed",
                    "updated_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
