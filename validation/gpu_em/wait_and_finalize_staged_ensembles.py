#!/usr/bin/env python3
"""Wait for one CPU and one CUDA supplement, then run a fail-closed finalizer.

Arguments after ``--`` are the complete readiness-audit command.  The helper
first runs that command without ``--execute`` and requires an
``audit_complete`` readiness record.  It then repeats the same command with
``--execute`` and requires the final record to reach ``complete``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-run-manifest", type=Path, required=True)
    parser.add_argument("--gpu-campaign-manifest", type=Path, required=True)
    parser.add_argument("--expected-supplement-events", type=int, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("finalizer_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.finalizer_command and args.finalizer_command[0] == "--":
        args.finalizer_command = args.finalizer_command[1:]
    return args


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def readiness(cpu_path: Path, gpu_path: Path, expected: int) -> dict[str, Any]:
    if not cpu_path.is_file() or not gpu_path.is_file():
        return {
            "ready": False,
            "reason": "waiting for supplement manifests",
            "cpu_manifest_exists": cpu_path.is_file(),
            "gpu_manifest_exists": gpu_path.is_file(),
        }
    cpu = read_json(cpu_path)
    gpu = read_json(gpu_path)
    cpu_complete = int(cpu.get("completed", -1))
    cpu_failed = int(cpu.get("failed", -1))
    gpu_complete = int(gpu.get("completed_total_events", -1))
    cpu_ok = (
        cpu.get("status") == "complete"
        and cpu_complete == expected
        and cpu_failed == 0
    )
    gpu_ok = (
        gpu.get("status") == "complete"
        and gpu_complete == expected
        and gpu.get("failure") is None
    )
    return {
        "ready": cpu_ok and gpu_ok,
        "reason": "supplements complete" if cpu_ok and gpu_ok else "waiting for exact supplements",
        "cpu_status": cpu.get("status"),
        "cpu_completed": cpu_complete,
        "cpu_failed": cpu_failed,
        "gpu_status": gpu.get("status"),
        "gpu_completed": gpu_complete,
        "gpu_failed": gpu.get("failure") is not None,
    }


def run_logged(command: list[str], log: Path, mode: str) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open(mode, encoding="utf-8") as stream:
        stream.write("command: " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main() -> int:
    args = parse_args()
    if args.expected_supplement_events <= 0:
        raise ValueError("--expected-supplement-events must be positive")
    if not math.isfinite(args.poll_seconds) or not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [5, 60]")
    if not args.finalizer_command:
        raise ValueError("a finalizer command is required after --")
    if "--execute" in args.finalizer_command:
        raise ValueError("the supplied audit command must not contain --execute")

    base = {
        "schema_version": 1,
        "cpu_run_manifest": str(args.cpu_run_manifest),
        "gpu_campaign_manifest": str(args.gpu_campaign_manifest),
        "expected_supplement_events": args.expected_supplement_events,
        "final_root": str(args.final_root),
    }
    while True:
        try:
            state = readiness(
                args.cpu_run_manifest,
                args.gpu_campaign_manifest,
                args.expected_supplement_events,
            )
            atomic_json(
                args.status_json,
                {**base, "status": "ready" if state["ready"] else "waiting", "supplements": state},
            )
            if state["ready"]:
                break
        except Exception as error:
            atomic_json(
                args.status_json,
                {**base, "status": "poll_failed", "error": f"{type(error).__name__}: {error}"},
            )
        time.sleep(args.poll_seconds)

    atomic_json(args.status_json, {**base, "status": "auditing", "supplements": state})
    audit_code = run_logged(args.finalizer_command, args.log, "w")
    readiness_path = args.final_root / "finalization_readiness.json"
    audit_state = read_json(readiness_path) if readiness_path.is_file() else {}
    if audit_code != 0 or audit_state.get("status") != "audit_complete":
        atomic_json(
            args.status_json,
            {
                **base,
                "status": "audit_failed",
                "audit_returncode": audit_code,
                "readiness_status": audit_state.get("status"),
            },
        )
        return 1

    atomic_json(args.status_json, {**base, "status": "finalizing"})
    final_code = run_logged(args.finalizer_command + ["--execute"], args.log, "a")
    final_state = read_json(readiness_path) if readiness_path.is_file() else {}
    status = "complete" if final_code == 0 and final_state.get("status") == "complete" else "failed"
    atomic_json(
        args.status_json,
        {
            **base,
            "status": status,
            "audit_returncode": audit_code,
            "finalizer_returncode": final_code,
            "readiness_status": final_state.get("status"),
        },
    )
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
