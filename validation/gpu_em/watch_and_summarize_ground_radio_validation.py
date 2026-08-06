#!/usr/bin/env python3
"""Run the ground/radio conclusion only after every evidence layer closes."""

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
        read_json,
        sha256_file,
        utc_now,
    )
except ImportError:
    from watch_and_finalize_complete_ensembles import (  # type: ignore
        atomic_json,
        read_json,
        sha256_file,
        utc_now,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument("--expected-summarizer-sha256", required=True)
    parser.add_argument("--finalizer-status", type=Path, required=True)
    parser.add_argument("--radio-config-watcher-status", type=Path, required=True)
    parser.add_argument("--radio-waveform-status", type=Path, required=True)
    parser.add_argument("--radio-pulse-status", type=Path, required=True)
    parser.add_argument("--finalization", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--ground-attribution", type=Path, required=True)
    parser.add_argument("--radio-configuration", type=Path, required=True)
    parser.add_argument("--radio-waveform-oracle", type=Path, required=True)
    parser.add_argument("--radio-pulse-oracle", type=Path, required=True)
    parser.add_argument("--independent-radio-pulses", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def prerequisite_state(args: argparse.Namespace) -> dict[str, Any]:
    status_paths = {
        "finalizer": args.finalizer_status,
        "radio_configuration": args.radio_config_watcher_status,
        "radio_waveform": args.radio_waveform_status,
        "radio_pulse": args.radio_pulse_status,
    }
    expected_terminal = {
        "finalizer": "complete",
        "radio_configuration": "complete",
        "radio_waveform": "complete",
        "radio_pulse": "complete",
    }
    observed: dict[str, str] = {}
    terminal_failure = False
    for label, path in status_paths.items():
        if not path.is_file():
            observed[label] = "missing"
            continue
        value = read_json(path)
        status = str(value.get("status", "unknown"))
        observed[label] = status
        if status in {
            "failed",
            "analysis_failed",
            "audit_failed",
            "not_run_campaign_failed",
            "prerequisite_failed",
        }:
            terminal_failure = True
    artifacts = {
        "finalization": args.finalization,
        "comparison": args.comparison,
        "ground_attribution": args.ground_attribution,
        "radio_configuration": args.radio_configuration,
        "radio_waveform_oracle": args.radio_waveform_oracle,
        "radio_pulse_oracle": args.radio_pulse_oracle,
        "independent_radio_pulses": args.independent_radio_pulses,
    }
    artifact_present = {
        label: path.is_file() and path.stat().st_size > 4
        for label, path in artifacts.items()
    }
    statuses_complete = all(
        observed.get(label) == expected
        for label, expected in expected_terminal.items()
    )
    ready = bool(statuses_complete and all(artifact_present.values()))
    return {
        "ready": ready,
        "terminal_failure": terminal_failure,
        "statuses": observed,
        "artifact_present": artifact_present,
        "reason": (
            "all evidence layers are complete"
            if ready
            else "a prerequisite failed"
            if terminal_failure
            else "waiting for all evidence layers"
        ),
    }


def summary_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(args.summarizer),
        "--finalization",
        str(args.finalization),
        "--comparison",
        str(args.comparison),
        "--ground-attribution",
        str(args.ground_attribution),
        "--radio-configuration",
        str(args.radio_configuration),
        "--radio-waveform-oracle",
        str(args.radio_waveform_oracle),
        "--radio-pulse-oracle",
        str(args.radio_pulse_oracle),
        "--independent-radio-pulses",
        str(args.independent_radio_pulses),
        "--expected-events",
        str(args.expected_events),
        "--familywise-alpha",
        f"{args.familywise_alpha:.17g}",
        "--output",
        str(args.output),
    ]


def main() -> int:
    args = parse_args()
    if args.expected_events <= 0:
        raise ValueError("expected events must be positive")
    if not 0.0 < args.familywise_alpha < 1.0:
        raise ValueError("familywise alpha must lie in (0, 1)")
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 60.0:
        raise ValueError("poll seconds must lie in [1, 60]")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    observed_hash = sha256_file(args.summarizer)
    if observed_hash != args.expected_summarizer_sha256:
        raise ValueError(
            "summarizer script hash differs: "
            f"expected {args.expected_summarizer_sha256}, observed {observed_hash}"
        )
    base = {
        "schema_version": 1,
        "summarizer": str(args.summarizer),
        "summarizer_sha256": observed_hash,
        "output": str(args.output),
    }
    while True:
        state = prerequisite_state(args)
        status = {
            **base,
            "status": "ready" if state["ready"] else "waiting",
            "updated_utc": utc_now(),
            "prerequisites": state,
        }
        if state["terminal_failure"]:
            status["status"] = "prerequisite_failed"
            atomic_json(args.status_json, status)
            return 1
        if state["ready"]:
            command = summary_command(args)
            log_path = args.status_json.with_suffix(".summary.log")
            status.update(
                {"status": "summarizing", "command": command, "log": str(log_path)}
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
            status["returncode"] = result.returncode
            status["status"] = "complete" if result.returncode == 0 else "failed"
            atomic_json(args.status_json, status)
            return result.returncode
        atomic_json(args.status_json, status)
        print(json.dumps(state, sort_keys=True), flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
