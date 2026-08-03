#!/usr/bin/env python3
"""Run explicitly biased interim diagnostics at distributed campaign milestones.

This watcher is deliberately separate from the fail-closed 500+500 finalizer.
It selects only outputs that have already closed and therefore treats the CPU
sample as completion-conditioned.  The generated products are useful for
early regression detection, but their manifest permanently marks them as
ineligible for final physics acceptance.
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


SCRIPT_DIR = Path(__file__).resolve().parent


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


def proposal_records(distributed: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    campaigns = distributed.get("remote")
    if not isinstance(campaigns, list):
        raise ValueError("distributed status lacks remote campaigns")
    for campaign in campaigns:
        if not isinstance(campaign, dict):
            raise ValueError("invalid remote campaign record")
        for record in campaign.get("records", []):
            if not isinstance(record, dict):
                raise ValueError("invalid staged proposal record")
            root = Path(str(record.get("path"))).resolve()
            seed = int(record.get("seed", -1))
            if seed < 0 or not root.is_dir():
                raise ValueError(f"invalid staged proposal source: {root}")
            records.append(
                {
                    "backend": "proposal",
                    "root": str(root),
                    "seed": seed,
                    "events": 1,
                    "runtime_seconds": float(record.get("runtime_seconds", math.nan)),
                }
            )
    return validate_distinct_records(records, "proposal")


def cuda_records(campaign_roots: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for campaign_root in campaign_roots:
        manifest = read_json(campaign_root / "campaign_manifest.json")
        immutable = manifest.get("immutable_configuration")
        batches = manifest.get("batches", [])
        if not isinstance(immutable, dict) or not isinstance(batches, list):
            raise ValueError(f"invalid CUDA campaign manifest: {campaign_root}")
        expected_seed = int(immutable.get("cuda_seed_start", -1))
        observed_events = 0
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError(f"invalid CUDA batch record: {campaign_root}")
            root = Path(str(batch.get("cuda_output"))).resolve()
            seed = int(batch.get("seed", -1))
            events = int(batch.get("events", -1))
            if (
                batch.get("status") != "complete"
                or events <= 0
                or seed != expected_seed
                or not root.is_dir()
            ):
                raise ValueError(f"incomplete or non-contiguous CUDA batch: {root}")
            records.append(
                {
                    "backend": "cuda",
                    "root": str(root),
                    "seed": seed,
                    "events": events,
                    "runtime_seconds": float(batch.get("runtime_seconds", math.nan)),
                }
            )
            observed_events += events
            expected_seed += events
        if observed_events != int(manifest.get("completed_total_events", 0)):
            raise ValueError(f"CUDA manifest event total differs: {campaign_root}")
    return validate_distinct_records(records, "CUDA")


def validate_distinct_records(
    records: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    roots = [str(record["root"]) for record in records]
    if len(roots) != len(set(roots)):
        raise ValueError(f"duplicate {label} source path")
    occupied: set[int] = set()
    for record in records:
        seed = int(record["seed"])
        events = int(record["events"])
        seeds = set(range(seed, seed + events))
        if occupied.intersection(seeds):
            raise ValueError(f"overlapping {label} seed intervals")
        occupied.update(seeds)
    return sorted(records, key=lambda record: (int(record["seed"]), str(record["root"])))


def select_exact_events(
    records: list[dict[str, Any]], requested: int, label: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    count = 0
    for record in records:
        events = int(record["events"])
        if count + events > requested:
            break
        selected.append(record)
        count += events
        if count == requested:
            return selected
    raise ValueError(
        f"cannot select exactly {requested} closed {label} events; available whole-record count={count}"
    )


def run_logged(command: list[str], path: Path) -> dict[str, Any]:
    started = utc_now()
    with path.open("w", encoding="utf-8") as log:
        log.write("command: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    record = {
        "command": command,
        "log": str(path),
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"interim analysis command exited {result.returncode}; see {path}"
        )
    return record


def configuration(args: argparse.Namespace, events: int) -> dict[str, Any]:
    return {
        "energy_GeV": args.energy_gev,
        "events_per_backend": events,
        "combined_proposal_events": events,
        "combined_cuda_events": events,
        "primary_pdg": args.primary_pdg,
        "zenith_deg": args.zenith_deg,
        "azimuth_deg": args.azimuth_deg,
        "geomagnetic_model": args.geomagnetic_model,
        "geomagnetic_year": args.geomagnetic_year,
        "em_cut_GeV": args.em_cut_gev,
        "em_thinning": args.em_thinning,
        "maximum_weight": 0.0,
        "maximum_weight_cli_omitted": True,
        "hadron_cut_GeV": args.had_cut_gev,
        "muon_cut_GeV": args.mu_cut_gev,
        "tau_cut_GeV": args.tau_cut_gev,
        "ring": args.ring,
        "paired_seed_control": False,
    }


def milestone_commands(
    args: argparse.Namespace,
    root: Path,
    manifest: Path,
    proposal: list[dict[str, Any]],
    cuda: list[dict[str, Any]],
    events: int,
) -> list[list[str]]:
    python = sys.executable
    comparison = root / "ensemble_comparison"
    compare = [python, str(SCRIPT_DIR / "compare_ensembles.py")]
    for record in proposal:
        compare.extend(("--proposal", str(record["root"])))
    for record in cuda:
        compare.extend(("--cuda", str(record["root"])))
    compare.extend(
        (
            "--output",
            str(comparison),
            "--minimum-events",
            str(events),
            "--allow-cross-build-reference",
            "--allow-mixed-proposal-builds",
            "--proposal-implicit-geomagnetic-model",
            args.geomagnetic_model,
            "--proposal-implicit-geomagnetic-year",
            f"{args.geomagnetic_year:.17g}",
        )
    )
    return [
        compare,
        [
            python,
            str(SCRIPT_DIR / "analyze_shower_feature_distributions.py"),
            "--ensemble-root",
            str(comparison),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(root / "validation_plots_all_components"),
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_post_xmax_em_profiles.py"),
            "--ensemble-root",
            str(comparison),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(root / "post_xmax_em_profile_analysis"),
            "--resamples",
            str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "diagnose_longitudinal_mean_difference.py"),
            "--ensemble-root",
            str(comparison),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(root / "fixed_depth_profile_diagnosis"),
            "--resamples",
            str(args.bootstrap_repetitions),
            "--proposal-implicit-geomagnetic-model",
            args.geomagnetic_model,
            "--proposal-implicit-geomagnetic-year",
            f"{args.geomagnetic_year:.17g}",
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_geomagnetic_pulse_distributions.py"),
            "--dataset",
            str(comparison),
            "--manifest",
            str(manifest),
            "--output",
            str(root / "geomagnetic_pulse_validation"),
            "--pulse-analysis-root",
            str(args.pulse_analysis_root),
            "--radius-m",
            "100",
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
            "--minimum-count-per-backend",
            str(events),
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_geomagnetic_radial_comparison.py"),
            "--dataset",
            str(comparison),
            "--manifest",
            str(manifest),
            "--output",
            str(root / "geomagnetic_radial_validation"),
            "--pulse-analysis-root",
            str(args.pulse_analysis_root),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "plot_single_event_runtime_histograms.py"),
            str(comparison),
            "--manifest",
            str(manifest),
            "--output",
            str(root / "runtime_distribution_analysis"),
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_runtime_selection_bias.py"),
            "--ensemble-root",
            str(comparison),
            "--output",
            str(root / "runtime_selection_bias_diagnostic"),
        ],
    ]


def execute_milestone(
    args: argparse.Namespace,
    events: int,
    proposal: list[dict[str, Any]],
    cuda: list[dict[str, Any]],
) -> dict[str, Any]:
    root = args.final_root / f"interim_milestone_{events:03d}cpu_{events:03d}cuda"
    marker = root / "interim_milestone_status.json"
    if marker.is_file():
        previous = read_json(marker)
        if previous.get("status") == "complete":
            return previous
        raise RuntimeError(f"previous interim milestone is incomplete: {marker}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "interim_manifest.json"
    selected_proposal_seeds = [
        seed
        for record in proposal
        for seed in range(int(record["seed"]), int(record["seed"]) + int(record["events"]))
    ]
    selected_cuda_seeds = [
        seed
        for record in cuda
        for seed in range(int(record["seed"]), int(record["seed"]) + int(record["events"]))
    ]
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "label": f"runtime_conditioned_interim_{events}cpu_{events}cuda",
            "purpose": "Early regression diagnostic only; not final acceptance evidence.",
            "selection_warning": (
                "The CPU sample contains only jobs completed when this milestone was triggered. "
                "It is conditioned on runtime and is not an unbiased random subset."
            ),
            "configuration": configuration(args, events),
            "acceptance_policy": {
                "eligible_for_final_acceptance": False,
                "required_final_sample": "Exactly 500 CPU and 500 CUDA showers after terminal exact-seed audit.",
            },
            "additional_sources": {
                "proposal": [str(record["root"]) for record in proposal],
                "cuda": [str(record["root"]) for record in cuda],
            },
            "selected_seeds": {
                "proposal": selected_proposal_seeds,
                "cuda": selected_cuda_seeds,
            },
            "created_utc": utc_now(),
        },
    )
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "events_per_backend": events,
        "manifest": str(manifest),
        "updated_utc": utc_now(),
        "commands": [],
    }
    atomic_json(marker, status)
    try:
        for index, command in enumerate(
            milestone_commands(args, root, manifest, proposal, cuda, events)
        ):
            status["commands"].append(
                run_logged(command, root / f"interim_step_{index:02d}.log")
            )
            status["updated_utc"] = utc_now()
            atomic_json(marker, status)
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "updated_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(marker, status)
        raise
    status.update({"status": "complete", "updated_utc": utc_now()})
    atomic_json(marker, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed-status", type=Path, required=True)
    parser.add_argument("--automatic-status", type=Path, required=True)
    parser.add_argument("--local-campaign", type=Path, action="append", required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--pulse-analysis-root", type=Path, required=True)
    parser.add_argument("--milestone", type=int, action="append", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--energy-gev", type=float, default=1.0e8)
    parser.add_argument("--zenith-deg", type=float, default=47.0)
    parser.add_argument("--azimuth-deg", type=float, default=180.0)
    parser.add_argument("--geomagnetic-model", choices=("IGRF13", "IGRF14"), default="IGRF13")
    parser.add_argument("--geomagnetic-year", type=float, default=2025.0)
    parser.add_argument("--em-cut-gev", type=float, default=5.0e-4)
    parser.add_argument("--em-thinning", type=float, default=1.0e-4)
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--ring", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.distributed_status = args.distributed_status.resolve()
    args.automatic_status = args.automatic_status.resolve()
    args.local_campaign = [path.resolve() for path in args.local_campaign]
    args.final_root = args.final_root.resolve()
    args.status_json = args.status_json.resolve()
    args.pulse_analysis_root = args.pulse_analysis_root.resolve()
    milestones = sorted(set(args.milestone))
    if any(value <= 1 for value in milestones):
        raise ValueError("milestones must exceed one event")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 3600.0:
        raise ValueError("poll seconds must lie in [1, 3600]")
    for path in (
        args.distributed_status,
        args.automatic_status,
        *args.local_campaign,
        args.pulse_analysis_root,
    ):
        if not path.exists():
            raise ValueError(f"required path is missing: {path}")

    completed: dict[str, Any] = {}
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "watching",
        "updated_utc": utc_now(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "milestones": milestones,
        "completed": completed,
    }
    atomic_json(args.status_json, status)
    while True:
        try:
            distributed = read_json(args.distributed_status)
            automatic = read_json(args.automatic_status)
            if automatic.get("status") in {"failed", "error", "analysis_failed"}:
                raise RuntimeError("production completion watcher has failed")
            proposal = proposal_records(distributed)
            cuda = cuda_records(args.local_campaign)
            proposal_events = sum(int(record["events"]) for record in proposal)
            cuda_events = sum(int(record["events"]) for record in cuda)
            audited_events = int(
                automatic.get("cuda_batch_audit", {}).get(
                    "strictly_audited_events", 0
                )
            )
            for milestone in milestones:
                key = str(milestone)
                marker = (
                    args.final_root
                    / f"interim_milestone_{milestone:03d}cpu_{milestone:03d}cuda"
                    / "interim_milestone_status.json"
                )
                if key not in completed and marker.is_file():
                    previous = read_json(marker)
                    if previous.get("status") == "complete":
                        completed[key] = previous
                if (
                    key not in completed
                    and proposal_events >= milestone
                    and cuda_events >= milestone
                    and audited_events >= milestone
                ):
                    selected_proposal = select_exact_events(
                        proposal, milestone, "proposal"
                    )
                    selected_cuda = select_exact_events(cuda, milestone, "CUDA")
                    completed[key] = execute_milestone(
                        args, milestone, selected_proposal, selected_cuda
                    )
            status.update(
                {
                    "status": "complete" if len(completed) == len(milestones) else "watching",
                    "updated_utc": utc_now(),
                    "available_proposal_events": proposal_events,
                    "available_cuda_events": cuda_events,
                    "strictly_audited_cuda_events": audited_events,
                    "completed": completed,
                }
            )
            atomic_json(args.status_json, status)
            if status["status"] == "complete" or args.once:
                return 0
        except Exception as error:
            status.update(
                {
                    "status": "failed",
                    "updated_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            atomic_json(args.status_json, status)
            raise
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
