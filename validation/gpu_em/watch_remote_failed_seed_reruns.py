#!/usr/bin/env python3
"""Launch exact failed-seed reruns after remote CPU campaigns reach a terminal state.

This watcher never modifies an active campaign.  It waits until every input
manifest is either ``complete`` or ``failed``, proves that completed and
failed indices form an exact partition of each scheduled campaign, proves the
combined seed schedule, verifies all remote artifacts by SHA-256, runs the
replacement launcher in dry-run mode, and only then starts one tmux session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"complete", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_schedule(configuration: dict[str, Any]) -> list[int]:
    events = int(configuration.get("events", -1))
    explicit = configuration.get("seed_schedule")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("seed_schedule must be a list")
        schedule = [int(seed) for seed in explicit]
    else:
        start = int(configuration.get("seed_start", -1))
        schedule = list(range(start, start + events))
    if events <= 0 or len(schedule) != events:
        raise ValueError("manifest event count and seed schedule differ")
    if min(schedule) < 0 or len(schedule) != len(set(schedule)):
        raise ValueError("manifest seed schedule is negative or duplicated")
    return schedule


def audit_terminal_manifests(
    manifests: list[dict[str, Any]],
    *,
    expected_seed_start: int,
    expected_events: int,
    expected_executable_sha256: str,
    expected_antenna_sha256: str,
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("no input manifests")
    all_scheduled: list[int] = []
    completed_seeds: list[int] = []
    failed_seeds: list[int] = []
    campaign_records: list[dict[str, Any]] = []
    physics_configurations: list[dict[str, Any]] = []
    for manifest in manifests:
        status = str(manifest.get("status"))
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(f"campaign is not terminal: {status}")
        configuration = manifest.get("immutable_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("manifest lacks immutable_configuration")
        executable = configuration.get("executable")
        antenna = configuration.get("antenna_file")
        if not isinstance(executable, dict) or not isinstance(antenna, dict):
            raise ValueError("manifest lacks executable or antenna provenance")
        if executable.get("sha256") != expected_executable_sha256:
            raise ValueError("original campaign executable hash differs")
        if antenna.get("sha256") != expected_antenna_sha256:
            raise ValueError("original campaign antenna hash differs")
        physics = configuration.get("physics")
        if not isinstance(physics, dict):
            raise ValueError("manifest lacks immutable physics configuration")
        physics_configurations.append(physics)

        schedule = seed_schedule(configuration)
        events = len(schedule)
        completed_raw = manifest.get("completed_indices", [])
        failures_raw = manifest.get("failures", [])
        if not isinstance(completed_raw, list) or not isinstance(failures_raw, list):
            raise ValueError("manifest completion/failure records are invalid")
        completed = [int(index) for index in completed_raw]
        failed = []
        for failure in failures_raw:
            if not isinstance(failure, dict) or "index" not in failure:
                raise ValueError("failure record lacks an index")
            failed.append(int(failure["index"]))
        if len(completed) != len(set(completed)) or len(failed) != len(set(failed)):
            raise ValueError("manifest contains duplicate completion/failure indices")
        if set(completed).intersection(failed):
            raise ValueError("a shard is marked both completed and failed")
        expected_indices = set(range(events))
        observed_indices = set(completed).union(failed)
        if observed_indices != expected_indices:
            missing = sorted(expected_indices.difference(observed_indices))
            extra = sorted(observed_indices.difference(expected_indices))
            raise ValueError(
                f"terminal manifest does not partition its schedule: "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )
        if status == "complete" and failed:
            raise ValueError("complete campaign contains failures")
        if status == "failed" and not failed:
            raise ValueError("failed campaign contains no failure records")

        all_scheduled.extend(schedule)
        completed_seeds.extend(schedule[index] for index in completed)
        failed_seeds.extend(schedule[index] for index in failed)
        campaign_records.append(
            {
                "status": status,
                "events": events,
                "completed": len(completed),
                "failed": len(failed),
                "failed_indices": sorted(failed),
                "failed_seeds": sorted(schedule[index] for index in failed),
            }
        )

    expected_schedule = list(
        range(expected_seed_start, expected_seed_start + expected_events)
    )
    if len(all_scheduled) != len(set(all_scheduled)):
        raise ValueError("input campaigns have overlapping seed schedules")
    if sorted(all_scheduled) != expected_schedule:
        raise ValueError("combined input campaigns do not cover the expected seed interval")
    if sorted(completed_seeds + failed_seeds) != expected_schedule:
        raise ValueError("completed and failed seeds do not partition the expected interval")
    if any(physics != physics_configurations[0] for physics in physics_configurations[1:]):
        raise ValueError("input campaign physics configurations differ")
    return {
        "campaigns": campaign_records,
        "scheduled_events": expected_events,
        "completed_events": len(completed_seeds),
        "failed_events": len(failed_seeds),
        "failed_seeds": sorted(failed_seeds),
        "physics": physics_configurations[0],
    }


def audit_precompleted_repairs(
    manifests: list[dict[str, Any]],
    *,
    failed_seeds: list[int],
    expected_executable_sha256: str,
    expected_antenna_sha256: str,
    expected_fluka_sha256: str,
    expected_physics: dict[str, Any],
) -> dict[str, Any]:
    """Audit writer-fix work launched on CPUs released by original jobs.

    A repair seed is reusable only when the terminal original manifests also
    classify that exact seed as failed.  Repair failures are deliberately not
    fatal here: they remain in ``remaining_failed_seeds`` and are submitted by
    the canonical terminal rerun campaign.
    """

    expected_failed = set(failed_seeds)
    scheduled: list[int] = []
    completed: list[int] = []
    failed: list[int] = []
    campaigns: list[dict[str, Any]] = []
    for manifest in manifests:
        status = str(manifest.get("status"))
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(f"precompleted repair campaign is not terminal: {status}")
        configuration = manifest.get("immutable_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("precompleted repair manifest lacks immutable_configuration")
        executable = configuration.get("executable")
        antenna = configuration.get("antenna_file")
        flupro = configuration.get("flupro")
        physics = configuration.get("physics")
        if not all(isinstance(value, dict) for value in (executable, antenna, flupro, physics)):
            raise ValueError("precompleted repair manifest lacks artifact or physics provenance")
        if executable.get("sha256") != expected_executable_sha256:
            raise ValueError("precompleted repair executable hash differs")
        if antenna.get("sha256") != expected_antenna_sha256:
            raise ValueError("precompleted repair antenna hash differs")
        if flupro.get("sha256") != expected_fluka_sha256:
            raise ValueError("precompleted repair FLUKA hash differs")
        if physics != expected_physics:
            raise ValueError("precompleted repair physics configuration differs")

        schedule = seed_schedule(configuration)
        completed_indices = [int(index) for index in manifest.get("completed_indices", [])]
        failures_raw = manifest.get("failures", [])
        if not isinstance(failures_raw, list):
            raise ValueError("precompleted repair failures are invalid")
        failed_indices = []
        for failure in failures_raw:
            if not isinstance(failure, dict) or "index" not in failure:
                raise ValueError("precompleted repair failure lacks an index")
            failed_indices.append(int(failure["index"]))
        if len(completed_indices) != len(set(completed_indices)) or len(failed_indices) != len(set(failed_indices)):
            raise ValueError("precompleted repair contains duplicate indices")
        if set(completed_indices).intersection(failed_indices):
            raise ValueError("precompleted repair index is both complete and failed")
        expected_indices = set(range(len(schedule)))
        if set(completed_indices).union(failed_indices) != expected_indices:
            raise ValueError("terminal precompleted repair does not partition its schedule")
        if status == "complete" and failed_indices:
            raise ValueError("complete precompleted repair contains failures")
        if status == "failed" and not failed_indices:
            raise ValueError("failed precompleted repair contains no failures")
        if not set(schedule).issubset(expected_failed):
            unexpected = sorted(set(schedule).difference(expected_failed))
            raise ValueError(
                f"precompleted repair contains seeds not failed by the original run: {unexpected[:20]}"
            )

        scheduled.extend(schedule)
        completed.extend(schedule[index] for index in completed_indices)
        failed.extend(schedule[index] for index in failed_indices)
        campaigns.append(
            {
                "status": status,
                "events": len(schedule),
                "completed": len(completed_indices),
                "failed": len(failed_indices),
                "seed_schedule": schedule,
                "completed_seeds": sorted(schedule[index] for index in completed_indices),
                "failed_seeds": sorted(schedule[index] for index in failed_indices),
            }
        )

    if len(scheduled) != len(set(scheduled)):
        raise ValueError("precompleted repair campaigns contain overlapping seeds")
    completed_set = set(completed)
    remaining = sorted(expected_failed.difference(completed_set))
    return {
        "campaigns": campaigns,
        "scheduled_events": len(scheduled),
        "completed_events": len(completed),
        "failed_events": len(failed),
        "completed_seeds": sorted(completed),
        "failed_seeds": sorted(failed),
        "remaining_failed_events": len(remaining),
        "remaining_failed_seeds": remaining,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--ssh-control-path", type=Path, required=True)
    parser.add_argument("--remote-manifest", action="append", required=True)
    parser.add_argument(
        "--precompleted-manifest",
        action="append",
        default=[],
        help=(
            "Terminal writer-fix campaign manifest containing failed seeds "
            "repaired early on CPUs released by the original campaign; repeatable."
        ),
    )
    parser.add_argument("--remote-runner", required=True)
    parser.add_argument("--remote-fixed-executable", required=True)
    parser.add_argument("--remote-antenna-file", required=True)
    parser.add_argument("--remote-flupro", required=True)
    parser.add_argument("--remote-seed-list", required=True)
    parser.add_argument("--remote-rerun-root", required=True)
    parser.add_argument("--remote-workdir", required=True)
    parser.add_argument("--tmux-session", default="c8_p100pev_writerfix_rerun")
    parser.add_argument("--local-seed-list", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--expected-seed-start", type=int, required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    parser.add_argument("--expected-original-executable-sha256", required=True)
    parser.add_argument("--expected-fixed-executable-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-antenna-sha256", required=True)
    parser.add_argument("--expected-fluka-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--energy-gev", type=float, default=1.0e8)
    parser.add_argument("--zenith-deg", type=float, default=47.0)
    parser.add_argument("--azimuth-deg", type=float, default=180.0)
    parser.add_argument("--em-cut-gev", type=float, default=5.0e-4)
    parser.add_argument("--em-thinning", type=float, default=1.0e-4)
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    return parser.parse_args()


def ssh_prefix(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-S",
        str(args.ssh_control_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        args.remote_host,
    ]


def fetch_manifest_paths(
    args: argparse.Namespace, paths: list[str]
) -> list[dict[str, Any]]:
    code = (
        "import json,sys; "
        "print(json.dumps([json.load(open(path)) for path in sys.argv[1:]]))"
    )
    remote = shlex.join(["python3", "-c", code, *paths])
    result = subprocess.run(
        [*ssh_prefix(args), remote],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote manifest read failed: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("remote manifest response is invalid")
    return value


def fetch_manifests(args: argparse.Namespace) -> list[dict[str, Any]]:
    return fetch_manifest_paths(args, args.remote_manifest)


def remote_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "runner": args.remote_runner,
        "fixed_executable": args.remote_fixed_executable,
        "antenna": args.remote_antenna_file,
        "fluka": f"{args.remote_flupro.rstrip('/')}/libflukahp.a",
    }
    remote = shlex.join(["sha256sum", *paths.values()])
    result = subprocess.run(
        [*ssh_prefix(args), remote],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote artifact hashing failed: {result.stderr.strip()}")
    lines = result.stdout.splitlines()
    if len(lines) != len(paths):
        raise ValueError("unexpected remote sha256sum response")
    return {
        label: line.split()[0]
        for label, line in zip(paths, lines)
    }


def rerun_command(args: argparse.Namespace) -> list[str]:
    return [
        "python3",
        args.remote_runner,
        "--executable",
        args.remote_fixed_executable,
        "--output-root",
        args.remote_rerun_root,
        "--antenna-file",
        args.remote_antenna_file,
        "--seed-list-file",
        args.remote_seed_list,
        "--jobs",
        "130",
        "--cpu-list",
        "120-249",
        "--primary-pdg",
        str(args.primary_pdg),
        "--energy-gev",
        f"{args.energy_gev:.17g}",
        "--zenith-deg",
        f"{args.zenith_deg:.17g}",
        "--azimuth-deg",
        f"{args.azimuth_deg:.17g}",
        "--shower-core-x-m",
        "0",
        "--shower-core-y-m",
        "0",
        "--ring",
        "0",
        "--em-cut-gev",
        f"{args.em_cut_gev:.17g}",
        "--em-thinning",
        f"{args.em_thinning:.17g}",
        "--had-cut-gev",
        f"{args.had_cut_gev:.17g}",
        "--mu-cut-gev",
        f"{args.mu_cut_gev:.17g}",
        "--tau-cut-gev",
        f"{args.tau_cut_gev:.17g}",
        "--maximum-weight",
        "0",
        "--flupro",
        args.remote_flupro,
    ]


def sync_seed_list(args: argparse.Namespace, seeds: list[int]) -> None:
    args.local_seed_list.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.local_seed_list.with_name(
        f".{args.local_seed_list.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        "# Exact failed seeds derived from terminal original manifests.\n"
        + "".join(f"{seed}\n" for seed in seeds),
        encoding="utf-8",
    )
    temporary.replace(args.local_seed_list)
    ssh_transport = (
        f"ssh -S {shlex.quote(str(args.ssh_control_path))} "
        "-o BatchMode=yes -o ConnectTimeout=15"
    )
    result = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            ssh_transport,
            str(args.local_seed_list),
            f"{args.remote_host}:{args.remote_seed_list}",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("failed to synchronize the exact failed-seed list")


def dry_run_and_launch(args: argparse.Namespace) -> dict[str, Any]:
    command = rerun_command(args)
    dry_run = [*command, "--dry-run"]
    result = subprocess.run(
        [*ssh_prefix(args), shlex.join(dry_run)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote rerun dry-run failed: {result.stderr.strip()}")
    dry_configuration = json.loads(result.stdout)

    launch_log = f"{args.remote_rerun_root}.launch.log"
    runner_shell = (
        f"exec {shlex.join(command)} >> {shlex.quote(launch_log)} 2>&1"
    )
    tmux = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        args.tmux_session,
        "-c",
        args.remote_workdir,
        "bash",
        "-lc",
        runner_shell,
    ]
    remote_launch = (
        f"set -e; test ! -e {shlex.quote(args.remote_rerun_root)}; "
        f"mkdir -p {shlex.quote(args.remote_workdir)}; "
        + shlex.join(tmux)
    )
    result = subprocess.run(
        [*ssh_prefix(args), remote_launch],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote rerun launch failed: {result.stderr.strip()}")
    return {
        "command": command,
        "dry_run_configuration": dry_configuration,
        "tmux_session": args.tmux_session,
        "launch_log": launch_log,
    }


def main() -> int:
    args = parse_args()
    args.poll_seconds = min(max(args.poll_seconds, 1.0), 60.0)
    for value, label in (
        (args.expected_original_executable_sha256, "original executable"),
        (args.expected_fixed_executable_sha256, "fixed executable"),
        (args.expected_runner_sha256, "runner"),
        (args.expected_antenna_sha256, "antenna"),
        (args.expected_fluka_sha256, "FLUKA library"),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid expected {label} SHA-256")
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "watching",
        "updated_utc": utc_now(),
        "remote_manifests": args.remote_manifest,
        "precompleted_manifests": args.precompleted_manifest,
    }
    atomic_json(args.status_json, status)
    while True:
        try:
            manifests = fetch_manifests(args)
            states = [str(manifest.get("status")) for manifest in manifests]
            status.update(
                {
                    "status": "waiting_for_terminal_manifests",
                    "updated_utc": utc_now(),
                    "remote_statuses": states,
                }
            )
            atomic_json(args.status_json, status)
            if not all(state in TERMINAL_STATUSES for state in states):
                time.sleep(args.poll_seconds)
                continue

            audit = audit_terminal_manifests(
                manifests,
                expected_seed_start=args.expected_seed_start,
                expected_events=args.expected_events,
                expected_executable_sha256=args.expected_original_executable_sha256,
                expected_antenna_sha256=args.expected_antenna_sha256,
            )
            expected_physics = {
                "primary_pdg": args.primary_pdg,
                "energy_GeV": args.energy_gev,
                "zenith_deg": args.zenith_deg,
                "azimuth_deg": args.azimuth_deg,
                "shower_core_x_m": 0.0,
                "shower_core_y_m": 0.0,
                "ring": 0,
                "em_cut_GeV": args.em_cut_gev,
                "em_thinning": args.em_thinning,
                "had_cut_GeV": args.had_cut_gev,
                "mu_cut_GeV": args.mu_cut_gev,
                "tau_cut_GeV": args.tau_cut_gev,
                "maximum_weight": 0.0,
                "maximum_weight_cli_omitted": True,
                "em_backend": "legacy_scalar_proposal",
                "radio_backend": "cpu_coreas_and_zhs",
            }
            if audit["physics"] != expected_physics:
                raise ValueError(
                    "original campaign physics differs from the requested rerun: "
                    f"observed={audit['physics']}, expected={expected_physics}"
                )
            if args.precompleted_manifest:
                repair_manifests = fetch_manifest_paths(
                    args, args.precompleted_manifest
                )
                repair_states = [
                    str(manifest.get("status")) for manifest in repair_manifests
                ]
                if not all(state in TERMINAL_STATUSES for state in repair_states):
                    status.update(
                        {
                            "status": "waiting_for_precompleted_repairs",
                            "updated_utc": utc_now(),
                            "precompleted_repair_statuses": repair_states,
                        }
                    )
                    atomic_json(args.status_json, status)
                    time.sleep(args.poll_seconds)
                    continue
                precompleted = audit_precompleted_repairs(
                    repair_manifests,
                    failed_seeds=audit["failed_seeds"],
                    expected_executable_sha256=args.expected_fixed_executable_sha256,
                    expected_antenna_sha256=args.expected_antenna_sha256,
                    expected_fluka_sha256=args.expected_fluka_sha256,
                    expected_physics=expected_physics,
                )
            else:
                precompleted = {
                    "campaigns": [],
                    "scheduled_events": 0,
                    "completed_events": 0,
                    "failed_events": 0,
                    "completed_seeds": [],
                    "failed_seeds": [],
                    "remaining_failed_events": len(audit["failed_seeds"]),
                    "remaining_failed_seeds": list(audit["failed_seeds"]),
                }
            audit["precompleted_repairs"] = precompleted
            audit["remaining_failed_events"] = precompleted[
                "remaining_failed_events"
            ]
            audit["remaining_failed_seeds"] = precompleted[
                "remaining_failed_seeds"
            ]
            status["terminal_audit"] = audit
            if not audit["failed_seeds"]:
                status.update({"status": "complete_no_reruns_required", "updated_utc": utc_now()})
                atomic_json(args.status_json, status)
                return 0
            if not audit["remaining_failed_seeds"]:
                status.update(
                    {
                        "status": "complete_precompleted_repairs",
                        "updated_utc": utc_now(),
                    }
                )
                atomic_json(args.status_json, status)
                return 0

            hashes = remote_hashes(args)
            if hashes["runner"] != args.expected_runner_sha256:
                raise ValueError("remote rerun runner hash changed")
            if hashes["fixed_executable"] != args.expected_fixed_executable_sha256:
                raise ValueError("remote fixed executable hash changed")
            if hashes["antenna"] != args.expected_antenna_sha256:
                raise ValueError("remote antenna hash changed")
            if hashes["fluka"] != args.expected_fluka_sha256:
                raise ValueError("remote FLUKA library hash changed")
            sync_seed_list(args, audit["remaining_failed_seeds"])
            launch = dry_run_and_launch(args)
            status.update(
                {
                    "status": "reruns_launched",
                    "updated_utc": utc_now(),
                    "remote_artifact_hashes": hashes,
                    "seed_list": {
                        "local": str(args.local_seed_list),
                        "remote": args.remote_seed_list,
                        "sha256": sha256_file(args.local_seed_list),
                    },
                    "launch": launch,
                }
            )
            atomic_json(args.status_json, status)
            return 0
        except RuntimeError as error:
            # A non-terminal campaign is represented by the normal waiting
            # branch above.  Runtime failures here are operational errors.
            status.update(
                {
                    "status": "error",
                    "updated_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            atomic_json(args.status_json, status)
            raise
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
