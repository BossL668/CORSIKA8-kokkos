#!/usr/bin/env python3
"""Incrementally stage immutable completed CUDA batches on a remote host.

The CUDA ensemble runner appends a batch to ``campaign_manifest.json`` only
after the corresponding shower process has closed all outputs.  This helper
copies only those completed batches, verifies each copy with an rsync checksum
dry-run, and performs one exact whole-campaign synchronization after the
manifest reaches ``complete``.  It never removes local source data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--ssh-socket", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=45.0)
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def rsync_transport(socket: Path) -> str:
    return f"ssh -S {shlex.quote(str(socket))}"


def sync_tree(
    source: Path,
    remote_host: str,
    remote_path: str,
    socket: Path,
    *,
    delete: bool,
) -> None:
    command = ["rsync", "-a", "--partial"]
    if delete:
        command.append("--delete")
    command.extend(
        (
            "-e",
            rsync_transport(socket),
            f"{source}/",
            f"{remote_host}:{remote_path.rstrip('/')}/",
        )
    )
    run(command)


def checksum_differences(
    source: Path,
    remote_host: str,
    remote_path: str,
    socket: Path,
) -> str:
    result = run(
        [
            "rsync",
            "-acni",
            "--delete",
            "-e",
            rsync_transport(socket),
            f"{source}/",
            f"{remote_host}:{remote_path.rstrip('/')}/",
        ],
        capture=True,
    )
    return result.stdout.strip()


def sync_file(source: Path, remote_host: str, remote_root: str, socket: Path) -> None:
    run(
        [
            "rsync",
            "-a",
            "-e",
            rsync_transport(socket),
            str(source),
            f"{remote_host}:{remote_root.rstrip('/')}/",
        ]
    )


def completed_batches(manifest: dict[str, Any], campaign_root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for record in manifest.get("batches", []):
        if record.get("status") != "complete":
            continue
        path = Path(record["output"]).resolve()
        try:
            relative = path.relative_to(campaign_root)
        except ValueError as error:
            raise ValueError(f"batch escaped campaign root: {path}") from error
        if len(relative.parts) != 1 or not relative.name.startswith("batch_"):
            raise ValueError(f"unexpected batch path: {path}")
        result.append((relative.name, path))
    return sorted(result)


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.poll_seconds) or not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in [5, 60]")
    campaign_root = args.campaign_root.resolve()
    socket = args.ssh_socket.resolve()
    status_json = args.status_json.resolve()
    if not socket.exists():
        raise FileNotFoundError(f"SSH control socket is absent: {socket}")

    mkdir_command = "mkdir -p -- " + shlex.quote(args.remote_root)
    run(["ssh", "-S", str(socket), args.remote_host, mkdir_command])
    synced: set[str] = set()
    base: dict[str, Any] = {
        "schema_version": 1,
        "campaign_root": str(campaign_root),
        "remote_host": args.remote_host,
        "remote_root": args.remote_root,
    }

    while True:
        manifest_path = campaign_root / "campaign_manifest.json"
        if not manifest_path.is_file():
            atomic_json(status_json, {**base, "status": "waiting_for_manifest"})
            time.sleep(args.poll_seconds)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("failure") is not None or manifest.get("status") == "failed":
            atomic_json(
                status_json,
                {**base, "status": "source_failed", "failure": manifest.get("failure")},
            )
            return 1

        for batch_name, batch_path in completed_batches(manifest, campaign_root):
            if batch_name in synced:
                continue
            remote_batch = f"{args.remote_root.rstrip('/')}/{batch_name}"
            sync_tree(batch_path, args.remote_host, remote_batch, socket, delete=True)
            difference = checksum_differences(
                batch_path, args.remote_host, remote_batch, socket
            )
            if difference:
                raise RuntimeError(
                    f"checksum verification failed for {batch_name}:\n{difference}"
                )
            synced.add(batch_name)
            sync_file(manifest_path, args.remote_host, args.remote_root, socket)
            atomic_json(
                status_json,
                {
                    **base,
                    "status": "syncing",
                    "source_status": manifest.get("status"),
                    "synced_batches": sorted(synced),
                    "synced_events": sum(
                        int(record.get("events", 0))
                        for record in manifest.get("batches", [])
                        if record.get("status") == "complete"
                        and Path(record["output"]).name in synced
                    ),
                },
            )

        expected_batches = len(manifest.get("batches", []))
        if manifest.get("status") == "complete" and expected_batches == len(synced):
            sync_tree(
                campaign_root,
                args.remote_host,
                args.remote_root,
                socket,
                delete=True,
            )
            difference = checksum_differences(
                campaign_root, args.remote_host, args.remote_root, socket
            )
            if difference:
                raise RuntimeError(
                    "whole-campaign checksum verification failed:\n" + difference
                )
            atomic_json(
                status_json,
                {
                    **base,
                    "status": "complete",
                    "source_status": "complete",
                    "synced_batches": sorted(synced),
                    "synced_events": int(manifest.get("completed_total_events", 0)),
                    "checksum_dry_run_changes": 0,
                },
            )
            return 0

        atomic_json(
            status_json,
            {
                **base,
                "status": "waiting_for_completed_batch",
                "source_status": manifest.get("status"),
                "synced_batches": sorted(synced),
            },
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
