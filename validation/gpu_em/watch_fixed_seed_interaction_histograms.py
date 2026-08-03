#!/usr/bin/env python3
"""Watch restarted CUDA outputs for byte-exact fixed-seed histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpectedFile:
    shard: str
    relative_path: str
    old_path: str
    new_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed watcher for byte-identical lab/CMS interaction "
            "histograms in fixed-seed CUDA restarts."
        )
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("SHARD", "NEW_ROOT", "OLD_ROOT"),
        required=True,
    )
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def histogram_identity(relative: Path) -> tuple[str, int, str]:
    parts = relative.parts
    if len(parts) != 4:
        raise ValueError(f"unexpected histogram path: {relative}")
    batch, cuda, directory, filename = parts
    if cuda != "cuda" or directory != "interaction_hist":
        raise ValueError(f"unexpected histogram path: {relative}")
    if not batch.startswith("batch_"):
        raise ValueError(f"invalid batch in histogram path: {relative}")
    stem = Path(filename).stem
    prefix = "inthist_"
    if not stem.startswith(prefix):
        raise ValueError(f"invalid histogram filename: {relative}")
    frame_and_shower = stem[len(prefix) :].rsplit("_", maxsplit=1)
    if len(frame_and_shower) != 2 or frame_and_shower[0] not in ("lab", "cms"):
        raise ValueError(f"invalid histogram filename: {relative}")
    return batch, int(frame_and_shower[1]), frame_and_shower[0]


def discover_expected(
    shard: str, new_root: Path, old_root: Path
) -> list[ExpectedFile]:
    if not new_root.is_dir() or not old_root.is_dir():
        raise ValueError(f"missing new/old root for shard {shard}")
    files: list[ExpectedFile] = []
    for old_path in sorted(
        old_root.glob("batch_*/cuda/interaction_hist/inthist_*.npz")
    ):
        relative = old_path.relative_to(old_root)
        histogram_identity(relative)
        files.append(
            ExpectedFile(
                shard=shard,
                relative_path=str(relative),
                old_path=str(old_path),
                new_path=str(new_root / relative),
            )
        )
    if not files:
        raise ValueError(f"no archived histograms discovered for shard {shard}")
    identities = [
        (shard, *histogram_identity(Path(record.relative_path)))
        for record in files
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate archived histogram identity for shard {shard}")
    return files


def group_showers(files: list[ExpectedFile]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for record in files:
        batch, shower, frame = histogram_identity(Path(record.relative_path))
        key = f"{record.shard}/{batch}/shower_{shower}"
        groups.setdefault(key, set()).add(frame)
    invalid = {key: value for key, value in groups.items() if value != {"lab", "cms"}}
    if invalid:
        raise ValueError(f"archived shower lacks lab/CMS pair: {invalid}")
    return groups


def audit_available(
    files: list[ExpectedFile], previous: dict[str, Any] | None = None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    completed = dict(previous or {})
    pending: list[str] = []
    for record in files:
        key = f"{record.shard}/{record.relative_path}"
        if key in completed:
            continue
        new_path = Path(record.new_path)
        if not new_path.is_file() or new_path.stat().st_size == 0:
            pending.append(key)
            continue
        old_path = Path(record.old_path)
        old_hash = sha256(old_path)
        new_hash = sha256(new_path)
        if old_path.stat().st_size != new_path.stat().st_size or old_hash != new_hash:
            raise RuntimeError(
                f"fixed-seed histogram mismatch: {key}; "
                f"old={old_hash}, new={new_hash}"
            )
        completed[key] = {
            "sha256": new_hash,
            "size_bytes": new_path.stat().st_size,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "audited_utc": utc_now(),
        }
    return completed, pending


def completed_showers(
    groups: dict[str, set[str]], completed_files: dict[str, Any]
) -> list[str]:
    result: list[str] = []
    for shower in sorted(groups):
        shard, batch, name = shower.split("/")
        number = int(name.removeprefix("shower_"))
        prefix = f"{shard}/{batch}/cuda/interaction_hist/inthist_"
        keys = {
            f"{prefix}lab_{number}.npz",
            f"{prefix}cms_{number}.npz",
        }
        if keys.issubset(completed_files):
            result.append(shower)
    return result


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0.0:
        raise ValueError("poll seconds must be positive")
    status_path = args.status_json.resolve()
    expected: list[ExpectedFile] = []
    shard_roots: dict[str, dict[str, str]] = {}
    for shard, new, old in args.pair:
        if shard in shard_roots:
            raise ValueError(f"duplicate shard label: {shard}")
        new_root = Path(new).resolve()
        old_root = Path(old).resolve()
        shard_roots[shard] = {"new": str(new_root), "old": str(old_root)}
        expected.extend(discover_expected(shard, new_root, old_root))
    groups = group_showers(expected)
    completed: dict[str, dict[str, Any]] = {}

    while True:
        try:
            completed, pending = audit_available(expected, completed)
            showers = completed_showers(groups, completed)
            status = {
                "schema_version": 1,
                "status": "complete" if not pending else "watching",
                "updated_utc": utc_now(),
                "shards": shard_roots,
                "expected_showers": len(groups),
                "completed_showers": len(showers),
                "completed_shower_ids": showers,
                "expected_files": len(expected),
                "completed_files": len(completed),
                "pending_files": pending,
                "file_audits": completed,
                "comparison": "physical NPZ files must be byte-identical",
            }
            atomic_json(status_path, status)
            if not pending:
                return 0
        except (OSError, RuntimeError, ValueError) as error:
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "updated_utc": utc_now(),
                    "error": str(error),
                    "expected_files": len(expected),
                    "completed_files": len(completed),
                    "file_audits": completed,
                },
            )
            raise
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        raise SystemExit(2)
