#!/usr/bin/env python3
"""Fail-closed audit of CPU/CUDA radio configuration equivalence.

This tool is intentionally independent of the pulse-feature comparison.  It
answers the narrower provenance question first: did every completed CPU and
CUDA source use byte-identical CoREAS/ZHS configuration files and the same
normalized observer layout?  A successful result rules out run-setting drift;
it does not by itself prove radio-algorithm equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from .compare_ensembles import observer_layout_fingerprint
    from .finalize_distributed_campaign import discover_sources
except ImportError:  # Direct execution from validation/gpu_em.
    from compare_ensembles import observer_layout_fingerprint  # type: ignore
    from finalize_distributed_campaign import discover_sources  # type: ignore


@dataclass(frozen=True)
class RadioConfigurationRecord:
    backend: str
    root: str
    events: int
    coreas_config_sha256: str
    zhs_config_sha256: str
    observer_layout_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    if not path.is_file() or path.stat().st_size <= 4:
        raise ValueError(f"missing or incomplete radio configuration: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events(root: Path) -> int:
    path = root / "summary.yaml"
    if not path.is_file():
        raise ValueError(f"missing source summary: {path}")
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"invalid source summary: {path}")
    events = int(value.get("showers", -1))
    if events <= 0:
        raise ValueError(f"invalid shower count in {path}: {events}")
    return events


def record_source(root: Path, backend: str) -> RadioConfigurationRecord:
    layout = observer_layout_fingerprint(root)
    if layout is None:
        raise ValueError(f"cannot fingerprint radio observer layout: {root}")
    return RadioConfigurationRecord(
        backend=backend,
        root=str(root.resolve()),
        events=read_events(root),
        coreas_config_sha256=file_sha256(root / "CoREAS" / "config.yaml"),
        zhs_config_sha256=file_sha256(root / "ZHS" / "config.yaml"),
        observer_layout_sha256=layout,
    )


def audit_configuration_equivalence(
    proposal_roots: list[Path],
    cuda_roots: list[Path],
    *,
    expected_events: int | None = None,
) -> dict[str, Any]:
    proposal_sources = discover_sources(proposal_roots, "proposal")
    cuda_sources = discover_sources(cuda_roots, "cuda")
    if not proposal_sources or not cuda_sources:
        raise ValueError(
            "radio configuration audit requires at least one closed CPU and CUDA source"
        )
    records = [
        *(record_source(root, "proposal") for root in proposal_sources),
        *(record_source(root, "cuda") for root in cuda_sources),
    ]
    events = {
        backend: sum(record.events for record in records if record.backend == backend)
        for backend in ("proposal", "cuda")
    }
    if expected_events is not None:
        if expected_events <= 0:
            raise ValueError("expected event count must be positive")
        for backend, observed in events.items():
            if observed != expected_events:
                raise ValueError(
                    f"{backend} radio configuration audit has {observed} events; "
                    f"expected {expected_events}"
                )

    invariant_fields = (
        "coreas_config_sha256",
        "zhs_config_sha256",
        "observer_layout_sha256",
    )
    invariants: dict[str, str] = {}
    for field in invariant_fields:
        values = {str(getattr(record, field)) for record in records}
        if len(values) != 1:
            by_backend = {
                backend: sorted(
                    {
                        str(getattr(record, field))
                        for record in records
                        if record.backend == backend
                    }
                )
                for backend in ("proposal", "cuda")
            }
            raise ValueError(
                f"CPU/CUDA radio setting differs for {field}: {by_backend}"
            )
        invariants[field] = next(iter(values))

    return {
        "schema_version": 1,
        "status": "equivalent",
        "updated_utc": utc_now(),
        "interpretation": (
            "All audited CPU/CUDA sources use byte-identical CoREAS and ZHS "
            "configuration files and the same normalized observer layout. "
            "This excludes run-setting drift but does not replace the "
            "identical-track algorithm oracle or independent-shower statistics."
        ),
        "expected_events_per_backend": expected_events,
        "events_by_backend": events,
        "configuration_sha256": invariants,
        "records": [asdict(record) for record in records],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-root", type=Path, action="append", required=True)
    parser.add_argument("--cuda-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_configuration_equivalence(
        args.proposal_root,
        args.cuda_root,
        expected_events=args.expected_events,
    )
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
