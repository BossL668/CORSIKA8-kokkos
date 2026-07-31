#!/usr/bin/env python3
"""Validate strict per-shower CUDA electromagnetic energy ledgers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


ENERGY_TERMS = (
    "initial_total_GeV",
    "medium_rest_mass_input_GeV",
    "deposited_GeV",
    "cut_rest_mass_energy_GeV",
    "observed_total_energy_GeV",
    "escaped_total_energy_GeV",
    "source_GeV",
    "terminal_GeV",
    "residual_GeV",
    "relative_closure_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute and validate the deterministic CUDA EM energy ledger."
        )
    )
    parser.add_argument(
        "--cuda",
        type=Path,
        action="append",
        required=True,
        help="CUDA CORSIKA output directory; repeat for multiple shards.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--minimum-events", type=int, default=1)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Reject ledgers marked as partial coverage.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return exit code 2 if any acceptance condition fails.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required YAML file is missing: {path}")
    with path.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


def finite_energy_terms(ledger: dict[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for key in ENERGY_TERMS:
        value = float(ledger.get(key, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"energy ledger term {key} is not finite")
        terms[key] = value
    for key in ENERGY_TERMS[:8]:
        if key not in ("residual_GeV",) and terms[key] < 0.0:
            raise ValueError(f"energy ledger term {key} is negative")
    return terms


def validate_record(
    root: Path,
    shower: str,
    record: dict[str, Any],
    tolerance: float,
    require_complete: bool,
) -> dict[str, Any]:
    if (
        record.get("complete") is not True
        or record.get("status") != "complete"
    ):
        raise ValueError(f"{root}: {shower} is not a complete CUDA record")
    statistics = record.get("statistics", {})
    ledger = (
        statistics.get("energy_ledger")
        if isinstance(statistics, dict)
        else None
    )
    if ledger is None:
        ledger = record.get("energy_ledger")
    if not isinstance(ledger, dict):
        raise ValueError(f"{root}: {shower} has no strict energy ledger")
    terms = finite_energy_terms(ledger)
    complete = ledger.get("complete_coverage") is True

    source = (
        terms["initial_total_GeV"]
        + terms["medium_rest_mass_input_GeV"]
    )
    terminal = (
        terms["deposited_GeV"]
        + terms["cut_rest_mass_energy_GeV"]
        + terms["observed_total_energy_GeV"]
        + terms["escaped_total_energy_GeV"]
    )
    residual = source - terminal
    relative = abs(residual) / max(source, float.fromhex("0x1p-1022"))
    storage_scale = max(1.0, source, terminal)
    storage_tolerance = 64.0 * math.ulp(storage_scale)
    internally_consistent = (
        abs(terms["source_GeV"] - source) <= storage_tolerance
        and abs(terms["terminal_GeV"] - terminal) <= storage_tolerance
        and abs(terms["residual_GeV"] - residual) <= storage_tolerance
        and abs(terms["relative_closure_error"] - relative)
        <= 64.0 * math.ulp(max(1.0, relative))
    )
    accepted = (
        complete
        and internally_consistent
        and relative <= tolerance
        and ledger.get("accepted") is True
    )
    if require_complete and not complete:
        accepted = False
    return {
        "root": str(root),
        "shower": shower,
        "complete_coverage": complete,
        "internally_consistent": internally_consistent,
        "source_GeV": source,
        "terminal_GeV": terminal,
        "residual_GeV": residual,
        "relative_closure_error": relative,
        "tolerance": tolerance,
        "accepted": accepted,
    }


def validate_outputs(
    roots: list[Path],
    tolerance: float,
    minimum_events: int,
    require_complete: bool,
) -> dict[str, Any]:
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0.0
        or minimum_events <= 0
    ):
        raise ValueError("tolerance and minimum-events must be positive")
    events: list[dict[str, Any]] = []
    for requested in roots:
        root = requested.resolve()
        summary = read_yaml(root / "gpu_em" / "summary.yaml")
        if not isinstance(summary, dict):
            raise ValueError(f"invalid CUDA summary: {root}")
        for shower, value in sorted(summary.items()):
            if not str(shower).startswith("shower_") or not isinstance(
                value, dict
            ):
                raise ValueError(f"invalid CUDA shower record in {root}")
            events.append(
                validate_record(
                    root,
                    str(shower),
                    value,
                    tolerance,
                    require_complete,
                )
            )
    passed = (
        len(events) >= minimum_events
        and all(event["accepted"] for event in events)
    )
    return {
        "status": "passed" if passed else "failed",
        "tolerance": tolerance,
        "minimum_events": minimum_events,
        "events": len(events),
        "complete_events": sum(
            event["complete_coverage"] for event in events
        ),
        "maximum_relative_closure_error": max(
            (
                event["relative_closure_error"]
                for event in events
            ),
            default=math.inf,
        ),
        "records": events,
    }


def main() -> int:
    args = parse_args()
    try:
        result = validate_outputs(
            args.cuda,
            args.tolerance,
            args.minimum_events,
            args.require_complete,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        if args.require_pass and result["status"] != "passed":
            return 2
        return 0
    except Exception as error:
        print(f"energy-ledger validation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
