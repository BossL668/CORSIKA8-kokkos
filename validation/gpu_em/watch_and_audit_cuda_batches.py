#!/usr/bin/env python3
"""Strictly audit each completed batch of a running CUDA ensemble.

The production runner only appends a batch to ``campaign_manifest.json`` after
the shower process has closed its outputs.  This watcher consumes those
immutable records, applies the same ``audit_source`` checks used by the final
500-vs-500 finalizer, and keeps a small content/provenance report in the
campaign root.  A failed audit terminates the watcher but deliberately does not
kill or modify the simulation process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    from .finalize_distributed_campaign import audit_source
except ImportError:  # Direct execution from validation/gpu_em.
    from finalize_distributed_campaign import audit_source  # type: ignore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping: {label}")
    return value


def expected_namespace(immutable: dict[str, Any]) -> SimpleNamespace:
    physics = require_mapping(immutable.get("physics"), "immutable physics")
    artifacts = require_mapping(immutable.get("artifacts"), "immutable artifacts")
    antenna = require_mapping(artifacts.get("antenna_file"), "antenna artifact")
    antenna_sha256 = str(antenna.get("sha256", ""))
    if len(antenna_sha256) != 64:
        raise ValueError("immutable antenna SHA-256 is invalid")
    return SimpleNamespace(
        antenna_sha256=antenna_sha256,
        primary_pdg=int(immutable["primary_pdg"]),
        energy_gev=float(immutable["energy_GeV"]),
        zenith_deg=float(physics["zenith_deg"]),
        azimuth_deg=float(physics["azimuth_deg"]),
        em_cut_gev=float(physics["em_cut_GeV"]),
        em_thinning=float(physics["em_thinning"]),
        maximum_weight=float(physics.get("maximum_weight", 0.0)),
        had_cut_gev=float(physics["had_cut_GeV"]),
        mu_cut_gev=float(physics["mu_cut_GeV"]),
        tau_cut_gev=float(physics["tau_cut_GeV"]),
        ring=int(physics["ring"]),
        geomagnetic_model=str(physics["geomagnetic_model"]),
        geomagnetic_year=float(physics["geomagnetic_year"]),
    )


PARQUET_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile/profile.parquet": (
        "shower", "X", "charged", "hadron", "photon", "electron",
        "positron", "muplus", "muminus",
    ),
    "production_profile/profile.parquet": (
        "shower", "X", "pion", "kaon", "heavy", "hadron", "photon",
        "electron-positron", "muon", "all",
    ),
    "energyloss/dEdX.parquet": ("shower", "X", "total"),
    "particles/particles.parquet": (
        "shower", "pdg", "kinetic_energy", "x", "y", "nx", "ny", "nz",
        "time", "weight",
    ),
    "CoREAS/observers.parquet": ("shower", "Time", "Ex", "Ey", "Ez"),
    "ZHS/observers.parquet": ("shower", "Time", "Ex", "Ey", "Ez"),
}

RADIO_CONFIG_FILES: dict[str, str] = {
    "coreas_config_sha256": "CoREAS/config.yaml",
    "zhs_config_sha256": "ZHS/config.yaml",
}


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required configuration file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_radio_config_hashes(
    root: Path,
    record: dict[str, Any],
) -> None:
    """Pin the complete radio configuration for a closed batch.

    ``audit_source`` verifies the expected physical settings and observer layout.
    These content hashes add a stricter invariant: a completed batch may not be
    edited after it was cached, and every batch in one campaign must use byte-for-
    byte identical CoREAS and ZHS configuration files.
    """

    for field, relative in RADIO_CONFIG_FILES.items():
        observed = file_sha256(root / relative)
        cached = record.get(field)
        if cached is None:
            record[field] = observed
        elif cached != observed:
            raise ValueError(
                f"closed batch radio configuration changed: {root / relative}"
            )


def deep_scan_cuda_output(root: Path, expected_events: int) -> dict[str, Any]:
    """Stream every physics-bearing Parquet value without loading a batch at once."""

    expected_showers = set(range(expected_events))
    report: dict[str, Any] = {}
    radio_rows_per_shower: dict[str, dict[int, int]] = {}
    for relative, required_columns in PARQUET_COLUMNS.items():
        path = root / relative
        parquet = pq.ParquetFile(path)
        missing = set(required_columns).difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        showers: set[int] = set()
        rows_per_shower: dict[int, int] = {}
        nonfinite = 0
        minimum_physical = math.inf
        maximum_abs_field = 0.0
        weight_minimum = math.inf
        weight_maximum = -math.inf
        direction_error = 0.0
        rows = 0
        for batch in parquet.iter_batches(
            batch_size=250_000,
            columns=list(required_columns),
        ):
            values = batch.to_pydict()
            batch_showers = np.asarray(values["shower"], dtype=np.int64)
            rows += int(batch_showers.size)
            for shower in np.unique(batch_showers):
                shower_int = int(shower)
                showers.add(shower_int)
                rows_per_shower[shower_int] = (
                    rows_per_shower.get(shower_int, 0)
                    + int(np.count_nonzero(batch_showers == shower))
                )
            arrays: dict[str, np.ndarray] = {}
            for name in required_columns:
                if name == "shower":
                    continue
                array = np.asarray(values[name], dtype=np.float64)
                arrays[name] = array
                nonfinite += int(np.count_nonzero(~np.isfinite(array)))
            if relative.startswith("profile/"):
                physical = np.concatenate(
                    [arrays[name] for name in required_columns if name not in {"shower", "X"}]
                )
                minimum_physical = min(minimum_physical, float(np.min(physical)))
            elif relative.startswith("production_profile/"):
                physical = np.concatenate(
                    [arrays[name] for name in required_columns if name not in {"shower", "X"}]
                )
                minimum_physical = min(minimum_physical, float(np.min(physical)))
            elif relative.startswith("energyloss/"):
                minimum_physical = min(minimum_physical, float(np.min(arrays["total"])))
            elif relative.startswith("particles/"):
                minimum_physical = min(
                    minimum_physical, float(np.min(arrays["kinetic_energy"]))
                )
                weight_minimum = min(weight_minimum, float(np.min(arrays["weight"])))
                weight_maximum = max(weight_maximum, float(np.max(arrays["weight"])))
                norm = np.sqrt(
                    arrays["nx"] * arrays["nx"]
                    + arrays["ny"] * arrays["ny"]
                    + arrays["nz"] * arrays["nz"]
                )
                direction_error = max(
                    direction_error, float(np.max(np.abs(norm - 1.0)))
                )
            elif relative.startswith(("CoREAS/", "ZHS/")):
                maximum_abs_field = max(
                    maximum_abs_field,
                    *(float(np.max(np.abs(arrays[name]))) for name in ("Ex", "Ey", "Ez")),
                )
        if rows <= 0 or showers != expected_showers:
            raise ValueError(
                f"{path} has incomplete shower coverage: rows={rows}, "
                f"showers={sorted(showers)}, expected={sorted(expected_showers)}"
            )
        if nonfinite != 0:
            raise ValueError(f"{path} contains {nonfinite} non-finite values")
        entry: dict[str, Any] = {
            "rows": rows,
            "rows_per_shower_min": min(rows_per_shower.values()),
            "rows_per_shower_max": max(rows_per_shower.values()),
            "nonfinite_values": nonfinite,
        }
        if minimum_physical != math.inf:
            if minimum_physical < 0.0:
                raise ValueError(f"{path} contains a negative physical value")
            entry["minimum_physical_value"] = minimum_physical
        if relative.startswith("particles/"):
            if not weight_minimum > 0.0:
                raise ValueError(f"{path} contains a non-positive weight")
            if direction_error > 1.0e-6:
                raise ValueError(
                    f"{path} direction norm error exceeds 1e-6: {direction_error}"
                )
            entry.update(
                {
                    "weight_minimum": weight_minimum,
                    "weight_maximum": weight_maximum,
                    "direction_norm_max_abs_error": direction_error,
                }
            )
        if relative.startswith(("CoREAS/", "ZHS/")):
            entry["maximum_abs_field_V_per_m"] = maximum_abs_field
            radio_rows_per_shower[relative] = rows_per_shower
        report[relative] = entry

    profile_rows = {
        report[name]["rows"]
        for name in (
            "profile/profile.parquet",
            "production_profile/profile.parquet",
            "energyloss/dEdX.parquet",
        )
    }
    if len(profile_rows) != 1:
        raise ValueError("profile and energy-loss row counts differ")
    if radio_rows_per_shower["CoREAS/observers.parquet"] != radio_rows_per_shower[
        "ZHS/observers.parquet"
    ]:
        raise ValueError("CoREAS and ZHS radio row counts differ by shower")
    return report


def audit_manifest(
    campaign_root: Path,
    cached_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest = read_mapping(manifest_path)
    immutable = require_mapping(
        manifest.get("immutable_configuration"), "immutable configuration"
    )
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ValueError("campaign manifest batches are missing")
    expected = expected_namespace(immutable)
    target_events = int(immutable["target_events"])
    expected_seed = int(immutable["cuda_seed_start"])
    completed_events = 0
    observed_roots: set[str] = set()

    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise ValueError(f"batch {index} is not a mapping")
        if batch.get("status") != "complete":
            raise ValueError(f"manifest contains non-complete batch {index}")
        events = int(batch.get("events", -1))
        seed = int(batch.get("seed", -1))
        if events <= 0 or seed != expected_seed:
            raise ValueError(
                f"batch {index} seed range is not contiguous: "
                f"expected {expected_seed}, observed {seed}, events {events}"
            )
        root_text = batch.get("cuda_output")
        if not isinstance(root_text, str):
            raise ValueError(f"batch {index} has no CUDA output root")
        root = Path(root_text).resolve()
        key = str(root)
        if key in observed_roots:
            raise ValueError(f"duplicate CUDA output root: {root}")
        observed_roots.add(key)
        if key not in cached_records:
            record = audit_source(root, "cuda", expected=expected)
            if record.events != events or record.seed != seed:
                raise ValueError(f"strict audit differs from batch {index} manifest")
            encoded_record = asdict(record)
            encoded_record["deep_scan"] = deep_scan_cuda_output(root, events)
            cached_records[key] = encoded_record
        elif not isinstance(cached_records[key].get("deep_scan"), dict):
            cached_records[key]["deep_scan"] = deep_scan_cuda_output(root, events)
        audit_radio_config_hashes(root, cached_records[key])
        completed_events += events
        expected_seed += events

    radio_config_sha256: dict[str, str] = {}
    for field in RADIO_CONFIG_FILES:
        values = {
            str(cached_records[key][field])
            for key in observed_roots
        }
        if len(values) > 1:
            raise ValueError(
                f"completed CUDA batches use different {field} values: "
                f"{sorted(values)}"
            )
        if values:
            radio_config_sha256[field] = next(iter(values))

    manifest_completed = int(manifest.get("completed_total_events", -1))
    if manifest_completed != completed_events:
        raise ValueError(
            "manifest completed-event count differs from its closed batches: "
            f"{manifest_completed} versus {completed_events}"
        )
    if completed_events > target_events:
        raise ValueError("completed CUDA events exceed the immutable target")
    complete = completed_events == target_events
    if complete and manifest.get("status") != "complete":
        raise ValueError("all CUDA events are present but campaign is not complete")
    if not complete and manifest.get("status") not in {"running", "initializing"}:
        raise ValueError(
            f"incomplete CUDA campaign has terminal status {manifest.get('status')!r}"
        )
    return {
        "schema_version": 1,
        "status": "complete" if complete else "monitoring",
        "updated_utc": utc_now(),
        "campaign_root": str(campaign_root),
        "campaign_manifest": str(manifest_path),
        "campaign_manifest_status": manifest.get("status"),
        "target_events": target_events,
        "audited_events": completed_events,
        "audited_batches": len(batches),
        "next_expected_seed": expected_seed,
        "radio_config_sha256": radio_config_sha256,
        "records": [cached_records[key] for key in sorted(cached_records)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--status-json",
        type=Path,
        help="Defaults to <campaign-root>/incremental_batch_audit_status.json.",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    status_path = (
        args.status_json.resolve()
        if args.status_json is not None
        else campaign_root / "incremental_batch_audit_status.json"
    )
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    cache: dict[str, dict[str, Any]] = {}
    if status_path.is_file():
        previous = read_mapping(status_path)
        for record in previous.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("root"), str):
                cache[str(Path(record["root"]).resolve())] = record

    while True:
        try:
            status = audit_manifest(campaign_root, cache)
        except Exception as error:
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "audit_failed",
                    "updated_utc": utc_now(),
                    "campaign_root": str(campaign_root),
                    "error": f"{type(error).__name__}: {error}",
                    "records": [cache[key] for key in sorted(cache)],
                },
            )
            raise
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        if args.once or status["status"] == "complete":
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
