#!/usr/bin/env python3
"""Audit and finalize a distributed original-CPU/CUDA shower campaign.

The script is intentionally fail closed.  It will not start the statistical
comparison unless both backends contain the exact requested seed sets, every
output has closed timing records, CUDA integrity counters pass, the canonical
physics configuration and radio observer layout agree, and all immutable
input provenance is present.  By default it performs only the readiness
audit; ``--execute`` runs the complete comparison and plotting suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from .compare_ensembles import (
        canonical_physics_configuration,
        observer_layout_fingerprint,
        validate_completion,
    )
except ImportError:  # Direct execution from validation/gpu_em.
    from compare_ensembles import (  # type: ignore
        canonical_physics_configuration,
        observer_layout_fingerprint,
        validate_completion,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_OUTPUTS = (
    "config.yaml",
    "summary.yaml",
    "validation_provenance.json",
    "simulation_timing/summary.yaml",
    "profile/profile.parquet",
    "production_profile/profile.parquet",
    "energyloss/dEdX.parquet",
    "particles/particles.parquet",
    "CoREAS/config.yaml",
    "CoREAS/summary.yaml",
    "CoREAS/observers.parquet",
    "ZHS/config.yaml",
    "ZHS/summary.yaml",
    "ZHS/observers.parquet",
)


@dataclass(frozen=True)
class SourceRecord:
    backend: str
    root: str
    seed: int
    events: int
    executable_sha256: str
    table_sha256: str | None
    antenna_sha256: str
    observer_layout_sha256: str
    gpu_physics_source: str | None = None

    @property
    def seeds(self) -> range:
        return range(self.seed, self.seed + self.events)


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


def read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as source:
            value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


_EXCLUDED_SOURCE_DIRECTORY_NAMES = {
    "failed_attempt",
    "failed_attempts",
    "failed-attempt",
    "failed-attempts",
    "archive",
    "archives",
}
_CANONICAL_SOURCE_DIRECTORY_NAMES = {
    "proposal": {"proposal"},
    "cuda": {"cuda", "c8emrt", "proposal-native"},
}


def is_canonical_source_directory(
    source: Path, search_root: Path, backend: str
) -> bool:
    """Return whether ``source`` is a canonical completed-output directory.

    Distributed runners write live attempts below ``.attempt-*`` and may keep
    failed copies below a ``failed_attempts`` or archive directory.  Those
    trees can contain a syntactically valid provenance file, so accepting
    every recursive match can silently double-count an event.  A caller may
    still pass one completed output directly; recursively discovered outputs
    must use one of the stable runner directory names.
    """

    if backend not in _CANONICAL_SOURCE_DIRECTORY_NAMES:
        raise ValueError(f"unsupported source backend: {backend!r}")
    try:
        relative_parts = source.relative_to(search_root).parts
    except ValueError as error:
        raise ValueError(
            f"source {source} is outside requested campaign root {search_root}"
        ) from error
    inspected_parts = (*search_root.parts[-1:], *relative_parts)
    for part in inspected_parts:
        lowered = part.lower()
        if part.startswith(".attempt-") or lowered in _EXCLUDED_SOURCE_DIRECTORY_NAMES:
            return False
    if source == search_root:
        return True
    leaf = source.name
    if backend == "proposal":
        return leaf.startswith("proposal_shard_") or leaf in (
            _CANONICAL_SOURCE_DIRECTORY_NAMES[backend]
        )
    return leaf in _CANONICAL_SOURCE_DIRECTORY_NAMES[backend]


def discover_sources(search_roots: Iterable[Path], backend: str) -> list[Path]:
    discovered: set[Path] = set()
    for requested in search_roots:
        root = requested.resolve()
        if not root.is_dir():
            raise ValueError(f"campaign root is absent: {root}")
        direct = root / "validation_provenance.json"
        candidates = [direct] if direct.is_file() else root.rglob(
            "validation_provenance.json"
        )
        for provenance_path in candidates:
            source = provenance_path.parent.resolve()
            if not is_canonical_source_directory(source, root, backend):
                continue
            provenance = read_mapping(provenance_path)
            if provenance.get("backend") == backend:
                discovered.add(source)
    return sorted(discovered, key=str)


def cuda_generic_fallback_allowlist(
    records: Iterable[SourceRecord],
) -> tuple[str, ...]:
    """Return the exact scalar-fallback allowlist for one CUDA ensemble."""

    sources = {record.gpu_physics_source for record in records}
    if sources == {"proposal-native"}:
        return (
            "unsupported_particle",
            "unsupported_medium",
            "unsupported_geometry",
        )
    if "proposal-native" in sources:
        raise ValueError(
            "CUDA campaign mixes proposal-native with another or unrecorded "
            f"physics source: {sorted(str(value) for value in sources)}"
        )
    return ()


def append_presentation_arguments(
    command: list[str],
    reference_label: str | None,
    candidate_label: str | None,
) -> None:
    """Append labels only to child analyzers that explicitly support them."""

    if reference_label is not None:
        command.extend(("--reference-label", reference_label))
    if candidate_label is not None:
        command.extend(("--candidate-label", candidate_label))


def require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid {label} SHA-256: {text!r}")
    return text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cuda_build_equivalence_attestation(
    attestation_path: Path | None,
    cuda_builds: set[str],
    cuda_tables: set[str | None],
) -> dict[str, Any] | None:
    """Validate explicit replay evidence before accepting mixed CUDA builds.

    A single executable hash needs no attestation.  More than one hash is
    accepted only when a fail-closed attestation names exactly those builds,
    the one physics table used by the campaign, and a content-addressed replay
    report whose own status is ``pass``.
    """

    if len(cuda_tables) != 1 or None in cuda_tables:
        raise ValueError("CUDA batches mix physics-table builds")
    table_sha256 = require_sha256(next(iter(cuda_tables)), "CUDA table")
    if len(cuda_builds) == 1:
        return None
    if attestation_path is None:
        raise ValueError(
            "CUDA batches mix executable builds; "
            "--cuda-build-equivalence-attestation is required"
        )

    path = attestation_path.resolve()
    attestation = read_mapping(path)
    if attestation.get("schema_version") != 1:
        raise ValueError(f"unsupported CUDA build attestation schema: {path}")
    if attestation.get("status") != "verified":
        raise ValueError(f"CUDA build attestation is not verified: {path}")
    allowed = attestation.get("allowed_executable_sha256")
    if not isinstance(allowed, list):
        raise ValueError(f"CUDA build attestation lacks allowed hashes: {path}")
    attested_builds = {
        require_sha256(value, "attested CUDA executable") for value in allowed
    }
    if len(attested_builds) != len(allowed):
        raise ValueError(f"CUDA build attestation repeats executable hashes: {path}")
    if attested_builds != cuda_builds:
        raise ValueError(
            "CUDA build attestation hash set differs from campaign builds: "
            f"attested={sorted(attested_builds)}, observed={sorted(cuda_builds)}"
        )
    if require_sha256(attestation.get("table_sha256"), "attested CUDA table") != table_sha256:
        raise ValueError("CUDA build attestation physics-table hash differs")

    replay = attestation.get("replay_report")
    if not isinstance(replay, dict):
        raise ValueError(f"CUDA build attestation lacks replay report: {path}")
    report_path_value = replay.get("path")
    if not isinstance(report_path_value, str) or not report_path_value:
        raise ValueError(f"CUDA build attestation has invalid replay path: {path}")
    report_path = Path(report_path_value)
    if not report_path.is_absolute():
        report_path = path.parent / report_path
    report_path = report_path.resolve()
    expected_report_hash = require_sha256(
        replay.get("sha256"), "CUDA replay report"
    )
    if not report_path.is_file() or file_sha256(report_path) != expected_report_hash:
        raise ValueError(f"CUDA replay report is missing or has changed: {report_path}")
    report = read_mapping(report_path)
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        raise ValueError(f"CUDA replay report did not pass: {report_path}")
    reference = report.get("reference")
    candidate = report.get("candidate")
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise ValueError("CUDA replay report lacks reference/candidate metadata")
    report_builds = {
        require_sha256(reference.get("executable_sha256"), "replay reference executable"),
        require_sha256(candidate.get("executable_sha256"), "replay candidate executable"),
    }
    if report_builds != cuda_builds:
        raise ValueError("CUDA replay report executable hashes differ")
    report_tables = {
        require_sha256(reference.get("table_sha256"), "replay reference table"),
        require_sha256(candidate.get("table_sha256"), "replay candidate table"),
    }
    if report_tables != {table_sha256}:
        raise ValueError("CUDA replay report physics-table hash differs")
    identity_checks = report.get("identity_checks")
    yaml_artifacts = report.get("yaml_artifacts")
    parquet_artifacts = report.get("parquet_artifacts")
    if not isinstance(identity_checks, dict) or not identity_checks:
        raise ValueError("CUDA replay report contains no identity checks")
    if not isinstance(yaml_artifacts, dict) or not yaml_artifacts:
        raise ValueError("CUDA replay report contains no YAML comparisons")
    if not isinstance(parquet_artifacts, dict) or not parquet_artifacts:
        raise ValueError("CUDA replay report contains no Parquet comparisons")
    if not all(value is True for value in identity_checks.values()):
        raise ValueError("CUDA replay report contains a failed identity check")
    if not all(
        isinstance(item, dict) and item.get("equal") is True
        for item in yaml_artifacts.values()
    ):
        raise ValueError("CUDA replay report contains a non-identical YAML output")
    if not all(
        isinstance(item, dict) and item.get("logical_values_equal") is True
        for item in parquet_artifacts.values()
    ):
        raise ValueError("CUDA replay report contains a non-identical Parquet output")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "replay_report": str(report_path),
        "replay_report_sha256": expected_report_hash,
    }


def command_option(command: list[str], *names: str) -> str | None:
    result: str | None = None
    for index, token in enumerate(command):
        for name in names:
            if token == name:
                if index + 1 >= len(command):
                    raise ValueError(f"command option {name} has no value")
                if result is not None:
                    raise ValueError(f"command repeats option {name}")
                result = command[index + 1]
            elif token.startswith(name + "="):
                if result is not None:
                    raise ValueError(f"command repeats option {name}")
                result = token.split("=", 1)[1]
    return result


def require_close(observed: str | None, expected: float, label: str) -> None:
    if observed is None:
        raise ValueError(f"required command option is absent: {label}")
    value = float(observed)
    if not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{label} differs: expected {expected}, observed {value}")


def require_maximum_weight(command: list[str], expected: float) -> None:
    observed = command_option(command, "--max-weight")
    if expected == 0.0:
        if observed is not None:
            raise ValueError(
                "campaign requires automatic maximum weight, but "
                "--max-weight is present"
            )
        return
    require_close(observed, expected, "maximum weight")


def require_cuda_geomagnetic_configuration(
    root: Path,
    command: list[str],
    expected_model: str,
    expected_year: float,
) -> None:
    """Validate explicit CLI values or the immutable CUDA environment record."""

    model = command_option(command, "--geomagnetic-model")
    year = command_option(command, "--geomagnetic-year")
    if (model is None) != (year is None):
        raise ValueError(
            f"geomagnetic model and year must both be explicit or both be omitted: {root}"
        )
    if model is None:
        gpu_config = read_mapping(root / "gpu_em/config.yaml")
        environment = gpu_config.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"missing CUDA environment metadata in {root}")
        model = str(environment.get("geomagnetic_model"))
        recorded_year = environment.get("geomagnetic_year")
        year = None if recorded_year is None else str(recorded_year)
    if model != expected_model:
        raise ValueError(
            f"geomagnetic model differs in {root}: expected {expected_model}, observed {model}"
        )
    require_close(year, expected_year, "geomagnetic year")


def audit_source(
    root: Path,
    backend: str,
    *,
    expected: argparse.Namespace,
) -> SourceRecord:
    for relative in REQUIRED_OUTPUTS:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 4:
            raise ValueError(f"missing or unclosed {backend} output: {path}")

    provenance = read_mapping(root / "validation_provenance.json")
    if provenance.get("schema_version") != 1 or provenance.get("backend") != backend:
        raise ValueError(f"invalid {backend} provenance in {root}")
    executable = provenance.get("executable")
    if not isinstance(executable, dict):
        raise ValueError(f"missing executable provenance in {root}")
    executable_hash = require_sha256(executable.get("sha256"), "executable")
    table_hash: str | None = None
    if backend == "cuda":
        table = provenance.get("table")
        if not isinstance(table, dict):
            raise ValueError(f"missing CUDA table provenance in {root}")
        table_hash = require_sha256(table.get("sha256"), "CUDA table")
    elif provenance.get("table") is not None:
        raise ValueError(f"CPU source unexpectedly contains CUDA table provenance: {root}")

    antenna = provenance.get("antenna_file")
    if not isinstance(antenna, dict):
        # Deferred CUDA batches record the antenna fingerprint in their
        # immutable parent campaign, while the actual observer layout remains
        # authoritative here.  Recover it from the command's existing file.
        command = provenance.get("command")
        if not isinstance(command, list):
            raise ValueError(f"invalid command provenance in {root}")
        antenna_path = command_option([str(item) for item in command], "--antenna-file")
        if antenna_path is None or not Path(antenna_path).is_file():
            raise ValueError(f"missing antenna provenance for {root}")
        antenna_hash = hashlib.sha256(Path(antenna_path).read_bytes()).hexdigest()
    else:
        antenna_hash = require_sha256(antenna.get("sha256"), "antenna")
    if antenna_hash != expected.antenna_sha256:
        raise ValueError(
            f"antenna content hash differs in {root}: {antenna_hash}"
        )

    summary = read_mapping(root / "summary.yaml")
    events = int(summary.get("showers", -1))
    seed = int(summary.get("seed", -1))
    if events <= 0 or seed < 0:
        raise ValueError(f"invalid shower count or seed in {root / 'summary.yaml'}")
    configured_gpu_source: str | None = None
    permitted_generic_fallback_reasons: tuple[str, ...] = ()
    if backend == "cuda":
        candidate_source = provenance.get("gpu_physics_source")
        if isinstance(candidate_source, str) and candidate_source:
            configured_gpu_source = candidate_source
        if configured_gpu_source == "proposal-native":
            permitted_generic_fallback_reasons = (
                "unsupported_particle",
                "unsupported_medium",
                "unsupported_geometry",
            )
    showers, _gpu_integrity = validate_completion(
        root,
        backend == "cuda",
        expected_gpu_source=configured_gpu_source,
        permitted_generic_fallback_reasons=permitted_generic_fallback_reasons,
    )
    if len(showers) != events:
        raise ValueError(
            f"summary/timing event count differs in {root}: {events} versus {len(showers)}"
        )

    config = read_mapping(root / "config.yaml")
    args_text = config.get("args")
    if not isinstance(args_text, str):
        raise ValueError(f"missing exact command metadata in {root / 'config.yaml'}")
    command = shlex.split(args_text)
    require_close(command_option(command, "-p"), float(expected.primary_pdg), "primary PDG")
    require_close(command_option(command, "-E"), expected.energy_gev, "primary energy")
    require_close(command_option(command, "-N"), float(events), "event count")
    require_close(command_option(command, "--seed"), float(seed), "seed")
    require_close(command_option(command, "--zenith"), expected.zenith_deg, "zenith")
    require_close(command_option(command, "--azimuth"), expected.azimuth_deg, "azimuth")
    require_close(command_option(command, "--emcut"), expected.em_cut_gev, "EM cut")
    require_close(command_option(command, "--emthin"), expected.em_thinning, "EM thinning")
    require_close(command_option(command, "--hadcut"), expected.had_cut_gev, "hadron cut")
    require_close(command_option(command, "--mucut"), expected.mu_cut_gev, "muon cut")
    require_close(command_option(command, "--taucut"), expected.tau_cut_gev, "tau cut")
    require_close(command_option(command, "--shower-core-x"), 0.0, "core x")
    require_close(command_option(command, "--shower-core-y"), 0.0, "core y")
    require_close(command_option(command, "--ring"), float(expected.ring), "ring")
    require_maximum_weight(command, expected.maximum_weight)
    if backend == "cuda":
        require_cuda_geomagnetic_configuration(
            root,
            command,
            expected.geomagnetic_model,
            expected.geomagnetic_year,
        )
        if command_option(command, "--em-backend") != "cuda":
            raise ValueError(f"CUDA EM backend is not selected in {root}")
        if command_option(command, "--radio-backend") != "cuda":
            raise ValueError(f"CUDA radio backend is not selected in {root}")

    layout = observer_layout_fingerprint(root)
    if layout is None:
        raise ValueError(f"cannot fingerprint radio observer layout in {root}")
    return SourceRecord(
        backend=backend,
        root=str(root),
        seed=seed,
        events=events,
        executable_sha256=executable_hash,
        table_sha256=table_hash,
        antenna_sha256=antenna_hash,
        observer_layout_sha256=layout,
        gpu_physics_source=configured_gpu_source,
    )


def exact_seed_audit(
    records: list[SourceRecord], expected_start: int, expected_events: int, label: str
) -> None:
    observed: list[int] = []
    for record in records:
        observed.extend(record.seeds)
    expected = list(range(expected_start, expected_start + expected_events))
    duplicates = sorted(seed for seed in set(observed) if observed.count(seed) > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate seeds: {duplicates[:20]}")
    if sorted(observed) != expected:
        missing = sorted(set(expected).difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        raise ValueError(
            f"{label} seed set is incomplete: observed={len(observed)}, "
            f"expected={expected_events}, missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def read_explicit_seed_schedule(
    path: Path, expected_events: int, label: str
) -> tuple[int, ...]:
    """Read a fail-closed, non-contiguous seed schedule.

    This is intended for distributed campaigns whose scheduler deliberately
    stopped after reaching a valid-event quota.  Blank lines and comments are
    accepted, but duplicate, negative, or incorrectly sized schedules are not.
    """

    source = path.resolve()
    if not source.is_file():
        raise ValueError(f"{label} seed schedule is missing: {source}")
    seeds: list[int] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        field = line.strip()
        if not field or field.startswith("#"):
            continue
        try:
            seed = int(field)
        except ValueError as error:
            raise ValueError(
                f"invalid {label} seed at {source}:{line_number}: {field!r}"
            ) from error
        if seed < 0:
            raise ValueError(
                f"negative {label} seed at {source}:{line_number}: {seed}"
            )
        seeds.append(seed)
    duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
    if duplicates:
        raise ValueError(
            f"{label} explicit schedule contains duplicate seeds: {duplicates[:20]}"
        )
    if len(seeds) != expected_events:
        raise ValueError(
            f"{label} explicit schedule has {len(seeds)} seeds; "
            f"expected {expected_events}"
        )
    return tuple(seeds)


def select_records_for_seed_schedule(
    records: list[SourceRecord], expected_seeds: tuple[int, ...], label: str
) -> list[SourceRecord]:
    """Select complete source records matching an explicit seed schedule.

    Whole records outside the schedule are ignored.  A multi-event record that
    only partly overlaps the requested set is rejected because silently
    splitting one output library would make provenance ambiguous.
    """

    expected = set(expected_seeds)
    selected: list[SourceRecord] = []
    for record in records:
        record_seeds = set(record.seeds)
        overlap = record_seeds.intersection(expected)
        if not overlap:
            continue
        if overlap != record_seeds:
            raise ValueError(
                f"{label} explicit schedule partially selects source record "
                f"{record.root}: selected={sorted(overlap)[:20]}, "
                f"record_seed={record.seed}, record_events={record.events}"
            )
        selected.append(record)

    observed: list[int] = []
    for record in selected:
        observed.extend(record.seeds)
    duplicates = sorted(seed for seed in set(observed) if observed.count(seed) > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate seeds: {duplicates[:20]}")
    if set(observed) != expected or len(observed) != len(expected_seeds):
        missing = sorted(expected.difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        raise ValueError(
            f"{label} explicit seed set is incomplete: observed={len(observed)}, "
            f"expected={len(expected_seeds)}, missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    return selected


def audit_configuration(
    proposal_roots: list[Path], cuda_roots: list[Path], model: str, year: float
) -> tuple[str, ...]:
    implicit = {
        "--geomagnetic-model": model,
        "--geomagnetic-year": f"{year:.17g}",
    }
    proposal_configs = {
        canonical_physics_configuration(root, implicit_physics_options=implicit)
        for root in proposal_roots
    }
    cuda_configs = {
        canonical_physics_configuration(root, implicit_physics_options=implicit)
        for root in cuda_roots
    }
    if len(proposal_configs) != 1 or len(cuda_configs) != 1:
        raise ValueError("physics configurations differ within a backend")
    proposal = next(iter(proposal_configs))
    cuda = next(iter(cuda_configs))
    if proposal != cuda:
        raise ValueError(
            "canonical CPU/CUDA physics configurations differ:\n"
            f"CPU={proposal}\nCUDA={cuda}"
        )
    return proposal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, action="append", required=True)
    parser.add_argument("--cuda-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-events", type=int, default=500)
    proposal_seed_group = parser.add_mutually_exclusive_group(required=True)
    proposal_seed_group.add_argument("--proposal-seed-start", type=int)
    proposal_seed_group.add_argument(
        "--proposal-seed-file",
        type=Path,
        help=(
            "Explicit proposal seed schedule, one seed per line. This permits "
            "fail-closed selection from quota-stopped distributed campaigns."
        ),
    )
    parser.add_argument("--cuda-seed-start", type=int, required=True)
    parser.add_argument("--primary-pdg", type=int, default=2212)
    parser.add_argument("--energy-gev", type=float, default=1.0e8)
    parser.add_argument("--zenith-deg", type=float, default=47.0)
    parser.add_argument("--azimuth-deg", type=float, default=180.0)
    parser.add_argument("--geomagnetic-model", choices=("IGRF13", "IGRF14"), default="IGRF13")
    parser.add_argument("--geomagnetic-year", type=float, default=2025.0)
    parser.add_argument("--em-cut-gev", type=float, default=5.0e-4)
    parser.add_argument("--em-thinning", type=float, default=1.0e-4)
    parser.add_argument(
        "--maximum-weight",
        type=float,
        default=0.0,
        help="Expected explicit --max-weight; zero requires the option to be omitted.",
    )
    parser.add_argument("--had-cut-gev", type=float, default=0.3)
    parser.add_argument("--mu-cut-gev", type=float, default=0.3)
    parser.add_argument("--tau-cut-gev", type=float, default=0.3)
    parser.add_argument("--ring", type=int, default=0)
    parser.add_argument("--antenna-sha256", required=True)
    parser.add_argument(
        "--pulse-analysis-root",
        type=Path,
        required=True,
        help="Path to the pulse_analysis_modular reference implementation.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument(
        "--reference-label",
        help=(
            "Optional presentation label for the scalar-PROPOSAL/reference "
            "arm. Statistical column names and acceptance semantics are not "
            "changed."
        ),
    )
    parser.add_argument(
        "--candidate-label",
        help=(
            "Optional presentation label for the CUDA/candidate arm. "
            "Statistical column names and acceptance semantics are not changed."
        ),
    )
    parser.add_argument(
        "--cuda-build-equivalence-attestation",
        type=Path,
        help=(
            "Required when the accepted CUDA sources contain more than one "
            "executable hash; must reference a passing same-seed replay report."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> dict[str, Any]:
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    record = {
        "command": command,
        "log": str(log_path),
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"finalization command failed with exit code {result.returncode}: "
            f"{shlex.join(command)}; see {log_path}"
        )
    return record


def main() -> int:
    args = parse_args()
    if args.expected_events <= 1 or args.bootstrap_repetitions <= 0:
        raise ValueError("event and bootstrap counts must be positive")
    if not math.isfinite(args.maximum_weight) or args.maximum_weight < 0.0:
        raise ValueError("maximum weight must be finite and non-negative")
    args.antenna_sha256 = require_sha256(args.antenna_sha256, "expected antenna")
    final_root = args.final_root.resolve()
    final_root.mkdir(parents=True, exist_ok=True)
    readiness_path = final_root / "finalization_readiness.json"

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "auditing",
        "updated_utc": utc_now(),
        "expected_events_per_backend": args.expected_events,
    }
    atomic_json(readiness_path, report)
    try:
        proposal_roots = discover_sources(args.proposal_root, "proposal")
        cuda_roots = discover_sources(args.cuda_root, "cuda")
        proposal_records = [
            audit_source(root, "proposal", expected=args) for root in proposal_roots
        ]
        cuda_records = [audit_source(root, "cuda", expected=args) for root in cuda_roots]
        permitted_cuda_generic_fallback_reasons = (
            cuda_generic_fallback_allowlist(cuda_records)
        )
        proposal_seed_schedule: tuple[int, ...] | None = None
        if args.proposal_seed_file is not None:
            proposal_seed_schedule = read_explicit_seed_schedule(
                args.proposal_seed_file, args.expected_events, "proposal"
            )
            proposal_records = select_records_for_seed_schedule(
                proposal_records, proposal_seed_schedule, "proposal"
            )
        else:
            exact_seed_audit(
                proposal_records,
                args.proposal_seed_start,
                args.expected_events,
                "proposal",
            )
        exact_seed_audit(cuda_records, args.cuda_seed_start, args.expected_events, "CUDA")
        proposal_roots = [Path(record.root) for record in proposal_records]
        cuda_roots = [Path(record.root) for record in cuda_records]
        canonical = audit_configuration(
            proposal_roots, cuda_roots, args.geomagnetic_model, args.geomagnetic_year
        )
        layouts = {
            record.observer_layout_sha256
            for record in (*proposal_records, *cuda_records)
        }
        if len(layouts) != 1:
            raise ValueError(f"CPU/CUDA observer layouts differ: {sorted(layouts)}")
        cuda_tables = {record.table_sha256 for record in cuda_records}
        cuda_builds = {record.executable_sha256 for record in cuda_records}
        build_equivalence = validate_cuda_build_equivalence_attestation(
            args.cuda_build_equivalence_attestation,
            cuda_builds,
            cuda_tables,
        )
    except Exception as error:
        report.update(
            {
                "status": "not_ready",
                "updated_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(readiness_path, report)
        raise

    event_count_label = str(args.expected_events)
    analysis_manifest = final_root / (
        f"final_cpu{event_count_label}_cuda{event_count_label}_analysis_manifest.json"
    )
    manifest_payload = {
        "schema_version": 1,
        "label": (
            f"pdg{args.primary_pdg}_E{args.energy_gev:.17g}GeV_"
            f"theta{args.zenith_deg:.17g}_phi{args.azimuth_deg:.17g}_"
            f"emthin{args.em_thinning:.17g}_cpu{args.expected_events}_"
            f"cuda{args.expected_events}"
        ),
        "purpose": "Immutable source list for the complete distributed CPU/CUDA comparison",
        "presentation": {
            "reference_label": args.reference_label,
            "candidate_label": args.candidate_label,
        },
        "configuration": {
            "energy_GeV": args.energy_gev,
            "events_per_backend": args.expected_events,
            "combined_proposal_events": args.expected_events,
            "combined_cuda_events": args.expected_events,
            "proposal_seed": args.proposal_seed_start,
            "proposal_seed_file": (
                {
                    "path": str(args.proposal_seed_file.resolve()),
                    "sha256": file_sha256(args.proposal_seed_file.resolve()),
                    "count": len(proposal_seed_schedule),
                }
                if args.proposal_seed_file is not None
                and proposal_seed_schedule is not None
                else None
            ),
            "cuda_seed": args.cuda_seed_start,
            "paired_seed_control": False,
            "primary_pdg": args.primary_pdg,
            "zenith_deg": args.zenith_deg,
            "azimuth_deg": args.azimuth_deg,
            "IGRF": {"model": args.geomagnetic_model, "year": args.geomagnetic_year},
            "em_cut_GeV": args.em_cut_gev,
            "em_thinning": args.em_thinning,
            "maximum_weight": args.maximum_weight,
            "maximum_weight_cli_omitted": args.maximum_weight == 0.0,
            "hadron_cut_GeV": args.had_cut_gev,
            "muon_cut_GeV": args.mu_cut_gev,
            "tau_cut_GeV": args.tau_cut_gev,
            "ring": args.ring,
            "antenna_sha256": args.antenna_sha256,
            "radio_sampling_rate_GHz": 1.0,
            "radio_window_duration_ns": 400.0,
            "radio_pretrigger_ns": 10.0,
        },
        "canonical_physics_configuration": list(canonical),
        "additional_sources": {
            "proposal": [record.root for record in proposal_records],
            "cuda": [record.root for record in cuda_records],
        },
        "provenance_strata": {
            "proposal_executable_sha256": sorted(
                {record.executable_sha256 for record in proposal_records}
            ),
            "cuda_executable_sha256": sorted(cuda_builds),
            "cuda_table_sha256": sorted(str(value) for value in cuda_tables),
            "observer_layout_sha256": next(iter(layouts)),
            "cuda_build_equivalence_attestation": build_equivalence,
            "cuda_gpu_physics_sources": sorted(
                str(value)
                for value in {record.gpu_physics_source for record in cuda_records}
            ),
            "permitted_cuda_generic_fallback_reasons": list(
                permitted_cuda_generic_fallback_reasons
            ),
        },
    }
    atomic_json(analysis_manifest, manifest_payload)
    report.update(
        {
            "status": "ready" if args.execute else "audit_complete",
            "updated_utc": utc_now(),
            "proposal_events": sum(record.events for record in proposal_records),
            "cuda_events": sum(record.events for record in cuda_records),
            "proposal_sources": [asdict(record) for record in proposal_records],
            "cuda_sources": [asdict(record) for record in cuda_records],
            "analysis_manifest": str(analysis_manifest),
        }
    )
    atomic_json(readiness_path, report)
    if not args.execute:
        print(f"readiness audit passed; use --execute to finalize: {readiness_path}")
        return 0

    python = sys.executable
    comparison = final_root / f"ensemble_comparison_{event_count_label}"
    compare_command = [python, str(SCRIPT_DIR / "compare_ensembles.py")]
    for root in proposal_roots:
        compare_command.extend(("--proposal", str(root)))
    for root in cuda_roots:
        compare_command.extend(("--cuda", str(root)))
    compare_command.extend(
        (
            "--output", str(comparison),
            "--minimum-events", str(args.expected_events),
            "--allow-cross-build-reference",
            "--allow-mixed-proposal-builds",
            "--proposal-implicit-geomagnetic-model", args.geomagnetic_model,
            "--proposal-implicit-geomagnetic-year", f"{args.geomagnetic_year:.17g}",
            "--cuda-implicit-geomagnetic-model", args.geomagnetic_model,
            "--cuda-implicit-geomagnetic-year", f"{args.geomagnetic_year:.17g}",
        )
    )
    if build_equivalence is not None:
        compare_command.append("--allow-mixed-cuda-builds")
    for reason in permitted_cuda_generic_fallback_reasons:
        compare_command.extend(
            ("--permitted-cuda-generic-fallback-reason", reason)
        )

    feature_command = [
            python,
            str(SCRIPT_DIR / "analyze_shower_feature_distributions.py"),
            "--ensemble-root", str(comparison),
            "--manifest", str(analysis_manifest),
            "--output-dir", str(final_root / "validation_plots_all_components"),
    ]
    append_presentation_arguments(
        feature_command, args.reference_label, args.candidate_label
    )

    commands = [
        compare_command,
        feature_command,
        [
            python,
            str(SCRIPT_DIR / "analyze_post_xmax_em_profiles.py"),
            "--ensemble-root", str(comparison),
            "--manifest", str(analysis_manifest),
            "--output-dir",
            str(final_root / f"post_xmax_em_profile_analysis_{event_count_label}"),
            "--resamples", str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "diagnose_longitudinal_mean_difference.py"),
            "--ensemble-root", str(comparison),
            "--manifest", str(analysis_manifest),
            "--output-dir",
            str(final_root / f"fixed_depth_profile_diagnosis_{event_count_label}"),
            "--resamples", str(args.bootstrap_repetitions),
            "--proposal-implicit-geomagnetic-model", args.geomagnetic_model,
            "--proposal-implicit-geomagnetic-year", f"{args.geomagnetic_year:.17g}",
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_geomagnetic_pulse_distributions.py"),
            "--dataset", str(comparison),
            "--manifest", str(analysis_manifest),
            "--output", str(final_root / "geomagnetic_pulse_validation"),
            "--pulse-analysis-root", str(args.pulse_analysis_root.resolve()),
            "--radius-m", "100",
            "--bootstrap-repetitions", str(args.bootstrap_repetitions),
            "--minimum-count-per-backend", str(args.expected_events),
        ],
        [
            python,
            str(SCRIPT_DIR / "analyze_geomagnetic_radial_comparison.py"),
            "--dataset", str(comparison),
            "--manifest", str(analysis_manifest),
            "--output", str(final_root / "geomagnetic_radial_validation"),
            "--pulse-analysis-root", str(args.pulse_analysis_root.resolve()),
            "--bootstrap-repetitions", str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "diagnose_ground_radio_distribution_shift.py"),
            "--observables",
            str(comparison / "per_shower_observables.csv"),
            "--radio-features",
            str(
                final_root
                / "geomagnetic_pulse_validation"
                / "per_shower_pulse_features.csv"
            ),
            "--output",
            str(final_root / "ground_radio_attribution"),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "diagnose_radio_radial_attribution.py"),
            "--observables",
            str(comparison / "per_shower_observables.csv"),
            "--radial-features",
            str(
                final_root
                / "geomagnetic_radial_validation"
                / "per_shower_radius_features.csv"
            ),
            "--output",
            str(final_root / "ground_radio_attribution"),
            "--bootstrap-repetitions",
            str(args.bootstrap_repetitions),
        ],
        [
            python,
            str(SCRIPT_DIR / "plot_single_event_runtime_histograms.py"),
            str(comparison),
            "--manifest", str(analysis_manifest),
            "--output",
            str(final_root / f"runtime_distribution_analysis_{event_count_label}"),
        ],
    ]
    command_records: list[dict[str, Any]] = []
    try:
        for index, command in enumerate(commands):
            command_records.append(
                run_logged(command, final_root / f"finalization_step_{index:02d}.log")
            )
            report["commands"] = command_records
            report["updated_utc"] = utc_now()
            atomic_json(readiness_path, report)
    except Exception as error:
        report.update(
            {
                "status": "analysis_failed",
                "updated_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "commands": command_records,
            }
        )
        atomic_json(readiness_path, report)
        raise

    report.update(
        {
            "status": "complete",
            "updated_utc": utc_now(),
            "commands": command_records,
        }
    )
    atomic_json(readiness_path, report)
    print(f"distributed campaign finalization complete: {final_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
