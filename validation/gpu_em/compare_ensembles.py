#!/usr/bin/env python3
"""
Compare independent scalar-PROPOSAL and CUDA-EM shower ensembles.

The comparison is intentionally event based.  Particle rows and longitudinal
bins are first reduced to one value or one histogram per shower; treating
individual particles as independent samples would underestimate shower-to-
shower fluctuations by many orders of magnitude.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROFILE_COLUMNS = (
    "charged",
    "hadron",
    "photon",
    "electron",
    "positron",
    "muplus",
    "muminus",
)
DERIVED_PROFILE_COLUMNS = (
    "electron_positron",
    "muon",
    "em",
)
PRODUCTION_PROFILE_COLUMNS = (
    "pion",
    "kaon",
    "heavy",
    "hadron",
    "photon",
    "electron-positron",
    "muon",
    "all",
)
DIAGNOSTIC_CURVE_PREFIXES = ("muon_production_parent_",)
VALIDATION_PROVENANCE_FILENAME = "validation_provenance.json"
VALIDATION_PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_KEY_SCALAR_METRICS = (
    "profile_xmax_charged_gcm2",
    "profile_charged_max",
    "profile_charged_integral",
    "profile_photon_integral",
    "energy_deposit_sum_GeV",
    "energy_deposit_xmax_gcm2",
    "ground_em_weighted_count",
    "ground_em_kinetic_energy_GeV",
    "energy_closure_fraction",
)
OPTIONAL_KEY_SCALAR_METRICS = (
    "energy_deposit_max_GeV",
    "ground_radius_mean_m",
    "ground_time_rms_s",
)
AVAILABLE_KEY_SCALAR_METRICS = (
    DEFAULT_KEY_SCALAR_METRICS
    + OPTIONAL_KEY_SCALAR_METRICS
)
RADIAL_EDGES_M = np.asarray(
    [
        0.0,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        2000.0,
        5000.0,
        10000.0,
        np.inf,
    ]
)
ENERGY_EDGES_GEV = np.geomspace(1.0e-4, 1.0e10, 57)
TIME_RESIDUAL_EDGES_S = np.concatenate(
    ([0.0], np.geomspace(1.0e-12, 1.0e-2, 51), [np.inf])
)

# These options may differ between statistically independent shards or between
# the scalar and CUDA invocations without changing the simulated physics.
NON_PHYSICS_OPTIONS_WITH_VALUE = frozenset(
    {
        "--nevent",
        "--filename",
        "--seed",
        "--verbosity",
        "--em-backend",
        "--radio-backend",
        "--gpu-device",
        "--gpu-min-batch",
        "--gpu-memory-fraction",
        "--gpu-table-cache",
        "--gpu-table-tolerance",
        "--gpu-deterministic",
        "--gpu-resident-cross-species",
        "--gpu-radio-field-limit",
        "--hadronic-backend",
        "--hadronic-workers",
        "--hadronic-min-batch",
        "--hadronic-target-batch-ms",
        "--hadronic-max-batch",
        "--hadronic-initial-cost-ms",
        "--hadronic-worker-executable",
    }
)
NON_PHYSICS_FLAGS = frozenset(
    {
        "--compress",
        "--gpu-detailed-stage-timing",
        "--gpu-full-step-records",
        "--gpu-radio-track-diagnostics",
        "--cpu-detailed-step-timing",
    }
)
OPTION_ALIASES = {
    "-N": "--nevent",
    "-f": "--filename",
    "-s": "--seed",
    "-v": "--verbosity",
    "-E": "--energy",
    "-p": "--pdg",
    "-Z": "--atomic-number",
    "-A": "--atomic-mass",
    "-z": "--zenith",
    "-a": "--azimuth",
    "-M": "--hadronModel",
    "-T": "--hadronModelTransitionEnergy",
}

OMITTABLE_PHYSICS_OPTION_DEFAULTS = {
    # The original 21CMA c8_air_shower hard-codes these values and therefore
    # cannot write the equivalent explicit CLI tokens used by the refactor.
    # Only exact legacy defaults are omitted; non-default values remain
    # physics-bearing and must match on both sides.
    "--radio-sampling-rate-ghz": "1",
    "--radio-window-duration-ns": "400",
    "--radio-pretrigger-ns": "10",
}

# These options were hard-coded rather than exposed through the CLI in the
# original 21CMA application.  Keep them physics-bearing, but place them at a
# stable position in the canonical command so an explicitly documented legacy
# value can be compared with the refactor's explicit CLI value.
NORMALIZED_PHYSICS_OPTIONS_WITH_VALUE = (
    "--geomagnetic-model",
    "--geomagnetic-year",
)


def canonical_command_token(token: str) -> str:
    alias = OPTION_ALIASES.get(token)
    if alias is not None:
        return alias
    try:
        number = float(token)
    except ValueError:
        return token
    if not math.isfinite(number):
        raise ValueError(
            f"run command contains a non-finite numeric token: {token}"
        )
    return f"{number:.17g}"


@dataclass
class Ensemble:
    name: str
    root: Path
    showers: tuple[int, ...]
    scalars: pd.DataFrame
    curves: dict[str, tuple[np.ndarray, np.ndarray]]
    histograms: dict[str, tuple[np.ndarray, np.ndarray]]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-shower observables from scalar PROPOSAL and CUDA EM outputs."
        )
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        action="append",
        required=True,
        help="Scalar PROPOSAL output; repeat for independent shards.",
    )
    parser.add_argument(
        "--cuda",
        type=Path,
        action="append",
        required=True,
        help="CUDA EM output; repeat for independent shards.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument(
        "--active-fraction",
        type=float,
        default=1.0e-4,
        help="Ignore curve bins below this fraction of the CPU mean peak.",
    )
    parser.add_argument(
        "--minimum-bin-pass-fraction",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--key-scalar",
        action="append",
        choices=AVAILABLE_KEY_SCALAR_METRICS,
        help=(
            "Scalar mean receiving both the 1%% and statistical gate. "
            "Repeat to replace the default core list."
        ),
    )
    parser.add_argument(
        "--fail-on-acceptance",
        action="store_true",
        help="Return exit code 2 if any acceptance gate fails.",
    )
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help=(
            "Permit diagnostic comparison of outputs that predate "
            "validation_provenance.json. Such outputs cannot be mixed with "
            "provenanced outputs and are not valid production evidence."
        ),
    )
    parser.add_argument(
        "--allow-cross-build-reference",
        action="store_true",
        help=(
            "Permit a separately provenanced original CORSIKA executable as "
            "the scalar reference. Physics configurations must still match."
        ),
    )
    parser.add_argument(
        "--allow-mixed-proposal-builds",
        action="store_true",
        help=(
            "Permit scalar shards produced by separately built original "
            "CORSIKA executables. Every shard must remain provenanced and "
            "have an identical canonical physics configuration."
        ),
    )
    parser.add_argument(
        "--allow-mixed-cuda-builds",
        action="store_true",
        help=(
            "Permit CUDA shards produced by separately validated executable "
            "builds. Every shard must remain provenanced and have an "
            "identical canonical physics configuration."
        ),
    )
    parser.add_argument(
        "--proposal-implicit-geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        help=(
            "Document the geomagnetic coefficient model hard-coded by a "
            "legacy scalar executable whose config.yaml cannot contain the "
            "equivalent option. Must be used together with "
            "--proposal-implicit-geomagnetic-year."
        ),
    )
    parser.add_argument(
        "--proposal-implicit-geomagnetic-year",
        type=float,
        help=(
            "Document the geomagnetic epoch hard-coded by a legacy scalar "
            "executable. Must be used together with "
            "--proposal-implicit-geomagnetic-model."
        ),
    )
    parser.add_argument(
        "--cuda-implicit-geomagnetic-model",
        choices=("IGRF13", "IGRF14"),
        help=(
            "Document the geomagnetic coefficient model recorded by a CUDA "
            "run whose command omitted the equivalent default option. Must "
            "be used together with --cuda-implicit-geomagnetic-year."
        ),
    )
    parser.add_argument(
        "--cuda-implicit-geomagnetic-year",
        type=float,
        help=(
            "Document the geomagnetic epoch recorded by a CUDA run whose "
            "command omitted the equivalent default option. Must be used "
            "together with --cuda-implicit-geomagnetic-model."
        ),
    )
    return parser.parse_args()


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required YAML file is missing: {path}")
    with path.open("r", encoding="utf-8") as source:
        # GPU summaries can contain several megabytes of nested process
        # statistics per batch.  The C loader is both substantially faster
        # and avoids the pure-Python reader's pathological behaviour on
        # these long mappings.  SafeLoader remains the portable fallback.
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return yaml.load(source, Loader=loader)


def read_validation_provenance(
    root: Path,
    expected_backend: str,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any] | None:
    """Read the immutable build/table identity captured when an output was made."""

    path = root / VALIDATION_PROVENANCE_FILENAME
    if not path.is_file():
        if allow_legacy:
            return None
        raise ValueError(
            f"required validation provenance is missing: {path}; "
            "legacy outputs may only be inspected with "
            "--allow-legacy-provenance"
        )
    with path.open("r", encoding="utf-8") as source:
        provenance = json.load(source)
    if not isinstance(provenance, dict):
        raise ValueError(f"invalid validation provenance object: {path}")
    if provenance.get("schema_version") != VALIDATION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported validation provenance schema in {path}: "
            f"{provenance.get('schema_version')}"
        )
    if provenance.get("backend") != expected_backend:
        raise ValueError(
            f"validation provenance backend differs in {path}: "
            f"expected {expected_backend}, observed {provenance.get('backend')}"
        )

    executable = provenance.get("executable")
    if not isinstance(executable, dict):
        raise ValueError(f"missing executable provenance in {path}")
    executable_hash = executable.get("sha256")
    if (
        not isinstance(executable_hash, str)
        or len(executable_hash) != 64
        or any(character not in "0123456789abcdef" for character in executable_hash)
    ):
        raise ValueError(f"invalid executable SHA-256 in {path}")

    table = provenance.get("table")
    if expected_backend == "cuda":
        if not isinstance(table, dict):
            raise ValueError(f"missing CUDA table provenance in {path}")
        table_hash = table.get("sha256")
        if (
            not isinstance(table_hash, str)
            or len(table_hash) != 64
            or any(character not in "0123456789abcdef" for character in table_hash)
        ):
            raise ValueError(f"invalid CUDA table SHA-256 in {path}")
    elif table is not None:
        raise ValueError(
            f"scalar PROPOSAL provenance unexpectedly contains a CUDA table in {path}"
        )
    return provenance


def provenance_fingerprint(
    provenance: dict[str, Any] | None,
) -> tuple[int, str, str, str | None] | None:
    if provenance is None:
        return None
    table = provenance.get("table")
    return (
        int(provenance["schema_version"]),
        str(provenance["backend"]),
        str(provenance["executable"]["sha256"]),
        str(table["sha256"]) if isinstance(table, dict) else None,
    )


def observer_layout_fingerprint(root: Path) -> str | None:
    """Hash the radio geometry actually written to the shower output.

    Input antenna paths are machine-specific and therefore cannot be compared
    literally when an ensemble is produced on another host.  The output
    configuration is authoritative: it records every observer location and
    the time-grid definition used by both radio algorithms.
    """

    def normalized_float(value: Any) -> float:
        # CPU and CUDA YAML writers may serialize the same configured value
        # as 400 versus 399.99999999999994.  Twelve significant digits are
        # far tighter than any observer-position or sampling tolerance here.
        return float(f"{float(value):.12g}")

    algorithms: dict[str, list[dict[str, Any]]] = {}
    for algorithm in ("CoREAS", "ZHS"):
        path = root / algorithm / "config.yaml"
        if not path.is_file():
            return None
        config = read_yaml(path)
        observers = (
            config.get("observers")
            if isinstance(config, dict)
            else None
        )
        if not isinstance(observers, dict):
            return None
        records: list[dict[str, Any]] = []
        for name, observer in sorted(observers.items()):
            if not isinstance(observer, dict):
                raise ValueError(f"invalid radio observer in {path}: {name}")
            location = observer.get("location")
            if not isinstance(location, list) or len(location) != 3:
                raise ValueError(
                    f"invalid radio observer location in {path}: {name}"
                )
            records.append(
                {
                    "name": str(name),
                    "location_m": [
                        normalized_float(value) for value in location
                    ],
                    "duration_ns": normalized_float(observer["duration"]),
                    "number_of_bins": int(observer["number of bins"]),
                    "sampling_frequency_GHz": normalized_float(
                        observer["sampling frequency"]
                    ),
                }
            )
        algorithms[algorithm] = records
    encoded = json.dumps(
        algorithms,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_physics_configuration(
    root: Path,
    *,
    implicit_physics_options: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the physics-bearing CLI tokens stored by OutputManager.

    Seeds, event counts, output paths, verbosity and backend implementation
    controls are removed.  Everything else is retained in order, so combining
    ensembles with different primaries, atmosphere geometry, cuts, thinning or
    interaction settings fails before any means are calculated.

    Equivalent commands expressed with omitted defaults are deliberately not
    guessed to be compatible.  A false rejection is safer than silently
    pooling different physics configurations.
    """

    config = read_yaml(root / "config.yaml")
    if not isinstance(config, dict) or not isinstance(config.get("args"), str):
        raise ValueError(f"invalid run command metadata in {root / 'config.yaml'}")
    command = shlex.split(config["args"])
    if not command:
        raise ValueError(f"empty run command metadata in {root / 'config.yaml'}")

    implicit = {
        str(option): canonical_command_token(str(value))
        for option, value in (implicit_physics_options or {}).items()
    }
    unsupported_implicit = set(implicit).difference(
        NORMALIZED_PHYSICS_OPTIONS_WITH_VALUE
    )
    if unsupported_implicit:
        raise ValueError(
            "unsupported implicit physics options: "
            + ", ".join(sorted(unsupported_implicit))
        )

    canonical: list[str] = []
    normalized: dict[str, str] = {}
    index = 1  # executable path is build provenance, not a physics option
    while index < len(command):
        token = command[index]
        attached_value: str | None = None
        option = token
        if token.startswith("--") and "=" in token:
            option, attached_value = token.split("=", 1)
        option = canonical_command_token(option)

        if option in NORMALIZED_PHYSICS_OPTIONS_WITH_VALUE:
            if attached_value is None:
                if index + 1 >= len(command):
                    raise ValueError(
                        f"run command option {option} has no value in {root}"
                    )
                value = canonical_command_token(command[index + 1])
                index += 2
            else:
                value = canonical_command_token(attached_value)
                index += 1
            if option in normalized:
                raise ValueError(
                    f"run command repeats physics option {option} in {root}"
                )
            normalized[option] = value
            continue

        if option == "--antenna-file":
            if attached_value is None:
                if index + 1 >= len(command):
                    raise ValueError(
                        f"run command option {option} has no value in {root}"
                    )
                index += 2
            else:
                index += 1
            layout = observer_layout_fingerprint(root)
            canonical.extend(
                (
                    "--observer-layout-sha256",
                    layout if layout is not None else "radio-output-unavailable",
                )
            )
            continue

        if option in OMITTABLE_PHYSICS_OPTION_DEFAULTS:
            if attached_value is None:
                if index + 1 >= len(command):
                    raise ValueError(
                        f"run command option {option} has no value in {root}"
                    )
                value = canonical_command_token(command[index + 1])
                index += 2
            else:
                value = canonical_command_token(attached_value)
                index += 1
            if value != OMITTABLE_PHYSICS_OPTION_DEFAULTS[option]:
                canonical.extend((option, value))
            continue

        if option in NON_PHYSICS_OPTIONS_WITH_VALUE:
            if attached_value is None:
                if index + 1 >= len(command):
                    raise ValueError(
                        f"run command option {option} has no value in {root}"
                    )
                index += 2
            else:
                index += 1
            continue
        if option in NON_PHYSICS_FLAGS:
            if attached_value is not None:
                raise ValueError(
                    f"flag {option} unexpectedly has a value in {root}"
                )
            index += 1
            continue

        canonical.append(option)
        if attached_value is not None:
            canonical.append(
                canonical_command_token(attached_value)
            )
        index += 1

    for option in NORMALIZED_PHYSICS_OPTIONS_WITH_VALUE:
        explicit_value = normalized.get(option)
        implicit_value = implicit.get(option)
        if (
            explicit_value is not None
            and implicit_value is not None
            and explicit_value != implicit_value
        ):
            raise ValueError(
                f"explicit {option}={explicit_value} in {root} conflicts "
                f"with documented implicit value {implicit_value}"
            )
        value = explicit_value if explicit_value is not None else implicit_value
        if value is not None:
            canonical.extend((option, value))
    return tuple(canonical)


def shower_number(name: str) -> int:
    prefix = "shower_"
    if not name.startswith(prefix):
        raise ValueError(f"invalid shower key: {name}")
    return int(name[len(prefix) :])


def validate_completion(root: Path, expect_gpu: bool) -> tuple[int, ...]:
    timing = read_yaml(root / "simulation_timing" / "summary.yaml")
    if not isinstance(timing, dict) or not timing:
        raise ValueError(f"invalid simulation timing summary in {root}")
    showers: list[int] = []
    for name, record in timing.items():
        if (
            not isinstance(record, dict)
            or record.get("closed") is not True
            or record.get("status") != "closed"
        ):
            raise ValueError(f"incomplete timing record {name} in {root}")
        showers.append(shower_number(str(name)))
    showers.sort()
    if showers != list(range(len(showers))):
        raise ValueError(f"shower IDs are not contiguous from zero in {root}")

    if expect_gpu:
        gpu = read_yaml(root / "gpu_em" / "summary.yaml")
        for shower in showers:
            record = gpu.get(f"shower_{shower}") if isinstance(gpu, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("complete") is not True
                or record.get("status") != "complete"
            ):
                raise ValueError(f"incomplete CUDA record shower_{shower} in {root}")
            statistics = record.get("statistics", {})
            if (
                statistics.get("queue_overflows", 0) != 0
                or statistics.get("cross_species", {}).get("host_spills", 0) != 0
                or statistics.get("profile", {}).get("fixed_point_overflows", 0) != 0
                or statistics.get("profile", {}).get("invalid_records", 0) != 0
            ):
                raise ValueError(
                    f"CUDA integrity counter is nonzero for shower_{shower} in {root}"
                )
            ledger = statistics.get(
                "energy_ledger", record.get("energy_ledger")
            )
            if isinstance(ledger, dict) and ledger.get(
                "complete_coverage"
            ) is True:
                relative_error = float(
                    ledger.get("relative_closure_error", math.inf)
                )
                tolerance = float(
                    ledger.get("acceptance_tolerance", 1.0e-4)
                )
                if (
                    ledger.get("accepted") is not True
                    or not math.isfinite(relative_error)
                    or relative_error > tolerance
                ):
                    raise ValueError(
                        "CUDA strict energy ledger failed for "
                        f"shower_{shower} in {root}"
                    )
    return tuple(showers)


def read_parquet(
    path: Path,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"required Parquet file is missing: {path}")
    required = tuple(required_columns)
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(required) - schema_names
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = pq.read_table(path, columns=list(required)).to_pandas()
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains NaN or infinity")
    return frame


def validate_shower_ids(
    frame: pd.DataFrame,
    showers: tuple[int, ...],
    label: str,
    allow_missing: bool = False,
) -> None:
    observed = set(int(value) for value in frame["shower"].unique())
    expected = set(showers)
    if observed - expected or (not allow_missing and observed != expected):
        raise ValueError(
            f"{label} shower IDs differ: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )


def quadratic_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    maximum = int(np.argmax(y))
    x_peak = float(x[maximum])
    y_peak = float(y[maximum])
    if maximum == 0 or maximum + 1 == x.size:
        return x_peak, y_peak
    left, center, right = (
        float(y[maximum - 1]),
        float(y[maximum]),
        float(y[maximum + 1]),
    )
    denominator = left - 2.0 * center + right
    spacing_left = float(x[maximum] - x[maximum - 1])
    spacing_right = float(x[maximum + 1] - x[maximum])
    if (
        denominator >= 0.0
        or denominator == 0.0
        or not math.isclose(spacing_left, spacing_right, rel_tol=1.0e-6)
    ):
        return x_peak, y_peak
    offset = 0.5 * (left - right) / denominator
    if abs(offset) > 1.0:
        return x_peak, y_peak
    x_peak += offset * spacing_left
    y_peak = center - 0.25 * (left - right) * offset
    return x_peak, y_peak


def curve_matrix(
    frame: pd.DataFrame,
    showers: tuple[int, ...],
    value: str,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate: np.ndarray | None = None
    rows: list[np.ndarray] = []
    for shower in showers:
        selected = frame.loc[frame["shower"] == shower].sort_values("X")
        x = selected["X"].to_numpy(dtype=np.float64)
        values = selected[value].to_numpy(dtype=np.float64)
        if coordinate is None:
            coordinate = x
        elif coordinate.shape != x.shape or not np.array_equal(coordinate, x):
            raise ValueError(f"inconsistent X grid for {value}, shower {shower}")
        rows.append(values)
    assert coordinate is not None
    return coordinate, np.stack(rows)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0.0 else 0.0


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    total = float(np.sum(weights))
    if total <= 0.0 or values.size == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * total, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def normalized_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    result, _ = np.histogram(values, bins=edges, weights=weights)
    result = result.astype(np.float64)
    total = float(np.sum(result))
    if total > 0.0:
        result /= total
    return result


def primary_energies(
    root: Path,
    showers: tuple[int, ...],
) -> dict[int, float]:
    summary = read_yaml(root / "primary" / "summary.yaml")
    energies: dict[int, float] = {}
    for shower in showers:
        record = summary.get(f"shower_{shower}") if isinstance(summary, dict) else None
        if not isinstance(record, dict):
            raise ValueError(f"missing primary summary for shower_{shower} in {root}")
        energies[shower] = float(record["total_energy"])
    return energies


def energy_summary(
    root: Path,
    showers: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    summary = read_yaml(root / "energyloss" / "summary.yaml")
    result: dict[int, dict[str, float]] = {}
    for shower in showers:
        record = summary.get(f"shower_{shower}") if isinstance(summary, dict) else None
        if not isinstance(record, dict):
            raise ValueError(f"missing energy summary for shower_{shower} in {root}")
        result[shower] = {
            "sum_dEdX": float(record["sum_dEdX"]),
            "Xmax": float(record["Xmax"]),
            "dEdXmax": float(record["dEdXmax"]),
        }
    return result


def extract_ensemble(
    name: str,
    root: Path,
    expect_gpu: bool,
    *,
    allow_legacy_provenance: bool = False,
    implicit_physics_options: dict[str, str] | None = None,
) -> Ensemble:
    provenance = read_validation_provenance(
        root,
        "cuda" if expect_gpu else "proposal",
        allow_legacy=allow_legacy_provenance,
    )
    showers = validate_completion(root, expect_gpu)
    profiles = read_parquet(
        root / "profile" / "profile.parquet",
        ("shower", "X", *PROFILE_COLUMNS),
    )
    production_profiles = read_parquet(
        root / "production_profile" / "profile.parquet",
        ("shower", "X", *PRODUCTION_PROFILE_COLUMNS),
    )
    losses = read_parquet(
        root / "energyloss" / "dEdX.parquet",
        ("shower", "X", "total"),
    )
    particles = read_parquet(
        root / "particles" / "particles.parquet",
        (
            "shower",
            "pdg",
            "kinetic_energy",
            "x",
            "y",
            "time",
            "weight",
        ),
    )
    validate_shower_ids(profiles, showers, f"{name} profile")
    validate_shower_ids(
        production_profiles,
        showers,
        f"{name} muon-production profile",
    )
    validate_shower_ids(losses, showers, f"{name} energy loss")
    validate_shower_ids(particles, showers, f"{name} particles", allow_missing=True)
    if (
        (profiles[list(PROFILE_COLUMNS)] < 0.0).any().any()
        or (
            production_profiles[list(PRODUCTION_PROFILE_COLUMNS)] < 0.0
        ).any().any()
        or (losses["total"] < 0.0).any()
        or (particles[["kinetic_energy", "weight"]] < 0.0).any().any()
    ):
        raise ValueError(f"{name} output contains a negative physical weight or value")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in PROFILE_COLUMNS:
        curves[f"profile_{column}"] = curve_matrix(
            profiles, showers, column
        )
    profile_axis = curves["profile_charged"][0]
    em_matrix = (
        curves["profile_photon"][1]
        + curves["profile_electron"][1]
        + curves["profile_positron"][1]
    )
    curves["profile_electron_positron"] = (
        profile_axis,
        curves["profile_electron"][1]
        + curves["profile_positron"][1],
    )
    curves["profile_muon"] = (
        profile_axis,
        curves["profile_muplus"][1]
        + curves["profile_muminus"][1],
    )
    curves["profile_em"] = (profile_axis, em_matrix)
    for column in PRODUCTION_PROFILE_COLUMNS:
        observable = column.replace("-", "_")
        curves[f"muon_production_parent_{observable}"] = curve_matrix(
            production_profiles,
            showers,
            column,
        )
    curves["energy_deposit"] = curve_matrix(losses, showers, "total")

    primaries = primary_energies(root, showers)
    loss_summary = energy_summary(root, showers)
    scalar_rows: list[dict[str, float | int]] = []
    radial_histograms: list[np.ndarray] = []
    energy_histograms: list[np.ndarray] = []
    time_histograms: list[np.ndarray] = []

    for row_index, shower in enumerate(showers):
        charged = curves["profile_charged"][1][row_index]
        charged_xmax, charged_max = quadratic_peak(profile_axis, charged)
        row: dict[str, float | int] = {
            "shower": shower,
            "primary_total_energy_GeV": primaries[shower],
            "profile_xmax_charged_gcm2": charged_xmax,
            "profile_charged_max": charged_max,
        }
        for column in PROFILE_COLUMNS:
            coordinate, matrix = curves[f"profile_{column}"]
            row[f"profile_{column}_integral"] = float(
                np.trapz(matrix[row_index], coordinate)
            )
        for column in DERIVED_PROFILE_COLUMNS:
            coordinate, matrix = curves[f"profile_{column}"]
            row[f"profile_{column}_integral"] = float(
                np.trapz(matrix[row_index], coordinate)
            )
        summary = loss_summary[shower]
        deposit_coordinate, deposit_matrix = curves["energy_deposit"]
        deposit_xmax, deposit_max = quadratic_peak(
            deposit_coordinate,
            deposit_matrix[row_index],
        )
        row["energy_deposit_sum_GeV"] = summary["sum_dEdX"]
        row["energy_deposit_xmax_gcm2"] = deposit_xmax
        row["energy_deposit_max_GeV"] = deposit_max
        row["energy_deposit_fit_xmax_gcm2"] = summary["Xmax"]
        row["energy_deposit_fit_max_GeV"] = summary["dEdXmax"]
        row["energy_deposit_fraction"] = (
            summary["sum_dEdX"] / primaries[shower]
        )

        ground = particles.loc[particles["shower"] == shower]
        pdg = ground["pdg"].to_numpy(dtype=np.int32)
        kinetic = ground["kinetic_energy"].to_numpy(dtype=np.float64)
        weights = ground["weight"].to_numpy(dtype=np.float64)
        em_mask = np.isin(pdg, (-11, 11, 22))
        em_pdg = pdg[em_mask]
        em_kinetic = kinetic[em_mask]
        em_weights = weights[em_mask]
        radius = np.hypot(
            ground["x"].to_numpy(dtype=np.float64)[em_mask],
            ground["y"].to_numpy(dtype=np.float64)[em_mask],
        )
        times = ground["time"].to_numpy(dtype=np.float64)[em_mask]

        for particle_name, particle_pdg in (
            ("photon", 22),
            ("electron", 11),
            ("positron", -11),
        ):
            particle_mask = em_pdg == particle_pdg
            row[f"ground_{particle_name}_weighted_count"] = float(
                np.sum(em_weights[particle_mask])
            )
        weighted_count = float(np.sum(em_weights))
        weighted_kinetic = float(np.sum(em_weights * em_kinetic))
        electron_mass = 0.00051099895
        weighted_total_energy = float(
            np.sum(
                em_weights
                * (
                    em_kinetic
                    + np.where(np.abs(em_pdg) == 11, electron_mass, 0.0)
                )
            )
        )
        radius_mean = weighted_mean(radius, em_weights)
        radius_rms = (
            math.sqrt(weighted_mean(radius * radius, em_weights))
            if weighted_count > 0.0
            else 0.0
        )
        time_center = weighted_quantile(times, em_weights, 0.5)
        time_rms = (
            math.sqrt(
                weighted_mean((times - time_center) ** 2, em_weights)
            )
            if weighted_count > 0.0
            else 0.0
        )
        row.update(
            {
                "ground_em_weighted_count": weighted_count,
                "ground_em_kinetic_energy_GeV": weighted_kinetic,
                "ground_em_total_energy_GeV": weighted_total_energy,
                "ground_radius_mean_m": radius_mean,
                "ground_radius_rms_m": radius_rms,
                "ground_time_median_s": time_center,
                "ground_time_rms_s": time_rms,
                "ground_energy_fraction": (
                    weighted_total_energy / primaries[shower]
                ),
                "energy_closure_fraction": (
                    summary["sum_dEdX"] + weighted_total_energy
                )
                / primaries[shower],
            }
        )
        scalar_rows.append(row)
        radial_histograms.append(
            normalized_histogram(radius, em_weights, RADIAL_EDGES_M)
        )
        energy_histograms.append(
            normalized_histogram(em_kinetic, em_weights, ENERGY_EDGES_GEV)
        )
        time_histograms.append(
            normalized_histogram(
                np.abs(times - time_center),
                em_weights,
                TIME_RESIDUAL_EDGES_S,
            )
        )

    histograms = {
        "ground_radial_fraction": (
            RADIAL_EDGES_M,
            np.stack(radial_histograms),
        ),
        "ground_energy_fraction": (
            ENERGY_EDGES_GEV,
            np.stack(energy_histograms),
        ),
        "ground_time_residual_fraction": (
            TIME_RESIDUAL_EDGES_S,
            np.stack(time_histograms),
        ),
    }
    metadata = {
        "events": len(showers),
        "source": str(root),
        "gpu_integrity_checked": expect_gpu,
        "physics_configuration": canonical_physics_configuration(
            root,
            implicit_physics_options=implicit_physics_options,
        ),
        "implicit_physics_options": dict(
            implicit_physics_options or {}
        ),
        "validation_provenance": provenance,
        "provenance_fingerprint": provenance_fingerprint(provenance),
    }
    return Ensemble(
        name=name,
        root=root,
        showers=showers,
        scalars=pd.DataFrame(scalar_rows).set_index("shower"),
        curves=curves,
        histograms=histograms,
        metadata=metadata,
    )


def concatenate_ensembles(
    name: str,
    ensembles: list[Ensemble],
    *,
    allow_mixed_provenance: bool = False,
) -> Ensemble:
    if not ensembles:
        raise ValueError(f"cannot concatenate an empty {name} ensemble list")
    scalar_columns = list(ensembles[0].scalars.columns)
    curve_names = set(ensembles[0].curves)
    histogram_names = set(ensembles[0].histograms)
    reference_configuration = ensembles[0].metadata.get(
        "physics_configuration"
    )
    reference_provenance = ensembles[0].metadata.get(
        "provenance_fingerprint"
    )
    for ensemble in ensembles[1:]:
        if list(ensemble.scalars.columns) != scalar_columns:
            raise ValueError(f"{name} shard scalar observable sets differ")
        if set(ensemble.curves) != curve_names:
            raise ValueError(f"{name} shard curve observable sets differ")
        if set(ensemble.histograms) != histogram_names:
            raise ValueError(f"{name} shard histogram observable sets differ")
        if (
            ensemble.metadata.get("physics_configuration")
            != reference_configuration
        ):
            raise ValueError(
                f"{name} shard physics configurations differ: "
                f"{ensembles[0].root} versus {ensemble.root}"
            )
        if (
            not allow_mixed_provenance
            and ensemble.metadata.get("provenance_fingerprint")
            != reference_provenance
        ):
            raise ValueError(
                f"{name} shard build/table provenance differs: "
                f"{ensembles[0].root} versus {ensemble.root}"
            )

    scalars = pd.concat(
        [ensemble.scalars for ensemble in ensembles],
        ignore_index=True,
    )
    scalars.index = pd.Index(
        range(len(scalars)),
        name="shower",
    )
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for observable in sorted(curve_names):
        axis = ensembles[0].curves[observable][0]
        matrices: list[np.ndarray] = []
        for ensemble in ensembles:
            candidate_axis, matrix = ensemble.curves[observable]
            if (
                candidate_axis.shape != axis.shape
                or not np.array_equal(candidate_axis, axis)
            ):
                raise ValueError(
                    f"{name} shard {observable} coordinate grids differ"
                )
            matrices.append(matrix)
        curves[observable] = (axis, np.concatenate(matrices, axis=0))

    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for observable in sorted(histogram_names):
        edges = ensembles[0].histograms[observable][0]
        matrices = []
        for ensemble in ensembles:
            candidate_edges, matrix = ensemble.histograms[observable]
            if (
                candidate_edges.shape != edges.shape
                or not np.array_equal(candidate_edges, edges)
            ):
                raise ValueError(
                    f"{name} shard {observable} bin edges differ"
                )
            matrices.append(matrix)
        histograms[observable] = (
            edges,
            np.concatenate(matrices, axis=0),
        )

    total = len(scalars)
    provenance_fingerprints = list(
        dict.fromkeys(
            ensemble.metadata.get("provenance_fingerprint")
            for ensemble in ensembles
        )
    )
    return Ensemble(
        name=name,
        root=ensembles[0].root,
        showers=tuple(range(total)),
        scalars=scalars,
        curves=curves,
        histograms=histograms,
        metadata={
            "events": total,
            "shards": len(ensembles),
            "sources": [
                str(ensemble.root)
                for ensemble in ensembles
            ],
            "gpu_integrity_checked": all(
                bool(
                    ensemble.metadata.get(
                        "gpu_integrity_checked"
                    )
                )
                for ensemble in ensembles
            ),
            "physics_configuration": reference_configuration,
            "validation_provenance":
                ensembles[0].metadata.get(
                    "validation_provenance"
                ),
            "provenance_fingerprint": reference_provenance,
            "provenance_fingerprints": provenance_fingerprints,
            "mixed_provenance_allowed": bool(
                allow_mixed_provenance
            ),
        },
    )


def sample_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if count > 1 else 0.0
    return {
        "events": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(count),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def empirical_ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return math.nan
    support = np.sort(np.concatenate((x, y)))
    cdf_x = np.searchsorted(x, support, side="right") / x.size
    cdf_y = np.searchsorted(y, support, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def relative_quantile_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return math.nan
    quantiles = np.linspace(0.0, 1.0, 1001)
    distance = float(
        np.mean(
            np.abs(
                np.quantile(x, quantiles) - np.quantile(y, quantiles)
            )
        )
    )
    scale = max(
        float(np.mean(np.abs(x))),
        float(np.mean(np.abs(y))),
        np.finfo(np.float64).tiny,
    )
    return distance / scale


def bootstrap_relative_mean_interval(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> list[float]:
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    a_means = np.mean(
        a[rng.integers(0, a.size, size=(repetitions, a.size))], axis=1
    )
    b_means = np.mean(
        b[rng.integers(0, b.size, size=(repetitions, b.size))], axis=1
    )
    floor = max(
        float(np.mean(np.abs(a))) * 1.0e-12,
        np.finfo(np.float64).tiny,
    )
    shifts = (b_means - a_means) / np.maximum(np.abs(a_means), floor)
    return [
        float(np.quantile(shifts, 0.025)),
        float(np.quantile(shifts, 0.975)),
    ]


def compare_scalar(
    cpu_values: np.ndarray,
    gpu_values: np.ndarray,
    relative_tolerance: float,
    sigma_limit: float,
    gate: bool,
    bootstrap_seed: int = 20260729,
    bootstrap_repetitions: int = 5000,
) -> dict[str, Any]:
    cpu = sample_statistics(cpu_values)
    gpu = sample_statistics(gpu_values)
    difference = float(gpu["mean"] - cpu["mean"])
    scale = abs(float(cpu["mean"]))
    if scale > 0.0:
        relative_difference = abs(difference) / scale
    else:
        relative_difference = 0.0 if difference == 0.0 else None
    standard_error = math.hypot(
        float(cpu["standard_error"]),
        float(gpu["standard_error"]),
    )
    if standard_error > 0.0:
        z_score = abs(difference) / standard_error
    else:
        z_score = 0.0 if difference == 0.0 else math.inf
    if scale > 0.0:
        relative_standard_error = standard_error / scale
        sigma_scaled_relative_precision = (
            sigma_limit * standard_error / scale
        )
    else:
        relative_standard_error = (
            0.0 if standard_error == 0.0 else None
        )
        sigma_scaled_relative_precision = (
            0.0 if standard_error == 0.0 else None
        )
    relative_pass = relative_difference is not None and (
        relative_difference <= relative_tolerance
    )
    statistical_pass = z_score <= sigma_limit
    ks_distance = empirical_ks_distance(cpu_values, gpu_values)
    ks_95_critical = 1.36 * math.sqrt(
        (cpu_values.size + gpu_values.size)
        / (cpu_values.size * gpu_values.size)
    )
    cpu_median = float(np.median(cpu_values))
    gpu_median = float(np.median(gpu_values))
    median_scale = max(abs(cpu_median), np.finfo(np.float64).tiny)
    pooled_variance_denominator = cpu_values.size + gpu_values.size - 2
    pooled_standard_deviation = (
        math.sqrt(
            (
                (cpu_values.size - 1) * float(np.var(cpu_values, ddof=1))
                + (gpu_values.size - 1) * float(np.var(gpu_values, ddof=1))
            )
            / pooled_variance_denominator
        )
        if pooled_variance_denominator > 0
        else 0.0
    )
    if not gate:
        outcome = "diagnostic_only"
    elif relative_pass and statistical_pass:
        outcome = "passed"
    elif not relative_pass and statistical_pass:
        outcome = "relative_threshold_failed_but_statistically_inconclusive"
    elif relative_pass and not statistical_pass:
        outcome = "statistically_significant_difference_within_relative_tolerance"
    else:
        outcome = "relative_and_statistical_thresholds_failed"
    return {
        "proposal": cpu,
        "cuda": gpu,
        "difference": difference,
        "relative_difference": relative_difference,
        "combined_standard_error": standard_error,
        "relative_combined_standard_error": relative_standard_error,
        "sigma_scaled_relative_precision": sigma_scaled_relative_precision,
        "absolute_z_score": z_score,
        "distribution_diagnostics": {
            "proposal_median": cpu_median,
            "cuda_median": gpu_median,
            "signed_relative_median_shift": (
                gpu_median - cpu_median
            )
            / median_scale,
            "empirical_KS_distance": ks_distance,
            "KS_95pct_critical_value": ks_95_critical,
            "KS_below_95pct_critical_value": ks_distance <= ks_95_critical,
            "relative_quantile_wasserstein": (
                relative_quantile_wasserstein(cpu_values, gpu_values)
            ),
            "standardized_mean_difference": (
                difference / pooled_standard_deviation
                if pooled_standard_deviation > 0.0
                else (0.0 if difference == 0.0 else math.inf)
            ),
            "variance_ratio_cuda_over_proposal": (
                float(np.var(gpu_values, ddof=1))
                / float(np.var(cpu_values, ddof=1))
                if cpu_values.size > 1
                and float(np.var(cpu_values, ddof=1)) > 0.0
                else None
            ),
            "bootstrap_signed_relative_mean_shift_95pct": (
                bootstrap_relative_mean_interval(
                    cpu_values,
                    gpu_values,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                )
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
        },
        "acceptance_gate": gate,
        "relative_pass": relative_pass,
        "statistical_pass": statistical_pass,
        "outcome": outcome,
        "passed": (relative_pass and statistical_pass) if gate else True,
    }


def curve_comparison(
    coordinate: np.ndarray,
    cpu: np.ndarray,
    gpu: np.ndarray,
    relative_tolerance: float,
    sigma_limit: float,
    active_fraction: float,
    minimum_bin_pass_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, float | bool | None]]]:
    if cpu.ndim != 2 or gpu.ndim != 2 or cpu.shape[1] != gpu.shape[1]:
        raise ValueError(
            f"curve bin counts differ: {cpu.shape} versus {gpu.shape}"
        )
    cpu_mean = np.mean(cpu, axis=0)
    gpu_mean = np.mean(gpu, axis=0)
    cpu_std = np.std(cpu, axis=0, ddof=1) if cpu.shape[0] > 1 else np.zeros_like(cpu_mean)
    gpu_std = np.std(gpu, axis=0, ddof=1) if gpu.shape[0] > 1 else np.zeros_like(gpu_mean)
    combined_se = np.sqrt(
        cpu_std * cpu_std / cpu.shape[0]
        + gpu_std * gpu_std / gpu.shape[0]
    )
    difference = gpu_mean - cpu_mean
    scale = float(np.max(np.abs(cpu_mean))) if cpu_mean.size else 0.0
    active = np.abs(cpu_mean) >= active_fraction * scale if scale > 0.0 else np.zeros_like(cpu_mean, dtype=bool)
    z_score = np.zeros_like(difference)
    nonzero_error = combined_se > 0.0
    z_score[nonzero_error] = np.abs(difference[nonzero_error]) / combined_se[nonzero_error]
    z_score[~nonzero_error & (difference != 0.0)] = np.inf
    active_count = int(np.count_nonzero(active))
    within = active & (z_score <= sigma_limit)
    bin_pass_fraction = (
        float(np.count_nonzero(within)) / active_count if active_count else 1.0
    )
    denominator = float(np.sum(np.abs(cpu_mean[active])))
    relative_l1 = (
        float(np.sum(np.abs(difference[active]))) / denominator
        if denominator > 0.0
        else 0.0
    )
    finite_active_z = z_score[active & np.isfinite(z_score)]
    rms_z = (
        float(np.sqrt(np.mean(finite_active_z * finite_active_z)))
        if finite_active_z.size
        else 0.0
    )
    summary = {
        "bins": int(coordinate.size),
        "active_bins": active_count,
        "relative_l1": relative_l1,
        "rms_active_z_score": rms_z,
        "active_bin_pass_fraction": bin_pass_fraction,
        "relative_pass": relative_l1 <= relative_tolerance,
        "statistical_pass": bin_pass_fraction >= minimum_bin_pass_fraction,
        # The production requirement for a distribution is statistical
        # consistency. The 1% requirement applies to the explicitly gated
        # scalar means. Relative L1 remains a useful convergence diagnostic,
        # but does not reject a statistically consistent finite ensemble.
        "relative_acceptance_gate": False,
        "passed": bin_pass_fraction >= minimum_bin_pass_fraction,
    }
    rows: list[dict[str, float | bool | None]] = []
    for index, x in enumerate(coordinate):
        relative = (
            abs(float(difference[index])) / abs(float(cpu_mean[index]))
            if cpu_mean[index] != 0.0
            else None
        )
        rows.append(
            {
                "coordinate": float(x),
                "proposal_mean": float(cpu_mean[index]),
                "cuda_mean": float(gpu_mean[index]),
                "difference": float(difference[index]),
                "relative_difference": relative,
                "combined_standard_error": float(combined_se[index]),
                "absolute_z_score": float(z_score[index]),
                "active": bool(active[index]),
            }
        )
    return summary, rows


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def compare_ensembles(
    proposal: Ensemble,
    cuda: Ensemble,
    relative_tolerance: float,
    sigma_limit: float,
    active_fraction: float,
    minimum_bin_pass_fraction: float,
    key_scalar_metrics: tuple[str, ...] =
        DEFAULT_KEY_SCALAR_METRICS,
    allow_cross_build_reference: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    proposal_configuration = proposal.metadata.get(
        "physics_configuration"
    )
    cuda_configuration = cuda.metadata.get(
        "physics_configuration"
    )
    if proposal_configuration != cuda_configuration:
        raise ValueError(
            "proposal and CUDA physics configurations differ"
        )
    proposal_provenance = proposal.metadata.get(
        "provenance_fingerprint"
    )
    cuda_provenance = cuda.metadata.get(
        "provenance_fingerprint"
    )
    if (proposal_provenance is None) != (cuda_provenance is None):
        raise ValueError(
            "proposal and CUDA validation provenance availability differs"
        )
    if (
        proposal_provenance is not None
        and cuda_provenance is not None
        and proposal_provenance[2] != cuda_provenance[2]
        and not allow_cross_build_reference
    ):
        raise ValueError(
            "proposal and CUDA executable build provenance differs"
        )
    scalar_names = list(proposal.scalars.columns)
    if scalar_names != list(cuda.scalars.columns):
        raise ValueError("proposal and CUDA scalar observable sets differ")

    scalar_results: dict[str, Any] = {}
    scalar_pass = True
    scalar_outcomes: dict[str, list[str]] = {}
    for metric_index, metric in enumerate(scalar_names):
        gated = metric in key_scalar_metrics
        comparison = compare_scalar(
            proposal.scalars[metric].to_numpy(dtype=np.float64),
            cuda.scalars[metric].to_numpy(dtype=np.float64),
            relative_tolerance,
            sigma_limit,
            gated,
            bootstrap_seed=20260729 + metric_index,
        )
        scalar_results[metric] = comparison
        scalar_pass = scalar_pass and bool(comparison["passed"])
        if gated:
            scalar_outcomes.setdefault(
                str(comparison["outcome"]), []
            ).append(metric)

    curve_rows: list[dict[str, Any]] = []
    curve_results: dict[str, Any] = {}
    curve_pass = True
    for family, cpu_collection, gpu_collection in (
        ("longitudinal", proposal.curves, cuda.curves),
        ("ground_histogram", proposal.histograms, cuda.histograms),
    ):
        if set(cpu_collection) != set(gpu_collection):
            raise ValueError(f"{family} observable sets differ")
        for observable in sorted(cpu_collection):
            cpu_axis, cpu_matrix = cpu_collection[observable]
            gpu_axis, gpu_matrix = gpu_collection[observable]
            if cpu_axis.shape != gpu_axis.shape or not np.array_equal(cpu_axis, gpu_axis):
                raise ValueError(f"{observable} coordinate grids differ")
            # Histograms have N+1 edges for N values. Use the lower edge as
            # a stable coordinate; full edges remain in the JSON metadata.
            coordinate = (
                cpu_axis[:-1]
                if cpu_axis.size == cpu_matrix.shape[1] + 1
                else cpu_axis
            )
            comparison, rows = curve_comparison(
                coordinate,
                cpu_matrix,
                gpu_matrix,
                relative_tolerance,
                sigma_limit,
                active_fraction,
                minimum_bin_pass_fraction,
            )
            comparison["family"] = family
            acceptance_gate = not any(
                observable.startswith(prefix)
                for prefix in DIAGNOSTIC_CURVE_PREFIXES
            )
            comparison["acceptance_gate"] = acceptance_gate
            if cpu_axis.size == cpu_matrix.shape[1] + 1:
                comparison["bin_edges"] = cpu_axis.tolist()
            curve_results[observable] = comparison
            if acceptance_gate:
                curve_pass = curve_pass and bool(comparison["passed"])
            for row in rows:
                row.update({"family": family, "observable": observable})
                curve_rows.append(row)

    passed = scalar_pass and curve_pass
    report = {
        "status": "passed" if passed else "failed",
        "configuration": {
            "relative_tolerance": relative_tolerance,
            "sigma_limit": sigma_limit,
            "active_fraction": active_fraction,
            "minimum_bin_pass_fraction": minimum_bin_pass_fraction,
            "key_scalar_metrics": list(key_scalar_metrics),
        },
        "proposal": proposal.metadata,
        "cuda": cuda.metadata,
        "acceptance": {
            "scalar_pass": scalar_pass,
            "curve_pass": curve_pass,
            "passed": passed,
            "key_scalar_outcomes": scalar_outcomes,
        },
        "scalars": scalar_results,
        "curves": curve_results,
    }
    return finite_json(report), pd.DataFrame(curve_rows)


def main() -> int:
    args = parse_args()
    if args.minimum_events < 2:
        raise ValueError("minimum event count must be at least two")
    if not 0.0 < args.relative_tolerance < 1.0:
        raise ValueError("relative tolerance must be in (0, 1)")
    if args.sigma_limit <= 0.0:
        raise ValueError("sigma limit must be positive")
    if not 0.0 <= args.active_fraction < 1.0:
        raise ValueError("active fraction must be in [0, 1)")
    if not 0.0 < args.minimum_bin_pass_fraction <= 1.0:
        raise ValueError("minimum bin pass fraction must be in (0, 1]")
    implicit_model_set = (
        args.proposal_implicit_geomagnetic_model is not None
    )
    implicit_year_set = (
        args.proposal_implicit_geomagnetic_year is not None
    )
    if implicit_model_set != implicit_year_set:
        raise ValueError(
            "proposal implicit geomagnetic model and year must be provided "
            "together"
        )
    proposal_implicit_physics_options: dict[str, str] = {}
    if implicit_model_set:
        if not math.isfinite(args.proposal_implicit_geomagnetic_year):
            raise ValueError(
                "proposal implicit geomagnetic year must be finite"
            )
        proposal_implicit_physics_options = {
            "--geomagnetic-model":
                args.proposal_implicit_geomagnetic_model,
            "--geomagnetic-year":
                f"{args.proposal_implicit_geomagnetic_year:.17g}",
        }
    cuda_implicit_model_set = args.cuda_implicit_geomagnetic_model is not None
    cuda_implicit_year_set = args.cuda_implicit_geomagnetic_year is not None
    if cuda_implicit_model_set != cuda_implicit_year_set:
        raise ValueError(
            "CUDA implicit geomagnetic model and year must be provided together"
        )
    cuda_implicit_physics_options: dict[str, str] = {}
    if cuda_implicit_model_set:
        if not math.isfinite(args.cuda_implicit_geomagnetic_year):
            raise ValueError("CUDA implicit geomagnetic year must be finite")
        cuda_implicit_physics_options = {
            "--geomagnetic-model": args.cuda_implicit_geomagnetic_model,
            "--geomagnetic-year": f"{args.cuda_implicit_geomagnetic_year:.17g}",
        }

    proposal_roots = [
        require_directory(path, "proposal output")
        for path in args.proposal
    ]
    cuda_roots = [
        require_directory(path, "CUDA output")
        for path in args.cuda
    ]
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")

    proposal = concatenate_ensembles(
        "proposal",
        [
            extract_ensemble(
                "proposal", root,
                expect_gpu=False,
                allow_legacy_provenance=
                    args.allow_legacy_provenance,
                implicit_physics_options=
                    proposal_implicit_physics_options,
            )
            for root in proposal_roots
        ],
        allow_mixed_provenance=args.allow_mixed_proposal_builds,
    )
    cuda = concatenate_ensembles(
        "cuda",
        [
            extract_ensemble(
                "cuda", root,
                expect_gpu=True,
                allow_legacy_provenance=
                    args.allow_legacy_provenance,
                implicit_physics_options=
                    cuda_implicit_physics_options,
            )
            for root in cuda_roots
        ],
        allow_mixed_provenance=args.allow_mixed_cuda_builds,
    )
    if (
        len(proposal.showers) < args.minimum_events
        or len(cuda.showers) < args.minimum_events
    ):
        raise ValueError(
            f"at least {args.minimum_events} events are required in each ensemble"
        )
    report, curve_rows = compare_ensembles(
        proposal,
        cuda,
        args.relative_tolerance,
        args.sigma_limit,
        args.active_fraction,
        args.minimum_bin_pass_fraction,
        tuple(args.key_scalar)
        if args.key_scalar
        else DEFAULT_KEY_SCALAR_METRICS,
        args.allow_cross_build_reference,
    )

    output.mkdir(parents=True)
    combined_scalars = pd.concat(
        [
            proposal.scalars.assign(backend="proposal"),
            cuda.scalars.assign(backend="cuda"),
        ]
    ).reset_index()
    combined_scalars.to_csv(output / "per_shower_observables.csv", index=False)
    curve_rows.to_csv(output / "curve_comparison.csv", index=False)
    with (output / "comparison.json").open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")

    print(json.dumps(report["acceptance"], indent=2))
    print(f"summary: {output / 'comparison.json'}")
    return 2 if args.fail_on_acceptance and report["status"] != "passed" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ensemble comparison failed: {error}")
        raise SystemExit(1)
