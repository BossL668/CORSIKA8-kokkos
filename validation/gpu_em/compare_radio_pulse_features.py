#!/usr/bin/env python3
"""Validate geomagnetic radio-pulse features for two CORSIKA ensembles.

The program is analysis-only: it never launches a shower simulation.  It can
either discover the formal arms produced by
``run_gpu_physics_source_ensemble.py`` or consume repeated, explicit CORSIKA
output directories.  The latter form is suitable for CPU-PROPOSAL versus
proposal-native comparisons.

Pulse projection, peak selection and pulse-width fitting are imported from
the user's ``pulse_analysis_modular`` checkout.  The statistical unit is one
shower after antennas at the same shower-axis distance have been aggregated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import ks_2samp


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_geomagnetic_pulse_distributions import (  # noqa: E402
    ALGORITHMS,
    aggregate_by_shower,
    apply_reference_width_filter,
    extract_antenna_rows,
    load_reference_apis,
)
from analyze_geomagnetic_radial_comparison import (  # noqa: E402
    add_aggregate_r_perp_column,
    add_extracted_coordinate_columns,
    bootstrap_ratio,
    central_statistic,
    clustered_available_radii,
    load_records,
)
from compare_cuda_replay import radio_observer_layout  # noqa: E402


ROLES = ("reference", "candidate")
ROLE_COLORS = {"reference": "#1565c0", "candidate": "#d84315"}
METRICS = (
    (
        "amplitude",
        "geomagnetic_amplitude_geomean_V_per_m",
        r"$|E_{\mathbf{v}\times\mathbf{B}}|$ peak [V m$^{-1}$]",
        True,
    ),
    (
        "width",
        "pulse_width_mean_ns",
        "Pulse width [ns]",
        False,
    ),
)
PHYSICS_ARGUMENTS: dict[str, tuple[str, Any]] = {
    "-p": ("primary_pdg", int),
    "-E": ("energy_GeV", float),
    "--zenith": ("zenith_deg", float),
    "--azimuth": ("azimuth_deg", float),
    "--geomagnetic-model": ("geomagnetic_model", str),
    "--geomagnetic-year": ("geomagnetic_year", float),
    "--shower-core-x": ("shower_core_x_m", float),
    "--shower-core-y": ("shower_core_y_m", float),
    "--ring": ("ring", int),
    "--radio-sampling-rate-ghz": ("radio_sampling_rate_GHz", float),
    "--radio-window-duration-ns": ("radio_window_duration_ns", float),
    "--radio-pretrigger-ns": ("radio_pretrigger_ns", float),
    "--max-deflection-angle": ("maximum_deflection_angle_rad", float),
    "--emcut": ("em_cut_GeV", float),
    "--emthin": ("em_thinning", float),
    "--hadcut": ("had_cut_GeV", float),
    "--mucut": ("mu_cut_GeV", float),
    "--taucut": ("tau_cut_GeV", float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--campaign-root",
        type=Path,
        help=(
            "Completed run_gpu_physics_source_ensemble.py campaign; formal "
            "c8emrt/proposal-native shards are discovered automatically."
        ),
    )
    source.add_argument(
        "--reference-output",
        type=Path,
        action="append",
        help=(
            "Reference CORSIKA output; repeat for independent shards. Must be "
            "paired with at least one --candidate-output."
        ),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        action="append",
        help="Candidate CORSIKA output; repeat for independent shards.",
    )
    parser.add_argument("--reference-label", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument(
        "--comparison-mode",
        choices=("auto", "gpu-sources", "cpu-native", "custom"),
        default="auto",
        help=(
            "Identity gate for the two arms. auto selects gpu-sources for a "
            "campaign root and cpu-native for explicit outputs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pulse-analysis-root",
        type=Path,
        required=True,
        help="Checkout containing pulse_analysis_modular/pulse_analysis.",
    )
    parser.add_argument(
        "--zenith-deg",
        type=float,
        help="Override/inject zenith when it cannot be read from config.yaml.",
    )
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        help="Override/inject azimuth when it cannot be read from config.yaml.",
    )
    parser.add_argument(
        "--magnetic-field-tesla",
        type=float,
        nargs=3,
        metavar=("BN", "BW", "BU"),
        help=(
            "NWU magnetic field. By default it is read from a GPU output's "
            "gpu_em/config.yaml and cross-checked wherever available."
        ),
    )
    parser.add_argument("--shower-core-x-m", type=float)
    parser.add_argument("--shower-core-y-m", type=float)
    parser.add_argument("--minimum-r-perp-m", type=float, default=1.0)
    parser.add_argument("--maximum-r-perp-m", type=float, default=600.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument(
        "--equivalence-relative-tolerance",
        type=float,
        default=0.10,
        help=(
            "Require each usable point's full 95%% bootstrap ratio interval "
            "inside [1-tolerance, 1+tolerance]."
        ),
    )
    parser.add_argument(
        "--minimum-count-per-arm",
        type=int,
        default=20,
        help="Minimum valid shower values per arm and radial point.",
    )
    parser.add_argument(
        "--minimum-width-valid-shower-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--shape-familywise-alpha",
        type=float,
        default=0.05,
        help="Holm-corrected familywise alpha for the pointwise KS diagnostics.",
    )
    parser.add_argument("--minimum-points-per-curve", type=int, default=2)
    parser.add_argument(
        "--allow-incomplete-campaign",
        action="store_true",
        help="Permit an interim campaign root whose manifest is not complete.",
    )
    parser.add_argument(
        "--fail-on-acceptance",
        action="store_true",
        help="Return exit code 2 when the radio feature gate does not pass.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document is not a mapping: {path}")
    return value


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pulse_implementation_identity(root: Path) -> dict[str, Any]:
    relative_paths = (
        Path("pulse_analysis/core/analysis.py"),
        Path("pulse_analysis/core/geometry.py"),
        Path("pulse_analysis/plotting/stats.py"),
        Path("pulse_analysis/plotting/config.py"),
    )
    artifacts: dict[str, Any] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"pulse-analysis implementation file is missing: {path}")
        artifacts[str(relative)] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"root": str(root), "artifacts": artifacts}


def require_outputs(paths: Iterable[Path], label: str) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"{label} output is not a directory: {path}")
        if path in seen:
            raise ValueError(f"duplicate {label} output: {path}")
        for algorithm in ALGORITHMS:
            for relative in ("config.yaml", "observers.parquet"):
                required = path / algorithm / relative
                if not required.is_file() or required.stat().st_size == 0:
                    raise ValueError(f"missing/empty radio artifact: {required}")
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise ValueError(f"no {label} outputs were selected")
    return resolved


def discover_campaign(
    root: Path, allow_incomplete: bool
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    campaign = root.expanduser().resolve()
    manifest_path = campaign / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"campaign manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a JSON object")
    if manifest.get("status") != "complete" and not allow_incomplete:
        raise ValueError(
            "campaign is not complete; use --allow-incomplete-campaign only "
            "for explicitly labelled interim diagnostics"
        )
    reference = sorted(campaign.glob("formal/batch_*/c8emrt"))
    candidate = sorted(campaign.glob("formal/batch_*/proposal-native"))
    return (
        require_outputs(reference, "c8emrt reference"),
        require_outputs(candidate, "proposal-native candidate"),
        {
            "path": str(manifest_path),
            "status": manifest.get("status"),
            "event_counts": manifest.get("event_counts"),
            "immutable_configuration": manifest.get("immutable_configuration"),
        },
    )


def command_physics(path: Path) -> dict[str, Any]:
    root_config = read_yaml(path / "config.yaml")
    raw = root_config.get("args")
    if not isinstance(raw, str):
        raise ValueError(f"CORSIKA root config has no command line: {path}")
    tokens = shlex.split(raw)
    result: dict[str, Any] = {}
    for option, (name, converter) in PHYSICS_ARGUMENTS.items():
        positions = [index for index, token in enumerate(tokens) if token == option]
        if len(positions) > 1:
            raise ValueError(f"duplicate {option} in {path / 'config.yaml'}")
        if positions:
            position = positions[0]
            if position + 1 >= len(tokens):
                raise ValueError(f"missing value after {option} in {path}")
            result[name] = converter(tokens[position + 1])
    return result


def command_option(path: Path, option: str) -> str | None:
    root_config = read_yaml(path / "config.yaml")
    raw = root_config.get("args")
    if not isinstance(raw, str):
        raise ValueError(f"CORSIKA root config has no command line: {path}")
    tokens = shlex.split(raw)
    positions = [index for index, token in enumerate(tokens) if token == option]
    if len(positions) > 1:
        raise ValueError(f"duplicate {option} in {path / 'config.yaml'}")
    if not positions:
        return None
    position = positions[0]
    if position + 1 >= len(tokens):
        raise ValueError(f"missing value after {option} in {path}")
    return tokens[position + 1]


def validate_backend_identity(
    sources: dict[str, list[Path]], mode: str
) -> dict[str, Any]:
    expected = {
        "gpu-sources": {"reference": "c8emrt", "candidate": "proposal-native"},
        "cpu-native": {"reference": "cpu-proposal", "candidate": "proposal-native"},
    }
    records: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for role in ROLES:
        for path in sources[role]:
            em_backend = command_option(path, "--em-backend") or "proposal"
            radio_backend = command_option(path, "--radio-backend") or "cpu"
            gpu_source: str | None = None
            gpu_config = path / "gpu_em" / "config.yaml"
            if gpu_config.is_file():
                gpu_source = str(read_yaml(gpu_config).get("gpu_physics_source"))
            observed = (
                "cpu-proposal"
                if em_backend == "proposal" and radio_backend == "cpu"
                else gpu_source
            )
            records[role].append(
                {
                    "path": str(path),
                    "em_backend": em_backend,
                    "radio_backend": radio_backend,
                    "gpu_physics_source": gpu_source,
                    "observed_kind": observed,
                }
            )
            if mode != "custom" and observed != expected[mode][role]:
                raise ValueError(
                    f"{role} backend identity differs in {path}: expected "
                    f"{expected[mode][role]}, observed {observed}"
                )
    return {
        "passed": True,
        "comparison_mode": mode,
        "expected": expected.get(mode),
        "outputs": records,
    }


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=1.0e-12, atol=1.0e-12))
    return left == right


def common_physics_configuration(
    sources: dict[str, list[Path]], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [
        {"role": role, "path": str(path), "values": command_physics(path)}
        for role in ROLES
        for path in sources[role]
    ]
    common: dict[str, Any] = {}
    names = sorted({name for record in records for name in record["values"]})
    checked: list[str] = []
    incomplete: list[str] = []
    for name in names:
        present = [record["values"][name] for record in records if name in record["values"]]
        if any(not equivalent(present[0], value) for value in present[1:]):
            detail = [
                {"path": record["path"], "value": record["values"].get(name)}
                for record in records
            ]
            raise ValueError(f"physics configuration differs for {name}: {detail}")
        common[name] = present[0]
        if len(present) == len(records):
            checked.append(name)
        else:
            incomplete.append(name)

    overrides = {
        "zenith_deg": args.zenith_deg,
        "azimuth_deg": args.azimuth_deg,
        "shower_core_x_m": args.shower_core_x_m,
        "shower_core_y_m": args.shower_core_y_m,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if name in common and not equivalent(common[name], value):
            raise ValueError(
                f"explicit {name}={value} differs from output value {common[name]}"
            )
        common[name] = float(value)
    for required in ("primary_pdg", "energy_GeV", "zenith_deg", "azimuth_deg"):
        if required not in common:
            raise ValueError(
                f"cannot determine {required}; provide the corresponding explicit option"
            )
    common.setdefault("shower_core_x_m", 0.0)
    common.setdefault("shower_core_y_m", 0.0)
    return common, {
        "passed": True,
        "outputs_checked": len(records),
        "fields_checked_in_every_output": checked,
        "fields_not_explicit_in_every_output": incomplete,
        "per_output": records,
    }


def magnetic_field(
    sources: dict[str, list[Path]], explicit: list[float] | None
) -> tuple[np.ndarray, dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    for role in ROLES:
        for path in sources[role]:
            gpu_path = path / "gpu_em" / "config.yaml"
            if not gpu_path.is_file():
                continue
            config = read_yaml(gpu_path)
            field = config.get("environment", {}).get("magnetic_field_T")
            if not isinstance(field, dict):
                raise ValueError(f"GPU config has no magnetic field: {gpu_path}")
            value = np.asarray([field[axis] for axis in ("x", "y", "z")], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"invalid magnetic field in {gpu_path}")
            observed.append({"role": role, "path": str(gpu_path), "value": value})
    if explicit is not None:
        selected = np.asarray(explicit, dtype=float)
    elif observed:
        selected = observed[0]["value"]
    else:
        raise ValueError(
            "no GPU magnetic field is available; provide --magnetic-field-tesla"
        )
    if not np.isfinite(selected).all() or np.linalg.norm(selected) == 0.0:
        raise ValueError("magnetic field must be finite and non-zero")
    for record in observed:
        if not np.allclose(record["value"], selected, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError(f"magnetic fields differ: {record['path']}")
    return selected, {
        "passed": True,
        "selected_NWU_T": selected.tolist(),
        "gpu_configs_cross_checked": len(observed),
    }


def layout_signature(path: Path, algorithm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = radio_observer_layout(path, algorithm)
    if not layout:
        raise ValueError(f"empty {algorithm} observer layout in {path}")
    locations = np.asarray([item["location_m"] for item in layout], dtype=float)
    bins = np.asarray([item["bins"] for item in layout], dtype=np.int64)
    sampling = np.asarray(
        [item["sampling_frequency_GHz"] for item in layout], dtype=float
    )
    if not np.isfinite(locations).all() or not np.isfinite(sampling).all():
        raise ValueError(f"non-finite {algorithm} observer layout in {path}")
    return locations, bins, sampling


def validate_layouts(sources: dict[str, list[Path]]) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": True, "algorithms": {}}
    for algorithm in ALGORITHMS:
        expected: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        checked = 0
        for role in ROLES:
            for path in sources[role]:
                current = layout_signature(path, algorithm)
                if expected is None:
                    expected = current
                elif not (
                    np.allclose(current[0], expected[0], rtol=0.0, atol=1.0e-6)
                    and np.array_equal(current[1], expected[1])
                    and np.allclose(current[2], expected[2], rtol=1.0e-12, atol=1.0e-12)
                ):
                    raise ValueError(f"{algorithm} observer layout differs in {path}")
                checked += 1
        assert expected is not None
        result["algorithms"][algorithm] = {
            "outputs_checked": checked,
            "observers": int(expected[0].shape[0]),
            "bins_per_shower": int(np.sum(expected[1])),
            "location_tolerance_m": 1.0e-6,
        }
    return result


def validate_waveform_records(
    records: list[dict[str, Any]], algorithm: str, role: str
) -> dict[str, Any]:
    showers = sorted({int(record["shower"]) for record in records})
    observers = sorted({int(record["observer"]) for record in records})
    if showers != list(range(len(showers))):
        raise ValueError(f"{role}/{algorithm} pooled shower IDs are not contiguous")
    expected_pairs = len(showers) * len(observers)
    if len(records) != expected_pairs:
        raise ValueError(
            f"{role}/{algorithm} has {len(records)} shower/observer waveforms; "
            f"expected {expected_pairs}"
        )
    for record in records:
        time = np.asarray(record["time_ns"], dtype=float)
        field = np.asarray(record["field"], dtype=float)
        if time.ndim != 1 or field.shape != (time.size, 3) or time.size < 3:
            raise ValueError(f"malformed waveform in {role}/{algorithm}")
        if not np.isfinite(time).all() or not np.isfinite(field).all():
            raise ValueError(f"non-finite waveform in {role}/{algorithm}")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError(f"non-increasing waveform time in {role}/{algorithm}")
    return {
        "showers": len(showers),
        "observers": len(observers),
        "waveforms": len(records),
        "all_finite": True,
        "strictly_increasing_time": True,
        "complete_shower_observer_product": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values_at(
    rows: list[dict[str, Any]], role: str, algorithm: str, radius: float, column: str
) -> np.ndarray:
    values = np.asarray(
        [
            row[column]
            for row in rows
            if row["backend"] == role
            and row["algorithm"] == algorithm
            and np.isclose(row["radius_m"], radius, rtol=0.0, atol=1.0e-5)
        ],
        dtype=float,
    )
    return values[np.isfinite(values) & (values > 0.0)]


def holm_rejections(p_values: list[float], alpha: float) -> list[bool]:
    result = [False] * len(p_values)
    for rank, (index, value) in enumerate(sorted(enumerate(p_values), key=lambda item: item[1])):
        if value > alpha / (len(p_values) - rank):
            break
        result[index] = True
    return result


def summarize(
    shower_rows: list[dict[str, Any]],
    radii: list[float],
    shower_counts: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for metric, column, _, _ in METRICS:
            for radius in radii:
                arrays = {
                    role: values_at(shower_rows, role, algorithm, radius, column)
                    for role in ROLES
                }
                record: dict[str, Any] = {
                    "algorithm": algorithm,
                    "metric": metric,
                    "r_perp_m": radius,
                }
                for role, values in arrays.items():
                    record[f"{role}_showers"] = int(values.size)
                    record[f"{role}_central"] = (
                        central_statistic(metric, values) if values.size else math.nan
                    )
                    record[f"{role}_q16"] = (
                        float(np.quantile(values, 0.16)) if values.size else math.nan
                    )
                    record[f"{role}_q84"] = (
                        float(np.quantile(values, 0.84)) if values.size else math.nan
                    )
                    record[f"{role}_valid_fraction"] = (
                        float(values.size / shower_counts[role][algorithm])
                        if shower_counts[role][algorithm]
                        else 0.0
                    )
                    record[f"{role}_unique_values_1e-6"] = int(
                        np.unique(np.round(values, 6)).size
                    )
                enough_for_statistics = all(values.size >= 2 for values in arrays.values())
                if enough_for_statistics:
                    ratio, low, high = bootstrap_ratio(
                        arrays["reference"],
                        arrays["candidate"],
                        metric=metric,
                        repetitions=args.bootstrap_repetitions,
                        seed=args.bootstrap_seed + len(rows),
                    )
                    ks = ks_2samp(arrays["reference"], arrays["candidate"], method="auto")
                    record.update(
                        {
                            "candidate_over_reference": ratio,
                            "ratio_ci95_low": low,
                            "ratio_ci95_high": high,
                            "ks_statistic": float(ks.statistic),
                            "ks_p_value": float(ks.pvalue),
                        }
                    )
                else:
                    record.update(
                        {
                            "candidate_over_reference": math.nan,
                            "ratio_ci95_low": math.nan,
                            "ratio_ci95_high": math.nan,
                            "ks_statistic": math.nan,
                            "ks_p_value": math.nan,
                        }
                    )
                record["fitter_floor_diagnostic"] = bool(
                    metric == "width"
                    and min(
                        record["reference_unique_values_1e-6"],
                        record["candidate_unique_values_1e-6"],
                    )
                    <= 2
                )
                count_pass = all(
                    values.size >= args.minimum_count_per_arm for values in arrays.values()
                )
                coverage_pass = metric != "width" or all(
                    record[f"{role}_valid_fraction"]
                    >= args.minimum_width_valid_shower_fraction
                    for role in ROLES
                )
                interval_pass = bool(
                    enough_for_statistics
                    and record["ratio_ci95_low"]
                    >= 1.0 - args.equivalence_relative_tolerance
                    and record["ratio_ci95_high"]
                    <= 1.0 + args.equivalence_relative_tolerance
                )
                record.update(
                    {
                        "count_pass": count_pass,
                        "coverage_pass": coverage_pass,
                        "ratio_interval_pass": interval_pass,
                        "eligible_for_gate": bool(
                            enough_for_statistics
                            and count_pass
                            and coverage_pass
                            and not record["fitter_floor_diagnostic"]
                        ),
                    }
                )
                rows.append(record)

    eligible_indices = [
        index
        for index, row in enumerate(rows)
        if row["eligible_for_gate"] and math.isfinite(row["ks_p_value"])
    ]
    rejected = holm_rejections(
        [rows[index]["ks_p_value"] for index in eligible_indices],
        args.shape_familywise_alpha,
    )
    for row in rows:
        row["holm_shape_rejected"] = None
        row["passed"] = False
    for index, is_rejected in zip(eligible_indices, rejected):
        rows[index]["holm_shape_rejected"] = is_rejected
        rows[index]["passed"] = bool(
            rows[index]["ratio_interval_pass"] and not is_rejected
        )

    curves: dict[str, Any] = {}
    for algorithm in ALGORITHMS:
        curves[algorithm] = {}
        for metric, _, _, _ in METRICS:
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm and row["metric"] == metric
            ]
            eligible = [row for row in selected if row["eligible_for_gate"]]
            curves[algorithm][metric] = {
                "radial_points": len(selected),
                "eligible_points": len(eligible),
                "passed_points": sum(bool(row["passed"]) for row in eligible),
                "minimum_points_required": args.minimum_points_per_curve,
                "passed": bool(
                    len(eligible) >= args.minimum_points_per_curve
                    and all(bool(row["passed"]) for row in eligible)
                ),
            }
    passed = all(
        curves[algorithm][metric]["passed"]
        for algorithm in ALGORITHMS
        for metric, _, _, _ in METRICS
    )
    return rows, {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "equivalence_relative_tolerance": args.equivalence_relative_tolerance,
        "minimum_count_per_arm": args.minimum_count_per_arm,
        "minimum_width_valid_shower_fraction": (
            args.minimum_width_valid_shower_fraction
        ),
        "shape_test": "pointwise two-sample KS with Holm correction",
        "shape_familywise_alpha": args.shape_familywise_alpha,
        "shape_family_size": len(eligible_indices),
        "curves": curves,
    }


def plot_curves(
    rows: list[dict[str, Any]], output: Path, labels: dict[str, str], title: str
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (metric, _, ylabel, logarithmic) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            selected = sorted(
                [row for row in rows if row["algorithm"] == algorithm and row["metric"] == metric],
                key=lambda row: row["r_perp_m"],
            )
            radii = np.asarray([row["r_perp_m"] for row in selected])
            for role in ROLES:
                center = np.asarray([row[f"{role}_central"] for row in selected])
                q16 = np.asarray([row[f"{role}_q16"] for row in selected])
                q84 = np.asarray([row[f"{role}_q84"] for row in selected])
                axis.plot(
                    radii,
                    center,
                    "o-",
                    color=ROLE_COLORS[role],
                    linewidth=1.8,
                    markersize=4,
                    label=labels[role],
                )
                axis.fill_between(
                    radii, q16, q84, color=ROLE_COLORS[role], alpha=0.16, linewidth=0.0
                )
            axis.set_xscale("log")
            if logarithmic:
                axis.set_yscale("log")
            axis.set_xlabel(r"Shower-axis distance $r_\perp$ [m]")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{algorithm}: {metric}")
            axis.grid(alpha=0.24, which="both")
            axis.legend(frameon=False)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_ratios(
    rows: list[dict[str, Any]],
    output: Path,
    labels: dict[str, str],
    tolerance: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (metric, _, _, _) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            selected = sorted(
                [row for row in rows if row["algorithm"] == algorithm and row["metric"] == metric],
                key=lambda row: row["r_perp_m"],
            )
            radii = np.asarray([row["r_perp_m"] for row in selected])
            ratio = np.asarray([row["candidate_over_reference"] for row in selected])
            low = np.asarray([row["ratio_ci95_low"] for row in selected])
            high = np.asarray([row["ratio_ci95_high"] for row in selected])
            axis.axhspan(1.0 - tolerance, 1.0 + tolerance, color="0.75", alpha=0.22)
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
            axis.plot(radii, ratio, "o-", color="#6a1b9a", linewidth=1.8, markersize=4)
            axis.fill_between(radii, low, high, color="#6a1b9a", alpha=0.18)
            failed = np.asarray(
                [
                    row["eligible_for_gate"] and not row["passed"]
                    for row in selected
                ]
            )
            if np.any(failed):
                axis.scatter(
                    radii[failed],
                    ratio[failed],
                    marker="x",
                    color="black",
                    s=55,
                    zorder=5,
                )
            axis.set_xscale("log")
            axis.set_xlabel(r"Shower-axis distance $r_\perp$ [m]")
            axis.set_ylabel(f"{labels['candidate']} / {labels['reference']}")
            axis.set_title(f"{algorithm}: {metric}")
            axis.grid(alpha=0.24, which="both")
    figure.suptitle("Geomagnetic pulse-feature ratios (95% shower bootstrap)", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def markdown(report: dict[str, Any]) -> str:
    labels = report["labels"]
    lines = [
        "# 地磁射电脉冲特征验收",
        "",
        f"- 总状态：**{report['acceptance']['status']}**。",
        f"- Reference：`{labels['reference']}`；candidate：`{labels['candidate']}`。",
        (
            "- 输入完整性：通过；所有 CoREAS/ZHS waveform 均为有限值，"
            "时间严格递增，且 observer layout 与 shower 覆盖完整。"
        ),
        (
            "- 地磁投影、主峰选择及宽度拟合直接复用 "
            "`pulse_analysis_modular`；统计单位是同一 r_perp 下方位天线"
            "聚合后的独立 shower。"
        ),
        (
            f"- 等价区间：candidate/reference 的 95% shower-bootstrap 区间须"
            f"完整落入 ±{100.0 * report['acceptance']['equivalence_relative_tolerance']:.1f}%；"
            "分布形状另用 Holm 校正的逐点 KS 检验。"
        ),
        "",
        "![振幅和宽度径向曲线](geomagnetic_amplitude_width_vs_rperp.png)",
        "",
        "![径向比值](geomagnetic_candidate_reference_ratio_vs_rperp.png)",
        "",
        "## 曲线门禁",
        "",
        "| 算法 | 指标 | 可用点/总点 | 通过点 | 判定 |",
        "|---|---|---:|---:|---|",
    ]
    for algorithm, metrics in report["acceptance"]["curves"].items():
        for metric, result in metrics.items():
            lines.append(
                f"| {algorithm} | {metric} | {result['eligible_points']}/"
                f"{result['radial_points']} | {result['passed_points']} | "
                f"{'PASS' if result['passed'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## 可复现输出",
            "",
            "- `radio_feature_acceptance.json`：输入身份、完整性、配置和全部门禁。",
            "- `radio_feature_comparison.csv`：逐算法、逐指标、逐 r_perp 统计量。",
            "- `per_shower_rperp_features.csv`：实际 shower 级统计样本。",
            "- `per_antenna_radio_features.csv`：脉冲拟合的逐天线明细。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_cli(args: argparse.Namespace) -> None:
    if args.campaign_root is None:
        if not args.reference_output or not args.candidate_output:
            raise ValueError(
                "explicit mode requires both --reference-output and --candidate-output"
            )
    elif args.candidate_output:
        raise ValueError("--candidate-output cannot be combined with --campaign-root")
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    if not (
        math.isfinite(args.minimum_r_perp_m)
        and math.isfinite(args.maximum_r_perp_m)
        and 0.0 <= args.minimum_r_perp_m < args.maximum_r_perp_m
    ):
        raise ValueError("invalid r_perp interval")
    if not 0.0 < args.equivalence_relative_tolerance < 1.0:
        raise ValueError("equivalence tolerance must be in (0, 1)")
    if args.minimum_count_per_arm < 2 or args.minimum_points_per_curve < 1:
        raise ValueError("minimum counts and points are too small")
    if not 0.0 < args.minimum_width_valid_shower_fraction <= 1.0:
        raise ValueError("minimum width-valid fraction must be in (0, 1]")
    if not 0.0 <= args.shape_familywise_alpha <= 1.0:
        raise ValueError("shape familywise alpha must be in [0, 1]")


def main() -> int:
    args = parse_args()
    validate_cli(args)
    campaign_record: dict[str, Any] | None = None
    if args.campaign_root is not None:
        reference, candidate, campaign_record = discover_campaign(
            args.campaign_root, args.allow_incomplete_campaign
        )
        default_labels = {"reference": "c8emrt", "candidate": "proposal-native"}
    else:
        reference = require_outputs(args.reference_output or [], "reference")
        candidate = require_outputs(args.candidate_output or [], "candidate")
        default_labels = {"reference": "CPU PROPOSAL", "candidate": "proposal-native"}
    labels = {
        "reference": args.reference_label or default_labels["reference"],
        "candidate": args.candidate_label or default_labels["candidate"],
    }
    sources = {"reference": reference, "candidate": candidate}
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pulse_root = args.pulse_analysis_root.expanduser().resolve()
    (
        build_polarization_basis,
        analyze_pulse_parameters,
        read_radio_records,
        band_limited_waveform,
        robust_pulse_width_mask,
        PulseWidthFilterConfig,
    ) = load_reference_apis(pulse_root)

    physics, physics_integrity = common_physics_configuration(sources, args)
    field, magnetic_integrity = magnetic_field(sources, args.magnetic_field_tesla)
    layout_integrity = validate_layouts(sources)
    basis = build_polarization_basis(
        float(physics["zenith_deg"]), float(physics["azimuth_deg"]), field
    )
    core = (float(physics["shower_core_x_m"]), float(physics["shower_core_y_m"]))

    antenna_rows: list[dict[str, Any]] = []
    waveform_integrity: dict[str, Any] = {role: {} for role in ROLES}
    shower_counts: dict[str, dict[str, int]] = {role: {} for role in ROLES}
    radii: list[float] | None = None
    for role in ROLES:
        for algorithm in ALGORITHMS:
            records = load_records(
                output_directories=sources[role],
                algorithm=algorithm,
                shower_axis_nwu=np.asarray(basis["n"], dtype=float),
                core_xy_m=core,
                read_radio_records=read_radio_records,
            )
            integrity = validate_waveform_records(records, algorithm, role)
            waveform_integrity[role][algorithm] = integrity
            shower_counts[role][algorithm] = int(integrity["showers"])
            stream_radii = clustered_available_radii(
                records,
                minimum_m=args.minimum_r_perp_m,
                maximum_m=args.maximum_r_perp_m,
            )
            if radii is None:
                radii = stream_radii
            elif stream_radii != radii:
                raise ValueError(
                    f"r_perp grid differs for {role}/{algorithm}: "
                    f"{stream_radii} != {radii}"
                )
            for radius in stream_radii:
                extracted = extract_antenna_rows(
                    records=records,
                    backend=role,
                    algorithm=algorithm,
                    radius_m=radius,
                    yprime=np.asarray(basis["yprime"], dtype=float),
                    analyze_pulse_parameters=analyze_pulse_parameters,
                    band_limited_waveform=band_limited_waveform,
                    band_MHz=None,
                    analysis_sampling_rate_GHz=None,
                )
                add_extracted_coordinate_columns(extracted, records)
                antenna_rows.extend(extracted)
            del records
    if radii is None or len(radii) < 2:
        raise ValueError("fewer than two common non-zero r_perp values are available")
    for role in ROLES:
        if shower_counts[role]["CoREAS"] != shower_counts[role]["ZHS"]:
            raise ValueError(f"CoREAS/ZHS shower counts differ for {role}")

    apply_reference_width_filter(
        antenna_rows,
        robust_pulse_width_mask=robust_pulse_width_mask,
        filter_config=PulseWidthFilterConfig(),
    )
    shower_rows = aggregate_by_shower(antenna_rows)
    add_aggregate_r_perp_column(shower_rows)
    comparisons, acceptance = summarize(shower_rows, radii, shower_counts, args)
    input_integrity = {
        "passed": True,
        "physics_configuration": physics_integrity,
        "magnetic_field": magnetic_integrity,
        "observer_layout": layout_integrity,
        "waveforms": waveform_integrity,
        "algorithms_have_equal_shower_counts_within_arm": True,
    }
    report = {
        "schema_version": 1,
        "status": acceptance["status"],
        "comparison_semantics": (
            "independent shower ensembles; one statistical sample is one shower "
            "after azimuthal aggregation at fixed r_perp"
        ),
        "labels": labels,
        "source_directories": {
            role: [str(path) for path in sources[role]] for role in ROLES
        },
        "campaign": campaign_record,
        "physics_configuration": physics,
        "pulse_analysis_root": str(pulse_root),
        "coordinate": {
            "name": "r_perp_m",
            "definition": "norm(d - dot(d, n) * n)",
            "frame": "local NWU relative to shower core",
            "shower_axis_NWU": np.asarray(basis["n"]).tolist(),
            "geomagnetic_unit_vector_NWU": np.asarray(basis["yprime"]).tolist(),
            "r_perp_m": radii,
        },
        "input_integrity": input_integrity,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "acceptance": acceptance,
        "points": comparisons,
    }
    write_csv(output / "per_antenna_radio_features.csv", antenna_rows)
    write_csv(output / "per_shower_rperp_features.csv", shower_rows)
    write_csv(output / "radio_feature_comparison.csv", comparisons)
    (output / "radio_feature_acceptance.json").write_text(
        json.dumps(strict_json(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    title = (
        f"{labels['reference']} vs {labels['candidate']} geomagnetic pulses\n"
        f"{physics['energy_GeV']:g} GeV PDG {physics['primary_pdg']}, "
        f"zenith {physics['zenith_deg']:g}°, azimuth {physics['azimuth_deg']:g}°"
    )
    plot_curves(
        comparisons,
        output / "geomagnetic_amplitude_width_vs_rperp.png",
        labels,
        title,
    )
    plot_ratios(
        comparisons,
        output / "geomagnetic_candidate_reference_ratio_vs_rperp.png",
        labels,
        args.equivalence_relative_tolerance,
    )
    (output / "RADIO_FEATURE_ACCEPTANCE_CN.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": acceptance["status"],
                "reference_showers": shower_counts["reference"]["CoREAS"],
                "candidate_showers": shower_counts["candidate"]["CoREAS"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 2 if args.fail_on_acceptance and not acceptance["passed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"radio pulse-feature acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
