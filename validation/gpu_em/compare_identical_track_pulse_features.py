#!/usr/bin/env python3
"""Compare pulse-analysis features from identical CPU/CUDA radio tracks.

The two input showers must use the same CUDA transport tracks and differ only
in the radio projection backend.  Waveform agreement is checked separately by
``run_radio_acceptance.py``; this diagnostic deliberately runs the user's
``pulse_analysis_modular`` estimator on both products and pairs every pulse by
algorithm, shower, observer, and shower-axis radius.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .analyze_geomagnetic_pulse_distributions import (
        ALGORITHMS,
        apply_reference_width_filter,
        extract_antenna_rows,
        load_reference_apis,
    )
except ImportError:  # Direct execution from validation/gpu_em.
    from analyze_geomagnetic_pulse_distributions import (  # type: ignore
        ALGORITHMS,
        apply_reference_width_filter,
        extract_antenna_rows,
        load_reference_apis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-radio-output", type=Path, required=True)
    parser.add_argument("--cuda-radio-output", type=Path, required=True)
    parser.add_argument("--pulse-analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zenith-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument(
        "--amplitude-relative-tolerance", type=float, default=2.0e-4
    )
    parser.add_argument(
        "--amplitude-absolute-tolerance-v-per-m",
        type=float,
        default=1.0e-18,
    )
    parser.add_argument(
        "--width-absolute-tolerance-ns", type=float, default=0.11
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def magnetic_field(output: Path) -> np.ndarray:
    path = output / "gpu_em" / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    magnetic = config["environment"]["magnetic_field_T"]
    result = np.asarray(
        [magnetic["x"], magnetic["y"], magnetic["z"]],
        dtype=np.float64,
    )
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid magnetic field in {path}")
    return result


def extract_rows(
    output: Path,
    backend: str,
    yprime: np.ndarray,
    read_radio_records: Any,
    analyze_pulse_parameters: Any,
    band_limited_waveform: Any,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    layouts: dict[str, np.ndarray] = {}
    for algorithm in ALGORITHMS:
        records, locations = read_radio_records(output, algorithm)
        layouts[algorithm] = np.asarray(locations, dtype=np.float64)
        radii = sorted(
            {
                round(float(record["radius_m"]), 6)
                for record in records
            }
        )
        for radius in radii:
            rows.extend(
                extract_antenna_rows(
                    records=records,
                    backend=backend,
                    algorithm=algorithm,
                    radius_m=radius,
                    yprime=yprime,
                    analyze_pulse_parameters=analyze_pulse_parameters,
                    band_limited_waveform=band_limited_waveform,
                    band_MHz=None,
                    analysis_sampling_rate_GHz=None,
                )
            )
    if not rows:
        raise ValueError(f"no radio pulses were extracted from {output}")
    return rows, layouts


def pulse_key(row: dict[str, Any]) -> tuple[str, int, int, float]:
    return (
        str(row["algorithm"]),
        int(row["shower"]),
        int(row["observer"]),
        round(float(row["radius_m"]), 6),
    )


def paired_rows(rows: list[dict[str, Any]]) -> dict[str, dict[Any, dict[str, Any]]]:
    result: dict[str, dict[Any, dict[str, Any]]] = {
        "cpu_radio": {},
        "cuda_radio": {},
    }
    for row in rows:
        backend = str(row["backend"])
        if backend not in result:
            raise ValueError(f"unexpected radio feature backend: {backend}")
        key = pulse_key(row)
        if key in result[backend]:
            raise ValueError(f"duplicate pulse feature key for {backend}: {key}")
        result[backend][key] = row
    if set(result["cpu_radio"]) != set(result["cuda_radio"]):
        missing_cpu = sorted(set(result["cuda_radio"]) - set(result["cpu_radio"]))
        missing_cuda = sorted(set(result["cpu_radio"]) - set(result["cuda_radio"]))
        raise ValueError(
            "CPU/CUDA pulse feature keys differ: "
            f"missing_cpu={missing_cpu[:5]}, missing_cuda={missing_cuda[:5]}"
        )
    return result


def compare_features(
    rows: list[dict[str, Any]],
    amplitude_relative_tolerance: float,
    amplitude_absolute_tolerance: float,
    width_absolute_tolerance_ns: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = paired_rows(rows)
    per_pair: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    overall_pass = True
    for algorithm in ALGORITHMS:
        selected_keys = sorted(
            key for key in pairs["cpu_radio"] if key[0] == algorithm
        )
        if not selected_keys:
            raise ValueError(f"no paired {algorithm} pulse features")
        algorithm_rows: list[dict[str, Any]] = []
        for key in selected_keys:
            cpu = pairs["cpu_radio"][key]
            cuda = pairs["cuda_radio"][key]
            cpu_amplitude = float(cpu["geomagnetic_peak_abs_V_per_m"])
            cuda_amplitude = float(cuda["geomagnetic_peak_abs_V_per_m"])
            amplitude_difference = abs(cuda_amplitude - cpu_amplitude)
            amplitude_scale = max(abs(cpu_amplitude), abs(cuda_amplitude))
            amplitude_relative = (
                amplitude_difference / amplitude_scale
                if amplitude_scale > 0.0
                else 0.0
            )
            amplitude_pass = bool(
                amplitude_difference
                <= amplitude_absolute_tolerance
                + amplitude_relative_tolerance * amplitude_scale
            )
            cpu_valid = bool(cpu["pulse_width_valid"])
            cuda_valid = bool(cuda["pulse_width_valid"])
            validity_match = cpu_valid == cuda_valid
            width_difference: float | None = None
            width_pass = validity_match
            if cpu_valid and cuda_valid:
                width_difference = abs(
                    float(cuda["pulse_width_ns"])
                    - float(cpu["pulse_width_ns"])
                )
                width_pass = bool(
                    width_difference <= width_absolute_tolerance_ns
                )
            filter_match = bool(cpu["pulse_width_filter_pass"]) == bool(
                cuda["pulse_width_filter_pass"]
            )
            method_match = str(cpu["pulse_method"]) == str(cuda["pulse_method"])
            row = {
                "algorithm": algorithm,
                "shower": key[1],
                "observer": key[2],
                "r_perp_m": key[3],
                "cpu_amplitude_V_per_m": cpu_amplitude,
                "cuda_amplitude_V_per_m": cuda_amplitude,
                "amplitude_absolute_difference_V_per_m": amplitude_difference,
                "amplitude_relative_difference": amplitude_relative,
                "amplitude_pass": amplitude_pass,
                "cpu_width_valid": cpu_valid,
                "cuda_width_valid": cuda_valid,
                "width_validity_match": validity_match,
                "cpu_width_ns": float(cpu["pulse_width_ns"]) if cpu_valid else None,
                "cuda_width_ns": float(cuda["pulse_width_ns"]) if cuda_valid else None,
                "width_absolute_difference_ns": width_difference,
                "width_pass": width_pass,
                "width_filter_match": filter_match,
                "pulse_method_match": method_match,
            }
            algorithm_rows.append(row)
            per_pair.append(row)
        summary = {
            "pairs": len(algorithm_rows),
            "maximum_amplitude_relative_difference": max(
                row["amplitude_relative_difference"] for row in algorithm_rows
            ),
            "maximum_amplitude_absolute_difference_V_per_m": max(
                row["amplitude_absolute_difference_V_per_m"]
                for row in algorithm_rows
            ),
            "amplitude_failures": sum(
                not row["amplitude_pass"] for row in algorithm_rows
            ),
            "width_pairs": sum(
                row["cpu_width_valid"] and row["cuda_width_valid"]
                for row in algorithm_rows
            ),
            "maximum_width_absolute_difference_ns": max(
                (
                    row["width_absolute_difference_ns"]
                    for row in algorithm_rows
                    if row["width_absolute_difference_ns"] is not None
                ),
                default=None,
            ),
            "width_failures": sum(
                not row["width_pass"] for row in algorithm_rows
            ),
            "width_validity_mismatches": sum(
                not row["width_validity_match"] for row in algorithm_rows
            ),
            "width_filter_mismatches": sum(
                not row["width_filter_match"] for row in algorithm_rows
            ),
            "pulse_method_mismatches": sum(
                not row["pulse_method_match"] for row in algorithm_rows
            ),
        }
        summary["status"] = (
            "passed"
            if all(
                summary[name] == 0
                for name in (
                    "amplitude_failures",
                    "width_failures",
                    "width_validity_mismatches",
                    "width_filter_mismatches",
                    "pulse_method_mismatches",
                )
            )
            else "failed"
        )
        summaries[algorithm] = summary
        overall_pass &= summary["status"] == "passed"
    return {
        "status": "passed" if overall_pass else "failed",
        "algorithms": summaries,
        "tolerances": {
            "amplitude_relative": amplitude_relative_tolerance,
            "amplitude_absolute_V_per_m": amplitude_absolute_tolerance,
            "width_absolute_ns": width_absolute_tolerance_ns,
        },
    }, per_pair


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Identical-track pulse-analysis acceptance",
        "",
        "CPU-radio and CUDA-radio waveforms were produced from the same deterministic CUDA transport tracks. The geomagnetic amplitude and pulse width below are extracted with the user's `pulse_analysis_modular` implementation.",
        "",
        "| Algorithm | Pairs | Max amplitude rel. diff. | Width pairs | Max width abs. diff. [ns] | Validity/filter/method mismatches | Status |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        item = report["algorithms"][algorithm]
        mismatch = (
            item["width_validity_mismatches"]
            + item["width_filter_mismatches"]
            + item["pulse_method_mismatches"]
        )
        width = item["maximum_width_absolute_difference_ns"]
        width_text = "n/a" if width is None else f"{width:.6g}"
        lines.append(
            f"| {algorithm} | {item['pairs']} | "
            f"{item['maximum_amplitude_relative_difference']:.3e} | "
            f"{item['width_pairs']} | "
            f"{width_text} | {mismatch} | {item['status']} |"
        )
    lines.extend(("", f"Overall status: **{report['status']}**", ""))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    for value, label in (
        (args.amplitude_relative_tolerance, "amplitude relative tolerance"),
        (args.amplitude_absolute_tolerance_v_per_m, "amplitude absolute tolerance"),
        (args.width_absolute_tolerance_ns, "width absolute tolerance"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    if not args.cpu_radio_output.is_dir() or not args.cuda_radio_output.is_dir():
        raise FileNotFoundError("CPU/CUDA radio output is absent")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)

    cpu_field = magnetic_field(args.cpu_radio_output)
    cuda_field = magnetic_field(args.cuda_radio_output)
    if not np.allclose(cpu_field, cuda_field, rtol=0.0, atol=1.0e-15):
        raise ValueError("CPU/CUDA identical-track magnetic fields differ")
    (
        build_polarization_basis,
        analyze_pulse_parameters,
        read_radio_records,
        band_limited_waveform,
        robust_pulse_width_mask,
        PulseWidthFilterConfig,
    ) = load_reference_apis(args.pulse_analysis_root.resolve())
    basis = build_polarization_basis(
        args.zenith_deg, args.azimuth_deg, cpu_field
    )
    all_rows: list[dict[str, Any]] = []
    layouts: dict[str, dict[str, np.ndarray]] = {}
    for backend, output in (
        ("cpu_radio", args.cpu_radio_output),
        ("cuda_radio", args.cuda_radio_output),
    ):
        rows, backend_layouts = extract_rows(
            output,
            backend,
            np.asarray(basis["yprime"], dtype=np.float64),
            read_radio_records,
            analyze_pulse_parameters,
            band_limited_waveform,
        )
        all_rows.extend(rows)
        layouts[backend] = backend_layouts
    for algorithm in ALGORITHMS:
        if not np.allclose(
            layouts["cpu_radio"][algorithm],
            layouts["cuda_radio"][algorithm],
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(f"{algorithm} identical-track observer layouts differ")
    apply_reference_width_filter(
        all_rows,
        robust_pulse_width_mask=robust_pulse_width_mask,
        filter_config=PulseWidthFilterConfig(),
    )
    report, rows = compare_features(
        all_rows,
        args.amplitude_relative_tolerance,
        args.amplitude_absolute_tolerance_v_per_m,
        args.width_absolute_tolerance_ns,
    )
    report["inputs"] = {
        "cpu_radio_output": str(args.cpu_radio_output.resolve()),
        "cuda_radio_output": str(args.cuda_radio_output.resolve()),
        "pulse_analysis_root": str(args.pulse_analysis_root.resolve()),
        "cpu_CoREAS_config_sha256": sha256(
            args.cpu_radio_output / "CoREAS" / "config.yaml"
        ),
        "cuda_CoREAS_config_sha256": sha256(
            args.cuda_radio_output / "CoREAS" / "config.yaml"
        ),
        "cpu_ZHS_config_sha256": sha256(
            args.cpu_radio_output / "ZHS" / "config.yaml"
        ),
        "cuda_ZHS_config_sha256": sha256(
            args.cuda_radio_output / "ZHS" / "config.yaml"
        ),
    }
    (args.output / "pulse_feature_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "per_pulse_comparison.csv", rows)
    (args.output / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["algorithms"], indent=2))
    return 0 if report["status"] == "passed" or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
