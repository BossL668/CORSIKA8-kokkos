#!/usr/bin/env python3
"""Audit whether two independently sampled backend builds may be pooled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from compare_ensembles import (
    DEFAULT_KEY_SCALAR_METRICS,
    compare_ensembles,
    concatenate_ensembles,
    extract_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--backend", choices=("proposal", "cuda"), default="cuda"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--sigma-limit", type=float, default=3.0)
    parser.add_argument("--active-fraction", type=float, default=1.0e-4)
    parser.add_argument("--minimum-bin-pass-fraction", type=float, default=0.95)
    return parser.parse_args()


def write_markdown(
    path: Path,
    report: dict[str, Any],
    compatibility: dict[str, Any],
) -> None:
    lines = [
        "# Backend build-stratum compatibility audit",
        "",
        f"- Reference sources: {len(compatibility['reference_sources'])}",
        f"- Candidate sources: {len(compatibility['candidate_sources'])}",
        f"- Backend: `{compatibility['backend']}`",
        (
            "- Executable SHA-256: `"
            f"{compatibility['reference_executable_sha256']}` vs `"
            f"{compatibility['candidate_executable_sha256']}`"
        ),
        (
            f"- Shared rate-table SHA-256: `{compatibility['table_sha256']}`"
            if compatibility["table_sha256"] is not None
            else "- Rate table: not applicable to scalar PROPOSAL output."
        ),
        (
            "- Pooling diagnostic: **"
            + ("passed" if compatibility["passed"] else "failed")
            + "**."
        ),
        "",
        (
            "The pooling diagnostic requires exact physics configuration and "
            "table identity, every key scalar to remain within the configured "
            "sigma limit, every key-scalar empirical KS distance below its "
            "95% critical value, and the longitudinal/ground curve gate to pass."
        ),
        "",
        "| Metric | Candidate/reference mean shift | |z| | KS below 95% critical |",
        "|---|---:|---:|---|",
    ]
    for metric in DEFAULT_KEY_SCALAR_METRICS:
        result = report["scalars"][metric]
        signed = (
            result["difference"] / result["proposal"]["mean"]
            if result["proposal"]["mean"] != 0.0
            else None
        )
        lines.append(
            f"| {metric} | "
            + (f"{100.0 * signed:+.3f}%" if signed is not None else "n/a")
            + f" | {result['absolute_z_score']:.3f} | "
            + (
                "yes"
                if result["distribution_diagnostics"][
                    "KS_below_95pct_critical_value"
                ]
                else "no"
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- Key-scalar sigma gate: {compatibility['key_scalar_sigma_pass']}",
            f"- Key-scalar KS gate: {compatibility['key_scalar_ks_pass']}",
            f"- Curve gate: {compatibility['curve_pass']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    reference_paths = [path.resolve() for path in args.reference]
    candidate_paths = [path.resolve() for path in args.candidate]
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    expect_gpu = args.backend == "cuda"
    reference = concatenate_ensembles(
        f"{args.backend}_build_reference",
        [
            extract_ensemble(
                f"{args.backend}_build_reference",
                path,
                expect_gpu,
            )
            for path in reference_paths
        ],
    )
    candidate = concatenate_ensembles(
        f"{args.backend}_build_candidate",
        [
            extract_ensemble(
                f"{args.backend}_build_candidate",
                path,
                expect_gpu,
            )
            for path in candidate_paths
        ],
    )
    if (
        reference.metadata["physics_configuration"]
        != candidate.metadata["physics_configuration"]
    ):
        raise ValueError("CUDA build strata have different physics configurations")
    reference_provenance = reference.metadata["provenance_fingerprint"]
    candidate_provenance = candidate.metadata["provenance_fingerprint"]
    if reference_provenance[3] != candidate_provenance[3]:
        raise ValueError("backend build strata use different rate tables")
    if expect_gpu and reference_provenance[3] is None:
        raise ValueError("CUDA build strata do not record a rate table")
    report, curve_rows = compare_ensembles(
        reference,
        candidate,
        args.relative_tolerance,
        args.sigma_limit,
        args.active_fraction,
        args.minimum_bin_pass_fraction,
        DEFAULT_KEY_SCALAR_METRICS,
        allow_cross_build_reference=True,
    )
    key_results = [
        report["scalars"][metric]
        for metric in DEFAULT_KEY_SCALAR_METRICS
    ]
    key_sigma_pass = all(bool(item["statistical_pass"]) for item in key_results)
    key_ks_pass = all(
        bool(
            item["distribution_diagnostics"][
                "KS_below_95pct_critical_value"
            ]
        )
        for item in key_results
    )
    curve_pass = bool(report["acceptance"]["curve_pass"])
    compatibility = {
        "reference_sources": [str(path) for path in reference_paths],
        "candidate_sources": [str(path) for path in candidate_paths],
        "backend": args.backend,
        "reference_events": len(reference.showers),
        "candidate_events": len(candidate.showers),
        "reference_executable_sha256": reference_provenance[2],
        "candidate_executable_sha256": candidate_provenance[2],
        "table_sha256": reference_provenance[3],
        "physics_configuration_equal": True,
        "key_scalar_sigma_pass": key_sigma_pass,
        "key_scalar_ks_pass": key_ks_pass,
        "curve_pass": curve_pass,
        "passed": key_sigma_pass and key_ks_pass and curve_pass,
    }
    output.mkdir(parents=True)
    pd.concat(
        [
            reference.scalars.assign(build="reference"),
            candidate.scalars.assign(build="candidate"),
        ],
        ignore_index=True,
    ).to_csv(output / "per_shower_observables.csv", index=False)
    curve_rows.to_csv(output / "curve_comparison.csv", index=False)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "compatibility.json").write_text(
        json.dumps(compatibility, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output / "README.md", report, compatibility)
    print(json.dumps(compatibility, indent=2))
    return 0 if compatibility["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
