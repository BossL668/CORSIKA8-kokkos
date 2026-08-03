#!/usr/bin/env python3
"""Create a diagnostic scalar-PROPOSAL/CUDA comparison for one shower each.

One shower per backend is useful for checking output completeness, gross
physics mistakes, curve shapes and end-to-end timing.  It is not an ensemble
test and must not be interpreted as evidence of statistical equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from analyze_cpu_cuda_radio import json_compatible
from compare_ensembles import Ensemble, extract_ensemble


PROFILE_GROUPS = (
    (
        "Electromagnetic",
        ("profile_photon", "profile_electron", "profile_positron"),
    ),
    ("Hadronic", ("profile_hadron",)),
    ("Muonic", ("profile_muplus", "profile_muminus")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    return parser.parse_args()


def only_event(ensemble: Ensemble) -> None:
    if ensemble.showers != (0,) or len(ensemble.scalars.index) != 1:
        raise ValueError(
            f"{ensemble.name} must contain exactly shower_0; "
            f"found {ensemble.showers}"
        )


def relative_difference(reference: float, candidate: float) -> float | None:
    if not (math.isfinite(reference) and math.isfinite(candidate)):
        return None
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - reference) / abs(reference)


def scalar_comparison(reference: Ensemble, candidate: Ensemble) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in reference.scalars.columns:
        if name not in candidate.scalars.columns:
            continue
        scalar = float(reference.scalars.iloc[0][name])
        cuda = float(candidate.scalars.iloc[0][name])
        result[name] = {
            "proposal": scalar,
            "cuda": cuda,
            "signed_relative_difference": relative_difference(scalar, cuda),
        }
    return result


def curve_comparison(reference: Ensemble, candidate: Ensemble) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(reference.curves) & set(candidate.curves)):
        x_reference, matrix_reference = reference.curves[name]
        x_candidate, matrix_candidate = candidate.curves[name]
        if not np.array_equal(x_reference, x_candidate):
            raise ValueError(f"different depth grids for {name}")
        scalar = np.asarray(matrix_reference[0], dtype=np.float64)
        cuda = np.asarray(matrix_candidate[0], dtype=np.float64)
        difference = cuda - scalar
        l1_scale = max(
            float(np.sum(np.abs(scalar))),
            float(np.sum(np.abs(cuda))),
            np.finfo(np.float64).tiny,
        )
        l2_scale = max(
            float(np.linalg.norm(scalar)),
            float(np.linalg.norm(cuda)),
            np.finfo(np.float64).tiny,
        )
        denominator = float(np.linalg.norm(scalar) * np.linalg.norm(cuda))
        result[name] = {
            "bins": int(scalar.size),
            "proposal_integral_on_output_grid": float(np.sum(scalar)),
            "cuda_integral_on_output_grid": float(np.sum(cuda)),
            "signed_integral_difference": relative_difference(
                float(np.sum(scalar)), float(np.sum(cuda))
            ),
            "relative_L1": float(np.sum(np.abs(difference)) / l1_scale),
            "relative_L2": float(np.linalg.norm(difference) / l2_scale),
            "cosine_similarity": (
                float(np.dot(scalar, cuda) / denominator)
                if denominator > 0.0
                else 1.0
            ),
        }
    return result


def timing(root: Path) -> float:
    with (root / "simulation_timing" / "summary.yaml").open(
        "r", encoding="utf-8"
    ) as source:
        value = yaml.safe_load(source)
    record = value.get("shower_0", {}) if isinstance(value, dict) else {}
    if record.get("closed") is not True or record.get("status") != "closed":
        raise ValueError(f"incomplete timing record in {root}")
    return float(record["wall_time_ms"]) / 1000.0


def plot_profiles(
    reference: Ensemble, candidate: Ensemble, destination: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    colors = {
        "profile_photon": "#d95f02",
        "profile_electron": "#1b9e77",
        "profile_positron": "#7570b3",
        "profile_hadron": "#e7298a",
        "profile_muplus": "#66a61e",
        "profile_muminus": "#e6ab02",
    }
    labels = {
        "profile_photon": r"$\gamma$",
        "profile_electron": r"$e^-$",
        "profile_positron": r"$e^+$",
        "profile_hadron": "hadrons",
        "profile_muplus": r"$\mu^+$",
        "profile_muminus": r"$\mu^-$",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True)
    active_depth = 0.0
    for axis, (title, names) in zip(axes.flat[:3], PROFILE_GROUPS):
        for name in names:
            x_reference, y_reference = reference.curves[name]
            x_candidate, y_candidate = candidate.curves[name]
            for x, values in (
                (x_reference, y_reference[0]),
                (x_candidate, y_candidate[0]),
            ):
                active = np.flatnonzero(values > 0.0)
                if active.size:
                    active_depth = max(active_depth, float(x[active[-1]]))
            axis.plot(
                x_reference,
                np.maximum(y_reference[0], 1.0e-12),
                color=colors[name],
                linestyle="-",
                label=f"PROPOSAL {labels[name]}",
            )
            axis.plot(
                x_candidate,
                np.maximum(y_candidate[0], 1.0e-12),
                color=colors[name],
                linestyle="--",
                label=f"CUDA {labels[name]}",
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("weighted particle count")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)

    axis = axes.flat[3]
    for ensemble, linestyle, label in (
        (reference, "-", "PROPOSAL"),
        (candidate, "--", "CUDA"),
    ):
        x, values = ensemble.curves["energy_deposit"]
        active = np.flatnonzero(values[0] > 0.0)
        if active.size:
            active_depth = max(active_depth, float(x[active[-1]]))
        axis.plot(x, values[0], linestyle=linestyle, label=label)
    axis.set_title("Energy deposition")
    axis.set_ylabel(r"$dE/dX$ [GeV per bin]")
    axis.grid(alpha=0.25)
    axis.legend()
    for axis in axes[-1, :]:
        axis.set_xlabel(r"slant depth $X$ [g cm$^{-2}$]")
    if active_depth > 0.0:
        axes.flat[0].set_xlim(0.0, active_depth + 50.0)
    figure.suptitle(
        "Single-event diagnostic (independent showers; not a statistical test)"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    reference = extract_ensemble(
        "proposal", args.proposal.resolve(), False, allow_legacy_provenance=True
    )
    candidate = extract_ensemble(
        "cuda", args.cuda.resolve(), True, allow_legacy_provenance=True
    )
    only_event(reference)
    only_event(candidate)
    proposal_seconds = timing(args.proposal.resolve())
    cuda_seconds = timing(args.cuda.resolve())
    report = {
        "semantics": (
            "one independent shower per backend; diagnostic comparison only, "
            "not evidence of statistical equivalence"
        ),
        "proposal": str(args.proposal.resolve()),
        "cuda": str(args.cuda.resolve()),
        "timing": {
            "proposal_seconds": proposal_seconds,
            "cuda_seconds": cuda_seconds,
            "observed_turnaround_ratio": proposal_seconds / cuda_seconds,
        },
        "scalars": scalar_comparison(reference, candidate),
        "curves": curve_comparison(reference, candidate),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as destination:
        json.dump(json_compatible(report), destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_profiles(reference, candidate, args.plot)
    print(
        json.dumps(
            {
                "status": "diagnostic_complete",
                "report": str(args.report),
                "plot": str(args.plot),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
