#!/usr/bin/env python3
"""Compare muon-production parent profiles at equal shower age.

The fixed-depth ensemble mean remains an important observable, but its apparent
peak can move by one or more 10 g/cm^2 bins when two independent ensembles have
slightly different shower-development distributions.  This module therefore
adds a complementary comparison: every production profile is interpolated on
``Delta X = X - Xmax_charged`` before the ensemble mean is formed.  The raw
per-shower parent-peak distributions and integrals are reported separately so
that alignment cannot hide a genuine displacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


BACKENDS = ("proposal", "cuda")
COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {
    "proposal": "Original CPU (PROPOSAL)",
    "cuda": "CUDA EM",
}
PARENT_COLUMNS = (
    ("pion", "pion parent"),
    ("kaon", "kaon parent"),
    ("heavy", "heavy-hadron parent"),
    ("hadron", "all hadron parents"),
    ("photon", "photon parent"),
    ("electron-positron", r"$e^\pm$ parent"),
    ("muon", "muon parent"),
    ("all", "all parent species"),
)


def quadratic_peak_coordinate(
    coordinate: np.ndarray, values: np.ndarray
) -> float:
    """Return a stable three-point peak coordinate on a regular grid."""

    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3:
        raise ValueError("peak input must be two equal one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("peak input contains a non-finite value")
    index = int(np.argmax(y))
    if index == 0 or index + 1 == x.size:
        return float(x[index])
    denominator = y[index - 1] - 2.0 * y[index] + y[index + 1]
    if denominator == 0.0:
        return float(x[index])
    spacing = 0.5 * (x[index + 1] - x[index - 1])
    offset = 0.5 * (y[index - 1] - y[index + 1]) / denominator
    offset = float(np.clip(offset, -1.0, 1.0))
    return float(x[index] + offset * spacing)


def _read_events(
    roots: list[Path], backend: str
) -> dict[str, Any]:
    coordinate: np.ndarray | None = None
    anchors: list[float] = []
    profiles: dict[str, list[np.ndarray]] = {
        column: [] for column, _ in PARENT_COLUMNS
    }
    source_events: list[dict[str, Any]] = []
    for root in roots:
        production_path = root / "production_profile" / "profile.parquet"
        longitudinal_path = root / "profile" / "profile.parquet"
        if not production_path.is_file() or not longitudinal_path.is_file():
            raise ValueError(f"missing parent-profile source below {root}")
        production = pd.read_parquet(
            production_path,
            columns=["shower", "X", *(column for column, _ in PARENT_COLUMNS)],
        )
        longitudinal = pd.read_parquet(
            longitudinal_path, columns=["shower", "X", "charged"]
        )
        production_showers = sorted(int(value) for value in production["shower"].unique())
        longitudinal_showers = sorted(
            int(value) for value in longitudinal["shower"].unique()
        )
        if production_showers != longitudinal_showers:
            raise ValueError(f"profile shower IDs differ below {root}")
        for shower in production_showers:
            parent = production.loc[production["shower"] == shower].sort_values("X")
            charged = longitudinal.loc[
                longitudinal["shower"] == shower
            ].sort_values("X")
            current_coordinate = parent["X"].to_numpy(dtype=np.float64)
            if coordinate is None:
                coordinate = current_coordinate
            elif not np.array_equal(coordinate, current_coordinate):
                raise ValueError(f"parent-profile grid differs below {root}")
            charged_coordinate = charged["X"].to_numpy(dtype=np.float64)
            charged_values = charged["charged"].to_numpy(dtype=np.float64)
            anchor = quadratic_peak_coordinate(charged_coordinate, charged_values)
            anchors.append(anchor)
            for column, _ in PARENT_COLUMNS:
                values = parent[column].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all() or np.any(values < 0.0):
                    raise ValueError(
                        f"invalid {backend} {column} parent profile below {root}"
                    )
                profiles[column].append(values)
        source_events.append({"root": str(root), "events": len(production_showers)})
    if coordinate is None or not anchors:
        raise ValueError(f"no {backend} parent profiles were found")
    return {
        "coordinate": coordinate,
        "anchors": np.asarray(anchors, dtype=np.float64),
        "profiles": {
            column: np.vstack(rows) for column, rows in profiles.items()
        },
        "sources": source_events,
    }


def load_parent_profile_events(
    manifest: dict[str, Any], expected_events: int
) -> dict[str, dict[str, Any]]:
    sources = manifest.get("additional_sources")
    if not isinstance(sources, dict):
        raise ValueError("manifest has no immutable additional_sources")
    result: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        values = sources.get(backend)
        if not isinstance(values, list) or not values:
            raise ValueError(f"manifest has no {backend} source list")
        result[backend] = _read_events(
            [Path(str(value)).resolve() for value in values], backend
        )
        observed = int(result[backend]["anchors"].size)
        if observed != expected_events:
            raise ValueError(
                f"{backend} parent-profile count differs: "
                f"expected {expected_events}, observed {observed}"
            )
    if not np.array_equal(
        result["proposal"]["coordinate"], result["cuda"]["coordinate"]
    ):
        raise ValueError("CPU and CUDA parent-profile grids differ")
    return result


def _aligned_moments(
    coordinate: np.ndarray,
    matrix: np.ndarray,
    anchors: np.ndarray,
    aligned_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned = np.vstack(
        [
            np.interp(
                aligned_coordinate,
                coordinate - anchor,
                row,
                left=np.nan,
                right=np.nan,
            )
            for row, anchor in zip(matrix, anchors)
        ]
    )
    finite = np.isfinite(aligned)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.nansum(aligned, axis=0),
        count,
        out=np.full(aligned_coordinate.shape, np.nan),
        where=count > 0,
    )
    squared = np.nansum((aligned - mean) ** 2, axis=0)
    standard_error = np.divide(
        np.sqrt(np.divide(squared, count - 1, out=np.zeros_like(mean), where=count > 1)),
        np.sqrt(count),
        out=np.full_like(mean, np.nan),
        where=count > 1,
    )
    return mean, standard_error, count.astype(np.int64)


def _bootstrap_mean_difference(
    proposal: np.ndarray,
    cuda: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> list[float] | None:
    if proposal.size < 2 or cuda.size < 2:
        return None
    proposal_indices = rng.integers(
        0, proposal.size, size=(repetitions, proposal.size)
    )
    cuda_indices = rng.integers(0, cuda.size, size=(repetitions, cuda.size))
    differences = cuda[cuda_indices].mean(axis=1) - proposal[
        proposal_indices
    ].mean(axis=1)
    return [float(value) for value in np.quantile(differences, (0.025, 0.975))]


def analyze_parent_alignment(
    manifest_path: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = 5000,
    minimum_coverage_fraction: float = 0.90,
) -> dict[str, Any]:
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.0 < minimum_coverage_fraction <= 1.0:
        raise ValueError("coverage fraction must be in (0, 1]")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("manifest has no configuration")
    expected_events = int(configuration["events_per_backend"])
    events = load_parent_profile_events(manifest, expected_events)
    coordinate = events["proposal"]["coordinate"]
    spacing = float(np.median(np.diff(coordinate)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("invalid production-profile coordinate spacing")
    extent = float(np.ceil(np.max(np.abs(coordinate)) / spacing) * spacing)
    aligned_coordinate = np.arange(-extent, extent + 0.5 * spacing, spacing)

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    rng = np.random.default_rng(20260804)
    required_count = int(np.ceil(minimum_coverage_fraction * expected_events))
    aligned_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        backend: {} for backend in BACKENDS
    }
    for column, _ in PARENT_COLUMNS:
        for backend in BACKENDS:
            aligned_cache[backend][column] = _aligned_moments(
                coordinate,
                events[backend]["profiles"][column],
                events[backend]["anchors"],
                aligned_coordinate,
            )
        proposal_mean, proposal_sem, proposal_count = aligned_cache["proposal"][column]
        cuda_mean, cuda_sem, cuda_count = aligned_cache["cuda"][column]
        for index, delta_x in enumerate(aligned_coordinate):
            rows.append(
                {
                    "parent": column,
                    "delta_X_g_per_cm2": float(delta_x),
                    "proposal_mean": float(proposal_mean[index]),
                    "proposal_standard_error": float(proposal_sem[index]),
                    "proposal_count": int(proposal_count[index]),
                    "cuda_mean": float(cuda_mean[index]),
                    "cuda_standard_error": float(cuda_sem[index]),
                    "cuda_count": int(cuda_count[index]),
                }
            )

        raw: dict[str, dict[str, np.ndarray]] = {}
        for backend in BACKENDS:
            matrix = events[backend]["profiles"][column]
            present = np.max(matrix, axis=1) > 0.0
            raw[backend] = {
                "peak": np.asarray(
                    [
                        quadratic_peak_coordinate(coordinate, row)
                        for row in matrix[present]
                    ],
                    dtype=np.float64,
                ),
                "integral": np.trapz(matrix, coordinate, axis=1),
            }
        proposal_peak = raw["proposal"]["peak"]
        cuda_peak = raw["cuda"]["peak"]
        proposal_integral = raw["proposal"]["integral"]
        cuda_integral = raw["cuda"]["integral"]
        peak_test = (
            stats.ttest_ind(proposal_peak, cuda_peak, equal_var=False)
            if proposal_peak.size >= 2 and cuda_peak.size >= 2
            else None
        )
        peak_ks = (
            stats.ks_2samp(proposal_peak, cuda_peak, method="exact")
            if proposal_peak.size >= 2 and cuda_peak.size >= 2
            else None
        )
        common = (
            (proposal_count >= required_count)
            & (cuda_count >= required_count)
            & np.isfinite(proposal_mean)
            & np.isfinite(cuda_mean)
        )
        active = common & (
            proposal_mean
            > 1.0e-4 * max(float(np.nanmax(proposal_mean)), np.finfo(float).tiny)
        )
        denominator = float(np.sum(np.abs(proposal_mean[active])))
        summaries[column] = {
            "events": {backend: expected_events for backend in BACKENDS},
            "nonzero_peak_events": {
                "proposal": int(proposal_peak.size),
                "cuda": int(cuda_peak.size),
            },
            "raw_peak_depth_g_per_cm2": {
                "proposal_mean": (
                    float(np.mean(proposal_peak)) if proposal_peak.size else None
                ),
                "cuda_mean": float(np.mean(cuda_peak)) if cuda_peak.size else None,
                "cuda_minus_proposal": (
                    float(np.mean(cuda_peak) - np.mean(proposal_peak))
                    if proposal_peak.size and cuda_peak.size
                    else None
                ),
                "bootstrap_difference_95pct": _bootstrap_mean_difference(
                    proposal_peak, cuda_peak, bootstrap_repetitions, rng
                ),
                "welch_pvalue": float(peak_test.pvalue) if peak_test else None,
                "ks_pvalue": float(peak_ks.pvalue) if peak_ks else None,
            },
            "raw_integral": {
                "proposal_mean": float(np.mean(proposal_integral)),
                "cuda_mean": float(np.mean(cuda_integral)),
                "cuda_over_proposal": (
                    float(np.mean(cuda_integral) / np.mean(proposal_integral))
                    if np.mean(proposal_integral) > 0.0
                    else None
                ),
            },
            "xmax_aligned_shape": {
                "minimum_coverage_fraction": minimum_coverage_fraction,
                "common_bins": int(np.sum(common)),
                "active_bins": int(np.sum(active)),
                "relative_L1": (
                    float(
                        np.sum(np.abs(cuda_mean[active] - proposal_mean[active]))
                        / denominator
                    )
                    if denominator > 0.0
                    else None
                ),
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    curve_frame = pd.DataFrame(rows)
    curve_frame.to_csv(
        output_dir / "muon_production_parent_xmax_aligned_curves.csv", index=False
    )
    report = {
        "schema_version": 1,
        "alignment": "per-shower charged-profile quadratic Xmax",
        "coordinate": "Delta X = X - Xmax_charged",
        "interpretation": (
            "Use the aligned curves to compare parent-profile shape at equal "
            "shower age. Use the raw peak-depth and integral distributions to "
            "test absolute-location and normalization differences."
        ),
        "configuration": configuration,
        "bootstrap_repetitions": bootstrap_repetitions,
        "ratio_plot_reference_floor_fraction": 1.0e-2,
        "sources": {
            backend: events[backend]["sources"] for backend in BACKENDS
        },
        "parents": summaries,
    }
    (output_dir / "muon_production_parent_alignment_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot_alignment(
        curve_frame, output_dir, minimum_coverage_fraction
    )
    return report


def _plot_alignment(
    curves: pd.DataFrame,
    output_dir: Path,
    minimum_coverage_fraction: float,
) -> None:
    figure, axes = plt.subplots(
        4,
        4,
        figsize=(20.4, 12.4),
        squeeze=False,
        gridspec_kw={"height_ratios": (2.2, 1.0, 2.2, 1.0)},
    )
    for panel, (column, label) in enumerate(PARENT_COLUMNS):
        block, panel_column = divmod(panel, 4)
        upper = axes[2 * block, panel_column]
        lower = axes[2 * block + 1, panel_column]
        selected = curves.loc[curves["parent"] == column].sort_values(
            "delta_X_g_per_cm2"
        )
        expected = max(
            int(selected["proposal_count"].max()),
            int(selected["cuda_count"].max()),
        )
        valid = (
            (
                selected["proposal_count"].to_numpy()
                >= int(np.ceil(minimum_coverage_fraction * expected))
            )
            & (
                selected["cuda_count"].to_numpy()
                >= int(np.ceil(minimum_coverage_fraction * expected))
            )
        )
        selected = selected.loc[valid]
        x = selected["delta_X_g_per_cm2"].to_numpy(dtype=np.float64)
        proposal = selected["proposal_mean"].to_numpy(dtype=np.float64)
        cuda = selected["cuda_mean"].to_numpy(dtype=np.float64)
        proposal_sem = selected["proposal_standard_error"].to_numpy(dtype=np.float64)
        cuda_sem = selected["cuda_standard_error"].to_numpy(dtype=np.float64)
        active = (proposal > 0.0) | (cuda > 0.0)
        if not np.any(active):
            upper.text(0.5, 0.5, "no resolved entries", ha="center", va="center")
            lower.set_visible(False)
            upper.set_title(label)
            continue
        first, last = np.flatnonzero(active)[[0, -1]]
        first = max(0, first - 2)
        last = min(x.size - 1, last + 2)
        selection = slice(first, last + 1)
        x = x[selection]
        proposal = proposal[selection]
        cuda = cuda[selection]
        proposal_sem = proposal_sem[selection]
        cuda_sem = cuda_sem[selection]
        upper.plot(x, proposal, color=COLORS["proposal"], linewidth=1.8, label=LABELS["proposal"])
        upper.plot(x, cuda, color=COLORS["cuda"], linestyle="--", linewidth=1.6, label=LABELS["cuda"])
        upper.axvline(0.0, color="0.45", linewidth=0.9, linestyle=":")
        upper.set_ylabel(label)
        upper.grid(alpha=0.2)
        relative = np.full_like(proposal, np.nan)
        relative_error = np.full_like(proposal, np.nan)
        # A percentage residual is undefined or visually unbounded once the
        # CPU reference has entered its numerical tail.  Keep the complete
        # absolute curves above, but only draw the residual where the CPU
        # signal exceeds 1% of its peak.  This display-only mask is stricter
        # than the L1 metric's 1e-4 active-bin floor; no samples are removed
        # from the absolute curves or the reported statistics.
        reference_floor = 1.0e-2 * max(
            float(np.nanmax(proposal)), np.finfo(float).tiny
        )
        ratio_active = proposal > reference_floor
        relative[ratio_active] = (
            cuda[ratio_active] - proposal[ratio_active]
        ) / proposal[ratio_active]
        relative_error[ratio_active] = np.sqrt(
            proposal_sem[ratio_active] ** 2 + cuda_sem[ratio_active] ** 2
        ) / np.abs(proposal[ratio_active])
        lower.axhline(0.0, color="0.35", linewidth=1.0)
        lower.axvline(0.0, color="0.45", linewidth=0.9, linestyle=":")
        lower.plot(x, 100.0 * relative, color=COLORS["cuda"], linewidth=1.2)
        lower.fill_between(
            x,
            100.0 * (relative - relative_error),
            100.0 * (relative + relative_error),
            color=COLORS["cuda"],
            alpha=0.18,
            linewidth=0.0,
        )
        lower.set_xlabel(r"$\Delta X=X-X_{\max}^{\rm charged}$ [g cm$^{-2}$]")
        lower.set_ylabel(r"$\Delta/\mathrm{CPU}$ [%]")
        lower.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Muon-production parent profiles aligned shower by shower\n"
        r"(shape at equal shower age; band is $\pm1$ combined standard error)"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "muon_production_parent_xmax_aligned_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--minimum-coverage-fraction", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_parent_alignment(
        args.manifest.resolve(),
        args.output_dir.resolve(),
        bootstrap_repetitions=args.bootstrap_repetitions,
        minimum_coverage_fraction=args.minimum_coverage_fraction,
    )
    print(
        json.dumps(
            {
                "events_per_backend": report["configuration"]["events_per_backend"],
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
