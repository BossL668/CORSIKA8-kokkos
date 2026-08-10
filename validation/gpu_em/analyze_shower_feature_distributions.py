#!/usr/bin/env python3
"""Summarize and plot original-CPU versus CUDA shower-feature distributions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

try:
    from .analyze_muon_parent_alignment import analyze_parent_alignment
except ImportError:  # Direct execution from validation/gpu_em.
    _PARENT_MODULE_PATH = (
        Path(__file__).resolve().parent
        / "analyze_muon_parent_alignment.py"
    )
    _PARENT_SPEC = importlib.util.spec_from_file_location(
        "analyze_muon_parent_alignment", _PARENT_MODULE_PATH
    )
    if _PARENT_SPEC is None or _PARENT_SPEC.loader is None:
        raise ImportError(
            f"cannot load parent-profile analyzer: {_PARENT_MODULE_PATH}"
        )
    _PARENT_MODULE = importlib.util.module_from_spec(_PARENT_SPEC)
    _PARENT_SPEC.loader.exec_module(_PARENT_MODULE)
    analyze_parent_alignment = _PARENT_MODULE.analyze_parent_alignment


BACKENDS = ("proposal", "cuda")
COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {
    "proposal": "Original CPU (PROPOSAL)",
    "cuda": "CUDA EM",
}
FEATURES = (
    (
        "profile_xmax_charged_gcm2",
        r"$X_{\max}^{\rm charged}$ [g cm$^{-2}$]",
        False,
    ),
    (
        "profile_charged_max",
        r"$N_{\rm charged}(X_{\max})$",
        False,
    ),
    (
        "profile_charged_integral",
        r"$\int N_{\rm charged}\,dX$ [g cm$^{-2}$]",
        False,
    ),
    (
        "profile_photon_integral",
        r"$\int N_\gamma\,dX$ [g cm$^{-2}$]",
        False,
    ),
    (
        "ground_em_weighted_count",
        "ground EM weighted particle count",
        True,
    ),
    (
        "ground_em_kinetic_energy_GeV",
        "ground EM kinetic energy [GeV]",
        True,
    ),
    (
        "energy_deposit_xmax_gcm2",
        r"$X_{\max}^{dE/dX}$ [g cm$^{-2}$]",
        False,
    ),
    (
        "energy_deposit_sum_GeV",
        r"total deposited energy [GeV]",
        False,
    ),
)

REFERENCE_SCALAR_FEATURES = (
    ("energy_deposit_sum_GeV", r"$E_{\rm dep}$ [GeV]"),
    ("profile_charged_integral", "charged-profile integral"),
    ("profile_photon_integral", "photon-profile integral"),
    (
        "profile_xmax_charged_gcm2",
        r"$X_{\max}^{\rm charged}$ [g cm$^{-2}$]",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        required=True,
        help="run_physics_acceptance output containing CSV and JSON results",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional external run manifest for a pooled ensemble. If "
            "omitted, run_manifest.json is searched beside the comparison."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def finite_values(
    frame: pd.DataFrame, backend: str, metric: str
) -> np.ndarray:
    values = frame.loc[
        frame["backend"] == backend, metric
    ].to_numpy(dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError(
            f"{backend} {metric} must contain at least two finite values"
        )
    return values


def feature_summary(
    frame: pd.DataFrame,
    comparison: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    result = comparison["scalars"][metric]
    proposal_values = finite_values(frame, "proposal", metric)
    cuda_values = finite_values(frame, "cuda", metric)
    ks_test = stats.ks_2samp(
        proposal_values, cuda_values, method="exact"
    )
    welch_test = stats.ttest_ind(
        proposal_values, cuda_values, equal_var=False
    )
    brown_forsythe = stats.levene(
        proposal_values, cuda_values, center="median"
    )
    summary: dict[str, Any] = {
        "metric": metric,
        "events_proposal": int(result["proposal"]["events"]),
        "events_cuda": int(result["cuda"]["events"]),
        "proposal_mean": float(result["proposal"]["mean"]),
        "proposal_standard_deviation": float(
            result["proposal"]["standard_deviation"]
        ),
        "cuda_mean": float(result["cuda"]["mean"]),
        "cuda_standard_deviation": float(
            result["cuda"]["standard_deviation"]
        ),
        "signed_relative_mean_shift": (
            float(result["difference"])
            / max(
                abs(float(result["proposal"]["mean"])),
                np.finfo(np.float64).tiny,
            )
        ),
        "absolute_z_score": float(result["absolute_z_score"]),
        "empirical_KS_distance": float(
            result["distribution_diagnostics"]["empirical_KS_distance"]
        ),
        "KS_95pct_critical_value": float(
            result["distribution_diagnostics"][
                "KS_95pct_critical_value"
            ]
        ),
        "KS_below_95pct_critical_value": bool(
            result["distribution_diagnostics"][
                "KS_below_95pct_critical_value"
            ]
        ),
        "KS_exact_pvalue": float(ks_test.pvalue),
        "welch_mean_pvalue": float(welch_test.pvalue),
        "brown_forsythe_scale_pvalue": float(
            brown_forsythe.pvalue
        ),
        "variance_ratio_cuda_over_proposal": float(
            result["distribution_diagnostics"][
                "variance_ratio_cuda_over_proposal"
            ]
        ),
        "bootstrap_signed_relative_mean_shift_95pct": [
            float(value)
            for value in result["distribution_diagnostics"][
                "bootstrap_signed_relative_mean_shift_95pct"
            ]
        ],
        "one_percent_gate": bool(result["relative_pass"]),
        "three_sigma_gate": bool(result["statistical_pass"]),
        "comparison_outcome": result["outcome"],
    }
    for backend in BACKENDS:
        values = (
            proposal_values
            if backend == "proposal"
            else cuda_values
        )
        summary[f"{backend}_median"] = float(np.median(values))
        summary[f"{backend}_q16"] = float(np.quantile(values, 0.16))
        summary[f"{backend}_q84"] = float(np.quantile(values, 0.84))
    return summary


def common_bins(arrays: dict[str, np.ndarray], log_x: bool) -> np.ndarray:
    support = np.concatenate(tuple(arrays.values()))
    if log_x:
        positive = support[support > 0.0]
        if positive.size != support.size:
            raise ValueError("logarithmic feature contains non-positive values")
        return np.geomspace(
            float(np.min(positive)),
            float(np.max(positive)),
            31,
        )
    bins = np.histogram_bin_edges(support, bins="fd")
    if bins.size < 12:
        bins = np.linspace(float(np.min(support)), float(np.max(support)), 21)
    return bins


def histogram_normalization(
    values: np.ndarray, log_x: bool
) -> tuple[bool, np.ndarray | None]:
    """Return a visually honest normalization for linear or log bins.

    ``density=True`` divides counts by the *linear* bin width.  Applied to
    geometrically spaced bins on a logarithmic x axis, that convention makes
    the narrow low-value bins appear disproportionately tall.  Ground-EM
    observables therefore use equal-width log bins with one-event probability
    weights, while linear observables retain the conventional density.
    """

    if not log_x:
        return True, None
    if values.size == 0:
        raise ValueError("cannot normalize an empty logarithmic histogram")
    return False, np.full(values.shape, 1.0 / values.size, dtype=np.float64)


def plot_feature_distributions(
    frame: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for axis, (metric, label, log_x) in zip(axes.flat, FEATURES):
        arrays = {
            backend: finite_values(frame, backend, metric)
            for backend in BACKENDS
        }
        bins = common_bins(arrays, log_x)
        for backend in BACKENDS:
            density, weights = histogram_normalization(
                arrays[backend], log_x
            )
            axis.hist(
                arrays[backend],
                bins=bins,
                density=density,
                weights=weights,
                histtype="step",
                linewidth=1.8,
                color=COLORS[backend],
                label=LABELS[backend],
            )
        if log_x:
            axis.set_xscale("log")
        result = summaries[metric]
        ks_relation = (
            "<"
            if result["KS_below_95pct_critical_value"]
            else r"\geq"
        )
        axis.text(
            0.03,
            0.96,
            (
                rf"$\Delta\mu/\mu={100.0 * result['signed_relative_mean_shift']:+.2f}\%$"
                "\n"
                rf"$D_{{KS}}={result['empirical_KS_distance']:.3f}"
                rf"{ks_relation}{result['KS_95pct_critical_value']:.3f}$"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.85,
            },
        )
        axis.set_xlabel(label)
        axis.set_ylabel(
            "probability per logarithmic bin"
            if log_x
            else "probability density"
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(f"Original CPU versus CUDA EM: {title}")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_reference_scalar_distributions(
    frame: pd.DataFrame,
    output: Path,
    primary_title: str,
    events_per_backend: int,
) -> None:
    """Reproduce the four-panel scalar layout used by validation_plots_v1."""
    figure, axes = plt.subplots(
        1,
        len(REFERENCE_SCALAR_FEATURES),
        figsize=(16, 4.2),
    )
    for axis, (metric, label) in zip(
        axes, REFERENCE_SCALAR_FEATURES
    ):
        arrays = {
            backend: finite_values(frame, backend, metric)
            for backend in BACKENDS
        }
        bins = common_bins(arrays, log_x=False)
        for backend in BACKENDS:
            axis.hist(
                arrays[backend],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=COLORS[backend],
                label=LABELS[backend],
            )
        axis.set_xlabel(label)
        axis.set_ylabel("probability density")
        axis.set_title(primary_title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Original CPU versus CUDA EM — "
        f"{events_per_backend} independent showers per backend",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def format_energy(energy_gev: float) -> str:
    for scale, unit in (
        (1.0e9, "EeV"),
        (1.0e6, "PeV"),
        (1.0e3, "TeV"),
    ):
        value = energy_gev / scale
        if value >= 1.0:
            return f"{value:g} {unit}"
    return f"{energy_gev:g} GeV"


def primary_name(pdg: int) -> str:
    return {
        11: "electron",
        -11: "positron",
        22: "photon",
        2212: "proton",
    }.get(pdg, f"PDG {pdg}")


def primary_label(configuration: dict[str, Any]) -> str:
    """Return a human-readable elementary-particle or nuclear-primary label."""
    primary_z = configuration.get("primary_Z")
    primary_a = configuration.get("primary_A")
    if primary_z is not None or primary_a is not None:
        if primary_z is None or primary_a is None:
            raise ValueError(
                "nuclear-primary metadata must provide both primary_Z "
                "and primary_A"
            )
        z = int(primary_z)
        a = int(primary_a)
        if z == 26 and a == 56:
            return r"$^{56}$Fe"
        return rf"$^{{{a}}}Z_{{{z}}}$ nucleus"
    primary_pdg = configuration.get("primary_pdg")
    if primary_pdg is None:
        raise ValueError(
            "run configuration contains neither primary_pdg nor a complete "
            "primary_Z/primary_A pair"
        )
    return primary_name(int(primary_pdg))


def plot_longitudinal_grid(
    curves: pd.DataFrame,
    selected_features: tuple[tuple[str, str], ...],
    output: Path,
    title: str,
    *,
    columns: int = 3,
) -> None:
    available = set(
        curves.loc[curves["family"] == "longitudinal", "observable"]
    )
    selected_features = tuple(
        feature
        for feature in selected_features
        if feature[0] in available
    )
    if not selected_features:
        return
    columns = min(columns, len(selected_features))
    rows = (len(selected_features) + columns - 1) // columns
    height_ratios = tuple(
        ratio
        for _ in range(rows)
        for ratio in (2.2, 1.0)
    )
    figure, axes = plt.subplots(
        2 * rows,
        columns,
        figsize=(5.1 * columns, 6.2 * rows),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    active_coordinates = []
    for observable, _ in selected_features:
        selected = curves.loc[
            (curves["family"] == "longitudinal")
            & (curves["observable"] == observable)
            & curves["active"]
        ]
        if not selected.empty:
            active_coordinates.append(
                selected["coordinate"].to_numpy(dtype=np.float64)
            )
    if active_coordinates:
        support = np.concatenate(active_coordinates)
        support_low = max(0.0, float(np.min(support)))
        support_high = float(np.max(support))
        margin = max(
            10.0,
            0.04 * max(support_high - support_low, 1.0),
        )
        x_limits = (
            max(0.0, support_low - margin),
            support_high + margin,
        )
    else:
        x_limits = None
    for panel, (observable, label) in enumerate(selected_features):
        panel_row, panel_column = divmod(panel, columns)
        selected = curves.loc[
            (curves["family"] == "longitudinal")
            & (curves["observable"] == observable)
        ].sort_values("coordinate")
        selected = selected.loc[selected["active"]]
        coordinate = selected["coordinate"].to_numpy(dtype=np.float64)
        proposal = selected["proposal_mean"].to_numpy(dtype=np.float64)
        cuda = selected["cuda_mean"].to_numpy(dtype=np.float64)
        combined_error = selected[
            "combined_standard_error"
        ].to_numpy(dtype=np.float64)
        has_backend_errors = {
            "proposal_standard_error",
            "cuda_standard_error",
            "proposal_events",
            "cuda_events",
        }.issubset(selected.columns)
        if has_backend_errors:
            proposal_error = selected[
                "proposal_standard_error"
            ].to_numpy(dtype=np.float64)
            cuda_error = selected[
                "cuda_standard_error"
            ].to_numpy(dtype=np.float64)
            proposal_events = selected[
                "proposal_events"
            ].to_numpy(dtype=np.int64)
            cuda_events = selected[
                "cuda_events"
            ].to_numpy(dtype=np.int64)
            proposal_half_width = stats.t.ppf(
                0.975, np.maximum(proposal_events - 1, 1)
            ) * proposal_error
            cuda_half_width = stats.t.ppf(
                0.975, np.maximum(cuda_events - 1, 1)
            ) * cuda_error
            mean_variance_sum = (
                proposal_error * proposal_error
                + cuda_error * cuda_error
            )
            welch_denominator = (
                np.divide(
                    proposal_error**4,
                    np.maximum(proposal_events - 1, 1),
                )
                + np.divide(
                    cuda_error**4,
                    np.maximum(cuda_events - 1, 1),
                )
            )
            welch_degrees = np.full_like(
                mean_variance_sum, np.inf
            )
            nonzero_welch = welch_denominator > 0.0
            welch_degrees[nonzero_welch] = (
                mean_variance_sum[nonzero_welch] ** 2
                / welch_denominator[nonzero_welch]
            )
            difference_half_width = stats.t.ppf(
                0.975, welch_degrees
            ) * combined_error
        else:
            proposal_half_width = None
            cuda_half_width = None
            difference_half_width = (
                stats.norm.ppf(0.975) * combined_error
            )
        upper = axes[2 * panel_row, panel_column]
        if proposal_half_width is not None:
            upper.fill_between(
                coordinate,
                np.maximum(proposal - proposal_half_width, 0.0),
                proposal + proposal_half_width,
                color=COLORS["proposal"],
                alpha=0.18,
                linewidth=0.0,
            )
            upper.fill_between(
                coordinate,
                np.maximum(cuda - cuda_half_width, 0.0),
                cuda + cuda_half_width,
                color=COLORS["cuda"],
                alpha=0.16,
                linewidth=0.0,
            )
        upper.plot(
            coordinate,
            proposal,
            color=COLORS["proposal"],
            linewidth=1.8,
            label=LABELS["proposal"],
        )
        upper.plot(
            coordinate,
            cuda,
            color=COLORS["cuda"],
            linewidth=1.6,
            linestyle="--",
            label=LABELS["cuda"],
        )
        upper.set_ylabel(label)
        if x_limits is not None:
            upper.set_xlim(*x_limits)
        upper.grid(alpha=0.2)

        lower = axes[2 * panel_row + 1, panel_column]
        relative = np.full_like(proposal, np.nan)
        uncertainty = np.full_like(proposal, np.nan)
        nonzero = proposal != 0.0
        relative[nonzero] = (
            cuda[nonzero] - proposal[nonzero]
        ) / proposal[nonzero]
        uncertainty[nonzero] = (
            difference_half_width[nonzero]
            / np.abs(proposal[nonzero])
        )
        lower.axhline(0.0, color="0.35", linewidth=1.0)
        lower.plot(
            coordinate,
            100.0 * relative,
            color=COLORS["cuda"],
            linewidth=1.2,
        )
        lower.fill_between(
            coordinate,
            100.0 * (relative - uncertainty),
            100.0 * (relative + uncertainty),
            color=COLORS["cuda"],
            alpha=0.18,
            linewidth=0.0,
        )
        lower.set_xlabel(r"slant depth [g cm$^{-2}$]")
        lower.set_ylabel(r"$\Delta/\mathrm{CPU}$ [%]")
        if x_limits is not None:
            lower.set_xlim(*x_limits)
        lower.grid(alpha=0.2)
    for panel in range(len(selected_features), rows * columns):
        panel_row, panel_column = divmod(panel, columns)
        axes[2 * panel_row, panel_column].set_visible(False)
        axes[2 * panel_row + 1, panel_column].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        title
        + "\n(shading: pointwise 95% confidence interval of each mean; "
        + "relative panel: Welch 95% interval)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_longitudinal_means(curves: pd.DataFrame, output: Path) -> None:
    plot_longitudinal_grid(
        curves,
        (
            (
                "profile_electron_positron",
                r"$N_{e^-}+N_{e^+}$",
            ),
            ("profile_photon", r"$N_\gamma$"),
            (
                "profile_em",
                r"$N_\gamma+N_{e^-}+N_{e^+}$",
            ),
            ("energy_deposit", r"$dE/dX$ [GeV per bin]"),
        ),
        output,
        "Ensemble-mean electromagnetic shower development",
        columns=4,
    )


def plot_non_em_longitudinal_means(
    curves: pd.DataFrame,
    output: Path,
) -> None:
    plot_longitudinal_grid(
        curves,
        (
            ("profile_hadron", r"$N_{\rm hadron}$"),
            ("profile_muon", r"$N_{\mu^-}+N_{\mu^+}$"),
            ("profile_muplus", r"$N_{\mu^+}$"),
            ("profile_muminus", r"$N_{\mu^-}$"),
            (
                "profile_charged",
                r"$N_{\rm charged}$ (all charged species)",
            ),
        ),
        output,
        "Ensemble-mean non-EM and all-charged longitudinal profiles",
        columns=3,
    )


def plot_muon_production_parent_means(
    curves: pd.DataFrame,
    output: Path,
) -> None:
    plot_longitudinal_grid(
        curves,
        (
            ("muon_production_parent_pion", "pion parent"),
            ("muon_production_parent_kaon", "kaon parent"),
            ("muon_production_parent_heavy", "heavy-hadron parent"),
            ("muon_production_parent_hadron", "all hadron parents"),
            ("muon_production_parent_photon", "photon parent"),
            (
                "muon_production_parent_electron_positron",
                r"$e^\pm$ parent",
            ),
            ("muon_production_parent_muon", "muon parent"),
            ("muon_production_parent_all", "all parent species"),
        ),
        output,
        "Muon-production vertex profiles grouped by parent species",
        columns=4,
    )


def main() -> int:
    args = parse_args()
    root = args.ensemble_root.resolve()
    output = args.output_dir.resolve()
    per_shower_path = root / "per_shower_observables.csv"
    comparison_path = root / "comparison.json"
    curves_path = root / "curve_comparison.csv"
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else root / "run_manifest.json"
    )
    if args.manifest is None and not manifest_path.is_file():
        manifest_path = root.parent / "run_manifest.json"
    for path in (
        per_shower_path,
        comparison_path,
        curves_path,
        manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"missing ensemble artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(per_shower_path)
    with comparison_path.open(encoding="utf-8") as source:
        comparison = json.load(source)
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    curves = pd.read_csv(curves_path)
    configuration = manifest["configuration"]
    proposal_events = int(
        (frame["backend"] == "proposal").sum()
    )
    cuda_events = int((frame["backend"] == "cuda").sum())
    if proposal_events != cuda_events:
        raise ValueError(
            "the scalar-distribution plot requires equal CPU and CUDA "
            f"sample sizes, got {proposal_events} and {cuda_events}"
        )
    primary_title = (
        f"{format_energy(float(configuration['energy_GeV']))} "
        f"{primary_label(configuration)}"
    )
    ensemble_title = (
        f"{proposal_events} independent {primary_title} showers per backend"
    )

    summaries = {
        metric: feature_summary(frame, comparison, metric)
        for metric, _, _ in FEATURES
    }
    summary_rows = []
    for metric, label, _ in FEATURES:
        row = dict(summaries[metric])
        row["label"] = label
        low, high = row.pop(
            "bootstrap_signed_relative_mean_shift_95pct"
        )
        row["bootstrap_relative_shift_95pct_low"] = low
        row["bootstrap_relative_shift_95pct_high"] = high
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        output / "selected_feature_summary.csv", index=False
    )
    payload = {
        "interpretation": (
            "Two independent random ensembles are compared at the shower "
            "level. Agreement is assessed with mean shifts, bootstrap "
            "confidence intervals, variance ratios and empirical two-sample "
            "KS distances; individual longitudinal bins or particles are "
            "not treated as independent events."
        ),
        "source": str(root),
        "configuration": configuration,
        "features": summaries,
    }
    with (
        output / "selected_feature_summary.json"
    ).open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, allow_nan=False)
        destination.write("\n")
    plot_feature_distributions(
        frame,
        summaries,
        output / "shower_feature_distributions.png",
        ensemble_title,
    )
    plot_reference_scalar_distributions(
        frame,
        output / "shower_scalar_distributions.png",
        primary_title,
        proposal_events,
    )
    plot_longitudinal_means(
        curves,
        output / "longitudinal_mean_comparison.png",
    )
    plot_non_em_longitudinal_means(
        curves,
        output / "longitudinal_non_em_mean_comparison.png",
    )
    plot_muon_production_parent_means(
        curves,
        output / "muon_production_parent_mean_comparison.png",
    )
    if isinstance(manifest.get("additional_sources"), dict):
        analyze_parent_alignment(
            manifest_path,
            output,
        )
    print(
        json.dumps(
            {
                "events_per_backend": int(
                    proposal_events
                ),
                "features": len(summaries),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
