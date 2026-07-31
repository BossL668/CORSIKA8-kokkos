#!/usr/bin/env python3
"""Create publication-style diagnostic plots for original CPU versus CUDA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_cpu_cuda_radio import (  # noqa: E402
    ALGORITHMS,
    BANDS,
    aggregate_radial_features,
    radiation_proxy_distributions,
    read_radio_records,
)
from compare_cuda_replay import (  # noqa: E402
    RADIAL_GROUP_DECIMALS,
    energy_observables,
    radio_fluence_maps,
    radio_observer_layout,
)


COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {"proposal": "Original CPU (PROPOSAL)", "cuda": "CUDA EM"}


def finish_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_scalar_distributions(
    electron_root: Path, proton_root: Path, output: Path
) -> None:
    metrics = (
        ("energy_deposit_sum_GeV", r"$E_{\rm dep}$ [GeV]"),
        ("profile_charged_integral", "charged-profile integral"),
        ("profile_photon_integral", "photon-profile integral"),
        ("profile_xmax_charged_gcm2", r"$X_{\max}^{\rm charged}$ [g cm$^{-2}$]"),
    )
    figure, axes = plt.subplots(2, len(metrics), figsize=(15, 7))
    for row, (root, primary) in enumerate(
        ((electron_root, "1 TeV electron"), (proton_root, "1 TeV proton"))
    ):
        frame = pd.read_csv(root / "per_shower_observables.csv")
        for column, (metric, label) in enumerate(metrics):
            axis = axes[row, column]
            arrays = {
                backend: frame.loc[
                    frame["backend"] == backend, metric
                ].to_numpy(dtype=np.float64)
                for backend in ("proposal", "cuda")
            }
            support = np.concatenate(tuple(arrays.values()))
            bins = np.histogram_bin_edges(support, bins="fd")
            if bins.size < 8:
                bins = np.linspace(float(np.min(support)), float(np.max(support)), 16)
            for backend, values in arrays.items():
                axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.7,
                    color=COLORS[backend],
                    label=LABELS[backend],
                )
            axis.set_xlabel(label)
            axis.set_ylabel("probability density")
            axis.set_title(primary)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    finish_figure(figure, output)


def plot_longitudinal_curves(root: Path, title: str, output: Path) -> None:
    frame = pd.read_csv(root / "curve_comparison.csv")
    observables = (
        ("energy_deposit", r"$dE/dX$"),
        ("profile_charged", r"$N_{e^-}+N_{e^+}$"),
        ("profile_photon", r"$N_\gamma$"),
        ("profile_electron", r"$N_{e^-}$"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (observable, label) in zip(axes.flat, observables):
        selected = frame[
            (frame["family"] == "longitudinal")
            & (frame["observable"] == observable)
            & frame["active"]
        ].sort_values("coordinate")
        for backend, column in (
            ("proposal", "proposal_mean"),
            ("cuda", "cuda_mean"),
        ):
            axis.plot(
                selected["coordinate"],
                selected[column],
                color=COLORS[backend],
                linewidth=1.7,
                label=LABELS[backend],
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"slant depth [g cm$^{-2}$]")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(title)
    finish_figure(figure, output)


def plot_radiation_distributions(
    radio_root: Path,
    output: Path,
    *,
    normalize_by_em_deposit: bool,
) -> None:
    reference = radio_root / "legacy_proposal"
    candidate = radio_root / "cuda"
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, algorithm, band in zip(
        axes.flat,
        (algorithm for algorithm in ALGORITHMS for _ in BANDS),
        (band for _ in ALGORITHMS for band in BANDS),
    ):
        locations = np.asarray(
            [
                observer["location_m"]
                for observer in radio_observer_layout(reference, algorithm)
            ]
        )
        distributions = {
            "proposal": np.asarray(
                list(
                    radiation_proxy_distributions(
                        reference,
                        algorithm,
                        band,
                        locations,
                        normalize_by_em_deposit=normalize_by_em_deposit,
                    ).values()
                )
            ),
            "cuda": np.asarray(
                list(
                    radiation_proxy_distributions(
                        candidate,
                        algorithm,
                        band,
                        locations,
                        normalize_by_em_deposit=normalize_by_em_deposit,
                    ).values()
                )
            ),
        }
        support = np.concatenate(tuple(distributions.values()))
        bins = np.histogram_bin_edges(support, bins="fd")
        for backend, values in distributions.items():
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.7,
                color=COLORS[backend],
                label=LABELS[backend],
            )
        axis.set_title(f"{algorithm}, {band[0]:g}–{band[1]:g} MHz")
        axis.set_xlabel(
            r"$E_{\rm rad}/E_{\rm EM}^2$ proxy"
            if normalize_by_em_deposit
            else r"raw $E_{\rm rad}$ proxy"
        )
        axis.set_ylabel("probability density")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    finish_figure(figure, output)


def plot_aligned_templates(radio_root: Path, output: Path) -> None:
    reference = radio_root / "legacy_proposal"
    candidate = radio_root / "cuda"
    selected_radii = (0.0, 100.0, 300.0, 600.0)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, algorithm, band in zip(
        axes.flat,
        (algorithm for algorithm in ALGORITHMS for _ in BANDS),
        (band for _ in ALGORITHMS for band in BANDS),
    ):
        templates = {}
        records_for_dt = None
        for backend, root in (("proposal", reference), ("cuda", candidate)):
            records, _ = read_radio_records(root, algorithm)
            if records_for_dt is None:
                records_for_dt = records
            _, by_radius, _ = aggregate_radial_features(
                records, energy_observables(root), band
            )
            templates[backend] = {
                radius: np.mean(np.stack(values), axis=0)
                for radius, values in by_radius.items()
            }
        assert records_for_dt is not None
        dt_ns = float(
            np.median(np.diff(records_for_dt[0]["time_ns"]))
        )
        palette = plt.get_cmap("viridis")
        available = np.asarray(sorted(templates["proposal"]))
        for radius_index, requested in enumerate(selected_radii):
            radius = float(available[np.argmin(np.abs(available - requested))])
            color = palette(radius_index / max(len(selected_radii) - 1, 1))
            for backend, linestyle in (("proposal", "-"), ("cuda", "--")):
                values = templates[backend][radius]
                time = (
                    np.arange(values.size, dtype=np.float64) - values.size // 2
                ) * dt_ns
                axis.plot(
                    time,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.25,
                    label=(
                        f"{radius:g} m, {LABELS[backend]}"
                        if backend == "proposal"
                        else f"{radius:g} m, CUDA"
                    ),
                )
        axis.set_xlim(-100.0, 100.0)
        axis.set_title(f"{algorithm}, {band[0]:g}–{band[1]:g} MHz")
        axis.set_xlabel("peak-aligned time [ns]")
        axis.set_ylabel("unit-energy pulse power")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    finish_figure(figure, output)


def plot_radial_fluence(radio_root: Path, output: Path) -> None:
    reference = radio_root / "legacy_proposal"
    candidate = radio_root / "cuda"
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, algorithm, band in zip(
        axes.flat,
        (algorithm for algorithm in ALGORITHMS for _ in BANDS),
        (band for _ in ALGORITHMS for band in BANDS),
    ):
        for backend, root in (("proposal", reference), ("cuda", candidate)):
            layout = radio_observer_layout(root, algorithm)
            locations = np.asarray(
                [observer["location_m"] for observer in layout]
            )
            radii = np.round(
                np.linalg.norm(
                    locations[:, :2] - locations[0, :2], axis=1
                ),
                decimals=RADIAL_GROUP_DECIMALS,
            )
            maps = radio_fluence_maps(root, algorithm, [band])[band]
            per_shower = []
            unique_radii = np.unique(radii)
            for shower in sorted(maps):
                total = maps[shower][:, 3] / 1000.0**2
                per_shower.append(
                    [
                        float(np.mean(total[radii == radius]))
                        for radius in unique_radii
                    ]
                )
            values = np.asarray(per_shower)
            mean = np.mean(values, axis=0)
            standard_error = (
                np.std(values, axis=0, ddof=1)
                / math.sqrt(values.shape[0])
                if values.shape[0] > 1
                else np.zeros_like(mean)
            )
            axis.plot(
                unique_radii,
                mean,
                color=COLORS[backend],
                linewidth=1.7,
                label=LABELS[backend],
            )
            axis.fill_between(
                unique_radii,
                np.maximum(mean - standard_error, 0.0),
                mean + standard_error,
                color=COLORS[backend],
                alpha=0.15,
                linewidth=0.0,
            )
        axis.set_yscale("log")
        axis.set_title(f"{algorithm}, {band[0]:g}–{band[1]:g} MHz")
        axis.set_xlabel("axis distance [m]")
        axis.set_ylabel(r"mean fluence / $(1\,{\rm TeV})^2$")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    finish_figure(figure, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--electron-root", type=Path, required=True)
    parser.add_argument("--proton-root", type=Path, required=True)
    parser.add_argument("--radio-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plot_scalar_distributions(
        args.electron_root,
        args.proton_root,
        args.output_directory / "shower_scalar_distributions.png",
    )
    plot_longitudinal_curves(
        args.electron_root,
        "Original CPU versus CUDA: 1 TeV electron",
        args.output_directory / "electron_longitudinal_profiles.png",
    )
    plot_longitudinal_curves(
        args.proton_root,
        "Original CPU versus CUDA: 1 TeV proton",
        args.output_directory / "proton_longitudinal_profiles.png",
    )
    plot_radiation_distributions(
        args.radio_root,
        args.output_directory / "radio_energy_distributions.png",
        normalize_by_em_deposit=True,
    )
    plot_radiation_distributions(
        args.radio_root,
        args.output_directory / "radio_energy_distributions_raw.png",
        normalize_by_em_deposit=False,
    )
    plot_aligned_templates(
        args.radio_root,
        args.output_directory / "radio_aligned_pulse_templates.png",
    )
    plot_radial_fluence(
        args.radio_root,
        args.output_directory / "radio_radial_fluence_profiles.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
