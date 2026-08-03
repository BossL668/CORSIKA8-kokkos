#!/usr/bin/env python3
"""Plot longitudinal-particle and radio profiles for one CORSIKA 8 shower."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml


COLORS = {
    "CoREAS": "#1769aa",
    "ZHS": "#d95f02",
    "photon": "#6a3d9a",
    "electron": "#1f78b4",
    "positron": "#e31a1c",
    "em": "#009e73",
    "charged": "#222222",
    "hadron": "#b15928",
    "muon": "#cc79a7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="single-shower result directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="plot destination (default: RESULT/plots)",
    )
    parser.add_argument(
        "--maximum-depth",
        type=float,
        default=1600.0,
        help="maximum slant depth shown in g/cm^2",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_parquet(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
    }


def finish(figure: plt.Figure, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def plot_particle_profile(
    result: Path, output: Path, maximum_depth: float
) -> None:
    profile = read_parquet(result / "profile" / "profile.parquet")
    loss = read_parquet(result / "energyloss" / "dEdX.parquet")
    loss_summary = read_yaml(result / "energyloss" / "summary.yaml")[
        "shower_0"
    ]
    production_summary = read_yaml(
        result / "production_profile" / "summary.yaml"
    )["shower_0"]
    interaction_summary = read_yaml(
        result / "interactions" / "summary.yaml"
    )["shower_0"]

    x = profile["X"].astype(float)
    mask = x <= maximum_depth
    x = x[mask]
    electron = profile["electron"][mask]
    positron = profile["positron"][mask]
    photon = profile["photon"][mask]
    muon = profile["muplus"][mask] + profile["muminus"][mask]

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True)

    axes[0].plot(
        x,
        positive(photon),
        label=r"$\gamma$",
        color=COLORS["photon"],
        linewidth=1.7,
    )
    axes[0].plot(
        x,
        positive(electron),
        label=r"$e^-$",
        color=COLORS["electron"],
        linewidth=1.5,
    )
    axes[0].plot(
        x,
        positive(positron),
        label=r"$e^+$",
        color=COLORS["positron"],
        linewidth=1.5,
    )
    axes[0].plot(
        x,
        positive(electron + positron),
        label=r"$e^-+e^+$",
        color=COLORS["em"],
        linewidth=2.0,
    )
    axes[0].set_title("Electromagnetic components")
    axes[0].set_ylabel("weighted particle count")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(
        x,
        positive(profile["charged"][mask]),
        label="all charged",
        color=COLORS["charged"],
        linewidth=1.8,
    )
    axes[1].plot(
        x,
        positive(profile["hadron"][mask]),
        label="hadrons",
        color=COLORS["hadron"],
        linewidth=1.7,
    )
    axes[1].plot(
        x,
        positive(muon),
        label=r"$\mu^-+\mu^+$",
        color=COLORS["muon"],
        linewidth=1.7,
    )
    axes[1].set_title("Charged and non-EM components")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=9)

    loss_x = loss["X"].astype(float)
    loss_mask = loss_x <= maximum_depth
    axes[2].plot(
        loss_x[loss_mask],
        loss["total"][loss_mask],
        color="#0072b2",
        linewidth=1.8,
    )
    xmax = float(loss_summary["Xmax"])
    axes[2].axvline(
        xmax,
        color="#d55e00",
        linestyle="--",
        linewidth=1.4,
        label=rf"$X_{{\max}}={xmax:.1f}$ g cm$^{{-2}}$",
    )
    axes[2].set_title("Energy-deposit profile")
    axes[2].set_ylabel(r"$dE/dX$ [GeV per bin]")
    axes[2].legend(frameon=False, fontsize=9)

    first_interaction = float(interaction_summary["slant_depth"])
    muon_max = float(production_summary["XmuMax"])
    for axis in axes:
        axis.axvline(
            first_interaction,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            label=None,
        )
        axis.axvline(
            muon_max,
            color=COLORS["muon"],
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
            label=None,
        )
        axis.set_xlim(0.0, maximum_depth)
        axis.set_xlabel(r"slant depth $X$ [g cm$^{-2}$]")
        axis.grid(alpha=0.22)

    figure.suptitle(
        "100 PeV proton, zenith 47°, azimuth 180°, EM thinning $10^{-6}$\n"
        "dotted lines: first interaction and muon-production maximum",
        fontsize=12,
    )
    finish(figure, output)


def geomagnetic_axis(result: Path) -> np.ndarray:
    primary = read_yaml(result / "primary" / "summary.yaml")["shower_0"]
    environment = read_yaml(result / "gpu_em" / "config.yaml")[
        "environment"
    ]
    velocity = np.array(
        [primary["nx"], primary["ny"], primary["nz"]], dtype=float
    )
    field_record = environment["magnetic_field_T"]
    field = np.array(
        [field_record["x"], field_record["y"], field_record["z"]],
        dtype=float,
    )
    axis = np.cross(velocity, field)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot construct the v cross B polarization axis")
    return axis / norm


def read_radio(
    result: Path, algorithm: str, polarization: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    config = read_yaml(result / algorithm / "config.yaml")
    observer_items = list(config["observers"].items())
    counts = [int(round(item[1]["number of bins"])) for item in observer_items]
    if len(set(counts)) != 1:
        raise ValueError("radio observers do not share a common bin count")
    bins = counts[0]
    table = read_parquet(result / algorithm / "observers.parquet")
    expected = len(observer_items) * bins
    if len(table["Time"]) != expected:
        raise ValueError(
            f"{algorithm}: expected {expected} samples, found {len(table['Time'])}"
        )
    time = table["Time"].astype(float).reshape(len(observer_items), bins)
    field = np.stack(
        [table["Ex"], table["Ey"], table["Ez"]], axis=-1
    ).reshape(len(observer_items), bins, 3)
    projected = np.einsum("ijk,k->ij", field, polarization)
    locations = np.array(
        [item[1]["location"] for item in observer_items], dtype=float
    )
    names = [item[0] for item in observer_items]
    return time, projected, locations, names


def select_positive_y_observers(locations: np.ndarray) -> list[int]:
    targets = (0.0, 100.0, 300.0, 600.0)
    selected = []
    for radius in targets:
        target = np.array([0.0, radius])
        index = int(np.argmin(np.linalg.norm(locations[:, :2] - target, axis=1)))
        selected.append(index)
    return selected


def pulse_width(time: np.ndarray, signal: np.ndarray) -> float:
    amplitude = np.abs(signal)
    peak_index = int(np.argmax(amplitude))
    threshold = 0.5 * amplitude[peak_index]
    left = peak_index
    while left > 0 and amplitude[left - 1] >= threshold:
        left -= 1
    right = peak_index
    while right + 1 < len(amplitude) and amplitude[right + 1] >= threshold:
        right += 1
    if right == left:
        return float(np.median(np.diff(time)))
    return float(time[right] - time[left])


def plot_radio_time_profiles(result: Path, output: Path) -> None:
    polarization = geomagnetic_axis(result)
    radio = {
        algorithm: read_radio(result, algorithm, polarization)
        for algorithm in ("CoREAS", "ZHS")
    }
    locations = radio["CoREAS"][2]
    selected = select_positive_y_observers(locations)
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), sharex=True)
    for axis, observer_index in zip(axes.flat, selected):
        radius = np.linalg.norm(locations[observer_index, :2])
        for algorithm in ("CoREAS", "ZHS"):
            time, projected, _, _ = radio[algorithm]
            axis.plot(
                time[observer_index],
                1.0e3 * projected[observer_index],
                color=COLORS[algorithm],
                linewidth=1.6,
                label=algorithm,
            )
        axis.set_title(rf"$\rho={radius:.0f}$ m on the $+y$ spoke")
        axis.set_ylabel(r"$E_{\mathbf{v}\times\mathbf{B}}$ [mV m$^{-1}$]")
        axis.set_xlim(0.0, 140.0)
        axis.grid(alpha=0.22)
    for axis in axes[-1, :]:
        axis.set_xlabel("observer-window time [ns]")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Geomagnetic-polarization radio pulse profiles\n"
        "81 antennas, 1 ns sampling; CoREAS and ZHS use the same shower tracks"
    )
    finish(figure, output)


def radial_spokes(locations: np.ndarray) -> dict[float, list[int]]:
    spokes: dict[float, list[int]] = {}
    for index, (x, y, _) in enumerate(locations):
        radius = math.hypot(x, y)
        if radius < 0.5:
            continue
        angle = round(math.degrees(math.atan2(y, x)) % 360.0, 3)
        spokes.setdefault(angle, []).append(index)
    for indices in spokes.values():
        indices.sort(key=lambda i: math.hypot(locations[i, 0], locations[i, 1]))
    return spokes


def plot_radio_lateral_profiles(result: Path, output: Path) -> None:
    polarization = geomagnetic_axis(result)
    radio = {
        algorithm: read_radio(result, algorithm, polarization)
        for algorithm in ("CoREAS", "ZHS")
    }
    locations = radio["CoREAS"][2]
    spokes = radial_spokes(locations)
    unique_radii = np.array(
        sorted(
            {
                round(math.hypot(locations[i, 0], locations[i, 1]), 3)
                for indices in spokes.values()
                for i in indices
            }
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    for algorithm in ("CoREAS", "ZHS"):
        time, projected, _, _ = radio[algorithm]
        peak_curves = []
        width_curves = []
        for indices in spokes.values():
            radii = np.array(
                [math.hypot(locations[i, 0], locations[i, 1]) for i in indices]
            )
            peaks = np.array([np.max(np.abs(projected[i])) for i in indices])
            widths = np.array(
                [pulse_width(time[i], projected[i]) for i in indices]
            )
            axes[0].plot(
                radii,
                1.0e3 * peaks,
                color=COLORS[algorithm],
                alpha=0.16,
                linewidth=0.8,
            )
            axes[1].plot(
                radii,
                widths,
                color=COLORS[algorithm],
                alpha=0.16,
                linewidth=0.8,
            )
            peak_curves.append(peaks)
            width_curves.append(widths)
        peaks = np.asarray(peak_curves)
        widths = np.asarray(width_curves)
        for axis, curves, scale in (
            (axes[0], peaks, 1.0e3),
            (axes[1], widths, 1.0),
        ):
            mean = curves.mean(axis=0) * scale
            lower = curves.min(axis=0) * scale
            upper = curves.max(axis=0) * scale
            axis.fill_between(
                unique_radii,
                lower,
                upper,
                color=COLORS[algorithm],
                alpha=0.09,
                linewidth=0.0,
            )
            axis.plot(
                unique_radii,
                mean,
                color=COLORS[algorithm],
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                label=f"{algorithm}: 8-spoke mean",
            )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"peak $|E_{\mathbf{v}\times\mathbf{B}}|$ [mV m$^{-1}$]")
    axes[0].set_title("Peak geomagnetic amplitude")
    axes[1].set_ylabel("absolute-pulse FWHM [ns]")
    axes[1].set_title("Pulse width")
    for axis in axes:
        axis.set_xlabel(r"ground-plane radius $\rho$ [m]")
        axis.set_xlim(0.0, 610.0)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Single-shower radio lateral profiles\n"
        "thin curves/range bands show the eight antenna spokes"
    )
    finish(figure, output)


def main() -> int:
    args = parse_args()
    result = args.result.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_particle_profile(
        result,
        output_dir / "particle_longitudinal_profile.png",
        args.maximum_depth,
    )
    plot_radio_time_profiles(result, output_dir / "radio_time_profiles.png")
    plot_radio_lateral_profiles(
        result, output_dir / "radio_lateral_profiles.png"
    )
    print(output_dir / "particle_longitudinal_profile.png")
    print(output_dir / "radio_time_profiles.png")
    print(output_dir / "radio_lateral_profiles.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
