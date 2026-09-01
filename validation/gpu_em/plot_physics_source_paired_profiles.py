#!/usr/bin/env python3
"""Plot same-seed c8emrt/proposal-native longitudinal diagnostics.

The two showers are autonomous simulations.  Sharing a seed is useful for
diagnosis, but it does not force the two physics sources to consume identical
random draws or to grow an identical shower tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "c8emrt": "#0072B2",
    "proposal-native": "#D55E00",
    "photon": "#6A3D9A",
    "electron-positron": "#009E73",
    "hadron": "#B15928",
    "muon": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c8emrt", required=True, type=Path)
    parser.add_argument("--proposal-native", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-start", type=int, default=2026180001)
    parser.add_argument("--shower-index", type=int, default=0)
    return parser.parse_args()


def load(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def by_shower(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(index): group.sort_values("X").reset_index(drop=True)
        for index, group in frame.groupby("shower", sort=True)
    }


def component(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "electron-positron":
        return frame["electron"].to_numpy(float) + frame["positron"].to_numpy(float)
    if name == "muon":
        return frame["muplus"].to_numpy(float) + frame["muminus"].to_numpy(float)
    return frame[name].to_numpy(float)


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def normalized(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / denominator) if denominator else 0.0


def profile_metrics(
    c8_profiles: dict[int, pd.DataFrame],
    native_profiles: dict[int, pd.DataFrame],
) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    for shower in sorted(c8_profiles.keys() & native_profiles.keys()):
        c8 = c8_profiles[shower]
        native = native_profiles[shower]
        if not np.array_equal(c8["X"].to_numpy(), native["X"].to_numpy()):
            raise ValueError(f"shower {shower}: longitudinal grids differ")
        c8_em = component(c8, "electron-positron")
        native_em = component(native, "electron-positron")
        c8_peak = int(np.argmax(c8_em))
        native_peak = int(np.argmax(native_em))
        metrics.append(
            {
                "shower": shower,
                "em_relative_l2": relative_l2(c8_em, native_em),
                "em_correlation": float(np.corrcoef(c8_em, native_em)[0, 1]),
                "c8emrt_xmax_gcm2": float(c8["X"].iloc[c8_peak]),
                "native_xmax_gcm2": float(native["X"].iloc[native_peak]),
                "xmax_difference_gcm2": float(
                    native["X"].iloc[native_peak] - c8["X"].iloc[c8_peak]
                ),
                "c8emrt_nmax": float(c8_em[c8_peak]),
                "native_nmax": float(native_em[native_peak]),
                "nmax_ratio": float(native_em[native_peak] / c8_em[c8_peak])
                if c8_em[c8_peak] > 0.0
                else float("nan"),
            }
        )
    return metrics


def finish(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_single_event(
    c8: pd.DataFrame,
    native: pd.DataFrame,
    c8_loss: pd.DataFrame,
    native_loss: pd.DataFrame,
    shower: int,
    seed: int,
    output: Path,
) -> None:
    x = c8["X"].to_numpy(float)
    if not np.array_equal(x, native["X"].to_numpy(float)):
        raise ValueError("longitudinal grids differ")

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True)
    styles = (("c8emrt", c8, "-"), ("proposal-native", native, "--"))
    for label, frame, linestyle in styles:
        axes[0, 0].plot(
            x,
            positive(component(frame, "electron-positron")),
            color=COLORS[label],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[0, 1].plot(
            x,
            positive(component(frame, "photon")),
            color=COLORS[label],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[1, 0].plot(
            x,
            positive(component(frame, "hadron")),
            color=COLORS[label],
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{label}: hadrons",
        )
        axes[1, 0].plot(
            x,
            positive(component(frame, "muon")),
            color=COLORS[label],
            linestyle=":" if linestyle == "-" else "-.",
            linewidth=1.8,
            alpha=0.85,
            label=f"{label}: muons",
        )

    loss_styles = (
        ("c8emrt", c8_loss, "-"),
        ("proposal-native", native_loss, "--"),
    )
    for label, frame, linestyle in loss_styles:
        axes[1, 1].plot(
            frame["X"].to_numpy(float),
            frame["total"].to_numpy(float),
            color=COLORS[label],
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )

    titles = (r"$e^-+e^+$", r"$\gamma$", "Non-EM components", "Energy deposit")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel(r"slant depth $X$ [g cm$^{-2}$]")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[0, 0].set_ylabel("weighted particle count")
    axes[1, 0].set_ylabel("weighted particle count")
    axes[1, 1].set_ylabel(r"$dE/dX$ [GeV per bin]")

    c8_em = component(c8, "electron-positron")
    native_em = component(native, "electron-positron")
    c8_xmax = x[int(np.argmax(c8_em))]
    native_xmax = x[int(np.argmax(native_em))]
    occupied = np.zeros_like(x, dtype=bool)
    for frame in (c8, native):
        for name in ("electron-positron", "photon", "hadron", "muon"):
            occupied |= component(frame, name) > 0.0
    maximum_depth = float(np.max(x[occupied]) + 50.0) if np.any(occupied) else float(np.max(x))
    for axis in axes.flat:
        axis.set_xlim(0.0, maximum_depth)
    figure.suptitle(
        "Same-seed autonomous shower diagnostic\n"
        f"seed={seed}, shower={shower}; "
        f"EM relative $L_2$={relative_l2(c8_em, native_em):.3f}, "
        rf"$\Delta X_{{\max}}$={native_xmax-c8_xmax:+.1f} g cm$^{{-2}}$",
        fontsize=12,
    )
    finish(figure, output)


def plot_shape_residual(
    c8: pd.DataFrame,
    native: pd.DataFrame,
    c8_loss: pd.DataFrame,
    native_loss: pd.DataFrame,
    seed: int,
    output: Path,
) -> None:
    x = c8["X"].to_numpy(float)
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True)
    items = (
        ("electron-positron", r"$e^-+e^+$ normalized shape"),
        ("photon", r"$\gamma$ normalized shape"),
    )
    for axis, (name, title) in zip(axes[:2], items):
        c8_values = normalized(component(c8, name))
        native_values = normalized(component(native, name))
        axis.plot(x, c8_values, color=COLORS["c8emrt"], linewidth=2.0, label="c8emrt")
        axis.plot(
            x,
            native_values,
            color=COLORS["proposal-native"],
            linewidth=2.0,
            linestyle="--",
            label="proposal-native",
        )
        axis.fill_between(
            x,
            0.0,
            native_values - c8_values,
            color="#999999",
            alpha=0.22,
            label="native − c8emrt",
        )
        axis.set_title(title)
        axis.set_ylabel("peak-normalized count")
        axis.legend(frameon=False, fontsize=8)

    c8_loss_values = normalized(c8_loss["total"].to_numpy(float))
    native_loss_values = normalized(native_loss["total"].to_numpy(float))
    loss_x = c8_loss["X"].to_numpy(float)
    axes[2].plot(loss_x, c8_loss_values, color=COLORS["c8emrt"], linewidth=2.0, label="c8emrt")
    axes[2].plot(
        loss_x,
        native_loss_values,
        color=COLORS["proposal-native"],
        linewidth=2.0,
        linestyle="--",
        label="proposal-native",
    )
    axes[2].set_title("Energy-deposit normalized shape")
    axes[2].set_ylabel("peak-normalized $dE/dX$")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel(r"slant depth $X$ [g cm$^{-2}$]")
        axis.grid(alpha=0.22)
    occupied = (
        (component(c8, "electron-positron") > 0.0)
        | (component(native, "electron-positron") > 0.0)
        | (component(c8, "photon") > 0.0)
        | (component(native, "photon") > 0.0)
    )
    maximum_depth = float(np.max(x[occupied]) + 50.0) if np.any(occupied) else float(np.max(x))
    for axis in axes:
        axis.set_xlim(0.0, maximum_depth)
    figure.suptitle(
        f"Same seed {seed}: shape comparison (not a forced decision-tape replay)",
        fontsize=12,
    )
    finish(figure, output)


def plot_pair_summary(metrics: list[dict[str, float | int]], seed_start: int, output: Path) -> None:
    frame = pd.DataFrame(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    axes[0].hist(frame["em_relative_l2"], bins="auto", color="#56B4E9", edgecolor="white")
    axes[0].axvline(frame["em_relative_l2"].median(), color="#D55E00", linestyle="--")
    axes[0].set_xlabel(r"EM profile relative $L_2$")
    axes[0].set_ylabel("paired showers")
    axes[0].set_title("Whole-profile difference")

    low = float(min(frame["c8emrt_xmax_gcm2"].min(), frame["native_xmax_gcm2"].min()))
    high = float(max(frame["c8emrt_xmax_gcm2"].max(), frame["native_xmax_gcm2"].max()))
    axes[1].scatter(
        frame["c8emrt_xmax_gcm2"],
        frame["native_xmax_gcm2"],
        s=28,
        color="#009E73",
        alpha=0.8,
    )
    axes[1].plot([low, high], [low, high], color="0.35", linestyle="--")
    axes[1].set_xlabel(r"c8emrt $X_{\max}$ [g cm$^{-2}$]")
    axes[1].set_ylabel(r"native $X_{\max}$ [g cm$^{-2}$]")
    axes[1].set_title("Same-seed pair correlation")

    axes[2].scatter(
        frame["shower"] + 1,
        frame["nmax_ratio"],
        s=28,
        color="#CC79A7",
        alpha=0.8,
    )
    axes[2].axhline(1.0, color="0.35", linestyle="--")
    axes[2].set_xlabel(f"paired shower index (seed = {seed_start} + index − 1)")
    axes[2].set_ylabel(r"native/c8emrt EM $N_{\max}$")
    axes[2].set_title("Peak-size pair ratio")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        f"Autonomous same-seed diagnostic, {len(frame)} paired showers",
        fontsize=12,
    )
    finish(figure, output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    c8_profiles = by_shower(load(args.c8emrt, "profile/profile.parquet"))
    native_profiles = by_shower(load(args.proposal_native, "profile/profile.parquet"))
    c8_losses = by_shower(load(args.c8emrt, "energyloss/dEdX.parquet"))
    native_losses = by_shower(load(args.proposal_native, "energyloss/dEdX.parquet"))
    shared = sorted(c8_profiles.keys() & native_profiles.keys())
    if not shared:
        raise ValueError("no shared shower indices")
    if args.shower_index not in shared:
        raise ValueError(f"shower {args.shower_index} is not shared")

    seed = args.seed_start + args.shower_index
    plot_single_event(
        c8_profiles[args.shower_index],
        native_profiles[args.shower_index],
        c8_losses[args.shower_index],
        native_losses[args.shower_index],
        args.shower_index,
        seed,
        args.output_dir / f"same_seed_single_event_profile_seed{seed}.png",
    )
    plot_shape_residual(
        c8_profiles[args.shower_index],
        native_profiles[args.shower_index],
        c8_losses[args.shower_index],
        native_losses[args.shower_index],
        seed,
        args.output_dir / f"same_seed_single_event_shape_seed{seed}.png",
    )
    metrics = profile_metrics(c8_profiles, native_profiles)
    plot_pair_summary(
        metrics,
        args.seed_start,
        args.output_dir / "same_seed_pair_summary_25.png",
    )
    payload = {
        "note": (
            "Same seed does not force identical random-draw addressing or an "
            "identical shower tree; these are autonomous paired diagnostics."
        ),
        "c8emrt": str(args.c8emrt.resolve()),
        "proposal_native": str(args.proposal_native.resolve()),
        "seed_start": args.seed_start,
        "selected_shower": args.shower_index,
        "selected_seed": seed,
        "pairs": metrics,
    }
    (args.output_dir / "same_seed_pair_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
