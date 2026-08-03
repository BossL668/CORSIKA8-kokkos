#!/usr/bin/env python3
"""Plot the device-resident muon-parent production-profile accounting fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--old-comparison", type=Path, required=True)
    parser.add_argument("--fixed-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_cpu_profiles(root: Path) -> tuple[np.ndarray, np.ndarray]:
    curves: list[np.ndarray] = []
    coordinate: np.ndarray | None = None
    paths = sorted(root.glob("proposal_shard_*/production_profile/profile.parquet"))
    for path in paths:
        frame = pd.read_parquet(path)
        for _, shower in frame.groupby("shower", sort=True):
            shower = shower.sort_values("X")
            current_coordinate = shower["X"].to_numpy(dtype=float)
            if coordinate is None:
                coordinate = current_coordinate
            elif not np.array_equal(coordinate, current_coordinate):
                raise RuntimeError(f"inconsistent production-profile grid: {path}")
            curves.append(shower["muon"].to_numpy(dtype=float))
    if coordinate is None or not curves:
        raise RuntimeError(f"no CPU production profiles found below {root}")
    return coordinate, np.vstack(curves)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coordinate, cpu = load_cpu_profiles(args.cpu_root)
    old = pd.read_csv(args.old_comparison)
    old = old[old["observable"] == "muon_production_parent_muon"].sort_values(
        "coordinate"
    )
    if old.empty:
        raise RuntimeError("old comparison has no muon-parent curve")
    fixed = pd.read_parquet(args.fixed_profile).sort_values("X")
    fixed_coordinate = fixed["X"].to_numpy(dtype=float)
    fixed_curve = fixed["muon"].to_numpy(dtype=float)
    if not np.array_equal(coordinate, fixed_coordinate):
        raise RuntimeError("fixed CUDA and CPU production-profile grids differ")

    cpu_mean = cpu.mean(axis=0)
    cpu_q16, cpu_q84 = np.quantile(cpu, [0.16, 0.84], axis=0)
    old_coordinate = old["coordinate"].to_numpy(dtype=float)
    old_cuda_mean = old["cuda_mean"].to_numpy(dtype=float)
    if not np.array_equal(coordinate, old_coordinate):
        old_cuda_mean = np.interp(coordinate, old_coordinate, old_cuda_mean)

    diagnostic = pd.DataFrame(
        {
            "X_g_per_cm2": coordinate,
            "cpu500_mean": cpu_mean,
            "cpu500_q16": cpu_q16,
            "cpu500_q84": cpu_q84,
            "old_cuda500_mean": old_cuda_mean,
            "fixed_cuda_seed10100051": fixed_curve,
        }
    )
    diagnostic.to_csv(args.output_dir / "muon_parent_profile_fix_curves.csv", index=False)

    totals = cpu.sum(axis=1)
    fixed_total = float(fixed_curve.sum())
    old_total = float(old_cuda_mean.sum())
    summary = {
        "cpu_events": int(cpu.shape[0]),
        "cpu_total_mean": float(totals.mean()),
        "cpu_total_standard_deviation": float(totals.std(ddof=1)),
        "cpu_total_q025_q50_q975": np.quantile(totals, [0.025, 0.5, 0.975]).tolist(),
        "old_cuda500_mean_total": old_total,
        "fixed_cuda_total": fixed_total,
        "fixed_cuda_z_in_cpu_distribution": float(
            (fixed_total - totals.mean()) / totals.std(ddof=1)
        ),
        "fixed_cuda_percentile_in_cpu_distribution": float(
            np.mean(totals <= fixed_total)
        ),
    }
    (args.output_dir / "muon_parent_profile_fix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.8), constrained_layout=True)

    active = (cpu_mean > 0) | (fixed_curve > 0) | (old_cuda_mean > 0)
    x_max = min(float(coordinate[active].max() + 20), float(coordinate.max()))
    ax = axes[0]
    ax.fill_between(
        coordinate,
        cpu_q16,
        cpu_q84,
        color="#777777",
        alpha=0.22,
        label="Original CPU: central 68% of 500 showers",
    )
    ax.plot(coordinate, cpu_mean, color="black", linewidth=2.2, label="Original CPU500 mean")
    ax.plot(
        coordinate,
        old_cuda_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="CUDA500 before fix (mean)",
    )
    ax.plot(
        coordinate,
        fixed_curve,
        color="#0072b2",
        linewidth=1.7,
        label="CUDA after fix (seed 10100051)",
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Slant depth $X$ [g cm$^{-2}$]")
    ax.set_ylabel("Muon productions from muon parent / 10 g cm$^{-2}$")
    ax.set_title("Muon-parent production profile: device accounting repair")
    ax.legend(frameon=True, fontsize=9)

    ax = axes[1]
    labels = ["Old CUDA500\nmean", "Original CPU500\nmean", "Fixed CUDA\none shower"]
    values = [old_total, float(totals.mean()), fixed_total]
    colors = ["#d62728", "#444444", "#0072b2"]
    bars = ax.bar(labels, values, color=colors, alpha=0.88, width=0.62)
    ax.errorbar(
        [1],
        [totals.mean()],
        yerr=[totals.std(ddof=1)],
        fmt="none",
        color="black",
        capsize=6,
        linewidth=1.5,
        label="CPU inter-shower standard deviation",
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Profile integral per shower")
    ax.set_title(
        f"Fixed event is {(fixed_total - totals.mean()) / totals.std(ddof=1):+.2f} "
        "standard deviations from the CPU500 mean"
    )
    ax.legend(frameon=True, fontsize=9)

    figure.savefig(args.output_dir / "muon_parent_profile_fix_smoke.png", dpi=180)
    figure.savefig(args.output_dir / "muon_parent_profile_fix_smoke.pdf")


if __name__ == "__main__":
    main()
