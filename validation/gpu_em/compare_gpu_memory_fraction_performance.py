#!/usr/bin/env python3
"""Compare scalar CPU, 70%-cap CUDA, and 90%-cap CUDA shower runtimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-root", type=Path, action="append", required=True)
    parser.add_argument("--gpu70-root", type=Path, required=True)
    parser.add_argument("--gpu90-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cpu", type=int, default=1000)
    parser.add_argument("--expected-gpu70", type=int, default=500)
    parser.add_argument("--expected-gpu90", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def timing_files(roots: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        files.update(path.resolve() for path in root.rglob("simulation_timing/summary.yaml"))
    return sorted(files)


def read_timings(roots: list[Path], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in timing_files(roots):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"invalid timing summary: {path}")
        for shower, value in document.items():
            if not isinstance(value, dict):
                raise ValueError(f"invalid timing record in {path}: {shower}")
            if value.get("closed") is not True or value.get("status") != "closed":
                raise ValueError(f"unclosed timing record in {path}: {shower}")
            seconds = float(value["wall_time_ms"]) / 1000.0
            if not np.isfinite(seconds) or seconds <= 0.0:
                raise ValueError(f"invalid wall time in {path}: {seconds}")
            records.append(
                {
                    "group": label,
                    "seconds": seconds,
                    "source": str(path),
                    "shower": str(shower),
                }
            )
    return records


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "std_seconds": float(np.std(values, ddof=1)),
        "q05_seconds": float(np.quantile(values, 0.05)),
        "q95_seconds": float(np.quantile(values, 0.95)),
        "total_seconds": float(np.sum(values)),
    }


def bootstrap_ratio_and_difference(
    reference: np.ndarray, candidate: np.ndarray, repetitions: int
) -> dict[str, Any]:
    rng = np.random.default_rng(20260810)
    ref_indices = rng.integers(0, reference.size, size=(repetitions, reference.size))
    cand_indices = rng.integers(0, candidate.size, size=(repetitions, candidate.size))
    ref_means = reference[ref_indices].mean(axis=1)
    cand_means = candidate[cand_indices].mean(axis=1)
    ratio = cand_means / ref_means
    difference = cand_means - ref_means
    return {
        "mean_runtime_ratio_90_over_70": float(np.mean(candidate) / np.mean(reference)),
        "mean_runtime_ratio_95pct": [float(x) for x in np.quantile(ratio, [0.025, 0.975])],
        "mean_difference_seconds_90_minus_70": float(np.mean(candidate) - np.mean(reference)),
        "mean_difference_95pct_seconds": [
            float(x) for x in np.quantile(difference, [0.025, 0.975])
        ],
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_all(cpu: np.ndarray, gpu70: np.ndarray, gpu90: np.ndarray, output: Path) -> None:
    configure_style()
    colors = {"CPU": "#4C78A8", "CUDA 70%": "#F58518", "CUDA 90%": "#54A24B"}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    bins = np.geomspace(min(gpu70.min(), gpu90.min()), cpu.max(), 42)
    axes[0].hist(cpu, bins=bins, density=True, histtype="step", linewidth=1.8, label="CPU", color=colors["CPU"])
    axes[0].hist(gpu70, bins=bins, density=True, histtype="step", linewidth=1.8, label="CUDA 70%", color=colors["CUDA 70%"])
    axes[0].hist(gpu90, bins=bins, density=True, histtype="step", linewidth=1.8, label="CUDA 90%", color=colors["CUDA 90%"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Wall time per shower [s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("Cross-system turnaround distributions")
    axes[0].legend(frameon=False)

    for values, label in ((cpu, "CPU"), (gpu70, "CUDA 70%"), (gpu90, "CUDA 90%")):
        ordered = np.sort(values)
        probability = np.arange(1, ordered.size + 1) / ordered.size
        axes[1].plot(ordered, probability, linewidth=1.7, label=label, color=colors[label])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Wall time per shower [s]")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Runtime cumulative distributions")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_gpu(gpu70: np.ndarray, gpu90: np.ndarray, output: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    low = min(gpu70.min(), gpu90.min())
    high = max(gpu70.max(), gpu90.max())
    bins = np.linspace(low, high, 36)
    axes[0].hist(gpu70, bins=bins, density=True, alpha=0.52, color="#F58518", label="70% cap")
    axes[0].hist(gpu90, bins=bins, density=True, alpha=0.52, color="#54A24B", label="90% cap")
    axes[0].axvline(np.mean(gpu70), color="#F58518", linestyle="--", linewidth=1.4)
    axes[0].axvline(np.mean(gpu90), color="#54A24B", linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Wall time per shower [s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("CUDA runtime distributions")
    axes[0].legend(frameon=False)

    boxes = axes[1].boxplot(
        [gpu70, gpu90], labels=["70% cap", "90% cap"], showfliers=False, patch_artist=True
    )
    for patch, color in zip(boxes["boxes"], ("#F58518", "#54A24B")):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ratio = np.mean(gpu90) / np.mean(gpu70)
    axes[1].set_ylabel("Wall time per shower [s]")
    axes[1].set_title(f"Mean ratio 90% / 70% = {ratio:.3f}")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = (
        read_timings(args.cpu_root, "CPU")
        + read_timings([args.gpu70_root], "CUDA 70%")
        + read_timings([args.gpu90_root], "CUDA 90%")
    )
    groups = {
        name: np.asarray([record["seconds"] for record in records if record["group"] == name])
        for name in ("CPU", "CUDA 70%", "CUDA 90%")
    }
    expected = {"CPU": args.expected_cpu, "CUDA 70%": args.expected_gpu70, "CUDA 90%": args.expected_gpu90}
    for name, count in expected.items():
        if groups[name].size != count:
            raise ValueError(f"{name} timing count is {groups[name].size}, expected {count}")

    cpu, gpu70, gpu90 = groups["CPU"], groups["CUDA 70%"], groups["CUDA 90%"]
    comparison = bootstrap_ratio_and_difference(gpu70, gpu90, args.bootstrap_repetitions)
    welch = stats.ttest_ind(gpu90, gpu70, equal_var=False)
    mann = stats.mannwhitneyu(gpu90, gpu70, alternative="two-sided")
    summary = {
        "groups": {name: describe(values) for name, values in groups.items()},
        "cross_system_speedup_from_mean": {
            "cpu_over_cuda70": float(np.mean(cpu) / np.mean(gpu70)),
            "cpu_over_cuda90": float(np.mean(cpu) / np.mean(gpu90)),
            "cpu_over_pooled_cuda1000": float(np.mean(cpu) / np.mean(np.concatenate((gpu70, gpu90)))),
        },
        "same_laptop_memory_cap_comparison": {
            **comparison,
            "welch_t_pvalue": float(welch.pvalue),
            "mann_whitney_pvalue": float(mann.pvalue),
        },
        "interpretation_note": "CPU and CUDA ran on different hosts; CPU/CUDA ratios are production turnaround ratios. The 70%/90% comparison used the same laptop GPU and executable/table hashes.",
    }
    atomic_json(output / "runtime_memory_fraction_summary.json", summary)
    with (output / "runtime_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group", "seconds", "source", "shower"))
        writer.writeheader()
        writer.writerows(records)
    plot_all(cpu, gpu70, gpu90, output / "runtime_cpu_cuda70_cuda90_distributions.png")
    plot_gpu(gpu70, gpu90, output / "runtime_cuda70_cuda90_comparison.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
