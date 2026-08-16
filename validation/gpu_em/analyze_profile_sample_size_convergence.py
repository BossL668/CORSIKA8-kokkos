#!/usr/bin/env python3
"""Plot nested-subsample longitudinal means with pointwise 95% confidence bands."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


PROFILE_COLUMNS = (
    "charged",
    "hadron",
    "photon",
    "electron",
    "positron",
    "muplus",
    "muminus",
)
PRODUCTION_COLUMNS = (
    "pion",
    "kaon",
    "heavy",
    "hadron",
    "photon",
    "electron-positron",
    "muon",
    "all",
)
CPU_COLOR = "#1f77b4"
CUDA_COLOR = "#d62728"

EM_FEATURES = (
    ("profile_electron_positron", r"$N_{e^-}+N_{e^+}$"),
    ("profile_photon", r"$N_\gamma$"),
    ("profile_em", r"$N_\gamma+N_{e^-}+N_{e^+}$"),
    ("energy_deposit", r"$dE/dX$ [GeV per bin]"),
)
NON_EM_FEATURES = (
    ("profile_hadron", r"$N_{\rm hadron}$"),
    ("profile_muon", r"$N_{\mu^-}+N_{\mu^+}$"),
    ("profile_muplus", r"$N_{\mu^+}$"),
    ("profile_muminus", r"$N_{\mu^-}$"),
    ("profile_charged", r"$N_{\rm charged}$"),
)
PARENT_FEATURES = (
    ("muon_production_parent_pion", "pion parent"),
    ("muon_production_parent_kaon", "kaon parent"),
    ("muon_production_parent_heavy", "heavy-hadron parent"),
    ("muon_production_parent_hadron", "all hadron parents"),
    ("muon_production_parent_photon", "photon parent"),
    ("muon_production_parent_electron_positron", r"$e^\pm$ parent"),
    ("muon_production_parent_muon", "muon parent"),
    ("muon_production_parent_all", "all parent species"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-root",
        type=Path,
        action="append",
        required=True,
        help="CPU campaign root; repeat for a split campaign",
    )
    parser.add_argument(
        "--cuda-root",
        type=Path,
        action="append",
        required=True,
        help="CUDA campaign root; repeat for a split campaign",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, action="append", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--include-full-ensemble", action="store_true")
    parser.add_argument("--active-fraction", type=float, default=1.0e-4)
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


def matrix(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    shower_ids = sorted(int(value) for value in frame["shower"].unique())
    coordinate: np.ndarray | None = None
    rows: list[np.ndarray] = []
    for shower in shower_ids:
        selected = frame.loc[frame["shower"] == shower].sort_values("X")
        x = selected["X"].to_numpy(dtype=np.float64)
        values = selected[column].to_numpy(dtype=np.float64)
        if coordinate is None:
            coordinate = x
        elif coordinate.shape != x.shape or not np.array_equal(coordinate, x):
            raise ValueError(f"inconsistent X grid for {column}, shower {shower}")
        rows.append(values)
    if coordinate is None or not rows:
        raise ValueError(f"empty curve matrix for {column}")
    return coordinate, np.stack(rows)


def read_source(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    profile = pd.read_parquet(
        root / "profile" / "profile.parquet",
        columns=["shower", "X", *PROFILE_COLUMNS],
    )
    production = pd.read_parquet(
        root / "production_profile" / "profile.parquet",
        columns=["shower", "X", *PRODUCTION_COLUMNS],
    )
    losses = pd.read_parquet(
        root / "energyloss" / "dEdX.parquet",
        columns=["shower", "X", "total"],
    )
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in PROFILE_COLUMNS:
        curves[f"profile_{column}"] = matrix(profile, column)
    axis = curves["profile_charged"][0]
    curves["profile_electron_positron"] = (
        axis,
        curves["profile_electron"][1] + curves["profile_positron"][1],
    )
    curves["profile_muon"] = (
        axis,
        curves["profile_muplus"][1] + curves["profile_muminus"][1],
    )
    curves["profile_em"] = (
        axis,
        curves["profile_photon"][1]
        + curves["profile_electron"][1]
        + curves["profile_positron"][1],
    )
    for column in PRODUCTION_COLUMNS:
        name = column.replace("-", "_")
        curves[f"muon_production_parent_{name}"] = matrix(production, column)
    curves["energy_deposit"] = matrix(losses, "total")
    return curves


def source_roots(roots: list[Path], backend: str) -> list[Path]:
    sources: list[Path] = []
    for root in roots:
        if backend == "cpu":
            discovered = root.glob("proposal_shard_*")
        else:
            discovered = root.glob("batch_*/cuda")
        sources.extend(path for path in discovered if path.is_dir())
    sources = sorted(set(path.resolve() for path in sources))
    if not sources:
        joined = ", ".join(str(root) for root in roots)
        raise ValueError(f"no {backend} sources found under {joined}")
    return sources


def load_backend(
    roots: list[Path], backend: str
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    combined: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    labels: list[str] = []
    for source in source_roots(roots, backend):
        curves = read_source(source)
        count = next(iter(curves.values()))[1].shape[0]
        labels.extend(f"{source}:{index}" for index in range(count))
        if not combined:
            combined = {name: (axis, [values]) for name, (axis, values) in curves.items()}
            continue
        if set(combined) != set(curves):
            raise ValueError(f"observable set changed in {source}")
        for name, (axis, values) in curves.items():
            reference, chunks = combined[name]
            if reference.shape != axis.shape or not np.array_equal(reference, axis):
                raise ValueError(f"coordinate grid changed for {name} in {source}")
            chunks.append(values)
    result = {
        name: (axis, np.concatenate(chunks, axis=0))
        for name, (axis, chunks) in combined.items()
    }
    counts = {values.shape[0] for _, values in result.values()}
    if counts != {len(labels)}:
        raise ValueError(f"inconsistent event counts for {backend}: {counts}, labels={len(labels)}")
    return result, labels


def mean_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = values.shape[0]
    if count < 2:
        raise ValueError("at least two showers are required")
    mean = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(count)
    critical = float(stats.t.ppf(0.975, count - 1))
    half_width = critical * standard_error
    return mean, standard_error, mean - half_width, mean + half_width


def profile_statistics(
    axis: np.ndarray,
    cpu: np.ndarray,
    cuda: np.ndarray,
    active_fraction: float,
) -> dict[str, Any]:
    cpu_mean, cpu_se, cpu_low, cpu_high = mean_interval(cpu)
    cuda_mean, cuda_se, cuda_low, cuda_high = mean_interval(cuda)
    difference = cuda_mean - cpu_mean
    combined_se = np.sqrt(cpu_se * cpu_se + cuda_se * cuda_se)
    critical = float(stats.norm.ppf(0.975))
    scale = float(np.max(np.abs(cpu_mean)))
    active = np.abs(cpu_mean) >= active_fraction * scale if scale > 0.0 else np.zeros_like(cpu_mean, dtype=bool)
    denominator = float(np.sum(np.abs(cpu_mean[active])))
    relative_l1 = float(np.sum(np.abs(difference[active])) / denominator) if denominator else 0.0
    integral_cpu = float(np.trapz(cpu_mean, axis))
    integral_cuda = float(np.trapz(cuda_mean, axis))
    peak_cpu = float(np.max(cpu_mean))
    peak_cuda = float(np.max(cuda_mean))
    return {
        "axis": axis,
        "cpu_mean": cpu_mean,
        "cpu_low": cpu_low,
        "cpu_high": cpu_high,
        "cuda_mean": cuda_mean,
        "cuda_low": cuda_low,
        "cuda_high": cuda_high,
        "difference": difference,
        "difference_half_width": critical * combined_se,
        "active": active,
        "relative_l1": relative_l1,
        "integral_ratio_cuda_over_cpu": integral_cuda / integral_cpu if integral_cpu else math.nan,
        "peak_ratio_cuda_over_cpu": peak_cuda / peak_cpu if peak_cpu else math.nan,
        "cpu_median_95ci_half_width_over_peak": float(np.median((cpu_high - cpu_low)[active] / 2.0) / peak_cpu) if peak_cpu and np.any(active) else math.nan,
        "cuda_median_95ci_half_width_over_peak": float(np.median((cuda_high - cuda_low)[active] / 2.0) / peak_cuda) if peak_cuda and np.any(active) else math.nan,
    }


def plot_grid(
    statistics: dict[str, dict[str, Any]],
    features: tuple[tuple[str, str], ...],
    output: Path,
    title: str,
    columns: int,
) -> None:
    columns = min(columns, len(features))
    rows = math.ceil(len(features) / columns)
    active_coordinates = [
        statistics[observable]["axis"][statistics[observable]["active"]]
        for observable, _ in features
        if np.any(statistics[observable]["active"])
    ]
    if active_coordinates:
        support = np.concatenate(active_coordinates)
        support_low = max(0.0, float(np.min(support)))
        support_high = float(np.max(support))
        margin = max(10.0, 0.04 * max(support_high - support_low, 1.0))
        x_limits = (max(0.0, support_low - margin), support_high + margin)
    else:
        full_axis = statistics[features[0][0]]["axis"]
        x_limits = (max(0.0, float(np.min(full_axis))), float(np.max(full_axis)))
    figure, axes = plt.subplots(
        2 * rows,
        columns,
        figsize=(5.0 * columns, 5.9 * rows),
        squeeze=False,
        gridspec_kw={"height_ratios": tuple(value for _ in range(rows) for value in (2.2, 1.0))},
    )
    for panel, (observable, label) in enumerate(features):
        row, column = divmod(panel, columns)
        item = statistics[observable]
        x = item["axis"]
        upper = axes[2 * row, column]
        upper.fill_between(x, np.maximum(item["cpu_low"], 0.0), item["cpu_high"], color=CPU_COLOR, alpha=0.18, linewidth=0.0)
        upper.fill_between(x, np.maximum(item["cuda_low"], 0.0), item["cuda_high"], color=CUDA_COLOR, alpha=0.16, linewidth=0.0)
        upper.plot(x, item["cpu_mean"], color=CPU_COLOR, linewidth=1.7, label="Original CPU mean")
        upper.plot(x, item["cuda_mean"], color=CUDA_COLOR, linewidth=1.5, linestyle="--", label="CUDA mean")
        upper.set_xlim(*x_limits)
        upper.set_ylabel(label)
        upper.grid(alpha=0.2)

        lower = axes[2 * row + 1, column]
        cpu_mean = item["cpu_mean"]
        active = item["active"] & (cpu_mean != 0.0)
        ratio = np.full_like(cpu_mean, np.nan)
        half = np.full_like(cpu_mean, np.nan)
        ratio[active] = item["difference"][active] / cpu_mean[active]
        half[active] = item["difference_half_width"][active] / np.abs(cpu_mean[active])
        lower.axhline(0.0, color="0.35", linewidth=1.0)
        lower.fill_between(x, 100.0 * (ratio - half), 100.0 * (ratio + half), color=CUDA_COLOR, alpha=0.18, linewidth=0.0)
        lower.plot(x, 100.0 * ratio, color=CUDA_COLOR, linewidth=1.2)
        lower.set_xlim(*x_limits)
        lower.set_xlabel(r"slant depth [g cm$^{-2}$]")
        lower.set_ylabel(r"$(\bar N_{\rm CUDA}-\bar N_{\rm CPU})/\bar N_{\rm CPU}$ [%]")
        lower.grid(alpha=0.2)
    for panel in range(len(features), rows * columns):
        row, column = divmod(panel, columns)
        axes[2 * row, column].set_visible(False)
        axes[2 * row + 1, column].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle(
        title + "\nshading: pointwise 95% confidence interval of the ensemble mean",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    figure.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(figure)


def plot_convergence(rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    selected = ("profile_em", "profile_hadron", "profile_muon", "energy_deposit")
    labels = {
        "profile_em": "EM",
        "profile_hadron": "hadrons",
        "profile_muon": "muons",
        "energy_deposit": "energy deposit",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for observable in selected:
        data = frame.loc[frame["observable"] == observable].sort_values("sample_size")
        axes[0].plot(data["sample_size"], 100.0 * data["relative_l1"], marker="o", label=labels[observable])
        axes[1].plot(data["sample_size"], data["peak_ratio_cuda_over_cpu"], marker="o", label=labels[observable])
        axes[2].plot(data["sample_size"], 100.0 * data["cpu_median_95ci_half_width_over_peak"], marker="o", color=CPU_COLOR, alpha=0.75)
        axes[2].plot(data["sample_size"], 100.0 * data["cuda_median_95ci_half_width_over_peak"], marker="s", linestyle="--", color=CUDA_COLOR, alpha=0.75)
    axes[0].set_ylabel("Mean-curve relative $L_1$ difference [%]")
    axes[1].axhline(1.0, color="0.35", linewidth=1.0)
    axes[1].set_ylabel("CUDA / CPU mean peak")
    axes[2].set_ylabel("Median 95% CI half-width / peak [%]")
    for axis in axes:
        axis.set_xlabel("Showers per backend")
        axis.set_xscale("log")
        axis.set_xticks(sorted(frame["sample_size"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[2].plot([], [], color=CPU_COLOR, marker="o", label="CPU")
    axes[2].plot([], [], color=CUDA_COLOR, marker="s", linestyle="--", label="CUDA")
    axes[2].legend(frameon=False)
    figure.suptitle("Nested-subsample convergence of longitudinal-profile means")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    sample_sizes = sorted(set(args.sample_size))
    if any(value < 2 for value in sample_sizes):
        raise ValueError("sample sizes must be at least two")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cpu_roots = [root.resolve() for root in args.cpu_root]
    cuda_roots = [root.resolve() for root in args.cuda_root]
    cpu_curves, cpu_labels = load_backend(cpu_roots, "cpu")
    cuda_curves, cuda_labels = load_backend(cuda_roots, "cuda")
    if set(cpu_curves) != set(cuda_curves):
        raise ValueError("CPU/CUDA observable sets differ")
    cpu_events, cuda_events = len(cpu_labels), len(cuda_labels)
    if args.include_full_ensemble:
        if cpu_events != cuda_events:
            raise ValueError("full-ensemble comparison requires equal backend counts")
        sample_sizes.append(cpu_events)
        sample_sizes = sorted(set(sample_sizes))
    if max(sample_sizes) > min(cpu_events, cuda_events):
        raise ValueError(f"sample size exceeds available events: CPU={cpu_events}, CUDA={cuda_events}")

    rng_cpu = np.random.default_rng(args.selection_seed)
    rng_cuda = np.random.default_rng(args.selection_seed + 1)
    cpu_order = rng_cpu.permutation(cpu_events)
    cuda_order = rng_cuda.permutation(cuda_events)
    atomic_json(
        output / "nested_selection.json",
        {
            "selection_seed_cpu": args.selection_seed,
            "selection_seed_cuda": args.selection_seed + 1,
            "cpu_roots": [str(root) for root in cpu_roots],
            "cuda_roots": [str(root) for root in cuda_roots],
            "sample_sizes": sample_sizes,
            "cpu_order": [cpu_labels[index] for index in cpu_order],
            "cuda_order": [cuda_labels[index] for index in cuda_order],
            "interval": "pointwise 95% Student-t confidence interval for each backend mean; normal approximation for the independent mean difference",
        },
    )

    summary_rows: list[dict[str, Any]] = []
    for count in sample_sizes:
        statistics: dict[str, dict[str, Any]] = {}
        for observable in sorted(cpu_curves):
            cpu_axis, cpu_matrix = cpu_curves[observable]
            cuda_axis, cuda_matrix = cuda_curves[observable]
            if cpu_axis.shape != cuda_axis.shape or not np.array_equal(cpu_axis, cuda_axis):
                raise ValueError(f"coordinate grid differs for {observable}")
            item = profile_statistics(
                cpu_axis,
                cpu_matrix[cpu_order[:count]],
                cuda_matrix[cuda_order[:count]],
                args.active_fraction,
            )
            statistics[observable] = item
            summary_rows.append(
                {
                    "sample_size": count,
                    "observable": observable,
                    "relative_l1": item["relative_l1"],
                    "integral_ratio_cuda_over_cpu": item["integral_ratio_cuda_over_cpu"],
                    "peak_ratio_cuda_over_cpu": item["peak_ratio_cuda_over_cpu"],
                    "cpu_median_95ci_half_width_over_peak": item["cpu_median_95ci_half_width_over_peak"],
                    "cuda_median_95ci_half_width_over_peak": item["cuda_median_95ci_half_width_over_peak"],
                }
            )
        prefix = f"n{count:03d}"
        plot_grid(statistics, EM_FEATURES, output / f"{prefix}_longitudinal_em_with_95ci.png", f"Electromagnetic profiles: {count} CPU vs {count} CUDA showers", 4)
        plot_grid(statistics, NON_EM_FEATURES, output / f"{prefix}_longitudinal_non_em_with_95ci.png", f"Non-EM profiles: {count} CPU vs {count} CUDA showers", 3)
        plot_grid(statistics, PARENT_FEATURES, output / f"{prefix}_muon_parent_profiles_with_95ci.png", f"Muon-parent profiles: {count} CPU vs {count} CUDA showers", 4)

    with (output / "profile_convergence_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_convergence(summary_rows, output / "profile_sample_size_convergence.png")
    print(json.dumps({"cpu_events": cpu_events, "cuda_events": cuda_events, "sample_sizes": sample_sizes, "figures": 3 * len(sample_sizes) + 1, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
