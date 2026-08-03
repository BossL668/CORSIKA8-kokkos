#!/usr/bin/env python3
"""Summarize fixed-energy proton/iron original-CPU versus CUDA validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import yaml  # noqa: E402


BACKENDS = ("proposal", "cuda")
COLORS = {"proposal": "#1f77b4", "cuda": "#d62728"}
LABELS = {
    "proposal": "Original CPU",
    "cuda": "CUDA full acceleration",
}
FEATURES = (
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
    parser.add_argument("--proton-root", type=Path, required=True)
    parser.add_argument("--iron-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_dataset(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_shower = root / "per_shower_observables.csv"
    manifest_path = root / "run_manifest.json"
    if not per_shower.is_file():
        raise FileNotFoundError(per_shower)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    frame = pd.read_csv(per_shower)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for backend in BACKENDS:
        count = int((frame["backend"] == backend).sum())
        expected = int(manifest["configuration"]["events_per_backend"])
        if count != expected:
            raise ValueError(
                f"{root}: {backend} contains {count} showers, expected "
                f"{expected}"
            )
    return frame, manifest


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


def common_bins(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    support = np.concatenate((first, second))
    bins = np.histogram_bin_edges(support, bins="fd")
    if bins.size < 12:
        low = float(np.min(support))
        high = float(np.max(support))
        if low == high:
            width = max(abs(low), 1.0) * 1.0e-6
            low -= width
            high += width
        bins = np.linspace(low, high, 21)
    return bins


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


def shared_physics_label(
    manifests: dict[str, dict[str, Any]]
) -> str:
    fields = (
        "energy_GeV",
        "zenith_deg",
        "azimuth_deg",
        "em_cut_GeV",
        "em_thinning",
        "maximum_weight",
    )
    configurations = [
        manifest["configuration"] for manifest in manifests.values()
    ]
    reference = configurations[0]
    for configuration in configurations[1:]:
        for field in fields:
            if configuration[field] != reference[field]:
                raise ValueError(
                    "proton and iron plotting configurations differ for "
                    f"{field}: {reference[field]!r} versus "
                    f"{configuration[field]!r}"
                )
    direction = (
        "vertical"
        if abs(float(reference["zenith_deg"])) < 1.0e-12
        else (
            f"zenith {float(reference['zenith_deg']):g}°, "
            f"azimuth {float(reference['azimuth_deg']):g}°"
        )
    )
    return (
        f"{format_energy(float(reference['energy_GeV']))} {direction} showers, "
        f"emthin = {float(reference['em_thinning']):.0e}"
    )


def summarize_metric(
    frame: pd.DataFrame, primary: str, metric: str
) -> dict[str, Any]:
    cpu = finite_values(frame, "proposal", metric)
    cuda = finite_values(frame, "cuda", metric)
    cpu_mean = float(np.mean(cpu))
    cuda_mean = float(np.mean(cuda))
    denominator = max(abs(cpu_mean), np.finfo(np.float64).tiny)
    ks = stats.ks_2samp(cpu, cuda, method="exact")
    welch = stats.ttest_ind(cpu, cuda, equal_var=False)
    brown_forsythe = stats.levene(cpu, cuda, center="median")
    combined_standard_error = math.sqrt(
        float(np.var(cpu, ddof=1)) / cpu.size
        + float(np.var(cuda, ddof=1)) / cuda.size
    )
    return {
        "primary": primary,
        "metric": metric,
        "events_cpu": int(cpu.size),
        "events_cuda": int(cuda.size),
        "cpu_mean": cpu_mean,
        "cpu_standard_deviation": float(np.std(cpu, ddof=1)),
        "cuda_mean": cuda_mean,
        "cuda_standard_deviation": float(np.std(cuda, ddof=1)),
        "relative_mean_shift": (cuda_mean - cpu_mean) / denominator,
        "absolute_mean_z_score": (
            abs(cuda_mean - cpu_mean)
            / max(combined_standard_error, np.finfo(np.float64).tiny)
        ),
        "KS_statistic": float(ks.statistic),
        "KS_pvalue": float(ks.pvalue),
        "Welch_pvalue": float(welch.pvalue),
        "Brown_Forsythe_pvalue": float(brown_forsythe.pvalue),
    }


def read_internal_timing_seconds(root: Path, backend: str) -> float:
    if backend == "proposal":
        timing_paths = sorted(
            root.glob("proposal_shard_*/simulation_timing/summary.yaml")
        )
        if not timing_paths:
            timing_paths = [root / "proposal" / "simulation_timing" / "summary.yaml"]
    else:
        timing_paths = [root / "cuda" / "simulation_timing" / "summary.yaml"]
    if not timing_paths or any(not path.is_file() for path in timing_paths):
        missing = [str(path) for path in timing_paths if not path.is_file()]
        raise FileNotFoundError(
            "missing simulation timing summary: " + ", ".join(missing)
        )
    milliseconds = 0.0
    for path in timing_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid timing payload: {path}")
        for shower, record in payload.items():
            if not isinstance(record, dict) or "wall_time_ms" not in record:
                raise ValueError(
                    f"invalid shower timing record {shower!r} in {path}"
                )
            milliseconds += float(record["wall_time_ms"])
    return milliseconds / 1000.0


def read_per_shower_performance(
    root: Path, backend: str, primary: str
) -> list[dict[str, Any]]:
    if backend == "proposal":
        directories = sorted(
            path
            for path in root.glob("proposal_shard_*")
            if path.is_dir()
        )
        if not directories:
            directories = [root / "proposal"]
    else:
        directories = [root / "cuda"]
    rows: list[dict[str, Any]] = []
    global_shower = 0
    for directory in directories:
        timing_path = directory / "simulation_timing" / "summary.yaml"
        radio_path = directory / "CoREAS" / "summary.yaml"
        gpu_path = directory / "gpu_em" / "summary.yaml"
        if not timing_path.is_file():
            raise FileNotFoundError(timing_path)
        if not radio_path.is_file():
            raise FileNotFoundError(radio_path)
        timings = yaml.safe_load(timing_path.read_text(encoding="utf-8"))
        radio = yaml.safe_load(radio_path.read_text(encoding="utf-8"))
        gpu = (
            yaml.safe_load(gpu_path.read_text(encoding="utf-8"))
            if backend == "cuda"
            else None
        )
        if not isinstance(timings, dict) or not isinstance(radio, dict):
            raise ValueError(f"invalid timing/radio summary under {directory}")
        if set(timings) != set(radio):
            raise ValueError(
                f"timing and CoREAS shower IDs differ under {directory}"
            )
        if backend == "cuda" and (
            not isinstance(gpu, dict) or set(timings) != set(gpu)
        ):
            raise ValueError(
                f"timing and CUDA summary shower IDs differ under {directory}"
            )
        shower_keys = sorted(
            timings,
            key=lambda key: int(str(key).removeprefix("shower_")),
        )
        for local_index, shower_key in enumerate(shower_keys):
            timing = timings[shower_key]
            radio_record = radio[shower_key]
            scalar_radio_tracks = int(radio_record["segment_count"])
            resident_radio_tracks = (
                int(gpu[shower_key]["statistics"]["radio_tracks"])
                if gpu is not None
                else 0
            )
            rows.append(
                {
                    "primary": primary,
                    "backend": backend,
                    "global_shower": global_shower,
                    "source_directory": str(directory.resolve()),
                    "local_shower": local_index,
                    "internal_wall_seconds": (
                        float(timing["wall_time_ms"]) / 1000.0
                    ),
                    "scalar_radio_segment_count": scalar_radio_tracks,
                    "resident_gpu_radio_segment_count": (
                        resident_radio_tracks
                    ),
                    "coreas_segment_count": (
                        scalar_radio_tracks + resident_radio_tracks
                    ),
                }
            )
            global_shower += 1
    return rows


def summarize_performance_scaling(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (primary, backend), group in frame.groupby(
        ["primary", "backend"], sort=False
    ):
        segment_count = group["coreas_segment_count"].to_numpy(
            dtype=np.float64
        )
        runtime = group["internal_wall_seconds"].to_numpy(dtype=np.float64)
        if (
            segment_count.size < 2
            or np.any(segment_count <= 0.0)
            or np.any(runtime <= 0.0)
        ):
            raise ValueError(
                f"invalid runtime scaling sample for {primary}/{backend}"
            )
        regression = stats.linregress(
            np.log10(segment_count), np.log10(runtime)
        )
        rows.append(
            {
                "primary": primary,
                "backend": backend,
                "showers": int(segment_count.size),
                "median_internal_wall_seconds": float(np.median(runtime)),
                "median_coreas_segment_count": float(
                    np.median(segment_count)
                ),
                "median_microseconds_per_coreas_segment": float(
                    1.0e6 * np.median(runtime / segment_count)
                ),
                "pearson_runtime_vs_segment_count": float(
                    stats.pearsonr(segment_count, runtime).statistic
                ),
                "spearman_runtime_vs_segment_count": float(
                    stats.spearmanr(segment_count, runtime).statistic
                ),
                "log_log_power_exponent": float(regression.slope),
                "log_log_r_squared": float(regression.rvalue**2),
            }
        )
    return rows


def summarize_timing(
    root: Path, primary: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    external = manifest["external_wall_seconds"]
    proposal_shards = np.asarray(
        external["proposal_shards"], dtype=np.float64
    )
    events = int(manifest["configuration"]["events_per_backend"])
    cpu_serial_work = float(np.sum(proposal_shards))
    cpu_parallel_wall = float(external["proposal_group"])
    cuda_wall = float(external["cuda"])
    cpu_internal = read_internal_timing_seconds(root, "proposal")
    cuda_internal = read_internal_timing_seconds(root, "cuda")
    return {
        "primary": primary,
        "events_per_backend": events,
        "cpu_shards": int(proposal_shards.size),
        "cpu_parallelism": int(
            manifest["configuration"]["proposal_parallelism"]
        ),
        "cpu_sum_of_shard_wall_seconds": cpu_serial_work,
        "cpu_parallel_group_wall_seconds": cpu_parallel_wall,
        "cuda_wall_seconds": cuda_wall,
        "cpu_internal_shower_seconds": cpu_internal,
        "cuda_internal_shower_seconds": cuda_internal,
        "cpu_sum_of_shard_wall_seconds_per_event": (
            cpu_serial_work / events
        ),
        "cpu_parallel_group_wall_seconds_per_event": (
            cpu_parallel_wall / events
        ),
        "cuda_wall_seconds_per_event": cuda_wall / events,
        "cpu_internal_shower_seconds_per_event": cpu_internal / events,
        "cuda_internal_shower_seconds_per_event": cuda_internal / events,
        "speedup_cuda_vs_sum_of_cpu_shard_walls": (
            cpu_serial_work / cuda_wall
        ),
        "speedup_cuda_vs_cpu_internal_shower_time": (
            cpu_internal / cuda_internal
        ),
        "throughput_gain_cuda_vs_parallel_cpu_group": (
            cpu_parallel_wall / cuda_wall
        ),
    }


def plot_scalar_distributions(
    datasets: dict[str, pd.DataFrame],
    summaries: dict[tuple[str, str], dict[str, Any]],
    output: Path,
    physics_label: str,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17.5, 8.0))
    for row, (primary, frame) in enumerate(datasets.items()):
        for column, (metric, label) in enumerate(FEATURES):
            axis = axes[row, column]
            arrays = {
                backend: finite_values(frame, backend, metric)
                for backend in BACKENDS
            }
            bins = common_bins(arrays["proposal"], arrays["cuda"])
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
            summary = summaries[(primary, metric)]
            axis.text(
                0.03,
                0.96,
                (
                    rf"$\Delta\mu/\mu={100.0 * summary['relative_mean_shift']:+.2f}\%$"
                    "\n"
                    rf"$D_{{KS}}={summary['KS_statistic']:.3f}$, "
                    rf"$p={summary['KS_pvalue']:.3g}$"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.85,
                },
            )
            axis.set_xlabel(label)
            axis.set_ylabel("probability density")
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_title(f"{primary}\n{label}")
            else:
                axis.set_title(label)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        f"{physics_label} — original CPU versus fully accelerated CUDA",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_timing(timings: list[dict[str, Any]], output: Path) -> None:
    primaries = [record["primary"] for record in timings]
    cpu = np.asarray(
        [
            record["cpu_sum_of_shard_wall_seconds_per_event"]
            for record in timings
        ],
        dtype=np.float64,
    )
    cuda = np.asarray(
        [record["cuda_wall_seconds_per_event"] for record in timings],
        dtype=np.float64,
    )
    positions = np.arange(len(primaries), dtype=np.float64)
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.bar(
        positions - width / 2.0,
        cpu,
        width,
        color=COLORS["proposal"],
        label="Original CPU: sum of shard walls/event",
    )
    axis.bar(
        positions + width / 2.0,
        cuda,
        width,
        color=COLORS["cuda"],
        label="CUDA full acceleration/event",
    )
    for index, record in enumerate(timings):
        axis.text(
            positions[index],
            max(cpu[index], cuda[index]) * 1.08,
            (
                f"{record['speedup_cuda_vs_sum_of_cpu_shard_walls']:.2f}"
                r"$\times$"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_yscale("log")
    axis.set_xticks(positions, primaries)
    axis.set_ylabel("external wall time per shower [s]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title(
        "Runtime comparison (CUDA uses all enabled acceleration paths)"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_runtime_scaling(frame: pd.DataFrame, output: Path) -> None:
    primaries = list(dict.fromkeys(frame["primary"].tolist()))
    figure, axes = plt.subplots(
        1, len(primaries), figsize=(7.0 * len(primaries), 5.2), squeeze=False
    )
    for axis, primary in zip(axes.flat, primaries):
        for backend in BACKENDS:
            selected = frame.loc[
                (frame["primary"] == primary)
                & (frame["backend"] == backend)
            ]
            segment_count = selected["coreas_segment_count"].to_numpy(
                dtype=np.float64
            )
            runtime = selected["internal_wall_seconds"].to_numpy(
                dtype=np.float64
            )
            axis.scatter(
                segment_count,
                runtime,
                s=24,
                alpha=0.62,
                edgecolors="none",
                color=COLORS[backend],
                label=LABELS[backend],
            )
            regression = stats.linregress(
                np.log10(segment_count), np.log10(runtime)
            )
            support = np.geomspace(
                float(np.min(segment_count)),
                float(np.max(segment_count)),
                100,
            )
            prediction = 10.0 ** (
                regression.intercept
                + regression.slope * np.log10(support)
            )
            axis.plot(
                support,
                prediction,
                linewidth=1.5,
                color=COLORS[backend],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("CoREAS electron/positron track segments per shower")
        axis.set_ylabel("internal shower wall time [s]")
        axis.set_title(primary)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Per-shower runtime scaling with radio-track workload",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    roots = {
        "proton": args.proton_root.resolve(),
        r"$^{56}$Fe": args.iron_root.resolve(),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for primary, root in roots.items():
        frame, manifest = load_dataset(root)
        datasets[primary] = frame
        manifests[primary] = manifest
        for metric, _ in FEATURES:
            row = summarize_metric(frame, primary, metric)
            summaries[(primary, metric)] = row
            summary_rows.append(row)
        timings.append(summarize_timing(root, primary, manifest))
        for backend in BACKENDS:
            performance_rows.extend(
                read_per_shower_performance(root, backend, primary)
            )
    physics_label = shared_physics_label(manifests)
    plot_scalar_distributions(
        datasets,
        summaries,
        output / "proton_iron_shower_scalar_distributions.png",
        physics_label,
    )
    plot_timing(timings, output / "proton_iron_runtime_comparison.png")
    performance_frame = pd.DataFrame(performance_rows)
    performance_scaling = summarize_performance_scaling(performance_frame)
    plot_runtime_scaling(
        performance_frame,
        output / "proton_iron_runtime_vs_radio_tracks.png",
    )
    pd.DataFrame(summary_rows).to_csv(
        output / "proton_iron_feature_summary.csv", index=False
    )
    pd.DataFrame(timings).to_csv(
        output / "proton_iron_timing_summary.csv", index=False
    )
    performance_frame.to_csv(
        output / "per_shower_runtime_and_radio_tracks.csv", index=False
    )
    pd.DataFrame(performance_scaling).to_csv(
        output / "runtime_scaling_summary.csv", index=False
    )
    payload = {
        "interpretation": {
            "statistics": (
                "CPU and CUDA use independent seed ranges. Tests therefore "
                "compare shower-level distributions rather than event pairs."
            ),
            "timing": (
                "The CPU serial-work estimate is the sum of one-event shard "
                "wall times measured while the shards ran concurrently. The "
                "parallel-group wall and internal shower timers are retained "
                "separately; no CPU and CUDA run overlapped."
            ),
        },
        "roots": {key: str(value) for key, value in roots.items()},
        "physics_label": physics_label,
        "feature_summaries": summary_rows,
        "timings": timings,
        "runtime_scaling": performance_scaling,
        "physics_status": {
            key: manifest["physics_status"]
            for key, manifest in manifests.items()
        },
    }
    (output / "proton_iron_validation_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "primaries": list(roots),
                "features": len(summary_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
