#!/usr/bin/env python3
"""Compare CPU/CUDA geomagnetic pulse amplitude and width versus radius.

Each point first aggregates antennas at the same radius within one shower.
The plotted bands are the 16th--84th percentiles across independent showers.
CUDA/CPU ratios use an independent two-sample, pointwise bootstrap whose
resampling unit is a complete shower, never an individual antenna.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_geomagnetic_pulse_distributions import (  # noqa: E402
    ALGORITHMS,
    BACKENDS,
    BACKEND_COLORS,
    BACKEND_LABELS,
    DEFAULT_PULSE_ROOT,
    aggregate_by_shower,
    apply_reference_width_filter,
    backend_output_directories,
    extract_antenna_rows,
    finite,
    geometric_mean,
    load_reference_apis,
    load_run_configuration,
    write_csv,
)


METRICS = (
    (
        "amplitude",
        "geomagnetic_amplitude_geomean_V_per_m",
        r"$|E_{\mathbf{v}\times\mathbf{B}}|$ peak [V m$^{-1}$]",
        True,
    ),
    (
        "width",
        "pulse_width_mean_ns",
        "Pulse width [ns]",
        False,
    ),
)


def load_records(
    *,
    output_directories: list[Path],
    algorithm: str,
    read_radio_records: Callable[
        ..., tuple[list[dict[str, Any]], np.ndarray]
    ],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    shower_offset = 0
    for output_directory in output_directories:
        shard_records, _ = read_radio_records(output_directory, algorithm)
        local_showers = sorted(
            {int(record["shower"]) for record in shard_records}
        )
        if local_showers and local_showers != list(
            range(local_showers[-1] + 1)
        ):
            raise ValueError(
                "radio shard shower IDs are not contiguous: "
                f"{output_directory}"
            )
        for record in shard_records:
            item = dict(record)
            item["shower"] = int(record["shower"]) + shower_offset
            records.append(item)
        if local_showers:
            shower_offset += local_showers[-1] + 1
    if not records:
        raise ValueError(
            f"no {algorithm} records in {output_directories}"
        )
    return records


def common_radii(
    records_by_stream: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    minimum_m: float,
    maximum_m: float,
) -> list[float]:
    stream_radii = [
        {
            round(float(record["radius_m"]), 6)
            for record in records
            if (
                float(record["radius_m"]) > 0.0
                and minimum_m <= float(record["radius_m"]) <= maximum_m
            )
        }
        for records in records_by_stream.values()
    ]
    common = sorted(set.intersection(*stream_radii))
    if len(common) < 2:
        raise ValueError(
            "fewer than two common non-zero antenna radii are available"
        )
    return common


def clustered_available_radii(
    records: list[dict[str, Any]],
    *,
    minimum_m: float,
    maximum_m: float,
    tolerance_m: float = 1.e-4,
) -> list[float]:
    """Collapse text-rounding variants of one nominal antenna radius."""
    raw = sorted(
        {
            float(record["radius_m"])
            for record in records
            if (
                float(record["radius_m"]) > 0.0
                and minimum_m - tolerance_m
                <= float(record["radius_m"])
                <= maximum_m + tolerance_m
            )
        }
    )
    clusters: list[list[float]] = []
    for value in raw:
        if (
            not clusters
            or value - float(np.mean(clusters[-1])) > tolerance_m
        ):
            clusters.append([value])
        else:
            clusters[-1].append(value)
    result: list[float] = []
    for cluster in clusters:
        representative = float(np.median(cluster))
        nearest_integer = round(representative)
        if abs(representative - nearest_integer) <= tolerance_m:
            representative = float(nearest_integer)
        result.append(round(representative, 6))
    return result


def central_statistic(metric: str, values: np.ndarray) -> float:
    if metric == "amplitude":
        return geometric_mean(values)
    if metric == "width":
        return float(np.mean(values))
    raise ValueError(metric)


def bootstrap_ratio(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    metric: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        cpu = reference[
            rng.integers(0, reference.size, size=reference.size)
        ]
        cuda = candidate[
            rng.integers(0, candidate.size, size=candidate.size)
        ]
        denominator = central_statistic(metric, cpu)
        ratios[index] = (
            central_statistic(metric, cuda) / denominator
            if denominator > 0.0
            else math.nan
        )
    ratios = ratios[np.isfinite(ratios)]
    return (
        central_statistic(metric, candidate)
        / central_statistic(metric, reference),
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )


def values_at(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    algorithm: str,
    radius_m: float,
    column: str,
) -> np.ndarray:
    return finite(
        (
            row[column]
            for row in rows
            if (
                row["backend"] == backend
                and row["algorithm"] == algorithm
                and np.isclose(
                    float(row["radius_m"]),
                    radius_m,
                    rtol=0.0,
                    atol=1.e-5,
                )
            )
        ),
        positive=True,
    )


def summarize_curves(
    rows: list[dict[str, Any]],
    *,
    radii: list[float],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    point_index = 0
    for algorithm in ALGORITHMS:
        for metric, column, _, _ in METRICS:
            for radius_m in radii:
                arrays = {
                    backend: values_at(
                        rows,
                        backend=backend,
                        algorithm=algorithm,
                        radius_m=radius_m,
                        column=column,
                    )
                    for backend in BACKENDS
                }
                if any(array.size < 2 for array in arrays.values()):
                    continue
                ratio, ratio_low, ratio_high = bootstrap_ratio(
                    arrays["legacy_proposal"],
                    arrays["cuda"],
                    metric=metric,
                    repetitions=repetitions,
                    seed=seed + point_index,
                )
                point_index += 1
                row: dict[str, Any] = {
                    "algorithm": algorithm,
                    "metric": metric,
                    "radius_m": radius_m,
                    "cuda_over_cpu": ratio,
                    "ratio_ci95_low": ratio_low,
                    "ratio_ci95_high": ratio_high,
                }
                for backend, values in arrays.items():
                    prefix = (
                        "cpu" if backend == "legacy_proposal" else "cuda"
                    )
                    row[f"{prefix}_showers"] = int(values.size)
                    row[f"{prefix}_central"] = central_statistic(
                        metric, values
                    )
                    row[f"{prefix}_q16"] = float(
                        np.quantile(values, 0.16)
                    )
                    row[f"{prefix}_q84"] = float(
                        np.quantile(values, 0.84)
                    )
                    row[f"{prefix}_unique_values_1e-6"] = int(
                        np.unique(np.round(values, 6)).size
                    )
                row["diagnostic_only"] = bool(
                    metric == "width"
                    and min(
                        int(row["cpu_unique_values_1e-6"]),
                        int(row["cuda_unique_values_1e-6"]),
                    )
                    <= 2
                )
                summaries.append(row)
    return summaries


def metric_rows(
    summaries: list[dict[str, Any]], algorithm: str, metric: str
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in summaries
            if row["algorithm"] == algorithm and row["metric"] == metric
        ),
        key=lambda row: row["radius_m"],
    )


def read_feature_csv(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "shower",
        "observer",
        "antenna_count",
        "amplitude_valid_count",
        "pulse_width_raw_valid_count",
        "pulse_width_filter_pass_count",
    }
    text_fields = {"backend", "algorithm", "pulse_method"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_fields:
                    row[key] = value
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def plot_curves(
    summaries: list[dict[str, Any]],
    *,
    output: Path,
    title_suffix: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (metric, _, ylabel, logarithmic) in enumerate(
            METRICS
        ):
            axis = axes[row_index, column_index]
            selected = metric_rows(summaries, algorithm, metric)
            radii = np.asarray(
                [row["radius_m"] for row in selected], dtype=np.float64
            )
            for backend in BACKENDS:
                prefix = (
                    "cpu" if backend == "legacy_proposal" else "cuda"
                )
                centers = np.asarray(
                    [row[f"{prefix}_central"] for row in selected]
                )
                q16 = np.asarray(
                    [row[f"{prefix}_q16"] for row in selected]
                )
                q84 = np.asarray(
                    [row[f"{prefix}_q84"] for row in selected]
                )
                axis.plot(
                    radii,
                    centers,
                    "o-",
                    color=BACKEND_COLORS[backend],
                    lw=1.8,
                    ms=4,
                    label=BACKEND_LABELS[backend],
                )
                axis.fill_between(
                    radii,
                    q16,
                    q84,
                    color=BACKEND_COLORS[backend],
                    alpha=0.16,
                    linewidth=0.0,
                )
            diagnostic = np.asarray(
                [bool(row["diagnostic_only"]) for row in selected]
            )
            if np.any(diagnostic):
                axis.scatter(
                    radii[diagnostic],
                    np.asarray(
                        [row["cuda_central"] for row in selected]
                    )[diagnostic],
                    marker="x",
                    s=58,
                    linewidths=1.8,
                    color="black",
                    zorder=5,
                    label="fitter-floor diagnostic",
                )
            axis.set_xscale("log")
            if logarithmic:
                axis.set_yscale("log")
            axis.set_xlabel("Core distance $r$ [m]")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{algorithm}: {metric}")
            axis.grid(alpha=0.24, which="both")
            axis.legend(frameon=False)
    figure.suptitle(
        "Original CPU vs CUDA geomagnetic pulse radial curves\n"
        + title_suffix,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_ratios(
    summaries: list[dict[str, Any]],
    *,
    output: Path,
    title_suffix: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for row_index, algorithm in enumerate(ALGORITHMS):
        for column_index, (metric, _, _, _) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            selected = metric_rows(summaries, algorithm, metric)
            radii = np.asarray([row["radius_m"] for row in selected])
            ratio = np.asarray([row["cuda_over_cpu"] for row in selected])
            low = np.asarray([row["ratio_ci95_low"] for row in selected])
            high = np.asarray([row["ratio_ci95_high"] for row in selected])
            axis.axhspan(0.9, 1.1, color="0.75", alpha=0.22)
            axis.axhline(1.0, color="0.25", ls="--", lw=1.1)
            axis.plot(
                radii,
                ratio,
                "o-",
                color="#6a1b9a",
                lw=1.8,
                ms=4,
            )
            axis.fill_between(
                radii,
                low,
                high,
                color="#6a1b9a",
                alpha=0.18,
                linewidth=0.0,
            )
            diagnostic = np.asarray(
                [bool(row["diagnostic_only"]) for row in selected]
            )
            if np.any(diagnostic):
                axis.scatter(
                    radii[diagnostic],
                    ratio[diagnostic],
                    marker="x",
                    s=58,
                    linewidths=1.8,
                    color="black",
                    zorder=5,
                    label="fitter-floor diagnostic",
                )
                axis.legend(frameon=False, fontsize=8.5)
            axis.set_xscale("log")
            axis.set_xlabel("Core distance $r$ [m]")
            axis.set_ylabel("CUDA / original CPU")
            axis.set_title(f"{algorithm}: {metric}")
            axis.grid(alpha=0.24, which="both")
    figure.suptitle(
        "Pointwise CPU/CUDA radial ratios (95% shower bootstrap)\n"
        + title_suffix,
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_summary_markdown(
    *,
    path: Path,
    dataset: Path,
    title_suffix: str,
    radii: list[float],
    antenna_rows: list[dict[str, Any]],
    shower_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    repetitions: int,
) -> None:
    lines = [
        "# CPU/CUDA 地磁脉冲随核心距变化",
        "",
        f"- 数据集：`{dataset}`。",
        f"- 配置：{title_suffix}。",
        (
            "- 半径："
            + ", ".join(f"{radius:g}" for radius in radii)
            + " m；不在天线文件中的半径没有插值。"
        ),
        (
            "- 地磁投影、主峰和宽度定义复用 `pulse_analysis_modular`；"
            "同一 shower、同一半径的方位天线先聚合，统计单位是 shower。"
        ),
        (
            "- 曲线中心：振幅为 shower 分布的几何均值，宽度为算术均值；"
            "阴影为 shower 的 16%--84% 分位区间。"
        ),
        (
            f"- 比值阴影为 {repetitions} 次独立两样本、逐半径 shower "
            "bootstrap 的 95% 区间；它是 pointwise 区间，不是整条曲线的"
            " simultaneous confidence band。"
        ),
        (
            f"- 提取了 {len(antenna_rows)} 条逐天线记录和 "
            f"{len(shower_rows)} 条逐 shower-半径记录。"
        ),
    ]
    diagnostic = [
        row for row in summaries if bool(row["diagnostic_only"])
    ]
    if diagnostic:
        labels = ", ".join(
            f"{row['algorithm']} {row['metric']}@{row['radius_m']:g} m"
            for row in diagnostic
        )
        lines.append(
            "- 黑色叉号是拟合器最小窗宽退化点，只作诊断、不能作为"
            f" CPU/CUDA 物理等价证据：{labels}。"
        )
    lines.extend(
        [
            "",
            "![径向主曲线](geomagnetic_radial_curves.png)",
            "",
            "![CUDA/CPU 径向比值](geomagnetic_radial_ratios.png)",
            "",
            "## 数值表",
            "",
            (
                "| 算法 | 指标 | r [m] | CPU | CUDA | "
                "CUDA/CPU [95% CI] | N CPU/CUDA | 状态 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['algorithm']} | {row['metric']} | "
            f"{row['radius_m']:g} | {row['cpu_central']:.6g} | "
            f"{row['cuda_central']:.6g} | "
            f"{row['cuda_over_cpu']:.4f} "
            f"[{row['ratio_ci95_low']:.4f}, "
            f"{row['ratio_ci95_high']:.4f}] | "
            f"{row['cpu_showers']}/{row['cuda_showers']} | "
            f"{'diagnostic_only' if row['diagnostic_only'] else 'usable'} |"
        )
    lines.extend(
        [
            "",
            "## 可复现文件",
            "",
            "- `per_antenna_radial_features.csv`：逐天线脉冲参数。",
            "- `per_shower_radius_features.csv`：实际统计单位。",
            "- `radial_summary.csv/json`：曲线点、分位数和比值区间。",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pulse-analysis-root", type=Path, default=DEFAULT_PULSE_ROOT
    )
    parser.add_argument("--minimum-radius-m", type=float, default=1.0)
    parser.add_argument("--maximum-radius-m", type=float, default=600.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help=(
            "Reuse per_antenna_radial_features.csv and "
            "per_shower_radius_features.csv already in --output."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("--bootstrap-repetitions must be at least 100")
    if not (
        0.0 <= args.minimum_radius_m < args.maximum_radius_m
        and math.isfinite(args.maximum_radius_m)
    ):
        raise ValueError("invalid radius interval")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (
        build_polarization_basis,
        analyze_pulse_parameters,
        read_radio_records,
        band_limited_waveform,
        robust_pulse_width_mask,
        PulseWidthFilterConfig,
    ) = load_reference_apis(args.pulse_analysis_root.resolve())
    run_config = load_run_configuration(dataset)
    basis = build_polarization_basis(
        run_config["theta_deg"],
        run_config["phi_deg"],
        run_config["magnetic_field_T"],
    )
    sources = {
        backend: backend_output_directories(dataset, backend)
        for backend in BACKENDS
    }
    antenna_path = output / "per_antenna_radial_features.csv"
    shower_path = output / "per_shower_radius_features.csv"
    if args.reuse_extracted:
        if not antenna_path.is_file() or not shower_path.is_file():
            raise FileNotFoundError(
                "--reuse-extracted requires existing antenna and shower CSVs"
            )
        antenna_rows = read_feature_csv(antenna_path)
        shower_rows = read_feature_csv(shower_path)
        radii = sorted(
            {float(row["radius_m"]) for row in shower_rows}
        )
        print(
            f"Reused {len(antenna_rows)} antenna and "
            f"{len(shower_rows)} shower-radius rows",
            flush=True,
        )
    else:
        antenna_rows = []
        radii: list[float] | None = None
        # Radio waveforms dominate analysis memory.  Load, analyze and release
        # one backend/algorithm stream at a time.
        for backend in BACKENDS:
            for algorithm in ALGORITHMS:
                records = load_records(
                    output_directories=sources[backend],
                    algorithm=algorithm,
                    read_radio_records=read_radio_records,
                )
                stream_radii = clustered_available_radii(
                    records,
                    minimum_m=args.minimum_radius_m,
                    maximum_m=args.maximum_radius_m,
                )
                if radii is None:
                    radii = stream_radii
                elif stream_radii != radii:
                    raise ValueError(
                        f"observer radii differ for {backend}/{algorithm}: "
                        f"{stream_radii} != {radii}"
                    )
                if len(radii) < 2:
                    raise ValueError(
                        "fewer than two common non-zero antenna radii are "
                        "available"
                    )
                for radius_m in radii:
                    antenna_rows.extend(
                        extract_antenna_rows(
                            records=records,
                            backend=backend,
                            algorithm=algorithm,
                            radius_m=radius_m,
                            yprime=np.asarray(
                                basis["yprime"], dtype=np.float64
                            ),
                            analyze_pulse_parameters=(
                                analyze_pulse_parameters
                            ),
                            band_limited_waveform=band_limited_waveform,
                            band_MHz=None,
                            analysis_sampling_rate_GHz=None,
                        )
                    )
                print(
                    f"{backend}/{algorithm}: {len(records)} waveforms, "
                    f"{len(radii)} radii",
                    flush=True,
                )
                del records
        if radii is None:
            raise ValueError("no radio streams were loaded")
        width_filter_config = PulseWidthFilterConfig()
        apply_reference_width_filter(
            antenna_rows,
            robust_pulse_width_mask=robust_pulse_width_mask,
            filter_config=width_filter_config,
        )
        shower_rows = aggregate_by_shower(antenna_rows)
    summaries = summarize_curves(
        shower_rows,
        radii=radii,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    if not summaries:
        raise ValueError("no radial curve points could be summarized")
    manifest_config = run_config["manifest"]["configuration"]
    primary = (
        f"Z={manifest_config['primary_Z']}, A={manifest_config['primary_A']}"
        if manifest_config.get("primary_Z") is not None
        else f"PDG {manifest_config.get('primary_pdg')}"
    )
    title_suffix = (
        f"{manifest_config['energy_GeV']:g} GeV {primary}, "
        f"zenith {manifest_config['zenith_deg']:g}°, "
        f"{manifest_config['combined_proposal_events']}+"
        f"{manifest_config['combined_cuda_events']} showers"
    )
    write_csv(output / "per_antenna_radial_features.csv", antenna_rows)
    write_csv(output / "per_shower_radius_features.csv", shower_rows)
    with (output / "radial_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "dataset": str(dataset),
        "configuration": manifest_config,
        "statistical_unit": "one shower after azimuth aggregation",
        "radii_m": radii,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_scope": "pointwise independent two-sample shower bootstrap",
        "geomagnetic_unit_vector_NWU": np.asarray(
            basis["yprime"], dtype=np.float64
        ).tolist(),
        "source_directories": {
            backend: [str(path) for path in paths]
            for backend, paths in sources.items()
        },
        "curves": summaries,
    }
    (output / "radial_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_curves(
        summaries,
        output=output / "geomagnetic_radial_curves.png",
        title_suffix=title_suffix,
    )
    plot_ratios(
        summaries,
        output=output / "geomagnetic_radial_ratios.png",
        title_suffix=title_suffix,
    )
    write_summary_markdown(
        path=output / "README.md",
        dataset=dataset,
        title_suffix=title_suffix,
        radii=radii,
        antenna_rows=antenna_rows,
        shower_rows=shower_rows,
        summaries=summaries,
        repetitions=args.bootstrap_repetitions,
    )
    print(f"Wrote radial comparison to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
