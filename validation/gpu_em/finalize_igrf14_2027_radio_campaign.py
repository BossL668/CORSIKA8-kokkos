#!/usr/bin/env python3
"""Wait for, validate, analyze, and plot the IGRF14/2027 radio campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pulse-project", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_logged(
    command: list[str],
    log: Path,
    environment: dict,
    *,
    working_directory: Path | None = None,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("command: " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=working_directory,
            check=True,
        )


def finite_radio(path: Path) -> dict:
    table = pq.read_table(path)
    arrays = [
        table[name].to_numpy(zero_copy_only=False)
        for name in ("Time", "Ex", "Ey", "Ez")
    ]
    return {
        "rows": table.num_rows,
        "nulls": int(sum(table[name].null_count for name in table.column_names)),
        "all_finite": bool(all(np.isfinite(array).all() for array in arrays)),
        "maximum_absolute_field_V_per_m": float(
            max(np.max(np.abs(array)) for array in arrays[1:])
        ),
    }


def validate_cuda_event(event: dict, configuration: dict) -> dict:
    root = Path(event["cuda_library"])
    cpu_root = Path(event["cpu_library"])
    summary = read_yaml(root / "summary.yaml")
    primary = read_yaml(root / "primary" / "summary.yaml")["shower_0"]
    cpu_summary = read_yaml(cpu_root / "summary.yaml")
    cpu_primary = read_yaml(cpu_root / "primary" / "summary.yaml")["shower_0"]
    cpu_config = read_yaml(cpu_root / "config.yaml")
    gpu_config = read_yaml(root / "gpu_em" / "config.yaml")
    gpu_summary = read_yaml(root / "gpu_em" / "summary.yaml")["shower_0"]
    timing = read_yaml(root / "simulation_timing" / "summary.yaml")["shower_0"]
    statistics = gpu_summary["statistics"]
    radio = finite_radio(root / "CoREAS" / "observers.parquet")
    cpu_radio = finite_radio(cpu_root / "CoREAS" / "observers.parquet")
    cpu_accepted = (
        int(cpu_summary.get("showers", -1)) == 1
        and int(cpu_summary.get("seed", -1)) == int(event["seed"])
        and int(cpu_primary.get("pdg", -1)) == int(configuration["primary_pdg"])
        and math.isclose(
            float(cpu_primary.get("total_energy", 0.0)),
            float(configuration["energy_GeV"]),
            rel_tol=1.0e-12,
        )
        and cpu_radio["nulls"] == 0
        and cpu_radio["all_finite"]
    )
    accepted = (
        cpu_accepted
        and int(summary.get("showers", -1)) == 1
        and int(summary.get("seed", -1)) == int(event["seed"])
        and int(primary.get("pdg", -1)) == int(configuration["primary_pdg"])
        and math.isclose(
            float(primary.get("total_energy", 0.0)),
            float(configuration["energy_GeV"]),
            rel_tol=1.0e-12,
        )
        and gpu_summary.get("complete") is True
        and gpu_summary.get("status") == "complete"
        and timing.get("closed") is True
        and gpu_config["environment"]["geomagnetic_model"]
        == configuration["geomagnetic_model"]
        and float(gpu_config["environment"]["geomagnetic_year"])
        == float(configuration["geomagnetic_year"])
        and int(statistics["queue_overflows"]) == 0
        and int(statistics["cpu_memory_spill_particles"]) == 0
        and radio["rows"] == cpu_radio["rows"]
        and radio["nulls"] == 0
        and radio["all_finite"]
    )
    return {
        "index": int(event["index"]),
        "seed": int(event["seed"]),
        "path": str(root),
        "cpu_path": str(cpu_root),
        "accepted": accepted,
        "cpu_accepted": cpu_accepted,
        "cpu_runtime_s": float(cpu_summary["runtime_raw"]),
        "cpu_command": str(cpu_config.get("args", "")),
        "runtime_s": float(summary["runtime_raw"]),
        "shower_wall_time_s": float(timing["wall_time_ms"]) / 1000.0,
        "geomagnetic_model": gpu_config["environment"]["geomagnetic_model"],
        "geomagnetic_year": float(gpu_config["environment"]["geomagnetic_year"]),
        "magnetic_field_T": gpu_config["environment"]["magnetic_field_T"],
        "queue_overflows": int(statistics["queue_overflows"]),
        "memory_spill_particles": int(statistics["cpu_memory_spill_particles"]),
        "coreas": radio,
        "cpu_coreas": cpu_radio,
    }


def energy_label(energy_GeV: float) -> str:
    if energy_GeV >= 1.0e9:
        return f"{energy_GeV / 1.0e9:g} EeV"
    if energy_GeV >= 1.0e6:
        return f"{energy_GeV / 1.0e6:g} PeV"
    if energy_GeV >= 1.0e3:
        return f"{energy_GeV / 1.0e3:g} TeV"
    return f"{energy_GeV:g} GeV"


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "sum_s": float(np.sum(array)),
        "mean_s": float(np.mean(array)),
        "median_s": float(np.median(array)),
        "minimum_s": float(np.min(array)),
        "maximum_s": float(np.max(array)),
    }


def runtime_histogram_edges(values: np.ndarray) -> np.ndarray:
    """Return stable, readable bin edges for a ten-event campaign."""
    if np.allclose(values, values[0]):
        half_width = max(abs(float(values[0])) * 0.025, 0.5)
        return np.linspace(values[0] - half_width, values[0] + half_width, 7)
    edge_count = len(np.histogram_bin_edges(values, bins="fd"))
    number_of_bins = min(8, max(5, edge_count - 1))
    return np.linspace(float(np.min(values)), float(np.max(values)), number_of_bins + 1)


def write_runtime_products(
    root: Path,
    events: list[dict],
    cpu_runtime: list[float],
    cuda_runtime: list[float],
    cuda_observed_runtime: list[float],
    configuration: dict,
) -> None:
    output = root / "runtime_analysis"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "single_event_runtimes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "backend",
                "event_index",
                "seed",
                "timing_definition",
                "runtime_seconds",
                "runtime_minutes",
                "runtime_hours",
                "top_level_observed_seconds",
            ],
        )
        writer.writeheader()
        groups = (
            (
                "CPU",
                cpu_runtime,
                cpu_runtime,
                "summary.yaml runtime_raw",
            ),
            (
                "CUDA",
                cuda_runtime,
                cuda_observed_runtime,
                "simulation_timing wall_time_ms",
            ),
        )
        for backend, values, observed_values, timing_definition in groups:
            for event, seconds, observed_seconds in zip(
                events, values, observed_values
            ):
                writer.writerow(
                    {
                        "backend": backend,
                        "event_index": int(event["index"]),
                        "seed": int(event["seed"]),
                        "timing_definition": timing_definition,
                        "runtime_seconds": seconds,
                        "runtime_minutes": seconds / 60.0,
                        "runtime_hours": seconds / 3600.0,
                        "top_level_observed_seconds": observed_seconds,
                    }
                )

    cpu_hours = np.asarray(cpu_runtime, dtype=float) / 3600.0
    cuda_minutes = np.asarray(cuda_runtime, dtype=float) / 60.0
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), constrained_layout=True)
    panels = (
        (axes[0], cpu_hours, "Original CPU", "Runtime per shower [h]", "#4472C4"),
        (axes[1], cuda_minutes, "CUDA (RTX 4060 Laptop)", "Runtime per shower [min]", "#ED7D31"),
    )
    for axis, values, title, xlabel, color in panels:
        axis.hist(
            values,
            bins=runtime_histogram_edges(values),
            color=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.9,
        )
        mean = float(np.mean(values))
        median = float(np.median(values))
        axis.axvline(mean, color="#B22222", linewidth=1.8, label="Mean")
        axis.axvline(
            median,
            color="#202020",
            linewidth=1.8,
            linestyle="--",
            label="Median",
        )
        unit = "h" if "[h]" in xlabel else "min"
        axis.text(
            0.97,
            0.95,
            "\n".join(
                (
                    f"N = {len(values)}",
                    f"Mean = {mean:.2f} {unit}",
                    f"Median = {median:.2f} {unit}",
                    f"Range = {np.min(values):.2f}–{np.max(values):.2f} {unit}",
                )
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.3,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#B0B0B0",
                "alpha": 0.92,
            },
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of showers")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        f"Active shower runtime: {energy_label(float(configuration['energy_GeV']))} "
        f"proton, zenith {float(configuration['zenith_deg']):g}°, "
        f"azimuth {float(configuration['azimuth_deg']):g}°, "
        f"EM thinning {float(configuration['em_thinning']):.0e}",
        fontsize=13,
    )
    figure.savefig(output / "cpu_cuda_runtime_histograms.png", dpi=220)
    plt.close(figure)


def write_markdown(root: Path, report: dict) -> None:
    configuration = report["configuration"]
    comparison = report["pulse_comparison"]
    amplitude = comparison["amplitude_ratio"]
    width = comparison["pulse_width_ratio"]
    paired_amplitude = comparison["paired_event_antenna"]["amplitude_ratio"]
    paired_width = comparison["paired_event_antenna"]["pulse_width_ratio"]
    cuda_time = report["runtime"]["cuda"]
    cpu_time = report["runtime"]["cpu"]
    lines = [
        f"# {configuration['geomagnetic_model']}/{float(configuration['geomagnetic_year']):g}："
        f"{energy_label(float(configuration['energy_GeV']))} 质子 CPU–CUDA CoREAS 比较",
        "",
        f"- 初级粒子：proton，{energy_label(float(configuration['energy_GeV']))}；",
        f"- 方向：zenith {float(configuration['zenith_deg']):g}°，"
        f"azimuth {float(configuration['azimuth_deg']):g}°；",
        f"- EM thinning：`{float(configuration['em_thinning']):.0e}`；",
        f"- 地磁场：{configuration['geomagnetic_model']}，"
        f"{float(configuration['geomagnetic_year']):g}，21CMA 站址；",
        f"- 样本：CPU {report['cuda_complete_count']} 个、CUDA "
        f"{report['cuda_complete_count']} 个，使用相同 seed；",
        "- 脉冲提取：`pulse_analysis_modular`，CoREAS `EyPrime`（地磁分量）；",
        "- 横坐标：到 shower axis 的垂直距离 `r_perp`。",
        "",
        "## 完整性",
        "",
        f"CUDA 完整事例：{report['cuda_complete_count']}/{report['cuda_complete_count']}；"
        "所有 CoREAS 文件与配对 CPU 文件具有相同的行数，且无 NaN/Inf。",
        "",
        "## 运行时间",
        "",
        "CPU 使用顶层 `runtime_raw`；CUDA 使用闭合的 `simulation_timing` shower wall time。",
        "",
        f"- CUDA：中位数 {cuda_time['median_s']:.3f} s，10 例合计 {cuda_time['sum_s']:.3f} s；",
        f"- 原 CPU：中位数 {cpu_time['median_s']:.3f} s，10 例合计 {cpu_time['sum_s']:.3f} s；",
        f"- 以中位数计的单事例加速：{report['runtime']['median_speedup']:.3f}×。",
        "",
        "## CoREAS profile",
        "",
        f"共同 r_perp 点数：{comparison['common_r_perp_points']}。",
        "",
        f"- 振幅 CUDA/CPU 几何平均比：{amplitude.get('geometric_mean_cuda_over_cpu', math.nan):.6g}；",
        f"- 脉宽 CUDA/CPU 几何平均比：{width.get('geometric_mean_cuda_over_cpu', math.nan):.6g}。",
        f"- 同 seed、同天线配对振幅比："
        f"{paired_amplitude.get('event_cluster_geometric_mean_cuda_over_cpu', math.nan):.6g}，"
        f"shower-cluster bootstrap 95% 区间 "
        f"{paired_amplitude.get('event_cluster_bootstrap_95pct', [math.nan, math.nan])}；",
        f"- 同 seed、同天线配对脉宽比："
        f"{paired_width.get('event_cluster_geometric_mean_cuda_over_cpu', math.nan):.6g}，"
        f"shower-cluster bootstrap 95% 区间 "
        f"{paired_width.get('event_cluster_bootstrap_95pct', [math.nan, math.nan])}。",
        "",
        "图：",
        "",
        "- `plots/EyPrime_amplitude_vs_r_perp_cpu_cuda.png`",
        "- `plots/EyPrime_pulse_width_vs_r_perp_cpu_cuda.png`",
        "- `plots/EyPrime_paired_event_antenna_ratios.csv`",
        "- `runtime_analysis/cpu_cuda_runtime_histograms.png`",
        "",
    ]
    (root / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.campaign_root.resolve()
    finalizer_state = root / "finalization_manifest.json"
    state = {"status": "waiting_for_cuda", "campaign_root": str(root)}
    atomic_json(finalizer_state, state)
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    campaign_path = root / "campaign_manifest.json"
    while True:
        if campaign_path.is_file():
            campaign = read_json(campaign_path)
            statuses = [event.get("status") for event in campaign.get("events", [])]
            if "failed" in statuses:
                raise RuntimeError("CUDA campaign contains a failed event")
            if campaign.get("status") == "complete":
                break
        if time.monotonic() >= deadline:
            raise TimeoutError("CUDA campaign did not complete before finalizer timeout")
        time.sleep(args.poll_seconds)

    state["status"] = "validating_cuda"
    atomic_json(finalizer_state, state)
    events = campaign["events"]
    configuration = campaign["configuration"]
    expected_count = int(configuration.get("event_count", len(events)))
    validation = [validate_cuda_event(event, configuration) for event in events]
    if len(validation) != expected_count or not all(item["accepted"] for item in validation):
        raise RuntimeError("one or more CUDA events failed final integrity validation")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.pulse_project.resolve())
    pulse_root = root / "pulse_analysis"
    cpu_analysis = pulse_root / "cpu"
    cuda_analysis = pulse_root / "cuda"
    cpu_input_root = root / "analysis_inputs" / "cpu"
    cpu_dataset = cpu_input_root / (
        f"proton_cpu_{configuration['geomagnetic_model']}_"
        f"{float(configuration['geomagnetic_year']):g}_"
        f"E{float(configuration['energy_GeV']):.0e}_"
        f"TH{float(configuration['zenith_deg']):g}_"
        f"PH{float(configuration['azimuth_deg']):g}_"
        f"EM{float(configuration['em_thinning']):.0e}"
    )
    cpu_dataset.mkdir(parents=True, exist_ok=True)
    for event in events:
        source = Path(event["cpu_library"]).resolve()
        link = cpu_dataset / source.name
        if link.is_symlink():
            if link.resolve() != source:
                raise RuntimeError(f"CPU analysis link points to the wrong library: {link}")
        elif link.exists():
            raise RuntimeError(f"CPU analysis input exists but is not a symlink: {link}")
        else:
            link.symlink_to(source, target_is_directory=True)

    state["status"] = "analyzing_cpu_pulses"
    atomic_json(finalizer_state, state)
    run_logged(
        [
            str(args.python.resolve()),
            "-m", "pulse_analysis",
            "--input-dir", str(cpu_input_root),
            "--output-dir", str(cpu_analysis),
            "--jobs", "8",
            "--parallel-chunksize", "4",
            "--igrf-date", f"{float(configuration['geomagnetic_year']):g}",
            "--waveform-type", "CoREAS",
            "--particle-type", "proton",
            "--force-reanalyze",
        ],
        root / "logs" / "pulse_analysis_cpu.log",
        environment,
        working_directory=args.pulse_project.resolve(),
    )

    state["status"] = "analyzing_cuda_pulses"
    atomic_json(finalizer_state, state)
    run_logged(
        [
            str(args.python.resolve()),
            "-m", "pulse_analysis",
            "--input-dir", str(root),
            "--output-dir", str(cuda_analysis),
            "--jobs", "8",
            "--parallel-chunksize", "4",
            "--igrf-date", f"{float(configuration['geomagnetic_year']):g}",
            "--waveform-type", "CoREAS",
            "--particle-type", "proton",
            "--force-reanalyze",
        ],
        root / "logs" / "pulse_analysis_cuda.log",
        environment,
        working_directory=args.pulse_project.resolve(),
    )

    state["status"] = "plotting_profiles"
    atomic_json(finalizer_state, state)
    plot_root = root / "plots"
    run_logged(
        [
            str(args.python.resolve()),
            str(args.pulse_project.resolve() / "scripts" / "compare_cpu_cuda_rperp_profiles.py"),
            "--cpu-csv",
            str(pulse_root / "cpu" / "all_antennas_pulse_analysis_all_components_with_basis.csv"),
            "--cuda-csv",
            str(cuda_analysis / "all_antennas_pulse_analysis_all_components_with_basis.csv"),
            "--output-dir", str(plot_root),
            "--component", "EyPrime",
            "--energy-gev", f"{float(configuration['energy_GeV']):.17g}",
            "--zenith-deg", f"{float(configuration['zenith_deg']):.17g}",
            "--azimuth-deg", f"{float(configuration['azimuth_deg']):.17g}",
            "--thinning", f"{float(configuration['em_thinning']):.17g}",
            "--geomagnetic-model", str(configuration["geomagnetic_model"]),
            "--geomagnetic-year", f"{float(configuration['geomagnetic_year']):.17g}",
        ],
        root / "logs" / "plot_r_perp_comparison.log",
        environment,
        working_directory=args.pulse_project.resolve(),
    )

    cpu_runtime = []
    for event in events:
        cpu_summary = read_yaml(Path(event["cpu_library"]) / "summary.yaml")
        cpu_runtime.append(float(cpu_summary["runtime_raw"]))
    cuda_runtime = [item["shower_wall_time_s"] for item in validation]
    cuda_observed_runtime = [item["runtime_s"] for item in validation]
    write_runtime_products(
        root,
        events,
        cpu_runtime,
        cuda_runtime,
        cuda_observed_runtime,
        configuration,
    )
    pulse_comparison = read_json(plot_root / "comparison_summary.json")
    report = {
        "status": "complete",
        "configuration": configuration,
        "cuda_complete_count": len(validation),
        "cuda_validation": validation,
        "runtime": {
            "timing_definition": {
                "cpu": "summary.yaml runtime_raw",
                "cuda": "simulation_timing/summary.yaml wall_time_ms",
                "cuda_top_level_observed": "summary.yaml runtime_raw",
            },
            "cpu": describe(cpu_runtime),
            "cuda": describe(cuda_runtime),
            "cuda_top_level_observed": describe(cuda_observed_runtime),
            "median_speedup": float(np.median(cpu_runtime) / np.median(cuda_runtime)),
        },
        "pulse_comparison": pulse_comparison,
    }
    atomic_json(root / "final_report.json", report)
    write_markdown(root, report)
    state.update({"status": "complete", "final_report": str(root / "final_report.json")})
    atomic_json(finalizer_state, state)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
