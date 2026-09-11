#!/usr/bin/env python3
"""Plot bounded N>1 measurements and compare aligned, same-backend controls."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(root):
    return json.loads((root / "acceptance.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    args = parser.parse_args()
    short, long, aligned = read(args.short), read(args.long), read(args.aligned)
    if not long["pass"] or not aligned["pass"]:
        raise RuntimeError("Incomplete run matrix")
    comparisons = {}
    checker = Path(__file__).with_name("run_nmulti_memory_acceptance.py")
    for backend in ("cuda", "openmp"):
        output = args.aligned / (backend + "_aligned_comparison.json")
        subprocess.run([sys.executable, str(checker), "--compare",
                        str(args.short / f"{backend}_dual_fixed_N64"),
                        str(args.aligned / f"{backend}_independent_fixed_N64"),
                        str(output)], check=True, timeout=180)
        comparisons[backend] = json.loads(output.read_text())
    selected = {k: v for k, v in short["runs"].items() if "_dual_" in k}
    selected.update(aligned["runs"])
    selected.update(long["runs"])
    metrics = []
    for label, run in selected.items():
        ends = [r for r in run["checkpoints"] if r["kind"] == "end_log"]
        n = len(run["output"]["showers"])
        warm = next(r for r in ends if r["observed_shower"] >= min(8, n))
        late = [r for r in ends if r["observed_shower"] >= max(8, n // 2)]
        slope = np.polyfit([r["observed_shower"] for r in late], [r["rss_mib"] for r in late], 1)[0]
        device = [r["peak_device_bytes"] / 1048576 for r in run["output"]["showers"]]
        metrics.append({"run": label, "pid": run["pid"], "showers": n,
                        "rss_at_8_mib": warm["rss_mib"], "rss_last_mib": ends[-1]["rss_mib"],
                        "peak_rss_mib": max(r["rss_mib"] for r in run["samples"]),
                        "late_rss_slope_mib_per_shower": float(slope),
                        "peak_private_mib": max(r.get("private_mib", 0) for r in run["samples"]),
                        "minimum_host_available_mib": min(r["host_available_mib"] for r in run["samples"]),
                        "maximum_process_swap_mib": max(r.get("swap_mib", 0) for r in run["samples"]),
                        "tracked_allocation_first_mib": device[0], "tracked_allocation_last_mib": device[-1],
                        "gpu_devicewide_peak_mib": max(r.get("device_used_mib", 0) for r in run["samples"]),
                        "elapsed_s": run["elapsed_s"]})
    with (args.aligned / "memory_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for backend, color in (("cuda", "#0072B2"), ("openmp", "#D55E00")):
        run = long["runs"][f"{backend}_dual_fixed_N256"]
        checkpoints = [r for r in run["checkpoints"] if r["kind"] == "end_log"]
        x = [r["observed_shower"] for r in checkpoints]
        axes[0, 0].plot(x, [r["rss_mib"] for r in checkpoints], color=color, label=backend.upper())
        rows = run["output"]["showers"]
        axes[0, 1].plot([r["shower"] for r in rows], [r["workspace_bytes"] / 1048576 for r in rows],
                        color=color, label=backend.upper())
        a = short["runs"][f"{backend}_dual_auto_N16"]
        ax = axes[1, 0]
        ax.plot([r["elapsed_s"] for r in a["samples"]], [r.get("device_used_mib", np.nan) for r in a["samples"]],
                color=color, label=backend.upper() + " mode (device-wide)")
    with (args.short / "yaml_report_retention.csv").open() as stream:
        yaml_rows = list(csv.DictReader(stream))
    retained = [r for r in yaml_rows if r["phase"] == "retained"]
    axes[1, 1].plot([int(r["shower"]) for r in retained], [float(r["rss_mib"]) for r in retained],
                    color="#009E73", label="Retained YAML reports (isolated probe)")
    released = float(yaml_rows[-1]["rss_mib"])
    axes[1, 1].axhline(released, color="gray", linestyle="--", label="After destruction + diagnostic trim")
    for ax, title, xlabel, ylabel in zip(axes.flat,
            ("Same process: N=256", "Transport workspaces stabilize", "Default 70% memory budget, N=16", "Report retention is not unreachable memory"),
            ("Shower ordinal", "Shower ordinal", "Elapsed time [s]", "Stored reports"),
            ("Process RSS [MiB]", "Tracked workspace [MiB]", "Device-wide VRAM [MiB]", "Probe RSS [MiB]")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.savefig(args.aligned / "nmulti_memory_diagnosis.png")
    fig.savefig(args.aligned / "nmulti_memory_diagnosis.pdf")
    plt.close(fig)
    summary = {"memory_test_completed": True,
               "no_simulation_memory_guard_triggered": all(
                   r["returncode"] == 0 and not r["failure"] for r in selected.values()),
               "flat_host_memory_claim": False,
               "aligned_array_comparisons": comparisons,
               "all_aligned_arrays_equal": all(r["pass"] for r in comparisons.values()),
               "metrics": metrics, "yaml_probe": yaml_rows,
               "excluded_control": "short/independent: stale install binaries, different RNG/physics revision"}
    (args.aligned / "memory_review.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arrays_equal": summary["all_aligned_arrays_equal"], "metrics": metrics}, indent=2))
    return 0 if summary["all_aligned_arrays_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
