#!/usr/bin/env python3
"""Plot the isolated 16-thread cooperative pilot; never a statistical acceptance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    a = p.parse_args()
    root, build = a.output.resolve(), a.build.resolve()
    results, samples = {}, {}
    cases = [("CUDA", "air-cuda-fe100tev-1", "cuda_fe100tev_seed85000001"),
             ("CUDA + OpenMP 16", "air-cooperative16-fe100tev-1-retry",
              "cooperative16_fe100tev_seed85000001_retry")]
    for label, log, data in cases:
        guard = json.loads((build / log / "summary.json").read_text())
        assert guard["pass"]
        m = yaml.safe_load((root / data / "gpu_em/summary.yaml").read_text())["shower_0"]
        assert m["complete"]
        s = m["statistics"]
        assert s["profile"]["fixed_point_overflows"] == s["radio"]["fixed_point_overflows"] == s["queue_overflows"] == 0
        assert s["cross_species"]["final_pending_photons"] == s["cross_species"]["final_pending_leptons"] == 0
        samples[label] = [json.loads(x) for x in (build / log / "resources.jsonl").read_text().splitlines()]
        results[label] = {"end_to_end_s": guard["elapsed_s"],
            "shower_s": yaml.safe_load((root / data / "simulation_timing/summary.yaml").read_text())["shower_0"]["wall_time_ms"] / 1000,
            "peak_rss_mib": guard["peak_tree_rss_bytes"] / 1024**2,
            "profile_steps": s["profile"]["steps"],
            "gpu_final_states": s["gpu_final_states"],
            "cooperative": s["accelerator"].get("cooperative"),
            "energy_ledger_complete_coverage": s["energy_ledger"]["complete_coverage"],
            "energy_ledger_accepted": s["energy_ledger"]["accepted"],
            "device_peak_mib": max(x.get("device_used_mib", 0) for x in samples[label])}
    nroot = root / "regression16/cooperative16-N32"
    n = yaml.safe_load((nroot / "gpu_em/summary.yaml").read_text())
    assert len(n) == 32 and all(v["complete"] for v in n.values())
    stats = [n[f"shower_{i}"]["statistics"] for i in range(32)]
    assert all(s["accelerator"]["host_threads"] == 16 for s in stats)
    assert all(s["profile"]["fixed_point_overflows"] == s["radio"]["fixed_point_overflows"] == s["queue_overflows"] == 0 for s in stats)
    assert all(s["cross_species"]["final_pending_photons"] == s["cross_species"]["final_pending_leptons"] == 0 for s in stats)
    assert all(s["backend_lifecycle"]["reused"] for s in stats[1:])
    nmonitor = json.loads((root / "regression16/cooperative16-N32-guard/summary.json").read_text())
    workspace = [s["workspace_bytes"] for s in stats]
    cpu_workspace = [s["accelerator"]["cooperative"]["openmp_workspace_bytes"] for s in stats]
    report = {"pilot_only": True, "Fe100TeV": results,
        "observed_ratio_not_ensemble_speedup": results["CUDA"]["end_to_end_s"] / results["CUDA + OpenMP 16"]["end_to_end_s"],
        "N32": {"completed": 32, "peak_rss_mib": nmonitor["peak_tree_rss_bytes"] / 1024**2,
            "backend_reused_after_first": True, "gpu_workspace_bytes": workspace,
            "openmp_workspace_bytes": cpu_workspace,
            "final_16_workspaces_stable": len(set(workspace[16:])) == len(set(cpu_workspace[16:])) == 1},
        "limitations": ["one Fe shower per mode; dynamic splitting changes the tree",
            "GPU telemetry is device-wide including desktop", "pending-window CPU time is not actual kernel overlap",
            "accelerator-only energy ledger has incomplete coverage in BOTH modes; no full energy-closure claim",
            "no 500-event statistical acceptance"]}
    with (root / "PILOT_SUMMARY.json").open("x") as f:
        json.dump(report, f, indent=2)
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    colors = ["#3675ad", "#cf7140"]
    for label, color in zip(results, colors):
        row = samples[label]
        time = np.array([v["elapsed_s"] for v in row])
        ax[0, 1].plot(time, [v.get("device_util_percent", np.nan) for v in row], label=label, color=color)
        ax[1, 0].plot(time, [v.get("device_used_mib", np.nan) / 1024 for v in row], color=color)
        cpu = np.array([v.get("cpu_seconds", np.nan) for v in row])
        ax[1, 1].plot(time[1:], np.maximum(0, np.diff(cpu) / np.diff(time)), color=color)
    for index, (label, result) in enumerate(results.items()):
        ax[0, 0].bar(index, result["end_to_end_s"], color=colors[index], width=.55)
        ax[0, 0].text(index, result["end_to_end_s"] + 2, f'{result["end_to_end_s"]:.2f} s', ha="center")
    ax[0, 0].set_xticks(range(2), list(results))
    ax[0, 0].set_ylabel("End-to-end time [s]")
    ax[0, 0].set_ylim(0, max(v["end_to_end_s"] for v in results.values()) * 1.2)
    ax[0, 1].set_ylabel("Device-wide GPU utilization [%]")
    ax[0, 1].set_ylim(0, 105); ax[0, 1].legend(frameon=False)
    ax[1, 0].set_ylabel("Device-wide used memory [GiB]")
    ax[1, 1].set_ylabel("CPU time / wall time [core equivalents]")
    for item in (ax[0, 1], ax[1, 0], ax[1, 1]):
        item.set_xlabel("Elapsed time [s]"); item.grid(alpha=.18)
        item.set_xlim(0, max(v["end_to_end_s"] for v in results.values()))
    fig.suptitle("100 TeV vertical Fe-56: full radio, one shower per mode (pilot)")
    fig.text(.5, .015, "Identical primary parameters/seed, different trees allowed; this is not an ensemble speedup measurement.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96)); fig.savefig(root / "pilot_fe100tev_resources.png"); plt.close(fig)
    row = [json.loads(x) for x in (root / "regression16/cooperative16-N32-guard/resources.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot([v["elapsed_s"] for v in row], [v["rss_bytes"] / 1024**2 for v in row])
    ax[0].set(xlabel="Elapsed time [s]", ylabel="Process RSS [MiB]", title="32 consecutive 1 GeV photon showers")
    ax[1].plot(range(1, 33), np.array(workspace) / 1024**2, label="CUDA transport workspace")
    ax[1].plot(range(1, 33), np.array(cpu_workspace) / 1024**2, label="OpenMP transport workspace")
    ax[1].set(xlabel="Shower ordinal", ylabel="Retained workspace [MiB]", xlim=(1, 32))
    for item in ax: item.grid(alpha=.18)
    ax[1].legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(root / "N32_memory.png"); plt.close(fig)
    print(json.dumps({"pilot_times": {k:v["end_to_end_s"] for k,v in results.items()}, "N32": report["N32"]}, indent=2))


if __name__ == "__main__":
    main()
