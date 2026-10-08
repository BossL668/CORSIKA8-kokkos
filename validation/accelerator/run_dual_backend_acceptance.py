#!/usr/bin/env python3
"""Bounded, same-backend regression of the experimental dual executable.

Run in corsika_venv with CORSIKA_DATA pointing to an existing PROPOSAL cache.
Each backend/case uses a fresh process with N=2 to exercise session reuse.
This is a refactor regression, not an ensemble or performance acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd, env, log, timeout, rss_limit):
    started = time.monotonic()
    samples = []
    failure = None
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    rss = process.memory_info().rss
                    samples.append({"elapsed_s": elapsed, "rss": rss,
                                    "threads": process.num_threads(),
                                    "cpu_s": sum(process.cpu_times()[:2])})
                except psutil.NoSuchProcess:
                    break
                if rss > rss_limit:
                    failure = "RSS limit exceeded"
                elif psutil.virtual_memory().available < 3 * 1024**3:
                    failure = "host available memory below 3 GiB"
                elif elapsed > timeout:
                    failure = "timeout"
                if failure:
                    break
                time.sleep(0.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    result = {"command": command, "returncode": code, "failure": failure,
              "elapsed_s": time.monotonic() - started,
              "peak_rss_bytes": max((x["rss"] for x in samples), default=0),
              "samples": samples, "log": str(log)}
    return result


def compare(left, right):
    rows = []
    left_files = {p.relative_to(left) for suffix in ("*.parquet", "*.npz")
                  for p in left.rglob(suffix)}
    right_files = {p.relative_to(right) for suffix in ("*.parquet", "*.npz")
                   for p in right.rglob(suffix)}
    if not left_files or left_files != right_files:
        return {"pass": False, "file_set_difference":
                sorted(map(str, left_files ^ right_files))}
    for relative in sorted(left_files):
        if relative.suffix == ".parquet":
            a, b = pq.read_table(left / relative), pq.read_table(right / relative)
            equal = a.equals(b, check_metadata=False)
            rows.append({"file": str(relative), "rows": a.num_rows,
                         "exact": equal})
        else:
            with np.load(left / relative, allow_pickle=False) as a, \
                    np.load(right / relative, allow_pickle=False) as b:
                equal = (set(a.files) == set(b.files) and
                         all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in "fc")
                             for k in a.files))
            rows.append({"file": str(relative), "exact": bool(equal)})
    physics_fields = (
        "rng_domain_version", "physics_alignment_revision", "hadronic_models",
        "thinning", "process_registry", "forced_primary", "gpu_particles",
        "cpu_particle_steps", "particles_staged", "gpu_muon_transport_enabled",
        "wavefronts", "resident_photon_wavefronts", "resident_lepton_wavefronts",
        "gpu_final_states", "physical_secondaries", "cpu_generic_fallbacks",
        "cpu_decay_particles", "cpu_specified_final_states",
        "cpu_completed_selected_losses", "cpu_completed_native_selection_replays",
        "cpu_fallbacks_by_process", "cpu_fallbacks_by_reason", "observed",
        "escaped", "cut", "weighted_deposit_GeV", "weighted_muon_parent_productions",
        "radio_tracks", "processes", "epair_sampler", "queue_overflows",
        "gpu_physics_source", "proposal_native", "energy_ledger")
    a = yaml.safe_load((left / "gpu_em/summary.yaml").read_text())
    b = yaml.safe_load((right / "gpu_em/summary.yaml").read_text())
    for shower in ("shower_0", "shower_1"):
        for key in physics_fields:
            av, bv = a[shower]["statistics"], b[shower]["statistics"]
            va, vb = av.get(key), bv.get(key)
            if key == "hadronic_models":
                # Keep names, versions, interaction counts and transition
                # energy; wall-clock timing is not a physical observable.
                def without_timing(value):
                    if isinstance(value, dict):
                        return {k: without_timing(v) for k, v in value.items()
                                if k != "final_state_time_ms"}
                    return value
                va, vb = without_timing(va), without_timing(vb)
            rows.append({"file": "gpu_em/summary.yaml:" + shower + "." + key,
                         "exact": key in av and key in bv and va == vb})
    return {"pass": all(row["exact"] for row in rows), "files": rows}


def inspect_output(output, backend, require_nonzero_radio=False):
    summary = yaml.safe_load((output / "gpu_em/summary.yaml").read_text())
    if set(summary) != {"shower_0", "shower_1"}:
        raise RuntimeError("Expected two completed showers")
    result = []
    for key in sorted(summary):
        item = summary[key]
        if not item.get("complete"):
            raise RuntimeError("Incomplete shower " + key)
        stats = item["statistics"]
        accelerator = stats["accelerator"]
        if accelerator["backend"] != backend:
            raise RuntimeError("Executed the wrong backend")
        if bool(accelerator["gpu"]) != (backend == "cuda"):
            raise RuntimeError("Execution-space metadata is inconsistent")
        if backend == "cuda" and accelerator["host_threads"] != 1:
            raise RuntimeError("GPU scheduling used multiple host threads")
        result.append({"shower": key, "gpu_particles": stats.get("gpu_particles"),
                       "backend_lifecycle": stats.get("backend_lifecycle"),
                       "accelerator": accelerator})
    if not result[1]["backend_lifecycle"]["reused"]:
        raise RuntimeError("N=2 did not reuse the backend")
    if not any(row["gpu_particles"] for row in result):
        raise RuntimeError("No particles used the accelerated transport")
    waveform_checks = {}
    for radio in ("CoREAS", "ZHS"):
        data = pq.read_table(output / radio / "observers.parquet")
        fields = np.column_stack([data[k].to_numpy() for k in ("Ex", "Ey", "Ez")])
        waveform_checks[radio] = {"finite": bool(np.isfinite(fields).all()),
                                  "nonzero": bool(np.any(fields)),
                                  "peak_V_per_m": float(np.max(np.abs(fields)))}
        # A low-energy hadronic shower can legitimately have no contributions
        # in the configured observer time window. The photon case is the
        # nonzero full-radio regression; do not reject a zero baseline proton.
        if not np.isfinite(fields).all() or (require_nonzero_radio and not np.any(fields)):
            raise RuntimeError(radio + " waveforms are nonfinite or all zero")
    result[0]["waveform_checks"] = waveform_checks
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--recheck":
        # Re-evaluate completed data without rerunning showers or overwriting
        # the original report. Intended for auditing test criteria themselves.
        original = Path(sys.argv[2]).resolve()
        report = json.loads(original.read_text())
        for label, run_result in report["runs"].items():
            if run_result["returncode"] != 0 or run_result["failure"]:
                raise RuntimeError("Cannot accept failed run " + label)
            backend = "openmp" if "_openmp_" in label else "cuda"
            run_result["output_summary"] = inspect_output(
                original.parent / label, backend, label.startswith("photon"))
        if len(report["runs"]) != 8:
            raise RuntimeError("Cannot accept an incomplete eight-process matrix")
        for label in report["comparisons"]:
            report["comparisons"][label] = compare(
                original.parent / (label + "_baseline"),
                original.parent / (label + "_dual"))
        report["pass"] = all(c["pass"] for c in report["comparisons"].values())
        report["rechecked_from"] = str(original)
        report["review_criteria"] = (
            "Exact arrays/counters/table hashes; exclude hadronic final_state_time_ms; "
            "require finite radio for all cases and nonzero photon radio")
        destination = original.with_name("acceptance_review.json")
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print("RECHECK " + str(report["pass"]) + ": " + str(destination))
        return 0 if report["pass"] else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", required=True, type=Path)
    parser.add_argument("--baseline-cuda", required=True, type=Path)
    parser.add_argument("--baseline-openmp", required=True, type=Path)
    parser.add_argument("--antennas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {"description": "same-backend full-radio N=2 refactor regression",
              "executables": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                              for name, path in [("dual", args.dual),
                                                 ("baseline_cuda", args.baseline_cuda),
                                                 ("baseline_openmp", args.baseline_openmp)]},
              "runs": {}, "comparisons": {}}
    try:
        for case, pdg, energy in [("photon1GeV", 22, 1), ("proton10GeV", 2212, 10)]:
            for backend in ("cuda", "openmp"):
                for variant in ("baseline", "dual"):
                    label = case + "_" + backend + "_" + variant
                    env = os.environ.copy()
                    threads = args.threads if backend == "openmp" else 1
                    env.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                               OMP_PROC_BIND="spread", OMP_PLACES="cores",
                               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    executable = args.dual if variant == "dual" else getattr(args, "baseline_" + backend)
                    destination = root / label
                    command = [str(executable.resolve()), "--pdg", str(pdg), "-E", str(energy),
                               "-N", "2", "-s", "26090611", "-z", "0", "-a", "0",
                               "--emthin", "1e-6", "--em-backend", "kokkos-proposal",
                               "--radio-backend", "kokkos", "--gpu-min-batch", "16",
                               "--gpu-resident-batch-limit", "4096", "--gpu-memory-fraction", "0.15",
                               "--kokkos-num-threads", str(threads),
                               "--antenna-file", str(args.antennas.resolve()),
                               "--geomagnetic-model", "IGRF14", "--geomagnetic-year", "2027",
                               "--verbosity", "warn", "-f", str(destination)]
                    if variant == "dual":
                        command += ["--kokkos-execution", backend]
                    print("RUN " + label, flush=True)
                    result = run(command, root, env, root / (label + ".log"),
                                 args.timeout, 3 * 1024**3)
                    report["runs"][label] = result
                    if result["returncode"] != 0 or result["failure"]:
                        raise RuntimeError(label + " failed: " + str(result["failure"]))
                    result["output_summary"] = inspect_output(
                        destination, backend, require_nonzero_radio=(pdg == 22))
                comparison = compare(root / (case + "_" + backend + "_baseline"),
                                     root / (case + "_" + backend + "_dual"))
                report["comparisons"][case + "_" + backend] = comparison
                print("COMPARE " + case + " " + backend + ": " + str(comparison["pass"]), flush=True)
        report["pass"] = all(c["pass"] for c in report["comparisons"].values())
    except Exception as error:
        report["pass"] = False
        report["error"] = str(error)
        raise
    finally:
        (root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
