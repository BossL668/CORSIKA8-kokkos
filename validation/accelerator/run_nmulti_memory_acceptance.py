#!/usr/bin/env python3
"""Bounded same-process N>1 memory regression; no physics configuration edits.

Run under tools/run_memory_guarded.py in a memory-limited systemd user service.
RSS/PSS checkpoints follow observed log messages, not instrumented C++ barriers.
NVIDIA telemetry is device-wide (including the desktop), not process allocation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
import yaml


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_sample():
    try:
        raw = subprocess.check_output([
            "/usr/lib/wsl/lib/nvidia-smi", "--id=0",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=2)
        used, total, util = [float(x.strip()) for x in raw.strip().split(",")]
        return {"device_used_mib": used, "device_total_mib": total,
                "device_util_percent": util}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def memory_sample(process):
    result = {"rss_mib": process.memory_info().rss / 1024**2,
              "threads": process.num_threads(),
              "cpu_s": sum(process.cpu_times()[:2]),
              "host_available_mib": psutil.virtual_memory().available / 1024**2}
    try:
        fields = {}
        for line in Path(f"/proc/{process.pid}/smaps_rollup").read_text().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB":
                fields[parts[0].rstrip(":")] = int(parts[1]) / 1024
        result.update(pss_mib=fields.get("Pss", 0),
                      private_mib=fields.get("Private_Clean", 0) + fields.get("Private_Dirty", 0),
                      swap_mib=fields.get("Swap", 0))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def stop(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def monitor(command, root, env, label, timeout):
    start = time.monotonic()
    result = {"command": command, "gpu_before": gpu_sample(),
              "samples": [], "checkpoints": [], "failure": None}
    selector = selectors.DefaultSelector()
    with (root / (label + ".log")).open("wb") as log:
        child = subprocess.Popen(["stdbuf", "-oL", "-eL"] + command,
                                 cwd=root, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        result["pid"] = child.pid
        process = psutil.Process(child.pid)
        selector.register(child.stdout, selectors.EVENT_READ)
        pending = b""
        next_sample = next_gpu = 0.
        gpu = {}
        ordinal = 0
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= next_gpu:
                    gpu = gpu_sample()
                    next_gpu = elapsed + 1.
                try:
                    sample = memory_sample(process)
                except psutil.NoSuchProcess:
                    break
                sample.update(elapsed_s=elapsed, observed_shower=ordinal, **gpu)
                if elapsed >= next_sample:
                    result["samples"].append(sample)
                    next_sample = elapsed + .25
                if sample["rss_mib"] > 3072:
                    result["failure"] = "child RSS exceeded 3 GiB"
                elif sample["host_available_mib"] < 3072:
                    result["failure"] = "host available memory below 3 GiB"
                elif elapsed > timeout:
                    result["failure"] = "timeout"
                if result["failure"]:
                    break
                for key, _ in selector.select(.05):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(block)
                    log.flush()
                    pending += block
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for raw in lines:
                        line = raw.decode(errors="replace")
                        match = re.search(r"Shower (\d+) / (\d+)", line)
                        kind = None
                        if match:
                            ordinal = int(match.group(1))
                            kind = "start_log"
                        elif "total energy budget (GeV)" in line:
                            kind = "end_log"
                        if kind:
                            try:
                                checkpoint = memory_sample(process)
                            except psutil.NoSuchProcess:
                                continue
                            checkpoint.update(elapsed_s=time.monotonic() - start,
                                              observed_shower=ordinal, kind=kind, **gpu)
                            result["checkpoints"].append(checkpoint)
                            if kind == "end_log" and ordinal % 8 == 0:
                                print(f"{label}: shower {ordinal}, RSS={checkpoint['rss_mib']:.1f} MiB", flush=True)
        finally:
            stop(child)
            # Output remaining in the pipe is retained in the log; it is not
            # assigned fabricated memory checkpoints after the process exits.
            log.write(child.stdout.read())
            child.stdout.close()
            selector.close()
            result.update(returncode=child.wait(), elapsed_s=time.monotonic() - start,
                          gpu_after=gpu_sample(), process_exited=True)
            save(root / (label + "_monitor.json"), result)
    if result["returncode"] or result["failure"]:
        raise RuntimeError(f"{label}: {result['failure']}, exit={result['returncode']}")
    return result


def inspect_output(output, backend, count):
    summary_path = output / "gpu_em/summary.yaml"
    summary = yaml.safe_load(summary_path.read_text())
    expected = {f"shower_{i}" for i in range(count)}
    if set(summary) != expected:
        raise RuntimeError("Unexpected shower set: " + str(output))
    rows = []
    hashes = set()
    for i in range(count):
        item = summary[f"shower_{i}"]
        stats = item["statistics"]
        lifecycle = stats["backend_lifecycle"]
        if not item["complete"] or stats["accelerator"]["backend"] != backend:
            raise RuntimeError("Incomplete or wrong backend")
        if lifecycle["reused"] != (i > 0) or lifecycle["shower_ordinal"] != i + 1:
            raise RuntimeError("Backend lifecycle did not reuse the same session")
        row = {"shower": i + 1, "reused": lifecycle["reused"],
               "rng_domain_version": stats.get("rng_domain_version"),
               "physics_alignment_revision": str(stats.get("physics_alignment_revision")),
               "gpu_particles": stats["gpu_particles"],
               "table_device_bytes": stats["table_device_bytes"],
               "workspace_bytes": stats["workspace_bytes"],
               "peak_device_bytes": stats["peak_device_bytes"],
               "radio_device_bytes": stats["radio"]["device_bytes"],
               "static_host_to_device_bytes": lifecycle["static_host_to_device_bytes"]}
        native = stats["proposal_native"]
        hashes.add(json.dumps({k: native[k] for k in (
            "proposal_version", "cubic_interpolation_version", "table_sha256",
            "node_count", "device_bytes", "aux_sha256")}, sort_keys=True))
        rows.append(row)
    if len(hashes) != 1 or not any(row["gpu_particles"] for row in rows):
        raise RuntimeError("Native table metadata changed or accelerator unused")
    radio = {}
    for name in ("CoREAS", "ZHS"):
        data = pq.read_table(output / name / "observers.parquet", columns=["shower", "Ex", "Ey", "Ez"])
        ids = data["shower"].to_numpy()
        fields = np.column_stack([data[k].to_numpy() for k in ("Ex", "Ey", "Ez")])
        if set(np.unique(ids)) != set(range(count)) or not np.isfinite(fields).all():
            raise RuntimeError("Missing/nonfinite radio events")
        radio[name] = {"finite": True, "nonzero_showers": int(sum(np.any(fields[ids == i]) for i in range(count)))}
    return {"showers": rows, "radio": radio, "summary_file_bytes": summary_path.stat().st_size}


def compare_arrays(left, right):
    files = {p.relative_to(left) for suffix in ("*.parquet", "*.npz") for p in left.rglob(suffix)}
    other = {p.relative_to(right) for suffix in ("*.parquet", "*.npz") for p in right.rglob(suffix)}
    if not files or files != other:
        raise RuntimeError("Output file sets differ")
    rows = []
    for name in sorted(files):
        if name.suffix == ".parquet":
            a, b = pq.read_table(left / name), pq.read_table(right / name)
            equal = a.equals(b, check_metadata=False)
        else:
            with np.load(left / name) as a, np.load(right / name) as b:
                equal = set(a.files) == set(b.files) and all(
                    np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind in "fc") for k in a.files)
        rows.append({"file": str(name), "exact": bool(equal)})
    return {"pass": all(r["exact"] for r in rows), "files": rows}


def main():
    # Arrow and YAML allocators may retain arenas after large reads. Isolate
    # each validation phase from the long-lived monitor and simulation process.
    if len(sys.argv) == 6 and sys.argv[1] == "--inspect":
        save(Path(sys.argv[5]), inspect_output(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])))
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "--compare":
        save(Path(sys.argv[4]), compare_arrays(Path(sys.argv[2]), Path(sys.argv[3])))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--antennas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--auto-count", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed matching runs; never overwrite failed simulations")
    parser.add_argument("--variants", nargs="+", choices=("dual", "independent"),
                        default=["dual", "independent"])
    parser.add_argument("--skip-auto", action="store_true")
    parser.add_argument("--independent-cuda", type=Path)
    parser.add_argument("--independent-openmp", type=Path)
    args = parser.parse_args()
    if min(args.count, args.auto_count) < 2:
        parser.error("Both counts must exceed one")
    if not os.environ.get("FLUPRO") or not (Path(os.environ["FLUPRO"]) / "libflukahp.a").is_file():
        parser.error("Activate corsika_venv and pass FLUPRO into the systemd service")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    binaries = {name: args.project.resolve() / "install" / name / "bin/c8_air_shower"
                for name in ("cuda-openmp", "cuda", "openmp")}
    for name in ("cuda", "openmp"):
        override = getattr(args, "independent_" + name)
        if override is not None:
            binaries[name] = override.resolve()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "note": "same PID N>1; approximate log checkpoints; GPU telemetry device-wide",
              "binaries": {k: {"path": str(v), "sha256": sha256(v)} for k, v in binaries.items()},
              "runs": {}, "comparisons": {}, "pass": False}
    if args.resume and (root / "acceptance.json").exists():
        old = json.loads((root / "acceptance.json").read_text())
        if old["binaries"] != report["binaries"]:
            raise RuntimeError("Refusing to reuse measurements of different binaries")
    cases = [(backend, variant, "fixed", args.count) for backend in ("cuda", "openmp")
             for variant in args.variants]
    if not args.skip_auto:
        cases += [(backend, "dual", "auto", args.auto_count) for backend in ("cuda", "openmp")]
    try:
        for backend, variant, capacity, count in cases:
            label = f"{backend}_{variant}_{capacity}_N{count}"
            binary = binaries["cuda-openmp" if variant == "dual" else backend]
            threads = 4 if backend == "openmp" else 1
            env = os.environ.copy()
            env.update(CORSIKA_DATA=str(args.project.resolve() / "install/cuda-openmp/share/corsika/data"),
                       OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                       OMP_PROC_BIND="spread", OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [str(binary), "--pdg", "22", "-E", "1", "-N", str(count),
                       "-s", "26090711", "-z", "0", "-a", "0", "--emthin", "1e-6",
                       "--em-backend", "kokkos", "--radio-backend", "kokkos",
                       "--kokkos-num-threads", str(threads),
                       "--antenna-file", str(args.antennas.resolve()),
                       "--geomagnetic-model", "IGRF14", "--geomagnetic-year", "2027",
                       "--verbosity", "info", "-f", str(root / label)]
            if variant == "dual":
                command += ["--kokkos-execution", backend]
            if capacity == "fixed":
                command += ["--gpu-min-batch", "16", "--gpu-resident-batch-limit", "4096",
                            "--gpu-memory-fraction", "0.15"]
            print("RUN " + label, flush=True)
            previous = root / (label + "_monitor.json")
            if args.resume and previous.exists():
                result = json.loads(previous.read_text())
                if result["command"] != command or result["returncode"] or result["failure"]:
                    raise RuntimeError("Cannot reuse failed/different run " + label)
                print("REUSE completed " + label, flush=True)
            else:
                if (root / label).exists():
                    raise RuntimeError("Refusing to overwrite existing output " + label)
                result = monitor(command, root, env, label, args.timeout)
            # Preserve raw measurements even if output validation fails.
            report["runs"][label] = result
            inspection_path = root / (label + "_inspection.json")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--inspect",
                            str(root / label), backend, str(count), str(inspection_path)],
                           check=True, timeout=180)
            result["output"] = json.loads(inspection_path.read_text())
            save(root / "acceptance.json", report)
        for backend in ("cuda", "openmp"):
            if set(args.variants) != {"dual", "independent"}:
                continue
            comparison_path = root / (backend + "_comparison.json")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--compare",
                            str(root / f"{backend}_dual_fixed_N{args.count}"),
                            str(root / f"{backend}_independent_fixed_N{args.count}"),
                            str(comparison_path)], check=True, timeout=180)
            report["comparisons"][backend] = json.loads(comparison_path.read_text())
        report["comparisons_requested"] = set(args.variants) == {"dual", "independent"}
        report["pass"] = (len(report["runs"]) == len(cases) and
                          all(x["pass"] for x in report["comparisons"].values()))
        print("ACCEPTANCE " + str(report["pass"]), flush=True)
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        save(root / "acceptance.json", report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
