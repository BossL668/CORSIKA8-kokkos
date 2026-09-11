#!/usr/bin/env python3
"""Bounded frozen single-endpoint replay and full cooperative N=32 smoke.

Does not modify installs, source, dependencies, or existing output directories.
The three pre-change cases use the exact commands captured by baseline freeze.
The cooperative case tests lifecycle, not equality of dynamically split trees.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    binary, baseline, output = a.binary.resolve(), a.baseline.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_PROC_BIND="false", OPENBLAS_NUM_THREADS="1")
    guard = Path(__file__).with_name("run_overlap_guarded.py")
    compare = Path(__file__).with_name("compare_backend_build_outputs.py")
    results = {}
    for mode in ("cuda", "openmp", "proposal"):
        command = json.loads((baseline / f"guarded-photon-{mode}/summary.json").read_text())["command"]
        command[0] = str(binary)
        command[command.index("-f") + 1] = str(output / mode)
        for flag, filename in (("--cuda-replay-trace", mode + "-trace.csv"),
                               ("--cuda-replay-tape-out", mode + "-tape")):
            if flag in command:
                command[command.index(flag) + 1] = str(output / filename)
        run = subprocess.run([sys.executable, str(guard), "--output", str(output / (mode + "-guard")),
                              "--timeout", "180", "--rss-limit-gib", "3", "--sample-resources",
                              "--"] + command, cwd=source, env=env)
        results[mode] = {"run_code": run.returncode}
        if run.returncode == 0:
            result = subprocess.run([sys.executable, str(compare), str(baseline / ("photon-" + mode)),
                str(output / mode), "--report", str(output / (mode + "-comparison.json"))], cwd=source, env=env)
            results[mode]["comparison_code"] = result.returncode
    command = [str(binary), "-p", "22", "-E", "1", "-N", "32", "-s", "26091021",
               "--emthin", "1e-6", "--em-backend", "kokkos", "--radio-backend", "kokkos",
               "--kokkos-execution", "cuda-openmp", "--kokkos-num-threads", "16",
               "--gpu-min-batch", "16", "--gpu-resident-batch-limit", "4096",
               "--gpu-memory-fraction", "0.1", "--geomagnetic-model", "IGRF14",
               "--geomagnetic-year", "2027", "--antenna-file",
               str(source / "validation/accelerator/overlap_probe/antennas.txt"),
               "--verbosity", "info", "-f", str(output / "cooperative16-N32")]
    result = subprocess.run([sys.executable, str(guard), "--output", str(output / "cooperative16-N32-guard"),
                             "--timeout", "300", "--rss-limit-gib", "3", "--sample-resources",
                             "--"] + command, cwd=source, env=env)
    results["cooperative16-N32"] = {"run_code": result.returncode}
    with (output / "regression.json").open("x") as f:
        json.dump(results, f, indent=2)
    return int(any(v for r in results.values() for v in r.values()))


if __name__ == "__main__":
    raise SystemExit(main())
