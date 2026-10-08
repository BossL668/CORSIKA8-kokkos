#!/usr/bin/env python3
"""Bounded runtime/tiling and N>1 checks for independent and dual builds.

Uses already-built binaries. Does not install, alter tuning defaults or run
production campaigns. Temporary tuning candidates stay in the report directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_backend_build_matrix import run, save, digest
from run_nmulti_memory_acceptance import monitor
from run_zhs_window_acceptance import radio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("cuda", "openmp", "cuda-openmp"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = Path(__file__).resolve().parents[2]
    project = source.parent
    build = project / "build" / args.mode
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = project / "install" / args.mode / "share/corsika/data"
    env = dict(os.environ, FLUPRO="/home/yuhanglu/fluka",
               CORSIKA_DATA=str(data), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OMP_PROC_BIND="spread", OMP_PLACES="cores")
    backends = ("cuda", "openmp") if args.mode == "cuda-openmp" else (args.mode,)
    report = {}
    exe = build / "applications/c8_air_shower"
    if not exe.is_file():
        raise RuntimeError("Build c8_air_shower first")
    save(root / "binaries.json", {p.name: digest(p) for p in
         (exe, build / "applications/kokkos_backend_probe", build / "applications/c8_kokkos_tune")})
    for backend in backends:
        threads = 4 if backend == "openmp" else 1
        selected_env = dict(env, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads))
        # No CUDA visibility is necessary or allowed to be relied upon in the
        # independent OpenMP build; the dual build's documented CUDA context
        # requirement is tested separately by Beta5DualBackendRuntime.
        if args.mode == "openmp":
            selected_env["CUDA_VISIBLE_DEVICES"] = ""
        probe = [str(build / "applications/kokkos_backend_probe"),
                 "--backend", backend, "--threads", str(threads), "--values", "1048576"]
        if not run(probe, root, root, backend + "_probe", selected_env, timeout=60):
            raise RuntimeError("Runtime probe failed")
        # Existing physical-table identity is only part of the isolated tuning
        # key. This utility times synthetic queue/tiling probes, not showers.
        tune = [str(build / "applications/c8_kokkos_tune"), "--backend", backend,
                "--threads", str(threads), "--repetitions", "3",
                "--radio-tracks", "129", "--radio-observers", "33",
                "--proposal-table-hash", "7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f",
                "--output", str(root / (backend + "_synthetic_tuning.json"))]
        if not run(tune, root, root, backend + "_tiling", selected_env, timeout=120):
            raise RuntimeError("Tiling probe failed")
        for primary, pdg, energy, count in (("photon", 22, 1, 4), ("proton", 2212, 10, 2)):
            label = backend + "_" + primary
            destination = root / label
            if destination.exists():
                raise RuntimeError("Preserve existing output; choose a new report directory")
            command = [str(exe), "-p", str(pdg), "-E", str(energy), "-N", str(count),
                       "-s", "26090729", "-z", "0", "-a", "0", "--emthin", "1e-6",
                       "--antenna-file", "/home/yuhanglu/21CMA/data/antennas_nwu_coordinates_test.txt",
                       "--geomagnetic-model", "IGRF14", "--geomagnetic-year", "2027",
                       "--verbosity", "info", "-f", str(destination),
                       "--em-backend", "kokkos-proposal", "--radio-backend", "kokkos",
                       "--kokkos-execution", backend, "--kokkos-num-threads", str(threads),
                       "--gpu-min-batch", "16", "--gpu-resident-batch-limit", "4096",
                       "--gpu-memory-fraction", ".15"]
            result = monitor(command, root, selected_env, label, timeout=600)
            save(root / (label + "_provenance.json"), result)
            if result["returncode"] != 0 or result["failure"]:
                raise RuntimeError(label + " failed")
            import yaml
            summary = yaml.load((destination / "gpu_em/summary.yaml").read_text(), Loader=yaml.CSafeLoader)
            assert list(summary) == [f"shower_{i}" for i in range(count)]
            for i, record in enumerate(summary.values()):
                assert record["complete"]
                stats = record["statistics"]
                assert stats["radio"]["fixed_point_overflows"] == 0
                assert stats["backend_lifecycle"]["reused"] == (i > 0)
                assert stats["accelerator"]["backend"] == backend
                assert stats["accelerator"]["gpu"] == (backend == "cuda")
                assert stats["accelerator"]["host_threads"] == threads
                assert stats["hadronic_models"]["low_energy"]["name"] == "FLUKA"
                assert stats["hadronic_models"]["high_energy"]["name"] == "SIBYLL-2.3d"
            shapes = {alg: list(radio(destination / alg, count)[0].shape) for alg in ("CoREAS", "ZHS")}
            report[label] = dict(complete=count, waveform_shapes=shapes)
            save(root / "runtime_acceptance.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
