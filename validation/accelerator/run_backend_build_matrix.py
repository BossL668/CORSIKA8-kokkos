#!/usr/bin/env python3
"""Rebuild first-party backends and run their registered regression matrix.

Keeps dependency caches, installs and production executables unchanged. Each
mode has its own existing CMake build tree. Logs/commands and source identity
are retained; a failure in one mode does not hide the other modes' results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(command, cwd, output, label, env, timeout=7200):
    start = time.monotonic()
    result = dict(command=command, cwd=str(cwd), peak_tree_rss_mib=0.,
                  minimum_available_mib=float("inf"), failure=None)
    with (output / (label + ".log")).open("wb") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                rss = 0
                try:
                    members = [process] + process.children(recursive=True)
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available / 1024**2
                result["peak_tree_rss_mib"] = max(result["peak_tree_rss_mib"], rss / 1024**2)
                result["minimum_available_mib"] = min(result["minimum_available_mib"], available)
                if available < 3072:
                    result["failure"] = "host available memory below 3 GiB"
                elif rss > 6 * 1024**3:
                    result["failure"] = "diagnostic process tree exceeded 6 GiB RSS"
                elif time.monotonic() - start > timeout:
                    result["failure"] = "timeout"
                if result["failure"]:
                    break
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            result["returncode"] = child.wait()
    result["seconds"] = time.monotonic() - start
    if result["minimum_available_mib"] == float("inf"):
        result["minimum_available_mib"] = None
    save(output / (label + ".json"), result)
    print(label, result["returncode"], round(result["seconds"], 1),
          "RSS MiB", round(result["peak_tree_rss_mib"], 1), flush=True)
    return result["returncode"] == 0 and result["failure"] is None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", choices=("cuda", "openmp", "cuda-openmp"),
                    default=("cuda", "openmp", "cuda-openmp"))
    ap.add_argument("--force-backend-recompile", action="store_true")
    args = ap.parse_args()
    source = Path(__file__).resolve().parents[2]
    project = source.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4",
               OMP_PROC_BIND="spread", OMP_PLACES="cores",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               FLUPRO="/home/yuhanglu/fluka")
    save(output / "source_identity.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "git_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=source)).hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "kernel_files": {str(p.relative_to(source)): digest(p)
                         for base in ("corsika/accelerator", "src/accelerator")
                         for p in sorted((source / base).rglob("*")) if p.is_file()}})
    tests = ["testKokkosScalarRadioAlignment", "testKokkosCpuDepositionAlignment",
             "testKokkosScalarThinningAlignment", "testKokkosCpuTransportAlignment",
             "testKokkosTuningCache", "testKokkosPendingParticleQueue",
             "testKokkosWavefrontBucketing", "testKokkosResidentMemoryProjection",
             "testKokkosCompositeExclusiveScan", "testKokkosFusedProfileStatistics",
             "testKokkosProposalNativeTable"]
    summary = {}
    for mode in args.modes:
        build = project / "build" / mode
        assert (build / "CMakeCache.txt").is_file(), build
        mode_env = dict(env, CORSIKA_DATA=str(project / "install" / mode / "share/corsika/data"))
        summary[mode] = {}
        if not run(["cmake", "-S", str(source), "-B", str(build)],
                   project, output, mode + "_configure", mode_env):
            summary[mode]["stage"] = "configure_failed"
            save(output / "summary.json", summary)
            continue
        if args.force_backend_recompile:
            folder = build / "src/accelerator/em/kokkos"
            clean = folder / "CMakeFiles/CORSIKA8KokkosEm.dir/cmake_clean.cmake"
            # CMake's target-local clean removes only generated backend objects
            # and its static archive, never source files or broad directories.
            assert clean.is_file() and folder.is_relative_to(build)
            if not run(["cmake", "-P", str(clean)], folder, output,
                       mode + "_clean_backend_objects", mode_env):
                raise RuntimeError("Target-local clean failed")
        targets = ["c8_air_shower", "fluka_batch_worker", "c8_kokkos_tune",
                   "kokkos_backend_probe"] + tests
        if mode == "cuda-openmp":
            targets += ["testKokkosScalarRadioAlignmentOpenMP"]
        if not run(["cmake", "--build", str(build), "--parallel", "1", "--target"] + targets,
                   project, output, mode + "_build", mode_env):
            summary[mode]["stage"] = "build_failed"
            save(output / "summary.json", summary)
            continue
        summary[mode]["binaries"] = {
            name: digest(build / "applications" / name)
            for name in targets[:4]}
        ok = run(["ctest", "--test-dir", str(build), "--output-on-failure",
                  "--timeout", "300", "-j", "1", "-R", "^(testKokkos|Beta5)"],
                 project, output, mode + "_ctest", mode_env)
        summary[mode]["stage"] = "passed" if ok else "ctest_failed"
        save(output / "summary.json", summary)
    if any(v["stage"] != "passed" for v in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
