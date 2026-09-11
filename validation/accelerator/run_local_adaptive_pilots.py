#!/usr/bin/env python3
"""Guarded RTX4060/20-thread adaptive pilots; isolated build and no production install."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil
import yaml


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("output", "build", "before", "fe-manifest", "proton-command"):
        p.add_argument("--" + option, type=Path, required=True)
    p.add_argument("--build-unit", required=True)
    p.add_argument("--fe-seeds", type=int, default=5)
    p.add_argument("--resume-preflight", action="store_true",
                   help="Resume an infrastructure failure only before any test was launched")
    p.add_argument("--resume", action="store_true",
                   help="Reuse only successful identical commands; incomplete guard runs are never skipped")
    a = p.parse_args()
    root = a.output.resolve()
    if root.exists():
        assert a.resume or (a.resume_preflight and not list(root.glob("*-guard"))), "prior tests require explicit resume"
        previous = json.loads((root/"STATUS.json").read_text())
        assert a.resume or not previous["records"], "cannot overwrite tested events"
    else:
        root.mkdir(parents=True)
    tools = Path(__file__).resolve().parent
    source = tools.parents[1]
    state = dict(phase="waiting-for-build", complete=False,
                 records=previous["records"] if a.resume else [])
    save(root/"STATUS.json", state)
    floor = 4*2**30
    # Watch the separate, persistent compiler unit before acquiring the GPU.
    while subprocess.run(["systemctl", "--user", "is-active", "--quiet", a.build_unit]).returncode == 0:
        if psutil.virtual_memory().available < floor:
            subprocess.run(["systemctl", "--user", "stop", a.build_unit], check=True)
            raise RuntimeError("build stopped: available memory below 4 GiB")
        time.sleep(2)
    status = subprocess.check_output(["systemctl", "--user", "show", a.build_unit,
                                     "-p", "ExecMainStatus", "--value"], text=True).strip()
    assert status == "0" or (a.resume and status == ""), "build failed"
    binary = a.build/"applications/c8_air_shower"
    fixture = a.build/"tests/accelerator/testKokkosCooperativeBackend"
    assert binary.is_file() and fixture.is_file()
    # No production task may contend with these timing runs.
    assert subprocess.run(["pgrep", "-x", "c8_air_shower"], stdout=subprocess.DEVNULL).returncode == 1
    (root/"binaries").mkdir(exist_ok=a.resume_preflight or a.resume)
    frozen = root/"binaries/c8_air_shower"
    if frozen.exists():
        assert (a.resume_preflight or a.resume) and digest(binary)==digest(frozen), "preflight binary changed"
    else:
        shutil.copy2(binary, frozen)
    fe = json.loads(a.fe_manifest.read_text())
    proton = json.loads(a.proton_command.read_text())["command"]
    provenance = dict(binary_sha256=digest(frozen), before_sha256=digest(a.before),
        fixture_sha256=digest(fixture), source=str(source), build=str(a.build),
        fe_manifest_sha256=digest(a.fe_manifest), proton_command_sha256=digest(a.proton_command),
        threads=20, gpu_memory_fraction=.7, emthin="1e-6", max_weight="unchanged",
        physical_comparison="same new binary; dynamic modes need not produce the same shower tree",
        historical_references=[str(a.fe_manifest), str(a.proton_command)],
        driver=subprocess.check_output([shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi",
                                        "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True))
    if (root/"PROVENANCE.json").exists():
        previous_provenance=json.loads((root/"PROVENANCE.json").read_text())
        for key in ("binary_sha256", "before_sha256", "fixture_sha256",
                    "fe_manifest_sha256", "proton_command_sha256"):
            assert provenance[key]==previous_provenance[key], "resume identity changed: "+key
    else:
        save(root/"PROVENANCE.json", provenance)
    env = dict(os.environ, FLUPRO="/home/yuhanglu/fluka",
        CORSIKA_DATA=str(source/"modules/data"), OMP_NUM_THREADS="20", OMP_THREAD_LIMIT="20",
        OMP_PROC_BIND="false", OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    def guarded(label, cmd, timeout, threads=20, rss=5):
        assert psutil.virtual_memory().available >= floor
        assert shutil.disk_usage(root).free > 15*2**30, "disk reserve below 15 GiB"
        state["phase"] = label
        state["updated_unix"] = time.time()
        save(root/"STATUS.json", state)
        saved = root/(label+"-guard/summary.json")
        if saved.exists():
            monitor = json.loads(saved.read_text())
            assert a.resume and monitor["pass"] and monitor["command"]==list(map(str,cmd)), "unsafe guard resume"
            return monitor
        run_env = dict(env, OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads))
        result = subprocess.run([sys.executable, str(tools/"run_overlap_guarded.py"),
            "--output", str(root/(label+"-guard")), "--timeout", str(timeout),
            "--rss-limit-gib", str(rss), "--sample-resources", "--sample-threads",
            "--", *list(map(str, cmd))], cwd=source, env=run_env)
        monitor = json.loads((root/(label+"-guard/summary.json")).read_text())
        assert result.returncode == 0 and monitor["pass"], label+" failed; production stays paused"
        return monitor

    # Old/new single endpoints plus adaptive N=32. Original formulas/arrays
    # and random decision traces are compared, not just return codes.
    guarded("single-end-and-N32", [sys.executable, str(tools/"run_independent_driver_regression.py"),
        "--before", str(a.before), "--after", str(frozen), "--output", str(root/"regression"),
        "--lifecycle-energy-gev", "100", "--allow-adaptive-option",
        "--cooperative-policy", "adaptive"], 1800)
    guarded("real-em-radio-N2", [str(fixture), str(source/"modules/data/PROPOSAL"),
                               "20", "subshowers", ".2", "adaptive"], 600)
    save(root/"CORRECTNESS_GATES.json", dict(passed=True, ensemble_acceptance=False))

    def put(cmd, flag, value):
        if flag in cmd:
            cmd[cmd.index(flag)+1] = str(value)
        else:
            cmd += [flag, str(value)]

    def run_event(family, mode, seed, warm=False):
        label = ("warmup-" if warm else "") + family+"-"+str(seed)+"-"+mode
        if family == "Fe100TeV":
            cmd = [str(frozen)] + list(fe["common_argv"])
            for flag, value in (("-N", 1), ("-s", seed), ("-f", root/label),
                ("--em-backend", "kokkos"), ("--radio-backend", "kokkos"),
                ("--gpu-physics-source", "proposal-native"), ("--gpu-min-batch", 4096),
                ("--gpu-memory-fraction", .7), ("--hadronic-workers", 1),
                ("--kokkos-device", 0)):
                put(cmd, flag, value)
        else:
            cmd = list(proton)
            cmd[0] = str(frozen)
            put(cmd, "-s", seed)
            put(cmd, "-f", root/label)
        if warm:
            put(cmd, "-E", 1000)
        put(cmd, "--kokkos-execution", "cuda" if mode == "single-cuda" else "cuda-openmp")
        put(cmd, "--kokkos-num-threads", 1 if mode == "single-cuda" else 20)
        put(cmd, "--kokkos-cooperative-policy", "adaptive" if mode == "adaptive20" else "legacy")
        assert digest(frozen) == provenance["binary_sha256"]
        record = dict(family=family, mode=mode, seed=seed, command=cmd, complete=False, warmup=warm)
        save(root/(label+".json"), record)
        monitor = guarded(label, cmd, 14400 if family=="proton100PeV" and not warm else 1200,
                          1 if mode=="single-cuda" else 20)
        record["monitor"] = monitor
        gpu = yaml.safe_load((root/label/"gpu_em/summary.yaml").read_text())["shower_0"]
        timing = yaml.safe_load((root/label/"simulation_timing/summary.yaml").read_text())["shower_0"]
        assert gpu["complete"] and timing["closed"]
        stats = gpu["statistics"]
        assert stats["queue_overflows"] == stats["profile"]["fixed_point_overflows"] == stats["radio"]["fixed_point_overflows"] == 0
        accelerator = stats["accelerator"]
        assert accelerator["gpu"]
        assert accelerator["host_threads"] == (1 if mode=="single-cuda" else 20)
        if mode!="single-cuda":
            coop = accelerator["cooperative"]
            assert coop["subshower_cuda_submissions"] == coop["subshower_cuda_commits"]
            if not warm:
                assert coop["subshower_cuda_submissions"] > 0 and coop["subshower_openmp_epochs"] > 0
            if mode=="adaptive20":
                assert coop["scheduling_policy"] == "adaptive-v2"
        # Stream finite-array checks; never load a high-energy event wholesale.
        import numpy as np
        import pyarrow.parquet as pq
        for algorithm in ("CoREAS", "ZHS"):
            file = pq.ParquetFile(root/label/algorithm/"observers.parquet")
            assert file.metadata.num_rows > 0
            for batch in file.iter_batches(columns=["Ex", "Ey", "Ez"], batch_size=65536, use_threads=False):
                assert all(np.isfinite(col.to_numpy()).all() for col in batch.columns)
        record.update(complete=True, process_s=monitor["elapsed_s"],
                      shower_s=timing["wall_time_ms"]/1000, accelerator=accelerator)
        save(root/(label+".json"), record)
        state["records"] = [r for r in state["records"]
            if (r["family"],r["mode"],r["seed"],r["warmup"]) != (family,mode,seed,warm)]
        state["records"].append(record)
        save(root/"STATUS.json", state)
        print(label, record["process_s"], record["shower_s"], flush=True)
        return record

    for mode in ("single-cuda", "legacy20", "adaptive20"):
        run_event("Fe100TeV", mode, 85000001, warm=True)
    modes = ("adaptive20", "single-cuda", "legacy20")
    for i in range(a.fe_seeds):
        order = modes[i%3:]+modes[:i%3]
        for mode in order:
            run_event("Fe100TeV", mode, 85000001+i)
        records = [r for r in state["records"] if not r["warmup"] and r["family"]=="Fe100TeV"]
        median = {mode: statistics.median(r["process_s"] for r in records if r["mode"]==mode)
                  for mode in modes}
        save(root/"Fe_PERFORMANCE_PARTIAL.json", dict(events=len(records), median_s=median,
            single_over_adaptive=median["single-cuda"]/median["adaptive20"],
            legacy_over_adaptive=median["legacy20"]/median["adaptive20"]))
    # Explicit requested high-energy case, first adaptive so its result arrives
    # before the longer baseline reruns. One seed is a pilot, not acceptance.
    for mode in modes:
        run_event("proton100PeV", mode, 2026110001)
    state.update(complete=True, phase="complete", production_resumed=False)
    save(root/"STATUS.json", state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            status = Path(sys.argv[sys.argv.index("--output")+1])/"STATUS.json"
            if status.exists():
                data = json.loads(status.read_text())
                data.update(complete=False, error=repr(error))
                save(status, data)
        raise
