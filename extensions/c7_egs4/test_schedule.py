"""Readback/serial replay for the C++ scheduler, not its runtime coordinator.

The reference reuses each shard, namespace and seed. An unsplit stochastic
shower has different allocated child IDs and is NOT an exact-array oracle.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pyarrow.parquet as pq

PRODUCTS = {"profile/profile.parquet": "X", "energyloss/dEdX.parquet": "X",
            "CoREAS/observers.parquet": "Time", "ZHS/observers.parquet": "Time"}


def arrays(path):
    table = pq.read_table(path)
    result = {k: table[k].to_numpy() for k in table.column_names}
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def run(command, cwd, log, success=True):
    with log.open("x") as stream:
        p = subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=900)
    assert (p.returncode == 0) == success, (p.returncode, log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scheduler", type=Path)
    p.add_argument("worker", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--devices")
    a = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix="thinning_schedule_", dir=a.root))
    results = {"root": str(root), "scope": "small integration, not high-energy statistics or scaling", "cases": {}}
    for case, physics in [
        ("gamma", ["--pdg", "22", "--energy-GeV", ".03", "--source-count", "8"]),
        ("proton", ["--pdg", "2212", "--energy-GeV", "10", "--force-interaction"]),
        ("empty_shards", ["--pdg", "22", "--energy-GeV", ".0002"])]:
        dest = root / case
        common = ["--table", a.table, "--height-m", "4000.1", "--thin-threshold-GeV", ".1", "--thin-max-weight", "8"]
        mode = ["--devices", a.devices] if a.devices else ["--cpu-workers", "4"]
        run([a.scheduler, "--worker", a.worker, *common, *physics, *mode, "-f", dest], root, root / (case+".log"))
        report = json.loads((dest / "COMPLETE.json").read_text())
        counts = report["partition"]["counts"]
        assert len(counts) == 4 and sum(counts) == report["prefix"]["native_injections"]
        if case == "gamma":
            assert min(counts) > 0
            assert sum(w["thinning_removed"] for w in report["workers"]) > 0
        if case == "proton":
            assert report["prefix"]["scalar_hadron_steps"] > 0
        if case == "empty_shards":
            assert counts.count(0) == 3
        # Independent numerical sum of every output family. Field components,
        # not squared amplitudes or per-worker fluence, must be merged.
        parts = [dest / "prefix", *[dest / f"worker_{i}" for i in range(1, 5)]]
        merge_error = {}
        for product, grid in PRODUCTS.items():
            merged = arrays(dest / "merged" / product)
            inputs = [arrays(part / product) for part in parts]
            error = 0.
            for key, v in merged.items():
                if key in ("shower", grid):
                    assert all(np.array_equal(v, entry[key]) for entry in inputs)
                else:
                    expected = np.sum([entry[key].astype(np.float64) for entry in inputs], axis=0)
                    scale = max(float(np.max(np.abs(expected))), 1.e-100)
                    error = max(error, float(np.max(np.abs(v-expected))) / scale)
            assert error < 1.e-12, (product, error)
            merge_error[product] = error
        # Repeat workers sequentially on the CPU with identical identities.
        replay_errors, branch_counter_differences = [], []
        flags = [x for x in physics if x != "--force-interaction"]
        for i in range(1, 5):
            replay = root / f"{case}_serial_{i}"
            run([a.worker, *common, *flags, "--frontier", dest / f"shard_{i}.txt",
                 "--worker-id", i, "--host-reference", "-f", replay], root, root / f"{case}_serial_{i}.log")
            record = json.loads((replay / "native_egs4_run.json").read_text())
            concurrent = report["workers"][i-1]
            for key in ("native_steps", "native_children", "native_host_requests", "scalar_steps",
                        "thinning_removed", "frontier_roots_consumed"):
                assert record[key] == concurrent[key], (case, i, key)
            branch_differences = {k: {"worker": concurrent[k], "cpu_replay": record[k]}
                for k in ("thinning_hillas", "thinning_statistical") if record[k] != concurrent[k]}
            if not a.devices:
                assert not branch_differences  # identical CPU arithmetic
            branch_counter_differences.append(branch_differences)
            difference = {}
            for product, grid in PRODUCTS.items():
                ref, other = arrays(replay / product), arrays(dest / f"worker_{i}" / product)
                keys = [k for k in ref if k not in ("shower", grid)]
                x = np.concatenate([ref[k] for k in keys]); y = np.concatenate([other[k] for k in keys])
                difference[product] = float(np.linalg.norm(y-x) / max(np.linalg.norm(x), 1.e-100))
                assert difference[product] < 1.e-4, (case, i, product, difference[product])
            replay_errors.append(difference)
        results["cases"][case] = {"topology": report["topology"], "counts": counts,
                                  "merge_max_scaled_error": merge_error, "cpu_replay_relative_L2": replay_errors,
                                  "thinning_branch_counter_differences": branch_counter_differences,
                                  "workers": report["workers"], "prefix": report["prefix"]}
        print(case, "accepted", counts, flush=True)
    # Failed worker leaves logs but cannot publish completion or a merged shower.
    bad = root / "failed_worker"
    run([a.scheduler, "--worker", "/usr/bin/false", "--table", a.table,
         "--cpu-workers", "4", "-f", bad], root, root / "failure.log", success=False)
    assert not (bad / "COMPLETE.json").exists() and not (bad / "merged").exists()
    results["failed_child_does_not_publish_complete"] = True
    (root / "acceptance.json").write_text(json.dumps(results, indent=2)+"\n")
    print("ACCEPTED", root, flush=True)


if __name__ == "__main__":
    main()
