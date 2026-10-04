#!/usr/bin/env python3
"""Same-executable factorial control: host/device histograms x fresh/reused queue.

No shower physics changes. Compare raw Parquet tables and discrete counters;
timing is diagnostic only (small single-event runs, not an ensemble speedup).
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq


def compare(reference, candidate):
    result = {}
    for relative, columns in {
        "profile/profile.parquet": ["photon", "electron", "positron", "mu-", "mu+"],
        "energyloss/dEdX.parquet": ["total"],
        "CoREAS/observers.parquet": ["Ex", "Ey", "Ez"],
        "ZHS/observers.parquet": ["Ex", "Ey", "Ez"],
    }.items():
        a, b = (pq.read_table(p / relative) for p in (reference, candidate))
        # Validate every numeric column, including depth/time/observer indices.
        assert a.schema == b.schema and a.num_rows == b.num_rows
        metrics = {}
        for column in a.column_names:
            x, y = a[column].to_numpy(), b[column].to_numpy()
            if not np.issubdtype(x.dtype, np.number):
                assert np.array_equal(x, y)
                continue
            assert np.isfinite(x).all() and np.isfinite(y).all()
            delta = x.astype(float) - y.astype(float)
            # NumPy 2 weak scalar promotion can turn the floor into float32
            # zero for an empty/zero prefix. Accumulate norms in float64 and
            # divide Python floats, including when the reference is all zero.
            relative_l2 = float(np.linalg.norm(delta)) / max(float(np.linalg.norm(x.astype(float))), 1e-300)
            if relative.startswith(("CoREAS", "ZHS")):
                assert np.array_equal(x, y), (relative, column, relative_l2)
            else:
                assert relative_l2 < 1e-7, (relative, column, relative_l2)
            metrics[column] = {"relative_L2": relative_l2, "max_abs": float(np.max(np.abs(delta), initial=0))}
        result[relative] = metrics
    for relative in reference.glob("particles/*.parquet"):
        assert pq.read_table(relative).equals(pq.read_table(candidate / relative.relative_to(reference)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix="resident_output_", dir=a.root))
    report = {"root": str(root), "scope": "same-seed output and workspace equivalence, not speed acceptance", "cases": {}}
    cases = {
        "electron": ["--pdg", "11", "--energy-GeV", ".1", "--height-m", "4000.1"],
        "mixed_thinned": ["--pdg", "2212", "--energy-GeV", "100", "--height-m", "5000", "--force-interaction",
                          "--thin-threshold-GeV", "1", "--thin-max-weight", "2", "--muon-backend", "kokkos", "--muon-cut-GeV", ".05"],
    }
    for name, flags in cases.items():
        baseline = None
        baseline_meta = None
        report["cases"][name] = {}
        for profile, workspace, split in [("cpu", "fresh", False), ("cpu", "reuse", False),
                                          ("kokkos", "fresh", False), ("kokkos", "reuse", False),
                                          ("kokkos", "reuse", True)]:
            label = profile + "_" + workspace + ("_split" if split else "")
            out = root / (name + "_" + label)
            command = [str(a.binary), "--table", str(a.table), "-f", str(out), "--profile-backend", profile,
                       "--queue-workspace", workspace, "--radio-backend", "kokkos", *flags]
            if split:
                command.append("--split-rare-kernels")
            with (root / (out.name + ".log")).open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            m = json.loads((out / "native_egs4_run.json").read_text())
            assert m["status"] == "completed" and m["scalar_em_steps"] == 0
            if "initialization_stages_s" in m:
                stages = m["initialization_stages_s"]
                assert len(stages) == 8 and all(np.isfinite(x) and x >= 0 for x in stages.values())
                unaccounted = m["initialization_s"] - sum(stages.values())
                assert -1e-9 <= unaccounted < max(.01, .001 * m["initialization_s"])
            if split:
                assert m["split_rare_kernels_requested"] and m["queue_split_rare_waves"] > 0
            if profile == "kokkos":
                assert m["profile_accumulated_steps"] == m["native_steps"] > 0
                assert m["profile_execution_space"] == m["execution_space"]
                assert m["profile_invalid_records"] == m["profile_fixed_point_overflows"] == 0
                assert m["host_output_records"] < m["native_steps"]
            if baseline is None:
                baseline, baseline_meta = out, m
            for key in ["native_steps", "native_children", "native_host_requests", "native_radio_tracks", "native_observations",
                        "proposal_muon_specified_cpu_final_states", "proposal_muon_cpu_decays", "scalar_hadron_steps"]:
                if key in baseline_meta:
                    assert m[key] == baseline_meta[key], (name, label, key)
            report["cases"][name][label] = {"metadata": m, "comparison": compare(baseline, out)}
    (root / "ACCEPTANCE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root / "ACCEPTANCE.json")


if __name__ == "__main__":
    main()
