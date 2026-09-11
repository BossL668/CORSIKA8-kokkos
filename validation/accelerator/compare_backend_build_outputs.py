#!/usr/bin/env python3
"""Strict same-execution-space comparison of build-matrix shower outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from compare_event_memory_metadata import compare
from run_backend_build_matrix import save


def compare_outputs(reference, candidate):
    result = {"reference": str(reference), "candidate": str(candidate), "arrays": []}
    for suffix in ("*.parquet", "*.npz"):
        files = sorted(p.relative_to(reference) for p in reference.rglob(suffix))
        assert files == sorted(p.relative_to(candidate) for p in candidate.rglob(suffix))
        for rel in files:
            if rel.suffix == ".parquet":
                a = pq.read_table(reference / rel, use_threads=False)
                b = pq.read_table(candidate / rel, use_threads=False)
                equal = a.equals(b, check_metadata=False)
            else:
                with np.load(reference / rel) as a, np.load(candidate / rel) as b:
                    equal = set(a.files) == set(b.files) and all(
                        np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
            result["arrays"].append({"file": str(rel), "equal": bool(equal)})
    metadata = compare(reference, candidate)
    oracle = []
    for root in (reference, candidate):
        summary = root / "gpu_em/summary.yaml"
        # Scalar PROPOSAL has no accelerator output. Its other summaries and
        # all physical arrays are still compared strictly above.
        oracle.append(yaml.load(summary.read_text(), Loader=yaml.CSafeLoader)
                      if summary.is_file() else {})
    ignored, unexpected = [], []
    for item in metadata["unexpected"]:
        path = item["path"].split(".")
        # Same documented wall-time-derived diagnostic as the ZHS acceptance:
        # this oracle does not dispatch production particles.
        retrospective = (
            item["file"] == "gpu_em/summary.yaml"
            and path[1:3] == ["statistics", "hadronic_worker_oracle"]
            and ((len(path) == 4 and path[-1] == "batches")
                 or (len(path) == 6 and path[3] == "worker_loads" and path[-1] == "batches"))
            and all(m.get(path[0], {}).get("statistics", {}).get("hadronic_worker_oracle", {}).get("mode")
                    == "retrospective_measured_final_state_only" for m in oracle))
        (ignored if retrospective else unexpected).append(item)
    metadata["walltime_dependent_retrospective_batches"] = ignored
    metadata["unexpected"] = unexpected
    metadata["pass_"] = not unexpected
    result["metadata"] = metadata
    result["pass"] = bool(result["arrays"]) and all(x["equal"] for x in result["arrays"]) and metadata["pass_"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    result = compare_outputs(args.reference.resolve(), args.candidate.resolve())
    save(args.report, result)
    print(json.dumps({"pass": result["pass"], "arrays": len(result["arrays"]),
                      "unequal": [x for x in result["arrays"] if not x["equal"]],
                      "metadata_unexpected": result["metadata"]["unexpected"]}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
