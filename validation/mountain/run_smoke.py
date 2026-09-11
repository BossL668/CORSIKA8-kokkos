#!/usr/bin/env python3
"""Bounded integration acceptance, not a physics-equivalence ensemble.

Uses new output directories, records every command/return code, and never
changes caches, production processes or existing simulation output.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
import yaml


def validate_output(path, kind, accelerated):
    report = yaml.safe_load((path / "mountain_run.yaml").read_text())
    assert report["complete"] is True
    diag = report["diagnostics"]
    assert diag["boundary_violations"] == 0
    assert np.isfinite(diag["deposited_energy_GeV"])
    assert diag["deposited_energy_GeV"] >= 0
    with (path / "mountain" / "profile.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 200
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())
    if kind == "natural":
        assert report["neutrino"]["interaction_count"] == 0
        assert diag["escaped_particles"] == 1
        assert diag["deposited_energy_GeV"] == 0
        assert diag["escaped_weighted_total_energy_GeV"] == 1.e4
    else:
        if accelerated:
            assert diag["accelerated_steps"] > 0, "Kokkos did not transport particles"
        if kind == "forced":
            assert report["neutrino"]["interaction_count"] >= 1
            assert report["neutrino"]["geometry_probability_weight_provided"] is False
        # Check actual saved E fields after the normal ZHS differentiation.
        for name in ("CoREAS", "ZHS"):
            files = list((path / name).rglob("*.parquet"))
            assert files, f"{name} parquet missing"
            numeric = []
            for file in files:
                table = pq.read_table(file)
                for column in table.column_names:
                    if column.lower() in ("ex", "ey", "ez", "e_x", "e_y", "e_z"):
                        values = np.asarray(table[column].to_pylist(), dtype=float)
                        assert np.isfinite(values).all()
                        numeric.append(values)
            # Different beta5 output versions use different array columns;
            # unknown schemas must be explicitly inspected, never auto-passed.
            assert numeric, f"unrecognized {name} waveform schema"
            assert any(np.any(x != 0) for x in numeric), f"{name} signal is identically zero"
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--execution", choices=["openmp", "cuda", "hip", "sycl"], default="openmp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aux-cache", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--kinds", nargs="+", choices=["photon", "natural", "forced"], default=["photon", "natural", "forced"])
    parser.add_argument("--arms", nargs="+", choices=["proposal", "kokkos"], default=["proposal", "kokkos"])
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    configs = source / "configs" / "mountain"
    inputs = {
        "photon": yaml.safe_load((configs / "photon_1gev_box.yaml").read_text()),
        "natural": yaml.safe_load((configs / "natural_nue_cc_tetrahedron.yaml").read_text()),
        "forced": yaml.safe_load((configs / "forced_nue_cc_box.yaml").read_text()),
    }
    # A short rock chord exercises true CC generation and both transports
    # without a 10 TeV fully contained, unthinned production shower.
    forced = inputs["forced"]
    forced["geometry"] = dict(shape="box", center_m=[0, 0, 0], half_lengths_m=[0.1, 0.1, 0.1])
    forced["primary"]["position_m"] = [0, 0, 0]
    forced["radio"]["observers_m"] = [[0.03, 0.01, 0.02]]
    summary = {"complete": False, "scope": "integration smoke, not statistical equivalence", "runs": []}
    status = args.output / "acceptance.json"
    env = dict(os.environ, OMP_NUM_THREADS="2", OMP_PROC_BIND="false")
    try:
        for kind in args.kinds:
            config = args.output / f"{kind}.yaml"
            config.write_text(yaml.safe_dump(inputs[kind]))
            for arm in args.arms:
                output = args.output / f"{kind}_{arm}"
                cmd = [str(args.binary), "--config", str(config), "--output", str(output),
                       "--em-backend", arm, "--seed", "67101"]
                if arm == "kokkos":
                    cmd += ["--kokkos-execution", args.execution, "--gpu-min-batch", "64",
                            "--gpu-resident-batch-limit", "4096", "--gpu-memory-fraction", "0.1",
                            "--gpu-aux-cache-dir", str(args.aux_cache.resolve())]
                    if args.execution == "openmp":
                        cmd += ["--kokkos-num-threads", "2"]
                record = {"kind": kind, "arm": arm, "command": cmd, "passed": False}
                summary["runs"].append(record)
                status.write_text(json.dumps(summary, indent=2))
                start = time.monotonic()
                with (args.output / f"{kind}_{arm}.log").open("w") as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            env=env, timeout=args.timeout, check=False)
                record.update(returncode=result.returncode, wall_seconds=time.monotonic()-start)
                if result.returncode:
                    raise RuntimeError(f"{kind}/{arm}: exit {result.returncode}; inspect log")
                validate_output(output, kind, arm == "kokkos")
                record["passed"] = True
                status.write_text(json.dumps(summary, indent=2))
        summary["complete"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        status.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
