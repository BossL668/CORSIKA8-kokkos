"""Small application regression, not a full C7 physics/timing comparison.

Python only orchestrates tests/readback; c8_egs4_shower is a native executable.
Each case runs in a new process so unchanged C8 generator RNG initialization is
identical. Outputs and logs are retained under a unique build subdirectory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq


def fields(root, relative, names):
    table = pq.read_table(root / relative, columns=names)
    data = np.column_stack([table[name].to_numpy() for name in names])
    if not np.isfinite(data).all() or data.size == 0:
        raise AssertionError(f"Invalid serialized output: {relative}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="mixed_shower_", dir=args.root))
    cases = [("gamma", 22, .1, 4000.1, []), ("muon", 13, 2., 2000., []),
             ("proton", 2212, 10., 5000., ["--force-interaction"])]
    results = {"root": str(root), "scope": "single-seed host/device application integration, unthinned", "cases": {}}
    for name, pdg, energy, height, flags in cases:
        outputs = []
        for backend in ("host", "device"):
            destination = root / f"{name}_{backend}"
            command = [str(args.binary), "--table", str(args.table), "--pdg", str(pdg),
                       "--energy-GeV", str(energy), "--height-m", str(height), "-f", str(destination), *flags]
            if backend == "host":
                command.extend(["--host-reference", "--radio-backend", "cpu"])
            elif args.gpu_memory_fraction is not None:
                command.extend(["--gpu-memory-fraction", str(args.gpu_memory_fraction)])
            with (root / f"{name}_{backend}.log").open("w") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError(f"Application failed ({result.returncode}); see {root / (name+'_'+backend+'.log')}")
            metadata = json.loads((destination / "native_egs4_run.json").read_text())
            assert metadata["status"] == "completed" and metadata["scalar_em_steps"] == 0
            assert metadata["native_steps"] > 0
            if backend == "device":
                assert metadata["radio_backend_requested"] == "kokkos"
                assert metadata["radio_projected_tracks"] == metadata["native_radio_tracks"] > 0
                assert metadata["radio_coreas_contributions"] > 0 and metadata["radio_zhs_contributions"] > 0
                assert metadata["radio_fixed_point_overflows"] == 0
                if metadata["execution_space"] == "Cuda":
                    assert metadata["radio_execution_space"] == "Cuda" and metadata["radio_cuda_kernel_ms"] > 0
            if backend == "device" and args.gpu_memory_fraction is not None:
                assert metadata["execution_space"] == "Cuda" and metadata["gpu_memory_plan_active"]
                assert metadata["gpu_memory_fraction"] == args.gpu_memory_fraction
                assert (metadata["queue_capacity"] * metadata["gpu_bytes_per_history_bound"]
                        <= metadata["gpu_working_budget_bytes"])
            if name == "muon":
                assert metadata["scalar_muon_steps"] > 0 and metadata["c8_em_daughters"] > 0
            if name == "proton":
                assert metadata["scalar_hadron_steps"] > 0 and metadata["c8_em_daughters"] > 0
            arrays = {
                "em_profile": fields(destination, "profile/profile.parquet", ["photon", "electron", "positron"]),
                "deposition": fields(destination, "energyloss/dEdX.parquet", ["total"]),
                "CoREAS": fields(destination, "CoREAS/observers.parquet", ["Ex", "Ey", "Ez"]),
                "ZHS": fields(destination, "ZHS/observers.parquet", ["Ex", "Ey", "Ez"]),
            }
            assert all(np.linalg.norm(x) > 0 for x in arrays.values())
            # All five writer families must be materialized, even if no particle
            # reaches the observation plane in this particular small shower.
            particle_files = list((destination / "particles").glob("*.parquet"))
            assert particle_files
            for path in particle_files:
                pq.read_metadata(path)
            outputs.append((metadata, arrays))
        host, device = outputs
        differences = {}
        for kind, reference in host[1].items():
            other = device[1][kind]
            assert reference.shape == other.shape
            error = float(np.linalg.norm(other - reference) / np.linalg.norm(reference))
            differences[kind] = error
            assert error < 1.e-5, (name, kind, error)
        for counter in ("native_injections", "native_steps", "native_children", "scalar_steps",
                        "native_host_requests", "scalar_hadron_steps", "scalar_muon_steps"):
            assert host[0][counter] == device[0][counter], (name, counter)
        results["cases"][name] = {"host": host[0], "device": device[0], "relative_L2": differences}
        print(name, json.dumps(differences), flush=True)
    (root / "acceptance.json").write_text(json.dumps(results, indent=2) + "\n")
    print("ACCEPTED", root, flush=True)


if __name__ == "__main__":
    main()
