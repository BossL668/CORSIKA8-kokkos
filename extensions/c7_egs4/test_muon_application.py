"""Small mixed-backend integration; not a muon physics equivalence ensemble.

The simulation/scheduler are native binaries. Python only reads test output.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pyarrow.parquet as pq


def check_output(path):
    stats = json.loads((path / "native_egs4_run.json").read_text())
    assert stats["status"] == "completed" and stats["scalar_em_steps"] == 0
    assert stats["proposal_muon_steps"] > 0 and stats["proposal_muon_waves"] > 0
    assert stats["native_steps"] > 0
    assert stats["radio_projected_tracks"] == stats["native_radio_tracks"] > 0
    assert stats["radio_fixed_point_overflows"] == 0
    assert stats["proposal_muon_memory_spills"] == 0
    if stats["execution_space"] == "Cuda":
        assert stats["proposal_muon_execution_space"] == "cuda"
        assert stats["radio_execution_space"] == "Cuda" and stats["radio_cuda_kernel_ms"] > 0
    norms = {}
    for relative in ("profile/profile.parquet", "energyloss/dEdX.parquet", "CoREAS/observers.parquet", "ZHS/observers.parquet"):
        table = pq.read_table(path / relative)
        assert table.num_rows > 0
        for name in table.column_names:
            data = table[name].to_numpy()
            if data.dtype.kind in "fiu":
                assert np.isfinite(data).all(), (relative, name)
        if relative.startswith(("CoREAS/", "ZHS/")):
            field = np.column_stack([table[c].to_numpy() for c in ("Ex", "Ey", "Ez")])
            norms[relative] = float(np.linalg.norm(field))
            assert norms[relative] > 0
    return {"statistics": stats, "field_norms": norms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("table", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="proposal_muon_", dir=args.root))
    results = {}
    for name, pdg, energy, height, count in (
        ("mu_minus", 13, 10., 2000., 1),
        ("mu_plus", -13, 10., 2000., 1),
        ("mu_decay", 13, .3, 5000., 1),
        ("mu_radiative", 13, 1000., 5000., 1),
        # Natural selected-loss CPU final states. The pre-fix .5 MeV
        # calculator override fails its interaction-hash check on this case.
        ("mu_specified_radiative", 13, 100000., 20000., 1),
    ):
        destination = root / name
        command = [str(args.binary), "--table", str(args.table), "--muon-backend", "kokkos",
                   "--muon-cut-GeV", ".05", "--pdg", str(pdg), "--energy-GeV", str(energy),
                   "--height-m", str(height), "--source-count", str(count), "-f", str(destination)]
        if name == "mu_specified_radiative":
            command += ["--thin-threshold-GeV", ".1", "--thin-max-weight", "100"]
        with (root / f"{name}.log").open("w") as log:
            run = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if run.returncode:
            raise RuntimeError(f"Failed {name}; see {root / (name + '.log')}")
        result = check_output(destination)
        stats = result["statistics"]
        if name in ("mu_minus", "mu_plus"):
            assert stats["proposal_muon_em_handoffs"] > 0
        if name == "mu_decay":
            assert stats["proposal_muon_cpu_decays"] > 0
        if name == "mu_specified_radiative":
            assert stats["proposal_muon_specified_cpu_final_states"] > 0
        results[name] = result
        print(name, "muon_steps", stats["proposal_muon_steps"], "handoffs", stats["proposal_muon_em_handoffs"],
              "decays", stats["proposal_muon_cpu_decays"], "CPU final states", stats["proposal_muon_specified_cpu_final_states"], flush=True)
    (root / "ACCEPTANCE.json").write_text(json.dumps({"scope": __doc__, "cases": results}, indent=2) + "\n")
    print("ACCEPTED", root, flush=True)


if __name__ == "__main__":
    main()
