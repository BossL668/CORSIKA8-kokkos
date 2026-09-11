#!/usr/bin/env python3
"""Exercise application preflight without constructing physics calculators."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((source / "configs/mountain/photon_1gev_box.yaml").read_text())
    cases = []

    def add(name, section, key, value, expected):
        config = copy.deepcopy(base)
        config[section][key] = value
        cases.append((name, config, expected))

    add("negative density", "material", "density_g_cm3", -1, "density must")
    add("unsupported composition", "material", "name", "granite", "silica")
    add("zero direction", "primary", "direction", [0, 0, 0], "direction norm")
    add("missed mountain", "primary", "position_m", [100, 0, 0], "misses mountain")
    add("forced photon", "primary", "sampling", "forced_vertex_cc", "requires a neutrino")
    add("exterior observer", "radio", "observers_m", [[100, 0, 0]], "inside mountain")
    add("excessive window", "radio", "duration_ns", 1.e9, "time window")
    add("nonfinite cut", "physics", "emcut_GeV", float("nan"), "transport cut")
    add("nonconvex input kind", "geometry", "shape", "dem", "convex triangular obj")
    for pdg, energy, message in ((12, 1., "CTW2011"), (14, 1.e4, "nu_e/anti-nu_e")):
        config = copy.deepcopy(base)
        config["primary"].update(pdg=pdg, energy_GeV=energy)
        cases.append((f"neutrino gate {pdg}/{energy}", config, message))

    results = []
    with tempfile.TemporaryDirectory(prefix="c8-mountain-config-") as directory:
        root = Path(directory)
        for i, (name, config, expected) in enumerate(cases):
            path = root / f"case_{i}.yaml"
            path.write_text(yaml.safe_dump(config))
            result = subprocess.run([str(binary), "--config", str(path), "--geometry-only"],
                                    text=True, capture_output=True, timeout=30)
            assert result.returncode != 0 and expected in result.stderr, (name, result)
            results.append({"case": name, "passed": True})
        for name in ("photon_1gev_box", "natural_nue_cc_tetrahedron", "forced_nue_cc_box"):
            path = source / "configs/mountain" / f"{name}.yaml"
            result = subprocess.run([str(binary), "--config", str(path), "--geometry-only"],
                                    text=True, capture_output=True, timeout=30, check=True)
            report = yaml.safe_load(result.stdout)
            assert report["chord_m"] > 0 and report["complete"] is False
            results.append({"case": name, "passed": True})
    print(json.dumps({"passed": True, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
