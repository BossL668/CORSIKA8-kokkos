#!/usr/bin/env python3
"""Verify frozen source/binary boundaries; describe baseline output artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--photon-front", action="store_true")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
source = Path(__file__).resolve().parents[2]
baseline = json.loads((args.baseline / "baseline.json").read_text())

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

changed = []
for text, expected in baseline["files"].items():
    path = Path(text)
    if not path.is_file() or digest(path) != expected:
        changed.append(text)
allowed = {str(source / relative) for relative in (
    "corsika/accelerator/em/KokkosRuntime.hpp",
    "src/accelerator/em/kokkos/KokkosRuntime.cpp",
    "tests/accelerator/CMakeLists.txt")}
if args.photon_front:
    allowed.update(str(source / relative) for relative in (
        "corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp",
        "corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp"))
report = dict(changed_preexisting_files=changed,
              unexpected_changed_files=sorted(set(changed) - allowed),
              installed_binaries_unchanged=not any("/install/" in p for p in changed),
              scope=("stage B1 photon front; complete cooperative EM/radio not integrated"
                     if args.photon_front else "stage A foundations; cooperative EM/radio not integrated"),
              baseline_cases={}, artifacts={})
for case in ("cuda", "openmp", "proposal"):
    root = args.baseline / ("photon-" + case)
    command = json.loads((args.baseline / ("guarded-photon-" + case) / "summary.json").read_text())
    row = dict(command_completed=command["pass"], N=2,
               peak_sampled_tree_rss_bytes=command["peak_tree_rss_bytes"],
               elapsed_s=command["elapsed_s"], execution=case)
    summary = root / "gpu_em/summary.yaml"
    if summary.is_file():
        showers = yaml.safe_load(summary.read_text())
        row["complete"] = len(showers) == 2 and all(v.get("complete") for v in showers.values())
        row["second_shower_reused_backend"] = showers["shower_1"]["statistics"]["backend_lifecycle"]["reused"]
    row["parquet_files"] = len(list(root.rglob("*.parquet")))
    report["baseline_cases"][case] = row
    for file in sorted(root.rglob("*")):
        if file.is_file():
            report["artifacts"][str(file.relative_to(args.baseline))] = digest(file)
for file in sorted(args.baseline.glob("photon-*-trace.csv")):
    report["artifacts"][file.name] = digest(file)
for file in sorted(args.baseline.glob("photon-proposal-tape*")):
    if file.is_file():
        report["artifacts"][file.name] = digest(file)
report["production_active"] = subprocess.run(
    ["systemctl", "--user", "is-active", "c8-beta5-uhe100-cuda-persistent.service"],
    capture_output=True, text=True).stdout.strip()
report["boundary_pass"] = not report["unexpected_changed_files"] and report["installed_binaries_unchanged"]
with (args.output or args.baseline / "boundary_and_output_audit.json").open("x") as out:
    json.dump(report, out, indent=2)
print(json.dumps({k:v for k,v in report.items() if k != "artifacts"}, indent=2))
if not report["boundary_pass"]:
    raise SystemExit(1)
