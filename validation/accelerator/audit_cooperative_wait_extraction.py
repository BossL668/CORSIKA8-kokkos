#!/usr/bin/env python3
"""Audit unchanged physics plus the explicitly enumerated optional wait hook."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
reports = {}
for species, expected in (("Photon", 3), ("Lepton", 4)):
    source = root / f"corsika/accelerator/em/kokkos/KokkosResident{species}Cascade.hpp"
    text = source.read_text()
    start = text.index(f"gpu::em::Resident{species}CascadeResult runResident")
    head, body = text[:start], text[start:]
    body, count = re.subn(
        r'waitResidentExecution\(execution,\s*("download resident Kokkos [^"]+"), cooperative_wait\);',
        r'execution.fence(\1);', body)
    assert count == expected, (species, count, expected)
    body, signature = re.subn(
        r',\s*ResidentExecutionWait<ExecutionSpace>\* const cooperative_wait = nullptr',
        '', body)
    assert signature == 1
    # The existing frozen-source audit verifies every remaining host-loop
    # token and every producer/profile/radio kernel, not just selected formulae.
    with tempfile.TemporaryDirectory(prefix="c8-wait-audit-") as tmp:
        normalized = Path(tmp) / "normalized.hpp"
        normalized.write_text(head + body)
        report = Path(tmp) / "report.json"
        subprocess.run([
            sys.executable, str(root / f"validation/accelerator/audit_{species.lower()}_front_extraction.py"),
            "--baseline", str(a.baseline / f"Resident{species}CascadeBefore.hpp"),
            "--source", str(normalized), "--output", str(report)], check=True)
        reports[species] = json.loads(report.read_text())
        reports[species]["enumerated_optional_waits"] = count
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x") as f:
    json.dump(reports, f, indent=2)
print("PASS: frozen kernels/loops identical except helper extraction and enumerated optional waits")
