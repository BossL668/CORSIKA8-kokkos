#!/usr/bin/env python3
"""Repeat campaigns exactly; independently check event reset on pure EM.

C8 host generators continue their RNG/state as in original multi-shower C8.
An isolated later seed is deliberately NOT claimed to reproduce a campaign event.
The rejected partial-reseed A/B/A diagnostics are retained separately.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from test_resident_output import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix="persistent_events_", dir=a.root))
    report = {"root": str(root), "cases": {}}
    for name, flags in {
        "electron": ["--pdg", "11", "--energy-GeV", ".1", "--height-m", "4000.1"],
        "mixed": ["--pdg", "2212", "--energy-GeV", "100", "--height-m", "5000", "--force-interaction",
                  "--thin-threshold-GeV", "1", "--thin-max-weight", "2", "--muon-backend", "kokkos", "--muon-cut-GeV", ".05"],
    }.items():
        outputs = {}
        for label, seeds in [("warm", [731, 732, 731]), ("replay", [731, 732, 731]), ("cold_a", [731])]:
            plan = [{"seed": seed, "output": str(root / f"{name}_{label}_{i}")} for i, seed in enumerate(seeds)]
            path = root / f"{name}_{label}.json"
            path.write_text(json.dumps(plan))
            command = [str(a.binary), "--table", str(a.table), "-f", plan[0]["output"],
                       "--event-plan", str(path), "--profile-backend", "kokkos", "--queue-workspace", "reuse",
                       "--radio-backend", "kokkos", *flags]
            with (root / f"{name}_{label}.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
            outputs[label] = [Path(row["output"]) for row in plan]
        warm = outputs["warm"]
        metas = [json.loads((out / "native_egs4_run.json").read_text()) for out in warm]
        assert len({m["process_pid"] for m in metas}) == 1
        for i, m in enumerate(metas):
            done = json.loads((warm[i] / "EVENT_DONE.json").read_text())
            assert done["event_index"] == m["event_index"] == i and done["pid"] == m["process_pid"]
            assert done["seed"] == m["seed"]
            assert m["native_session_reused"] == (i > 0) and m["persistent_process"]
            assert m["profile_invalid_records"] == m["profile_fixed_point_overflows"] == 0
            assert m["profile_accumulated_steps"] == m["native_steps"] > 0
            assert m["radio_fixed_point_overflows"] == 0
        pairs = [(outputs["cold_a"][0], warm[0]), *zip(warm, outputs["replay"])]
        if name == "electron":
            pairs.append((warm[0], warm[2]))
        assert all("campaign-continuous" in m["event_rng_policy"] for m in metas)
        assert len({m["host_campaign_seed"] for m in metas}) == 1
        comparisons = []
        for left, right in pairs:
            lm, rm = [json.loads((out / "native_egs4_run.json").read_text()) for out in (left, right)]
            for key in ["native_steps", "native_children", "native_host_requests", "native_radio_tracks", "native_observations",
                        "proposal_muon_specified_cpu_final_states", "proposal_muon_cpu_decays", "scalar_hadron_steps"]:
                if key in lm:
                    assert lm[key] == rm[key], (name, left, right, key, lm[key], rm[key])
            comparisons.append({"left": str(left), "right": str(right), "outputs": compare(left, right)})
        report["cases"][name] = {"metadata": metas, "comparisons": comparisons}
    (root / "ACCEPTANCE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root / "ACCEPTANCE.json")


if __name__ == "__main__":
    main()
