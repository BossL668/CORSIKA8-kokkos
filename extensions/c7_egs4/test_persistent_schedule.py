"""Five resident processes: exact campaign replay plus first-event singleton.

Host generators use continuous streams, like original multi-shower C8.
Later events require reproducing the preceding campaign, not just a seed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from test_resident_output import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scheduler", type=Path)
    p.add_argument("worker", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--devices")
    a = p.parse_args()
    root = Path(tempfile.mkdtemp(prefix="persistent_schedule_", dir=a.root))
    topology = ["--devices", a.devices] if a.devices else ["--cpu-workers", "4"]
    common = [str(a.scheduler), "--worker", str(a.worker), "--table", str(a.table), *topology,
              "--pdg", "2212", "--energy-GeV", "100", "--height-m", "5000", "--force-interaction",
              "--thin-threshold-GeV", "1", "--thin-max-weight", "2", "--muon-backend", "kokkos", "--muon-cut-GeV", ".05",
              "--profile-backend", "kokkos", "--queue-workspace", "reuse", "--radio-backend", "kokkos", "--persistent"]
    for label, seeds in [("warm", "731,732,731"), ("replay", "731,732,731"), ("cold", "731")]:
        with (root / (label + ".log")).open("w") as log:
            subprocess.run([*common, "--event-seeds", seeds, "-f", str(root / label)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    warm = json.loads((root / "warm/COMPLETE.json").read_text())
    assert warm["status"] == "completed" and len(set(warm["process_pids"])) == 5
    first = root / "warm/event_0_seed_731"
    cold = root / "cold/event_0_seed_731"
    results = {}
    for relative in ["prefix", "worker_1", "worker_2", "worker_3", "worker_4", "merged"]:
        results[relative] = {"first_cold_warm": compare(cold / relative, first / relative), "campaign_replay": []}
        for i,seed in enumerate([731,732,731]):
            left=root / f"warm/event_{i}_seed_{seed}"
            right=root / f"replay/event_{i}_seed_{seed}"
            assert (left / "frontier.txt").read_bytes() == (right / "frontier.txt").read_bytes()
            results[relative]["campaign_replay"].append(compare(left / relative, right / relative))
    (root / "ACCEPTANCE.json").write_text(json.dumps({"root": str(root), "campaign": warm, "outputs": results}, indent=2))
    print(root / "ACCEPTANCE.json")


if __name__ == "__main__":
    main()
