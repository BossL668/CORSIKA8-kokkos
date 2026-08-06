from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from watch_and_audit_radio_configuration_equivalence import audit_command  # noqa: E402


def test_audit_command_pins_exact_event_count_and_all_roots(tmp_path: Path) -> None:
    args = argparse.Namespace(
        audit_script=tmp_path / "audit.py",
        proposal_root=[tmp_path / "cpu_a", tmp_path / "cpu_b"],
        cuda_root=[tmp_path / "cuda"],
        expected_events=500,
        output=tmp_path / "result.json",
    )
    command = audit_command(args)
    assert command[0] == sys.executable
    assert command.count("--proposal-root") == 2
    assert command.count("--cuda-root") == 1
    assert command[command.index("--expected-events") + 1] == "500"
    assert command[command.index("--output") + 1] == str(args.output)
