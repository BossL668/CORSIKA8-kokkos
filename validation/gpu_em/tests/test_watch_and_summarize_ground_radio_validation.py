from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.gpu_em.watch_and_summarize_ground_radio_validation import (
    prerequisite_state,
    summary_command,
)


def args(tmp_path: Path) -> argparse.Namespace:
    names = (
        "finalization",
        "comparison",
        "ground_attribution",
        "radio_configuration",
        "radio_waveform_oracle",
        "radio_pulse_oracle",
        "independent_radio_pulses",
    )
    value = argparse.Namespace(
        summarizer=tmp_path / "summarizer.py",
        finalizer_status=tmp_path / "finalizer_status.json",
        radio_config_watcher_status=tmp_path / "config_status.json",
        radio_waveform_status=tmp_path / "waveform_status.json",
        radio_pulse_status=tmp_path / "pulse_status.json",
        expected_events=500,
        familywise_alpha=0.05,
        output=tmp_path / "output",
    )
    for name in names:
        setattr(value, name, tmp_path / f"{name}.json")
    return value


def test_ready_only_after_all_statuses_and_artifacts_complete(tmp_path: Path) -> None:
    value = args(tmp_path)
    statuses = (
        value.finalizer_status,
        value.radio_config_watcher_status,
        value.radio_waveform_status,
        value.radio_pulse_status,
    )
    for path in statuses:
        path.write_text(json.dumps({"status": "complete"}))
    for name in (
        "finalization",
        "comparison",
        "ground_attribution",
        "radio_configuration",
        "radio_waveform_oracle",
        "radio_pulse_oracle",
        "independent_radio_pulses",
    ):
        getattr(value, name).write_text(json.dumps({"status": "complete"}))
    state = prerequisite_state(value)
    assert state["ready"]
    assert not state["terminal_failure"]


def test_failed_oracle_is_terminal(tmp_path: Path) -> None:
    value = args(tmp_path)
    value.radio_waveform_status.write_text(json.dumps({"status": "failed"}))
    state = prerequisite_state(value)
    assert not state["ready"]
    assert state["terminal_failure"]


def test_summary_command_carries_exact_event_count(tmp_path: Path) -> None:
    value = args(tmp_path)
    command = summary_command(value)
    assert command[0] == sys.executable
    assert command[command.index("--expected-events") + 1] == "500"
    assert command[command.index("--output") + 1] == str(value.output)
