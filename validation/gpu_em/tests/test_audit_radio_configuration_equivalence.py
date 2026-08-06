from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import audit_radio_configuration_equivalence as audit  # noqa: E402


def write_source(
    root: Path,
    backend: str,
    *,
    events: int = 2,
    duration: float = 400.0,
) -> None:
    root.mkdir(parents=True)
    (root / "validation_provenance.json").write_text(
        json.dumps({"schema_version": 1, "backend": backend})
    )
    (root / "summary.yaml").write_text(
        yaml.safe_dump({"showers": events})
    )
    for algorithm in ("CoREAS", "ZHS"):
        directory = root / algorithm
        directory.mkdir()
        (directory / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "algorithm": algorithm,
                    "observers": {
                        "a0": {
                            "location": [100.0, 0.0, 0.0],
                            "duration": duration,
                            "number of bins": 400,
                            "sampling frequency": 1.0,
                        }
                    },
                },
                sort_keys=False,
            )
        )


def test_accepts_byte_identical_cpu_cuda_radio_configuration(
    tmp_path: Path,
) -> None:
    proposal = tmp_path / "proposal" / "shard_000"
    cuda = tmp_path / "cuda" / "batch_000"
    write_source(proposal, "proposal")
    write_source(cuda, "cuda")
    report = audit.audit_configuration_equivalence(
        [tmp_path / "proposal"],
        [tmp_path / "cuda"],
        expected_events=2,
    )
    assert report["status"] == "equivalent"
    assert report["events_by_backend"] == {"proposal": 2, "cuda": 2}
    assert len(report["configuration_sha256"]["coreas_config_sha256"]) == 64


def test_rejects_cpu_cuda_radio_configuration_drift(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal" / "shard_000"
    cuda = tmp_path / "cuda" / "batch_000"
    write_source(proposal, "proposal")
    write_source(cuda, "cuda", duration=401.0)
    with pytest.raises(ValueError, match="radio setting differs"):
        audit.audit_configuration_equivalence(
            [tmp_path / "proposal"],
            [tmp_path / "cuda"],
        )


def test_rejects_incomplete_expected_event_count(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal" / "shard_000"
    cuda = tmp_path / "cuda" / "batch_000"
    write_source(proposal, "proposal")
    write_source(cuda, "cuda")
    with pytest.raises(ValueError, match="expected 500"):
        audit.audit_configuration_equivalence(
            [tmp_path / "proposal"],
            [tmp_path / "cuda"],
            expected_events=500,
        )
