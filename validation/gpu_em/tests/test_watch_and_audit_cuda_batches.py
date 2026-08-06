from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import watch_and_audit_cuda_batches as watcher  # noqa: E402


@dataclass(frozen=True)
class FakeSourceRecord:
    backend: str
    root: str
    seed: int
    events: int
    executable_sha256: str
    table_sha256: str
    antenna_sha256: str
    observer_layout_sha256: str


def write_parquet_set(root: Path, *, second_weight: float = 1.0) -> None:
    showers = [0, 1]
    tables = {
        "profile/profile.parquet": {
            "shower": showers,
            "X": [1.0, 1.0],
            "charged": [1.0, 2.0],
            "hadron": [0.0, 0.0],
            "photon": [1.0, 1.0],
            "electron": [1.0, 1.0],
            "positron": [0.0, 1.0],
            "muplus": [0.0, 0.0],
            "muminus": [0.0, 0.0],
        },
        "production_profile/profile.parquet": {
            "shower": showers,
            "X": [1.0, 1.0],
            "pion": [0.0, 0.0],
            "kaon": [0.0, 0.0],
            "heavy": [0.0, 0.0],
            "hadron": [0.0, 0.0],
            "photon": [1.0, 1.0],
            "electron-positron": [1.0, 2.0],
            "muon": [0.0, 0.0],
            "all": [2.0, 3.0],
        },
        "energyloss/dEdX.parquet": {
            "shower": showers,
            "X": [1.0, 1.0],
            "total": [0.1, 0.2],
        },
        "particles/particles.parquet": {
            "shower": showers,
            "pdg": [11, -11],
            "kinetic_energy": [0.1, 0.2],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "nx": [0.0, 0.0],
            "ny": [0.0, 0.0],
            "nz": [-1.0, -1.0],
            "time": [1.0, 2.0],
            "weight": [1.0, second_weight],
        },
    }
    radio = {
        "shower": showers,
        "Time": [0.0, 0.0],
        "Ex": [0.0, 1.0e-9],
        "Ey": [0.0, 2.0e-9],
        "Ez": [0.0, 3.0e-9],
    }
    tables["CoREAS/observers.parquet"] = radio
    tables["ZHS/observers.parquet"] = radio
    for relative, columns in tables.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), path)


def manifest(root: Path, *, completed: int = 25, seed: int = 101) -> None:
    output = root / "batch_000" / "cuda"
    output.mkdir(parents=True)
    for algorithm in ("CoREAS", "ZHS"):
        config = output / algorithm / "config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(f"algorithm: {algorithm}\nsetting: fixed\n")
    value = {
        "status": "complete" if completed == 50 else "running",
        "completed_total_events": completed,
        "immutable_configuration": {
            "target_events": 50,
            "cuda_seed_start": 101,
            "primary_pdg": 2212,
            "energy_GeV": 1.0e5,
            "physics": {
                "zenith_deg": 0.0,
                "azimuth_deg": 0.0,
                "geomagnetic_model": "IGRF14",
                "geomagnetic_year": 2027.0,
                "em_cut_GeV": 5.0e-4,
                "em_thinning": 1.0e-6,
                "maximum_weight": 0.0,
                "had_cut_GeV": 0.3,
                "mu_cut_GeV": 0.3,
                "tau_cut_GeV": 0.3,
                "ring": 0,
            },
            "artifacts": {"antenna_file": {"sha256": "a" * 64}},
        },
        "batches": [
            {
                "status": "complete",
                "events": completed,
                "seed": seed,
                "cuda_output": str(output),
            }
        ],
    }
    (root / "campaign_manifest.json").write_text(json.dumps(value))


def test_audits_new_batch_from_immutable_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest(tmp_path)
    observed: list[SimpleNamespace] = []

    def fake_audit(root: Path, backend: str, *, expected: SimpleNamespace):
        assert backend == "cuda"
        observed.append(expected)
        return FakeSourceRecord(
            backend=backend,
            root=str(root),
            seed=101,
            events=25,
            executable_sha256="b" * 64,
            table_sha256="c" * 64,
            antenna_sha256="a" * 64,
            observer_layout_sha256="d" * 64,
        )

    monkeypatch.setattr(watcher, "audit_source", fake_audit)
    monkeypatch.setattr(
        watcher,
        "deep_scan_cuda_output",
        lambda root, events: {"checked_events": events},
    )
    cache: dict[str, dict] = {}
    status = watcher.audit_manifest(tmp_path, cache)
    assert status["status"] == "monitoring"
    assert status["audited_events"] == 25
    assert status["next_expected_seed"] == 126
    assert observed[0].energy_gev == pytest.approx(1.0e5)
    assert observed[0].em_thinning == pytest.approx(1.0e-6)
    assert observed[0].geomagnetic_model == "IGRF14"
    assert observed[0].maximum_weight == 0.0
    assert status["records"][0]["deep_scan"] == {"checked_events": 25}
    assert len(status["records"][0]["coreas_config_sha256"]) == 64
    assert len(status["records"][0]["zhs_config_sha256"]) == 64
    assert status["radio_config_sha256"] == {
        "coreas_config_sha256": status["records"][0]["coreas_config_sha256"],
        "zhs_config_sha256": status["records"][0]["zhs_config_sha256"],
    }


def test_cached_batch_is_not_reaudited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest(tmp_path)
    output = str((tmp_path / "batch_000" / "cuda").resolve())
    cache = {
        output: {
            "root": output,
            "seed": 101,
            "events": 25,
            "deep_scan": {"checked_events": 25},
        }
    }

    def unexpected(*args, **kwargs):
        raise AssertionError("cached batch was re-audited")

    monkeypatch.setattr(watcher, "audit_source", unexpected)
    status = watcher.audit_manifest(tmp_path, cache)
    assert status["audited_batches"] == 1
    assert len(cache[output]["coreas_config_sha256"]) == 64
    assert len(cache[output]["zhs_config_sha256"]) == 64


def test_rejects_noncontiguous_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest(tmp_path, seed=102)
    monkeypatch.setattr(watcher, "audit_source", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="not contiguous"):
        watcher.audit_manifest(tmp_path, {})


def test_rejects_radio_configuration_drift_between_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest(tmp_path, completed=50)
    manifest_value = json.loads((tmp_path / "campaign_manifest.json").read_text())
    second = tmp_path / "batch_001" / "cuda"
    for algorithm in ("CoREAS", "ZHS"):
        config = second / algorithm / "config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"algorithm: {algorithm}\nsetting: fixed\n")
    (second / "CoREAS" / "config.yaml").write_text(
        "algorithm: CoREAS\nsetting: changed\n"
    )
    manifest_value["batches"] = [
        {
            "status": "complete",
            "events": 25,
            "seed": 101,
            "cuda_output": str(tmp_path / "batch_000" / "cuda"),
        },
        {
            "status": "complete",
            "events": 25,
            "seed": 126,
            "cuda_output": str(second),
        },
    ]
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(manifest_value))

    def fake_audit(root: Path, backend: str, *, expected: SimpleNamespace):
        return FakeSourceRecord(
            backend=backend,
            root=str(root),
            seed=101 if root.name == "cuda" and root.parent.name == "batch_000" else 126,
            events=25,
            executable_sha256="b" * 64,
            table_sha256="c" * 64,
            antenna_sha256="a" * 64,
            observer_layout_sha256="d" * 64,
        )

    monkeypatch.setattr(watcher, "audit_source", fake_audit)
    monkeypatch.setattr(
        watcher,
        "deep_scan_cuda_output",
        lambda root, events: {"checked_events": events},
    )
    with pytest.raises(ValueError, match="different coreas_config_sha256"):
        watcher.audit_manifest(tmp_path, {})


def test_deep_scan_accepts_complete_finite_outputs(tmp_path: Path) -> None:
    write_parquet_set(tmp_path)
    report = watcher.deep_scan_cuda_output(tmp_path, 2)
    assert report["particles/particles.parquet"]["weight_minimum"] == 1.0
    assert report["particles/particles.parquet"][
        "direction_norm_max_abs_error"
    ] == 0.0
    assert report["CoREAS/observers.parquet"]["rows"] == 2


def test_deep_scan_rejects_nonpositive_weight(tmp_path: Path) -> None:
    write_parquet_set(tmp_path, second_weight=0.0)
    with pytest.raises(ValueError, match="non-positive weight"):
        watcher.deep_scan_cuda_output(tmp_path, 2)
