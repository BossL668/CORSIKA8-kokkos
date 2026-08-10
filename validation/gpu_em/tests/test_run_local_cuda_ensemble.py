from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "run_local_cuda_ensemble.py"
SPEC = importlib.util.spec_from_file_location("run_local_cuda_ensemble", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments() -> argparse.Namespace:
    return argparse.Namespace(
        executable=Path("/tmp/cuda"),
        proposal_executable=Path("/tmp/proposal"),
        table=Path("/tmp/table"),
        physics_runner=Path("/tmp/runner"),
        output_root=Path("/tmp/output"),
        existing_cuda=[],
        reference_proposal_root=Path("/tmp/references"),
        allow_mixed_proposal_builds=True,
        defer_reference_comparison=False,
        antenna_file=Path("/tmp/antennas"),
        flupro=Path("/tmp/fluka"),
        target_events=500,
        batch_events=25,
        cuda_seed_start=10400001,
        energy_gev=1.0e8,
        primary_pdg=2212,
        zenith_deg=47.0,
        azimuth_deg=180.0,
        geomagnetic_model="IGRF13",
        geomagnetic_year=2025.0,
        em_cut_gev=0.0005,
        em_thinning=1.0e-4,
        maximum_weight=0.0,
        had_cut_gev=0.3,
        mu_cut_gev=0.3,
        tau_cut_gev=0.3,
        shower_core_x_m=0.0,
        shower_core_y_m=0.0,
        ring=0,
        gpu_device=0,
        gpu_min_batch=4096,
        gpu_memory_fraction=0.70,
        gpu_table_tolerance=0.001,
        gpu_radio_field_limit=1.0,
        cuda_hadronic_workers=4,
        cuda_hadronic_min_batch=64,
        cuda_hadronic_target_batch_ms=5.0,
        cuda_hadronic_max_batch=256,
        label_prefix="",
    )


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_batch_command_propagates_campaign_physics() -> None:
    args = arguments()
    command = MODULE.batch_command(
        args,
        Path("/tmp/output/batch_000"),
        25,
        args.cuda_seed_start,
        [Path(f"/tmp/ref_{index}") for index in range(25)],
    )

    assert option(command, "--energy-gev") == "100000000"
    assert option(command, "--zenith-deg") == "47"
    assert option(command, "--azimuth-deg") == "180"
    assert option(command, "--em-thinning") == "0.0001"
    assert option(command, "--maximum-weight") == "0"
    assert option(command, "--cuda-radio-backend") == "cuda"
    assert option(command, "--cuda-hadronic-backend") == "fluka-process"
    assert option(command, "--label").startswith(
        "pdg2212_100PeV_theta47_phi180_emthin0.0001_cuda_batch_seed"
    )
    assert command.count("--additional-proposal") == 25
    assert "--allow-mixed-proposal-builds" in command


def test_direct_batch_command_preserves_full_acceleration_configuration() -> None:
    args = arguments()
    args.defer_reference_comparison = True
    command = MODULE.direct_batch_command(
        args,
        Path("/tmp/output/batch_000"),
        25,
        args.cuda_seed_start,
    )

    assert option(command, "-N") == "25"
    assert option(command, "--zenith") == "47"
    assert option(command, "--azimuth") == "180"
    assert option(command, "--geomagnetic-model") == "IGRF13"
    assert option(command, "--geomagnetic-year") == "2025"
    assert option(command, "--emthin") == "0.0001"
    assert "--max-weight" not in command
    assert option(command, "--radio-backend") == "cuda"
    assert option(command, "--hadronic-backend") == "fluka-process"
    assert option(command, "--gpu-table-cache") == "/tmp/table"
    assert option(command, "--gpu-deterministic") == "true"
    assert option(command, "--gpu-resident-cross-species") == "true"


def test_direct_provenance_fingerprints_antenna_and_fluka_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = arguments()
        args.executable = root / "c8_air_shower"
        args.table = root / "table.c8emrt"
        args.physics_runner = root / "runner.py"
        args.antenna_file = root / "antennas.txt"
        args.flupro = root / "fluka"
        args.flupro.mkdir()
        for path in (
            args.executable,
            args.table,
            args.physics_runner,
            args.antenna_file,
            args.flupro / "libflukahp.a",
        ):
            path.write_bytes(path.name.encode("utf-8"))
        output = root / "campaign"
        (output / "cuda").mkdir(parents=True)

        MODULE.write_direct_cuda_provenance(args, output, ["c8", "-N", "1"])
        provenance = json.loads(
            (output / "cuda/validation_provenance.json").read_text(encoding="utf-8")
        )

        assert provenance["antenna_file"]["path"] == str(args.antenna_file)
        assert provenance["flupro"]["path"] == str(args.flupro / "libflukahp.a")
        assert len(provenance["antenna_file"]["sha256"]) == 64
        assert len(provenance["flupro"]["sha256"]) == 64


def test_parse_summary_runtime_accepts_legacy_numeric_schema() -> None:
    assert MODULE.parse_summary_runtime_seconds({"runtime_raw": 12.5}) == 12.5


def test_parse_summary_runtime_accepts_current_duration_schema() -> None:
    observed = MODULE.parse_summary_runtime_seconds(
        {"runtime": "00:46:37.554719515"}
    )
    assert observed == pytest.approx(2797.554719515)


def test_parse_summary_runtime_accepts_day_duration_schema() -> None:
    observed = MODULE.parse_summary_runtime_seconds(
        {"runtime": "2 days, 01:02:03.5"}
    )
    assert observed == pytest.approx(176523.5)


def test_parse_summary_runtime_rejects_missing_or_zero_runtime() -> None:
    with pytest.raises(ValueError, match="positive finite runtime"):
        MODULE.parse_summary_runtime_seconds({"runtime": "00:00:00"})
