from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


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
