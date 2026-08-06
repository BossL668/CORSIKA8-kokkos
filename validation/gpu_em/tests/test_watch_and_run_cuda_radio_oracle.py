from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "watch_and_run_cuda_radio_oracle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "watch_and_run_cuda_radio_oracle", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_campaign_status_handles_missing_and_reads_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert MODULE.campaign_status(root) == "missing"
        (root / "campaign_manifest.json").write_text(
            json.dumps({"status": "running"}), encoding="utf-8"
        )
        assert MODULE.campaign_status(root) == "running"


def test_run_oracle_records_successful_ctest_result() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign = root / "campaign"
        build = root / "build"
        campaign.mkdir()
        build.mkdir()
        fake_ctest = root / "fake_ctest"
        fake_ctest.write_text(
            "#!/bin/sh\nprintf 'same-track oracle passed\\n'\nexit 0\n",
            encoding="utf-8",
        )
        fake_ctest.chmod(fake_ctest.stat().st_mode | 0o111)

        result = MODULE.run_oracle(
            campaign, build, str(fake_ctest), "testGpuRadioProjection"
        )

        assert result == 0
        status = json.loads(
            (campaign / "radio_projection_oracle_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "complete"
        assert status["returncode"] == 0
        assert "same-track oracle passed" in (
            campaign / "radio_projection_oracle_ctest.log"
        ).read_text(encoding="utf-8")


def test_run_oracle_records_failure() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign = root / "campaign"
        build = root / "build"
        campaign.mkdir()
        build.mkdir()
        fake_ctest = root / "fake_ctest"
        fake_ctest.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
        fake_ctest.chmod(fake_ctest.stat().st_mode | 0o111)

        result = MODULE.run_oracle(campaign, build, str(fake_ctest), "oracle")

        assert result == 7
        status = json.loads(
            (campaign / "radio_projection_oracle_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "failed"
        assert status["returncode"] == 7


def test_run_full_acceptance_records_success() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign = root / "campaign"
        campaign.mkdir()
        script = root / "acceptance.py"
        script.write_text(
            "import sys\nprint('full identical-track acceptance')\nsys.exit(0)\n",
            encoding="utf-8",
        )
        executable = root / "c8_air_shower"
        table = root / "table.c8emrt"
        executable.write_bytes(b"binary")
        table.write_bytes(b"table")
        args = argparse.Namespace(
            acceptance_script=script,
            executable=executable,
            table=table,
            acceptance_output=root / "acceptance_output",
            acceptance_energy_gev=1000.0,
            acceptance_events=2,
            acceptance_seed=123,
            acceptance_zenith_deg=0.0,
            acceptance_azimuth_deg=0.0,
            acceptance_geomagnetic_model="IGRF14",
            acceptance_geomagnetic_year=2027.0,
            acceptance_ring=1,
            acceptance_antenna_file=None,
            gpu_device=0,
            gpu_memory_fraction=0.7,
            campaign_root=campaign,
        )

        result = MODULE.run_full_acceptance(args)

        assert result == 0
        status = json.loads(
            (campaign / "radio_identical_track_acceptance_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "complete"
        assert status["returncode"] == 0
        assert "full identical-track acceptance" in (
            campaign / "radio_identical_track_acceptance.runner.log"
        ).read_text(encoding="utf-8")


def test_run_pulse_feature_acceptance_records_success() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        campaign = root / "campaign"
        acceptance = root / "acceptance"
        cpu = acceptance / "cpu_radio"
        cuda = acceptance / "cuda_radio"
        pulse_root = root / "pulse_analysis_modular"
        campaign.mkdir()
        cpu.mkdir(parents=True)
        cuda.mkdir()
        pulse_root.mkdir()
        script = root / "pulse_acceptance.py"
        script.write_text(
            "import sys\nprint('pulse feature acceptance')\nsys.exit(0)\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            pulse_feature_script=script,
            acceptance_output=acceptance,
            pulse_analysis_root=pulse_root,
            pulse_feature_output=root / "pulse_features",
            acceptance_zenith_deg=0.0,
            acceptance_azimuth_deg=0.0,
            pulse_amplitude_relative_tolerance=2.0e-4,
            pulse_width_absolute_tolerance_ns=0.11,
            campaign_root=campaign,
        )

        result = MODULE.run_pulse_feature_acceptance(args)

        assert result == 0
        status = json.loads(
            (
                campaign
                / "radio_identical_track_pulse_features_status.json"
            ).read_text(encoding="utf-8")
        )
        assert status["status"] == "complete"
        assert "pulse feature acceptance" in (
            campaign / "radio_identical_track_pulse_features.runner.log"
        ).read_text(encoding="utf-8")
