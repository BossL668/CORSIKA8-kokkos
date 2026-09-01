#!/usr/bin/env python3

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from validation.gpu_em.compare_radio_pulse_features import (
    command_physics,
    holm_rejections,
    strict_json,
)


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]


class CompareRadioPulseFeaturesTest(unittest.TestCase):
    def test_help_is_analysis_only_and_exposes_both_input_modes(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIRECTORY / "compare_radio_pulse_features.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--campaign-root", completed.stdout)
        self.assertIn("--reference-output", completed.stdout)
        self.assertIn("--candidate-output", completed.stdout)
        self.assertIn("analysis-only", completed.stdout)

    def test_command_physics_reads_relevant_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "args": (
                            "c8_air_shower -p 2212 -E 1000 --zenith 47 "
                            "--azimuth 180 --emthin 1e-6 --em-backend cuda"
                        )
                    }
                ),
                encoding="utf-8",
            )
            result = command_physics(root)
        self.assertEqual(result["primary_pdg"], 2212)
        self.assertEqual(result["energy_GeV"], 1000.0)
        self.assertEqual(result["zenith_deg"], 47.0)
        self.assertEqual(result["azimuth_deg"], 180.0)
        self.assertEqual(result["em_thinning"], 1.0e-6)
        self.assertNotIn("em_backend", result)

    def test_holm_stops_after_first_non_rejection(self):
        self.assertEqual(
            holm_rejections([0.001, 0.03, 0.20], 0.05),
            [True, False, False],
        )

    def test_strict_json_maps_nonfinite_values_to_null(self):
        self.assertEqual(
            strict_json({"nan": math.nan, "inf": math.inf, "ok": 2.0}),
            {"nan": None, "inf": None, "ok": 2.0},
        )


if __name__ == "__main__":
    unittest.main()
