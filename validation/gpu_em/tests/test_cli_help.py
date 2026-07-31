#!/usr/bin/env python3

import subprocess
import sys
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]


class ValidationCliHelpTest(unittest.TestCase):
    def test_percent_tolerance_is_rendered_without_argparse_metadata(self):
        for script_name in (
            "compare_ensembles.py",
            "run_physics_acceptance.py",
        ):
            with self.subTest(script=script_name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(MODULE_DIRECTORY / script_name),
                        "--help",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("1%", completed.stdout)
                self.assertIn("statistical", completed.stdout)
                self.assertNotIn("'option_strings'", completed.stdout)

    def test_replay_help_exposes_radio_sampling_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIRECTORY / "run_cuda_replay.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--radio-sampling-rate-ghz", completed.stdout)
        self.assertIn("--radio-window-duration-ns", completed.stdout)
        self.assertIn("--radio-pretrigger-ns", completed.stdout)

    def test_pooled_radio_help_accepts_repeated_outputs(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIRECTORY / "pool_radio_ensembles.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--reference-output", completed.stdout)
        self.assertIn("--candidate-output", completed.stdout)


if __name__ == "__main__":
    unittest.main()
