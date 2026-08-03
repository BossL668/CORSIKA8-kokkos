#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_paired_seed_control import analyse_frame, analyse_metric  # noqa: E402


class PairedSeedControlTest(unittest.TestCase):
    def test_identity_is_reported_exactly(self) -> None:
        result = analyse_metric(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(result["exact_pairs"], 2)
        self.assertAlmostEqual(result["pearson_correlation"], 1.0)
        self.assertEqual(result["paired_rms_difference"], 0.0)

    def test_frame_aligns_backends_by_shower_id(self) -> None:
        frame = pd.DataFrame(
            {
                "shower": [1, 0, 0, 1],
                "backend": ["proposal", "proposal", "cuda", "cuda"],
                "value": [2.0, 1.0, 1.0, 4.0],
            }
        )
        result = analyse_frame(frame)["metrics"]["value"]
        self.assertEqual(result["exact_pairs"], 1)
        self.assertEqual(result["candidate_mean"], 2.5)


if __name__ == "__main__":
    unittest.main()
