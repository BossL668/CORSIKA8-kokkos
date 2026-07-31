#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from pool_radio_ensembles import (  # noqa: E402
    geometric_mean_comparison,
    holm_bonferroni_rejections,
    leave_one_out_shift,
    paired_ratio_summary,
    trimmed_shift,
)


class PoolRadioEnsemblesTest(unittest.TestCase):
    def test_trimmed_shift_removes_symmetric_tails(self) -> None:
        reference = np.asarray([0.0, 1.0, 2.0, 3.0, 100.0])
        candidate = np.asarray([0.0, 2.0, 4.0, 6.0, 200.0])
        self.assertAlmostEqual(
            trimmed_shift(reference, candidate, 0.2),
            1.0,
        )

    def test_leave_one_out_range_contains_full_sample_shift(self) -> None:
        reference = np.asarray([1.0, 2.0, 3.0])
        candidate = np.asarray([2.0, 3.0, 4.0])
        result = leave_one_out_shift(reference, candidate)
        full_shift = (
            np.mean(candidate) - np.mean(reference)
        ) / np.mean(reference)
        self.assertLessEqual(result["minimum"], full_shift)
        self.assertGreaterEqual(result["maximum"], full_shift)

    def test_leave_one_out_requires_two_values_per_arm(self) -> None:
        self.assertEqual(
            leave_one_out_shift(
                np.asarray([1.0]),
                np.asarray([2.0, 3.0]),
            ),
            {},
        )

    def test_paired_ratio_summary_preserves_pairing(self) -> None:
        result = paired_ratio_summary(
            np.asarray([1.0, 2.0, 4.0]),
            np.asarray([2.0, 4.0, 8.0]),
            repetitions=1000,
            seed=1,
        )
        self.assertTrue(result["available"])
        self.assertEqual(
            result["aggregate_candidate_over_reference"], 2.0
        )
        self.assertAlmostEqual(
            result["geometric_mean_candidate_over_reference"], 2.0
        )
        self.assertAlmostEqual(result["pearson"], 1.0)

    def test_geometric_mean_comparison_handles_positive_heavy_tail(
        self,
    ) -> None:
        reference = np.asarray([1.0, 2.0, 4.0, 1000.0])
        candidate = reference * 0.95
        result = geometric_mean_comparison(
            reference,
            candidate,
            repetitions=1000,
            seed=3,
        )
        self.assertTrue(result["available"])
        self.assertAlmostEqual(
            result["candidate_over_reference"], 0.95
        )
        self.assertLessEqual(
            result["bootstrap_candidate_over_reference_95pct"][0],
            0.95,
        )
        self.assertGreaterEqual(
            result["bootstrap_candidate_over_reference_95pct"][1],
            0.95,
        )

    def test_holm_bonferroni_controls_four_shape_tests(self) -> None:
        self.assertEqual(
            holm_bonferroni_rejections(
                [0.029, 0.1, 0.2, 0.3], 0.05
            ),
            [False, False, False, False],
        )
        self.assertEqual(
            holm_bonferroni_rejections(
                [0.005, 0.1, 0.2, 0.3], 0.05
            ),
            [True, False, False, False],
        )


if __name__ == "__main__":
    unittest.main()
