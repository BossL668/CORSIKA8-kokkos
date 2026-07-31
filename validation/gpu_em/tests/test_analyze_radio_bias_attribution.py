#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_radio_bias_attribution import (  # noqa: E402
    cross_channel_correlation,
    resampled_shift_probability,
    signed_shift,
    trimmed,
    upper_tail_share,
)


class RadioBiasAttributionTest(unittest.TestCase):
    def test_signed_shift_and_symmetric_trim(self) -> None:
        reference = np.asarray([1.0, 2.0, 3.0, 100.0])
        candidate = np.asarray([2.0, 3.0, 4.0, 5.0])
        self.assertLess(signed_shift(reference, candidate), 0.0)
        np.testing.assert_array_equal(trimmed(reference, 1), [2.0, 3.0])

    def test_upper_tail_share(self) -> None:
        values = np.asarray([1.0, 1.0, 2.0, 6.0])
        self.assertAlmostEqual(upper_tail_share(values, 1), 0.6)

    def test_resampling_reports_extreme_probability(self) -> None:
        result = resampled_shift_probability(
            np.ones(20),
            np.full(20, 2.0),
            sample_size=10,
            threshold=0.5,
            repetitions=1000,
            seed=1,
        )
        self.assertEqual(
            result["probability_shift_at_or_below_threshold"], 0.0
        )
        self.assertEqual(result["shift_quantiles"]["median"], 1.0)

    def test_cross_channel_correlation_recognizes_shared_showers(self) -> None:
        proxies = {
            "a": {0: 1.0, 1: 2.0, 2: 4.0},
            "b": {0: 2.0, 1: 4.0, 2: 8.0},
            "c": {0: 3.0, 1: 6.0, 2: 12.0},
        }
        result = cross_channel_correlation(proxies)
        self.assertEqual(result["showers"], 3)
        self.assertAlmostEqual(result["minimum_off_diagonal"], 1.0)


if __name__ == "__main__":
    unittest.main()
