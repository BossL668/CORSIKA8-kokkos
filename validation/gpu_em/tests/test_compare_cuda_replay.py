#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from compare_cuda_replay import (  # noqa: E402
    band_limited_fluence,
    ks_distance,
    radio_acceptance_decision,
    radiation_energy_proxy,
)


class CompareCudaReplayTests(unittest.TestCase):
    def test_ks_distance(self) -> None:
        self.assertEqual(ks_distance([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(ks_distance([0, 0], [1, 1]), 1.0)

    def test_radiation_energy_proxy_for_constant_footprint(self) -> None:
        locations = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, -10.0, 0.0],
            ]
        )
        fluence = np.ones((len(locations), 4))
        proxy = radiation_energy_proxy(fluence, locations)
        np.testing.assert_allclose(proxy, math.pi * 10.0**2)

    def test_radiation_energy_proxy_merges_micrometre_ring_roundoff(self) -> None:
        locations = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [300.0, 0.0, 0.0],
                [299.999999, 0.0, 0.0],
                [600.0, 0.0, 0.0],
            ]
        )
        fluence = np.ones((len(locations), 4))
        proxy = radiation_energy_proxy(fluence, locations)
        np.testing.assert_allclose(proxy, np.full(4, math.pi * 600.0**2))

    def test_band_filter_keeps_only_requested_frequency(self) -> None:
        sampling_hz = 1.0e9
        sample_count = 1000
        time = np.arange(sample_count) / sampling_hz
        field = np.column_stack(
            [
                np.sin(2.0 * math.pi * 50.0e6 * time),
                np.sin(2.0 * math.pi * 200.0e6 * time),
                np.zeros(sample_count),
            ]
        )
        fluence = band_limited_fluence(
            field, 1.0 / sampling_hz, 30.0, 80.0
        )
        self.assertGreater(fluence[0], 0.0)
        self.assertLess(fluence[1], fluence[0] * 1.0e-20)
        self.assertEqual(fluence[2], 0.0)

    def test_radio_gate_does_not_confuse_low_significance_with_accuracy(self) -> None:
        decision = radio_acceptance_decision(
            reference_mean=1.0,
            candidate_mean=0.83,
            reference_standard_error=0.10,
            candidate_standard_error=0.05,
            relative_tolerance=0.10,
            z_limit=3.0,
        )
        self.assertFalse(decision["relative_accuracy_pass"])
        self.assertTrue(decision["statistically_consistent"])
        self.assertFalse(decision["accepted"])

    def test_radio_gate_requires_statistical_consistency_too(self) -> None:
        decision = radio_acceptance_decision(
            reference_mean=1.0,
            candidate_mean=0.99,
            reference_standard_error=0.001,
            candidate_standard_error=0.001,
            relative_tolerance=0.10,
            z_limit=3.0,
        )
        self.assertTrue(decision["relative_accuracy_pass"])
        self.assertFalse(decision["statistically_consistent"])
        self.assertFalse(decision["accepted"])

    def test_radio_gate_accepts_only_when_both_conditions_hold(self) -> None:
        decision = radio_acceptance_decision(
            reference_mean=1.0,
            candidate_mean=0.95,
            reference_standard_error=0.03,
            candidate_standard_error=0.03,
            relative_tolerance=0.10,
            z_limit=3.0,
        )
        self.assertTrue(decision["relative_accuracy_pass"])
        self.assertTrue(decision["statistically_consistent"])
        self.assertTrue(decision["accepted"])


if __name__ == "__main__":
    unittest.main()
