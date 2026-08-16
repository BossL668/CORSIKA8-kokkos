#!/usr/bin/env python3
"""Unit tests for shower-feature plot metadata helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analyze_shower_feature_distributions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_shower_feature_distributions", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrimaryLabelTests(unittest.TestCase):
    def test_elementary_primary_uses_pdg_label(self) -> None:
        self.assertEqual(MODULE.primary_label({"primary_pdg": 2212}), "proton")

    def test_iron_primary_uses_nuclear_metadata(self) -> None:
        self.assertEqual(
            MODULE.primary_label(
                {"primary_pdg": None, "primary_Z": 26, "primary_A": 56}
            ),
            r"$^{56}$Fe",
        )

    def test_incomplete_nuclear_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both primary_Z and primary_A"):
            MODULE.primary_label(
                {"primary_pdg": None, "primary_Z": 26, "primary_A": None}
            )


class HistogramNormalizationTests(unittest.TestCase):
    def test_log_bins_ignore_zero_but_cover_positive_support(self) -> None:
        bins = MODULE.common_bins(
            {
                "proposal": np.asarray([1.0, 2.0, 4.0]),
                "cuda": np.asarray([0.0, 3.0, 8.0]),
            },
            True,
        )
        self.assertEqual(float(bins[0]), 1.0)
        self.assertEqual(float(bins[-1]), 8.0)

    def test_log_bins_reject_all_nonpositive_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive values"):
            MODULE.common_bins(
                {
                    "proposal": np.asarray([0.0]),
                    "cuda": np.asarray([-1.0, 0.0]),
                },
                True,
            )

    def test_linear_histogram_uses_density(self) -> None:
        density, weights = MODULE.histogram_normalization(
            np.asarray([1.0, 2.0]), False
        )
        self.assertTrue(density)
        self.assertIsNone(weights)

    def test_log_histogram_uses_equal_event_probability(self) -> None:
        density, weights = MODULE.histogram_normalization(
            np.asarray([1.0, 2.0, 4.0, 8.0]), True
        )
        self.assertFalse(density)
        assert weights is not None
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        np.testing.assert_array_equal(weights, np.full(4, 0.25))

    def test_empty_log_histogram_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty logarithmic"):
            MODULE.histogram_normalization(np.asarray([]), True)


if __name__ == "__main__":
    unittest.main()
