#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd

from validation.gpu_em.diagnose_ground_radio_distribution_shift import (
    analyze_metric,
    bootstrap_quantile_ratio,
    empirical_ks_location,
    low_tail_fraction_comparison,
)


class GroundRadioDistributionShiftTest(unittest.TestCase):
    def test_empirical_ks_location_reports_signed_cdf_separation(self):
        result = empirical_ks_location(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([1.0, 2.0, 4.0, 5.0]),
        )
        self.assertEqual(result["value"], 3.0)
        self.assertEqual(result["statistic"], 0.25)
        self.assertEqual(result["signed_cuda_minus_proposal_cdf"], -0.25)
        self.assertEqual(result["proposal_cdf"], 0.75)
        self.assertEqual(result["cuda_cdf"], 0.5)
        self.assertFalse(result["inside_proposal_low_16pct"])

    def test_low_tail_bootstrap_reestimates_cpu_threshold(self):
        result = low_tail_fraction_comparison(
            np.arange(1.0, 101.0),
            np.arange(101.0, 201.0),
            quantile=0.05,
            repetitions=200,
            seed=7,
        )
        self.assertEqual(result["proposal_event_fraction"], 0.05)
        self.assertEqual(result["cuda_event_fraction"], 0.0)
        self.assertEqual(result["cuda_minus_proposal_fraction"], -0.05)
        self.assertLess(result["bootstrap_shower_95pct"][1], 0.0)

    def test_low_tail_quantile_is_validated(self):
        with self.assertRaisesRegex(ValueError, "quantile"):
            low_tail_fraction_comparison(
                np.arange(10.0),
                np.arange(10.0),
                quantile=0.5,
                repetitions=10,
                seed=3,
            )

    def test_low_tail_quantile_ratio_tracks_scale_displacement(self):
        reference = np.arange(1.0, 101.0)
        result = bootstrap_quantile_ratio(
            reference,
            2.0 * reference,
            quantile=0.16,
            repetitions=400,
            seed=11,
        )
        self.assertEqual(result["cuda_over_proposal_ratio"], 2.0)
        self.assertLess(result["bootstrap_shower_95pct"][0], 2.0)
        self.assertGreater(result["bootstrap_shower_95pct"][1], 2.0)

    def test_metric_report_contains_tail_sensitive_two_sample_tests(self):
        reference = np.linspace(10.0, 20.0, 30)
        candidate = np.linspace(10.5, 20.5, 30)
        frame = pd.DataFrame(
            {
                "backend_canonical": ["proposal"] * 30 + ["cuda"] * 30,
                "backend_indicator": [0.0] * 30 + [1.0] * 30,
                "metric": np.concatenate((reference, candidate)),
                "profile_xmax_charged_gcm2": np.linspace(500.0, 700.0, 60),
                "log_profile_em_integral": np.linspace(1.0, 2.0, 60),
            }
        )
        result = analyze_metric(frame, "metric", repetitions=40, seed=5)
        self.assertIn("anderson_darling_k_sample", result["tests"])
        self.assertIn("cramer_von_mises", result["tests"])
        self.assertIn("q05", result["low_tail_quantile_ratios"])


if __name__ == "__main__":
    unittest.main()
