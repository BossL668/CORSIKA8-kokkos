#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from analyze_geomagnetic_scaling import (  # noqa: E402
    group_power_fit,
    radial_power_fit,
    slope_difference,
)


class GeomagneticScalingAnalysisTest(unittest.TestCase):
    def test_stratified_energy_fit_recovers_known_exponent(self) -> None:
        rows = []
        for energy in (100.0, 300.0, 1000.0, 3000.0):
            for shower in range(20):
                fluctuation = np.exp(0.02 * (shower - 9.5) / 9.5)
                rows.append(
                    {
                        "energy_GeV": energy,
                        "value": 2.0e-8
                        * (energy / 1000.0) ** 1.07
                        * fluctuation,
                    }
                )
        result, samples = group_power_fit(
            rows,
            x_column="energy_GeV",
            y_column="value",
            pivot=1000.0,
            repetitions=500,
            seed=7,
            expected_slope=1.0,
        )
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["slope"], 1.07, places=10)
        self.assertEqual(samples.size, 500)

    def test_radial_bootstrap_resamples_complete_showers(self) -> None:
        rows = []
        for shower in range(15):
            shower_scale = np.exp(0.03 * shower)
            for radius in (50.0, 100.0, 200.0, 400.0):
                rows.append(
                    {
                        "shower": shower,
                        "effective_shower_plane_radius_m": radius,
                        "value": shower_scale
                        * (radius / 100.0) ** -1.4,
                    }
                )
        result, samples = radial_power_fit(
            rows,
            y_column="value",
            radius_low_m=50.0,
            radius_high_m=400.0,
            pivot_m=100.0,
            repetitions=500,
            seed=9,
        )
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["slope"], -1.4, places=10)
        self.assertEqual(samples.size, 500)
        self.assertEqual(result["shower_count"], 15)

    def test_slope_difference_detects_matching_exponents(self) -> None:
        reference = np.asarray([0.98, 1.00, 1.02])
        comparison = slope_difference(reference, reference)
        self.assertTrue(comparison["available"])
        self.assertAlmostEqual(comparison["cuda_minus_cpu"], 0.0)
        self.assertTrue(comparison["zero_inside_ci95"])


if __name__ == "__main__":
    unittest.main()
