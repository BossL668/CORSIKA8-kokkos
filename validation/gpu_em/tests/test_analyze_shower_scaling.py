from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from analyze_shower_scaling import (  # noqa: E402
    compare_bootstraps,
    power_fit,
    xmax_fit,
)


class ShowerScalingTests(unittest.TestCase):
    def test_power_fit_recovers_exact_exponent(self) -> None:
        energies = np.asarray([100.0, 300.0, 1000.0, 3000.0])
        arrays = [
            np.full(12, 7.0 * (energy / 1000.0) ** 1.25)
            for energy in energies
        ]
        result, slopes, _ = power_fit(
            energies, arrays, 200, 17, expected_slope=1.25
        )
        self.assertAlmostEqual(result["slope"], 1.25, places=12)
        self.assertTrue(np.allclose(slopes, 1.25))
        self.assertTrue(result["expected_inside_ci95"])

    def test_xmax_fit_recovers_elongation_rate(self) -> None:
        energies = np.asarray([100.0, 300.0, 1000.0, 3000.0])
        arrays = [
            np.full(
                10,
                320.0 + 82.0 * np.log10(energy / 1000.0),
            )
            for energy in energies
        ]
        result, slopes, _ = xmax_fit(energies, arrays, 200, 23)
        self.assertAlmostEqual(
            result["elongation_rate_gcm2_per_decade"], 82.0, places=12
        )
        self.assertTrue(np.allclose(slopes, 82.0))

    def test_bootstrap_comparison_reports_ratio_and_difference(self) -> None:
        cpu_slopes = np.full(200, 1.0)
        cuda_slopes = np.full(200, 1.1)
        cpu_means = np.full((200, 4), 10.0)
        cuda_means = np.full((200, 4), 9.0)
        result = compare_bootstraps(
            cpu_slopes, cuda_slopes, cpu_means, cuda_means
        )
        self.assertAlmostEqual(result["cuda_minus_cpu"], 0.1)
        self.assertFalse(result["zero_inside_slope_difference_ci95"])
        self.assertAlmostEqual(
            result["cuda_over_cpu_group_mean_ratios"][0]["ratio"], 0.9
        )


if __name__ == "__main__":
    unittest.main()
