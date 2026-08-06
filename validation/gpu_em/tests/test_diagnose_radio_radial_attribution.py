#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd

from validation.gpu_em.diagnose_radio_radial_attribution import (
    AMPLITUDE,
    WIDTH,
    add_reference_normalization,
    analyze_radius,
    canonicalize_radial_coordinate,
    ratio_with_bootstrap,
)


class RadioRadialAttributionTest(unittest.TestCase):
    def test_legacy_radius_alias_is_canonicalized_and_checked(self):
        legacy = canonicalize_radial_coordinate(
            pd.DataFrame({"radius_m": [1.0, 100.0]})
        )
        np.testing.assert_allclose(legacy.r_perp_m, [1.0, 100.0])

        current = canonicalize_radial_coordinate(
            pd.DataFrame(
                {
                    "radius_m": [1.0, 100.0],
                    "r_perp_m": [1.0, 100.0],
                }
            )
        )
        np.testing.assert_allclose(current.r_perp_m, [1.0, 100.0])

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            canonicalize_radial_coordinate(
                pd.DataFrame(
                    {
                        "radius_m": [1.0, 100.0],
                        "r_perp_m": [1.0, 101.0],
                    }
                )
            )

    def test_adjusted_log_model_recovers_known_backend_ratio(self):
        rows = []
        for backend, offset in (("proposal", 0.0), ("cuda", np.log(2.0))):
            for shower in range(12):
                x = float(shower)
                rows.append(
                    {
                        "backend_canonical": backend,
                        "log_response": 0.03 * x + offset,
                        "x": x,
                    }
                )
        result = ratio_with_bootstrap(
            pd.DataFrame(rows),
            "log_response",
            ("x",),
            repetitions=30,
            seed=17,
        )
        self.assertAlmostEqual(result["ratio"], 2.0, places=12)
        self.assertAlmostEqual(result["bootstrap_95pct"][0], 2.0, places=12)
        self.assertAlmostEqual(result["bootstrap_95pct"][1], 2.0, places=12)

    def test_reference_normalization_is_per_shower_and_backend(self):
        rows = []
        for backend, scale in (("proposal", 1.0), ("cuda", 3.0)):
            for shower in range(3):
                reference = scale * (shower + 1.0)
                for radius, shape in ((50.0, 2.0), (100.0, 1.0)):
                    rows.append(
                        {
                            "backend_canonical": backend,
                            "algorithm": "CoREAS",
                            "shower": shower,
                            "r_perp_m": radius,
                            AMPLITUDE: reference * shape,
                        }
                    )
        normalized = add_reference_normalization(pd.DataFrame(rows), 100.0)
        at_50 = normalized[normalized.r_perp_m == 50.0]
        at_100 = normalized[normalized.r_perp_m == 100.0]
        np.testing.assert_allclose(at_50.log_amplitude_shape, np.log(2.0))
        np.testing.assert_allclose(at_100.log_amplitude_shape, 0.0)

    def test_width_gate_uses_both_backend_shower_fractions(self):
        rows = []
        for backend in ("proposal", "cuda"):
            for shower in range(10):
                valid = not (backend == "cuda" and shower >= 8)
                rows.append(
                    {
                        "backend_canonical": backend,
                        "algorithm": "ZHS",
                        "shower": shower,
                        "r_perp_m": 100.0,
                        AMPLITUDE: 1.0 + 0.01 * shower,
                        "reference_amplitude": 1.0 + 0.01 * shower,
                        "log_amplitude_shape": 0.0,
                        WIDTH: 5.0 if valid else np.nan,
                        "pulse_width_filter_pass_count": 1 if valid else 0,
                        "profile_xmax_charged_gcm2": 400.0 + shower,
                        "log_profile_em_integral": 10.0 + 0.02 * shower**2,
                    }
                )
        frame = pd.DataFrame(rows)
        eligible = analyze_radius(
            frame,
            repetitions=10,
            seed=23,
            minimum_width_valid_fraction=0.8,
        )
        self.assertTrue(eligible["width_raw"]["acceptance_eligible"])
        self.assertEqual(eligible["width_raw"]["valid_fraction"]["cuda"], 0.8)
        rejected = analyze_radius(
            frame,
            repetitions=10,
            seed=23,
            minimum_width_valid_fraction=0.9,
        )
        self.assertFalse(rejected["width_raw"]["acceptance_eligible"])


if __name__ == "__main__":
    unittest.main()
