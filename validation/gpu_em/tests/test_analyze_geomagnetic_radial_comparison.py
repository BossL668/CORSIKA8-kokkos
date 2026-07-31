#!/usr/bin/env python3

import unittest

from validation.gpu_em.analyze_geomagnetic_radial_comparison import (
    clustered_available_radii,
)


class GeomagneticRadialComparisonTest(unittest.TestCase):
    def test_text_rounding_variants_collapse_to_nominal_radius(self):
        records = [
            {"radius_m": value}
            for value in (
                0.0,
                1.0,
                1.00000031,
                5.0,
                5.00000066,
                49.99999905,
                50.0,
                100.0,
                600.0,
                700.0,
            )
        ]
        self.assertEqual(
            clustered_available_radii(
                records, minimum_m=1.0, maximum_m=600.0
            ),
            [1.0, 5.0, 50.0, 100.0, 600.0],
        )


if __name__ == "__main__":
    unittest.main()
