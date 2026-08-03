#!/usr/bin/env python3

import unittest
from pathlib import Path

import numpy as np

from validation.gpu_em.analyze_geomagnetic_radial_comparison import (
    clustered_available_radii,
    load_records,
    observer_axis_coordinates,
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

    def test_observer_coordinates_match_pulse_analysis_r_perp(self):
        theta = np.deg2rad(47.0)
        phi = np.deg2rad(180.0)
        axis = np.asarray(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                -np.cos(theta),
            ]
        )
        diagonal = 100.0 / np.sqrt(2.0)
        locations = np.asarray(
            [
                [0.0, 0.0, 6_373_680.0],
                [100.0, 0.0, 6_373_680.0],
                [0.0, 100.0, 6_373_680.0],
                [diagonal, diagonal, 6_373_680.0],
            ]
        )
        ground, r_perp = observer_axis_coordinates(locations, axis)
        np.testing.assert_allclose(ground, [0.0, 100.0, 100.0, 100.0])
        self.assertAlmostEqual(r_perp[0], 0.0)
        self.assertAlmostEqual(r_perp[1], 100.0 * np.cos(theta))
        self.assertAlmostEqual(r_perp[2], 100.0)
        self.assertAlmostEqual(
            r_perp[3],
            100.0 * np.sqrt(1.0 - 0.5 * np.sin(theta) ** 2),
        )

    def test_load_records_replaces_ground_radius_with_r_perp(self):
        theta = np.deg2rad(47.0)
        axis = np.asarray([-np.sin(theta), 0.0, -np.cos(theta)])
        locations = np.asarray(
            [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0]]
        )

        def fake_reader(_root, _algorithm):
            return (
                [
                    {"shower": 0, "observer": 0, "radius_m": 0.0},
                    {"shower": 0, "observer": 1, "radius_m": 100.0},
                ],
                locations,
            )

        records = load_records(
            output_directories=[Path("first"), Path("second")],
            algorithm="CoREAS",
            shower_axis_nwu=axis,
            core_xy_m=(0.0, 0.0),
            read_radio_records=fake_reader,
        )
        self.assertEqual([row["shower"] for row in records], [0, 0, 1, 1])
        for row in (records[1], records[3]):
            self.assertAlmostEqual(row["ground_radius_m"], 100.0)
            self.assertAlmostEqual(row["r_perp_m"], 100.0 * np.cos(theta))
            self.assertAlmostEqual(row["radius_m"], row["r_perp_m"])


if __name__ == "__main__":
    unittest.main()
