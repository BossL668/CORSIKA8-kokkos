import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import analyze_post_xmax_em_profiles as analysis  # noqa: E402


class PostXmaxProfileAnalysisTest(unittest.TestCase):
    def test_quadratic_peak_refines_the_grid_maximum(self) -> None:
        coordinate = np.asarray([0.0, 10.0, 20.0])
        values = -((coordinate - 12.0) ** 2) + 200.0
        self.assertAlmostEqual(
            analysis.quadratic_peak_x(coordinate, values), 12.0
        )

    def test_alignment_masks_profile_below_observation_surface(self) -> None:
        coordinate = np.asarray([0.0, 10.0, 20.0, 30.0])
        matrix = np.asarray([[1.0, 2.0, 3.0, 0.0]])
        aligned = analysis.align_matrix(
            coordinate,
            matrix,
            np.asarray([10.0]),
            np.asarray([0.0, 10.0, 20.0]),
            np.asarray([20.0]),
        )
        np.testing.assert_allclose(aligned[0, :2], [2.0, 3.0])
        self.assertTrue(np.isnan(aligned[0, 2]))

    def test_per_shower_normalization_is_not_ensemble_normalization(self) -> None:
        matrix = np.asarray([[2.0, 1.0], [10.0, 2.0]])
        normalized = analysis.normalize_at_xmax(matrix)
        np.testing.assert_allclose(normalized, [[1.0, 0.5], [1.0, 0.2]])


if __name__ == "__main__":
    unittest.main()
