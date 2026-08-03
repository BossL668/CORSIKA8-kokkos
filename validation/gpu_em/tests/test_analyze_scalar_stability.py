import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from analyze_scalar_stability import analyse_frame


class ScalarStabilityTest(unittest.TestCase):
    def analyse(self, cpu, cuda, tolerance=0.01):
        frame = pd.DataFrame(
            {
                "shower": np.arange(len(cpu) + len(cuda)),
                "observable": [*cpu, *cuda],
                "backend": ["proposal"] * len(cpu) + ["cuda"] * len(cuda),
            }
        )
        return analyse_frame(
            frame,
            ["observable"],
            relative_tolerance=tolerance,
            sigma_limit=3.0,
            confidence=0.95,
            bootstrap_repetitions=2000,
            seed=17,
        )["metrics"]["observable"]

    def test_identical_constant_samples_are_equivalent(self):
        result = self.analyse([10.0] * 20, [10.0] * 20)
        self.assertEqual(result["classification"], "equivalent")
        self.assertEqual(result["signed_relative_difference"], 0.0)
        self.assertEqual(
            result["required_events_per_backend_for_sigma_precision"], 2
        )

    def test_resolved_bias_is_different(self):
        result = self.analyse([10.0] * 20, [10.3] * 20)
        self.assertEqual(result["classification"], "different")
        self.assertAlmostEqual(result["signed_relative_difference"], 0.03)

    def test_noisy_small_sample_is_inconclusive(self):
        cpu = [8.0, 12.0, 9.0, 11.0, 10.0, 13.0, 7.0, 10.0]
        cuda = [8.2, 12.2, 9.2, 11.2, 10.2, 13.2, 7.2, 10.2]
        result = self.analyse(cpu, cuda)
        self.assertEqual(result["classification"], "inconclusive")
        self.assertGreater(
            result["required_events_per_backend_for_sigma_precision"],
            len(cpu),
        )

    def test_sparse_zero_reference_resamples_are_reported(self):
        cpu = [0.0] * 18 + [10.0, 20.0]
        cuda = [0.0] * 18 + [9.0, 21.0]
        result = self.analyse(cpu, cuda)
        bootstrap = result["bootstrap"]
        self.assertGreater(
            bootstrap["undefined_zero_reference_repetitions"], 0
        )
        self.assertGreaterEqual(
            bootstrap["finite_relative_repetitions"], 100
        )
        self.assertIn(
            result["classification"],
            {"equivalent", "different", "inconclusive"},
        )

    def test_all_zero_samples_are_equivalent(self):
        result = self.analyse([0.0] * 20, [0.0] * 20)
        self.assertEqual(result["classification"], "equivalent")
        self.assertEqual(
            result["bootstrap"]["undefined_zero_reference_repetitions"],
            0,
        )
        self.assertEqual(result["signed_relative_difference"], 0.0)

    def test_missing_backend_is_rejected(self):
        frame = pd.DataFrame(
            {"observable": [1.0, 2.0], "backend": ["proposal", "proposal"]}
        )
        with self.assertRaisesRegex(ValueError, "must contain"):
            analyse_frame(
                frame,
                ["observable"],
                relative_tolerance=0.01,
                sigma_limit=3.0,
                confidence=0.95,
                bootstrap_repetitions=100,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
