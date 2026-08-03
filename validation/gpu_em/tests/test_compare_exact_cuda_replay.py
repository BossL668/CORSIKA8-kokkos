from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "compare_exact_cuda_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_exact_cuda_replay", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExactReplayMetricsTests(unittest.TestCase):
    def test_identical_waveform_is_exact(self) -> None:
        waveform = np.asarray([0.0, 1.0, -2.0, 0.5])
        metrics = MODULE.waveform_metrics(waveform, waveform.copy())
        self.assertEqual(metrics["maximum_absolute_V_per_m"], 0.0)
        self.assertEqual(metrics["peak_normalized_maximum"], 0.0)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertEqual(metrics["relative_fluence"], 0.0)

    def test_metrics_detect_a_changed_sample(self) -> None:
        reference = np.asarray([0.0, 1.0, 0.0])
        candidate = np.asarray([0.0, 1.1, 0.0])
        metrics = MODULE.waveform_metrics(reference, candidate)
        self.assertGreater(metrics["maximum_absolute_V_per_m"], 0.09)
        self.assertGreater(metrics["relative_l2"], 0.09)
        self.assertGreater(metrics["relative_fluence"], 0.17)


if __name__ == "__main__":
    unittest.main()
