#!/usr/bin/env python3
"""Unit tests for shower-age alignment of muon-parent profiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analyze_muon_parent_alignment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_muon_parent_alignment", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PeakCoordinateTests(unittest.TestCase):
    def test_quadratic_peak_recovers_sub_bin_vertex(self) -> None:
        coordinate = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
        values = -(coordinate - 23.0) ** 2 + 1000.0
        self.assertAlmostEqual(
            MODULE.quadratic_peak_coordinate(coordinate, values),
            23.0,
            places=12,
        )

    def test_alignment_removes_anchor_translation(self) -> None:
        coordinate = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
        aligned_coordinate = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0])
        cpu = np.asarray([[0.0, 1.0, 3.0, 1.0, 0.0]])
        cuda = np.asarray([[0.0, 0.0, 1.0, 3.0, 1.0]])
        cpu_mean, _, cpu_count = MODULE._aligned_moments(
            coordinate, cpu, np.asarray([20.0]), aligned_coordinate
        )
        cuda_mean, _, cuda_count = MODULE._aligned_moments(
            coordinate, cuda, np.asarray([30.0]), aligned_coordinate
        )
        common = (cpu_count == 1) & (cuda_count == 1)
        np.testing.assert_allclose(cpu_mean[common], cuda_mean[common])


if __name__ == "__main__":
    unittest.main()
