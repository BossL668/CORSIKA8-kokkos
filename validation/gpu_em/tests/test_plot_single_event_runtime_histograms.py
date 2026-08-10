#!/usr/bin/env python3
"""Unit tests for pooled timing-source metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "plot_single_event_runtime_histograms.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_single_event_runtime_histograms", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeSourceStatisticsTests(unittest.TestCase):
    def test_single_sample_standard_deviation_is_json_safe(self) -> None:
        result = MODULE.describe(np.asarray([12.5], dtype=np.float64))
        self.assertIsNone(result["std_seconds"])
        self.assertIsNone(result["coefficient_of_variation"])

    def test_source_strata_are_kept_separate(self) -> None:
        rows = [
            {"source": "/machine-a/timing.yaml", "runtime_seconds": 10.0},
            {"source": "/machine-a/timing.yaml", "runtime_seconds": 14.0},
            {"source": "/machine-b/timing.yaml", "runtime_seconds": 20.0},
        ]
        strata = MODULE.describe_sources(rows)
        self.assertEqual(len(strata), 2)
        self.assertEqual(strata[0]["count"], 2)
        self.assertEqual(strata[0]["mean_seconds"], 12.0)
        self.assertEqual(strata[1]["count"], 1)
        self.assertIsNone(strata[1]["std_seconds"])

    def test_legacy_provenance_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                MODULE.provenance_stratum(root)
            self.assertEqual(
                MODULE.provenance_stratum(root, allow_legacy=True),
                "legacy:unrecorded",
            )


if __name__ == "__main__":
    unittest.main()
