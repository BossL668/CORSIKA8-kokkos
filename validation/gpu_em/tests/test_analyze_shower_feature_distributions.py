#!/usr/bin/env python3
"""Unit tests for shower-feature plot metadata helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analyze_shower_feature_distributions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_shower_feature_distributions", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrimaryLabelTests(unittest.TestCase):
    def test_elementary_primary_uses_pdg_label(self) -> None:
        self.assertEqual(MODULE.primary_label({"primary_pdg": 2212}), "proton")

    def test_iron_primary_uses_nuclear_metadata(self) -> None:
        self.assertEqual(
            MODULE.primary_label(
                {"primary_pdg": None, "primary_Z": 26, "primary_A": 56}
            ),
            r"$^{56}$Fe",
        )

    def test_incomplete_nuclear_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both primary_Z and primary_A"):
            MODULE.primary_label(
                {"primary_pdg": None, "primary_Z": 26, "primary_A": None}
            )


if __name__ == "__main__":
    unittest.main()
