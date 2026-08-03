from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from compare_ensembles import Ensemble
from verify_reused_backend_equivalence import compare_paired


def make_ensemble(value: float, events: int = 3) -> Ensemble:
    axis = np.asarray([0.0, 1.0])
    matrix = np.full((events, 2), value)
    return Ensemble(
        name="test",
        root=Path("/tmp/test"),
        showers=tuple(range(events)),
        scalars=pd.DataFrame(
            {
                "a": np.full(events, value),
                "b": np.arange(events, dtype=np.float64),
            }
        ),
        curves={"curve": (axis, matrix)},
        histograms={"histogram": (axis, matrix)},
        metadata={},
    )


class ReusedBackendEquivalenceTest(unittest.TestCase):
    def test_identical_paired_ensembles_pass(self) -> None:
        result = compare_paired(
            make_ensemble(1.0), make_ensemble(1.0), 3
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["scalar_columns"], 2)

    def test_one_changed_value_fails(self) -> None:
        fresh = make_ensemble(1.0)
        reused = make_ensemble(1.0)
        reused.curves["curve"][1][1, 0] = 2.0
        result = compare_paired(fresh, reused, 3)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["curves"]["curve"]["values"][
                "different_values"
            ],
            1,
        )

    def test_rejects_excess_event_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            compare_paired(
                make_ensemble(1.0),
                make_ensemble(1.0),
                4,
            )

    def test_rejects_build_or_table_provenance_mismatch(self) -> None:
        fresh = make_ensemble(1.0)
        reused = make_ensemble(1.0)
        fresh.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "a" * 64,
            "b" * 64,
        )
        reused.metadata["provenance_fingerprint"] = (
            1,
            "cuda",
            "a" * 64,
            "c" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "build/table provenance differs",
        ):
            compare_paired(
                fresh,
                reused,
                3,
            )


if __name__ == "__main__":
    unittest.main()
