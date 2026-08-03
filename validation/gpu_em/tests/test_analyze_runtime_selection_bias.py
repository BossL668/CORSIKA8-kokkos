from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from analyze_runtime_selection_bias import (  # noqa: E402
    attach_proposal_timings,
    correlation_summary,
    source_timing_rows,
)


class RuntimeSelectionBiasTest(unittest.TestCase):
    def make_source(self, root: Path, seed: int, times_ms: list[float]) -> None:
        root.mkdir(parents=True)
        (root / "config.yaml").write_text(
            f'args: "c8_air_shower --seed {seed} -N {len(times_ms)}"\n',
            encoding="utf-8",
        )
        timing = root / "simulation_timing"
        timing.mkdir()
        (timing / "summary.yaml").write_text(
            "\n".join(
                (
                    f"shower_{index}:\n"
                    "  closed: true\n"
                    "  status: closed\n"
                    f"  wall_time_ms: {value}"
                )
                for index, value in enumerate(times_ms)
            )
            + "\n",
            encoding="utf-8",
        )

    def test_source_rows_recover_seed_and_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            self.make_source(root, 1234, [1000.0, 2500.0])
            rows = source_timing_rows(root)
            self.assertEqual([row["seed"] for row in rows], [1234, 1235])
            self.assertEqual(
                [row["runtime_seconds"] for row in rows], [1.0, 2.5]
            )

    def test_attach_timings_preserves_recorded_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            early = Path(temporary) / "early"
            late = Path(temporary) / "late"
            self.make_source(early, 10, [1000.0])
            self.make_source(late, 20, [2000.0, 3000.0])
            observables = pd.DataFrame(
                {
                    "backend": ["proposal", "proposal", "proposal", "cuda"],
                    "metric": [1.0, 2.0, 3.0, 4.0],
                }
            )
            frame = attach_proposal_timings(
                {"proposal": {"sources": [str(late), str(early)]}},
                observables,
            )
            self.assertEqual(frame["seed"].tolist(), [20, 21, 10])
            self.assertEqual(frame["metric"].tolist(), [1.0, 2.0, 3.0])

    def test_correlation_summary_detects_monotonic_relation(self) -> None:
        frame = pd.DataFrame(
            {
                "runtime_seconds": np.arange(1.0, 13.0),
                "metric": np.arange(1.0, 13.0) * 2.0,
            }
        )
        result = correlation_summary(frame, "metric")
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertEqual(result["tail_count"], 4)
        self.assertLess(
            result["fastest_tail_over_all_mean"],
            result["slowest_tail_over_all_mean"],
        )

    def test_duplicate_seeds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self.make_source(first, 10, [1000.0])
            self.make_source(second, 10, [2000.0])
            observables = pd.DataFrame(
                {"backend": ["proposal", "proposal"], "metric": [1.0, 2.0]}
            )
            with self.assertRaisesRegex(ValueError, "duplicate proposal seeds"):
                attach_proposal_timings(
                    {"proposal": {"sources": [str(first), str(second)]}},
                    observables,
                )


if __name__ == "__main__":
    unittest.main()
