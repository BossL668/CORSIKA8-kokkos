from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from watch_interim_campaign_milestones import (  # noqa: E402
    cuda_records,
    milestone_commands,
    proposal_records,
    select_exact_events,
    validate_distinct_records,
)


class InterimCampaignMilestoneWatcherTest(unittest.TestCase):
    def test_proposal_records_combine_remote_campaigns_and_sort_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ("late", "early")]
            for root in roots:
                root.mkdir()
            status = {
                "remote": [
                    {
                        "records": [
                            {
                                "path": str(roots[0]),
                                "seed": 10300009,
                                "runtime_seconds": 9.0,
                            }
                        ]
                    },
                    {
                        "records": [
                            {
                                "path": str(roots[1]),
                                "seed": 10300002,
                                "runtime_seconds": 2.0,
                            }
                        ]
                    },
                ]
            }
            observed = proposal_records(status)
            self.assertEqual([record["seed"] for record in observed], [10300002, 10300009])
            self.assertEqual([record["events"] for record in observed], [1, 1])

    def test_cuda_records_require_contiguous_complete_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            outputs = [campaign / f"batch_{index:03d}" / "cuda" for index in range(2)]
            for output in outputs:
                output.mkdir(parents=True)
            (campaign / "campaign_manifest.json").write_text(
                json.dumps(
                    {
                        "completed_total_events": 10,
                        "immutable_configuration": {"cuda_seed_start": 10400001},
                        "batches": [
                            {
                                "cuda_output": str(outputs[0]),
                                "seed": 10400001,
                                "events": 5,
                                "status": "complete",
                                "runtime_seconds": 10.0,
                            },
                            {
                                "cuda_output": str(outputs[1]),
                                "seed": 10400006,
                                "events": 5,
                                "status": "complete",
                                "runtime_seconds": 11.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            observed = cuda_records([campaign])
            self.assertEqual([record["seed"] for record in observed], [10400001, 10400006])
            self.assertEqual(sum(record["events"] for record in observed), 10)
            manifest = json.loads((campaign / "campaign_manifest.json").read_text())
            manifest["batches"][1]["seed"] = 10400007
            (campaign / "campaign_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                cuda_records([campaign])

    def test_exact_selection_never_splits_a_batch(self) -> None:
        records = [
            {"root": "/a", "seed": 10, "events": 5},
            {"root": "/b", "seed": 15, "events": 5},
        ]
        self.assertEqual(len(select_exact_events(records, 10, "CUDA")), 2)
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            select_exact_events(records, 7, "CUDA")

    def test_duplicate_or_overlapping_sources_are_rejected(self) -> None:
        duplicate_root = [
            {"root": "/same", "seed": 1, "events": 1},
            {"root": "/same", "seed": 2, "events": 1},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_distinct_records(duplicate_root, "test")
        overlapping = [
            {"root": "/a", "seed": 1, "events": 5},
            {"root": "/b", "seed": 5, "events": 5},
        ]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_distinct_records(overlapping, "test")

    def test_comparison_uses_a_child_of_the_manifest_directory(self) -> None:
        args = SimpleNamespace(
            geomagnetic_model="IGRF13",
            geomagnetic_year=2025.0,
            bootstrap_repetitions=20,
            pulse_analysis_root=Path("/pulse"),
        )
        root = Path("/milestone")
        commands = milestone_commands(
            args,
            root,
            root / "interim_manifest.json",
            [{"root": "/proposal"}],
            [{"root": "/cuda"}],
            10,
        )
        comparison = str(root / "ensemble_comparison")
        self.assertEqual(commands[0][commands[0].index("--output") + 1], comparison)
        self.assertIn(comparison, commands[1])
        self.assertIn(comparison, commands[-1])
        self.assertIn("analyze_runtime_selection_bias.py", commands[-1][1])


if __name__ == "__main__":
    unittest.main()
