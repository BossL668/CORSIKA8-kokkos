#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from monitor_distributed_campaign import (  # noqa: E402
    compact_status,
    expected_seed,
    parse_labeled_path,
    validate_staged_shard,
)


class DistributedCampaignMonitorTest(unittest.TestCase):
    def test_labeled_remote_path_and_seed_schedules(self) -> None:
        self.assertEqual(
            parse_labeled_path("main=/data/run"),
            ("main", "/data/run"),
        )
        with self.assertRaisesRegex(ValueError, "LABEL=PATH"):
            parse_labeled_path("/data/run")
        with self.assertRaisesRegex(ValueError, "absolute"):
            parse_labeled_path("main=relative/run")
        self.assertEqual(
            expected_seed(
                {"immutable_configuration": {"seed_start": 100}},
                3,
            ),
            103,
        )
        self.assertEqual(
            expected_seed(
                {"immutable_configuration": {"seed_schedule": [7, 11]}},
                1,
            ),
            11,
        )

    def test_staged_shard_requires_closed_matching_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "profile/profile.parquet",
                "production_profile/profile.parquet",
                "particles/particles.parquet",
                "energyloss/dEdX.parquet",
                "CoREAS/summary.yaml",
                "ZHS/summary.yaml",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "config.yaml").touch()
            with (root / "summary.yaml").open("w", encoding="utf-8") as output:
                yaml.safe_dump(
                    {"showers": 1, "seed": 123, "runtime_raw": 10.0},
                    output,
                )
            timing = root / "simulation_timing" / "summary.yaml"
            timing.parent.mkdir()
            with timing.open("w", encoding="utf-8") as output:
                yaml.safe_dump(
                    {
                        "shower_0": {
                            "closed": True,
                            "status": "closed",
                            "wall_time_ms": 9000.0,
                        }
                    },
                    output,
                )
            (root / "validation_provenance.json").write_text(
                json.dumps(
                    {
                        "backend": "proposal",
                        "seed": 123,
                        "executable": {"sha256": "a" * 64},
                    }
                ),
                encoding="utf-8",
            )
            record = validate_staged_shard(root, 123, "a" * 64)
            self.assertEqual(record["seed"], 123)
            self.assertEqual(record["wall_time_seconds"], 9.0)
            with self.assertRaisesRegex(ValueError, "summary seed differs"):
                validate_staged_shard(root, 124, "a" * 64)

    def test_compact_status_does_not_repeat_per_event_records(self) -> None:
        compact = compact_status(
            {
                "status": "monitoring",
                "updated_utc": "now",
                "remote": [
                    {
                        "label": "main",
                        "remote_status": "running",
                        "remote_completed": 2,
                        "remote_failed": 1,
                        "staged_events": 2,
                        "records": [
                            {"transferred_this_poll": False},
                            {"transferred_this_poll": True},
                        ],
                    }
                ],
                "local": [
                    {
                        "path": "/cuda-a",
                        "status": "running",
                        "completed_events": 5,
                        "target_events": 250,
                    }
                ],
            }
        )
        self.assertNotIn("records", compact["remote"][0])
        self.assertEqual(
            compact["remote"][0]["transferred_this_poll"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
