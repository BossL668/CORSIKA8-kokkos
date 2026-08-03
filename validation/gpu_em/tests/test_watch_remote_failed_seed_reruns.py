#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from watch_remote_failed_seed_reruns import (  # noqa: E402
    audit_precompleted_repairs,
    audit_terminal_manifests,
)


ORIGINAL_HASH = "a" * 64
ANTENNA_HASH = "b" * 64
PHYSICS = {
    "primary_pdg": 2212,
    "energy_GeV": 1.0e8,
}
FIXED_HASH = "c" * 64
FLUKA_HASH = "d" * 64


def manifest(
    *,
    status: str,
    seed_start: int,
    events: int,
    completed: list[int],
    failed: list[int],
) -> dict:
    return {
        "status": status,
        "immutable_configuration": {
            "events": events,
            "seed_start": seed_start,
            "executable": {"sha256": ORIGINAL_HASH},
            "antenna_file": {"sha256": ANTENNA_HASH},
            "physics": PHYSICS,
        },
        "completed_indices": completed,
        "failures": [{"index": index} for index in failed],
    }


def repair_manifest(
    *,
    status: str,
    seeds: list[int],
    completed: list[int],
    failed: list[int],
) -> dict:
    return {
        "status": status,
        "immutable_configuration": {
            "events": len(seeds),
            "seed_schedule": seeds,
            "executable": {"sha256": FIXED_HASH},
            "antenna_file": {"sha256": ANTENNA_HASH},
            "flupro": {"sha256": FLUKA_HASH},
            "physics": PHYSICS,
        },
        "completed_indices": completed,
        "failures": [{"index": index} for index in failed],
    }


class RemoteFailedSeedWatcherTest(unittest.TestCase):
    def test_precompleted_repairs_remove_only_successful_original_failures(self) -> None:
        result = audit_precompleted_repairs(
            [
                repair_manifest(
                    status="failed",
                    seeds=[103, 107, 109],
                    completed=[0, 2],
                    failed=[1],
                )
            ],
            failed_seeds=[103, 107, 109, 111],
            expected_executable_sha256=FIXED_HASH,
            expected_antenna_sha256=ANTENNA_HASH,
            expected_fluka_sha256=FLUKA_HASH,
            expected_physics=PHYSICS,
        )
        self.assertEqual(result["completed_seeds"], [103, 109])
        self.assertEqual(result["failed_seeds"], [107])
        self.assertEqual(result["remaining_failed_seeds"], [107, 111])

    def test_precompleted_repairs_reject_nonfailed_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "not failed"):
            audit_precompleted_repairs(
                [
                    repair_manifest(
                        status="complete",
                        seeds=[103, 104],
                        completed=[0, 1],
                        failed=[],
                    )
                ],
                failed_seeds=[103],
                expected_executable_sha256=FIXED_HASH,
                expected_antenna_sha256=ANTENNA_HASH,
                expected_fluka_sha256=FLUKA_HASH,
                expected_physics=PHYSICS,
            )

    def test_precompleted_repairs_must_be_terminal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not terminal"):
            audit_precompleted_repairs(
                [
                    repair_manifest(
                        status="running",
                        seeds=[103],
                        completed=[],
                        failed=[],
                    )
                ],
                failed_seeds=[103],
                expected_executable_sha256=FIXED_HASH,
                expected_antenna_sha256=ANTENNA_HASH,
                expected_fluka_sha256=FLUKA_HASH,
                expected_physics=PHYSICS,
            )

    def test_terminal_partition_recovers_exact_failed_seeds(self) -> None:
        result = audit_terminal_manifests(
            [
                manifest(
                    status="complete",
                    seed_start=100,
                    events=2,
                    completed=[0, 1],
                    failed=[],
                ),
                manifest(
                    status="failed",
                    seed_start=102,
                    events=3,
                    completed=[0, 2],
                    failed=[1],
                ),
            ],
            expected_seed_start=100,
            expected_events=5,
            expected_executable_sha256=ORIGINAL_HASH,
            expected_antenna_sha256=ANTENNA_HASH,
        )
        self.assertEqual(result["completed_events"], 4)
        self.assertEqual(result["failed_events"], 1)
        self.assertEqual(result["failed_seeds"], [103])
        self.assertEqual(result["physics"], PHYSICS)

    def test_nonterminal_manifest_is_not_interpreted_as_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not terminal"):
            audit_terminal_manifests(
                [
                    manifest(
                        status="running_with_failures",
                        seed_start=100,
                        events=2,
                        completed=[0],
                        failed=[1],
                    )
                ],
                expected_seed_start=100,
                expected_events=2,
                expected_executable_sha256=ORIGINAL_HASH,
                expected_antenna_sha256=ANTENNA_HASH,
            )

    def test_terminal_manifest_must_partition_every_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not partition"):
            audit_terminal_manifests(
                [
                    manifest(
                        status="failed",
                        seed_start=100,
                        events=3,
                        completed=[0],
                        failed=[2],
                    )
                ],
                expected_seed_start=100,
                expected_events=3,
                expected_executable_sha256=ORIGINAL_HASH,
                expected_antenna_sha256=ANTENNA_HASH,
            )

    def test_campaign_seed_intervals_may_not_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlapping"):
            audit_terminal_manifests(
                [
                    manifest(
                        status="complete",
                        seed_start=100,
                        events=2,
                        completed=[0, 1],
                        failed=[],
                    ),
                    manifest(
                        status="complete",
                        seed_start=101,
                        events=2,
                        completed=[0, 1],
                        failed=[],
                    ),
                ],
                expected_seed_start=100,
                expected_events=3,
                expected_executable_sha256=ORIGINAL_HASH,
                expected_antenna_sha256=ANTENNA_HASH,
            )


if __name__ == "__main__":
    unittest.main()
