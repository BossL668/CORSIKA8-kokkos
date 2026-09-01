#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validation.gpu_em.run_gpu_physics_source_ensemble import (
    BatchTask,
    next_attempt_number,
    reconcile_interrupted_attempts,
    summarize_completion,
    validate_fallback_accounting,
    validate_permitted_generic_fallbacks,
)


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]


class GpuPhysicsSourceEnsembleTest(unittest.TestCase):
    def dry_run_command(self, root: Path) -> list[str]:
        executable = root / "c8_air_shower"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        antenna = root / "antennas.txt"
        antenna.write_text("0 0 0\n", encoding="utf-8")
        flupro = root / "fluka"
        flupro.mkdir()
        (flupro / "libflukahp.a").write_bytes(b"test-fluka-library")
        return [
            sys.executable,
            str(MODULE_DIRECTORY / "run_gpu_physics_source_ensemble.py"),
            "--executable",
            str(executable),
            "--output-root",
            str(root / "output"),
            "--antenna-file",
            str(antenna),
            "--flupro",
            str(flupro),
            "--events-per-source",
            "5",
            "--batch-events",
            "2",
            "--dry-run",
        ]

    def test_help_exposes_explicit_maximum_weight(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIRECTORY / "run_gpu_physics_source_ensemble.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--maximum-weight", completed.stdout)
        self.assertIn("automatic value", completed.stdout)
        self.assertIn("--sources", completed.stdout)

    def test_native_only_dry_run_needs_no_c8emrt_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.dry_run_command(root)
            command.extend(
                (
                    "--sources",
                    "proposal-native",
                    "--native-seed-start",
                    "100",
                    # Deliberately overlap the unselected source's range.  It
                    # must not participate in single-source validation.
                    "--c8emrt-seed-start",
                    "100",
                )
            )
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            payload = json.loads(completed.stdout)
            immutable = payload["immutable_configuration"]
            self.assertEqual(immutable["sources"], ["proposal-native"])
            self.assertEqual(immutable["seeds"], {"proposal_native_start": 100})
            self.assertNotIn("c8emrt_table", immutable["artifacts"])
            self.assertEqual(
                payload["expected_event_counts"],
                {
                    "formal": {"proposal-native": 5},
                    "paired": {"proposal-native": 0},
                },
            )
            self.assertEqual(payload["task_count"], 3)
            self.assertEqual(
                [task["source"] for task in payload["tasks"]],
                ["proposal-native"] * 3,
            )
            self.assertEqual(
                [task["seed"] for task in payload["tasks"]], [100, 102, 104]
            )
            for task in payload["tasks"]:
                self.assertNotIn("--gpu-table-cache", task["command"])

    def test_default_dual_source_dry_run_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "air.c8emrt"
            table.write_bytes(b"test-c8emrt-table")
            command = self.dry_run_command(root)
            command.extend(("--c8emrt-table", str(table)))
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            payload = json.loads(completed.stdout)
            immutable = payload["immutable_configuration"]
            self.assertEqual(immutable["sources"], ["c8emrt", "proposal-native"])
            self.assertIn("c8emrt_table", immutable["artifacts"])
            self.assertEqual(
                payload["expected_event_counts"],
                {
                    "formal": {"c8emrt": 5, "proposal-native": 5},
                    "paired": {"c8emrt": 0, "proposal-native": 0},
                },
            )
            self.assertEqual(payload["task_count"], 6)
            self.assertEqual(
                {task["source"] for task in payload["tasks"]},
                {"c8emrt", "proposal-native"},
            )

    def test_single_source_rejects_paired_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = self.dry_run_command(root)
            command.extend(
                ("--sources", "proposal-native", "--paired-events", "1")
            )
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "--paired-events requires both c8emrt and proposal-native sources",
                completed.stderr,
            )

    def test_resume_reconciles_interrupted_attempt_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = BatchTask("formal", "proposal-native", 0, 25, 1234)
            batch_parent = root / "formal" / "batch_000"
            stale = batch_parent / ".attempt-proposal-native-000"
            stale.mkdir(parents=True)
            (stale / "partial-output.txt").write_text("partial", encoding="utf-8")
            manifest = {
                "attempts": [
                    {
                        "task": "formal:000:proposal-native",
                        "attempt": 0,
                        "status": "running",
                    }
                ]
            }

            changed = reconcile_interrupted_attempts(
                root, manifest, task, batch_parent
            )

            self.assertTrue(changed)
            self.assertFalse(stale.exists())
            record = manifest["attempts"][0]
            self.assertEqual(record["status"], "interrupted_orphaned")
            self.assertEqual(record["interrupted_status"], "running")
            self.assertTrue(record["private_output_was_present"])
            archived = Path(record["archived_output"])
            self.assertTrue((archived / "partial-output.txt").is_file())
            self.assertEqual(next_attempt_number(manifest, task), 1)

    def test_resume_archives_untracked_attempt_and_uses_maximum_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = BatchTask("formal", "proposal-native", 0, 25, 1234)
            batch_parent = root / "formal" / "batch_000"
            stale = batch_parent / ".attempt-proposal-native-004"
            stale.mkdir(parents=True)
            manifest = {
                "attempts": [
                    {
                        "task": "formal:000:proposal-native",
                        "attempt": 1,
                        "status": "failed",
                    }
                ]
            }

            changed = reconcile_interrupted_attempts(
                root, manifest, task, batch_parent
            )

            self.assertTrue(changed)
            orphan = next(
                item for item in manifest["attempts"] if item["attempt"] == 4
            )
            self.assertEqual(
                orphan["status"], "orphaned_untracked_before_restart"
            )
            self.assertTrue(Path(orphan["archived_output"]).is_dir())
            self.assertEqual(next_attempt_number(manifest, task), 5)

    def test_single_source_completion_summary_has_no_unselected_arm(self):
        task = BatchTask("formal", "proposal-native", 0, 5, 100)
        manifest = {
            "completed": {
                "formal:000:proposal-native": {"status": "complete"}
            }
        }
        self.assertEqual(
            summarize_completion(manifest, [task], ("proposal-native",)),
            {
                "formal": {"proposal-native": 5},
                "paired": {"proposal-native": 0},
            },
        )

    def test_declared_geometry_fallback_is_permitted(self):
        result = validate_permitted_generic_fallbacks(
            {
                "cpu_generic_fallbacks": 1,
                "cpu_specified_final_states": 23,
                "cpu_fallbacks_by_reason_name": {
                    "unsupported_geometry": 1,
                    "native_selection_replay": 23,
                },
            },
            shower_label="shower_0",
        )
        self.assertEqual(result["unsupported_geometry"], 1)
        self.assertEqual(result["unsupported_particle"], 0)
        self.assertEqual(result["unsupported_medium"], 0)

    def test_null_reason_map_is_permitted_only_when_generic_total_is_zero(self):
        result = validate_permitted_generic_fallbacks(
            {
                "cpu_generic_fallbacks": 0,
                "cpu_specified_final_states": 0,
                "cpu_fallbacks_by_reason_name": None,
            },
            shower_label="shower_0",
        )
        self.assertEqual(sum(result.values()), 0)

        with self.assertRaisesRegex(ValueError, "reason mapping is missing"):
            validate_permitted_generic_fallbacks(
                {
                    "cpu_generic_fallbacks": 1,
                    "cpu_specified_final_states": 0,
                    "cpu_fallbacks_by_reason_name": None,
                },
                shower_label="shower_0",
            )

        with self.assertRaisesRegex(ValueError, "reason mapping is missing"):
            validate_permitted_generic_fallbacks(
                {
                    "cpu_generic_fallbacks": 0,
                    "cpu_specified_final_states": 1,
                    "cpu_fallbacks_by_reason_name": None,
                },
                shower_label="shower_0",
            )

    def test_complete_fallback_ledger_accepts_native_replay(self):
        result = validate_fallback_accounting(
            {
                "cpu_generic_fallbacks": 0,
                "cpu_specified_final_states": 3,
                "deferred_cpu_fallbacks_queued": 3,
                "deferred_cpu_fallbacks_flushed": 3,
                "cpu_completed_selected_losses": 3,
                "cpu_completed_native_selection_replays": 3,
                "cpu_fallbacks_by_reason": {28: 3},
                "cpu_fallbacks_by_reason_name": {
                    "native_selection_replay": 3,
                },
                "cpu_fallbacks_by_process": {1000000010: 3},
                "cpu_fallbacks_by_process_name": {"compton": 3},
            },
            source="proposal-native",
            shower_label="shower_0",
        )
        self.assertEqual(result["specified"], 3)

    def test_complete_fallback_ledger_accepts_all_null_maps_when_empty(self):
        result = validate_fallback_accounting(
            {
                "cpu_generic_fallbacks": 0,
                "cpu_specified_final_states": 0,
                "deferred_cpu_fallbacks_queued": 0,
                "deferred_cpu_fallbacks_flushed": 0,
                "cpu_completed_selected_losses": 0,
                "cpu_completed_native_selection_replays": 0,
                "cpu_fallbacks_by_reason": None,
                "cpu_fallbacks_by_reason_name": None,
                "cpu_fallbacks_by_process": None,
                "cpu_fallbacks_by_process_name": None,
            },
            source="proposal-native",
            shower_label="shower_0",
        )
        self.assertEqual(result["specified"], 0)

    def test_complete_fallback_ledger_rejects_unclosed_maps(self):
        with self.assertRaisesRegex(ValueError, "does not close"):
            validate_fallback_accounting(
                {
                    "cpu_generic_fallbacks": 0,
                    "cpu_specified_final_states": 1,
                    "deferred_cpu_fallbacks_queued": 1,
                    "deferred_cpu_fallbacks_flushed": 1,
                    "cpu_completed_selected_losses": 1,
                    "cpu_completed_native_selection_replays": 1,
                    "cpu_fallbacks_by_reason": {28: 1},
                    "cpu_fallbacks_by_reason_name": {
                        "native_selection_replay": 1,
                    },
                    "cpu_fallbacks_by_process": {},
                    "cpu_fallbacks_by_process_name": {"compton": 1},
                },
                source="proposal-native",
                shower_label="shower_0",
            )

    def test_unknown_generic_fallback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not exactly"):
            validate_permitted_generic_fallbacks(
                {
                    "cpu_generic_fallbacks": 1,
                    "cpu_specified_final_states": 0,
                    "cpu_fallbacks_by_reason_name": {
                        "invalid_table_query": 1,
                    },
                },
                shower_label="shower_0",
            )

    def test_permitted_reason_count_must_equal_generic_total(self):
        with self.assertRaisesRegex(ValueError, "not exactly"):
            validate_permitted_generic_fallbacks(
                {
                    "cpu_generic_fallbacks": 0,
                    "cpu_specified_final_states": 0,
                    "cpu_fallbacks_by_reason_name": {
                        "unsupported_geometry": 1,
                    },
                },
                shower_label="shower_0",
            )


if __name__ == "__main__":
    unittest.main()
