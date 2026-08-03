from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from watch_and_finalize_distributed_campaign import (  # noqa: E402
    audit_new_cuda_batches,
    campaign_ready,
    local_cuda_state,
    mark_partition_complete,
    precompleted_repair_state,
    run_rerun_staging_once,
)
from finalize_distributed_campaign import SourceRecord  # noqa: E402


class DistributedCampaignCompletionWatcherTest(unittest.TestCase):
    def test_precompleted_repair_state_combines_unique_staged_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "wave1.json", Path(temporary) / "wave2.json"]
            for index, path in enumerate(paths):
                seed = 10300030 + index * 10
                path.write_text(
                    json.dumps(
                        {
                            "status": "monitoring",
                            "remote": [
                                {
                                    "label": f"wave{index + 1}",
                                    "remote_status": "complete",
                                    "remote_failed": 0,
                                    "staged_events": 1,
                                    "records": [{"seed": seed}],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            observed = precompleted_repair_state(paths)
            self.assertEqual(observed["staged_events"], 2)
            self.assertEqual(observed["staged_seeds"], [10300030, 10300040])

    def test_precompleted_repair_state_rejects_duplicate_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "wave1.json", Path(temporary) / "wave2.json"]
            payload = {
                "status": "monitoring",
                "remote": [
                    {
                        "label": "wave",
                        "remote_status": "complete",
                        "remote_failed": 0,
                        "staged_events": 1,
                        "records": [{"seed": 10300030}],
                    }
                ],
            }
            for path in paths:
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                precompleted_repair_state(paths)

    def test_local_cuda_state_combines_independent_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / "a", Path(temporary) / "b"]
            for index, root in enumerate(roots):
                root.mkdir()
                (root / "campaign_manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "completed_total_events": 250,
                            "immutable_configuration": {
                                "target_events": 250,
                                "cuda_seed_start": 10400001 + 250 * index,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            state = local_cuda_state(roots)
            self.assertEqual(state["target_events"], 500)
            self.assertEqual(state["completed_events"], 500)
            self.assertEqual(
                [record["seed_start"] for record in state["records"]],
                [10400001, 10400251],
            )

    def test_ready_requires_exact_original_rerun_and_cuda_counts(self) -> None:
        cuda = {
            "target_events": 500,
            "completed_events": 500,
            "records": [{"status": "complete"}, {"status": "complete"}],
        }
        self.assertTrue(
            campaign_ready(
                original_staged=496,
                original_completed=496,
                rerun_staged=4,
                failed_events=4,
                rerun_remote_status="complete",
                cuda=cuda,
                expected_events=500,
            )
        )
        for change in (
            {"original_staged": 495},
            {"rerun_staged": 3},
            {"rerun_remote_status": "running"},
            {"cuda": {**cuda, "completed_events": 499}},
        ):
            arguments = {
                "original_staged": 496,
                "original_completed": 496,
                "rerun_staged": 4,
                "failed_events": 4,
                "rerun_remote_status": "complete",
                "cuda": cuda,
                "expected_events": 500,
            }
            arguments.update(change)
            self.assertFalse(campaign_ready(**arguments))

    def test_no_rerun_case_still_requires_closed_cuda_campaigns(self) -> None:
        self.assertFalse(
            campaign_ready(
                original_staged=500,
                original_completed=500,
                rerun_staged=0,
                failed_events=0,
                rerun_remote_status="not_required",
                cuda={
                    "target_events": 500,
                    "completed_events": 500,
                    "records": [{"status": "complete"}, {"status": "running"}],
                },
                expected_events=500,
            )
        )

    def test_missing_new_rerun_manifest_is_a_retryable_poll_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "poll_failed",
                        "error_type": "CalledProcessError",
                        "error": "remote run_manifest.json is not present yet",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                monitor_script=Path("monitor.py"),
                remote_host="remote",
                ssh_control_path=Path("/tmp/control"),
                remote_rerun_root="/remote/rerun",
                staging_root=Path("/local/staging"),
                rerun_monitor_status=status_path,
            )
            with patch(
                "watch_and_finalize_distributed_campaign.subprocess.run"
            ) as run:
                run.return_value.returncode = 1
                observed = run_rerun_staging_once(args)
            self.assertEqual(observed["status"], "poll_failed")

    def test_partition_is_completed_only_from_running_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "partition.json"
            readiness = root / "readiness.json"
            path.write_text(
                json.dumps({"schema_version": 1, "status": "running"}),
                encoding="utf-8",
            )
            observed = mark_partition_complete(path, readiness)
            self.assertEqual(observed["status"], "complete")
            self.assertEqual(
                observed["finalization_readiness"], str(readiness.resolve())
            )
            with self.assertRaisesRegex(ValueError, "not in the running state"):
                mark_partition_complete(path, readiness)

    def test_new_closed_cuda_batches_are_strictly_audited_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            output = root / "batch_000" / "cuda"
            root.mkdir(parents=True)
            (root / "campaign_manifest.json").write_text(
                json.dumps(
                    {
                        "completed_total_events": 5,
                        "immutable_configuration": {"cuda_seed_start": 10400001},
                        "batches": [
                            {
                                "cuda_output": str(output),
                                "events": 5,
                                "seed": 10400001,
                                "status": "complete",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record = SourceRecord(
                backend="cuda",
                root=str(output),
                seed=10400001,
                events=5,
                executable_sha256="a" * 64,
                table_sha256="b" * 64,
                antenna_sha256="c" * 64,
                observer_layout_sha256="d" * 64,
            )
            cache = {}
            with patch(
                "watch_and_finalize_distributed_campaign.audit_source",
                return_value=record,
            ) as audit:
                first = audit_new_cuda_batches(
                    [root], cache, antenna_sha256="c" * 64
                )
                second = audit_new_cuda_batches(
                    [root], cache, antenna_sha256="c" * 64
                )
            self.assertEqual(audit.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first["strictly_audited_events"], 5)


if __name__ == "__main__":
    unittest.main()
