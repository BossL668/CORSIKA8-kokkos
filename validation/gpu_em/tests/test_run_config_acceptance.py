import argparse
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from validation.gpu_em.run_config_acceptance import (
    parse_task_physics,
    summarize_hadronic_pool,
    top_level_consistency_warnings,
    translated_command,
)


class ConfigAcceptanceTest(unittest.TestCase):
    def write_config(self, root: Path, args: list[str]) -> Path:
        path = root / "config.yaml"
        with path.open("w", encoding="utf-8") as destination:
            yaml.safe_dump(
                {
                    "energy_range": [1.0e9, 1.0e9],
                    "emthin_values": [1.0e-6],
                    "tasks": [
                        {
                            "name": "proton_config_task",
                            "args": args,
                        }
                    ],
                },
                destination,
                sort_keys=False,
            )
        return path

    def base_task(self) -> list[str]:
        return [
            "-p",
            "2212",
            "-E",
            "1e6",
            "-N",
            "1",
            "-z",
            "27",
            "-a",
            "180",
            "--shower-core-x",
            "0",
            "--shower-core-y",
            "0",
            "--ring",
            "2",
            "--emthin",
            "1e-6",
            "-s",
            "0",
            "-f",
            "configured_name",
        ]

    def test_task_args_are_authoritative_and_auto_weight_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(root, self.base_task())
            physics, document, raw = parse_task_physics(path, 0)
            self.assertEqual(physics.energy_GeV, 1.0e6)
            self.assertEqual(physics.em_thinning, 1.0e-6)
            self.assertEqual(physics.maximum_weight, 0.0)
            self.assertEqual(physics.ring, 2)
            self.assertEqual(raw, self.base_task())
            warnings = top_level_consistency_warnings(document, physics)
            self.assertEqual(
                warnings,
                [
                    "top-level energy_range differs from task -E; task args "
                    "are authoritative"
                ],
            )

    def test_unknown_duplicate_and_backend_options_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = self.write_config(
                root, self.base_task() + ["--force-interaction"]
            )
            with self.assertRaisesRegex(ValueError, "cannot yet be reproduced"):
                parse_task_physics(unknown, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = self.write_config(
                root, self.base_task() + ["--emthin", "1e-5"]
            )
            with self.assertRaisesRegex(ValueError, "more than once"):
                parse_task_physics(duplicate, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = self.write_config(
                root, self.base_task() + ["--em-backend", "cuda"]
            )
            with self.assertRaisesRegex(ValueError, "implementation option"):
                parse_task_physics(backend, 0)

    def test_translation_marks_development_changes(self):
        physics = parse_task_physics(
            Path("/home/yuhanglu/21CMA/python/config.yaml"), 0
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "c8"
            proposal = root / "proposal"
            table = root / "table"
            for path in (executable, proposal, table):
                path.touch()
            args = argparse.Namespace(
                proposal_seed=None,
                cuda_seed=None,
                development_emthin=1.0e-5,
                antenna_file=None,
                shower_only=True,
                cuda_radio_backend="cuda",
                cuda_executable=executable,
                proposal_executable=proposal,
                table=table,
                output_root=root / "output",
                events=10,
                proposal_shards=1,
                proposal_parallelism=1,
                hadronic_workers=4,
                hadronic_min_batch=64,
                hadronic_target_batch_ms=5.0,
                hadronic_max_batch=256,
                gpu_device=0,
                gpu_min_batch=4096,
                gpu_memory_fraction=0.7,
                gpu_table_tolerance=1.0e-3,
                stability_bootstrap_repetitions=5000,
                relative_tolerance=0.01,
                sigma_limit=3.0,
                require_pass=False,
            )
            command, mapping = translated_command(args, physics)
            self.assertFalse(mapping["production_equivalent"])
            self.assertTrue(mapping["em_thinning_overridden"])
            self.assertTrue(mapping["shower_only"])
            self.assertEqual(mapping["effective_em_thinning"], 1.0e-5)
            self.assertEqual(mapping["effective_maximum_weight"], 5.0)
            self.assertTrue(mapping["automatic_maximum_weight"])
            self.assertTrue(
                mapping["thinning_can_activate_from_unit_weight"]
            )
            self.assertEqual(mapping["effective_ring"], 0)
            self.assertEqual(mapping["effective_cuda_radio_backend"], "cpu")
            self.assertIn("--maximum-weight", command)
            self.assertEqual(
                command[command.index("--maximum-weight") + 1], "0"
            )

    def test_1pev_emthin_1e_minus_6_auto_weight_boundary_is_explicit(self):
        physics = parse_task_physics(
            Path("/home/yuhanglu/21CMA/python/config.yaml"), 0
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "c8"
            proposal = root / "proposal"
            table = root / "table"
            for path in (executable, proposal, table):
                path.touch()
            args = argparse.Namespace(
                proposal_seed=41001,
                cuda_seed=51001,
                development_emthin=None,
                antenna_file=None,
                shower_only=False,
                cuda_radio_backend="cuda",
                cuda_executable=executable,
                proposal_executable=proposal,
                table=table,
                output_root=root / "output",
                events=2,
                proposal_shards=1,
                proposal_parallelism=1,
                hadronic_workers=4,
                hadronic_min_batch=64,
                hadronic_target_batch_ms=5.0,
                hadronic_max_batch=256,
                gpu_device=0,
                gpu_min_batch=4096,
                gpu_memory_fraction=0.7,
                gpu_table_tolerance=1.0e-3,
                stability_bootstrap_repetitions=0,
                relative_tolerance=0.01,
                sigma_limit=3.0,
                require_pass=False,
            )
            _, mapping = translated_command(args, physics)
            self.assertEqual(mapping["effective_em_thinning"], 1.0e-6)
            self.assertEqual(mapping["effective_maximum_weight"], 0.5)
            self.assertTrue(mapping["automatic_maximum_weight"])
            self.assertFalse(
                mapping["thinning_can_activate_from_unit_weight"]
            )

    def test_hadronic_bottleneck_summary_uses_wall_share_and_balance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "cuda" / "gpu_em"
            summary.mkdir(parents=True)
            with (summary / "summary.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "shower_0": {
                            "complete": True,
                            "statistics": {
                                "hybrid_timing_ms": {"total_run": 1000.0},
                                "hadronic_models": {
                                    "high_energy": {
                                        "final_state_time_ms": 5.0
                                    },
                                    "low_energy": {
                                        "final_state_time_ms": 195.0
                                    },
                                },
                                "hadronic_process_pool": {
                                    "execute_time_ms": 120.0,
                                    "prepared_interactions": 20,
                                    "actual_worker_loads": [
                                        {
                                            "worker_id": 0,
                                            "final_state_time_ms": 100.0,
                                            "requests": 12,
                                        },
                                        {
                                            "worker_id": 1,
                                            "final_state_time_ms": 98.0,
                                            "requests": 8,
                                        },
                                    ],
                                    "flush_loads": [
                                        {
                                            "trigger": "estimated_cost",
                                            "requests": 64,
                                            "predicted_cost_by_worker": [
                                                5.0,
                                                4.9,
                                            ],
                                            "actual_final_state_time_by_worker_ms": [
                                                4.8,
                                                5.1,
                                            ],
                                        },
                                        {
                                            "trigger": "drain",
                                            "requests": 2,
                                            "predicted_cost_by_worker": [
                                                0.1,
                                                0.1,
                                            ],
                                            "actual_final_state_time_by_worker_ms": [
                                                0.01,
                                                0.0,
                                            ],
                                        },
                                    ],
                                },
                            },
                        }
                    },
                    destination,
                    sort_keys=False,
                )
            result = summarize_hadronic_pool(root)
            self.assertTrue(result["available"])
            self.assertAlmostEqual(result["process_pool_wall_fraction"], 0.12)
            self.assertAlmostEqual(
                result["worker_max_over_min_final_state_time"],
                100.0 / 98.0,
            )
            self.assertEqual(
                result["cost_triggered_flushes_with_loads"], 1
            )
            self.assertEqual(result["cost_flush_mean_requests"], 64.0)
            self.assertAlmostEqual(
                result["cost_flush_mean_predicted_worker_max_over_min"],
                5.0 / 4.9,
            )
            self.assertAlmostEqual(
                result["cost_flush_mean_actual_worker_max_over_min"],
                5.1 / 4.8,
            )
            self.assertTrue(result["gpu_hadronic_transport_recommended"])


if __name__ == "__main__":
    unittest.main()
