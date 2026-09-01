#!/usr/bin/env python3

import hashlib
import json
import shlex
import tempfile
import unittest
from pathlib import Path

import yaml

from validation.gpu_em.compare_ensembles import infer_legacy_c8emrt_source
from validation.gpu_em.compare_gpu_physics_sources import (
    gpu_source_identity,
    summarize_fallbacks,
)


class CompareGpuPhysicsSourcesTest(unittest.TestCase):
    @staticmethod
    def write_legacy_c8emrt_fixture(root: Path) -> None:
        table_path = "/cache/table.c8emrt"
        command = [
            "/build/c8_air_shower",
            "-p",
            "2212",
            "--em-backend",
            "cuda",
            "--gpu-table-cache",
            table_path,
        ]
        encoded_command = json.dumps(
            command,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        (root / "gpu_em").mkdir(parents=True)
        (root / "validation_provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "cuda",
                    "executable": {"sha256": "a" * 64},
                    "table": {
                        "path": table_path,
                        "size_bytes": 1234,
                        "sha256": "b" * 64,
                    },
                    "command": command,
                    "command_sha256": hashlib.sha256(encoded_command).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        (root / "config.yaml").write_text(
            yaml.safe_dump({"args": shlex.join(command)}),
            encoding="utf-8",
        )
        (root / "gpu_em" / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "backend": "CORSIKA8GpuEm",
                    "table": {
                        "path": table_path,
                        "format_version": 10,
                        "sha256": "c" * 64,
                        "proposal_version": "7.6.2",
                        "generator_version": "c8-gpu-em-tablegen-0.18",
                        "medium": "air_dry_1_atm",
                        "energy_min_MeV": 0.4,
                        "energy_max_MeV": 1.05e8,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "gpu_em" / "summary.yaml").write_text(
            yaml.safe_dump({"shower_0": {"statistics": {}}}),
            encoding="utf-8",
        )

    def test_legacy_c8emrt_inference_is_strict_and_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy_c8emrt_fixture(root)

            with self.assertRaisesRegex(ValueError, "observed None"):
                gpu_source_identity(root, "c8emrt")

            evidence = infer_legacy_c8emrt_source(root)
            self.assertEqual(evidence["inferred_source"], "c8emrt")
            self.assertEqual(evidence["rule"], "strict-legacy-c8emrt-v1")
            self.assertEqual(evidence["events_without_source_metadata"], 1)

    def test_legacy_c8emrt_inference_rejects_source_contradiction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy_c8emrt_fixture(root)
            path = root / "gpu_em" / "summary.yaml"
            summary = yaml.safe_load(path.read_text(encoding="utf-8"))
            summary["shower_0"]["statistics"]["gpu_physics_source"] = None
            path.write_text(yaml.safe_dump(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contradicts"):
                infer_legacy_c8emrt_source(root)

    def test_legacy_c8emrt_inference_rejects_native_table_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy_c8emrt_fixture(root)
            path = root / "gpu_em" / "config.yaml"
            configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
            configuration["table"]["cubic_interpolation_version"] = "0.1.5"
            path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "native table metadata"):
                infer_legacy_c8emrt_source(root)

    def test_legacy_c8emrt_inference_rejects_command_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy_c8emrt_fixture(root)
            path = root / "config.yaml"
            configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
            configuration["args"] += " --verbosity warn"
            path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "commands differ"):
                infer_legacy_c8emrt_source(root)

    def test_fallback_accounting_is_source_local_and_closed(self):
        result = summarize_fallbacks(
            [
                {
                    "cpu_generic_fallbacks": 1,
                    "cpu_completed_selected_losses": 3,
                    "cpu_completed_native_selection_replays": 2,
                    "cpu_specified_final_states": 4,
                    "deferred_cpu_fallbacks_queued": 4,
                    "deferred_cpu_fallbacks_flushed": 4,
                    "cpu_fallbacks_by_reason_name": {
                        "unsupported_geometry": 1,
                        "native_selection_replay": 2,
                        "inverse_cdf_unavailable": 1,
                        "cpu_only_process": 1,
                    },
                    "cpu_fallbacks_by_reason": {2: 1, 6: 1, 12: 1, 28: 2},
                    "cpu_fallbacks_by_process_name": {"compton": 5},
                    "cpu_fallbacks_by_process": {1000000010: 5},
                }
            ]
        )
        self.assertEqual(result["generic_by_reason"]["unsupported_geometry"], 1)
        self.assertEqual(
            result["selected_loss_by_reason"]["native_selection_replay"], 2
        )
        self.assertEqual(result["cpu_completed_selected_losses"], 3)

    def test_unaccounted_selected_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selected-loss"):
            summarize_fallbacks(
                [
                    {
                        "cpu_generic_fallbacks": 0,
                        "cpu_completed_selected_losses": 1,
                        "cpu_completed_native_selection_replays": 0,
                        "cpu_specified_final_states": 0,
                        "deferred_cpu_fallbacks_queued": 0,
                        "deferred_cpu_fallbacks_flushed": 0,
                        "cpu_fallbacks_by_reason_name": {},
                        "cpu_fallbacks_by_reason": {},
                        "cpu_fallbacks_by_process_name": {},
                        "cpu_fallbacks_by_process": {},
                    }
                ]
            )

    def test_all_null_fallback_maps_are_valid_only_for_zero_counts(self):
        result = summarize_fallbacks(
            [
                {
                    "cpu_generic_fallbacks": 0,
                    "cpu_completed_selected_losses": 0,
                    "cpu_completed_native_selection_replays": 0,
                    "cpu_specified_final_states": 0,
                    "deferred_cpu_fallbacks_queued": 0,
                    "deferred_cpu_fallbacks_flushed": 0,
                    "cpu_fallbacks_by_reason_name": None,
                    "cpu_fallbacks_by_reason": None,
                    "cpu_fallbacks_by_process_name": None,
                    "cpu_fallbacks_by_process": None,
                }
            ]
        )
        self.assertEqual(result["cpu_generic_fallbacks"], 0)

    def test_unknown_specified_reason_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unpermitted"):
            summarize_fallbacks(
                [
                    {
                        "cpu_generic_fallbacks": 0,
                        "cpu_completed_selected_losses": 0,
                        "cpu_completed_native_selection_replays": 0,
                        "cpu_specified_final_states": 1,
                        "deferred_cpu_fallbacks_queued": 1,
                        "deferred_cpu_fallbacks_flushed": 1,
                        "cpu_fallbacks_by_reason_name": {"invalid_table_query": 1},
                        "cpu_fallbacks_by_reason": {9: 1},
                        "cpu_fallbacks_by_process_name": {"compton": 1},
                        "cpu_fallbacks_by_process": {1000000010: 1},
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
