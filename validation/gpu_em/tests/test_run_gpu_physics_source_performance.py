#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_gpu_physics_source_performance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_gpu_physics_source_performance", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ControlledArgumentTest(unittest.TestCase):
    def test_all_script_owned_options_are_detected(self) -> None:
        for argument, expected in (
            ("--max-weight=100", "--max-weight"),
            ("--emthin", "--emthin"),
            ("--gpu-memory-fraction=0.9", "--gpu-memory-fraction"),
            ("--radio-backend=cpu", "--radio-backend"),
            ("-N12", "-N"),
            ("-E=1000", "-E"),
            ("--max-weight 100", "--max-weight"),
        ):
            with self.subTest(argument=argument):
                self.assertEqual(MODULE.controlled_extra_option(argument), expected)

    def test_unowned_extra_option_remains_available(self) -> None:
        self.assertIsNone(MODULE.controlled_extra_option("--hadronModel=SIBYLL-2.3d"))
        self.assertIsNone(MODULE.controlled_extra_option("--gpu-full-step-records"))

    def test_empty_extra_argument_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not be empty"):
            MODULE.controlled_extra_option("  ")

    def test_validation_rejects_maximum_weight_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "controlled option --max-weight"):
            MODULE.validate_extra_args(["--max-weight=7"])


class ThinningGateTest(unittest.TestCase):
    def test_automatic_maximum_weight_matches_primary_energy(self) -> None:
        errors: list[str] = []
        observed = MODULE.validate_requested_thinning(
            errors,
            {
                "thinning": {
                    "em_fraction": 1.0e-6,
                    "maximum_weight": 0.5,
                    "automatic_maximum_weight": True,
                    "can_activate_from_unit_weight": False,
                }
            },
            requested_em_thinning=1.0e-6,
            configured_maximum_weight=0.0,
            primary_energy_gev=1.0e6,
        )
        self.assertEqual(errors, [])
        self.assertTrue(observed["automatic_maximum_weight"])

    def test_explicit_maximum_weight_mismatch_fails(self) -> None:
        errors: list[str] = []
        MODULE.validate_requested_thinning(
            errors,
            {
                "thinning": {
                    "em_fraction": 1.0e-6,
                    "maximum_weight": 50.0,
                    "automatic_maximum_weight": True,
                    "can_activate_from_unit_weight": True,
                }
            },
            requested_em_thinning=1.0e-6,
            configured_maximum_weight=100.0,
            primary_energy_gev=1.0e6,
        )
        self.assertTrue(any("maximum_weight" in error for error in errors))
        self.assertTrue(any("automatic_maximum_weight" in error for error in errors))


class FallbackAccountingTest(unittest.TestCase):
    @staticmethod
    def native_record() -> dict:
        return {
            "complete": True,
            "status": "complete",
            "statistics": {
                "gpu_physics_source": "proposal-native",
                "gpu_muon_transport_enabled": True,
                "backend_lifecycle": {"reused": False, "shower_ordinal": 1},
                "process_registry": {"accepted": True},
                "queue_overflows": 0,
                "cpu_memory_spill_particles": 0,
                "cross_species": {
                    "host_spills": 0,
                    "particles_spilled_to_cpu": 0,
                    "final_pending_photons": 0,
                    "final_pending_leptons": 0,
                },
                "profile": {"fixed_point_overflows": 0, "invalid_records": 0},
                "radio": {"fixed_point_overflows": 0},
                "epair_sampler": {"cpu_fallbacks": 0, "envelope_violations": 0},
                "thinning": {
                    "em_fraction": 1.0e-6,
                    "maximum_weight": 0.5,
                    "automatic_maximum_weight": True,
                    "can_activate_from_unit_weight": False,
                },
                "cpu_generic_fallbacks": 0,
                "cpu_specified_final_states": 1,
                "deferred_cpu_fallbacks_queued": 1,
                "deferred_cpu_fallbacks_flushed": 1,
                "deferred_cpu_fallback_flushes": 1,
                "deferred_fallback_scalar_expansion_rounds": 0,
                "maximum_deferred_cpu_fallback_batch": 1,
                "cpu_fallback_steps_executed": 0,
                "cpu_specified_fallback_time_ms": 0.1,
                "cpu_completed_selected_losses": 1,
                "cpu_completed_native_selection_replays": 1,
                "cpu_fallbacks_by_reason": {28: 1},
                "cpu_fallbacks_by_reason_name": {"native_selection_replay": 1},
                "cpu_fallbacks_by_process": {1000000010: 1},
                "cpu_fallbacks_by_process_name": {"compton": 1},
                "proposal_native": {
                    "inverse_failures": 0,
                    "proposal_cache_all_hit": True,
                    "aux_cache_hit": True,
                },
                "transfer_timing": {"device_event_timing_enabled": False},
                "energy_ledger": {"complete_coverage": False},
            },
        }

    def test_empty_reason_map_may_be_null_only_without_generic_fallback(self) -> None:
        errors: list[str] = []
        result = MODULE.validate_permitted_generic_fallbacks(
            errors,
            {
                "cpu_generic_fallbacks": 0,
                "cpu_fallbacks_by_reason_name": None,
            },
        )
        self.assertEqual(errors, [])
        self.assertEqual(sum(result.values()), 0)

    def test_unaccounted_generic_fallback_is_rejected(self) -> None:
        errors: list[str] = []
        MODULE.validate_permitted_generic_fallbacks(
            errors,
            {
                "cpu_generic_fallbacks": 1,
                "cpu_fallbacks_by_reason_name": {
                    "invalid_table_query": 1,
                },
            },
        )
        self.assertTrue(any("not exactly" in error for error in errors))

    def test_counter_json_aggregation_is_deterministic(self) -> None:
        result = MODULE.aggregate_counter_json(
            [
                {"reasons": '{"z":2,"a":1}'},
                {"reasons": '{"a":3}'},
            ],
            "reasons",
        )
        self.assertEqual(result, {"a": 4, "z": 2})

    def test_complete_fallback_ledger_is_recorded(self) -> None:
        _, diagnostics, errors = MODULE.validate_gpu_record(
            self.native_record(),
            "proposal-native",
            0,
            False,
            1.0e-6,
            0.0,
            1.0e6,
        )
        self.assertEqual(errors, [])
        self.assertEqual(diagnostics["cpu_specified_final_states"], 1)
        self.assertEqual(
            diagnostics["cpu_fallbacks_by_reason_name"],
            {"native_selection_replay": 1},
        )

    def test_unclosed_fallback_process_map_fails(self) -> None:
        record = self.native_record()
        record["statistics"]["cpu_fallbacks_by_process"] = None
        _, _, errors = MODULE.validate_gpu_record(
            record,
            "proposal-native",
            0,
            False,
            1.0e-6,
            0.0,
            1.0e6,
        )
        self.assertTrue(any("process-id fallback map" in error for error in errors))


class ResumableAttemptTest(unittest.TestCase):
    def test_completed_attempt_is_revalidated_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task"

            def command_for_output(output: Path) -> list[str]:
                return ["fake-c8", "-f", str(output)]

            def validate_output(output: Path) -> dict:
                marker = output / "complete.json"
                return json.loads(marker.read_text(encoding="utf-8"))

            def fake_run(command, log_path, environment, timeout_seconds):
                output = Path(command[command.index("-f") + 1])
                output.mkdir(parents=True)
                (output / "complete.json").write_text(
                    '{"accepted": true}\n', encoding="utf-8"
                )
                log_path.write_text("complete\n", encoding="utf-8")
                return 1.25

            with mock.patch.object(MODULE, "run_command", side_effect=fake_run) as run:
                elapsed, parsed, attempt, reused, _ = MODULE.run_or_reuse_attempt(
                    task_root,
                    command_for_output,
                    {},
                    0.0,
                    validate_output,
                    resume=False,
                )
                self.assertEqual(run.call_count, 1)
            self.assertEqual(elapsed, 1.25)
            self.assertEqual(parsed, {"accepted": True})
            self.assertEqual(attempt.name, "attempt_001")
            self.assertFalse(reused)

            with mock.patch.object(MODULE, "run_command") as run:
                elapsed, parsed, attempt, reused, _ = MODULE.run_or_reuse_attempt(
                    task_root,
                    command_for_output,
                    {},
                    0.0,
                    validate_output,
                    resume=True,
                )
                run.assert_not_called()
            self.assertEqual(elapsed, 1.25)
            self.assertEqual(parsed, {"accepted": True})
            self.assertEqual(attempt.name, "attempt_001")
            self.assertTrue(reused)

    def test_failed_attempt_is_preserved_and_next_attempt_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task"
            failed = task_root / "attempt_001"
            failed.mkdir(parents=True)
            MODULE.atomic_write_json(
                failed / "attempt.json",
                {"status": "failed", "command": ["old"]},
            )

            def command_for_output(output: Path) -> list[str]:
                return ["fake-c8", "-f", str(output)]

            def validate_output(output: Path) -> dict:
                return {"output": str(output)}

            def fake_run(command, log_path, environment, timeout_seconds):
                Path(command[command.index("-f") + 1]).mkdir(parents=True)
                return 2.0

            with mock.patch.object(MODULE, "run_command", side_effect=fake_run):
                _, _, attempt, reused, _ = MODULE.run_or_reuse_attempt(
                    task_root,
                    command_for_output,
                    {},
                    0.0,
                    validate_output,
                    resume=True,
                )
            self.assertEqual(attempt.name, "attempt_002")
            self.assertFalse(reused)
            self.assertTrue((failed / "attempt.json").is_file())


if __name__ == "__main__":
    unittest.main()
