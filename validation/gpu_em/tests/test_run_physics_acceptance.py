#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

import run_physics_acceptance as runner  # noqa: E402


def make_arguments(root: Path) -> argparse.Namespace:
    executable = root / "c8_air_shower"
    table = root / "table.c8emrt"
    executable.touch()
    table.touch()
    return argparse.Namespace(
        executable=executable,
        proposal_executable=None,
        table=table,
        output_root=root / "output",
        additional_proposal=[],
        additional_cuda=[],
        energy_gev=1000.0,
        events=10,
        proposal_seed=100,
        cuda_seed=200,
        paired_seed_control=False,
        proposal_shards=3,
        proposal_parallelism=2,
        resume_completed_proposal=False,
        overlap_backends=False,
        antenna_file=Path("/dev/null"),
        em_cut_gev=0.0005,
        non_em_cut_gev=None,
        had_cut_gev=0.3,
        mu_cut_gev=0.3,
        tau_cut_gev=0.3,
        shower_core_x_m=0.0,
        shower_core_y_m=0.0,
        em_thinning=1.0e-4,
        maximum_weight=100.0,
        cuda_hadronic_backend="scalar",
        cuda_hadronic_workers=4,
        cuda_hadronic_min_batch=64,
        cuda_hadronic_target_batch_ms=5.0,
        cuda_hadronic_max_batch=256,
        cuda_radio_backend="cpu",
        gpu_radio_field_limit=1.0,
        gpu_radio_track_diagnostics=False,
        radio_sampling_rate_ghz=1.0,
        radio_window_duration_ns=400.0,
        radio_pretrigger_ns=10.0,
        gpu_memory_fraction=0.7,
        relative_tolerance=0.01,
        sigma_limit=3.0,
        active_fraction=1.0e-4,
        minimum_bin_pass_fraction=0.95,
        stability_bootstrap_repetitions=20000,
    )


class PhysicsAcceptanceRunnerTest(unittest.TestCase):
    def test_event_split_preserves_total_and_balance(self):
        self.assertEqual(runner.split_event_count(10, 3), [4, 3, 3])

    def test_completed_proposal_shard_is_verified_before_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "proposal_shard_000"
            output.mkdir()
            command = ["c8_air_shower", "-N", "1", "--seed", "100"]
            with (output / "summary.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {
                        "showers": 1,
                        "seed": 100,
                        "start time": "2026-01-01T00:00:00+0000",
                        "end time": "2026-01-01T00:00:05+0000",
                        "runtime_raw": 4.5,
                    },
                    destination,
                )
            with (output / "config.yaml").open(
                "w", encoding="utf-8"
            ) as destination:
                yaml.safe_dump(
                    {"args": runner.shlex.join(command)},
                    destination,
                )
            observed = runner.completed_proposal_shard(
                output,
                command,
                expected_seed=100,
                expected_events=1,
            )
            self.assertEqual(observed["elapsed_seconds"], 4.5)
            self.assertEqual(
                observed["start"],
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_resume_rejects_incomplete_existing_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "proposal_shard_000"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.completed_proposal_shard(
                    output,
                    ["c8_air_shower"],
                    expected_seed=100,
                    expected_events=1,
                )

    def test_resume_allows_unfinished_cpu_only_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.output_root.mkdir()
            args.resume_completed_proposal = True
            runner.validate_arguments(args)
            (args.output_root / "cuda").mkdir()
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                runner.validate_arguments(args)

    def test_stability_bootstrap_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            args.stability_bootstrap_repetitions = 0
            runner.validate_arguments(args)

    def test_equal_seeds_require_explicit_paired_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            args.cuda_seed = args.proposal_seed
            with self.assertRaisesRegex(ValueError, "must differ"):
                runner.validate_arguments(args)

    def test_paired_control_requires_equal_seed_and_one_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            args.paired_seed_control = True
            with self.assertRaisesRegex(ValueError, "requires equal seeds"):
                runner.validate_arguments(args)
            args.cuda_seed = args.proposal_seed
            args.proposal_shards = 1
            args.proposal_parallelism = 1
            runner.validate_arguments(args)

    def test_too_few_stability_bootstraps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            args.stability_bootstrap_repetitions = 99
            with self.assertRaisesRegex(ValueError, "zero or at least 100"):
                runner.validate_arguments(args)

    def test_missing_additional_ensemble_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.additional_cuda = [root / "missing"]
            with self.assertRaisesRegex(
                ValueError,
                "additional CUDA output is not a directory",
            ):
                runner.validate_arguments(args)

    def test_output_provenance_records_binary_and_table_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.executable.write_bytes(b"executable-v1")
            args.table.write_bytes(b"table-v1")
            identity = runner.validation_identity(args)
            output = root / "proposal"
            output.mkdir()
            command = [str(args.executable), "--em-backend", "proposal"]
            runner.write_output_provenance(
                output,
                identity,
                "proposal",
                command,
            )
            with (
                output / runner.VALIDATION_PROVENANCE_FILENAME
            ).open("r", encoding="utf-8") as source:
                observed = json.load(source)
            self.assertEqual(observed["backend"], "proposal")
            self.assertEqual(
                observed["executable"]["sha256"],
                identity["executable"]["sha256"],
            )
            self.assertIsNone(observed["table"])
            self.assertEqual(observed["command"], command)

    def test_changed_executable_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.executable.write_bytes(b"before")
            expected = runner.artifact_identity(args.executable)
            args.executable.write_bytes(b"after")
            with self.assertRaisesRegex(
                RuntimeError,
                "changed while validation showers were running",
            ):
                runner.verify_artifact_identity(
                    expected,
                    "test executable",
                )

    def test_additional_source_from_another_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.executable.write_bytes(b"current executable")
            args.table.write_bytes(b"current table")
            identity = runner.validation_identity(args)
            additional = root / "additional"
            additional.mkdir()
            incompatible = runner.output_provenance(
                identity,
                "proposal",
                ["c8_air_shower"],
            )
            incompatible["executable"]["sha256"] = "f" * 64
            with (
                additional / runner.VALIDATION_PROVENANCE_FILENAME
            ).open("w", encoding="utf-8") as destination:
                json.dump(incompatible, destination)
            args.additional_proposal = [additional]
            with self.assertRaisesRegex(
                ValueError,
                "build/table provenance differs",
            ):
                runner.validate_additional_provenance(
                    args,
                    identity,
                )

    def test_original_proposal_executable_omits_refactor_only_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            original = root / "original_c8_air_shower"
            original.touch()
            args.proposal_executable = original
            args.primary_pdg = 11
            args.zenith_deg = 0.0
            args.azimuth_deg = 0.0
            args.ring = 0
            args.gpu_device = 0
            args.gpu_min_batch = 64
            args.gpu_table_tolerance = 1.0e-3
            command = runner.proposal_command(args, root / "proposal")
            self.assertEqual(command[0], str(original))
            self.assertNotIn("--em-backend", command)
            self.assertNotIn("--radio-backend", command)
            self.assertNotIn("--radio-sampling-rate-ghz", command)
            cuda = runner.cuda_command(args, root / "cuda")
            self.assertNotIn("--radio-sampling-rate-ghz", cuda)

            args.radio_sampling_rate_ghz = 10.0
            with self.assertRaisesRegex(
                ValueError, "fixed 1 GHz"
            ):
                runner.validate_arguments(args)

    def test_cuda_process_pool_is_non_physics_command_detail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.primary_pdg = 2212
            args.zenith_deg = 27.0
            args.azimuth_deg = 180.0
            args.ring = 0
            args.gpu_device = 0
            args.gpu_min_batch = 4096
            args.gpu_table_tolerance = 1.0e-3
            args.cuda_hadronic_backend = "fluka-process"
            command = runner.cuda_command(args, root / "cuda")
            self.assertIn("--hadronic-backend", command)
            self.assertIn("--hadronic-workers", command)
            self.assertIn("--shower-core-x", command)
            self.assertIn("--shower-core-y", command)

    def test_cuda_radio_backend_is_explicit_and_configurable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.primary_pdg = 11
            args.zenith_deg = 27.0
            args.azimuth_deg = 180.0
            args.ring = 2
            args.gpu_device = 0
            args.gpu_min_batch = 4096
            args.gpu_table_tolerance = 1.0e-3

            cpu = runner.cuda_command(args, root / "cpu-radio")
            self.assertEqual(
                cpu[cpu.index("--radio-backend") + 1],
                "cpu",
            )
            self.assertNotIn("--gpu-radio-field-limit", cpu)

            args.cuda_radio_backend = "cuda"
            args.gpu_radio_track_diagnostics = True
            args.radio_sampling_rate_ghz = 10.0
            cuda = runner.cuda_command(args, root / "cuda-radio")
            self.assertEqual(
                cuda[cuda.index("--radio-backend") + 1],
                "cuda",
            )
            self.assertEqual(
                cuda[cuda.index("--gpu-radio-field-limit") + 1],
                "1",
            )
            self.assertIn("--gpu-radio-track-diagnostics", cuda)
            common = runner.common_command(
                args, args.executable, root / "common", seed=123
            )
            self.assertEqual(
                common[
                    common.index("--radio-sampling-rate-ghz") + 1
                ],
                "10",
            )

    def test_zero_maximum_weight_preserves_automatic_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.primary_pdg = 2212
            args.zenith_deg = 27.0
            args.azimuth_deg = 180.0
            args.ring = 0
            args.maximum_weight = 0.0
            command = runner.common_command(
                args,
                args.executable,
                root / "output",
                seed=0,
            )
            self.assertNotIn("--max-weight", command)

    def test_nuclear_primary_uses_atomic_number_and_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.primary_z = 26
            args.primary_a = 56
            args.zenith_deg = 0.0
            args.azimuth_deg = 0.0
            args.ring = 0
            command = runner.common_command(
                args,
                args.executable,
                root / "iron",
                seed=123,
            )
            self.assertEqual(command[1:5], ["-Z", "26", "-A", "56"])
            self.assertNotIn("-p", command)
            runner.validate_arguments(args)

    def test_nuclear_primary_requires_valid_z_and_a(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = make_arguments(root)
            args.primary_z = 26
            args.primary_a = None
            with self.assertRaisesRegex(ValueError, "requires both"):
                runner.validate_arguments(args)
            args.primary_a = 20
            with self.assertRaisesRegex(ValueError, "Z <= A"):
                runner.validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
