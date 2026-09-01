from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

import finalize_distributed_campaign as finalizer  # noqa: E402
from finalize_distributed_campaign import (  # noqa: E402
    REQUIRED_OUTPUTS,
    SourceRecord,
    audit_source,
    append_presentation_arguments,
    command_option,
    cuda_generic_fallback_allowlist,
    discover_sources,
    exact_seed_audit,
    file_sha256,
    is_canonical_source_directory,
    read_explicit_seed_schedule,
    require_cuda_geomagnetic_configuration,
    require_maximum_weight,
    select_records_for_seed_schedule,
    validate_cuda_build_equivalence_attestation,
)


def record(seed: int, events: int, backend: str = "cuda") -> SourceRecord:
    return SourceRecord(
        backend=backend,
        root=f"/{backend}/{seed}",
        seed=seed,
        events=events,
        executable_sha256="a" * 64,
        table_sha256="b" * 64 if backend == "cuda" else None,
        antenna_sha256="c" * 64,
        observer_layout_sha256="d" * 64,
        gpu_physics_source=None,
    )


class DistributedCampaignFinalizerTest(unittest.TestCase):
    def test_audit_source_unpacks_completion_and_gates_native_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in REQUIRED_OUTPUTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("dummy\n", encoding="utf-8")
            (root / "validation_provenance.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "backend": "cuda",
                        "gpu_physics_source": "proposal-native",
                        "executable": {"sha256": "a" * 64},
                        "table": {"sha256": "b" * 64},
                        "antenna_file": {"sha256": "c" * 64},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary.yaml").write_text(
                "showers: 1\nseed: 10\n", encoding="utf-8"
            )
            command = [
                "c8", "-p", "2212", "-E", "1000", "-N", "1",
                "--seed", "10", "--zenith", "0", "--azimuth", "0",
                "--geomagnetic-model", "IGRF14", "--geomagnetic-year", "2027",
                "--emcut", "0.0005", "--emthin", "1e-6",
                "--hadcut", "0.3", "--mucut", "0.3", "--taucut", "0.3",
                "--shower-core-x", "0", "--shower-core-y", "0", "--ring", "0",
                "--em-backend", "cuda", "--radio-backend", "cuda",
            ]
            (root / "config.yaml").write_text(
                "args: " + json.dumps(" ".join(command)) + "\n",
                encoding="utf-8",
            )
            expected = argparse.Namespace(
                antenna_sha256="c" * 64,
                primary_pdg=2212,
                energy_gev=1000.0,
                zenith_deg=0.0,
                azimuth_deg=0.0,
                em_cut_gev=0.0005,
                em_thinning=1.0e-6,
                had_cut_gev=0.3,
                mu_cut_gev=0.3,
                tau_cut_gev=0.3,
                ring=0,
                maximum_weight=0.0,
                geomagnetic_model="IGRF14",
                geomagnetic_year=2027.0,
            )
            with mock.patch.object(
                finalizer,
                "validate_completion",
                return_value=((0,), {"events_checked": 1}),
            ) as completion, mock.patch.object(
                finalizer,
                "observer_layout_fingerprint",
                return_value="d" * 64,
            ):
                result = audit_source(root, "cuda", expected=expected)

            self.assertEqual(result.events, 1)
            self.assertEqual(result.gpu_physics_source, "proposal-native")
            completion.assert_called_once_with(
                root,
                True,
                expected_gpu_source="proposal-native",
                permitted_generic_fallback_reasons=(
                    "unsupported_particle",
                    "unsupported_medium",
                    "unsupported_geometry",
                ),
            )

    def test_exact_seed_audit_accepts_disjoint_contiguous_batches(self) -> None:
        exact_seed_audit(
            [record(10400006, 5), record(10400001, 5)],
            10400001,
            10,
            "CUDA",
        )

    def test_exact_seed_audit_rejects_overlap_and_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            exact_seed_audit(
                [record(10, 4), record(13, 3)],
                10,
                6,
                "CUDA",
            )

    def test_explicit_seed_schedule_selects_only_requested_records(self) -> None:
        records = [record(10, 1, "proposal"), record(11, 1, "proposal"),
                   record(20, 1, "proposal")]
        selected = select_records_for_seed_schedule(
            records, (10, 20), "proposal"
        )
        self.assertEqual([item.seed for item in selected], [10, 20])

    def test_explicit_seed_schedule_rejects_partial_multi_event_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "partially selects"):
            select_records_for_seed_schedule(
                [record(10, 3, "proposal")], (10, 11), "proposal"
            )

    def test_read_explicit_seed_schedule_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seeds.txt"
            path.write_text("# selected\n10\n\n20\n", encoding="utf-8")
            self.assertEqual(
                read_explicit_seed_schedule(path, 2, "proposal"), (10, 20)
            )
            path.write_text("10\n10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate seeds"):
                read_explicit_seed_schedule(path, 2, "proposal")
        with self.assertRaisesRegex(ValueError, "seed set is incomplete"):
            exact_seed_audit(
                [record(10, 2), record(13, 2)],
                10,
                5,
                "CUDA",
            )

    def test_discover_sources_filters_backend_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal = root / "main" / "proposal_shard_000"
            cuda = root / "local" / "batch_000" / "cuda"
            proposal.mkdir(parents=True)
            cuda.mkdir(parents=True)
            (proposal / "validation_provenance.json").write_text(
                json.dumps({"backend": "proposal"}), encoding="utf-8"
            )
            (cuda / "validation_provenance.json").write_text(
                json.dumps({"backend": "cuda"}), encoding="utf-8"
            )

            self.assertEqual(
                discover_sources([root, root / "main"], "proposal"),
                [proposal.resolve()],
            )
            self.assertEqual(
                discover_sources([root], "cuda"),
                [cuda.resolve()],
            )

    def test_discover_sources_ignores_attempts_archives_and_noncanonical_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "formal" / "batch_000" / "proposal-native"
            attempt = root / "formal" / "batch_001" / ".attempt-proposal-native-000"
            archived = (
                root / "diagnostics" / "failed_attempts" / "batch_002"
                / "proposal-native"
            )
            arbitrary = root / "scratch" / "completed-looking-output"
            for source in (canonical, attempt, archived, arbitrary):
                source.mkdir(parents=True)
                (source / "validation_provenance.json").write_text(
                    json.dumps({"backend": "cuda"}), encoding="utf-8"
                )

            self.assertEqual(discover_sources([root], "cuda"), [canonical.resolve()])
            self.assertTrue(
                is_canonical_source_directory(canonical.resolve(), root.resolve(), "cuda")
            )
            self.assertFalse(
                is_canonical_source_directory(attempt.resolve(), root.resolve(), "cuda")
            )
            self.assertFalse(
                is_canonical_source_directory(archived.resolve(), root.resolve(), "cuda")
            )
            self.assertFalse(
                is_canonical_source_directory(arbitrary.resolve(), root.resolve(), "cuda")
            )

    def test_direct_completed_output_remains_a_valid_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "proposal-native"
            root.mkdir(parents=True)
            (root / "validation_provenance.json").write_text(
                json.dumps({"backend": "cuda"}), encoding="utf-8"
            )
            self.assertEqual(discover_sources([root], "cuda"), [root.resolve()])

    def test_native_cuda_records_receive_only_the_declared_allowlist(self) -> None:
        native = SourceRecord(
            **{
                **record(10, 1).__dict__,
                "gpu_physics_source": "proposal-native",
            }
        )
        self.assertEqual(
            cuda_generic_fallback_allowlist([native]),
            (
                "unsupported_particle",
                "unsupported_medium",
                "unsupported_geometry",
            ),
        )
        self.assertEqual(cuda_generic_fallback_allowlist([record(20, 1)]), ())
        with self.assertRaisesRegex(ValueError, "mixes proposal-native"):
            cuda_generic_fallback_allowlist([native, record(20, 1)])

    def test_presentation_labels_are_propagated_only_when_requested(self) -> None:
        command = ["python", "analyze.py"]
        append_presentation_arguments(
            command, "CPU PROPOSAL", "PROPOSAL-native CUDA"
        )
        self.assertEqual(
            command,
            [
                "python", "analyze.py",
                "--reference-label", "CPU PROPOSAL",
                "--candidate-label", "PROPOSAL-native CUDA",
            ],
        )
        unchanged = ["python", "analyze.py"]
        append_presentation_arguments(unchanged, None, None)
        self.assertEqual(unchanged, ["python", "analyze.py"])

    def test_command_option_supports_separate_and_attached_values(self) -> None:
        command = ["c8", "-E", "1e8", "--emthin=1e-4"]
        self.assertEqual(command_option(command, "-E"), "1e8")
        self.assertEqual(command_option(command, "--emthin"), "1e-4")
        self.assertIsNone(command_option(command, "--seed"))

    def test_maximum_weight_accepts_explicit_or_automatic_campaigns(self) -> None:
        require_maximum_weight(["c8", "--max-weight", "100"], 100.0)
        require_maximum_weight(["c8"], 0.0)
        with self.assertRaisesRegex(ValueError, "maximum weight"):
            require_maximum_weight(["c8", "--max-weight", "100"], 10.0)
        with self.assertRaisesRegex(ValueError, "automatic maximum weight"):
            require_maximum_weight(["c8", "--max-weight", "100"], 0.0)

    def test_cuda_geomagnetic_defaults_are_read_from_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "gpu_em").mkdir()
            (root / "gpu_em/config.yaml").write_text(
                "environment:\n  geomagnetic_model: IGRF14\n  geomagnetic_year: 2027\n",
                encoding="utf-8",
            )
            require_cuda_geomagnetic_configuration(root, ["c8"], "IGRF14", 2027.0)
            with self.assertRaisesRegex(ValueError, "geomagnetic model differs"):
                require_cuda_geomagnetic_configuration(root, ["c8"], "IGRF13", 2027.0)

    def test_single_cuda_build_needs_no_equivalence_attestation(self) -> None:
        self.assertIsNone(
            validate_cuda_build_equivalence_attestation(
                None,
                {"a" * 64},
                {"b" * 64},
            )
        )

    def test_mixed_cuda_builds_require_verified_replay_evidence(self) -> None:
        builds = {"a" * 64, "c" * 64}
        table = "b" * 64
        with self.assertRaisesRegex(ValueError, "attestation is required"):
            validate_cuda_build_equivalence_attestation(None, builds, {table})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "replay_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "passed": True,
                        "reference": {
                            "executable_sha256": "a" * 64,
                            "table_sha256": table,
                        },
                        "candidate": {
                            "executable_sha256": "c" * 64,
                            "table_sha256": table,
                        },
                        "identity_checks": {"seed_equal": True},
                        "yaml_artifacts": {
                            "gpu_em/config.yaml": {"equal": True}
                        },
                        "parquet_artifacts": {
                            "profile/profile.parquet": {
                                "logical_values_equal": True
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            attestation_path = root / "attestation.json"
            attestation_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "verified",
                        "allowed_executable_sha256": sorted(builds),
                        "table_sha256": table,
                        "replay_report": {
                            "path": report_path.name,
                            "sha256": file_sha256(report_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence = validate_cuda_build_equivalence_attestation(
                attestation_path,
                builds,
                {table},
            )
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence["replay_report"], str(report_path.resolve()))

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["parquet_artifacts"]["profile/profile.parquet"][
                "logical_values_equal"
            ] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or has changed"):
                validate_cuda_build_equivalence_attestation(
                    attestation_path,
                    builds,
                    {table},
                )


if __name__ == "__main__":
    unittest.main()
