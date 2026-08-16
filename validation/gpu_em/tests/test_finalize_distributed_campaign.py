from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from finalize_distributed_campaign import (  # noqa: E402
    SourceRecord,
    command_option,
    discover_sources,
    exact_seed_audit,
    file_sha256,
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
    )


class DistributedCampaignFinalizerTest(unittest.TestCase):
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
