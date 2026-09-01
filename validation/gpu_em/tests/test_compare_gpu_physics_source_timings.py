#!/usr/bin/env python3
"""Tests for the independent C8EMRT/native timing comparison."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "compare_gpu_physics_source_timings.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_gpu_physics_source_timings", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def make_shard(
    root: Path,
    source: str,
    times_ms: list[float],
    *,
    explicit_source: bool = True,
    memory_fraction: float = 0.9,
) -> None:
    table = (
        {
            "path": "/cache/table.c8emrt",
            "format_version": 10,
            "sha256": "a" * 64,
        }
        if source == "c8emrt"
        else {
            "format": "PROPOSAL native Hermite",
            "proposal_version": "7.6.2",
        }
    )
    config = {
        "backend": "CORSIKA8GpuEm",
        "backend_version": "test-backend",
        "device": {
            "index": 0,
            "name": "Test GPU",
            "compute_capability": 8.9,
            "total_memory_bytes": 8_000_000_000,
        },
        "cuda": {"driver_version": 12.6, "runtime_version": 12.6},
        "deterministic": True,
        "detailed_stage_timing": False,
        "radio_backend": "cuda",
        "radio_fixed_point_field_limit_V_per_m": 1.0,
        "radio_track_diagnostics": False,
        "minimum_batch_size": 4096,
        "memory_fraction": memory_fraction,
        "accepted_table_tolerance": 5.0e-4,
        "environment": {
            "geomagnetic_model": "IGRF14",
            "geomagnetic_year": 2027.0,
            "latitude_deg": 42.5,
            "longitude_deg": 86.4,
            "altitude_m": 2680.0,
            "magnetic_field_T": {"x": 1.0e-5, "y": 2.0e-6, "z": -5.0e-5},
            "maximum_magnetic_deflection_rad": 0.2,
            "observation_geometry": "plane",
            "observation_plane_point_m": {"x": 0.0, "y": 0.0, "z": 1.0},
            "observation_plane_normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        },
        "table": table,
    }
    command = [
        "/build/c8_air_shower",
        "-p", "2212",
        "-E", "1000",
        "-N", str(len(times_ms)),
        "-f", str(root),
        "--seed", "12345",
        "--zenith", "0",
        "--azimuth", "0",
        "--geomagnetic-model", "IGRF14",
        "--geomagnetic-year", "2027",
        "--shower-core-x", "0",
        "--shower-core-y", "0",
        "--ring", "0",
        "--antenna-file", "/data/antennas.txt",
        "--radio-sampling-rate-ghz", "1",
        "--radio-window-duration-ns", "400",
        "--radio-pretrigger-ns", "10",
        "--emcut", "0.0005",
        "--emthin", "1e-6",
        "--hadcut", "0.3",
        "--mucut", "0.3",
        "--taucut", "0.3",
        "--verbosity", "warn",
        "--em-backend", "cuda",
        "--radio-backend", "cuda",
        "--gpu-device", "0",
        "--gpu-min-batch", "4096",
        "--gpu-memory-fraction", str(memory_fraction),
        "--gpu-table-tolerance", "0.0005",
        "--gpu-deterministic", "true",
        "--gpu-resident-cross-species", "true",
        "--gpu-radio-field-limit", "1",
        "--hadronic-backend", "fluka-process",
        "--hadronic-workers", "4",
        "--hadronic-min-batch", "64",
        "--hadronic-target-batch-ms", "5",
        "--hadronic-max-batch", "256",
    ]
    provenance = {
        "backend": "cuda",
        "table": {"path": "/cache/table.c8emrt", "sha256": "b" * 64},
        "antenna_file": {"path": "/data/antennas.txt", "sha256": "c" * 64},
        "flupro": {"path": "/opt/fluka/libflukahp.a", "sha256": "d" * 64},
        "events": len(times_ms),
        "command": command,
    }
    if explicit_source:
        config["gpu_physics_source"] = source
        provenance["gpu_physics_source"] = source
        provenance["command"].extend(("--gpu-physics-source", source))
    elif source == "c8emrt":
        pass
    if source == "c8emrt":
        provenance["command"].extend(("--gpu-table-cache", "/cache/table.c8emrt"))
    else:
        provenance["command"].extend(("--gpu-aux-cache-dir", "/cache/aux"))
    encoded_command = json.dumps(
        provenance["command"], ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    provenance["command_sha256"] = hashlib.sha256(encoded_command).hexdigest()
    write_yaml(root / "gpu_em" / "config.yaml", config)
    write_yaml(
        root / "config.yaml",
        {
            "path": str(root),
            "creator": "CORSIKA8",
            "version": "test",
            "args": shlex.join(provenance["command"]),
        },
    )
    (root / "validation_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    write_yaml(
        root / "simulation_timing" / "summary.yaml",
        {
            f"shower_{index}": {
                "closed": True,
                "status": "closed",
                "wall_time_ms": value,
            }
            for index, value in enumerate(times_ms)
        },
    )


def rewrite_recorded_command(root: Path, command: list[str]) -> None:
    provenance_path = root / "validation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["command"] = command
    encoded = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    provenance["command_sha256"] = hashlib.sha256(encoded).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    output_config_path = root / "config.yaml"
    output_config = yaml.safe_load(output_config_path.read_text(encoding="utf-8"))
    output_config["args"] = shlex.join(command)
    write_yaml(output_config_path, output_config)


class TimingComparisonTests(unittest.TestCase):
    def test_legacy_c8emrt_requires_opt_in_and_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "c8"
            make_shard(root, "c8emrt", [1000.0, 1100.0], explicit_source=False)
            with self.assertRaisesRegex(ValueError, "requires explicit opt-in"):
                MODULE.validate_source_identity(
                    root, "c8emrt", allow_legacy_c8emrt=False
                )
            identity = MODULE.validate_source_identity(
                root, "c8emrt", allow_legacy_c8emrt=True
            )
            self.assertEqual(identity["identity_method"], "legacy-c8emrt-inferred")

            provenance_path = root / "validation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["command"] = ["c8_air_shower"]
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence is contradictory"):
                MODULE.validate_source_identity(
                    root, "c8emrt", allow_legacy_c8emrt=True
                )

    def test_timing_records_must_be_closed_and_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.yaml"
            write_yaml(
                path,
                {
                    "shower_0": {
                        "closed": True,
                        "status": "closed",
                        "wall_time_ms": 1000.0,
                    },
                    "shower_2": {
                        "closed": True,
                        "status": "closed",
                        "wall_time_ms": 1200.0,
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                MODULE.extract_timing_records(path)

    def test_no_legacy_opt_in_fails_before_malformed_command_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            make_shard(root, "c8emrt", [1000.0, 900.0], explicit_source=False)
            provenance_path = root / "validation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["command"] = ["broken"]
            provenance["command_sha256"] = "0" * 64
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires explicit opt-in"):
                MODULE.validate_source_identity(
                    root, "c8emrt", allow_legacy_c8emrt=False
                )

    def test_command_hash_and_output_manager_command_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "native"
            make_shard(root, "proposal-native", [1000.0, 900.0])
            provenance_path = root / "validation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["command_sha256"] = "0" * 64
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "command SHA-256 differs"):
                MODULE.validate_source_identity(
                    root, "proposal-native", allow_legacy_c8emrt=False
                )

            make_shard(root, "proposal-native", [1000.0, 900.0])
            output_path = root / "config.yaml"
            output_config = yaml.safe_load(output_path.read_text(encoding="utf-8"))
            output_command = shlex.split(output_config["args"])
            emthin_index = output_command.index("--emthin") + 1
            output_command[emthin_index] = "1e-5"
            output_config["args"] = shlex.join(output_command)
            write_yaml(output_path, output_config)
            with self.assertRaisesRegex(ValueError, "normalized commands differ"):
                MODULE.validate_source_identity(
                    root, "proposal-native", allow_legacy_c8emrt=False
                )

    def test_command_and_gpu_config_contradiction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "native"
            make_shard(root, "proposal-native", [1000.0, 900.0])
            config_path = root / "gpu_em" / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["memory_fraction"] = 0.7
            write_yaml(config_path, config)
            with self.assertRaisesRegex(
                ValueError, "command/config contradiction.*gpu_memory_fraction"
            ):
                MODULE.validate_source_identity(
                    root, "proposal-native", allow_legacy_c8emrt=False
                )

    def test_max_weight_presence_and_antenna_hash_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            c8 = base / "c8"
            native = base / "native"
            make_shard(c8, "c8emrt", [1000.0, 900.0])
            make_shard(native, "proposal-native", [1000.0, 900.0])
            provenance = json.loads(
                (native / "validation_provenance.json").read_text(encoding="utf-8")
            )
            rewrite_recorded_command(
                native, [*provenance["command"], "--max-weight", "100"]
            )
            native_provenance = json.loads(
                (native / "validation_provenance.json").read_text(encoding="utf-8")
            )
            native_provenance["antenna_file"]["sha256"] = "e" * 64
            (native / "validation_provenance.json").write_text(
                json.dumps(native_provenance), encoding="utf-8"
            )
            # Recompute because provenance itself changed but the command did not.
            _, c8_shards = MODULE.load_source(
                [c8], "c8emrt", allow_legacy_c8emrt=False
            )
            _, native_shards = MODULE.load_source(
                [native], "proposal-native", allow_legacy_c8emrt=False
            )
            fields = {
                mismatch["field"]
                for mismatch in MODULE.signature_mismatches(
                    [*c8_shards, *native_shards]
                )
            }
            self.assertTrue(any("maximum_weight" in field for field in fields))
            self.assertIn(
                "immutable_run_signature.antenna_sha256",
                fields,
            )

    def test_ratio_definition_and_independent_bootstrap(self) -> None:
        c8 = np.asarray([2.0, 4.0, 6.0])
        native = np.asarray([1.0, 2.0, 3.0])
        result = MODULE.ratio_summary(c8, native)
        self.assertEqual(result["ratio_of_means"], 0.5)
        self.assertEqual(result["ratio_of_medians"], 0.5)
        self.assertEqual(result["median_regression_percent"], -50.0)
        self.assertEqual(result["median_c8emrt_over_native_speed_factor"], 2.0)
        intervals = MODULE.bootstrap_ratio_intervals(c8, native, 50, 123)
        self.assertEqual(len(intervals["independent_mean_ratio_95pct"]), 2)
        self.assertEqual(len(intervals["independent_median_ratio_95pct"]), 2)

    def test_old_and_new_campaign_manifests_require_exact_shard_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            c8 = base / "c8"
            native = base / "native"
            make_shard(c8, "c8emrt", [1000.0, 900.0])
            make_shard(native, "proposal-native", [1100.0, 950.0])
            _, c8_shards = MODULE.load_source(
                [c8], "c8emrt", allow_legacy_c8emrt=False
            )
            _, native_shards = MODULE.load_source(
                [native], "proposal-native", allow_legacy_c8emrt=False
            )
            old_manifest = base / "old.json"
            old_manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "batches": [
                            {
                                "status": "complete",
                                "cuda_output": str(c8),
                                "events": 2,
                                "runtime_seconds": 4.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            native_command = json.loads(
                (native / "validation_provenance.json").read_text(encoding="utf-8")
            )["command"]
            new_manifest = base / "new.json"
            new_manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "attempts": [
                            {
                                "status": "complete",
                                "task": "formal:000:proposal-native",
                                "output": str(native),
                                "runtime_seconds": 5.0,
                                "command": native_command,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            old = MODULE.load_campaign_batch_timings(
                [old_manifest], "c8emrt", c8_shards
            )
            new = MODULE.load_campaign_batch_timings(
                [new_manifest], "proposal-native", native_shards
            )
            self.assertEqual(old["manifests"][0]["schema"], "legacy-batches")
            self.assertEqual(new["manifests"][0]["schema"], "attempts")
            self.assertEqual(old["batches"][0]["per_event_turnaround_seconds"], 2.0)
            self.assertEqual(new["batches"][0]["per_event_turnaround_seconds"], 2.5)

            old_payload = json.loads(old_manifest.read_text(encoding="utf-8"))
            old_payload["batches"][0]["cuda_output"] = str(base / "wrong")
            old_manifest.write_text(json.dumps(old_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not exactly cover"):
                MODULE.load_campaign_batch_timings(
                    [old_manifest], "c8emrt", c8_shards
                )

    def test_manifest_prefix_map_and_multiple_manifests_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            new_prefix = base / "server-data"
            first = new_prefix / "campaign-a" / "batch_000" / "cuda"
            second = new_prefix / "campaign-b" / "batch_000" / "cuda"
            make_shard(first, "c8emrt", [1000.0, 900.0])
            make_shard(second, "c8emrt", [1100.0, 950.0])
            _, shards = MODULE.load_source(
                [first, second], "c8emrt", allow_legacy_c8emrt=False
            )
            old_prefix = Path("/archive/CorsikaData")
            manifests: list[Path] = []
            for name, relative, runtime in (
                ("first.json", "campaign-a/batch_000/cuda", 4.0),
                ("second.json", "campaign-b/batch_000/cuda", 5.0),
            ):
                path = base / name
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "batches": [
                                {
                                    "status": "complete",
                                    "cuda_output": str(old_prefix / relative),
                                    "events": 2,
                                    "runtime_seconds": runtime,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(path)
            mapping = MODULE.parse_prefix_map(f"{old_prefix}={new_prefix}")
            result = MODULE.load_campaign_batch_timings(
                manifests, "c8emrt", shards, path_map=mapping
            )
            self.assertEqual(len(result["manifests"]), 2)
            self.assertEqual(len(result["batches"]), 2)
            self.assertEqual(result["path_map"], mapping)

            with self.assertRaisesRegex(ValueError, "duplicate campaign manifest"):
                MODULE.load_campaign_batch_timings(
                    [manifests[0], manifests[0]],
                    "c8emrt",
                    shards,
                    path_map=mapping,
                )

            outside = json.loads(manifests[1].read_text(encoding="utf-8"))
            outside["batches"][0]["cuda_output"] = "/outside/batch_000/cuda"
            manifests[1].write_text(json.dumps(outside), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside OLD path-map prefix"):
                MODULE.load_campaign_batch_timings(
                    manifests, "c8emrt", shards, path_map=mapping
                )

    def test_main_combines_shards_and_excludes_each_cold_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            c8_a = base / "c8-a"
            c8_b = base / "c8-b"
            native_a = base / "native-a"
            native_b = base / "native-b"
            make_shard(c8_a, "c8emrt", [10_000.0, 2_000.0, 4_000.0])
            make_shard(c8_b, "c8emrt", [11_000.0, 6_000.0, 8_000.0])
            make_shard(native_a, "proposal-native", [12_000.0, 1_000.0, 2_000.0])
            make_shard(native_b, "proposal-native", [13_000.0, 3_000.0, 4_000.0])
            output = base / "result"
            argv = [
                str(MODULE_PATH),
                "--c8emrt", str(c8_a),
                "--c8emrt", str(c8_b),
                "--proposal-native", str(native_a),
                "--proposal-native", str(native_b),
                "--output", str(output),
                "--expected-c8emrt-events", "6",
                "--expected-proposal-native-events", "6",
                "--bootstrap-repetitions", "20",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)
            result = json.loads(
                (output / "timing_comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                result["statistics"]["all_events"]["c8emrt"]["count"], 6
            )
            self.assertEqual(
                result["statistics"]["steady_state"]["c8emrt"]["count"], 4
            )
            self.assertAlmostEqual(
                result["comparisons"]["steady_state"]["ratio_of_medians"],
                0.5,
            )
            self.assertTrue(
                (output / "gpu_physics_source_runtime_histograms.png").is_file()
            )

    def test_performance_configuration_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            make_shard(first, "c8emrt", [1000.0, 900.0])
            make_shard(
                second,
                "proposal-native",
                [1000.0, 900.0],
                memory_fraction=0.7,
            )
            _, first_shards = MODULE.load_source(
                [first], "c8emrt", allow_legacy_c8emrt=False
            )
            _, second_shards = MODULE.load_source(
                [second], "proposal-native", allow_legacy_c8emrt=False
            )
            mismatches = MODULE.signature_mismatches(
                [*first_shards, *second_shards]
            )
            self.assertIn("memory_fraction", mismatches[0]["field"])


if __name__ == "__main__":
    unittest.main()
