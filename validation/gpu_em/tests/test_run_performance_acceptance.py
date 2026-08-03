#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import math
import argparse
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "run_performance_acceptance.py"
SPEC = importlib.util.spec_from_file_location("run_performance_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_sample(
    proposal_external: float,
    cuda_external: float,
    proposal_shower: float,
    cuda_shower: float,
) -> dict:
    return {
        "proposal": {
            "external_wall_seconds": proposal_external,
            "shower_timing": {"sum_ms": proposal_shower},
        },
        "cuda": {
            "external_wall_seconds": cuda_external,
            "shower_timing": {"sum_ms": cuda_shower},
        },
    }


class PerformanceAggregationTest(unittest.TestCase):
    def test_separate_muon_cut_is_forwarded(self) -> None:
        args = argparse.Namespace(
            executable=Path("/tmp/c8_air_shower"),
            primary_pdg=11,
            energy_gev=1.0e6,
            events=1,
            ring=0,
            seed=1,
            em_cut_gev=0.0005,
            em_thinning=1.0e-4,
            maximum_weight=100.0,
            non_em_cut_gev=1.0e13,
            mu_cut_gev=0.3,
        )
        command = MODULE.common_command(args, Path("/tmp/output"))
        self.assertEqual(
            command[command.index("--hadcut") + 1],
            "10000000000000",
        )
        self.assertEqual(
            command[command.index("--mucut") + 1],
            "0.29999999999999999",
        )
        self.assertEqual(
            command[command.index("--taucut") + 1],
            "10000000000000",
        )

    def test_cold_cache_refuses_preexisting_derived_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "rates.c8emrt"
            table.touch()
            cache = Path(str(table) + MODULE.MOLIERE_CACHE_SUFFIX)
            cache.write_bytes(b"owned")
            args = argparse.Namespace(
                table=table,
                cache_mode="cold-each",
                skip_cache_warmup=False,
            )

            with self.assertRaisesRegex(ValueError, "refusing to remove"):
                MODULE.configure_cache(args, root / "output", {})

            self.assertEqual(cache.read_bytes(), b"owned")

    def test_cold_each_removes_only_prior_generated_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "rates.c8emrt"
            table.touch()
            args = argparse.Namespace(
                table=table,
                cache_mode="cold-each",
                skip_cache_warmup=False,
            )
            cache = MODULE.configure_cache(args, root / "output", {})
            cache_path = Path(cache["cache_path"])

            first = MODULE.prepare_cuda_cache(cache, 0)
            self.assertFalse(first["present_before_run"])
            cache_path.write_bytes(b"generated-0")
            MODULE.finish_cuda_cache(cache, first)

            second = MODULE.prepare_cuda_cache(cache, 1)
            self.assertTrue(second["present_before_prepare"])
            self.assertTrue(second["removed_before_run"])
            self.assertFalse(cache_path.exists())
            cache_path.write_bytes(b"generated-1")
            MODULE.finish_cuda_cache(cache, second)

            self.assertEqual(len(cache["runs"]), 2)
            self.assertEqual(cache["runs"][1]["cache_bytes_after_run"], 11)

    def test_cmake_build_provenance_uses_nearest_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "applications" / "c8_air_shower"
            executable.parent.mkdir()
            executable.touch()
            (root / "CMakeCache.txt").write_text(
                "\n".join(
                    [
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++",
                        "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc",
                        "CMAKE_CUDA_ARCHITECTURES:STRING=89",
                    ]
                ),
                encoding="utf-8",
            )

            result = MODULE.cmake_build_provenance(executable)

            self.assertTrue(result["cmake_cache_found"])
            self.assertEqual(result["build_type"], "Release")
            self.assertEqual(result["cuda_architectures"], "89")

    def test_cmake_build_provenance_reports_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "c8_air_shower"
            executable.touch()

            result = MODULE.cmake_build_provenance(executable)

            self.assertFalse(result["cmake_cache_found"])
            self.assertIsNone(result["build_type"])

    def test_artifact_identity_detects_mid_benchmark_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "c8_air_shower"
            artifact.write_bytes(b"before")
            expected = MODULE.artifact_identity(artifact)
            artifact.write_bytes(b"after")
            with self.assertRaisesRegex(
                RuntimeError,
                "changed while performance repetitions were running",
            ):
                MODULE.verify_artifact_identity(
                    expected,
                    "test executable",
                )

    def test_ratio_uses_backend_medians(self) -> None:
        samples = [
            make_sample(10.0, 2.0, 100.0, 20.0),
            make_sample(30.0, 3.0, 300.0, 30.0),
            make_sample(20.0, 5.0, 200.0, 50.0),
        ]

        result = MODULE.aggregate_repetitions(samples)

        self.assertEqual(result["speedup"]["definition"], "ratio_of_backend_medians")
        self.assertAlmostEqual(result["speedup"]["external_wall"], 20.0 / 3.0)
        self.assertAlmostEqual(
            result["speedup"]["summed_shower_timing"], 200.0 / 30.0
        )
        self.assertEqual(
            result["proposal"]["external_wall"]["samples"], [10.0, 30.0, 20.0]
        )
        self.assertEqual(
            result["cuda"]["summed_shower_timing"]["samples"],
            [20.0, 30.0, 50.0],
        )
        self.assertEqual(
            result["speedup"]["paired_summed_shower_timing"]["samples"],
            [5.0, 10.0, 4.0],
        )
        self.assertEqual(
            result["speedup"]["paired_summed_shower_timing"]["median"], 5.0
        )

    def test_rejects_empty_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty benchmark"):
            MODULE.aggregate_repetitions([])

    def test_rejects_zero_cuda_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA timings must be positive"):
            MODULE.aggregate_repetitions(
                [make_sample(10.0, 0.0, 100.0, 20.0)]
            )

    def test_rejects_non_finite_timing(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    MODULE.sample_statistics([value], "ms")

    def test_main_writes_default_five_repetition_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "c8_air_shower"
            table = root / "rates.c8emrt"
            output = root / "benchmark"
            executable.touch()
            table.touch()

            def fake_run(command, log_path, environment):
                del command, environment
                return 100.0 if log_path.name.startswith("proposal") else 10.0

            def fake_times(output_path):
                return [1000.0] if output_path.name == "proposal" else [100.0]

            argv = [
                str(SCRIPT),
                "--executable",
                str(executable),
                "--table",
                str(table),
                "--output-root",
                str(output),
                "--skip-cache-warmup",
                "--minimum-speedup",
                "5",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                MODULE, "run_command", side_effect=fake_run
            ), mock.patch.object(
                MODULE, "read_shower_times", side_effect=fake_times
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(), 0)

            with (output / "benchmark_summary.json").open(
                "r", encoding="utf-8"
            ) as source:
                summary = json.load(source)
            self.assertEqual(summary["configuration"]["repetitions"], 5)
            self.assertEqual(len(summary["samples"]), 5)
            self.assertEqual(
                summary["samples"][0]["execution_order"],
                ["proposal", "cuda"],
            )
            self.assertEqual(
                summary["samples"][1]["execution_order"],
                ["cuda", "proposal"],
            )
            self.assertEqual(
                summary["speedup"]["summed_shower_timing"], 10.0
            )
            self.assertEqual(
                summary["aggregate"]["proposal"]["external_wall"]["count"], 5
            )
            self.assertFalse(
                summary["configuration"]["build"]["cmake_cache_found"]
            )
            self.assertEqual(
                summary["artifact_identity"]["executable"]["sha256"],
                MODULE.sha256_file(executable),
            )
            self.assertEqual(
                summary["artifact_identity"]["table"]["sha256"],
                MODULE.sha256_file(table),
            )


if __name__ == "__main__":
    unittest.main()
