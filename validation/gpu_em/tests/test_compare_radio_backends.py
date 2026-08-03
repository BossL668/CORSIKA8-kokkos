#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "compare_radio_backends.py"
SPEC = importlib.util.spec_from_file_location("compare_radio_backends", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_output(
    root: Path,
    delta: float = 0.0,
    transport_value: int = 1,
    time_delta: float = 0.0,
    radio_backend: str = "cpu",
) -> None:
    root.mkdir(parents=True)
    with (root / "summary.yaml").open("w", encoding="utf-8") as target:
        yaml.safe_dump({"showers": 1, "seed": 12345}, target)
    gpu_config = root / "gpu_em" / "config.yaml"
    gpu_config.parent.mkdir(parents=True)
    with gpu_config.open("w", encoding="utf-8") as target:
        yaml.safe_dump(
            {
                "backend": "CORSIKA8GpuEm",
                "deterministic": True,
                "radio_backend": radio_backend,
                "minimum_batch_size": 64,
            },
            target,
        )
    for algorithm in MODULE.ALGORITHMS:
        directory = root / algorithm
        directory.mkdir(parents=True)
        config = {
            "type": "RadioProcess",
            "algorithm": algorithm,
            "observers": {
                f"{algorithm}_observer": {
                    "type": "TimeDomainObserver",
                    "start time": 0.0,
                    "duration": 3.0,
                    "number of bins": 3,
                    "sampling frequency": 1.0,
                    "location": [10.0, 20.0, 30.0],
                }
            },
        }
        with (directory / "config.yaml").open("w", encoding="utf-8") as target:
            yaml.safe_dump(config, target, sort_keys=False)
        pq.write_table(
            pa.table(
                {
                    "shower": pa.array([0, 0, 0], type=pa.uint32()),
                    "Time": [0.0, 1.0, 2.0 + time_delta],
                    "Ex": [0.0, 1.0 + delta, 0.0],
                    "Ey": [0.0, 2.0 + delta, 0.0],
                    "Ez": [0.0, 3.0 + delta, 0.0],
                }
            ),
            directory / "observers.parquet",
        )
    for relative in MODULE.TRANSPORT_TABLES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"value": [transport_value]}),
            path,
        )


class RadioBackendComparisonTest(unittest.TestCase):
    def test_identical_transport_and_close_waveforms_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "cpu"
            candidate = root / "cuda"
            write_output(reference, radio_backend="cpu")
            write_output(candidate, delta=1.0e-8, radio_backend="cuda")

            report, rows = MODULE.compare_outputs(
                reference,
                candidate,
                relative_tolerance=1.0e-6,
                l2_tolerance=1.0e-6,
                fluence_tolerance=1.0e-6,
                absolute_tolerance=1.0e-7,
            )

            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["transport_identity"]["accepted"])
            self.assertEqual(report["component_comparisons"], 6)
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["accepted"] for row in rows))

    def test_transport_difference_fails_even_when_waveforms_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "cpu"
            candidate = root / "cuda"
            write_output(reference, transport_value=1, radio_backend="cpu")
            write_output(candidate, transport_value=2, radio_backend="cuda")

            report, _ = MODULE.compare_outputs(
                reference,
                candidate,
                relative_tolerance=1.0e-4,
                l2_tolerance=1.0e-4,
                fluence_tolerance=5.0e-4,
                absolute_tolerance=1.0e-18,
            )

            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["transport_identity"]["accepted"])

    def test_time_axis_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "cpu"
            candidate = root / "cuda"
            write_output(reference, radio_backend="cpu")
            write_output(candidate, time_delta=0.5, radio_backend="cuda")

            with self.assertRaisesRegex(RuntimeError, "time axes differ"):
                MODULE.compare_outputs(
                    reference,
                    candidate,
                    relative_tolerance=1.0e-4,
                    l2_tolerance=1.0e-4,
                    fluence_tolerance=5.0e-4,
                    absolute_tolerance=1.0e-18,
                )

    def test_zero_waveforms_compare_exactly(self) -> None:
        metrics = MODULE.component_metrics(
            reference=MODULE.np.zeros(4),
            candidate=MODULE.np.zeros(4),
            relative_tolerance=0.0,
            l2_tolerance=0.0,
            fluence_tolerance=0.0,
            absolute_tolerance=0.0,
        )
        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["normalized_maximum_difference"], 0.0)
        self.assertEqual(metrics["relative_l2_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
