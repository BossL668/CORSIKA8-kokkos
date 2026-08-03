from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from compare_fixed_seed_cuda_outputs import (  # noqa: E402
    logical_parquet_fingerprint,
    normalized_command,
    resolve_cuda_root,
)


class FixedSeedCudaOutputComparisonTest(unittest.TestCase):
    def test_normalized_command_removes_only_executable_and_output(self) -> None:
        provenance = {
            "command": [
                "/build/a/c8_air_shower",
                "-p",
                "2212",
                "-f",
                "/output/a",
                "--seed",
                "42",
            ]
        }
        self.assertEqual(
            normalized_command(provenance),
            [
                "<executable>",
                "-p",
                "2212",
                "-f",
                "<output>",
                "--seed",
                "42",
            ],
        )

    def test_logical_fingerprint_is_value_based_and_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.parquet"
            same = root / "same.parquet"
            changed = root / "changed.parquet"
            table = pa.table(
                {
                    "shower": pa.array([0, 0, 1], type=pa.uint32()),
                    "value": pa.array([1.0, -0.0, 3.0], type=pa.float32()),
                }
            )
            pq.write_table(table, first, compression="snappy")
            pq.write_table(table, same, compression="gzip")
            pq.write_table(
                table.set_column(
                    1,
                    "value",
                    pa.array([1.0, 0.0, 3.0], type=pa.float32()),
                ),
                changed,
            )
            a = logical_parquet_fingerprint(first, batch_rows=2)
            b = logical_parquet_fingerprint(same, batch_rows=1)
            c = logical_parquet_fingerprint(changed, batch_rows=2)
            self.assertEqual(a["logical_sha256"], b["logical_sha256"])
            self.assertNotEqual(a["file_sha256"], b["file_sha256"])
            self.assertNotEqual(a["logical_sha256"], c["logical_sha256"])

    def test_resolve_cuda_root_accepts_batch_or_cuda_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "batch"
            cuda = batch / "cuda"
            cuda.mkdir(parents=True)
            (cuda / "summary.yaml").write_text("showers: 1\nseed: 1\n")
            self.assertEqual(resolve_cuda_root(batch), cuda.resolve())
            self.assertEqual(resolve_cuda_root(cuda), cuda.resolve())


if __name__ == "__main__":
    unittest.main()
