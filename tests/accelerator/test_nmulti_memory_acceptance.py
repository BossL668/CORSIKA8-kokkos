"""Regression tests for the diagnostic's lifecycle/hash acceptance criteria."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "validation/accelerator/run_nmulti_memory_acceptance.py"
spec = importlib.util.spec_from_file_location("nmulti_memory", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NMultiAcceptanceTests(unittest.TestCase):
    def fixture(self, root, change=None):
        summary = {}
        for i in range(3):
            stats = {"accelerator": {"backend": "cuda"},
                     "backend_lifecycle": {"reused": i > 0, "shower_ordinal": i + 1,
                                           "static_host_to_device_bytes": 100},
                     "gpu_particles": i + 1, "table_device_bytes": 100,
                     "workspace_bytes": 32, "peak_device_bytes": 132,
                     "radio": {"device_bytes": 16},
                     "proposal_native": {"proposal_version": "7.6.2", "cubic_interpolation_version": "0.1.5",
                                         "table_sha256": "same", "node_count": 4, "device_bytes": 100,
                                         "aux_sha256": "aux", "newton_iterations": i * 17}}
            summary[f"shower_{i}"] = {"complete": True, "statistics": stats}
        if change:
            change(summary)
        (root / "gpu_em").mkdir()
        (root / "gpu_em/summary.yaml").write_text(yaml.safe_dump(summary))
        for name in ("CoREAS", "ZHS"):
            (root / name).mkdir()
            pq.write_table(pa.table({"shower": [0, 1, 2], "Ex": [1., 2., 3.],
                                     "Ey": [0., 0., 0.], "Ez": [0., 0., 0.]}),
                           root / name / "observers.parquet")

    def test_inverse_iteration_counts_are_not_part_of_immutable_table_hash(self):
        with tempfile.TemporaryDirectory(prefix="c8-nmulti-test-") as folder:
            root = Path(folder)
            self.fixture(root)
            result = module.inspect_output(root, "cuda", 3)
            self.assertEqual(len(result["showers"]), 3)
            self.assertEqual(result["radio"]["CoREAS"]["nonzero_showers"], 3)

    def test_changed_table_hash_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c8-nmulti-test-") as folder:
            root = Path(folder)
            def change(summary):
                summary["shower_1"]["statistics"]["proposal_native"]["table_sha256"] = "different"
            self.fixture(root, change)
            with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                module.inspect_output(root, "cuda", 3)

    def test_missing_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c8-nmulti-test-") as folder:
            root = Path(folder)
            def change(summary):
                summary["shower_1"]["statistics"]["backend_lifecycle"]["reused"] = False
            self.fixture(root, change)
            with self.assertRaisesRegex(RuntimeError, "lifecycle"):
                module.inspect_output(root, "cuda", 3)

    def test_incomplete_event_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c8-nmulti-test-") as folder:
            root = Path(folder)
            def change(summary):
                summary["shower_2"]["complete"] = False
            self.fixture(root, change)
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                module.inspect_output(root, "cuda", 3)


if __name__ == "__main__":
    unittest.main()
