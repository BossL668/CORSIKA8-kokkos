from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from watch_fixed_seed_interaction_histograms import (  # noqa: E402
    audit_available,
    completed_showers,
    discover_expected,
    group_showers,
    histogram_identity,
)


class FixedSeedInteractionHistogramWatcherTest(unittest.TestCase):
    def histogram(self, root: Path, batch: int, frame: str, shower: int) -> Path:
        path = (
            root
            / f"batch_{batch:03d}"
            / "cuda"
            / "interaction_hist"
            / f"inthist_{frame}_{shower}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def test_identity_parses_batch_frame_and_shower(self) -> None:
        self.assertEqual(
            histogram_identity(
                Path("batch_002/cuda/interaction_hist/inthist_cms_3.npz")
            ),
            ("batch_002", 3, "cms"),
        )

    def test_available_pairs_are_audited_byte_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / "old"
            new = Path(temporary) / "new"
            old.mkdir()
            new.mkdir()
            for frame in ("lab", "cms"):
                payload = f"fixed-{frame}".encode()
                self.histogram(old, 0, frame, 1).write_bytes(payload)
                self.histogram(new, 0, frame, 1).write_bytes(payload)
            expected = discover_expected("A", new, old)
            groups = group_showers(expected)
            completed, pending = audit_available(expected)
            self.assertFalse(pending)
            self.assertEqual(
                completed_showers(groups, completed),
                ["A/batch_000/shower_1"],
            )

    def test_missing_new_file_remains_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / "old"
            new = Path(temporary) / "new"
            old.mkdir()
            new.mkdir()
            for frame in ("lab", "cms"):
                self.histogram(old, 0, frame, 1).write_bytes(frame.encode())
            expected = discover_expected("A", new, old)
            completed, pending = audit_available(expected)
            self.assertFalse(completed)
            self.assertEqual(len(pending), 2)

    def test_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / "old"
            new = Path(temporary) / "new"
            old.mkdir()
            new.mkdir()
            for frame in ("lab", "cms"):
                self.histogram(old, 0, frame, 1).write_bytes(b"old")
                self.histogram(new, 0, frame, 1).write_bytes(
                    b"new" if frame == "lab" else b"old"
                )
            expected = discover_expected("A", new, old)
            with self.assertRaisesRegex(RuntimeError, "histogram mismatch"):
                audit_available(expected)

    def test_unpaired_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            old = Path(temporary) / "old"
            new = Path(temporary) / "new"
            old.mkdir()
            new.mkdir()
            self.histogram(old, 0, "lab", 1).write_bytes(b"old")
            expected = discover_expected("A", new, old)
            with self.assertRaisesRegex(ValueError, "lacks lab/CMS pair"):
                group_showers(expected)


if __name__ == "__main__":
    unittest.main()
