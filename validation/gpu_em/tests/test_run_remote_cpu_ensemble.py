#!/usr/bin/env python3

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from run_remote_cpu_ensemble import resolve_seed_schedule  # noqa: E402


class RemoteCpuSeedScheduleTest(unittest.TestCase):
    def test_contiguous_schedule_uses_event_count_and_start(self) -> None:
        args = argparse.Namespace(
            events=4,
            seed_start=10300001,
            seed_list_file=None,
        )
        self.assertEqual(
            resolve_seed_schedule(args),
            (10300001, 10300002, 10300003, 10300004),
        )

    def test_explicit_seed_list_preserves_failed_seed_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed_seeds.txt"
            path.write_text(
                "# exact failed production seeds\n"
                "10300030\n\n10300040\n10300067\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                events=500,
                seed_start=1,
                seed_list_file=path,
            )
            self.assertEqual(
                resolve_seed_schedule(args),
                (10300030, 10300040, 10300067),
            )
            self.assertEqual(args.seed_list_file, path.resolve())

    def test_duplicate_or_invalid_explicit_seeds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failed_seeds.txt"
            args = argparse.Namespace(
                events=500,
                seed_start=1,
                seed_list_file=path,
            )
            path.write_text("4\n4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicates"):
                resolve_seed_schedule(args)
            path.write_text("4\nnot-a-seed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid seed"):
                resolve_seed_schedule(args)
            path.write_text("-1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "negative seed"):
                resolve_seed_schedule(args)


if __name__ == "__main__":
    unittest.main()
