#!/usr/bin/env python3

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIRECTORY))

from run_remote_cpu_ensemble import c8_command, resolve_seed_schedule  # noqa: E402


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


class RemoteCpuCommandTest(unittest.TestCase):
    def test_geomagnetic_configuration_is_explicit(self) -> None:
        args = argparse.Namespace(
            executable=Path("/tmp/c8_air_shower"),
            primary_pdg=2212,
            energy_gev=1.0e5,
            zenith_deg=0.0,
            azimuth_deg=0.0,
            geomagnetic_model="IGRF14",
            geomagnetic_year=2027.0,
            shower_core_x_m=0.0,
            shower_core_y_m=0.0,
            ring=0,
            antenna_file=Path("/tmp/antennas.txt"),
            em_cut_gev=0.5e-3,
            em_thinning=1.0e-6,
            had_cut_gev=0.3,
            mu_cut_gev=0.3,
            tau_cut_gev=0.3,
            maximum_weight=0.0,
        )
        command = c8_command(args, Path("/tmp/output"), 10100051)
        model_index = command.index("--geomagnetic-model")
        year_index = command.index("--geomagnetic-year")
        self.assertEqual(command[model_index + 1], "IGRF14")
        self.assertEqual(command[year_index + 1], "2027")


if __name__ == "__main__":
    unittest.main()
