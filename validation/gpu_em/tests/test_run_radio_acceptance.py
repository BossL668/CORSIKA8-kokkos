#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_radio_acceptance.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_radio_acceptance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_arguments(root: Path) -> argparse.Namespace:
    executable = root / "c8_air_shower"
    table = root / "rates.c8emrt"
    executable.touch()
    table.touch()
    return argparse.Namespace(
        executable=executable,
        table=table,
        output_root=root / "output",
        energy_gev=1000.0,
        events=2,
        seed=12345,
        primary_pdg=11,
        zenith_deg=45.0,
        azimuth_deg=90.0,
        ring=1,
        em_cut_gev=0.0005,
        em_thinning=1.0e-4,
        maximum_weight=100.0,
        non_em_cut_gev=0.3,
        gpu_device=0,
        gpu_min_batch=64,
        gpu_memory_fraction=0.7,
        gpu_table_tolerance=1.0e-3,
        gpu_radio_field_limit=1.0,
        relative_tolerance=1.0e-4,
        l2_tolerance=1.0e-4,
        fluence_tolerance=5.0e-4,
        absolute_tolerance_v_per_m=1.0e-18,
        skip_cache_check=True,
        require_pass=True,
    )


class RadioAcceptanceRunnerTest(unittest.TestCase):
    def test_commands_differ_only_in_radio_backend_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            MODULE.validate_arguments(args)
            cpu = MODULE.radio_command(args, Path("/tmp/cpu"), "cpu")
            cuda = MODULE.radio_command(args, Path("/tmp/cuda"), "cuda")

            self.assertEqual(cpu[cpu.index("--seed") + 1], "12345")
            self.assertEqual(cuda[cuda.index("--seed") + 1], "12345")
            self.assertEqual(cpu[cpu.index("--em-backend") + 1], "cuda")
            self.assertEqual(cuda[cuda.index("--em-backend") + 1], "cuda")
            self.assertEqual(cpu[cpu.index("--radio-backend") + 1], "cpu")
            self.assertEqual(cuda[cuda.index("--radio-backend") + 1], "cuda")
            self.assertNotIn("--gpu-radio-field-limit", cpu)
            self.assertIn("--gpu-radio-field-limit", cuda)

    def test_zero_ring_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = make_arguments(Path(temporary))
            args.ring = 0
            with self.assertRaisesRegex(ValueError, "non-zero observer ring"):
                MODULE.validate_arguments(args)

    def test_non_finite_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for value in (math.nan, math.inf, -math.inf):
                args = make_arguments(Path(temporary))
                args.relative_tolerance = value
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        MODULE.validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
