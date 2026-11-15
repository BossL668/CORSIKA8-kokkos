#!/usr/bin/env python3
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
# Distributed under the 3-clause BSD license; see LICENSE.
"""Exercise HIP preflight diagnostics without requiring a GPU."""

import contextlib
import ctypes
import importlib.util
import io
from pathlib import Path
import unittest
from unittest import mock


script = Path(__file__).resolve().parents[2] / "tools/check_hip_runtime.py"
spec = importlib.util.spec_from_file_location("check_hip_runtime", script)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class HipRuntimeCheck(unittest.TestCase):
    def run_check(self, runtime=None, error=None):
        output = io.StringIO()
        with mock.patch.object(checker.ctypes, "CDLL", return_value=runtime,
                               side_effect=error), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(output):
            status = checker.main()
        return status, output.getvalue()

    @staticmethod
    def runtime(status, count):
        result = mock.Mock()

        def get_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = count
            return status

        result.hipGetDeviceCount.side_effect = get_count
        result.hipGetErrorString.return_value = b"no ROCm-capable device is detected"
        return result

    def test_missing_runtime_library(self):
        status, output = self.run_check(error=OSError("missing libamdhip64.so"))
        self.assertEqual(status, 1)
        self.assertIn("Source the selected ROCm/DTK", output)

    def test_missing_driver_or_device(self):
        status, output = self.run_check(self.runtime(100, 0))
        self.assertEqual(status, 1)
        self.assertIn("status=100, visible devices=0", output)
        self.assertIn("cannot replace a missing kernel driver", output)

    def test_success_with_no_visible_devices_is_not_accepted(self):
        status, output = self.run_check(self.runtime(0, 0))
        self.assertEqual(status, 1)
        self.assertIn("visible devices=0", output)

    def test_visible_device_does_not_claim_kernel_validation(self):
        status, output = self.run_check(self.runtime(0, 1))
        self.assertEqual(status, 0)
        self.assertIn("1 visible device(s)", output)
        self.assertIn("device discovery only", output)

    def test_wrong_library_api(self):
        status, output = self.run_check(object())
        self.assertEqual(status, 1)
        self.assertIn("does not provide the HIP runtime API", output)


if __name__ == "__main__":
    unittest.main()
