#!/usr/bin/env python3
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from run_kokkos_host_call_diagnosis import validate_guard, validate_comparison_tool


class GuardCompatibility(unittest.TestCase):
    @patch('run_kokkos_host_call_diagnosis.subprocess.check_output')
    def test_legacy_guard_rejected_before_launch(self, output):
        output.return_value = '--rss-limit-gib --sample-threads'
        with self.assertRaisesRegex(RuntimeError, 'require-exclusive-gpu'):
            validate_guard(Path('/unused/old-guard.py'))

    @patch('run_kokkos_host_call_diagnosis.subprocess.check_output')
    def test_required_protection_supported(self, output):
        output.return_value = '--rss-limit-gib --sample-threads --require-exclusive-gpu'
        validate_guard(Path('/unused/guard.py'))
        self.assertEqual(output.call_args[0][0][-1], '--help')

    @patch('run_kokkos_host_call_diagnosis.subprocess.check_output')
    def test_broken_help_is_not_accepted(self, output):
        output.side_effect = subprocess.CalledProcessError(1, ['guard','--help'])
        with self.assertRaises(subprocess.CalledProcessError):
            validate_guard(Path('/unused/guard.py'))

    @patch('run_kokkos_host_call_diagnosis.subprocess.check_output')
    def test_comparison_imports_checked_before_launch(self, output):
        output.return_value = 'usage: compare reference candidate --report PATH'
        validate_comparison_tool(Path('/unused/tools'))
        self.assertEqual(output.call_args[0][0][1:],
                         ['/unused/tools/compare_backend_build_outputs.py', '--help'])

    @patch('run_kokkos_host_call_diagnosis.subprocess.check_output')
    def test_missing_comparison_dependency_rejects_launch(self, output):
        output.side_effect = subprocess.CalledProcessError(1, ['compare', '--help'])
        with self.assertRaises(subprocess.CalledProcessError):
            validate_comparison_tool(Path('/unused/tools'))


if __name__ == '__main__':
    unittest.main()
