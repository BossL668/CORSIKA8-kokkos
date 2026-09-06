import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/run_memory_guarded.py'
spec = importlib.util.spec_from_file_location('memory_guard', SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class MemoryGuardTests(unittest.TestCase):
    def test_limits_cover_host_runner_child_and_whole_tree(self):
        limits = SimpleNamespace(host_reserve_mib=3072, max_runner_rss_mib=1024,
                                 max_child_rss_mib=3072, max_tree_rss_mib=4096)
        base = dict(host_available_mib=8000, runner_rss_mib=100,
                    largest_child_rss_mib=800, tree_rss_mib=900)
        self.assertIsNone(guard.reason(base, limits))
        for field, value, expected in (
                ('host_available_mib', 3000, 'host_available_below_reserve'),
                ('runner_rss_mib', 1025, 'runner_rss_mib_over_limit'),
                ('largest_child_rss_mib', 3073, 'largest_child_rss_mib_over_limit'),
                ('tree_rss_mib', 4097, 'tree_rss_mib_over_limit')):
            self.assertEqual(guard.reason(dict(base, **{field: value}), limits), expected)

    def run_guard(self, folder, code, extra=()):
        return subprocess.run([sys.executable, str(SCRIPT), '--output', str(folder),
                               '--interval', '.02', '--host-reserve-mib', '1',
                               *extra, '--', sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=15)

    def test_normal_completion(self):
        with tempfile.TemporaryDirectory(prefix='c8-guard-test-') as folder:
            result = self.run_guard(folder, 'import time; time.sleep(.1)')
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((Path(folder) / 'memory_guard_status.json').read_text())
            self.assertEqual(status['status'], 'runner_finished')
            self.assertFalse((Path(folder) / 'MEMORY_GUARD_STOP.json').exists())

    def test_child_in_separate_session_is_stopped_and_resume_is_blocked(self):
        # Allocate only 64 MiB: safely exercise an intentionally tiny 32 MiB guard.
        child = "import time; x=bytearray(64*1024*1024); time.sleep(60)"
        code = ('import subprocess,sys,time; '
                f'p=subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True); '
                'time.sleep(60)')
        with tempfile.TemporaryDirectory(prefix='c8-guard-test-') as folder:
            result = self.run_guard(folder, code, ('--max-child-rss-mib', '32'))
            self.assertEqual(result.returncode, 75, result.stderr)
            marker = json.loads((Path(folder) / 'MEMORY_GUARD_STOP.json').read_text())
            self.assertEqual(marker['reason'], 'largest_child_rss_mib_over_limit')
            for info in marker['last_sample']['processes']:
                current = guard.process_info(info['pid'])
                self.assertTrue(current is None or current['state'] == 'Z' or
                                current['start_ticks'] != info['start_ticks'])
            retry = self.run_guard(folder, 'raise SystemExit(0)')
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn('memory incident marker exists', retry.stderr)

    def test_long_lived_runner_growth_without_any_child_is_stopped(self):
        # Stand-in for a queue/validator retaining memory between events.
        code = ('import time\nretained=[]\n'
                'for i in range(30):\n'
                ' retained.append(bytearray(2*1024*1024))\n time.sleep(.03)\n')
        with tempfile.TemporaryDirectory(prefix='c8-guard-parent-test-') as folder:
            result = self.run_guard(folder, code, ('--max-runner-rss-mib', '32'))
            self.assertEqual(result.returncode, 75, result.stderr)
            marker = json.loads((Path(folder) / 'MEMORY_GUARD_STOP.json').read_text())
            self.assertEqual(marker['reason'], 'runner_rss_mib_over_limit')
            self.assertEqual(marker['last_sample']['largest_child_rss_mib'], 0)


if __name__ == '__main__':
    unittest.main()
