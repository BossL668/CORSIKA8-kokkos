import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from batch_liveness import ShutdownWatchdog, atomic_json, cpu_ticks, proc_diagnostic


class LivenessTests(unittest.TestCase):
    def test_active_showers_never_trigger_shutdown_limit(self):
        w = ShutdownWatchdog(10)
        self.assertFalse(w.stalled(0, False, 1, ()))
        self.assertFalse(w.stalled(10000, False, 1, ()))

    def test_shutdown_stall(self):
        w = ShutdownWatchdog(10)
        self.assertFalse(w.stalled(0, True, 1, ()))
        self.assertFalse(w.stalled(9, True, 1, ()))
        self.assertTrue(w.stalled(10, True, 1, ()))

    def test_cpu_and_output_progress_reset_clock(self):
        w = ShutdownWatchdog(10)
        self.assertFalse(w.stalled(0, True, 1, ()))
        self.assertFalse(w.stalled(9, True, 2, ()))
        self.assertFalse(w.stalled(18, True, 2, ('summary_written',)))
        self.assertTrue(w.stalled(28, True, 2, ('summary_written',)))

    def test_unreadable_or_incomplete_resets(self):
        w = ShutdownWatchdog(10)
        w.stalled(0, True, 1, ())
        self.assertFalse(w.stalled(100, True, None, ()))
        self.assertFalse(w.stalled(200, True, 1, ()))
        self.assertFalse(w.stalled(300, False, 1, ()))
        self.assertFalse(w.stalled(400, True, 1, ()))

    def test_atomic_writes_from_concurrent_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(atomic_json, path, {'i': i, 'data': [i]*100})
                           for i in range(120)]
                for job in futures:
                    job.result()
            value = json.loads(path.read_text())
            self.assertEqual(value['data'], [value['i']]*100)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_absent_process_is_harmless(self):
        self.assertIsNone(cpu_ticks(999999999))
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(proc_diagnostic(999999999, directory)['threads'], {})


if __name__ == '__main__':
    unittest.main()
