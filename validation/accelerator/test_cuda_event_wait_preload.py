"""CPU-only linker/interposer tests; tiny fake waits, no real CUDA, GPU or shower.

Run: python -B -m unittest discover -s validation/accelerator \
       -p test_cuda_event_wait_preload.py -v
Compiler outputs live in TemporaryDirectory and are removed after the tests.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parent


class CudaEventWaitPreloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which('cc')
        if cc is None:
            raise unittest.SkipTest('C compiler unavailable')
        cls.temporary = tempfile.TemporaryDirectory(prefix='c8-fake-cudart-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        flags = [cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread']
        cls.fake = cls.root / 'libfakecudart.so'
        cls.missing = cls.root / 'missing'
        cls.missing.mkdir()
        fixture = SOURCE / 'cuda_event_wait_fake.c'
        timer = SOURCE / 'cuda_event_wait_preload.c'
        commands = [
            [*flags, '-shared', '-fPIC', '-DC8_FAKE_LIBRARY', str(fixture), '-o', str(cls.fake)],
            [*flags, '-shared', '-fPIC', '-DC8_FAKE_LIBRARY', '-DC8_FAKE_OMIT_SYNC',
             str(fixture), '-o', str(cls.missing / cls.fake.name)],
            [*flags, str(fixture), '-L'+str(cls.root), '-lfakecudart',
             '-Wl,-rpath,'+str(cls.root), '-o', str(cls.root/'driver')],
        ]
        for name, limits in [('timer', []), ('small_events', ['-DC8_EVENT_TIMER_CAPACITY=2']),
                             ('small_rows', ['-DC8_EVENT_TIMER_ROWS=2'])]:
            commands.append([*flags, '-shared', '-fPIC', *limits, str(timer), '-ldl',
                             '-o', str(cls.root/(name+'.so'))])
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)

    def run_driver(self, mode, timer='timer', missing=False):
        env = dict(os.environ)
        env.pop('LD_PRELOAD', None)
        env.pop('LD_LIBRARY_PATH', None)
        if timer:
            env['LD_PRELOAD'] = str(self.root/(timer+'.so'))
        if missing:
            env['LD_LIBRARY_PATH'] = str(self.missing)
        run = subprocess.run([str(self.root/'driver'), mode], env=env,
                             capture_output=True, text=True, timeout=5)
        parsed = [json.loads(line) for line in run.stderr.splitlines() if line.startswith('{')]
        rows = [r for r in parsed if r['kind'] == 'C8_CUDA_EVENT_TIMER']
        summary = [r for r in parsed if r['kind'] == 'C8_CUDA_EVENT_TIMER_SUMMARY']
        return run, rows, summary, parsed

    def test_lifecycle_forwarding_errno_and_last_error(self):
        baseline, _, _, _ = self.run_driver('lifecycle', timer=None)
        run, rows, summary, _ = self.run_driver('lifecycle')
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, baseline.stdout)
        self.assertEqual(sum(r['calls'] for r in rows), 14)
        sync = {(r['flags'], r['retval']): r for r in rows if r['api']=='cudaEventSynchronize'}
        self.assertEqual(set(sync), {(1,0), (0,0), (13,0), (None,400)})
        self.assertEqual(sync[(0,0)]['calls'], 2)
        for row in rows:
            self.assertTrue(row['timing_valid'])
            self.assertGreaterEqual(row['wall_ns'], 0)
            self.assertGreaterEqual(row['thread_cpu_ns'], 0)
        self.assertGreater(sync[(1,0)]['wall_ns'], 0)
        self.assertEqual(summary[0]['tracked_live_events'], 1)  # Failed destroy.
        self.assertEqual(summary[0]['event_capacity_misses'], 0)

    def test_two_real_host_tids_and_repeated_event_addresses(self):
        run, rows, summary, _ = self.run_driver('threads')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len({row['tid'] for row in rows}), 2)
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(r['calls'] for r in rows), 24)
        self.assertTrue(all(r['calls']==3 and r['retval']==0 for r in rows))
        self.assertEqual(summary[0]['tracked_live_events'], 0)
        self.assertTrue(summary[0]['coverage_complete'])

    def test_event_table_is_bounded_and_overflow_explicit(self):
        run, rows, summary, _ = self.run_driver('capacity', timer='small_events')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(summary[0]['event_capacity'], 2)
        self.assertEqual(summary[0]['event_capacity_misses'], 2)
        self.assertEqual(summary[0]['unknown_event_calls'], 4)
        self.assertFalse(summary[0]['coverage_complete'])
        self.assertEqual(sum(r['calls'] for r in rows), 12)
        self.assertEqual(summary[0]['tracked_live_events'], 0)

    def test_result_table_is_bounded_and_does_not_change_return(self):
        run, _, summary, _ = self.run_driver('lifecycle', timer='small_rows')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(summary[0]['row_capacity'], 2)
        self.assertGreater(summary[0]['row_capacity_misses'], 0)
        self.assertFalse(summary[0]['coverage_complete'])

    def test_missing_real_symbol_exits_127_without_fallback(self):
        run, _, _, parsed = self.run_driver('lifecycle', missing=True)
        self.assertEqual(run.returncode, 127, run.stderr)
        self.assertEqual(parsed, [{'kind':'C8_CUDA_EVENT_TIMER_ERROR',
                                  'missing_symbol':'cudaEventSynchronize','exit_code':127}])
        self.assertNotIn('FAKE_DRIVER', run.stdout)

    def test_no_observed_cuda_calls_cannot_claim_coverage(self):
        run, rows, summary, _ = self.run_driver('quiet')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(rows, [])
        self.assertEqual(summary[0]['observed_calls'], 0)
        self.assertFalse(summary[0]['coverage_complete'])


if __name__ == '__main__':
    unittest.main()
