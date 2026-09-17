"""Pure parser/accounting tests: no /proc monitoring, GPU, SSH or writes."""
import unittest

from sample_priority_thread_contention import (
    counter_deltas, parse_proc_stat, parse_schedstat, thread_role, wait_for_start)


class PriorityThreadContentionTest(unittest.TestCase):
    def test_proc_stat_handles_spaces_parentheses_and_truncation(self):
        fields = ['0'] * 50
        fields[0] = 'R'
        fields[11], fields[12], fields[19], fields[36] = '123', '45', '67890', '511'
        for name in ('c8-cuda-driver', 'name with spaces', 'name (inner)) tail'):
            record = parse_proc_stat('321 (' + name + ') ' + ' '.join(fields))
            self.assertEqual(record, dict(tid=321, comm=name, state='R', user_ticks=123,
                                         system_ticks=45, start_ticks=67890, last_cpu=511))
        with self.assertRaises(ValueError):
            parse_proc_stat('321 (truncated) R 0')
        self.assertEqual(parse_schedstat('1200 4500 7\n'),
                         dict(runtime_ns=1200, runqueue_ns=4500, sched_slices=7))
        for bad in ('1 2', '1 -1 3', 'a b c'):
            with self.assertRaises(ValueError):
                parse_schedstat(bad)

    def test_deltas_are_lifetime_safe_and_missing_counters_are_not_zero(self):
        old = dict(pid=100, process_start_ticks=20, tid=101, start_ticks=21,
                   sample_time=1., cpu_ticks=100, runtime_ns=1000,
                   runqueue_ns=2000, sched_slices=3)
        now = dict(old, sample_time=3., cpu_ticks=150, runtime_ns=2000,
                   runqueue_ns=4500, sched_slices=5)
        self.assertEqual(counter_deltas(old, now, 100), dict(counter_baseline=False,
            delta_elapsed_s=2., delta_cpu_s=.5, delta_runtime_ns=1000,
            delta_runqueue_ns=2500, delta_sched_slices=2))
        for key in ('pid', 'process_start_ticks', 'tid', 'start_ticks'):
            self.assertEqual(counter_deltas(old, dict(now, **{key: now[key] + 1}), 100),
                             {'counter_baseline': True})
        missing = dict(now)
        del missing['runqueue_ns']
        self.assertNotIn('delta_runqueue_ns', counter_deltas(old, missing, 100))
        reset = counter_deltas(old, dict(now, runtime_ns=1), 100)
        self.assertIsNone(reset['delta_runtime_ns'])
        self.assertEqual(reset['counter_regressions'], ['runtime_ns'])
        self.assertEqual(thread_role('c8-cuda-driver', 101, 100), 'cuda_driver')
        self.assertEqual(thread_role('c8_air_shower', 101, 100), 'openmp_candidate')

    def test_queued_wait_starts_only_when_exact_observer_reports_a_pid(self):
        current = [100.]
        records, sleeps = [], []
        def sleep(seconds):
            self.assertGreater(seconds, 0)
            self.assertLessEqual(seconds, 2)
            sleeps.append(seconds)
            current[0] += seconds
        def observe():
            live = current[0] >= 172.
            return {'state': 'started' if live else 'waiting',
                    'pids': [42] if live else [], 'errors': []}
        result = wait_for_start(200, observe,
            lambda row: records.append(row) is None, lambda: current[0], sleep)
        self.assertEqual(result, {'reason': 'started', 'wait_elapsed_s': 72.})
        self.assertEqual([r['wait_elapsed_s'] for r in records], [0., 30., 60., 72.])
        self.assertEqual(records[-1]['pids'], [42])
        self.assertEqual(sum(sleeps), 72.)

    def test_queued_wait_exits_complete_without_active_sampling(self):
        current = [0.]
        records = []
        def observe():
            return {'state': 'status_complete' if current[0] >= 4 else 'waiting',
                    'pids': [], 'errors': []}
        result = wait_for_start(100, observe,
            lambda row: records.append(row) is None, lambda: current[0],
            lambda seconds: current.__setitem__(0, current[0] + seconds))
        self.assertEqual(result, {'reason': 'status_complete', 'wait_elapsed_s': 4.})
        self.assertEqual([r['state'] for r in records], ['waiting', 'status_complete'])

    def test_queued_wait_timeout_retains_errors_and_obeys_absolute_bound(self):
        current = [0.]
        records = []
        error = {'path': '/proc/42/exe', 'type': 'PermissionError', 'errno': 13}
        result = wait_for_start(5,
            lambda: {'state': 'waiting', 'pids': [], 'errors': [error]},
            lambda row: records.append(row) is None, lambda: current[0],
            lambda seconds: current.__setitem__(0, current[0] + seconds))
        self.assertEqual(result, {'reason': 'wait_start_timeout', 'wait_elapsed_s': 5.})
        self.assertEqual([r['wait_elapsed_s'] for r in records], [0., 5.])
        self.assertEqual(sum(e['occurrences'] for r in records for e in r['errors']), 4)
        self.assertEqual(records[-1]['errors'][0]['errno'], 13)


if __name__ == '__main__':
    unittest.main()
