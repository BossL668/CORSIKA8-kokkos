import unittest
from run_priority_endpoints_acceptance import wait_case_resources


class CaseIsolationTests(unittest.TestCase):
    def test_cpu_work_resets_idle_even_when_gpu_empty(self):
        now, observations = [0.], iter([[], [12], [], [], []])
        records = []
        wait_case_resources(4., 20., lambda: set(), lambda: next(observations), records.append,
            clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        self.assertEqual(now[0], 8.)
        self.assertEqual(records[1]['workflow_pids'], [12])
        self.assertFalse(records[1]['safe'])

    def test_unknown_resources_timeout_without_launch(self):
        for query in (lambda: None, lambda: (_ for _ in ()).throw(OSError('unknown'))):
            now, records = [0.], []
            with self.assertRaises(TimeoutError):
                wait_case_resources(2., 4., query, lambda: [], records.append,
                    clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))
            self.assertTrue(all(not r['launches_started'] for r in records))
            self.assertTrue(all(not r['safe'] for r in records))

    def test_slow_query_cannot_run_after_deadline(self):
        now = [0.]
        def query():
            now[0] += 5.
            return set()
        with self.assertRaises(TimeoutError):
            wait_case_resources(1., 4., query, lambda: [], lambda _: None,
                                clock=lambda: now[0], sleep=lambda _: None)

    def test_unknown_workflow_is_not_idle(self):
        now = [0.]
        with self.assertRaises(TimeoutError):
            wait_case_resources(1., 4., lambda: set(), lambda: None, lambda _: None,
                clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt))


if __name__ == '__main__':
    unittest.main()
