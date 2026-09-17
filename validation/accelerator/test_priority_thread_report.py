import unittest

from report_priority_thread_contention import driver_activity, nonnegative


class ThreadReportTest(unittest.TestCase):
    def test_second_active_driver_not_hidden_by_first_idle_driver(self):
        idle = dict(role='cuda_driver', delta_elapsed_s=1., delta_cpu_s=0.)
        active = dict(role='cuda_driver', delta_elapsed_s=1., delta_cpu_s=.4)
        self.assertFalse(driver_activity([idle]))
        self.assertTrue(driver_activity([idle, active]))
        self.assertTrue(driver_activity([active, idle]))

    def test_counter_baseline_and_regression_not_counted_as_work(self):
        self.assertFalse(driver_activity([dict(role='cuda_driver', cpu_s=100.)]))
        self.assertEqual(nonnegative(dict(delta_cpu_s=None), 'delta_cpu_s'), 0.)
        self.assertEqual(nonnegative(dict(delta_cpu_s=-1.), 'delta_cpu_s'), 0.)


if __name__ == '__main__':
    unittest.main()
