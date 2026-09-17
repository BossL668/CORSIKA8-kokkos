#!/usr/bin/env python3
"""Prevent a reused monitor from mislabelling newer pilots as adaptive-v4."""
import unittest
from report_psr_adaptive_v4 import report_settings


class SettingsTests(unittest.TestCase):
    def test_policy_and_threads_come_from_pilot(self):
        config = dict(arguments=dict(policy_version='adaptive-v10-continuation-learning',
            threads='130', expected_affinity='382-511'), affinity=list(range(382,512)))
        policy, threads, logical, affinity = report_settings(config)
        self.assertEqual(policy, 'adaptive-v10-continuation-learning')
        self.assertEqual((threads, logical, len(affinity)), (130, '382-511', 130))

    def test_missing_configuration_is_not_fabricated(self):
        self.assertEqual(report_settings({}), ('unknown-unrecorded',None,'unrecorded',[]))

    def test_inconsistent_affinity_rejected(self):
        with self.assertRaises(ValueError):
            report_settings(dict(arguments=dict(threads='130',expected_affinity='382-511'),
                affinity=list(range(130))))

    def test_invalid_thread_count_rejected(self):
        with self.assertRaises(ValueError):
            report_settings(dict(arguments=dict(threads='0')))


if __name__ == '__main__':
    unittest.main()
