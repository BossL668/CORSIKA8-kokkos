"""Small command-only tests. No simulator, GPU, service or production writes."""
import unittest

from run_priority_endpoints_acceptance import (
    override_fe_energy, override_timing_primary, timing_family)


class TimingPrimaryTest(unittest.TestCase):
    def test_legacy_template_and_labels_unchanged(self):
        cmd = ['c8', '-Z', '26', '-A', '56', '-E', '100000']
        before = list(cmd)
        override_timing_primary(cmd, None)
        self.assertEqual(cmd, before)
        self.assertEqual(timing_family(), 'Fe100TeV')
        self.assertEqual(timing_family(10000), 'Fe10TeV')

    def test_proton_removes_nuclear_selectors_only(self):
        cmd = ['c8', '-Z', '26', '-A', '56', '-E', '100000',
               '--emthin', '1e-6', '--max-weight', '0', '-s', '85000001']
        override_timing_primary(cmd, 'proton')
        self.assertEqual(cmd, ['c8', '-E', '100000', '--emthin', '1e-6',
                              '--max-weight', '0', '-s', '85000001', '-p', '2212'])
        self.assertEqual(timing_family(None, 'proton'), 'proton100TeV')

    def test_existing_pid_replaced_and_override_idempotent(self):
        cmd = ['c8', '-p', '22', '-E', '1000']
        override_timing_primary(cmd, 'proton')
        once = list(cmd)
        override_timing_primary(cmd, 'proton')
        self.assertEqual(cmd, once)
        self.assertEqual(cmd.count('-p'), 1)
        self.assertEqual(cmd[-1], '2212')

    def test_energy_override_keeps_other_physics(self):
        cmd = ['c8', '-Z', '26', '-A', '56', '-E', '10000', '--max-weight', '0']
        override_fe_energy(cmd, 100000)
        override_timing_primary(cmd, 'proton')
        self.assertEqual(cmd, ['c8', '-E', '100000.0', '--max-weight', '0', '-p', '2212'])
        self.assertEqual(timing_family(100000, 'proton'), 'proton100TeV')

    def test_invalid_override_fails_without_mutation(self):
        for cmd, primary in ((['c8', '-E'], 'proton'), ([], 'proton'),
                             (['c8', '-E', '1000'], 'unknown')):
            before = list(cmd)
            with self.assertRaises(ValueError):
                override_timing_primary(cmd, primary)
            self.assertEqual(cmd, before)


if __name__ == '__main__':
    unittest.main()
