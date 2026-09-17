"""The same physics conditions do not license mixing scheduler revisions."""
import unittest
from compare_priority_hosts import priority_policy, verify_rows


def row(policy):
    return dict(family='Fe100TeV', mode='openmp-cuda', accelerator=dict(
        accelerator=dict(cooperative=dict(scheduling_policy=policy))))


class PriorityHostComparisonTest(unittest.TestCase):
    def test_uniform_revision(self):
        self.assertEqual(priority_policy([row('cpu-primary-v2-simple')]*5), 'cpu-primary-v2-simple')

    def test_mixed_revisions_rejected(self):
        with self.assertRaises(AssertionError):
            priority_policy([row('cpu-primary-v1'), row('cpu-primary-v2-simple')])

    def test_missing_revision_rejected(self):
        with self.assertRaises(AssertionError):
            priority_policy([])

    def test_placement_does_not_require_unrequested_modes(self):
        rows=[dict(seed=s,mode='openmp-cuda') for s in range(85000001,85000006)]
        rows.append(dict(seed=85000001,mode='openmp'))
        modes=['openmp','cuda-openmp','openmp-cuda']
        verify_rows(rows,modes,'openmp-cuda')
        with self.assertRaises(AssertionError):
            verify_rows(rows,modes)
        with self.assertRaises(AssertionError):
            verify_rows(rows[1:],modes,'openmp-cuda')
        with self.assertRaises(AssertionError):
            verify_rows(rows+[rows[0]],modes,'openmp-cuda')


if __name__=='__main__':
    unittest.main()
