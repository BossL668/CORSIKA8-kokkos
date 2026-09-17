import math
import unittest
from report_kokkos_host_call_diagnosis import estimate


class SamplingEstimates(unittest.TestCase):
    def row(self):
        return dict(calls=2,host_inclusive_s=4.,host_seconds_squared_sum=10.,
                    kind='parallel_for',execution_type='OpenMP',label='kernel',host_max_s=3.)

    def test_full_profile_is_not_rescaled(self):
        value=estimate(self.row(),1.)
        self.assertEqual(value['estimated_host_s'],4.)
        self.assertEqual(value['approximate_sampling_se_s'],0.)

    def test_sampled_counts_and_totals_not_claimed_exact(self):
        value=estimate(self.row(),.25)
        self.assertEqual(value['estimated_calls'],8.)
        self.assertEqual(value['sampled_calls'],2)
        self.assertEqual(value['estimated_host_s'],16.)
        self.assertEqual(value['sampled_mean_ms'],2000.)
        self.assertAlmostEqual(value['approximate_sampling_se_s'],math.sqrt(7.5)/.25)
        self.assertTrue(value['sparse'])

    def test_reject_invalid_probability(self):
        for value in (0.,-1.,2.,float('nan')):
            with self.assertRaises(ValueError): estimate(self.row(),value)

    def test_reject_missing_variance(self):
        row=self.row(); del row['host_seconds_squared_sum']
        with self.assertRaises(ValueError): estimate(row,.25)
        self.assertEqual(estimate(row,1.)['estimated_host_s'],4.)

    def test_reject_inconsistent_sums(self):
        row=self.row(); row['host_seconds_squared_sum']=1.
        with self.assertRaises(ValueError): estimate(row,.25)


if __name__ == '__main__': unittest.main()
