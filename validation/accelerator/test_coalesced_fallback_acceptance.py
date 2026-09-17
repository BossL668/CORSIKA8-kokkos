"""Negative tests for the runtime gate, not a substitute for shower tests."""
import copy
import unittest
from run_adaptive_cooperative_acceptance import check_coalesced_fallbacks


class CoalescedFallbackAccounting(unittest.TestCase):
    def setUp(self):
        self.stats = dict(
            accelerator=dict(cooperative=dict(specified_fallback_batch_limit=4096)),
            deferred_cpu_fallbacks_queued=8193, deferred_cpu_fallbacks_flushed=8193,
            maximum_deferred_cpu_fallback_batch=4096,
            deferred_fallback_scalar_expansion_rounds=0, cpu_generic_fallbacks=1,
            cpu_fallbacks_by_reason_name=dict(native_selection_replay=8193, unsupported_geometry=1),
            cpu_fallbacks_by_process={0: 1, 1000000002: 8193},
            cpu_completed_selected_losses=8193, cpu_completed_native_selection_replays=8193)

    def test_full_and_partial_chunks(self):
        check_coalesced_fallbacks(self.stats)

    def test_invalid_accounting_rejected(self):
        for key, value in [('deferred_cpu_fallbacks_flushed', 8192),
                           ('maximum_deferred_cpu_fallback_batch', 4097),
                           ('cpu_generic_fallbacks', 4097),
                           ('cpu_completed_selected_losses', 8192),
                           ('deferred_fallback_scalar_expansion_rounds', 1)]:
            with self.subTest(key=key):
                state = copy.deepcopy(self.stats)
                state[key] = value
                with self.assertRaises(AssertionError):
                    check_coalesced_fallbacks(state)


if __name__ == '__main__':
    unittest.main()
