"""CPU-only acceptance-runner contracts; never start a service or GPU test."""
import copy
import unittest

from run_priority_endpoints_acceptance import (
    build_unit_complete, check_cpu_primary, timing_modes, timing_order, wait_for_build)


class BuildWaitTest(unittest.TestCase):
    def state(self, **changes):
        result = dict(LoadState='loaded', ActiveState='active', SubState='exited',
                      MainPID='0', Result='success', ExecMainStatus='0',
                      ExecMainCode='1', ExecMainStartTimestampMonotonic='1234')
        result.update(changes)
        return result

    def test_remain_after_exit_and_successful_inactive_complete(self):
        self.assertTrue(build_unit_complete(self.state()))
        self.assertTrue(build_unit_complete(self.state(ActiveState='inactive', SubState='dead')))

    def test_live_build_waits_then_exited_completes(self):
        states = iter([self.state(MainPID='123', SubState='running'), self.state()])
        pauses = []
        result = wait_for_build('not-a-real-unit', 10, lambda unit: next(states),
                                now=lambda: 0, sleep=pauses.append)
        self.assertEqual(pauses, [3.])
        self.assertEqual(result['MainPID'], '0')

    def test_failure_missing_and_unstarted_rejected(self):
        for changes in (dict(LoadState='not-found'), dict(ActiveState='failed'),
                        dict(Result='exit-code', ExecMainStatus='1'),
                        dict(ExecMainStatus='1'), dict(ExecMainCode='2'),
                        dict(ExecMainStartTimestampMonotonic='0')):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                build_unit_complete(self.state(**changes))
        with self.assertRaises(RuntimeError):
            build_unit_complete({})

    def test_wait_timeout_does_not_signal_or_restart(self):
        clock = iter([0., 11.])
        with self.assertRaises(TimeoutError):
            wait_for_build('not-a-real-unit', 10,
                           lambda unit: self.state(MainPID='123', SubState='running'),
                           now=lambda: next(clock), sleep=lambda seconds: self.fail('unexpected sleep'))


class TimingModesTest(unittest.TestCase):
    def test_paired_trials_really_alternate(self):
        modes = ['openmp', 'openmp-cuda']
        self.assertEqual([timing_order(modes, i) for i in range(4)],
                         [modes, modes[::-1], modes, modes[::-1]])

    def test_old_default_order_unchanged(self):
        modes = timing_modes('openmp')
        self.assertEqual(modes, ['openmp', 'cuda-openmp', 'openmp-cuda'])
        self.assertEqual(timing_order(modes, 1), ['openmp', 'openmp-cuda', 'cuda-openmp'])

    def test_arbitrary_selected_mode_counts(self):
        for modes in (['openmp-cuda'], ['openmp', 'openmp-cuda'],
                      ['cuda', 'openmp', 'cuda-openmp', 'openmp-cuda']):
            selected = timing_modes('openmp', modes)
            for index in range(9):
                self.assertCountEqual(timing_order(selected, index), modes)
        for modes in ([], ['cuda', 'cuda'], ['proposal']):
            with self.assertRaises(ValueError):
                timing_modes('cuda', modes)


class CpuPrimaryV3AuditTest(unittest.TestCase):
    def stats(self, queued=4097, flushes=2, maximum=4096):
        cooperative = dict(
            scheduling_policy='cpu-primary-v3-protected-front', primary_endpoint='openmp',
            auxiliary_endpoint='cuda', host_lepton_wave_limit=1024,
            auxiliary_lepton_wave_limit=8, host_profile_shards=0, host_profile_shard_bytes=0,
            specified_fallback_batch_limit=4096,
            specified_fallback_semantics='coordinator-owned batches; flush at 4096 or empty front without joining peer; protect EM products until front drains',
            primary_reserve_semantics='one full native OpenMP arena per species; only surplus may seed auxiliary work',
            auxiliary_grant_semantics='new work >= min(CUDA minimum batch, capacity); owned tails and pressure spills still drain',
            priority_endpoints={e: {k: dict(transport_records=10) for k in ('photon', 'lepton')}
                                for e in ('cuda', 'openmp')})
        return dict(accelerator=dict(cooperative=cooperative), gpu_particles=40,
                    profile=dict(steps=40), deferred_cpu_fallbacks_queued=queued,
                    deferred_cpu_fallbacks_flushed=queued, deferred_cpu_fallback_flushes=flushes,
                    maximum_deferred_cpu_fallback_batch=maximum,
                    deferred_fallback_scalar_expansion_rounds=0,
                    cpu_fallbacks_by_reason_name=dict(native_selection_replay=queued),
                    cpu_fallbacks_by_process={'3': queued}, cpu_generic_fallbacks=0,
                    cpu_completed_selected_losses=queued, cpu_completed_native_selection_replays=queued,
                    cross_species=dict(final_pending_photons=0, final_pending_leptons=0))

    def test_full_batch_and_final_partial_checked(self):
        audit = check_cpu_primary(self.stats())
        self.assertTrue(audit['full_batch_reached'])
        self.assertTrue(audit['at_least_one_partial_flush_required'])
        self.assertEqual(audit['flushed'], 4097)
        self.assertTrue(audit['final_flush_size_not_recorded'])
        for queued, flushes, maximum in ((4096, 1, 4096), (7, 1, 7), (0, 0, 0)):
            check_cpu_primary(self.stats(queued, flushes, maximum))

    def test_lost_partial_duplicate_or_unbounded_batch_rejected(self):
        for key, value in (('deferred_cpu_fallbacks_flushed', 4096),
                           ('deferred_cpu_fallback_flushes', 1),
                           ('deferred_cpu_fallback_flushes', 0),
                           ('maximum_deferred_cpu_fallback_batch', 4097),
                           ('cpu_completed_selected_losses', 4098),
                           ('deferred_fallback_scalar_expansion_rounds', 1)):
            stats = self.stats()
            stats[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                check_cpu_primary(stats)

    def test_new_metadata_and_empty_queues_required(self):
        for key, value in (('specified_fallback_batch_limit', 1),
                           ('host_lepton_wave_limit', 64), ('auxiliary_lepton_wave_limit', 16),
                           ('host_profile_shards', 8), ('primary_reserve_semantics', 'one small batch'),
                           ('auxiliary_grant_semantics', 'any tiny batch')):
            stats = self.stats()
            stats['accelerator']['cooperative'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_cpu_primary(stats)
        stats = self.stats()
        stats['cross_species']['final_pending_leptons'] = 1
        with self.assertRaises(AssertionError):
            check_cpu_primary(stats)

    def test_v2_archive_remains_readable(self):
        stats = copy.deepcopy(self.stats())
        c = stats['accelerator']['cooperative']
        c['scheduling_policy'] = 'cpu-primary-v2-simple'
        c['specified_fallback_batch_limit'] = 1
        self.assertIsNone(check_cpu_primary(stats))


class CpuPrimaryV4AuditTest(CpuPrimaryV3AuditTest):
    def stats(self, queued=4097, flushes=2, maximum=4096):
        stats = super().stats(queued, flushes, maximum)
        c = stats['accelerator']['cooperative']
        c.update(scheduling_policy='cpu-primary-v4-resident-helper',
                 primary_input_semantics='resident queue first, then waiting input; same fill order as standalone OpenMP',
                 auxiliary_continuation_call_limit=4,
                 subshower_cuda_submissions=7, subshower_cuda_commits=7,
                 subshower_cuda_autonomous_continuations=5,
                 cuda_completion_packets=2, maximum_cuda_packet_calls=4,
                 cuda_completion_mailbox_capacity=4,
                 cuda_completion_retention_budget_bytes=4096, cuda_completion_peak_bytes=1024,
                 auxiliary_retention_semantics='retention budget plus at most one ordinary bounded result; checked before the next call',
                 cuda_continuation_stop_order=[
                     'completed', 'no_progress', 'memory', 'call_limit', 'handoff', 'time', 'disabled'],
                 cuda_continuation_stops=[1, 0, 0, 1, 0, 0, 0],
                 cuda_completion_buffer_delay_ms=1., subshower_cuda_foreground_packets=1,
                 subshower_cuda_foreground_continuations=2)
        return stats

    def test_four_call_bound_and_continuation_accounting(self):
        result = check_cpu_primary(self.stats())['helper_continuation_audit']
        self.assertEqual(result['submitted_calls'], 7)
        self.assertEqual(result['continuation_calls'], 5)
        self.assertEqual(result['completion_packets'], 2)
        self.assertEqual(result['maximum_packet_calls'], 4)

    def test_no_helper_work_is_valid_not_forced(self):
        stats = self.stats()
        c = stats['accelerator']['cooperative']
        for key in ('subshower_cuda_submissions', 'subshower_cuda_commits',
                    'subshower_cuda_autonomous_continuations', 'cuda_completion_packets',
                    'maximum_cuda_packet_calls', 'subshower_cuda_foreground_packets',
                    'subshower_cuda_foreground_continuations'):
            c[key] = 0
        c['cuda_continuation_stops'] = [0] * 7
        self.assertEqual(check_cpu_primary(stats)['helper_continuation_audit']['submitted_calls'], 0)

    def test_bad_v4_bounds_or_ownership_rejected(self):
        for key, value in (('primary_input_semantics', 'waiting first'),
                           ('auxiliary_continuation_call_limit', 5),
                           ('maximum_cuda_packet_calls', 5),
                           ('maximum_cuda_packet_calls', 3),
                           ('cuda_completion_packets', 1),
                           ('subshower_cuda_autonomous_continuations', 8),
                           ('subshower_cuda_commits', 6),
                           ('cuda_completion_mailbox_capacity', 5),
                           ('cuda_completion_peak_bytes', -1),
                           ('cuda_completion_retention_budget_bytes', 0),
                           ('auxiliary_retention_semantics', 'strict byte ceiling'),
                           ('cuda_continuation_stops', [1, 0, 0, 1, 0, 1, 0]),
                           ('cuda_continuation_stops', [1, 0, 0, 0, 0, 1, 0]),
                           ('subshower_cuda_foreground_packets', 3),
                           ('subshower_cuda_foreground_continuations', 6)):
            stats = self.stats()
            stats['accelerator']['cooperative'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                check_cpu_primary(stats)

    def test_one_result_may_exceed_retention_budget(self):
        stats = self.stats()
        c = stats['accelerator']['cooperative']
        c['cuda_completion_peak_bytes'] = c['cuda_completion_retention_budget_bytes'] + 1
        c['cuda_continuation_stops'] = [1, 0, 1, 0, 0, 0, 0]
        audit = check_cpu_primary(stats)['helper_continuation_audit']
        self.assertGreater(audit['retention_peak_bytes'], audit['retention_budget_bytes'])
        self.assertFalse(audit['hard_retention_byte_ceiling_certified'])
        self.assertTrue(audit['rss_guard_evidence_required_separately'])
        # A completed packet can cross the threshold too: completion is checked
        # before memory, so exceeding the threshold does not imply a memory stop.
        c['cuda_continuation_stops'] = [2, 0, 0, 0, 0, 0, 0]
        check_cpu_primary(stats)

    def test_v4_requires_fields_but_v3_does_not(self):
        stats = self.stats()
        c = stats['accelerator']['cooperative']
        del c['subshower_cuda_autonomous_continuations']
        with self.assertRaises(KeyError):
            check_cpu_primary(stats)
        c['scheduling_policy'] = 'cpu-primary-v3-protected-front'
        self.assertNotIn('helper_continuation_audit', check_cpu_primary(stats))


class CpuPrimaryV5AuditTest(unittest.TestCase):
    def stats(self):
        stats = CpuPrimaryV4AuditTest().stats()
        stats['accelerator']['cooperative'].update(
            scheduling_policy='cpu-primary-v5-blocking-helper',
            auxiliary_blocking_wait_enabled=True, auxiliary_blocking_wait_calls=12,
            auxiliary_blocking_wait_ms=2.5,
            auxiliary_wait_semantics='reused blocking CUDA event on helper stream; default single-endpoint and GPU-primary fences unchanged; wait time overlaps primary work')
        return stats

    def test_valid_blocking_diagnostics_are_not_kernel_time(self):
        audit = check_cpu_primary(self.stats())['helper_continuation_audit']['blocking_wait']
        self.assertEqual(audit['calls'], 12)
        self.assertFalse(audit['kernel_time'])
        self.assertFalse(audit['global_cuda_flags_changed'])

    def test_bad_or_missing_wait_gate_rejected(self):
        for key, value in [('auxiliary_blocking_wait_enabled', False),
                           ('auxiliary_blocking_wait_calls', 0),
                           ('auxiliary_blocking_wait_calls', 1.5),
                           ('auxiliary_blocking_wait_ms', -1),
                           ('auxiliary_blocking_wait_ms', float('nan')),
                           ('auxiliary_wait_semantics', 'global device blocking')]:
            stats = self.stats()
            stats['accelerator']['cooperative'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                check_cpu_primary(stats)


class CpuPrimaryV6AuditTest(CpuPrimaryV5AuditTest):
    def stats(self):
        stats = super().stats()
        stats['accelerator']['cooperative'].update(
            scheduling_policy='cpu-primary-v6-species-coalescing',
            host_species_coalesces=3, host_species_maximum_deferrals=8,
            subshower_openmp_epochs=9,
            host_species_selection_semantics='defer sub-minimum species only while the other has a useful front; at most eight host choices; drain lone tails immediately')
        return stats

    def test_coalescing_audit_keeps_all_v5_gates(self):
        audit = check_cpu_primary(self.stats())['helper_continuation_audit']
        self.assertEqual(audit['host_species_coalesces'], 3)
        self.assertTrue(audit['blocking_wait']['enabled'])

    def test_bad_coalescing_metadata_rejected(self):
        for key, value in [('host_species_coalesces', -1),
                           ('host_species_coalesces', 10),
                           ('host_species_coalesces', True),
                           ('host_species_maximum_deferrals', 0),
                           ('host_species_selection_semantics', 'unbounded')]:
            stats = self.stats()
            stats['accelerator']['cooperative'][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_cpu_primary(stats)


class CpuPrimaryV7AuditTest(CpuPrimaryV6AuditTest):
    def stats(self):
        stats = super().stats()
        stats['accelerator']['cooperative'].update(
            scheduling_policy='cpu-primary-v7-tail-checkpoint',
            host_checkpoint_minimum_semantics='max(1, min(native minimum, initial batch / 4)); return surviving states without particle cuts',
            host_input_batch_log2_histogram=[0, 4, 5] + [0] * 18)
        return stats

    def test_tail_threshold_preserves_all_previous_gates(self):
        audit = check_cpu_primary(self.stats())['helper_continuation_audit']
        self.assertEqual(sum(audit['host_input_batch_log2_histogram']), 9)
        self.assertTrue(audit['blocking_wait']['enabled'])

    def test_bad_tail_metadata_rejected(self):
        for key, value in [
                ('host_checkpoint_minimum_semantics', 'drop tails'),
                ('host_input_batch_log2_histogram', []),
                ('host_input_batch_log2_histogram', [0] * 21),
                ('host_input_batch_log2_histogram', [True, 8] + [0] * 19),
                ('host_input_batch_log2_histogram', [-1, 10] + [0] * 19),
                ('host_input_batch_log2_histogram', [9.] + [0] * 20)]:
            stats = self.stats()
            stats['accelerator']['cooperative'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                check_cpu_primary(stats)


if __name__ == '__main__':
    unittest.main()
