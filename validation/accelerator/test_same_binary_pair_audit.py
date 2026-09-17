#!/usr/bin/env python3
"""Host-only audit contract tests; no simulation, compiler or GPU calls."""
import copy
import unittest

from accept_adaptive_same_binary_pair import command_flags, physics_flags, check_guard, endpoint_steps, fallback_counts


class PairAuditTests(unittest.TestCase):
    def test_only_execution_fields_are_ignored(self):
        base = ['binary', '-E', '1e8', '-s', '12', '--emthin', '1e-6',
                '--max-weight', '0', '--gpu-memory-fraction', '.7',
                '--radio-backend', 'kokkos']
        cuda = base + ['-f', 'a', '--kokkos-execution', 'cuda', '--kokkos-num-threads', '1']
        dual = base + ['-f', 'b', '--kokkos-execution', 'cuda-openmp',
                       '--kokkos-num-threads', '20', '--kokkos-cooperative-policy', 'adaptive']
        self.assertEqual(physics_flags(cuda), physics_flags(dual))
        for key in ('-E', '-s', '--emthin', '--max-weight', '--gpu-memory-fraction', '--radio-backend'):
            changed = dual.copy(); changed[changed.index(key) + 1] = 'different'
            self.assertNotEqual(physics_flags(cuda), physics_flags(changed), key)

    def test_ambiguous_command_rejected(self):
        for command in (['b', '-N'], ['b', '-N', '1', '-N', '2'], ['b', 'energy', '10']):
            with self.assertRaises(ValueError): command_flags(command)

    def test_gpu_evidence_not_inferred(self):
        minimal = dict(returncode=0, failure=None, **{'pass': True})
        self.assertFalse(check_guard(minimal, False))
        with self.assertRaises(ValueError): check_guard(minimal, True)
        valid = dict(minimal, exclusive_gpu_required=True, performance_valid=True,
                     gpu_checks=10, foreign_gpu_pids=[])
        self.assertTrue(check_guard(valid, True))
        for changed in ({'performance_valid': False}, {'foreign_gpu_pids': [123]},
                        {'gpu_checks': 0}, {'returncode': 1}, {'failure': 'interrupted'}):
            with self.assertRaises(ValueError): check_guard(dict(valid, **changed), True)

    def test_single_endpoint_not_double_counted(self):
        statistics = dict(gpu_particles=15, profile=dict(steps=15),
            accelerator=dict(backend='cuda', host_threads=1, openmp=False))
        self.assertEqual(endpoint_steps(statistics), dict(cuda=15, openmp=0))
        statistics['accelerator']['openmp'] = True
        with self.assertRaises(ValueError): endpoint_steps(statistics)

    def test_dual_counts_and_commit_must_close(self):
        c = dict(subshower_cuda_submissions=2, subshower_cuda_commits=2,
                 adaptive={e: {k: dict(transport_records=n) for k, n in zip(('photon', 'lepton'), ns)}
                           for e, ns in (('cuda', (2, 3)), ('openmp', (4, 6)))})
        s = dict(gpu_particles=15, profile=dict(steps=15),
                 accelerator=dict(backend='cuda-openmp', cooperative=c))
        self.assertEqual(endpoint_steps(s), dict(cuda=5, openmp=10))
        changed = copy.deepcopy(s); changed['accelerator']['cooperative']['subshower_cuda_commits'] = 1
        with self.assertRaises(ValueError): endpoint_steps(changed)
        changed = copy.deepcopy(s); changed['accelerator']['cooperative']['adaptive']['cuda']['photon']['transport_records'] += 1
        with self.assertRaises(ValueError): endpoint_steps(changed)
        changed = copy.deepcopy(s); changed['profile']['steps'] += 1
        with self.assertRaises(ValueError): endpoint_steps(changed)

    def test_immediate_and_deferred_fallback_accounting(self):
        shared = dict(cpu_specified_final_states=12,
                      cpu_fallbacks_by_reason_name={'native_selection_replay': 12, 'unsupported_geometry': 3},
                      cpu_fallbacks_by_process={1: 15}, cpu_completed_native_selection_replays=12,
                      cpu_completed_selected_losses=12)
        single = dict(shared, deferred_cpu_fallbacks_queued=12, deferred_cpu_fallbacks_flushed=12,
                      cpu_generic_fallbacks=3)
        dual = dict(shared, deferred_cpu_fallbacks_queued=0, deferred_cpu_fallbacks_flushed=0,
                    cpu_generic_fallbacks=15)
        self.assertEqual(fallback_counts(single), dict(total_events=15, immediate_specified=0,
                                                      deferred_specified=12, unspecified_scalar=3))
        self.assertEqual(fallback_counts(dual), dict(total_events=15, immediate_specified=12,
                                                    deferred_specified=0, unspecified_scalar=3))
        for key in ('deferred_cpu_fallbacks_flushed', 'cpu_generic_fallbacks',
                    'cpu_completed_selected_losses'):
            broken = dict(single); broken[key] += 1
            with self.assertRaises(ValueError): fallback_counts(broken)


if __name__ == '__main__':
    unittest.main()
