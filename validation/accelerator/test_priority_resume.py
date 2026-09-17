"""Host-only contracts; no SSH, services, simulation or archived data writes."""
import json
import tempfile
import unittest
from pathlib import Path

from resume_priority_endpoints_acceptance import (
    can_archive_interruption, event_command, frozen_experiment, interruption_kind,
    plan, select_template, verify_command, wait_until_idle)


class PriorityResumeTest(unittest.TestCase):
    def test_full_plan_preserves_original_order_and_all_seeds(self):
        schedule = plan('openmp', 5)
        self.assertEqual(len(schedule), 15)
        self.assertEqual(len(set(schedule)), 15)
        self.assertEqual(schedule[:6], [(85000001, 'openmp'), (85000001, 'cuda-openmp'),
            (85000001, 'openmp-cuda'), (85000002, 'openmp'),
            (85000002, 'openmp-cuda'), (85000002, 'cuda-openmp')])
        for mode in ('openmp', 'cuda-openmp', 'openmp-cuda'):
            self.assertEqual([s for s, m in schedule if m == mode], list(range(85000001, 85000006)))

    def test_only_verified_foreign_gpu_interruption_may_be_archived(self):
        guard = dict(pass_=False, returncode=-9, failure='foreign GPU work during diagnostic', foreign_gpu_pids=[123])
        guard['pass'] = guard.pop('pass_')
        self.assertTrue(can_archive_interruption(guard))
        for key, value in [('pass', True), ('returncode', 0), ('failure', 'timeout'),
                           ('foreign_gpu_pids', [])]:
            self.assertFalse(can_archive_interruption(dict(guard, **{key: value})))

    def test_revised_queue_runs_only_requested_endpoint_and_all_seeds(self):
        for mode in ('cuda-openmp', 'openmp-cuda'):
            schedule = plan('openmp', 5, mode)
            self.assertEqual(schedule, [(seed, mode) for seed in range(85000001, 85000006)])
            self.assertTrue(set(schedule) <= set(plan('openmp', 5)))
        with self.assertRaises(ValueError):
            plan('openmp', 5, 'cuda')

    def test_command_retains_physics_and_fixed_frozen_binary(self):
        template = ['old', '-E', '100000', '-Z', '26', '-A', '56', '--emthin', '1e-6',
                    '--max-weight', '0', '--antenna-file', '/antennas.txt',
                    '--kokkos-cooperative-policy', 'adaptive']
        original = list(template)
        command = event_command(template, Path('/frozen'), Path('/result'), 85000002, 'openmp-cuda', 130)
        self.assertEqual(template, original)
        self.assertEqual(command[0], '/frozen')
        for key, value in [('-E', '100000'), ('--emthin', '1e-6'), ('--max-weight', '0'),
                           ('--kokkos-num-threads', '130'), ('--gpu-memory-fraction', '0.7')]:
            self.assertEqual(command[command.index(key)+1], value)
        self.assertNotIn('--kokkos-cooperative-policy', command)
        verify_command({'command': command}, command)
        with self.assertRaises(RuntimeError):
            verify_command({'command': command}, command+['--emthin', '1e-4'])


class FrozenPriorityResumeTest(unittest.TestCase):
    def identity(self):
        return dict(modes=['openmp', 'openmp-cuda'], timing_seeds=5,
                    omp_proc_bind='spread', omp_places='threads')

    def test_two_mode_ab_ba_and_binding_from_provenance(self):
        experiment = frozen_experiment(self.identity(), 'openmp')
        self.assertEqual(experiment['omp_proc_bind'], 'spread')
        self.assertEqual(experiment['omp_places'], 'threads')
        schedule = plan('openmp', experiment['seeds'], modes=experiment['modes'])
        self.assertEqual(len(schedule), 10)
        self.assertEqual(schedule[:4], [(85000001, 'openmp'), (85000001, 'openmp-cuda'),
                                       (85000002, 'openmp-cuda'), (85000002, 'openmp')])
        self.assertEqual(plan('openmp', 5, 'openmp-cuda', experiment['modes']),
                         [(seed, 'openmp-cuda') for seed in range(85000001, 85000006)])
        with self.assertRaises(ValueError):
            frozen_experiment(self.identity(), 'openmp', 2)
        with self.assertRaises(ValueError):
            frozen_experiment(self.identity(), 'cuda')

    def test_new_provenance_cannot_silently_change_binding(self):
        identity = self.identity()
        del identity['omp_proc_bind']
        with self.assertRaises(ValueError):
            frozen_experiment(identity, 'openmp')
        legacy = frozen_experiment(dict(modes=['cuda-openmp', 'openmp-cuda', 'openmp']), 'openmp')
        self.assertEqual(legacy['modes'], ['openmp', 'cuda-openmp', 'openmp-cuda'])
        self.assertEqual(legacy['omp_proc_bind'], 'false')
        self.assertEqual(legacy['omp_places'], 'cores')

    def test_preserve_frozen_policy_and_all_physical_options(self):
        template = ['old', '-E', '100000', '--emthin', '1e-6', '--max-weight', '0',
                    '--gpu-memory-fraction', '0.70', '--gpu-min-batch', '4096',
                    '--kokkos-cooperative-policy', 'independent', '--kokkos-execution', 'openmp',
                    '--kokkos-num-threads', '130', '-N', '1', '-s', '85000001', '-f', 'old-result']
        result = event_command(template, Path('/frozen'), Path('/new'), 85000002, 'openmp-cuda',
                               130, preserve_frozen=True)
        self.assertEqual(result[result.index('--kokkos-cooperative-policy') + 1], 'independent')
        self.assertEqual(result[result.index('--gpu-memory-fraction') + 1], '0.70')
        self.assertEqual(result[result.index('--max-weight') + 1], '0')
        self.assertEqual(result[result.index('--emthin') + 1], '1e-6')
        self.assertEqual(len(result), len(template))
        self.assertEqual(template[0], 'old')

    def test_zero_records_uses_saved_fe_command_without_inventing_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = plan('openmp', 5, modes=['openmp', 'openmp-cuda'])
            with self.assertRaises(RuntimeError):
                select_template(root, [], self.identity(), schedule)
            path = root / 'Fe100TeV-85000001-openmp-command.json'
            template = ['frozen', '-E', '100000', '--emthin', '1e-6']
            path.write_text(json.dumps(dict(command=template)))
            command, provenance = select_template(root, [], self.identity(), schedule)
            self.assertEqual(command, template)
            self.assertEqual(provenance['source'], 'saved-event-command')
            self.assertEqual(len(provenance['sha256']), 64)

    def test_embedded_template_and_completed_records_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = ['frozen', '-E', '100000']
            command, source = select_template(root, [], dict(fe_command_template=template), [])
            self.assertEqual(command, template)
            records = [dict(label='done', guard=dict(command=template))]
            self.assertEqual(select_template(root, records, {}, [])[1]['source'], 'completed-record')

    def test_prelaunch_refusal_is_not_a_failed_shower(self):
        guard = dict(returncode=None, failure='foreign GPU work before starting',
                     foreign_gpu_pids=[637695], peak_tree_rss_bytes=0,
                     exclusive_gpu_required=True)
        guard['pass'] = False
        self.assertEqual(interruption_kind(guard), 'prelaunch-foreign-gpu-refusal-no-shower-started')
        self.assertTrue(can_archive_interruption(guard))
        for key, value in (('returncode', -9), ('child_pid', 123), ('process_wall_s', 0.),
                           ('peak_tree_rss_bytes', 1), ('foreign_gpu_pids', []),
                           ('exclusive_gpu_required', False), ('failure', 'timeout')):
            with self.subTest(key=key):
                self.assertIsNone(interruption_kind(dict(guard, **{key: value})))

    def test_wait_is_bounded_and_unknown_is_not_idle(self):
        clock = [0.]
        published = []
        def advance(seconds):
            clock[0] += seconds
        with self.assertRaises(TimeoutError):
            wait_until_idle(lambda: dict(observation_error='NVML unavailable'), published.append,
                            1., 5., now=lambda: clock[0], sleep=advance)
        self.assertEqual(clock[0], 5.)
        self.assertEqual(published[-1]['idle_seconds'], 0)

    def test_wait_requires_consecutive_observed_idle(self):
        clock = [0.]
        states = iter([False, True, False, True, True])
        def advance(seconds):
            clock[0] += seconds
        wait_until_idle(lambda: dict(safe=next(states)), lambda state: None, 2., 20.,
                        now=lambda: clock[0], sleep=advance)
        self.assertEqual(clock[0], 8.)


if __name__ == '__main__':
    unittest.main()
