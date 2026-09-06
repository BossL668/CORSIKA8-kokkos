import importlib.util
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace
import tempfile

import pyarrow as pa

spec = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[2] / 'tools/run_paired_acceptance.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PairedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'common_argv': ['-p', '2212', '-E', '1000', '--emthin', '1e-6'],
                         'executables': {b: {'path': f'/install/{b}/bin/c8_air_shower'} for b in ('cuda', 'openmp')},
                         'openmp_threads': 16}

    def test_each_backend_explicitly_restarts_same_seed(self):
        for backend in ('cuda', 'openmp'):
            command = runner.make_command(self.manifest, {'seed': 1234}, backend, Path('/output'))
            self.assertEqual(command[command.index('--seed') + 1], '1234')
            self.assertEqual(command[command.index('-N') + 1], '1')
            self.assertNotIn('--max-weight', command)
            self.assertEqual(command[command.index('--radio-backend') + 1], 'kokkos')
            self.assertEqual(command[command.index('--hadronic-workers') + 1], '1')

    def test_reference_argv_only_removes_run_identity(self):
        command = ['exe', '-p', '2212', '-N', '1', '-f', '/old', '--seed', '19', '--emthin', '1e-6']
        self.assertEqual(runner.common_argv(command), ['-p', '2212', '--emthin', '1e-6'])

    def test_backends_do_not_mix_thread_pools(self):
        gpu = runner.child_environment('cuda', 16)
        self.assertEqual(gpu['OMP_NUM_THREADS'], '1')
        cpu = runner.child_environment('openmp', 16)
        self.assertEqual(cpu['OMP_NUM_THREADS'], '16')
        self.assertEqual(cpu['CUDA_VISIBLE_DEVICES'], '')

    def test_dirac_uses_128_threads_in_one_shower(self):
        self.manifest['openmp_threads'] = 128
        command = runner.make_command(self.manifest, {'seed': 1234}, 'openmp', Path('/output'))
        self.assertEqual(command[command.index('--kokkos-num-threads') + 1], '128')
        self.assertEqual(command[command.index('-N') + 1], '1')
        env = runner.child_environment('openmp', 128)
        self.assertEqual(env['OMP_NUM_THREADS'], '128')
        self.assertEqual(env['OMP_THREAD_LIMIT'], '128')
        self.assertEqual(env['OMP_PROC_BIND'], 'spread')
        self.assertEqual(env['OMP_PLACES'], 'threads')
        self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

    def test_waveform_finite_check(self):
        self.assertTrue(runner.finite_array(pa.array([[0., 1.], [2., 3.]])))
        self.assertFalse(runner.finite_array(pa.array([[0., float('nan')]])))
        self.assertFalse(runner.finite_array(pa.array([float('inf')])))
        self.assertFalse(runner.finite_array(pa.array([[1.], None])))

    def test_memory_incident_is_not_an_ordinary_retryable_event_failure(self):
        with tempfile.TemporaryDirectory(prefix='c8-runner-stop-test-') as folder:
            self.manifest['executables']['cuda']['sha256'] = 'unchanged'
            proc = mock.Mock(pid=999999999)
            proc.poll.return_value = None
            with mock.patch.object(runner, 'digest', return_value='unchanged'), \
                 mock.patch.object(runner, 'memory', side_effect=[(8192, 0), (8192, 5000)]), \
                 mock.patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)), \
                 mock.patch.object(runner.subprocess, 'Popen', return_value=proc), \
                 mock.patch.object(runner, 'stop_child'):
                with self.assertRaises(runner.MemorySafetyStop):
                    runner.run_one(SimpleNamespace(output=Path(folder)), self.manifest,
                                   {'index': 0, 'seed': 1234, 'cpu_root': '/reference'}, 'cuda', None)
            self.assertTrue((Path(folder) / 'runner_memory_stop.json').is_file())


if __name__ == '__main__':
    unittest.main()
