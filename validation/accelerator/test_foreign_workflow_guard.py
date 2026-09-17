"""Mock-only workflow isolation tests. No simulation, service or child is run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
import run_overlap_guarded as guard


PREFIX = '/data/yhlu/CorsikaData/corsika_validation_results/foreign_case'


class Process:
    def __init__(self, pid, argv=(), parents=(), children=(), error=None, zombie=False):
        self.pid = pid
        self.argv = list(argv)
        self.ancestry = list(parents)
        self.descendants = list(children)
        self.error = error
        self.zombie = zombie

    def parents(self):
        return self.ancestry

    def children(self, recursive=False):
        return self.descendants

    def status(self):
        return psutil.STATUS_ZOMBIE if self.zombie else psutil.STATUS_SLEEPING

    def cmdline(self):
        if self.error:
            raise self.error
        return self.argv

    def memory_info(self):
        return SimpleNamespace(rss=1024)


class ForeignWorkflowGuardTests(unittest.TestCase):
    def test_configuration_schemas_and_weak_prefix_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prefixes.json'
            for value in ([PREFIX], {'prefixes': [PREFIX, PREFIX]},
                          {'isolation': {'wait_workflow_path': [PREFIX]}}):
                path.write_text(json.dumps(value))
                self.assertEqual(guard.read_foreign_workflow_file(path), [PREFIX])
            for value in ({'prefixes': []}, {'prefixes': ['/']},
                          {'prefixes': ['relative-but-long-path']},
                          {'prefixes': str(PREFIX)}, {'isolation': []}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    guard.read_foreign_workflow_file(path)

    def test_shell_python_payloads_match_but_own_tree_and_ancestors_do_not(self):
        init = Process(1)
        ancestor = Process(100, ['bash', '-c', PREFIX], [init])
        own = Process(900, ['python', PREFIX], [ancestor, init])
        child = Process(901, ['python', PREFIX], [own, ancestor, init])
        descendant = Process(902, ['bash', '-c', 'python ' + PREFIX], [child, own, ancestor, init])
        child.descendants = [descendant]
        # Born after the first child-tree snapshot; dynamic ancestry still owns it.
        late_descendant = Process(903, ['python', PREFIX], [child, own, ancestor, init])
        # Sharing an ancestor with us must NOT incorrectly exempt a real sibling.
        shell = Process(950, ['bash', '-c', 'python ' + PREFIX + '/run.py'], [ancestor, init])
        python = Process(951, ['python', PREFIX + '/other.py'], [init])
        unrelated = Process(952, ['python', '/unrelated/task.py'], [init])
        zombie = Process(953, ['python', PREFIX], [init], zombie=True)
        processes = [init, ancestor, own, child, descendant, late_descendant,
                     shell, python, unrelated, zombie]
        lookup = {p.pid: p for p in processes}
        with patch.object(guard.os, 'getpid', return_value=900), \
                patch.object(guard.psutil, 'Process', side_effect=lookup.__getitem__), \
                patch.object(guard.psutil, 'process_iter', return_value=iter(processes)):
            self.assertEqual(guard.foreign_workflow_pids([PREFIX], 901), [950, 951])

    def test_uncertain_proc_reads_propagate_instead_of_claiming_idle(self):
        for error in (psutil.AccessDenied(950), psutil.NoSuchProcess(950), OSError('unreadable /proc')):
            with self.subTest(error=type(error).__name__):
                own = Process(900)
                uncertain = Process(950, error=error)
                with patch.object(guard.os, 'getpid', return_value=900), \
                        patch.object(guard.psutil, 'Process', return_value=own), \
                        patch.object(guard.psutil, 'process_iter', return_value=iter([uncertain])):
                    with self.assertRaises(type(error)):
                        guard.foreign_workflow_pids([PREFIX])

    def run_mock_guard(self, *, configured=True, observations=None, running=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        output = folder / 'result'
        config = folder / 'workflow-prefixes.json'
        config.write_text(json.dumps({'prefixes': [PREFIX]}))
        argv = ['run_overlap_guarded.py', '--output', str(output)]
        if configured:
            argv += ['--foreign-workflow-file', str(config)]
        argv += ['--', 'mock-computation']
        polls = iter([None, 0, 0] if running else [0, 0])
        child = SimpleNamespace(pid=42, poll=lambda: next(polls, 0), wait=lambda **kw: 0)
        process = Process(42)
        thread = lambda target, daemon: SimpleNamespace(start=target, join=lambda: None)
        with patch('sys.argv', argv), \
                patch.object(guard.psutil, 'virtual_memory', return_value=SimpleNamespace(available=8*1024**3)), \
                patch.object(guard.psutil, 'Process', return_value=process), \
                patch.object(guard.subprocess, 'Popen', return_value=child) as launch, \
                patch.object(guard.threading, 'Thread', side_effect=thread), \
                patch.object(guard.time, 'sleep'), \
                patch.object(guard.os, 'killpg') as kill, \
                patch.object(guard, 'foreign_workflow_pids', side_effect=observations) as observe, \
                contextlib.redirect_stdout(io.StringIO()):
            code = guard.main()
        return (code, json.loads((output/'summary.json').read_text()), output,
                launch.call_count, kill.call_count, observe.call_count)

    def test_startup_foreign_work_refuses_to_launch(self):
        code, report, output, launches, kills, _ = self.run_mock_guard(observations=[[950]])
        self.assertEqual(code, 1)
        self.assertEqual((launches, kills), (0, 0))
        self.assertIsNone(report['returncode'])
        self.assertFalse(report['performance_valid'])
        self.assertEqual(report['foreign_workflow_pids'], [950])
        self.assertTrue((output/'KNOWN_INTERFERENCE.json').is_file())

    def test_startup_uncertainty_refuses_to_launch(self):
        code, report, output, launches, kills, _ = self.run_mock_guard(
            observations=[psutil.AccessDenied(950)])
        self.assertEqual(code, 1)
        self.assertEqual((launches, kills), (0, 0))
        self.assertEqual(report['foreign_workflow_observation_failures'], 1)
        self.assertIn('uncertain', report['failure'])
        self.assertTrue((output/'KNOWN_INTERFERENCE.json').is_file())

    def test_running_cpu_interference_invalidates_timing_without_kill(self):
        code, report, output, launches, kills, _ = self.run_mock_guard(observations=[[], [950]])
        self.assertEqual(code, 0)
        self.assertEqual((launches, kills), (1, 0))
        self.assertTrue(report['pass'])  # physics/process completion is preserved
        self.assertIsNone(report['failure'])
        self.assertFalse(report['performance_valid'])
        marker = json.loads((output/'KNOWN_INTERFERENCE.json').read_text())
        self.assertFalse(marker['physical_event_killed_for_cpu_interference'])
        self.assertEqual(marker['foreign_workflow_pids'], [950])

    def test_running_observation_error_invalidates_timing_without_kill(self):
        code, report, output, launches, kills, _ = self.run_mock_guard(
            observations=[[], psutil.AccessDenied(950)])
        self.assertEqual(code, 0)
        self.assertEqual((launches, kills), (1, 0))
        self.assertTrue(report['pass'])
        self.assertFalse(report['performance_valid'])
        self.assertEqual(report['foreign_workflow_observation_failures'], 1)
        self.assertTrue((output/'KNOWN_INTERFERENCE.json').is_file())

    def test_configured_clean_observations_allow_valid_timing(self):
        code, report, output, launches, kills, _ = self.run_mock_guard(observations=[[], []])
        self.assertEqual(code, 0)
        self.assertEqual((launches, kills), (1, 0))
        self.assertTrue(report['performance_valid'])
        self.assertFalse((output/'KNOWN_INTERFERENCE.json').exists())

    def test_legacy_no_option_performs_no_new_observation_or_schema_changes(self):
        code, report, output, launches, kills, observations = self.run_mock_guard(configured=False)
        self.assertEqual(code, 0)
        self.assertEqual((launches, kills, observations), (1, 0, 0))
        self.assertNotIn('performance_valid', report)
        self.assertFalse(any(key.startswith('foreign_workflow') for key in report))
        self.assertFalse((output/'KNOWN_INTERFERENCE.json').exists())


if __name__ == '__main__':
    unittest.main()
