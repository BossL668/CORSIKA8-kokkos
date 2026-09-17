import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from run_priority_followon_job import read_spec, run_step, stop_owned


class FollowonTests(unittest.TestCase):
    def test_spec_is_literal_pinned_and_unique(self):
        spec = dict(schema=1, output='/unused/fresh', timeout_seconds=100,
                    maximum_tree_rss_gib=4,
                    steps=[dict(label='probe', command=['/never-executed', 'argument with spaces'], timeout_seconds=10)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'SPEC.json'
            def load(value):
                path.write_text(json.dumps(value))
                return read_spec(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(load(spec), spec)
            changed = copy.deepcopy(spec)
            changed['steps'] *= 2
            with self.assertRaises(ValueError): load(changed)
            changed['steps'] = [dict(label='../escape', command=['/never'], timeout_seconds=10)]
            with self.assertRaises(ValueError): load(changed)
            changed['steps'] = [dict(label='shell-string', command='not an argv list', timeout_seconds=10)]
            with self.assertRaises(ValueError): load(changed)

    def test_no_signal_to_finished_child(self):
        with patch('run_priority_followon_job.psutil.Process') as signal:
            child = unittest.mock.Mock()
            child.poll.return_value = 0
            stop_owned(child)
            signal.assert_not_called()

    def test_expired_deadline_never_launches_child(self):
        with patch('run_priority_followon_job.time.monotonic', return_value=11.), \
                patch('run_priority_followon_job.subprocess.Popen') as launch:
            with self.assertRaisesRegex(TimeoutError, 'before child launch'):
                run_step({}, {}, Path('/never-created'), 10.)
            launch.assert_not_called()

    def test_captured_descendants_are_stopped_even_in_separate_sessions(self):
        child, root, descendant = [unittest.mock.Mock() for _ in range(3)]
        child.poll.return_value = None
        root.children.return_value = [descendant]
        with patch('run_priority_followon_job.psutil.Process', return_value=root), \
                patch('run_priority_followon_job.psutil.wait_procs', return_value=([], [])) as wait:
            stop_owned(child)
        descendant.terminate.assert_called_once()
        root.terminate.assert_called_once()
        self.assertEqual(wait.call_args_list[0].args[0], [descendant, root])

    def test_cpu_interference_preserves_current_event_but_invalidates_timing(self):
        import sys
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            step = dict(label='tiny-fake-event', command=[sys.executable, '-c',
                'import time; time.sleep(0.03); print("FAKE_EVENT_COMPLETE")'], timeout_seconds=10)
            spec = dict(cwd=directory, environment={}, maximum_tree_rss_gib=4,
                        isolation={'wait_workflow_path': ['/foreign/workflow']})
            actual_sleep = time.sleep
            with patch('run_priority_followon_job.foreign_workflows', return_value=[123456]), \
                    patch('run_priority_followon_job.psutil.virtual_memory', return_value=SimpleNamespace(available=32*2**30)), \
                    patch('run_priority_followon_job.time.sleep', side_effect=lambda _: actual_sleep(.01)):
                result = run_step(step, spec, out, time.monotonic()+10)
            self.assertTrue(result['complete'])
            self.assertFalse(result['performance_valid'])
            self.assertFalse(result['signals_only_to_owned_child'])
            self.assertIn('FAKE_EVENT_COMPLETE', (out/'tiny-fake-event.log').read_text())
            self.assertTrue((out/'KNOWN_INTERFERENCE.json').is_file())


if __name__ == '__main__':
    unittest.main()
