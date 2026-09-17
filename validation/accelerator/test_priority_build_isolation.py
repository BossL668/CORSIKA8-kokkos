import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

from wait_priority_build_isolation import main, prior_terminal, verify_bundle, wait_isolated


class BuildIsolationTests(unittest.TestCase):
    def unit(self, **changes):
        result = dict(LoadState='loaded', ActiveState='active', SubState='exited',
            MainPID='0', Result='success', ExecMainStatus='0', ExecMainCode='1',
            ExecMainStartTimestampMonotonic='123')
        result.update(changes)
        return result

    def test_remain_after_exit_is_terminal(self):
        self.assertTrue(prior_terminal(self.unit()))
        self.assertFalse(prior_terminal(self.unit(MainPID='42', SubState='running')))

    def test_missing_failed_unstarted_fail_closed(self):
        for changes in [dict(LoadState='not-found'), dict(Result='exit-code'),
                        dict(ExecMainStartTimestampMonotonic='0')]:
            with self.assertRaises(RuntimeError):
                prior_terminal(self.unit(**changes))

    def test_wait_requires_continuous_idle(self):
        now = [0.]
        states = iter([True, True, False, True, True, True])
        records = []
        args = SimpleNamespace(timeout=100., idle_seconds=20.)
        result = wait_isolated(args, records.append,
            observe_fn=lambda _: dict(safe=next(states)), clock=lambda: now[0],
            sleep=lambda dt: now.__setitem__(0, now[0] + dt))
        self.assertEqual(now[0], 50.)
        self.assertEqual(result['idle_seconds'], 20.)
        self.assertTrue(all(not x['launches_started'] for x in records))

    def test_error_is_not_idle_and_timeout_never_launches(self):
        now = [0.]
        records = []
        args = SimpleNamespace(timeout=20., idle_seconds=10.)
        with self.assertRaises(TimeoutError):
            wait_isolated(args, records.append,
                observe_fn=lambda _: (_ for _ in ()).throw(OSError('unknown GPU')),
                clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
        self.assertTrue(all(x['safe'] is False for x in records))
        self.assertTrue(all(x['signals_sent'] == 0 for x in records))

    def test_slow_observation_cannot_grant_late_ready(self):
        now = [0.]
        records = []
        args = SimpleNamespace(timeout=20., idle_seconds=10.)
        calls = [0]
        def observe(_):
            calls[0] += 1
            if calls[0] == 2:
                now[0] += 11.  # Starts at t=10, finishes after deadline t=20.
            return dict(safe=True)
        with self.assertRaises(TimeoutError):
            wait_isolated(args, records.append, observe_fn=observe,
                clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
        self.assertEqual(records[-1]['phase'], 'timed-out')
        self.assertEqual(records[-1]['wait_remaining_seconds'], 0.)
        self.assertTrue(all(not r['launches_started'] for r in records))

    def test_first_observation_duration_is_not_idle_credit(self):
        now = [0.]
        args = SimpleNamespace(timeout=100., idle_seconds=20.)
        records = []
        calls = [0]
        def observe(_):
            calls[0] += 1
            if calls[0] == 1:
                now[0] = 15.
            return dict(safe=True)
        wait_isolated(args, records.append, observe_fn=observe,
            clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
        self.assertEqual(records[0]['idle_seconds'], 0.)
        self.assertEqual(now[0], 35.)

    def test_ready_at_exact_deadline_refuses(self):
        now = [0.]
        records = []
        args = SimpleNamespace(timeout=10., idle_seconds=10.)
        with self.assertRaises(TimeoutError):
            wait_isolated(args, records.append, observe_fn=lambda _: dict(safe=True),
                clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt))
        self.assertEqual(records[-1]['phase'], 'timed-out')

    def test_final_recheck_after_deadline_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'queue.json'
            argv = ['wait_priority_build_isolation.py', '--wait-unit', 'previous',
                '--wait-status', '/not-read/STATUS.json', '--wait-binary', '/not-read/app',
                '--wait-workflow-path', '/foreign/workflow', '--bundle', directory,
                '--state', str(state), '--execute', '--', '/not-executed/builder']
            with patch('sys.argv', argv), \
                    patch('wait_priority_build_isolation.verify_bundle', return_value='fixed-hash'), \
                    patch('wait_priority_build_isolation.wait_isolated', return_value={
                        'phase': 'ready', 'wait_deadline_monotonic': 20.}), \
                    patch('wait_priority_build_isolation.observe', return_value={'safe': True}), \
                    patch('wait_priority_build_isolation.time.monotonic', return_value=20.), \
                    patch('wait_priority_build_isolation.os.execvp') as execute:
                with self.assertRaises(TimeoutError):
                    main()
            execute.assert_not_called()
            result = json.loads(state.read_text())
            self.assertEqual(result['phase'], 'refused')
            self.assertFalse(result['launches_started'])
            self.assertEqual(result['signals_sent'], 0)

    def test_bundle_rejects_changes_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'tool.py').write_text('one')
            row = dict(path='tool.py', sha256=hashlib.sha256(b'one').hexdigest())
            (p/'MANIFEST.json').write_text(json.dumps(dict(files=[row])))
            self.assertEqual(len(verify_bundle(p)), 64)
            (p/'extra').write_text('unexpected')
            with self.assertRaises(ValueError): verify_bundle(p)
            (p/'extra').unlink()
            (p/'tool.py').write_text('two')
            with self.assertRaises(ValueError): verify_bundle(p)


if __name__ == '__main__':
    unittest.main()
