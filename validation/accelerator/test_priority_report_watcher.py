"""Host-only reporting contracts: no services or showers are started."""
import unittest
from pathlib import Path
import contextlib
import io
import json
import tempfile
from unittest.mock import patch

import watch_priority_acceptance as watcher
from watch_priority_acceptance import (completion_evidence, report_commands, unit_waiting,
    wait_for_terminal, wait_for_run_initialization, prepare_internal_state_parent)


class CompletionEvidenceTest(unittest.TestCase):
    def test_collected_unit_is_not_certified_as_success(self):
        result=completion_evidence(dict(LoadState='not-found',ActiveState='inactive',
                                        Result='success',ExecMainStatus='0'))
        self.assertFalse(result['service_success_verified'])
        self.assertIn('unavailable',result['service_exit_status'])

    def test_live_unit_cannot_be_finalized(self):
        for state in ('active','activating','deactivating'):
            with self.assertRaises(RuntimeError):
                completion_evidence(dict(LoadState='loaded',ActiveState=state))

    def test_observed_terminal_status_is_retained(self):
        self.assertTrue(completion_evidence(dict(LoadState='loaded',ActiveState='inactive',
                         Result='success',ExecMainStatus='0'))['service_success_verified'])
        self.assertFalse(completion_evidence(dict(LoadState='loaded',ActiveState='failed',
                          Result='exit-code',ExecMainStatus='1'))['service_success_verified'])

    def test_active_exited_with_zero_pid_is_terminal(self):
        state=dict(LoadState='loaded',ActiveState='active',SubState='exited',MainPID='0',
                   Result='success',ExecMainStatus='0')
        self.assertFalse(unit_waiting(state))
        self.assertTrue(completion_evidence(state)['service_success_verified'])
        failed=dict(state,Result='exit-code',ExecMainStatus='1')
        self.assertFalse(completion_evidence(failed)['service_success_verified'])

    def test_live_pid_or_missing_exited_proof_still_waits(self):
        for state in (dict(ActiveState='active',SubState='exited',MainPID='123'),
                      dict(ActiveState='active',SubState='running',MainPID='0'),
                      dict(ActiveState='active',SubState='exited'),
                      dict(ActiveState='inactive',SubState='dead',MainPID='123')):
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                completion_evidence(state)

    def test_terminal_poll_runs_without_waiting_for_inactive(self):
        states=iter([dict(ActiveState='active',SubState='running',MainPID='123'),
                     dict(ActiveState='active',SubState='exited',MainPID='0')])
        records=[];clock=[0.]
        def pause(seconds):clock[0]+=seconds
        result=wait_for_terminal('mock',20,records.append,lambda name:next(states),
                                 now=lambda:clock[0],sleep=pause)
        self.assertEqual(result['phase'],'terminal')
        self.assertEqual(clock[0],10.)

    def test_timeout_explicitly_not_complete_and_does_not_report(self):
        records=[];clock=[0.]
        def pause(seconds):clock[0]+=seconds
        with self.assertRaises(TimeoutError):
            wait_for_terminal('mock',12,records.append,
                lambda name:dict(ActiveState='active',SubState='running',MainPID='123'),
                now=lambda:clock[0],sleep=pause)
        self.assertEqual(clock[0],12.)
        self.assertEqual(records[-1]['phase'],'timed-out')
        self.assertFalse(records[-1]['all_runs_complete'])
        self.assertFalse(records[-1]['service_success_verified'])
        self.assertFalse(records[-1]['report_generated'])

    def test_cpu_diagnosis_is_opt_in_and_runs_after_plots(self):
        commands=report_commands(Path('/tools'),Path('/run'),False)
        self.assertEqual(len(commands),2)
        extended=report_commands(Path('/tools'),Path('/run'),True)
        self.assertEqual(extended[:2],commands)
        self.assertEqual(extended[-1][1:],['/tools/report_cpu_primary_v3.py','/run',
                                          '--output','/run/diagnosis'])


class InitializationRaceTest(unittest.TestCase):
    active = dict(LoadState='loaded', ActiveState='active', SubState='running', MainPID='123')
    terminal = dict(LoadState='loaded', ActiveState='active', SubState='exited',
                    MainPID='0', Result='success', ExecMainStatus='0')

    def test_real_runner_mkdir_wins_before_internal_watcher_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'fresh-run'
            state = run/'reports'/'watch.json'
            seen = []
            def pause(_):
                self.assertFalse(run.exists(), 'watcher precreated the runner output')
                run.mkdir(exist_ok=False)  # exact operation that previously failed
                (run/'STATUS.json').write_text(json.dumps({'complete': False}))
            def initialization(*args, **kwargs):
                return wait_for_run_initialization(*args, **kwargs, sleep=pause)
            def observe(_):
                seen.append(run.exists())
                return self.terminal if run.exists() else self.active
            with patch.object(watcher, 'unit_state', side_effect=observe), \
                    patch.object(watcher, 'wait_for_run_initialization', side_effect=initialization), \
                    patch.object(watcher.subprocess, 'run') as reports, \
                    patch('sys.argv', ['watch', '--unit', 'mock', '--run', str(run),
                                       '--state', str(state), '--timeout', '20']), \
                    contextlib.redirect_stdout(io.StringIO()):
                watcher.main()
            self.assertEqual(seen[0], False)
            self.assertTrue(state.is_file())
            self.assertEqual(json.loads(state.read_text())['phase'], 'reported')
            self.assertEqual(reports.call_count, 2)
            self.assertTrue(all('priority_' in call.args[0][1] for call in reports.call_args_list))

    def test_missing_status_timeout_never_creates_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'absent'
            clock, records = [0.], []
            with self.assertRaises(TimeoutError):
                wait_for_run_initialization(run, 'mock', 12., records.append,
                    observe=lambda _: self.active, now=lambda: clock[0],
                    sleep=lambda dt: clock.__setitem__(0, clock[0]+dt))
            self.assertEqual(clock[0], 12.)
            self.assertFalse(run.exists())
            self.assertEqual(records[-1]['phase'], 'timed-out')
            self.assertFalse(records[-1]['report_generated'])
            self.assertEqual(records[-1]['signals_sent'], 0)

    def test_terminal_or_missing_unit_without_status_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as directory:
            for unit in (self.terminal, dict(LoadState='not-found', ActiveState='inactive', MainPID='0'),
                         dict(LoadState='loaded', ActiveState='failed', MainPID='0')):
                run, records = Path(directory)/'absent', []
                with self.subTest(unit=unit), self.assertRaisesRegex(RuntimeError, 'without initialized STATUS'):
                    wait_for_run_initialization(run, 'mock', 10., records.append,
                        observe=lambda _, unit=unit: unit, sleep=lambda _: self.fail('should not wait'))
                self.assertFalse(run.exists())
                self.assertEqual(records[-1]['phase'], 'initialization-failed')

    def test_partial_status_is_not_initialization_and_can_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            run, records = Path(directory), []
            (run/'STATUS.json').write_text('{')
            def finish(_):
                self.assertFalse((run/'watch.json').exists())
                (run/'STATUS.json').write_text('{"complete":false}')
            result = wait_for_run_initialization(run, 'mock', 20., records.append,
                observe=lambda _: self.active, sleep=finish)
            self.assertEqual(records[0]['initialization_problem'], 'JSONDecodeError')
            self.assertTrue(result['runner_initialized'])

    def test_slow_initialization_observation_cannot_bypass_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            run, clock, records = Path(directory), [0.], []
            (run/'STATUS.json').write_text('{"complete":false}')
            def slow(_):
                clock[0] = 11.
                return self.active
            with self.assertRaises(TimeoutError):
                wait_for_run_initialization(run, 'mock', 10., records.append,
                    observe=slow, now=lambda: clock[0], sleep=lambda _: self.fail('no wait'))
            self.assertFalse((run/'watch.json').exists())

    def test_state_parent_helper_does_not_recreate_missing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'missing'
            with self.assertRaisesRegex(RuntimeError, 'disappeared'):
                prepare_internal_state_parent(run, run/'reports'/'watch.json')
            self.assertFalse(run.exists())


if __name__=='__main__':
    unittest.main()
