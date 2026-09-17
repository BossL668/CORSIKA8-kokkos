#!/usr/bin/env python3
"""Host-only test doubles: no real GPU jobs, signals or simulator invocation."""
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_overlap_guarded as guard
from run_validated_adaptive_uhe import uhe_command


class MonitorContract(unittest.TestCase):
    def run_guard(self, observations, polls, status=0, expected_error=None, vanished_pids=()):
        with tempfile.TemporaryDirectory(prefix='c8-uhe-guard-test-') as temporary:
            output=Path(temporary)/'result'
            child=Mock(pid=123456)
            child.poll.side_effect=polls
            child.wait.return_value=status
            root=Mock(pid=child.pid)
            root.children.return_value=[]
            root.memory_info.return_value=SimpleNamespace(rss=1024)
            argv=['guard','--output',str(output),'--require-exclusive-gpu','--','mock-job']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch('sys.argv',argv))
                stack.enter_context(patch.object(guard,'compute_gpu_pids',side_effect=observations))
                stack.enter_context(patch.object(guard.psutil,'virtual_memory',return_value=SimpleNamespace(available=16*2**30)))
                def process(pid):
                    if pid in vanished_pids:raise guard.psutil.NoSuchProcess(pid)
                    return root
                stack.enter_context(patch.object(guard.psutil,'Process',side_effect=process))
                launch=stack.enter_context(patch.object(guard.subprocess,'Popen',return_value=child))
                kill=stack.enter_context(patch.object(guard.os,'killpg'))
                stack.enter_context(patch.object(guard.time,'sleep'))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if expected_error:
                    with self.assertRaises(expected_error):guard.main()
                    code=1
                else:code=guard.main()
                record=json.loads((output/'summary.json').read_text())
                return code,record,launch.call_count,kill.call_args_list

    def test_busy_gpu_never_launches_or_signals_foreign_job(self):
        code,r,launch,kills=self.run_guard([{999999}],[])
        self.assertEqual((code,launch,kills),(1,0,[]))
        self.assertFalse(r['performance_valid'])
        self.assertEqual(r['foreign_gpu_pids'],[999999])

    def test_own_gpu_process_is_accepted(self):
        code,r,launch,kills=self.run_guard([set(),{123456}],[None,0,0])
        self.assertEqual((code,launch,kills),(0,1,[]))
        self.assertTrue(r['pass'] and r['performance_valid'])
        self.assertEqual(r['gpu_checks'],2)

    def test_foreign_competition_stops_only_own_child_group(self):
        code,r,launch,kills=self.run_guard([set(),{123456,999999}],[None,None],-15)
        self.assertEqual((code,launch),(1,1))
        self.assertFalse(r['performance_valid'])
        self.assertEqual(r['foreign_gpu_pids'],[999999])
        self.assertEqual([c.args for c in kills],[(123456,signal.SIGTERM)])

    def test_observation_failure_is_not_silently_idle(self):
        code,r,_,kills=self.run_guard([set(),RuntimeError('smi unavailable')],
                                     [None,None],-15,RuntimeError)
        self.assertEqual(code,1)
        self.assertIn('smi unavailable',r['failure'])
        self.assertFalse(r['performance_valid'])
        self.assertEqual([c.args for c in kills],[(123456,signal.SIGTERM)])

    def test_transient_query_timeout_does_not_kill_simulator(self):
        error=subprocess.TimeoutExpired(['nvidia-smi'],3)
        code,r,launch,kills=self.run_guard([set(),error],[None,0,0])
        self.assertEqual((code,launch,kills),(0,1,[]))
        self.assertTrue(r['pass'])
        self.assertFalse(r['performance_valid'])
        self.assertEqual(r['gpu_checks'],1)  # unknown must not count as success
        self.assertEqual(r['gpu_observation_failures'],1)

    def test_query_failure_before_launch_still_rejects_launch(self):
        error=subprocess.TimeoutExpired(['nvidia-smi'],3)
        code,r,launch,kills=self.run_guard([error],[],expected_error=subprocess.TimeoutExpired)
        self.assertEqual((code,launch,kills),(1,0,[]))
        self.assertFalse(r['performance_valid'])

    def test_query_can_recover_without_erasing_gap(self):
        r=dict(gpu_checks=0,gpu_observation_failures=0,gpu_observation_error_samples=[])
        with patch.object(guard,'compute_gpu_pids',side_effect=[OSError('unavailable'),{123}, {123,456}]), \
             patch.object(guard.psutil,'Process',return_value=Mock()):
            self.assertIsNone(guard.observe_running_gpu(r,{123},1))
            self.assertEqual(guard.observe_running_gpu(r,{123},6),set())
            self.assertEqual(guard.observe_running_gpu(r,{123},11),{456})
        self.assertEqual((r['gpu_checks'],r['gpu_observation_failures']),(2,1))

    def test_vanished_predecessor_does_not_kill_simulator_or_certify_timing(self):
        code,r,launch,kills=self.run_guard([set(),{123456,999999}], [None,0,0],
                                         vanished_pids={999999})
        self.assertEqual((code,launch,kills),(0,1,[]))
        self.assertTrue(r['pass'])
        self.assertFalse(r['performance_valid'])
        self.assertEqual(r['foreign_gpu_pids'],[])
        self.assertEqual(r['gpu_observation_failures'],1)
        self.assertEqual(r['gpu_observation_error_samples'][0]['unresolved_gpu_pids'],[999999])

    def test_live_competitor_still_rejected_with_vanished_pid_present(self):
        code,r,launch,kills=self.run_guard([set(),{123456,999998,999999}], [None,None],
                                         status=-15, vanished_pids={999998})
        self.assertEqual((code,launch),(1,1))
        self.assertEqual(r['foreign_gpu_pids'],[999999])
        self.assertEqual([c.args for c in kills],[(123456,signal.SIGTERM)])
        self.assertFalse(r['performance_valid'])

    def test_query_error_samples_are_bounded(self):
        r=dict(gpu_checks=0,gpu_observation_failures=0,gpu_observation_error_samples=[])
        with patch.object(guard,'compute_gpu_pids',side_effect=OSError('unavailable')):
            for i in range(100): guard.observe_running_gpu(r,set(),i)
        self.assertEqual(r['gpu_observation_failures'],100)
        self.assertEqual(len(r['gpu_observation_error_samples']),16)

    def test_real_child_survives_mocked_query_timeout(self):
        # Actual process creation/poll/wait, real memory guard, no GPU access.
        with tempfile.TemporaryDirectory(prefix='c8-guard-live-child-') as temporary:
            output=Path(temporary)/'result'
            argv=['guard','--output',str(output),'--require-exclusive-gpu','--',
                  sys.executable,'-c','import time; time.sleep(0.3); print("completed")']
            with patch('sys.argv',argv), patch.object(guard,'compute_gpu_pids',
                    side_effect=[set(),subprocess.TimeoutExpired(['nvidia-smi'],3)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code=guard.main()
            r=json.loads((output/'summary.json').read_text())
            self.assertEqual((code,r['returncode']),(0,0))
            self.assertTrue(r['pass'])
            self.assertFalse(r['performance_valid'])
            self.assertIn('completed',(output/'command.log').read_text())

    def test_modes_preserve_physics_and_input(self):
        original=['old','-E','1e8','-z','47','-a','180','--emthin','1e-6',
                  '--max-weight','50','--cut','0.05','--antennas','a.txt',
                  '--kokkos-cooperative-policy','legacy']
        original_copy=list(original)
        for mode in ('cuda','adaptive','legacy'):
            c=uhe_command(original,'binary','output',mode,2026110001)
            for flag in ('-E','-z','-a','--emthin','--max-weight','--cut','--antennas'):
                self.assertEqual(c[c.index(flag)+1],original[original.index(flag)+1])
            self.assertEqual('--kokkos-cooperative-policy' in c,mode!='cuda')
            self.assertEqual(c[c.index('--kokkos-num-threads')+1],'1' if mode=='cuda' else '20')
        self.assertEqual(original,original_copy)
        wrong=list(original);wrong[wrong.index('-E')+1]='1e5'
        with self.assertRaises(AssertionError):uhe_command(wrong,'binary','output','adaptive',1)


if __name__=='__main__':unittest.main()
