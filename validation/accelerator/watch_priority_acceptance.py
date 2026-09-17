#!/usr/bin/env python3
"""Finalize a verified systemd-run experiment, never start/restart a simulation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def unit_state(unit):
    raw=subprocess.check_output(['systemctl','--user','show',unit,
        '-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID',
        '-p','Result','-p','ExecMainStatus'],text=True,timeout=10)
    return dict(line.split('=',1) for line in raw.splitlines() if '=' in line)


def unit_waiting(unit):
    if unit.get('MainPID') not in (None, '0'):
        return True
    if unit.get('ActiveState') in ('activating', 'deactivating', 'reloading'):
        return True
    if unit.get('ActiveState') == 'active':
        return not (unit.get('SubState') == 'exited' and unit.get('MainPID') == '0')
    return False


def wait_for_terminal(name, timeout, publish, observe=unit_state,
                      now=time.monotonic, sleep=time.sleep):
    if timeout <= 0:
        raise ValueError('report wait timeout must be positive')
    deadline = now() + timeout
    while True:
        unit = observe(name)
        waiting = unit_waiting(unit)
        record = dict(unit=name, observed_unix=time.time(), systemd=unit,
                      phase='waiting' if waiting else 'terminal')
        publish(record)
        if not waiting:
            return record
        remaining = deadline - now()
        if remaining <= 0:
            record.update(phase='timed-out', all_runs_complete=False,
                          service_success_verified=False, report_generated=False,
                          error='Live workflow did not become terminal before reporting deadline; no signal sent')
            publish(record)
            raise TimeoutError(record['error'])
        sleep(min(10., remaining))


def wait_for_run_initialization(run, name, timeout, publish, observe=unit_state,
                                now=time.monotonic, sleep=time.sleep):
    """Read only until the runner owns a valid STATUS; never create its run."""
    if timeout <= 0:
        raise ValueError('initialization wait timeout must be positive')
    deadline = now() + timeout
    while True:
        unit = observe(name)
        problem = None
        try:
            status = json.loads((run/'STATUS.json').read_text())
            initialized = isinstance(status, dict) and isinstance(status.get('complete'), bool)
            if not initialized:
                problem = 'STATUS lacks the runner completion field'
        except (OSError, ValueError) as error:
            initialized = False
            problem = type(error).__name__
        remaining = deadline - now()  # include time spent querying/reading
        record = dict(unit=name, observed_unix=time.time(), systemd=unit,
                      phase='waiting-for-run-initialization',
                      runner_initialized=initialized, initialization_problem=problem,
                      all_runs_complete=False, report_generated=False, signals_sent=0)
        if remaining <= 0:
            record.update(phase='timed-out', error='Runner did not initialize STATUS before reporting deadline; run directory was not created by watcher')
            publish(record)
            raise TimeoutError(record['error'])
        if initialized:
            record['phase'] = 'runner-initialized'
            publish(record)
            return record
        if not unit_waiting(unit):
            record.update(phase='initialization-failed',
                          error='Workflow is terminal or unavailable without initialized STATUS; no run directory or report was created')
            publish(record)
            raise RuntimeError(record['error'])
        publish(record)
        sleep(min(10., remaining))


def state_is_inside_run(run, state):
    run, state = run.resolve(), state.resolve()
    return state == run or run in state.parents


def prepare_internal_state_parent(run, state):
    """Create only report subdirectories, never the runner-owned root/parents."""
    relative = state.parent.relative_to(run)
    if not run.is_dir():
        raise RuntimeError('Runner directory disappeared before watcher state setup')
    parent = run
    for part in relative.parts:
        parent = parent/part
        # No parents=True: a disappearing root fails instead of recreating it.
        parent.mkdir(exist_ok=True)


def report_commands(tools, run, cpu_diagnosis):
    commands = [[sys.executable, str(tools/'audit_priority_completed_events.py'), str(run)],
                [sys.executable, str(tools/'summarize_priority_endpoints.py'),
                 str(run), '--output', str(run/'performance')]]
    if cpu_diagnosis:
        commands.append([sys.executable, str(tools/'report_cpu_primary_v3.py'),
                         str(run), '--output', str(run/'diagnosis')])
    return commands


def completion_evidence(unit):
    """Transient units may be collected before our next poll, even on success.

    Missing units are not successful units. They allow an offline artifact
    audit, with the unavailable service exit status explicitly retained.
    This helper never restarts or signals a simulation.
    """
    if unit_waiting(unit):
        raise RuntimeError('cannot finalize a live unit')
    if unit.get('LoadState')=='not-found':
        return dict(service_success_verified=False,
            service_exit_status='unavailable: transient unit unloaded',
            validation_scope='archived output audit only; not service exit certification')
    success=unit.get('ExecMainStatus')=='0' and unit.get('Result')=='success'
    return dict(service_success_verified=success,
        service_exit_status=dict(result=unit.get('Result'),exit_code=unit.get('ExecMainStatus')),
        validation_scope='archived output audit with observed terminal service status')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit',required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--cpu-diagnosis', action='store_true',
                   help='Generate CPU-primary paired work/throughput diagnosis after normal plots')
    p.add_argument('--timeout', type=float, default=32400.,
                   help='Maximum reporting wait in seconds; never signals or restarts the workflow')
    a=p.parse_args();tools=Path(__file__).resolve().parent
    if a.timeout <= 0:
        p.error('--timeout must be positive')
    a.run, a.state = a.run.resolve(), a.state.resolve()
    if a.state == a.run:
        p.error('--state must name a file, not the run directory')
    deadline = time.monotonic() + a.timeout
    internal_state = state_is_inside_run(a.run, a.state)
    state_writable = not internal_state
    if state_writable:
        a.state.parent.mkdir(parents=True,exist_ok=True)
    def publish(record):
        if state_writable:
            a.state.write_text(json.dumps(record,indent=2)+'\n')
        else:
            # Durable service stdout/journal carries startup progress without
            # racing the runner's mkdir(exist_ok=False).
            print(json.dumps(record), flush=True)
    if internal_state:
        wait_for_run_initialization(a.run, a.unit, a.timeout, publish, observe=unit_state)
        prepare_internal_state_parent(a.run, a.state)
        state_writable = True
    # Query the actual service handle, not only STATUS. A timed-out observer
    # is NOT a terminal workload and cannot trigger reports or a restart.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Reporting deadline expired before terminal-state wait; no signal sent')
    record=wait_for_terminal(a.unit,remaining,publish,observe=unit_state)
    unit=record['systemd']
    record.update(completion_evidence(unit))
    status=json.loads((a.run/'STATUS.json').read_text())
    if status.get('complete'):
        if unit.get('LoadState')!='not-found' and not record['service_success_verified']:
            raise RuntimeError('complete artifacts conflict with failed service exit')
        subprocess.run([sys.executable,str(tools/'check_priority_lifecycle.py'),str(a.run)],check=True)
    for command in report_commands(tools,a.run,a.cpu_diagnosis):
        subprocess.run(command,check=True)
    record.update(phase='reported',all_runs_complete=status.get('complete',False),
                  cpu_diagnosis_generated=a.cpu_diagnosis)
    a.state.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
