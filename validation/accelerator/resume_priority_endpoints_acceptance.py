#!/usr/bin/env python3
"""Resume only missing timing cases in a frozen, terminal priority-mode pilot.

Completed samples are re-read and never rerun or selected by speed. An explicitly
GPU-conflict-interrupted attempt is moved to an audit directory, not deleted.
No build, installation, physics change, or signal to unrelated work is allowed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from run_overlap_guarded import compute_gpu_pids
from run_priority_endpoints_acceptance import check_output, put, save, sha, timing_modes, timing_order


def plan(standalone, seeds, only_mode=None, modes=None):
    modes = timing_modes(standalone, modes)
    if only_mode is not None and only_mode not in modes:
        raise ValueError('Requested mode is absent from the frozen experiment')
    result = []
    for i in range(seeds):
        order = timing_order(modes, i)
        result.extend((85000001+i, mode) for mode in order
                      if only_mode is None or mode == only_mode)
    return result


def frozen_experiment(identity, standalone, requested_seeds=None):
    """New runners record exact order/count; old v1/v2 recorded an unordered set."""
    modern = 'timing_seeds' in identity
    seeds = identity['timing_seeds'] if modern else (requested_seeds or 5)
    if not isinstance(seeds, int) or seeds <= 0:
        raise ValueError('invalid frozen timing seed count')
    if modern and requested_seeds is not None and requested_seeds != seeds:
        raise ValueError('timing seed count differs from frozen provenance')
    modes = timing_modes(standalone, identity['modes'] if modern else None)
    if standalone not in modes or set(identity['modes']) != set(modes):
        raise ValueError('standalone/modes differ from frozen experiment')
    binding = {}
    for key, default, allowed in (
            ('omp_proc_bind', 'false', ('false', 'true', 'close', 'spread')),
            ('omp_places', 'cores', ('cores', 'threads', 'sockets'))):
        if modern and key not in identity:
            raise ValueError('modern frozen provenance lacks ' + key)
        value = identity.get(key, default)
        if value not in allowed:
            raise ValueError('unsupported frozen ' + key)
        binding[key] = value
    return dict(modes=modes, seeds=seeds, preserve_frozen_command=modern, **binding)


def select_template(out, records, identity, schedule):
    if records:
        command = records[0]['guard']['command']
        provenance = dict(source='completed-record', label=records[0]['label'])
    else:
        command = None
        for seed, mode in schedule:
            path = out / ('Fe100TeV-{}-{}-command.json'.format(seed, mode))
            if path.is_file() and not path.is_symlink():
                command = json.loads(path.read_text())['command']
                provenance = dict(source='saved-event-command', path=str(path), sha256=sha(path))
                break
        if command is None:
            value = identity.get('fe_command_template', identity.get('fe_command'))
            if isinstance(value, dict):
                command = value.get('command')
            elif isinstance(value, list):
                command = value
            if command is None:
                raise RuntimeError('No completed or saved Fe command; refusing to invent a zero-record template')
            provenance = dict(source='embedded-frozen-provenance')
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('invalid frozen command template')
    return list(command), provenance


def event_command(template, binary, out, seed, mode, threads, preserve_frozen=False):
    command = list(template)
    command[0] = str(binary)
    if not preserve_frozen and '--kokkos-cooperative-policy' in command:
        i = command.index('--kokkos-cooperative-policy')
        del command[i:i+2]
    updates = (('-N', 1), ('-s', seed), ('-f', out),
               ('--kokkos-execution', mode), ('--kokkos-num-threads', 1 if mode == 'cuda' else threads))
    if not preserve_frozen:
        updates += (
        ('--em-backend', 'kokkos'), ('--radio-backend', 'kokkos'),
        ('--gpu-physics-source', 'proposal-native'), ('--gpu-min-batch', 4096),
        ('--gpu-memory-fraction', .7), ('--hadronic-workers', 1), ('--kokkos-device', 0))
    for key, value in updates:
        put(command, key, value)
    return command


def verify_command(guard, expected):
    if guard['command'] != expected:
        raise RuntimeError('Archived command differs; refusing to mix configurations')


def interruption_kind(guard):
    if (guard.get('pass') is False and guard.get('returncode') == -9
            and guard.get('failure') == 'foreign GPU work during diagnostic'
            and bool(guard.get('foreign_gpu_pids'))):
        return 'interrupted-after-launch-by-foreign-gpu'
    if (guard.get('pass') is False and 'returncode' in guard and guard['returncode'] is None
            and guard.get('failure') == 'foreign GPU work before starting'
            and guard.get('exclusive_gpu_required') is True
            and bool(guard.get('foreign_gpu_pids'))
            and guard.get('child_pid') is None and guard.get('process_wall_s') is None
            and guard.get('peak_tree_rss_bytes') == 0):
        return 'prelaunch-foreign-gpu-refusal-no-shower-started'
    return None


def can_archive_interruption(guard):
    return interruption_kind(guard) is not None


def wait_until_idle(observe, publish, idle_seconds, timeout,
                    now=time.monotonic, sleep=time.sleep):
    if timeout <= 0 or idle_seconds < 0 or idle_seconds >= timeout:
        raise ValueError('resource timeout must exceed nonnegative idle interval')
    deadline = now() + timeout
    idle_since = None
    while True:
        observation = observe()
        instant = now()
        safe = observation.get('safe') is True
        idle_since = (idle_since if idle_since is not None else instant) if safe else None
        observation['idle_seconds'] = 0 if idle_since is None else instant - idle_since
        observation['wait_remaining_seconds'] = max(0., deadline - instant)
        publish(observation)
        if idle_since is not None and instant - idle_since >= idle_seconds:
            return
        if instant >= deadline:
            raise TimeoutError('Resources not idle before deadline; no unrelated work stopped or restarted')
        sleep(min(2., deadline - instant))


def workflow_pids(fragment):
    if not fragment:
        return []
    excluded = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    result = []
    for process in psutil.process_iter(['pid', 'cmdline', 'status']):
        info = process.info
        if info['pid'] in excluded or info['status'] == psutil.STATUS_ZOMBIE:
            continue
        if any(fragment in arg for arg in (info['cmdline'] or [])):
            result.append(info['pid'])
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-unit', required=True)
    parser.add_argument('--standalone', choices=('cuda', 'openmp'), required=True)
    parser.add_argument('--timing-seeds', type=int,
                        help='Must match recorded count; old provenance without count defaults to five')
    parser.add_argument('--only-mode', choices=('cuda-openmp', 'openmp-cuda'),
                        help='Revise remaining queue; retain every completed mode without rerunning it')
    parser.add_argument('--libraries', type=Path)
    parser.add_argument('--wait-workflow-path')
    parser.add_argument('--idle-seconds', type=float, default=60.)
    parser.add_argument('--resource-wait-timeout', type=float, default=21600.,
                        help='Maximum seconds to wait per case; never signals unrelated work')
    args = parser.parse_args()
    assert (args.timing_seeds is None or args.timing_seeds > 0) and args.idle_seconds >= 0
    assert args.resource_wait_timeout > args.idle_seconds
    out = args.run.resolve(strict=True)
    tools = Path(__file__).resolve().parent
    with (out/'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        properties = dict(line.split('=', 1) for line in subprocess.check_output(
            ['systemctl', '--user', 'show', args.previous_unit, '-p', 'ActiveState',
             '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], text=True).splitlines())
        if properties.get('MainPID') != '0' or properties.get('ActiveState') not in ('failed', 'inactive'):
            raise RuntimeError('Previous workflow is not terminal; refusing to start another')
        state = json.loads((out/'STATUS.json').read_text())
        if state['complete']:
            raise RuntimeError('Already complete; no simulation will be launched')
        assert json.loads((out/'CORRECTNESS_GATES.json').read_text())['passed']
        identity = json.loads((out/'PROVENANCE.json').read_text())
        binary = out/'binaries/c8_air_shower'
        assert sha(binary) == identity['binary_sha256'], 'frozen binary changed'
        assert sha(tools/'run_overlap_guarded.py') == identity['guard_sha256'], 'guard changed'
        experiment = frozen_experiment(identity, args.standalone, args.timing_seeds)
        source = Path(identity['source']).resolve(strict=True)
        threads = identity['threads']
        affinity = set(identity['affinity'])
        os.sched_setaffinity(0, affinity)
        # Refuse duplicate runs even if an old service lost its original handle.
        for process in psutil.process_iter(['pid', 'cmdline']):
            if process.info['cmdline'] and process.info['cmdline'][0] == str(binary):
                raise RuntimeError('Frozen binary is already running: '+str(process.pid))
        os.environ.update(FLUPRO='/home/yuhanglu/fluka', CORSIKA_DATA=str(source/'modules/data'),
            OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads), OMP_PROC_BIND=experiment['omp_proc_bind'],
            OMP_PLACES=experiment['omp_places'], OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        if args.libraries:
            os.environ['LD_LIBRARY_PATH'] = str(args.libraries)+':'+os.environ.get('LD_LIBRARY_PATH', '')
        records = {record['label']: record for record in state['records']}
        assert len(records) == len(state['records']), 'duplicate completed record'
        original_schedule = plan(args.standalone, experiment['seeds'], modes=experiment['modes'])
        schedule = plan(args.standalone, experiment['seeds'], args.only_mode, experiment['modes'])
        template, template_provenance = select_template(out, state['records'], identity, original_schedule)
        labels = {'Fe100TeV-'+str(seed)+'-'+mode for seed, mode in schedule}
        original_labels = {'Fe100TeV-'+str(seed)+'-'+mode for seed, mode in original_schedule}
        assert set(records) <= original_labels, 'unplanned completed record'
        archive = out/'continuations'/str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(out/'STATUS.json', archive/'STATUS.before.json')
        save(archive/'RESUME.json', dict(previous_unit=args.previous_unit,
            observed_terminal=properties, runner_sha256=sha(Path(__file__)),
            binary_sha256=sha(binary), guard_sha256=sha(tools/'run_overlap_guarded.py'),
            planned_cases=schedule, only_mode=args.only_mode,
            frozen_experiment=experiment, template_provenance=template_provenance,
            template_command=template, resource_wait_timeout=args.resource_wait_timeout,
            original_planned_cases=original_schedule, wait_workflow_path=args.wait_workflow_path,
            previous_completed_labels=list(records), time=time.time()))

        def status(phase, **extra):
            state.update(phase=phase, updated_unix=time.time(), **extra)
            save(out/'STATUS.json', state)

        def validate(label, mode, command):
            guard = json.loads((out/(label+'-guard/summary.json')).read_text())
            verify_command(guard, command)
            assert guard['pass'] and guard['returncode'] == 0, label+' did not complete'
            for line in (out/(label+'-guard/resources.jsonl')).open():
                for allowed in json.loads(line).get('thread_affinities', {}).values():
                    assert set(allowed) <= affinity, 'thread escaped affinity'
            checked = check_output(out/label, 1, mode)
            assert checked['accelerators'][0]['accelerator']['host_threads'] == (1 if mode == 'cuda' else threads)
            save(out/(label+'-CHECK.json'), checked)
            return guard, checked

        def wait_for_resources(label):
            last_log = [0.]
            def observe():
                observation = dict(case=label, time=time.time(), pid=os.getpid())
                try:
                    workloads = workflow_pids(args.wait_workflow_path)
                    gpu = compute_gpu_pids()
                    memory = psutil.virtual_memory().available
                    observation.update(workflow_pids=workloads, gpu_pids=list(gpu), available_bytes=memory)
                    observation['safe'] = not workloads and not gpu and memory >= 4*2**30
                except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
                    observation['safe'] = False
                    observation['observation_error'] = repr(error)
                return observation
            def publish(observation):
                save(out/'RESUME_WAIT.json', observation)
                instant = time.monotonic()
                if instant-last_log[0] >= 30:
                    print(json.dumps(observation), flush=True)
                    last_log[0] = instant
            wait_until_idle(observe, publish, args.idle_seconds, args.resource_wait_timeout)

        def command_for(label, seed, mode):
            return event_command(template, binary, out/label, seed, mode, threads,
                                 preserve_frozen=experiment['preserve_frozen_command'])

        try:
            # Re-read completed arrays before changing any interrupted output.
            for seed, mode in original_schedule:
                label = 'Fe100TeV-'+str(seed)+'-'+mode
                if label in records:
                    command = command_for(label, seed, mode)
                    guard, checked = validate(label, mode, command)
                    assert records[label]['guard'] == guard, 'guard differs from archived record'
                    assert records[label]['accelerator'] == checked['accelerators'][0]
            state.pop('error', None)
            if args.only_mode:
                state['revised_plan'] = dict(only_mode=args.only_mode,
                    requested_labels=sorted(labels), requested_cases=len(labels),
                    original_cases=len(original_schedule),
                    retained_other_labels=sorted(set(records)-labels),
                    reason='User requested local GPU-primary and server CPU-primary only',
                    completion_scope='requested mode only; original frozen mode matrix not required')
            status('resuming-missing-timing-cases', continuation=str(archive))
            for seed, mode in schedule:
                label = 'Fe100TeV-'+str(seed)+'-'+mode
                if label in records:
                    continue
                command = command_for(label, seed, mode)
                guard_path = out/(label+'-guard/summary.json')
                recovered = False
                if guard_path.exists():
                    old_guard = json.loads(guard_path.read_text())
                    verify_command(old_guard, command)
                    if old_guard.get('pass') and old_guard.get('returncode') == 0:
                        recovered = True  # finished between output and STATUS writes
                    else:
                        kind = interruption_kind(old_guard)
                        if kind is None:
                            raise RuntimeError('Not a verified GPU conflict: '+label)
                        if kind == 'prelaunch-foreign-gpu-refusal-no-shower-started':
                            if ((out/label).exists() or (out/(label+'-guard/command.log')).exists()
                                    or (out/(label+'-guard/resources.jsonl')).exists()):
                                raise RuntimeError('Prelaunch refusal contradicts process output: '+label)
                        save(archive/(label+'-ARCHIVE_REASON.json'), dict(kind=kind,
                            retained_not_deleted=True, foreign_gpu_pids=old_guard['foreign_gpu_pids'],
                            no_signal_to_foreign_process=True))
                        for suffix in ('', '-guard', '-command.json', '-CHECK.json'):
                            path = out/(label+suffix)
                            if path.exists():
                                assert not path.is_symlink(), 'refuse to move symlink'
                                path.rename(archive/path.name)
                elif (out/label).exists() or (out/(label+'-guard')).exists():
                    raise RuntimeError('Unclassified partial output: '+label)
                if not recovered:
                    status('waiting-for-resources-'+label)
                    wait_for_resources(label)
                    assert shutil.disk_usage(out).free >= 15*2**30
                    assert sha(binary) == identity['binary_sha256']
                    status(label)
                    save(out/(label+'-command.json'), dict(command=command))
                    result = subprocess.run([sys.executable, str(tools/'run_overlap_guarded.py'),
                        '--output', str(out/(label+'-guard')), '--timeout', '3600',
                        '--rss-limit-gib', '24', '--sample-resources', '--sample-threads',
                        '--require-exclusive-gpu', '--', *command], cwd=source)
                    assert result.returncode == 0, label+' guard failed; no automatic retry'
                guard, checked = validate(label, mode, command)
                record = dict(label=label, mode=mode, seed=seed, family='Fe100TeV', guard=guard,
                    shower_s=checked['timing']['shower_0']['wall_time_ms']/1000.,
                    accelerator=checked['accelerators'][0])
                state['records'].append(record)
                records[label] = record
                status(label+'-complete')
            assert labels <= set(records) <= original_labels
            if args.only_mode:
                state['revised_plan']['requested_completed'] = len(labels)
            state['complete'] = True
            status('completed')
        except BaseException as error:
            state['error'] = repr(error)
            status('failed')
            raise


if __name__ == '__main__':
    main()
