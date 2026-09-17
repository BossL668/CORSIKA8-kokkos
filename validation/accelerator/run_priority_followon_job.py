#!/usr/bin/env python3
"""SHA-pinned, bounded SERIAL validation plan; default is integrity-only.

Foreign workflow names live in SPEC.json, never expanded into OS argv. Reuse
the pinned build gate before every step. Guarded probes/acceptance retain their
own live GPU/RSS guards. Foreign CPU activity invalidates timing without
terminating the current physical event; memory/deadline failures stop only the
coordinator's own child tree, including the guard's separate child sessions.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import psutil

from launch_priority_build_job import pinned_bytes


def read_spec(path, expected):
    spec = json.loads(pinned_bytes(path, expected))
    if spec['schema'] != 1 or not Path(spec['output']).is_absolute():
        raise ValueError('unsupported follow-on spec')
    if spec['timeout_seconds'] <= 0 or spec['maximum_tree_rss_gib'] <= 0:
        raise ValueError('positive timeout/RSS bounds required')
    labels = []
    for step in spec['steps']:
        label = step['label']
        if not label or Path(label).name != label or label in ('.', '..'):
            raise ValueError('unsafe step label')
        labels.append(label)
        command = step['command']
        if not isinstance(command, list) or not command or any(
                not isinstance(value, str) or not value or '\0' in value for value in command):
            raise ValueError('commands must be literal nonempty argv lists')
        if step['timeout_seconds'] <= 0:
            raise ValueError('step needs a positive time bound')
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('missing or duplicate steps')
    for entry in spec.get('inputs', []):
        pinned_bytes(Path(entry['path']), entry['sha256'])
    return spec


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def foreign_workflows(prefixes, excluded):
    result = []
    for process in psutil.process_iter(['pid', 'uids', 'cmdline', 'status']):
        try:
            info = process.info
            if process.pid in excluded or info['uids'].real != os.getuid() or info['status'] == psutil.STATUS_ZOMBIE:
                continue
            if any(prefix in arg for prefix in prefixes for arg in info['cmdline'] or []):
                result.append(process.pid)
        except psutil.NoSuchProcess:
            continue
    return sorted(result)


def stop_owned(child):
    if child.poll() is None:
        try:
            root = psutil.Process(child.pid)
            # The existing resource guard deliberately puts its CORSIKA child
            # into a separate session. Capture descendants BEFORE terminating
            # the guard; killing only its process group would orphan that job.
            owned = list(reversed(root.children(recursive=True))) + [root]
        except psutil.NoSuchProcess:
            owned = []
        for process in owned:
            try:
                process.terminate()  # psutil verifies the captured PID identity.
            except psutil.NoSuchProcess:
                pass
        _, remaining = psutil.wait_procs(owned, timeout=10)
        for process in remaining:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(remaining, timeout=10)
        child.wait(timeout=10)


def run_step(step, spec, out, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('follow-on deadline before child launch')
    record = dict(label=step['label'], command=step['command'], complete=False,
        started_unix=time.time(), peak_tree_rss_bytes=0, foreign_workflow_pids=[],
        workflow_observation_errors=[], failure=None, signals_only_to_owned_child=False)
    started = time.monotonic()
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    with (out/(step['label']+'.log')).open('xb') as log:
        child = subprocess.Popen(step['command'], cwd=spec['cwd'],
            env=dict(os.environ, **spec['environment']), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        record['child_pid'] = child.pid
        try:
            while child.poll() is None:
                try:
                    tree = [psutil.Process(child.pid)] + psutil.Process(child.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    tree = []
                rss = 0
                for process in tree:
                    try:
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                record['peak_tree_rss_bytes'] = max(record['peak_tree_rss_bytes'], rss)
                try:
                    foreign = foreign_workflows(spec['isolation']['wait_workflow_path'], own | {p.pid for p in tree})
                except (OSError, ValueError, psutil.Error) as error:
                    foreign = []
                    record['workflow_observation_errors'].append(repr(error))
                record['foreign_workflow_pids'] = sorted(set(record['foreign_workflow_pids']) | set(foreign))
                if psutil.virtual_memory().available < 4*2**30:
                    record['failure'] = 'available RAM below 4 GiB'
                elif rss > spec['maximum_tree_rss_gib']*2**30:
                    record['failure'] = 'validation tree RSS limit'
                elif time.monotonic() >= min(deadline, started+step['timeout_seconds']):
                    record['failure'] = 'bounded validation deadline'
                if record['failure']:
                    break
                time.sleep(2)
        except BaseException as error:
            record['failure'] = repr(error)
        finally:
            record['signals_only_to_owned_child'] = child.poll() is None
            stop_owned(child)
            record['returncode'] = child.wait()
    record.update(seconds=time.monotonic()-started,
        complete=record['returncode'] == 0 and record['failure'] is None)
    record['performance_valid'] = record['complete'] and not record['foreign_workflow_pids'] and not record['workflow_observation_errors']
    save(out/(step['label']+'.json'), record)
    if record['foreign_workflow_pids'] or record['workflow_observation_errors']:
        save(out/'KNOWN_INTERFERENCE.json', record)
        marker = step.get('interference_directory')
        if marker and Path(marker).is_dir():
            save(Path(marker)/'KNOWN_INTERFERENCE.json', record)
    if not record['complete']:
        raise RuntimeError('step failed; outputs retained: '+step['label'])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    spec = read_spec(args.spec, args.spec_sha256)
    pinned_bytes(Path(spec['gate_bundle'])/'MANIFEST.json', spec['gate_bundle_sha256'])
    gate_file = Path(spec['gate_bundle'])/'tools/wait_priority_build_isolation.py'
    pinned_bytes(gate_file, spec['gate_sha256'])
    print(json.dumps(dict(steps=len(spec['steps']), spec_sha256=args.spec_sha256,
                          execute=args.execute, serial=True)), flush=True)
    if not args.execute:
        return
    out = Path(spec['output'])
    out.mkdir(parents=True, exist_ok=False)
    save(out/'SPEC.json', spec)
    loader = importlib.util.spec_from_file_location('pinned_priority_gate', gate_file)
    gate = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(gate)
    gate.verify_bundle(Path(spec['gate_bundle']))
    deadline = time.monotonic()+spec['timeout_seconds']
    status = dict(complete=False, records=[], phase='waiting-for-build')
    try:
        for step in spec['steps']:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('follow-on global deadline; no next step launched')
            options = dict(spec['isolation'], timeout=remaining)
            for key in ('wait_status', 'wait_binary'):
                options[key] = Path(options[key])
            isolation = SimpleNamespace(**options)
            status['phase'] = 'waiting-'+step['label']
            save(out/'STATUS.json', status)
            gate.wait_isolated(isolation, lambda record: save(out/'WAIT.json', record))
            final_observation = gate.observe(isolation)
            # An observation can block; test the absolute deadline AFTER it.
            if time.monotonic() >= deadline or final_observation.get('safe') is not True:
                raise RuntimeError('isolation changed before next step; no step launched')
            status['phase'] = step['label']
            save(out/'STATUS.json', status)
            status['records'].append(run_step(step, spec, out, deadline))
            save(out/'STATUS.json', status)
        status.update(complete=True, phase='completed',
            performance_valid=all(r['performance_valid'] for r in status['records']))
    except BaseException as error:
        status.update(phase='failed', error=repr(error))
        raise
    finally:
        save(out/'STATUS.json', status)


if __name__ == '__main__':
    main()
