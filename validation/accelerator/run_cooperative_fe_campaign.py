#!/usr/bin/env python3
"""Frozen-manifest Fe56 campaign: one fresh process per seed, sequential per arm.

Derived from the validated Fe campaign runner, frozen for cooperative16 validation.
The manifest is read-only. This campaign starts all 500 seeds with the new binary.
Attempt outputs/logs are never overwritten. Only operational timeouts receive one
same-seed retry. Other failed seeds remain visible while subsequent seeds run.
"""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import yaml

STOP = threading.Event()
REQUIRED_PARQUET = (
    'profile/profile.parquet', 'production_profile/profile.parquet',
    'energyloss/dEdX.parquet', 'particles/particles.parquet',
    'interactions/interactions.parquet', 'CoREAS/observers.parquet',
    'ZHS/observers.parquet',
)
OOM = re.compile(r'out of memory|bad_alloc|cudaErrorMemoryAllocation|'
                 r'cannot allocate memory|CUDA[^\n]*allocation[^\n]*fail', re.I)


class SafetyStop(RuntimeError):
    """Stop the whole arm for inadequate resources or unsafe resume state."""


class EventTimeout(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def yaml_load(path):
    return yaml.safe_load(Path(path).read_text())


def read_manifest(root, backend):
    raw = (root/'manifest.json').read_bytes()
    m = json.loads(raw)
    require(m['seeds'] == list(range(85000001, 85000501)),
            'manifest must contain exactly the 500 ordered reference seeds')
    require(backend == 'cuda-openmp' and m['threads'] == 16,
            'this campaign requires CUDA + 16 OpenMP threads')
    common = m.get('common_argv_by_backend', {}).get(backend, m['common_argv'])
    require(isinstance(common, list) and all(isinstance(x, str) for x in common),
            'common_argv must be a list of strings')
    forbidden = ('-N', '-s', '-f', '--em-backend', '--radio-backend',
                 '--gpu-physics-source', '--gpu-min-batch', '--gpu-memory-fraction',
                 '--hadronic-workers', '--kokkos-num-threads', '--kokkos-device',
                 '--kokkos-execution')
    require(not any(x.split('=')[0] in forbidden for x in common),
            'common_argv contains an event/backend override')
    executable = m['executables'][backend]
    require(sha(executable['path']) == executable['sha256'], 'executable hash changed')
    verified = {}
    for label, info in m['inputs'].items():
        path = info.get('path_by_backend', {}).get(backend, info.get('path'))
        # A label may intentionally describe a backend-specific input only.
        if path is None and 'path_by_backend' in info:
            continue
        require(path is not None, 'input has no path: '+label)
        require(sha(path) == info['sha256'], 'input hash changed: '+label)
        verified[label] = {'path': path, 'sha256': info['sha256']}
    require(float(m['memory'][backend]['rss_GiB']) > 0, 'invalid RSS limit')
    require(float(m['memory'][backend]['min_available_GiB']) > 0,
            'invalid host memory reserve')
    if backend == 'openmp':
        affinity = m['cpu_affinity_by_backend']['openmp']
        require(len(affinity) == 128 and len(set(affinity)) == 128,
                'OpenMP requires 128 distinct affinity CPUs')
        require(set(affinity) <= os.sched_getaffinity(0),
                'OpenMP affinity CPUs are not available to this runner')
    return m, hashlib.sha256(raw).hexdigest(), verified


def environment(m, backend):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in m['env_by_backend'][backend].items()})
    n = m['threads'] if backend in ('openmp', 'cuda-openmp') else 1
    env.update(OMP_NUM_THREADS=str(n), OMP_THREAD_LIMIT=str(n),
               OMP_PROC_BIND='spread', OMP_PLACES='cores',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1')
    if backend == 'cuda-openmp':
        env['OMP_PROC_BIND'] = 'false'  # Match the accepted 16-thread pilot.
    if backend == 'openmp':
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def command(m, backend, seed, output):
    common = m.get('common_argv_by_backend', {}).get(backend, m['common_argv'])
    args = [m['executables'][backend]['path'], *common, '-N', '1',
            '-s', str(seed), '-f', str(output), '--em-backend', 'kokkos',
            '--gpu-physics-source', 'proposal-native',
            '--radio-backend', 'kokkos', '--gpu-min-batch',
            str(m.get('gpu_min_batch', 4096)), '--gpu-memory-fraction',
            str(m.get('gpu_memory_fraction', 0.70)), '--hadronic-workers', '1']
    args += (['--kokkos-num-threads', str(m['threads'])] if backend == 'openmp'
             else ['--kokkos-device', '0'])
    if backend == 'cuda-openmp':
        args += ['--kokkos-execution', 'cuda-openmp',
                 '--kokkos-num-threads', str(m['threads'])]
    affinity = m.get('cpu_affinity_by_backend', {}).get(backend)
    if affinity:
        args = ['/usr/bin/taskset', '-c', ','.join(map(str, affinity)), *args]
    return args


def proc_state(pid):
    result = {'pid': pid, 'rss_bytes': 0, 'cpu_ticks': None,
              'threads': 0, 'start_ticks': None}
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        result.update(cpu_ticks=int(fields[11])+int(fields[12]),
                      start_ticks=int(fields[19]), state=fields[0])
        for line in Path('/proc', str(pid), 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_bytes'] = int(line.split()[1])*1024
            elif line.startswith('Threads:'):
                result['threads'] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return result


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise SafetyStop('MemAvailable cannot be read')


def resource_check(root, limits, rss=0):
    available = available_memory()
    if rss > float(limits['rss_GiB'])*2**30:
        raise SafetyStop('child RSS limit exceeded')
    if available < float(limits['min_available_GiB'])*2**30:
        raise SafetyStop('host memory reserve exhausted')
    free = shutil.disk_usage(root).free
    if free < 15*2**30:
        raise SafetyStop('disk reserve below 15 GiB')
    return available, free


def gpu_state(required=False):
    try:
        executable = ('/usr/lib/wsl/lib/nvidia-smi'
                      if Path('/usr/lib/wsl/lib/nvidia-smi').is_file()
                      else shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi')
        result = subprocess.run([
            executable, '-i', '0',
            '--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=4, check=True)
        values = [int(x.strip()) for x in result.stdout.strip().split(',')]
        require(len(values) == 4, 'unexpected nvidia-smi reply')
        return dict(zip(('index', 'used_MiB', 'total_MiB', 'utilization_percent'), values))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if required:
            raise SafetyStop('CUDA device unavailable: '+str(error)) from error
        return {'unavailable': str(error)}


def stop_child(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def signatures(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size,
                                     'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(root.rglob('*')) if p.is_file()}


def inspect_output(output, backend, seed, m):
    top = yaml_load(output/'summary.yaml')
    require(top['showers'] == 1 and top['seed'] == seed, 'wrong event count/seed')
    primary = yaml_load(output/'primary/summary.yaml')
    require(set(primary) == {'shower_0'}, 'primary must contain one shower')
    require(primary['shower_0']['pdg'] == 1000260560 and
            primary['shower_0']['total_energy'] == 1.e5, 'wrong Fe56 primary/energy')
    timing = yaml_load(output/'simulation_timing/summary.yaml')
    require(set(timing) == {'shower_0'} and timing['shower_0']['closed'] is True,
            'timing callback not closed')
    gpu = yaml_load(output/'gpu_em/summary.yaml')
    require(set(gpu) == {'shower_0'}, 'accelerator must contain one shower')
    event = gpu['shower_0']
    require(event['complete'] is True and event['status'] == 'complete',
            'accelerator event incomplete')
    s = event['statistics']
    accelerator = s['accelerator']
    require(accelerator['backend'] == backend and
            accelerator['gpu'] == (backend in ('cuda', 'cuda-openmp')) and
            accelerator['openmp'] == (backend in ('openmp', 'cuda-openmp')) and
            accelerator['host_threads'] == (m['threads'] if backend in ('openmp', 'cuda-openmp') else 1),
            'backend/thread identity mismatch')
    if backend == 'cuda-openmp':
        co = accelerator['cooperative']
        require(co['experimental'] is True and co['cuda_input_particles'] > 0 and
                co['openmp_input_particles'] > 0 and co['openmp_slices'] > 0,
                'Fe event did not execute both endpoints')
        require(co['openmp_workspace_bytes'] <= 256*2**20,
                'unexpected OpenMP workspace growth')
        require(s['hadronic_models']['high_energy']['name'] == 'SIBYLL-2.3d' and
                s['hadronic_models']['low_energy']['name'] == 'FLUKA',
                'hadronic model mismatch')
    require(s['gpu_physics_source'] == 'proposal-native' and
            s['rng_domain_version'] == 2 and
            str(s['physics_alignment_revision']) == '2026-09-06',
            'physics/RNG identity mismatch')
    require(s['process_registry']['accepted'] is True and
            s['backend_lifecycle']['reused'] is False, 'N1 lifecycle mismatch')
    require(s['gpu_particles'] >= 0, 'negative accelerator activity')
    require(s['queue_overflows'] == 0 and s['radio']['fixed_point_overflows'] == 0 and
            s['profile']['fixed_point_overflows'] == 0 and
            s['profile']['invalid_records'] == 0, 'overflow/invalid accelerator records')
    require(s['cross_species']['final_pending_photons'] == 0 and
            s['cross_species']['final_pending_leptons'] == 0 and
            s['deferred_cpu_fallbacks_queued'] == s['deferred_cpu_fallbacks_flushed'],
            'pending accelerator/fallback work')
    native = s['proposal_native']
    require(native['proposal_version'] == '7.6.2' and
            native['cubic_interpolation_version'] == '0.1.5' and
            native['table_sha256'] == m['expected_native_hash'] and
            native['aux_sha256'] == m['expected_aux_hash'], 'native table identity mismatch')
    files = {}
    for path in sorted(output.rglob('*.parquet')):
        parquet = pq.ParquetFile(path)
        rows = 0
        for batch in parquet.iter_batches(batch_size=16384, use_threads=False):
            for column in batch.columns:
                require(column.null_count == 0, 'null parquet data: '+str(path))
                array = column.to_numpy(zero_copy_only=False)
                if array.dtype.kind in 'fiu':
                    require(np.isfinite(array).all(), 'nonfinite parquet data: '+str(path))
            rows += batch.num_rows
        require(rows == parquet.metadata.num_rows, 'parquet row count mismatch')
        files[str(path.relative_to(output))] = rows
    require(all(name in files for name in REQUIRED_PARQUET), 'required parquet missing')
    radio = {}
    for algorithm in ('CoREAS', 'ZHS'):
        config = yaml_load(output/algorithm/'config.yaml')
        expected = sum(int(o['number of bins']) for o in config['observers'].values())
        require(expected > 0 and files[algorithm+'/observers.parquet'] == expected,
                'radio row count/layout mismatch: '+algorithm)
        nonzero, peak = 0, 0.
        for batch in pq.ParquetFile(output/algorithm/'observers.parquet').iter_batches(
                columns=['Ex', 'Ey', 'Ez'], batch_size=16384, use_threads=False):
            for column in batch.columns:
                array = column.to_numpy()
                nonzero += int(np.count_nonzero(array))
                peak = max(peak, float(np.abs(array).max(initial=0.)))
        radio[algorithm] = {'nonzero_components': nonzero, 'peak_Vpm': peak,
                            'config_sha256': sha(output/algorithm/'config.yaml')}
    return {'policy': 'Fe56-100TeV-rng2-N1-cooperative16-v1',
            'files': files, 'primary': primary['shower_0'], 'radio': radio,
            'advanced_em_records': s['gpu_particles'],
            'cooperative': accelerator.get('cooperative'),
            'simulation_timing': timing['shower_0'],
            'output_signatures': signatures(output)}


def closed_signature(output):
    result = []
    closed = False
    for name in ('simulation_timing/summary.yaml', 'summary.yaml'):
        path = output/name
        try:
            st = path.stat()
            result.append((name, st.st_size, st.st_mtime_ns))
        except OSError:
            result.append((name, None, None))
    try:
        timing = yaml_load(output/'simulation_timing/summary.yaml')
        closed = set(timing) == {'shower_0'} and timing['shower_0']['closed'] is True
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        pass
    return closed, tuple(result)


def log_tail(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-65536))
            return stream.read().decode(errors='replace')
    except OSError:
        return ''


def attempt(root, backend, index, record, m, digest):
    folder = root/backend
    number = len(record['attempts'])+1
    label = f'event_{index:04d}.attempt_{number}'
    output, log_path = folder/label, folder/(label+'.log')
    telemetry_path = folder/(label+'.telemetry.jsonl')
    marker = folder/f'event_{index:04d}.json'
    if any(p.exists() for p in (output, log_path, telemetry_path)):
        raise SafetyStop('unrecorded existing attempt files; inspection needed: '+label)
    limits = m['memory'][backend]
    resource_check(root, limits)
    if backend in ('cuda', 'cuda-openmp'):
        gpu = gpu_state(required=True)
        if gpu['total_MiB']-gpu['used_MiB'] < m.get('gpu_min_free_MiB', 256):
            raise SafetyStop('GPU memory reserve exhausted before launch')
    if STOP.is_set():
        raise InterruptedError('stop requested before launch')
    require(sha(root/'manifest.json') == digest, 'manifest changed during campaign')
    require(sha(m['executables'][backend]['path']) == m['executables'][backend]['sha256'],
            'archived executable changed during campaign')
    for label, info in m['inputs'].items():
        path = info.get('path_by_backend', {}).get(backend, info.get('path'))
        if path is not None:
            require(sha(path) == info['sha256'], 'runtime/input changed: '+label)
    argv = command(m, backend, record['seed'], output)
    item = {'attempt': number, 'output': str(output), 'log': str(log_path),
            'telemetry': str(telemetry_path), 'command': argv, 'state': 'starting',
            'started_unix': time.time(), 'execution_host': os.uname().nodename}
    record['attempts'].append(item)
    record.update(state='running', complete=False)
    atomic_json(marker, record)
    proc = None
    started = time.monotonic()
    recent = deque(maxlen=16)
    peak, max_threads, samples = 0, 0, 0
    min_available = None
    last_gpu, last_tick_signature, idle_since = -10., None, None
    last_thread_sample = -2.
    try:
        with log_path.open('x') as log, telemetry_path.open('x') as telemetry:
            proc = subprocess.Popen(argv, env=environment(m, backend), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            item.update(pid=proc.pid, proc_start_ticks=proc_state(proc.pid)['start_ticks'],
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        state='running')
            atomic_json(marker, record)
            while proc.poll() is None:
                elapsed = time.monotonic()-started
                sample = proc_state(proc.pid)
                sample['elapsed_s'] = round(elapsed, 3)
                if elapsed - last_thread_sample >= 2:
                    thread_ticks = {}
                    for path in Path('/proc', str(proc.pid), 'task').glob('*/stat'):
                        try:
                            fields = path.read_text().rsplit(')', 1)[1].split()
                            thread_ticks[path.parent.name] = int(fields[11])+int(fields[12])
                        except (OSError, ValueError, IndexError):
                            continue
                    sample['per_thread_cpu_ticks'] = thread_ticks
                    last_thread_sample = elapsed
                available, free = resource_check(root, limits, sample['rss_bytes'])
                sample.update(available_bytes=available, disk_free_bytes=free)
                if backend in ('cuda', 'cuda-openmp') and elapsed-last_gpu >= 5:
                    sample['gpu'] = gpu_state()
                    last_gpu = elapsed
                    gpu = sample['gpu']
                    if 'total_MiB' in gpu and gpu['total_MiB']-gpu['used_MiB'] < m.get('gpu_min_free_MiB', 256):
                        raise SafetyStop('GPU memory reserve exhausted during event')
                peak = max(peak, sample['rss_bytes'])
                max_threads = max(max_threads, sample['threads'])
                min_available = available if min_available is None else min(min_available, available)
                samples += 1
                recent.append(sample)
                telemetry.write(json.dumps(sample, allow_nan=False)+'\n')
                telemetry.flush()
                if STOP.is_set():
                    raise InterruptedError('runner received termination signal')
                deadline = m.get('event_timeout_seconds', m.get('timeout_seconds', 14400))
                if elapsed > min(float(deadline), 14400):
                    raise EventTimeout('event exceeded four-hour/configured deadline')
                closed, output_signature = closed_signature(output)
                current = (sample['cpu_ticks'], output_signature)
                if not closed or sample['cpu_ticks'] is None:
                    last_tick_signature, idle_since = None, None
                elif current != last_tick_signature:
                    last_tick_signature, idle_since = current, time.monotonic()
                elif time.monotonic()-idle_since >= 120:
                    raise EventTimeout('idle teardown for 120s after shower closed')
                STOP.wait(.25)
            item['returncode'] = proc.returncode
            item['simulation_process_wall_s'] = time.monotonic()-started
            if proc.returncode:
                if OOM.search(log_tail(log_path)) or proc.returncode == -signal.SIGKILL:
                    raise SafetyStop('child memory failure/SIGKILL; inspect log and cgroup')
                raise RuntimeError('child exited with code '+str(proc.returncode))
        validation_started = time.monotonic()
        try:
            validated = inspect_output(output, backend, record['seed'], m)
        finally:
            item['validation_s'] = time.monotonic()-validation_started
        item.update(state='complete', validation_unix=time.time())
        record.update(state='complete', complete=True, output=str(output), integrity=validated)
    except BaseException as error:
        kind = ('timeout' if isinstance(error, EventTimeout) else
                'safety_stop' if isinstance(error, SafetyStop) else
                'interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else
                'failure')
        if kind == 'failure' and OOM.search(log_tail(log_path)):
            kind = 'safety_stop'
        item.update(state='failed', error_kind=kind, error=repr(error))
        record.update(state='failed', complete=False)
        if kind in ('safety_stop', 'interrupted'):
            STOP.set()
        if isinstance(error, SystemExit):
            raise
    finally:
        if proc is not None:
            stop_child(proc)
            item['returncode'] = proc.returncode
            item.setdefault('simulation_process_wall_s', time.monotonic()-started)
        item.update(wall_s=time.monotonic()-started, finished_unix=time.time(),
                    peak_rss_bytes=peak, max_threads=max_threads,
                    minimum_available_bytes=min_available, memory_sample_count=samples,
                    recent_samples=list(recent))
        atomic_json(marker, record)
    return record


def existing_record(path, index, seed, backend, digest, validate_outputs=True):
    record = json.loads(path.read_text())
    require(record['index'] == index and record['seed'] == seed and
            record['backend'] == backend and record['events'] == 1 and
            record['manifest_sha256'] == digest, 'existing marker identity mismatch: '+str(path))
    if record.get('complete'):
        require(record['state'] == 'complete', 'inconsistent complete marker')
        if validate_outputs:
            require(signatures(Path(record['output'])) == record['integrity']['output_signatures'],
                    'accepted output files changed: '+str(path))
    if record['state'] == 'running':
        item = record['attempts'][-1]
        pid = item.get('pid')
        if not pid:
            raise SafetyStop('previous launch has no recorded PID; inspection needed: '+str(path))
        if pid and item.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            state = proc_state(pid)
            if state['start_ticks'] is not None and state['start_ticks'] == item.get('proc_start_ticks') and state.get('state') != 'Z':
                raise SafetyStop('previous child still running; refusing duplicate seed')
        # Never replace an unvalidated process crash with a new random seed.
        item.update(state='failed', error_kind='abandoned',
                    error='runner stopped before recording a verified outcome')
        record.update(state='failed', complete=False)
        atomic_json(path, record)
    return record


def progress(root, backend, m, digest, validate_outputs=True):
    completed, failed, missing = [], [], []
    for index, seed in enumerate(m['seeds']):
        path = root/backend/f'event_{index:04d}.json'
        if not path.exists():
            missing.append(index)
            continue
        record = existing_record(path, index, seed, backend, digest, validate_outputs)
        (completed if record.get('complete') else failed).append(index)
    result = {'backend': backend, 'manifest_sha256': digest, 'target_events': len(m['seeds']),
              'complete_events': len(completed), 'complete_indices': completed,
              'failed_indices': failed, 'missing_indices': missing,
              'complete': len(completed) == len(m['seeds']), 'updated_unix': time.time()}
    atomic_json(root/(backend+'_progress.json'), result)
    if result['complete'] and validate_outputs:
        atomic_json(root/('simulation_complete_'+backend+'.json'), result)
    return result


def retry_allowed(record):
    attempts = record['attempts']
    return not attempts or (len(attempts) == 1 and attempts[-1].get('error_kind') == 'timeout')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--backend', choices=('cuda-openmp',))
    parser.add_argument('--limit', type=int, default=500,
                        help='maximum number of manifest indices to visit from start-index')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.backend:
        parser.error('--backend is required except with --self-test')
    require(0 <= args.start_index < 500 and 0 <= args.limit <= 500, 'invalid index/limit')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root/args.backend).mkdir(exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    with (root/(args.backend+'_runner.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m, digest, inputs = read_manifest(root, args.backend)
        resource_check(root, m['memory'][args.backend])
        if args.backend in ('cuda', 'cuda-openmp'):
            gpu_state(required=True)
        # This is a separate audit record, never a manifest rewrite.
        atomic_json(root/(args.backend+'_runtime_audit.json'), {
            'manifest_sha256': digest, 'runner_sha256': sha(__file__), 'verified_inputs': inputs,
            'backend': args.backend, 'execution_host': os.uname().nodename,
            'started_unix': time.time(), 'runner_pid': os.getpid(),
            'available_affinity': sorted(os.sched_getaffinity(0))})
        for index in range(args.start_index, min(500, args.start_index+args.limit)):
            if STOP.is_set():
                break
            seed = m['seeds'][index]
            marker = root/args.backend/f'event_{index:04d}.json'
            if marker.exists():
                record = existing_record(marker, index, seed, args.backend, digest)
            else:
                record = {'index': index, 'seed': seed, 'events': 1, 'backend': args.backend,
                          'manifest_sha256': digest, 'runner_sha256': sha(__file__),
                          'executable_sha256': m['executables'][args.backend]['sha256'],
                          'complete': False, 'state': 'pending', 'attempts': []}
            while not record.get('complete'):
                if not retry_allowed(record):
                    break
                record = attempt(root, args.backend, index, record, m, digest)
                if STOP.is_set():
                    break
            status = progress(root, args.backend, m, digest, validate_outputs=False)
            print(json.dumps({'backend': args.backend, 'index': index, 'seed': seed,
                              'state': record['state'], 'complete_events': status['complete_events'],
                              'failed_events': len(status['failed_indices'])}), flush=True)
        status = progress(root, args.backend, m, digest)
        print(json.dumps(status), flush=True)
        return 130 if STOP.is_set() else (2 if status['failed_indices'] else 0)


def self_test():
    """Host-only tests; no CORSIKA binary or GPU is invoked."""
    import unittest
    import pyarrow as pa

    class Tests(unittest.TestCase):
        def test_atomic_and_resume_identity(self):
            with tempfile.TemporaryDirectory(prefix='fe_runner_test_') as temp:
                root = Path(temp)
                output = root/'output'
                output.mkdir()
                (output/'proof').write_text('unchanged')
                record = {'index': 0, 'seed': 85000001, 'backend': 'cuda', 'events': 1,
                          'manifest_sha256': 'frozen', 'complete': True, 'state': 'complete',
                          'output': str(output), 'integrity': {'output_signatures': signatures(output)}}
                path = root/'event.json'
                atomic_json(path, record)
                self.assertTrue(existing_record(path, 0, 85000001, 'cuda', 'frozen')['complete'])
                with self.assertRaises(ValueError):
                    existing_record(path, 0, 85000002, 'cuda', 'frozen')
                with self.assertRaises(ValueError):
                    existing_record(path, 0, 85000001, 'cuda', 'changed')
                (output/'proof').write_text('altered')
                with self.assertRaises(ValueError):
                    existing_record(path, 0, 85000001, 'cuda', 'frozen')

        def test_one_event_and_threads(self):
            m = {'executables': {b: {'path': '/no/simulation'} for b in ('cuda', 'openmp')},
                 'common_argv': ['-p', '1000260560', '-E', '100000'], 'threads': 128,
                 'env_by_backend': {'cuda': {}, 'openmp': {}},
                 'cpu_affinity_by_backend': {'openmp': list(range(128))}}
            for b in ('cuda', 'openmp'):
                args = command(m, b, 85000001, Path('/no/output'))
                self.assertEqual(args[args.index('-N')+1], '1')
                self.assertEqual(args.count('-N'), 1)
                self.assertEqual(args[args.index('-s')+1], '85000001')
                self.assertEqual(environment(m, b)['OMP_NUM_THREADS'], '128' if b == 'openmp' else '1')

        def test_cooperative16_command(self):
            m = {'executables': {'cuda-openmp': {'path': '/no/simulation'}},
                 'common_argv': ['-Z', '26', '-A', '56', '-E', '100000'],
                 'threads': 16, 'env_by_backend': {'cuda-openmp': {}}}
            args = command(m, 'cuda-openmp', 85000001, Path('/no/output'))
            self.assertEqual(args[args.index('--kokkos-execution')+1], 'cuda-openmp')
            self.assertEqual(args[args.index('--kokkos-num-threads')+1], '16')
            self.assertEqual(environment(m, 'cuda-openmp')['OMP_THREAD_LIMIT'], '16')
            self.assertEqual(args[args.index('--hadronic-workers')+1], '1')
            self.assertEqual(args[args.index('-N')+1], '1')

        def test_no_false_complete(self):
            with tempfile.TemporaryDirectory(prefix='fe_runner_test_') as temp:
                root = Path(temp)
                (root/'cuda').mkdir()
                result = progress(root, 'cuda', {'seeds': list(range(85000001, 85000501))}, 'frozen')
                self.assertFalse(result['complete'])
                self.assertEqual(len(result['missing_indices']), 500)
                self.assertFalse((root/'simulation_complete_cuda.json').exists())

        def test_retry_bound_and_live_child(self):
            self.assertTrue(retry_allowed({'attempts': []}))
            self.assertTrue(retry_allowed({'attempts': [{'error_kind': 'timeout'}]}))
            for kind in ('failure', 'safety_stop', 'interrupted', 'abandoned'):
                self.assertFalse(retry_allowed({'attempts': [{'error_kind': kind}]}))
            self.assertFalse(retry_allowed({'attempts': [{'error_kind': 'timeout'}]*2}))
            with tempfile.TemporaryDirectory(prefix='fe_runner_test_') as temp:
                path = Path(temp)/'event.json'
                atomic_json(path, {'index': 0, 'seed': 85000001, 'backend': 'cuda', 'events': 1,
                    'manifest_sha256': 'frozen', 'complete': False, 'state': 'running',
                    'attempts': [{'pid': os.getpid(),
                                  'proc_start_ticks': proc_state(os.getpid())['start_ticks'],
                                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}]})
                with self.assertRaises(SafetyStop):
                    existing_record(path, 0, 85000001, 'cuda', 'frozen')

        def test_parquet_integrity_and_lifecycle(self):
            with tempfile.TemporaryDirectory(prefix='fe_runner_test_') as temp:
                root = Path(temp)
                m = {'threads': 128, 'expected_native_hash': 'native', 'expected_aux_hash': 'aux'}
                statistics = {
                    'accelerator': {'backend': 'cuda', 'gpu': True, 'openmp': False, 'host_threads': 1},
                    'gpu_physics_source': 'proposal-native', 'rng_domain_version': 2,
                    'physics_alignment_revision': '2026-09-06',
                    'process_registry': {'accepted': True}, 'backend_lifecycle': {'reused': False},
                    'gpu_particles': 0, 'queue_overflows': 0,
                    'radio': {'fixed_point_overflows': 0},
                    'profile': {'fixed_point_overflows': 0, 'invalid_records': 0},
                    'cross_species': {'final_pending_photons': 0, 'final_pending_leptons': 0},
                    'deferred_cpu_fallbacks_queued': 0, 'deferred_cpu_fallbacks_flushed': 0,
                    'proposal_native': {'proposal_version': '7.6.2', 'cubic_interpolation_version': '0.1.5',
                                        'table_sha256': 'native', 'aux_sha256': 'aux'}}
                fixtures = {
                    'summary.yaml': {'showers': 1, 'seed': 85000001},
                    'primary/summary.yaml': {'shower_0': {'pdg': 1000260560, 'total_energy': 1e5}},
                    'simulation_timing/summary.yaml': {'shower_0': {'closed': True}},
                    'gpu_em/summary.yaml': {'shower_0': {'complete': True, 'status': 'complete',
                                                       'statistics': statistics}}}
                for name, data in fixtures.items():
                    path = root/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(data))
                table = pa.table({'Ex': [0.], 'Ey': [0.], 'Ez': [0.]})
                for name in REQUIRED_PARQUET:
                    path = root/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, path)
                for algorithm in ('CoREAS', 'ZHS'):
                    (root/algorithm/'config.yaml').write_text(yaml.safe_dump(
                        {'observers': {'one': {'number of bins': 1}}}))
                result = inspect_output(root, 'cuda', 85000001, m)
                self.assertEqual(result['radio']['CoREAS']['nonzero_components'], 0)
                pq.write_table(pa.table({'Ex': [float('nan')], 'Ey': [0.], 'Ez': [0.]}),
                               root/'CoREAS/observers.parquet')
                with self.assertRaises(ValueError):
                    inspect_output(root, 'cuda', 85000001, m)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (SafetyStop, ValueError, OSError) as error:
        print(type(error).__name__+': '+str(error), file=sys.stderr, flush=True)
        sys.exit(3)
