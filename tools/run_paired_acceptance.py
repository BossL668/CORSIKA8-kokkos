#!/usr/bin/env python3
"""Resumable beta5 CUDA/OpenMP cohorts against an archived scalar seed list.

Run under a memory-limited systemd user service. One child, one shower and one
execution backend at a time. No failed seed is silently replaced or discarded.
This is campaign orchestration, not part of the shower implementation.
"""
import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


class MemorySafetyStop(RuntimeError):
    """A memory incident stops the entire campaign, not only one seed."""


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return yaml.load(Path(path).read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def memory(pid=None):
    info = dict(line.replace(':', '').split()[:2] for line in Path('/proc/meminfo').read_text().splitlines())
    rss = 0.
    if pid:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss = float(line.split()[1]) / 1024
        except FileNotFoundError:
            pass
    return float(info['MemAvailable']) / 1024, rss


def common_argv(command):
    command = command[1:]
    result = []
    i = 0
    while i < len(command):
        if command[i] in ('-f', '--filename', '-s', '--seed', '-N', '--nevent'):
            i += 2
        else:
            result.append(command[i])
            i += 1
    return result


def inspect_reference(item):
    order, root = item
    root = Path(root)
    p = json.loads((root / 'validation_provenance.json').read_text())
    summary = load(root / 'summary.yaml')
    primary = load(root / 'primary/summary.yaml')['shower_0']
    timing = load(root / 'simulation_timing/summary.yaml')['shower_0']
    if p['events'] != 1 or summary['showers'] != 1 or not timing['closed']:
        raise ValueError(f'incomplete scalar reference: {root}')
    if p['backend'] != 'proposal' or summary['seed'] != p['seed']:
        raise ValueError(f'wrong scalar backend/seed: {root}')
    if primary['pdg'] != 2212 or primary['total_energy'] != 1000:
        raise ValueError(f'wrong primary: {root}')
    if shlex.split(load(root / 'config.yaml')['args']) != p['command']:
        raise ValueError(f'command provenance mismatch: {root}')
    return {'index': order, 'seed': p['seed'], 'cpu_root': str(root),
            'cpu_wall_ms': timing['wall_time_ms'], 'primary': primary,
            'physics': p['physics'], 'common_argv': common_argv(p['command']),
            'cpu_executable_sha256': p['executable']['sha256'],
            'antenna_sha256': p['antenna_file']['sha256'],
            'fluka_sha256': p['flupro']['sha256'],
            'raw_profile_present': (root / 'profile/profile.parquet').is_file(),
            'raw_radio_present': (root / 'CoREAS/observers.parquet').is_file()}


def prepare(args):
    path = args.output / 'campaign.json'
    if path.exists():
        raise ValueError('campaign already prepared; use preflight/run to resume')
    reference = args.reference.resolve()
    comparison = json.loads((reference / 'ensemble_comparison_2000/comparison.json').read_text())
    sources = comparison['proposal']['sources']
    if comparison['proposal']['events'] != 2000 or len(sources) != 2000:
        raise ValueError('expected exactly 2000 one-shower scalar references')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(inspect_reference, enumerate(sources)))
    if len({r['seed'] for r in records}) != 2000:
        raise ValueError('duplicate CPU seeds')
    for key in ('physics', 'common_argv', 'cpu_executable_sha256', 'antenna_sha256', 'fluka_sha256'):
        if any(r[key] != records[0][key] for r in records):
            raise ValueError(f'heterogeneous scalar references: {key}')
    command = records[0]['common_argv']
    antenna = Path(command[command.index('--antenna-file') + 1])
    if digest(antenna) != records[0]['antenna_sha256']:
        raise ValueError('antenna contents changed')
    if digest(Path(os.environ['FLUPRO']) / 'libflukahp.a') != records[0]['fluka_sha256']:
        raise ValueError('FLUKA library changed')
    installations = {}
    for backend in ('cuda', 'openmp'):
        exe = args.install.resolve() / backend / 'bin/c8_air_shower'
        installations[backend] = {'path': str(exe), 'sha256': digest(exe)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'reference').mkdir(exist_ok=True)
    for rel in ('ensemble_comparison_2000/per_shower_observables.csv',
                'ensemble_comparison_2000/curve_comparison.csv',
                'ensemble_comparison_2000/comparison.json',
                'geomagnetic_pulse_validation/per_shower_features.csv',
                'geomagnetic_radial_validation/per_shower_radius_features.csv',
                'runtime_distribution_analysis_2000/single_event_runtimes.csv'):
        source = reference / rel
        if source.is_file():
            target = args.output / 'reference' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    save(path, {'schema_version': 1, 'created_utc': now(), 'reference_analysis': str(reference),
                'reference_kind': 'archived beta4 P2.1 scalar PROPOSAL + CPU CoREAS/ZHS',
                'events_per_backend': 2000, 'executables': installations,
                'physics': records[0]['physics'], 'common_argv': command,
                'antenna_sha256': digest(antenna), 'openmp_threads': args.threads,
                'gpu_memory_fraction': 0.70, 'gpu_min_batch': 4096,
                'raw_cpu_profiles_available': sum(r['raw_profile_present'] for r in records),
                'raw_cpu_radio_available': sum(r['raw_radio_present'] for r in records),
                'events': records, 'analysis_tools': str(args.analysis_tools.resolve()),
                'analysis_tools_sha256': digest(args.analysis_tools / 'compare_ensembles.py'),
                'runner_sha256': digest(__file__),
                'full_raw_cpu_acceptance_pending': not all(r['raw_profile_present'] and r['raw_radio_present'] for r in records)})
    print('Prepared 2000 unique CPU seeds; CPU raw profiles locally present:',
          sum(r['raw_profile_present'] for r in records), flush=True)


def make_command(manifest, event, backend, output):
    command = [manifest['executables'][backend]['path'], *manifest['common_argv'],
               '-N', '1', '-f', str(output), '--seed', str(event['seed']),
               '--em-backend', 'kokkos', '--radio-backend', 'kokkos',
               '--gpu-physics-source', 'proposal-native', '--gpu-min-batch', '4096',
               '--gpu-memory-fraction', '0.70', '--hadronic-workers', '1']
    command += (['--kokkos-num-threads', str(manifest['openmp_threads'])] if backend == 'openmp'
                else ['--kokkos-device', '0'])
    return command


def child_environment(backend, threads):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads if backend == 'openmp' else 1),
               OMP_THREAD_LIMIT=str(threads if backend == 'openmp' else 1),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if backend == 'openmp':
        env.update(OMP_PROC_BIND='spread', OMP_PLACES='threads')
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def stop_child(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def finite_array(array):
    if array.null_count:
        return False
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        return finite_array(array.values)
    if pa.types.is_floating(array.type):
        return np.isfinite(array.to_numpy(zero_copy_only=False)).all()
    return True


def validate_output(output, event, manifest, compare, backend):
    summary = load(output / 'summary.yaml')
    if summary['seed'] != event['seed'] or summary['showers'] != 1:
        raise ValueError('seed/shower count mismatch')
    showers, integrity = compare.validate_completion(output, True, 'proposal-native', ('unsupported_geometry',))
    if showers != (0,):
        raise ValueError('wrong output shower IDs')
    primary = load(output / 'primary/summary.yaml')['shower_0']
    for key, value in event['primary'].items():
        if isinstance(value, (int, float)) and not np.isclose(primary[key], value, rtol=1e-12, atol=1e-15):
            raise ValueError(f'primary differs from scalar: {key}')
    expected_observers = manifest.get('reference_observer_layout_sha256')
    if expected_observers is None:
        expected_observers = compare.observer_layout_fingerprint(Path(event['cpu_root']))
    if compare.observer_layout_fingerprint(output) != expected_observers:
        raise ValueError('radio observer configuration differs from CPU')
    expected_physics = manifest.get('reference_physics_configuration')
    if expected_physics is None:
        expected_physics = compare.canonical_physics_configuration(Path(event['cpu_root']))
    if compare.canonical_physics_configuration(output) != tuple(expected_physics):
        raise ValueError('physical CLI differs from scalar reference')
    config = load(output / 'gpu_em/config.yaml')
    if config['execution_space'].lower() != backend or config['radio_backend'] != 'kokkos':
        raise ValueError('execution/radio backend mismatch')
    checked = {}
    required = ('profile/profile.parquet', 'production_profile/profile.parquet',
                'energyloss/dEdX.parquet', 'particles/particles.parquet',
                'CoREAS/observers.parquet', 'ZHS/observers.parquet')
    paths = sorted(set(required) | {str(f.relative_to(output)) for f in output.rglob('*.parquet')})
    for rel in paths:
        f = pq.ParquetFile(output / rel)
        count = 0
        for batch in f.iter_batches(batch_size=32, use_threads=False):
            if not all(finite_array(a) for a in batch.columns):
                raise ValueError(f'non-finite/null data: {rel}')
            count += batch.num_rows
        if count != f.metadata.num_rows:
            raise ValueError(f'Parquet row count mismatch: {rel}')
        checked[rel] = {'rows': count, 'sha256': digest(output / rel)}
    gpu = load(output / 'gpu_em/summary.yaml')['shower_0']
    native = gpu['statistics']['proposal_native']
    identity_path = output.parents[2] / 'physics_identity.json'
    if identity_path.exists():
        expected = json.loads(identity_path.read_text())
        if any(native[k] != v for k, v in expected.items()):
            raise ValueError('native/auxiliary table identity changed during campaign')
    timing = load(output / 'simulation_timing/summary.yaml')['shower_0']
    return {'integrity': integrity, 'parquet': checked,
            'simulation_wall_ms': timing['wall_time_ms'], 'native': native,
            'observer_layout_sha256': compare.observer_layout_fingerprint(output)}


def run_one(args, manifest, event, backend, compare):
    slot = args.output / backend / f"event_{event['index']:04d}"
    slot.mkdir(parents=True, exist_ok=True)
    result = slot / 'result.json'
    if result.exists():
        old = json.loads(result.read_text())
        if old.get('status') == 'complete':
            return True
        # Keep failed seeds visible. A subsequent repair must be explicit.
        if old.get('status') == 'failed':
            return False
    output = slot / 'shower'
    if output.exists():
        output.rename(slot / f'interrupted_{time.time_ns()}')
    exe = manifest['executables'][backend]
    if digest(exe['path']) != exe['sha256']:
        raise RuntimeError('installed executable changed; refusing to mix versions')
    command = make_command(manifest, event, backend, output)
    available, _ = memory()
    if available < 3072:
        raise RuntimeError(f'RAM start guard: only {available:.0f} MiB available; resume later')
    if shutil.disk_usage(args.output).free < 15 * 1024**3:
        raise RuntimeError('disk guard: less than 15 GiB free')
    state = {'status': 'running', 'backend': backend, 'index': event['index'],
             'seed': event['seed'], 'started_utc': now(), 'command': command,
             'executable_sha256': exe['sha256'], 'cpu_reference': event['cpu_root'],
             'execution_runner_sha256': digest(__file__)}
    save(result, state)
    proc = None
    start = time.monotonic()
    try:
        with (slot / 'shower.log').open('w') as log, (slot / 'resources.csv').open('w') as telemetry:
            writer = csv.writer(telemetry)
            writer.writerow(('elapsed_s', 'host_available_mib', 'process_rss_mib', 'process_cpu_seconds', 'threads'))
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=child_environment(backend, manifest['openmp_threads']), start_new_session=True)
            save(args.output / 'active.json', {**state, 'pid': proc.pid})
            peak = 0.
            while proc.poll() is None:
                avail, rss = memory(proc.pid)
                peak = max(peak, rss)
                try:
                    stat = Path(f'/proc/{proc.pid}/stat').read_text().split()
                    cpu = (int(stat[13]) + int(stat[14])) / os.sysconf('SC_CLK_TCK')
                    threads = stat[19]
                except (FileNotFoundError, ProcessLookupError):
                    cpu, threads = 0, 0
                writer.writerow((time.monotonic() - start, avail, rss, cpu, threads))
                telemetry.flush()
                if avail < 1536 or rss > 4300:
                    raise MemorySafetyStop('host memory guard stopped this event')
                if time.monotonic() - start > 1200:
                    raise RuntimeError('event exceeded 20 minute diagnostic time limit')
                if backend == 'cuda' and int(time.monotonic() - start) % 5 == 0:
                    smi = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
                        '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    with (slot / 'gpu.csv').open('a') as gpu_log:
                        gpu_log.write(smi.stdout)
                time.sleep(1)
        state.update(returncode=proc.returncode, process_wall_s=time.monotonic() - start,
                     peak_rss_mib=peak)
        if proc.returncode:
            raise RuntimeError(f'shower returned {proc.returncode}; see shower.log')
        validation = validate_output(output, event, manifest, compare, backend)
        p = {'schema_version': 1, 'backend': 'kokkos-' + backend, 'events': 1,
             'seed': event['seed'], 'command': command,
             'command_sha256': hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest(),
             'executable': exe, 'gpu_physics_source': 'proposal-native',
             'table': {'sha256': validation['native']['table_sha256']},
             'antenna_file': {'path': command[command.index('--antenna-file') + 1], 'sha256': manifest['antenna_sha256']}}
        save(output / 'validation_provenance.json', p)
        state.update(status='complete', finished_utc=now(), validation=validation)
        save(result, state)
        print(f"{backend} {event['index'] + 1}/2000 seed={event['seed']} complete wall={state['process_wall_s']:.2f}s", flush=True)
        return True
    except Exception as exc:
        stop_child(proc)
        state.update(status='failed', finished_utc=now(), error=traceback.format_exc())
        save(result, state)
        print(f"FAILED {backend} index={event['index']} seed={event['seed']}: {state['error']}", flush=True)
        if isinstance(exc, MemorySafetyStop):
            save(args.output / 'runner_memory_stop.json', state)
            raise
        return False
    finally:
        stop_child(proc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'preflight', 'run'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--install', type=Path)
    p.add_argument('--analysis-tools', type=Path)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--backends', choices=('cuda,openmp', 'cuda', 'openmp'), default='cuda,openmp',
                   help='Only execute these cohorts on this machine; never implicitly start the other backend')
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            prepare(args)
            return
        manifest = json.loads((args.output / 'campaign.json').read_text())
        if (args.output / 'runner_memory_stop.json').exists():
            raise MemorySafetyStop('previous memory incident requires diagnosis before resuming')
        backends = args.backends.split(',')
        if digest(Path(manifest['analysis_tools']) / 'compare_ensembles.py') != manifest['analysis_tools_sha256']:
            raise RuntimeError('analysis implementation changed')
        sys.path.insert(0, manifest['analysis_tools'])
        import compare_ensembles as compare
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        def interrupted(signum, frame):
            raise SystemExit(f'stopped by signal {signum}; current seed will restart on resume')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        pilot = manifest['events'][:3]
        for backend in backends:
            for event in pilot:
                if not run_one(args, manifest, event, backend, compare):
                    raise RuntimeError('preflight failed; bulk production not started')
        keys = ('table_sha256', 'aux_sha256', 'proposal_version', 'cubic_interpolation_version')
        identities = []
        for backend in backends:
            for event in pilot:
                path = args.output / backend / f"event_{event['index']:04d}" / 'result.json'
                native = json.loads(path.read_text())['validation']['native']
                identities.append({k: native[k] for k in keys})
        if any(i != identities[0] for i in identities):
            raise RuntimeError('preflight tables differ across seeds or backends')
        existing_identity = args.output / 'physics_identity.json'
        if existing_identity.exists() and json.loads(existing_identity.read_text()) != identities[0]:
            raise RuntimeError('preflight physics identity differs from the archived reference')
        save(existing_identity, identities[0])
        save(args.output / 'preflight.json', {'status': 'passed', 'backends': backends, 'events_each': 3, 'time': now()})
        if args.mode == 'preflight':
            return
        for backend in backends:
            failures = 0
            for event in manifest['events']:
                success = run_one(args, manifest, event, backend, compare)
                failures = 0 if success else failures + 1
                if failures >= 3:
                    raise RuntimeError('three consecutive failures; production paused for diagnosis')
        results = [json.loads(f.read_text()) for b in backends for f in (args.output / b).glob('event_*/result.json')]
        counts = {b: sum(r['backend'] == b and r['status'] == 'complete' for r in results) for b in backends}
        save(args.output / 'production_status.json', {'finished_utc': now(), 'counts': counts,
             'status': 'complete' if all(v == 2000 for v in counts.values()) else 'incomplete',
             'physics_acceptance': 'pending analysis; generation completion is not physics acceptance',
             'cpu_raw_data_needed_for_full_acceptance': manifest['full_raw_cpu_acceptance_pending']})
        analyzer = Path(__file__).with_name('analyze_paired_acceptance.py')
        if analyzer.is_file() and len(backends) == 2:
            subprocess.run([sys.executable, str(analyzer), '--campaign', str(args.output)], check=True)


if __name__ == '__main__':
    main()
