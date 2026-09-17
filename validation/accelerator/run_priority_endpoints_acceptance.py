#!/usr/bin/env python3
"""Isolated CPU/GPU-priority regression and alternating same-machine pilots.

No installation or production restart. Uses existing PROPOSAL caches. Every
completed sample is retained, including slow and telemetry-invalid samples.
"""
import argparse
import hashlib
import json
import math
from decimal import Decimal
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
import yaml

from compare_backend_build_outputs import compare_outputs
from run_adaptive_cooperative_acceptance import check_coalesced_fallbacks
from run_overlap_guarded import compute_gpu_pids


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def wait_case_resources(idle_seconds, timeout, gpu_query, workflow_query,
                        publish, clock=time.monotonic, sleep=time.sleep):
    """Optional per-case CPU-workflow + GPU isolation; errors are not idle."""
    deadline = clock() + timeout
    idle_since = None
    while True:
        try:
            gpu = gpu_query()
            workers = workflow_query()
            safe = gpu is not None and workers is not None and not gpu and not workers
            record = dict(gpu_pids=sorted(gpu) if gpu is not None else None,
                          workflow_pids=workers, safe=safe)
        except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
            record = dict(safe=False, observation_error=repr(error))
            safe = False
        now = clock()
        idle_since = (idle_since if idle_since is not None else now) if safe else None
        idle = now-idle_since if idle_since is not None else 0.
        record.update(idle_seconds=idle, wait_remaining_seconds=max(0., deadline-now),
                      launches_started=False, signals_sent=0)
        publish(record)
        if now >= deadline:
            raise TimeoutError('CPU workflow/GPU did not remain idle; no case launched')
        if idle_since is not None and idle >= idle_seconds:
            return record
        sleep(min(2., deadline-now))


def put(cmd, key, value):
    if key in cmd:
        cmd[cmd.index(key)+1] = str(value)
    else:
        cmd.extend([key, str(value)])


def positive_fe_energy(value):
    try:
        energy = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError('Fe energy must be a finite positive GeV value') from error
    if not math.isfinite(energy) or energy <= 0:
        raise argparse.ArgumentTypeError('Fe energy must be a finite positive GeV value')
    return energy


def fe_timing_family(energy=None):
    if energy is None:
        return 'Fe100TeV'  # preserve historical default command and labels
    value = Decimal(str(positive_fe_energy(energy))) / Decimal(1000)
    label = format(value, 'f')
    if '.' in label:
        label = label.rstrip('0').rstrip('.')
    if len(label) > 32:
        label = format(value.normalize(), 'g')
    return 'Fe' + label + 'TeV'


def override_fe_energy(command, energy):
    if energy is not None:
        put(command, '-E', positive_fe_energy(energy))


def override_timing_primary(command, primary):
    """Pilot-only explicit override; never change regression or old templates."""
    if primary is None:
        return
    if primary != 'proton':
        raise ValueError('unsupported timing primary: ' + str(primary))
    if not command or len(command) % 2 != 1:
        raise ValueError('timing template must contain option/value pairs')
    # Remove nuclear selectors as well as an existing PID: the application
    # must not receive both a nucleus and the requested proton.
    selectors = {'-p', '--pdg', '-Z', '--charge', '-A', '--mass-number'}
    changed = [command[0]]
    for key, value in zip(command[1::2], command[2::2]):
        if key not in selectors:
            changed.extend((key, value))
    changed.extend(('-p', '2212'))
    command[:] = changed


def timing_family(energy=None, primary=None):
    family = fe_timing_family(energy)
    if primary is None:
        return family
    if primary != 'proton':
        raise ValueError('unsupported timing primary: ' + str(primary))
    return 'proton' + family[2:]


def validate_correctness_reuse(reference, identity):
    """Reuse completed, same-binary correctness, never timing/statistical evidence."""
    reference = reference.resolve(strict=True)
    files = []
    def read(relative):
        path = reference/relative
        data = path.read_bytes()
        files.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()))
        return json.loads(data)
    def require(condition, message):
        if not condition:
            raise ValueError('correctness reuse refused: ' + message)
    status = read('STATUS.json')
    gates = read('CORRECTNESS_GATES.json')
    prior = read('PROVENANCE.json')
    require(status.get('complete') is True, 'reference STATUS is incomplete')
    require(gates.get('passed') is True, 'reference correctness gates did not pass')
    require(not prior.get('correctness_reused_from'), 'use the original full regression, not a reuse chain')
    keys = ('binary_sha256', 'fixture_sha256', 'before_sha256', 'source', 'threads')
    for key in keys:
        require(key in identity and prior.get(key) == identity[key], 'identity mismatch: ' + key)
    for relative, key in (('binaries/c8_air_shower', 'binary_sha256'),
                          ('binaries/testKokkosCooperativeBackend', 'fixture_sha256')):
        path = reference/relative
        actual = sha(path)
        require(actual == identity[key], 'archived binary mismatch: ' + key)
        files.append(dict(path=str(path), sha256=actual))
    for mode in ('proposal', 'cuda', 'openmp'):
        require(read(mode+'-regression.json').get('pass') is True, mode+' regression did not pass')
        if mode != 'proposal':
            pair = []
            for label in ('before', 'after'):
                path = reference/(label+'-'+mode+'-trace.csv')
                digest = sha(path)
                require(path.stat().st_size > 0, 'empty decision tape: '+mode)
                pair.append(digest)
                files.append(dict(path=str(path), sha256=digest))
            require(pair[0] == pair[1], 'decision tape divergence: '+mode)
    for mode in ('cuda-openmp', 'openmp-cuda'):
        for label in ('fixture-'+mode, 'N32-'+mode):
            guard = read(label+'-guard/summary.json')
            require(guard.get('pass') is True and guard.get('returncode') == 0,
                    label+' guard did not pass')
        checked = read('N32-'+mode+'-CHECK.json')
        require(checked.get('events') == 32, mode+' lifecycle count is not 32')
        timing = checked.get('timing', {})
        showers = [key for key in timing if re.fullmatch(r'shower_\d+', key)]
        require(len(showers) == 32 and all(timing[key].get('closed') is True for key in showers),
                mode+' lifecycle outputs were not closed')
        accelerators = checked.get('accelerators', [])
        require(len(accelerators) == 32 and all(
            row.get('accelerator', {}).get('backend') == mode for row in accelerators),
            mode+' lifecycle accelerator evidence incomplete')
    return dict(passed=True, reference_run=str(reference), files=files,
                matching_identity={key: identity[key] for key in keys},
                scope='previous same-binary full regression, decision tapes and N32 lifecycle only; no reuse of performance or large-sample acceptance')


def checked_help(text):
    text = re.sub(r'(?m)^\[corsika:info    \(c8_air_shower\.cpp:\d+\)\] (?=Please cite)',
                  '[corsika:info (source line)] ', text)
    match = re.search(r'(?m)^  --kokkos-execution[^\n]*\n(?: {4,}\S[^\n]*\n)*', text)
    assert match, 'execution CLI option missing'
    return text[:match.start()]+text[match.end():], match.group()


def timing_modes(standalone, requested=None):
    modes = list(requested) if requested is not None else [standalone, 'cuda-openmp', 'openmp-cuda']
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('timing modes must be nonempty and unique')
    if any(mode not in ('cuda', 'openmp', 'cuda-openmp', 'openmp-cuda') for mode in modes):
        raise ValueError('unsupported timing mode')
    return modes


def timing_order(modes, index):
    if len(modes) == 2:
        # Rotating two entries and reversing odd trials cancels out, giving
        # A/B on every seed. Alternate the actual measured order instead.
        return modes[::-1] if index % 2 else list(modes)
    offset = index % len(modes)
    order = modes[offset:] + modes[:offset]
    return order[::-1] if index % 2 else order


def build_unit_complete(state):
    """Do not confuse RemainAfterExit's active(exited) with a live compiler."""
    required = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result',
                'ExecMainStatus', 'ExecMainCode', 'ExecMainStartTimestampMonotonic')
    if any(key not in state for key in required):
        raise RuntimeError('incomplete systemd build-unit state')
    if state['LoadState'] != 'loaded' or state['ActiveState'] == 'failed':
        raise RuntimeError('build unit missing or failed: ' + repr(state))
    if state['Result'] not in ('', 'success'):
        raise RuntimeError('build failed: ' + repr(state))
    pid = int(state['MainPID'])
    if pid > 0 or state['ActiveState'] in ('activating', 'deactivating', 'reloading'):
        return False
    terminal = ((state['ActiveState'], state['SubState']) in
                (('active', 'exited'), ('inactive', 'dead')))
    if not terminal:
        raise RuntimeError('unexpected terminal build-unit state: ' + repr(state))
    if (state['Result'] != 'success' or state['ExecMainStatus'] != '0' or
            state['ExecMainCode'] != '1' or int(state['ExecMainStartTimestampMonotonic']) <= 0):
        raise RuntimeError('build lacks a verified successful executed command: ' + repr(state))
    return True


def read_build_unit(unit):
    fields = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result',
              'ExecMainStatus', 'ExecMainCode', 'ExecMainStartTimestampMonotonic')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=' + ','.join(fields)], text=True,
                            capture_output=True, timeout=10)
    if result.returncode:
        raise RuntimeError('cannot verify build unit: ' + result.stderr.strip())
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def wait_for_build(unit, timeout, read_state=read_build_unit,
                   now=time.monotonic, sleep=time.sleep):
    if timeout <= 0:
        raise ValueError('build wait timeout must be positive')
    deadline = now() + timeout
    while True:
        state = read_state(unit)
        if build_unit_complete(state):
            return state
        remaining = deadline - now()
        if remaining <= 0:
            raise TimeoutError('build-unit observation timed out; no build stopped or restarted')
        sleep(min(3., remaining))


def check_cpu_primary(stats):
    c = stats['accelerator']['cooperative']
    policy = c['scheduling_policy']
    assert policy in ('cpu-primary-v1', 'cpu-primary-v2-simple',
                      'cpu-primary-v3-protected-front', 'cpu-primary-v4-resident-helper',
                      'cpu-primary-v5-blocking-helper', 'cpu-primary-v6-species-coalescing',
                      'cpu-primary-v7-tail-checkpoint')
    assert c['primary_endpoint'] == 'openmp'
    steps = sum(c['priority_endpoints'][e][k]['transport_records']
                for e in ('cuda', 'openmp') for k in ('photon', 'lepton'))
    assert steps == stats['gpu_particles'] == stats['profile']['steps']
    if policy == 'cpu-primary-v1':
        assert c['host_lepton_wave_limit'] == 64 and c['auxiliary_lepton_wave_limit'] == 16
        check_coalesced_fallbacks(stats)
        return None
    assert c['host_lepton_wave_limit'] == 1024 and c['auxiliary_lepton_wave_limit'] == 8
    assert c['host_profile_shards'] == c['host_profile_shard_bytes'] == 0
    if policy == 'cpu-primary-v2-simple':
        assert c['specified_fallback_batch_limit'] == 1
        return None
    assert c['auxiliary_endpoint'] == 'cuda'
    assert c['primary_reserve_semantics'] == 'one full native OpenMP arena per species; only surplus may seed auxiliary work'
    assert c['auxiliary_grant_semantics'] == 'new work >= min(CUDA minimum batch, capacity); owned tails and pressure spills still drain'
    assert c['specified_fallback_semantics'] == 'coordinator-owned batches; flush at 4096 or empty front without joining peer; protect EM products until front drains'
    helper_audit = None
    if policy in ('cpu-primary-v4-resident-helper', 'cpu-primary-v5-blocking-helper',
                  'cpu-primary-v6-species-coalescing', 'cpu-primary-v7-tail-checkpoint'):
        assert c['primary_input_semantics'] == 'resident queue first, then waiting input; same fill order as standalone OpenMP'
        assert c['auxiliary_continuation_call_limit'] == 4
        submitted = c['subshower_cuda_submissions']
        assert submitted == c['subshower_cuda_commits']
        continuations = c['subshower_cuda_autonomous_continuations']
        packets = c['cuda_completion_packets']
        maximum_calls = c['maximum_cuda_packet_calls']
        assert 0 <= continuations <= submitted
        assert packets == submitted - continuations
        assert 0 <= maximum_calls <= 4
        assert c['cuda_completion_mailbox_capacity'] == 4
        assert c['auxiliary_retention_semantics'] == 'retention budget plus at most one ordinary bounded result; checked before the next call'
        assert c['cuda_completion_retention_budget_bytes'] > 0
        assert c['cuda_completion_peak_bytes'] >= 0
        # Retention is checked before the NEXT ordinary call. One valid result
        # may cross the budget; completed/no-progress take precedence over the
        # memory stop. These counters do not certify a hard total-byte ceiling.
        # The external process guard separately monitors actual RSS.
        assert c['cuda_continuation_stop_order'] == [
            'completed', 'no_progress', 'memory', 'call_limit', 'handoff', 'time', 'disabled']
        stops = c['cuda_continuation_stops']
        assert len(stops) == 7 and all(value >= 0 for value in stops)
        assert sum(stops) == packets
        assert stops[5] == 0, 'CPU-primary fixed packets must not use adaptive time stopping'
        assert c['cuda_completion_buffer_delay_ms'] >= 0
        assert 0 <= c['subshower_cuda_foreground_packets'] <= packets
        assert 0 <= c['subshower_cuda_foreground_continuations'] <= continuations
        if submitted == 0:
            assert packets == continuations == maximum_calls == 0
        else:
            assert 0 < packets <= submitted and maximum_calls > 0
            assert maximum_calls + packets - 1 <= submitted <= maximum_calls * packets
            assert continuations <= 3 * packets
        helper_audit = dict(submitted_calls=submitted, committed_calls=c['subshower_cuda_commits'],
                            continuation_calls=continuations, completion_packets=packets,
                            maximum_packet_calls=maximum_calls, bounded_call_limit=4,
                            continuation_stops=stops, bounded_mailbox_capacity=4,
                            retention_peak_bytes=c['cuda_completion_peak_bytes'],
                            retention_budget_bytes=c['cuda_completion_retention_budget_bytes'],
                            retention_semantics=c['auxiliary_retention_semantics'],
                            hard_retention_byte_ceiling_certified=False,
                            rss_guard_evidence_required_separately=True)
        if policy in ('cpu-primary-v5-blocking-helper', 'cpu-primary-v6-species-coalescing',
                      'cpu-primary-v7-tail-checkpoint'):
            assert c['auxiliary_blocking_wait_enabled'] is True
            calls, elapsed = c['auxiliary_blocking_wait_calls'], c['auxiliary_blocking_wait_ms']
            assert isinstance(calls, int) and calls >= 0
            assert np.isfinite(elapsed) and elapsed >= 0
            if submitted:
                assert calls >= submitted, 'helper ran without its blocking resident waits'
            else:
                assert calls == 0 and elapsed == 0
            assert c['auxiliary_wait_semantics'] == 'reused blocking CUDA event on helper stream; default single-endpoint and GPU-primary fences unchanged; wait time overlaps primary work'
            helper_audit['blocking_wait'] = dict(enabled=True, calls=calls, host_wait_ms=elapsed,
                                                kernel_time=False, global_cuda_flags_changed=False)
        if policy in ('cpu-primary-v6-species-coalescing', 'cpu-primary-v7-tail-checkpoint'):
            coalesces = c['host_species_coalesces']
            assert type(coalesces) is int and coalesces >= 0
            assert coalesces <= c['subshower_openmp_epochs']
            assert c['host_species_maximum_deferrals'] == 8
            assert c['host_species_selection_semantics'] == 'defer sub-minimum species only while the other has a useful front; at most eight host choices; drain lone tails immediately'
            helper_audit['host_species_coalesces'] = coalesces
        if policy == 'cpu-primary-v7-tail-checkpoint':
            assert c['host_checkpoint_minimum_semantics'] == 'max(1, min(native minimum, initial batch / 4)); return surviving states without particle cuts'
            histogram = c['host_input_batch_log2_histogram']
            assert isinstance(histogram, list) and len(histogram) == 21
            assert all(type(n) is int and n >= 0 for n in histogram)
            assert sum(histogram) == c['subshower_openmp_epochs']
            helper_audit['host_input_batch_log2_histogram'] = histogram
            helper_audit['host_checkpoint_minimum_semantics'] = c['host_checkpoint_minimum_semantics']
    check_coalesced_fallbacks(stats)
    assert stats['cross_species']['final_pending_photons'] == 0
    assert stats['cross_species']['final_pending_leptons'] == 0
    queued = stats['deferred_cpu_fallbacks_queued']
    flushes = stats['deferred_cpu_fallback_flushes']
    maximum = stats['maximum_deferred_cpu_fallback_batch']
    if queued == 0:
        assert flushes == maximum == 0
    else:
        assert 0 < flushes <= queued and 0 < maximum <= min(4096, queued)
        assert maximum + flushes - 1 <= queued <= maximum * flushes
    # Counters prove no queued full/partial batch survived termination. They
    # do not give the chronology or exact size of the final individual flush.
    audit = dict(queued=queued, flushed=stats['deferred_cpu_fallbacks_flushed'],
                flushes=flushes, maximum_batch=maximum,
                full_batch_reached=maximum == 4096,
                at_least_one_partial_flush_required=queued % 4096 != 0,
                terminal_queues_empty=True, final_flush_size_not_recorded=True)
    if helper_audit is not None:
        audit['helper_continuation_audit'] = helper_audit
    return audit


def check_output(path, count, mode):
    timing = yaml.safe_load((path/'simulation_timing/summary.yaml').read_text())
    showers = [x for x in timing if re.fullmatch(r'shower_\d+', x)]
    assert len(showers) == count and all(timing[x]['closed'] for x in showers)
    result = dict(events=count, physical_arrays=[], timing=timing)
    for p in sorted(path.rglob('*.parquet')):
        rows = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=32768, use_threads=False):
            rows += len(batch)
            for column in batch.columns:
                assert column.null_count == 0, str(p)+' nulls'
                values = column.to_numpy(zero_copy_only=False)
                if values.dtype.kind in 'fci':
                    assert np.isfinite(values).all(), str(p)+' nonfinite values'
        result['physical_arrays'].append(dict(path=str(p.relative_to(path)), rows=rows))
    for p in sorted(path.rglob('*.npz')):
        with np.load(p) as data:
            assert all(np.isfinite(data[k]).all() for k in data.files if data[k].dtype.kind in 'fci')
        result['physical_arrays'].append(dict(path=str(p.relative_to(path))))
    assert result['physical_arrays'], 'missing arrays'
    if mode == 'proposal':
        return result
    gpu = yaml.safe_load((path/'gpu_em/summary.yaml').read_text())
    result['accelerators'] = []
    for name in showers:
        assert gpu[name]['complete']
        stats = gpu[name]['statistics']
        acc = stats['accelerator']
        assert acc['backend'] == mode
        assert stats['queue_overflows'] == stats['profile']['fixed_point_overflows'] == stats['radio']['fixed_point_overflows'] == 0
        fallback_audit = None
        if mode in ('cuda-openmp', 'openmp-cuda'):
            c = acc['cooperative']
            assert c['subshower_cuda_submissions'] == c['subshower_cuda_commits']
            if mode == 'openmp-cuda':
                fallback_audit = check_cpu_primary(stats)
        result['accelerators'].append(dict(name=name, accelerator=acc,
            steps=stats['gpu_particles'], photon_waves=stats['resident_photon_wavefronts'],
            lepton_waves=stats['resident_lepton_wavefronts']))
        if fallback_audit is not None:
            result['accelerators'][-1]['specified_fallback_audit'] = fallback_audit
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'before', 'source', 'output', 'fe-command', 'antennas'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--build-unit', required=True)
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--standalone', choices=('cuda', 'openmp'), required=True)
    p.add_argument('--timing-modes', nargs='+', choices=('cuda', 'openmp', 'cuda-openmp', 'openmp-cuda'),
                   help='Explicit timing modes; default preserves standalone plus both cooperative modes')
    p.add_argument('--omp-proc-bind', choices=('false', 'true', 'close', 'spread'), default='false')
    p.add_argument('--omp-places', choices=('cores', 'threads', 'sockets'), default='cores')
    p.add_argument('--build-wait-timeout', type=float, default=7200.)
    p.add_argument('--affinity', help='PSR uses 382-511; inherited by all test threads')
    p.add_argument('--libraries', type=Path)
    p.add_argument('--timing-seeds', type=int, default=5)
    p.add_argument('--fe-energy-gev', type=positive_fe_energy,
                   help='Optional Fe timing energy override in GeV; default preserves archived 100 TeV command')
    p.add_argument('--timing-primary', choices=('proton',),
                   help='Explicitly replace Fe pilot primary with proton; archived template and regression cases remain unchanged')
    p.add_argument('--reuse-correctness', type=Path,
                   help='Completed full regression directory with identical binary/fixture/before SHA, source and threads')
    p.add_argument('--correctness-only', action='store_true',
                   help='Run all regression/lifecycle gates without claiming a performance comparison')
    p.add_argument('--gpu-idle-seconds', type=float, default=10.)
    p.add_argument('--foreign-workflow-file', type=Path,
                   help='Optional JSON prefixes file; per-case CPU-workflow idle and live guard evidence')
    p.add_argument('--proton-command', type=Path, help='Optional additional 100 PeV paired test after Fe pilots')
    a = p.parse_args()
    assert a.threads > 0 and a.timing_seeds > 0
    assert a.build_wait_timeout > 0
    modes = timing_modes(a.standalone, a.timing_modes)
    if a.affinity:
        first, last = map(int, a.affinity.split('-'))
        assert last-first+1 >= a.threads
        os.sched_setaffinity(0, set(range(first, last+1)))
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = a.source.resolve(); tools = Path(__file__).resolve().parent
    workflow_file = None
    if a.foreign_workflow_file is not None:
        from run_overlap_guarded import read_foreign_workflow_file, foreign_workflow_pids
        workflow_prefixes = read_foreign_workflow_file(a.foreign_workflow_file)
        workflow_file = out/'FOREIGN_WORKFLOWS.json'
        shutil.copy2(a.foreign_workflow_file, workflow_file)
    os.environ.update(FLUPRO='/home/yuhanglu/fluka', CORSIKA_DATA=str(source/'modules/data'),
        OMP_NUM_THREADS=str(a.threads), OMP_THREAD_LIMIT=str(a.threads), OMP_PROC_BIND=a.omp_proc_bind,
        OMP_PLACES=a.omp_places, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if a.libraries:
        os.environ['LD_LIBRARY_PATH'] = str(a.libraries)+':'+os.environ.get('LD_LIBRARY_PATH', '')
    state = dict(complete=False, phase='waiting-for-build', records=[])
    def status(phase):
        state.update(phase=phase, updated_unix=time.time()); save(out/'STATUS.json', state)

    def run(label, cmd, timeout=1200):
        status(label)
        assert psutil.virtual_memory().available >= 4*2**30
        assert shutil.disk_usage(out).free >= 15*2**30
        if workflow_file is not None:
            wait_case_resources(a.gpu_idle_seconds, 21600., compute_gpu_pids,
                lambda: foreign_workflow_pids(workflow_prefixes),
                lambda value: save(out/'CASE_RESOURCE_WAIT.json', dict(case=label, **value)))
        else:
            # Preserve legacy GPU-only preflight when no workflow file is supplied.
            start = time.monotonic(); idle = None
            while True:
                now = time.monotonic()
                occupied = compute_gpu_pids()  # unknown is never treated as idle
                idle = (idle if idle is not None else now) if not occupied else None
                if idle is not None and now-idle >= a.gpu_idle_seconds:
                    break
                if now-start > 600:
                    raise RuntimeError('GPU not observably idle; no test launched')
                time.sleep(2)
        cmd = list(map(str, cmd)); save(out/(label+'-command.json'), dict(command=cmd))
        workflow_guard = ['--foreign-workflow-file', str(workflow_file)] if workflow_file else []
        r = subprocess.run([sys.executable, str(tools/'run_overlap_guarded.py'),
            '--output', str(out/(label+'-guard')), '--timeout', str(timeout),
            '--rss-limit-gib', '24' if a.affinity else '6', '--sample-resources',
            '--sample-threads', '--require-exclusive-gpu', *workflow_guard, '--', *cmd], cwd=source)
        guard = json.loads((out/(label+'-guard/summary.json')).read_text())
        assert r.returncode == 0 and guard['pass'], label+' failed'
        if a.affinity:
            for line in (out/(label+'-guard/resources.jsonl')).open():
                sample = json.loads(line)
                for allowed in sample.get('thread_affinities', {}).values():
                    assert set(allowed) <= set(range(first,last+1)), 'thread escaped binding'
        return guard

    try:
        status('waiting-for-build')
        build_state = wait_for_build(a.build_unit, a.build_wait_timeout)
        save(out/'BUILD_UNIT.json', build_state)
        (out/'binaries').mkdir()
        binary = out/'binaries/c8_air_shower'; fixture = out/'binaries/testKokkosCooperativeBackend'
        shutil.copy2(a.build/'applications/c8_air_shower', binary)
        shutil.copy2(a.build/'tests/accelerator/testKokkosCooperativeBackend', fixture)
        provenance = dict(binary_sha256=sha(binary), before_sha256=sha(a.before),
            fixture_sha256=sha(fixture), runner_sha256=sha(Path(__file__)),
            guard_sha256=sha(tools/'run_overlap_guarded.py'), threads=a.threads,
            affinity=sorted(os.sched_getaffinity(0)), source=str(source),
            modes=modes, timing_seeds=a.timing_seeds, dynamic_schedule=True,
            correctness_only=a.correctness_only,
            omp_proc_bind=a.omp_proc_bind, omp_places=a.omp_places,
            build_unit=a.build_unit, build_unit_state=build_state)
        if workflow_file is not None:
            provenance.update(foreign_workflow_file=str(workflow_file),
                foreign_workflow_sha256=sha(workflow_file),
                foreign_workflow_preflight='CPU workflow and GPU idle before every guarded case',
                foreign_workflow_during_run='record timing-invalid interference; do not kill event for CPU interference')
        if a.fe_energy_gev is not None:
            provenance.update(fe_energy_override_GeV=a.fe_energy_gev,
                              fe_timing_family=fe_timing_family(a.fe_energy_gev))
        if a.timing_primary is not None:
            provenance.update(timing_primary_override=a.timing_primary,
                              timing_family=timing_family(a.fe_energy_gev, a.timing_primary))
        correctness_reuse = None
        if a.reuse_correctness is not None:
            correctness_reuse = validate_correctness_reuse(a.reuse_correctness, provenance)
            save(out/'CORRECTNESS_REUSE.json', correctness_reuse)
            provenance['correctness_reused_from'] = correctness_reuse['reference_run']
        save(out/'PROVENANCE.json', provenance)
        helps = [subprocess.check_output(['c8_air_shower','--help'], executable=str(b),
                    cwd=source, text=True) for b in (a.before,binary)]
        clean = [checked_help(x) for x in helps]
        save(out/'HELP.json', dict(before=helps[0], after=helps[1],
            unchanged_except_execution_option=clean[0][0]==clean[1][0]))
        assert clean[0][0] == clean[1][0], 'unexpected help change'
        assert 'openmp-cuda' in clean[1][1]
        if 'openmp-cuda' in clean[0][1]:
            assert clean[0][1] == clean[1][1], 'existing execution CLI changed'
        for label, extra, expected in (
            ('bad-policy',['--kokkos-cooperative-policy','adaptive'],'Adaptive scheduling requires'),
            ('bad-workers',['--hadronic-workers','2'],'--hadronic-workers must be 1')):
            cmd = [str(binary),'-p','22','-E','1','-f',str(out/label),
                '--em-backend','kokkos','--radio-backend','kokkos',
                '--kokkos-execution','openmp-cuda',*extra]
            r = subprocess.run(cmd,cwd=source,text=True,capture_output=True,timeout=60)
            save(out/(label+'.json'), dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
            assert r.returncode != 0 and expected in r.stdout+r.stderr, label+' did not fail as intended'

        probe_antennas = source/'validation/accelerator/overlap_probe/antennas.txt'
        def small(binary, label, mode, count=2, energy='1'):
            cmd = [str(binary),'-p','22','-E',energy,'-N',str(count),'-s','26091021',
                '-z','0','-a','0','--emthin','1e-6','--antenna-file',str(probe_antennas),
                '--geomagnetic-model','IGRF14','--geomagnetic-year','2027',
                '--verbosity','warn','-f',str(out/label)]
            if mode == 'proposal':
                cmd += ['--em-backend','proposal','--radio-backend','cpu']
            else:
                cmd += ['--em-backend','kokkos','--radio-backend','kokkos',
                    '--kokkos-execution',mode,'--kokkos-num-threads',str(1 if mode=='cuda' else a.threads),
                    '--gpu-min-batch','16','--gpu-resident-batch-limit','4096',
                    '--gpu-memory-fraction','.1','--hadronic-workers','1']
                if count == 2:
                    cmd += ['--cuda-replay-trace',str(out/(label+'-trace.csv'))]
            run(label,cmd,1800 if a.affinity else 600)
            checked = check_output(out/label,count,mode);save(out/(label+'-CHECK.json'),checked)

        if correctness_reuse is None:
            for mode in ('proposal','cuda','openmp'):
                small(a.before,'before-'+mode,mode)
                small(binary,'after-'+mode,mode)
                comparison = compare_outputs(out/('before-'+mode),out/('after-'+mode))
                save(out/(mode+'-regression.json'),comparison);assert comparison['pass']
                if mode != 'proposal':
                    assert (out/('before-'+mode+'-trace.csv')).read_bytes() == (out/('after-'+mode+'-trace.csv')).read_bytes(), 'new first divergence'
            for mode in ('cuda-openmp','openmp-cuda'):
                cmd = [fixture,source/'modules/data/PROPOSAL',a.threads,'subshowers','.25']
                if mode == 'openmp-cuda': cmd += [mode]
                run('fixture-'+mode,cmd,1200)
                small(binary,'N32-'+mode,mode,32,'100')
        gates = dict(passed=True, large_sample_physics=False)
        if correctness_reuse is not None:
            gates.update(reused=True, reference_run=correctness_reuse['reference_run'],
                         reference_manifest='CORRECTNESS_REUSE.json')
        save(out/'CORRECTNESS_GATES.json', gates)
        if a.correctness_only:
            state.update(complete=True, performance_tested=False,
                         large_sample_physics=False)
            status('correctness-gates-completed')
            return

        def event(mode, seed, template, family):
            label = family+'-'+str(seed)+'-'+mode
            cmd = json.loads(template.read_text())['command'];cmd[0]=str(binary)
            if template == a.fe_command:
                override_fe_energy(cmd, a.fe_energy_gev)
                override_timing_primary(cmd, a.timing_primary)
            if '--kokkos-cooperative-policy' in cmd:
                i=cmd.index('--kokkos-cooperative-policy');del cmd[i:i+2]
            for key,value in (('-N',1),('-s',seed),('-f',out/label),('--antenna-file',a.antennas),
                ('--em-backend','kokkos'),('--radio-backend','kokkos'),
                ('--gpu-physics-source','proposal-native'),('--gpu-min-batch',4096),
                ('--gpu-memory-fraction',.7),('--hadronic-workers',1),('--kokkos-device',0),
                ('--kokkos-execution',mode),('--kokkos-num-threads',1 if mode=='cuda' else a.threads)):
                put(cmd,key,value)
            guard = run(label,cmd,14400 if family=='proton100PeV' else 3600)
            checked = check_output(out/label,1,mode);save(out/(label+'-CHECK.json'),checked)
            record = dict(label=label, mode=mode, seed=seed, family=family, guard=guard,
                shower_s=checked['timing']['shower_0']['wall_time_ms']/1000.,
                accelerator=checked['accelerators'][0])
            assert record['accelerator']['accelerator']['host_threads'] == (1 if mode=='cuda' else a.threads)
            state['records'].append(record);status(label+'-complete')
        for i in range(a.timing_seeds):
            order = timing_order(modes, i)
            for mode in order: event(mode,85000001+i,a.fe_command,timing_family(a.fe_energy_gev, a.timing_primary))
        if a.proton_command:
            for mode in (modes if a.timing_modes is not None else ('cuda-openmp','openmp-cuda')):
                event(mode,2026110001,a.proton_command,'proton100PeV')
        state['complete']=True;status('completed')
    except BaseException as error:
        state['error']=repr(error);status('failed');raise


if __name__ == '__main__':
    main()
