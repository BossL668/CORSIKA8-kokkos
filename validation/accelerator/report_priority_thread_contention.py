#!/usr/bin/env python3
"""Stream an existing sampler log; report thread contention without touching jobs.

The sampler may contain TWO identically named CUDA driver threads. Keep every
TID/start identity: the old idle driver is not the independent active driver.
Runtime and scheduler wait are observed counters, not kernel/barrier profiling.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def nonnegative(row, key):
    value = row.get(key)
    return value if isinstance(value, (int, float)) and value >= 0 else 0.


def driver_activity(threads):
    drivers = [t for t in threads if t.get('role') == 'cuda_driver']
    return any(t.get('delta_elapsed_s', 0) > 0 and
               nonnegative(t, 'delta_cpu_s') / t['delta_elapsed_s'] > .05 for t in drivers)


def empty_group():
    return dict(samples=0, sampled_interval_s=0., driver_cpu_s=0., coordinator_cpu_s=0.,
                coordinator_runqueue_s=0., worker_candidate_cpu_s=0.,
                worker_candidate_runqueue_s=0., worker_candidate_runtime_s=0.,
                worker_wchan=Counter(), external_cpu_s=Counter(), shared_driver_cpu_samples=0)


def build_report(run):
    state = json.loads((run/'STATUS.json').read_text())
    records = state['records']
    by_pid = {r['guard']['child_pid']: r for r in records if r.get('guard', {}).get('child_pid')}
    events = {pid: dict(label=r['label'], mode=r['mode'], process_wall_s=r['guard'].get('process_wall_s'),
                        sampled_process_states=0, drivers={}, coordinator={}, phases={})
              for pid, r in by_pid.items()}
    header = None
    parse_errors = 0
    path = run/'THREAD_CONTENTION.jsonl'
    initial_bytes = path.stat().st_size
    with path.open() as stream:
        for line in stream:
            try:
                sample = json.loads(line)
            except ValueError:
                parse_errors += 1
                continue
            if sample.get('type') == 'header':
                header = sample
                continue
            if sample.get('type') != 'sample':
                continue
            for process in sample.get('processes', []):
                pid = process['pid']
                if pid not in events or process.get('identity_verified_after_sample') is not True:
                    continue
                event = events[pid]
                event['sampled_process_states'] += 1
                threads = process['threads']
                coordinator = next((t for t in threads if t.get('role') == 'coordinator'), None)
                if coordinator is None:
                    continue
                same_cpu = coordinator.get('last_cpu')
                for t in threads:
                    if t.get('role') not in ('coordinator', 'cuda_driver'):
                        continue
                    # TID reuse is excluded by the sampler's validated deltas;
                    # separate cumulative summaries by start_ticks as well.
                    key = '{}:{}'.format(t['tid'], t.get('start_ticks'))
                    bucket = event['drivers'] if t['role'] == 'cuda_driver' else event['coordinator']
                    row = bucket.setdefault(key, dict(tid=t['tid'], start_ticks=t.get('start_ticks'),
                        cpu_s=0., runtime_s=0., runqueue_s=0., sample_count=0,
                        last_cpu=Counter(), wchan=Counter(), maximum_interval_cpu_s=0.))
                    row['sample_count'] += 1
                    row['cpu_s'] += nonnegative(t, 'delta_cpu_s')
                    row['runtime_s'] += nonnegative(t, 'delta_runtime_ns') / 1e9
                    row['runqueue_s'] += nonnegative(t, 'delta_runqueue_ns') / 1e9
                    row['last_cpu'][str(t.get('last_cpu'))] += 1
                    row['wchan'][str(t.get('wchan'))] += 1
                    row['maximum_interval_cpu_s'] = max(row['maximum_interval_cpu_s'], nonnegative(t, 'delta_cpu_s'))
                dt = coordinator.get('delta_elapsed_s', 0)
                if dt <= 0:
                    continue
                active = driver_activity(threads)
                group = event['phases'].setdefault('driver_cpu_above_5pct' if active else 'driver_idle_or_below_5pct', empty_group())
                group['samples'] += 1
                group['sampled_interval_s'] += dt
                group['driver_cpu_s'] += sum(nonnegative(t, 'delta_cpu_s') for t in threads if t.get('role') == 'cuda_driver')
                group['coordinator_cpu_s'] += nonnegative(coordinator, 'delta_cpu_s')
                group['coordinator_runqueue_s'] += nonnegative(coordinator, 'delta_runqueue_ns') / 1e9
                # A name-sharing CUDA helper also appears as openmp_candidate
                # on the coordinator CPU. Report only off-coordinator candidates
                # (129 one-place workers in this spread/threads PSR campaign).
                workers = [t for t in threads if t.get('role') == 'openmp_candidate' and t.get('last_cpu') != same_cpu]
                group['worker_candidate_cpu_s'] += sum(nonnegative(t, 'delta_cpu_s') for t in workers)
                group['worker_candidate_runqueue_s'] += sum(nonnegative(t, 'delta_runqueue_ns') / 1e9 for t in workers)
                group['worker_candidate_runtime_s'] += sum(nonnegative(t, 'delta_runtime_ns') / 1e9 for t in workers)
                group['worker_wchan'].update(str(t.get('wchan')) for t in workers)
                group['shared_driver_cpu_samples'] += int(any(
                    t.get('role') == 'cuda_driver' and nonnegative(t, 'delta_cpu_s') > 0
                    and t.get('last_cpu') == same_cpu for t in threads))
                for external in sample.get('external_cpu_activity', []):
                    group['external_cpu_s'][external['comm']] += nonnegative(external, 'delta_cpu_s')
    for event in events.values():
        for group in event['phases'].values():
            dt = group['sampled_interval_s']
            group['coordinator_core_equivalents'] = group['coordinator_cpu_s'] / dt
            group['worker_candidate_core_equivalents'] = group['worker_candidate_cpu_s'] / dt
            group['coordinator_runqueue_fraction'] = group['coordinator_runqueue_s'] / dt
    return dict(phase=state['phase'], completed_records=len(records), source_bytes_before=initial_bytes,
                source_bytes_after=path.stat().st_size, parse_errors=parse_errors,
                kernel_sched_schedstats=header.get('kernel_sched_schedstats') if header else None,
                sampled_binary_sha256=header.get('binary_sha256') if header else None,
                events=list(events.values()), caveats=[
                    'No sampler coverage means missing evidence, not zero contention.',
                    'All identically named CUDA-driver TIDs are preserved; idle old driver must not hide active independent driver.',
                    'last_cpu is instantaneous placement; repeated same-CPU samples plus scheduler deltas support contention, not full causal attribution.',
                    'Scheduler runqueue wait is not OpenMP barrier wait. Kernel schedstats setting 0 limits interpretation of absent/zero delay.',
                    'Worker candidates are same-name off-coordinator-CPU threads, not an API-level membership proof.',
                    'CPU consumption/wchan 0 can include spinning; host CUDA-driver activity is not proof of GPU-kernel activity.',
                    'External CPU counters cover same-user readable processes only, exclude first baselines, and do not certify global host isolation.',
                    'No CUPTI/Nsight kernel timeline is present; exact GPU-busy versus OpenMP-barrier causality is not measured.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, help='Optional new small JSON; raw logs remain in place')
    args = parser.parse_args()
    report = build_report(args.run)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(text)
        print(json.dumps(dict(output=str(args.output), events=len(report['events']), bytes=len(text))))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
