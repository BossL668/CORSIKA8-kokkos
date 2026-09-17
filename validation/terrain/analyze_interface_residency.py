#!/usr/bin/env python3
"""Check actual CUPTI activity against labeled copies and residency counters."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def totals(rows):
    return dict(count=len(rows), bytes=sum(r['bytes'] for r in rows))


def analyze(row):
    with Path(row['trace']).open() as stream:
        events = [json.loads(line) for line in stream]
    groups = defaultdict(list)
    for event in events:
        groups[event['event']].append(event)
    accelerator = row['accelerator']
    batches = accelerator['batches']
    cascade = accelerator.get('resident_cascade_calls',0)>0
    kernels = groups['gpu_kernel']
    steps = [k for k in kernels if 'InterfaceStepKernel' in k['name']]
    first = min(k['start'] for k in steps)
    last = max(k['end'] for k in steps)
    copies = groups['copy_begin']
    inputs = [c for c in copies if c['src_space'] == 'Host' and c['dst_space'] == 'Cuda'
              and c['dst_label'] == 'terrain_step_input']
    record_label = 'interface_resident_records' if cascade else 'terrain_step_output'
    records = [c for c in copies if c['src_space'] == 'Cuda' and c['dst_space'] == 'Host'
               and c['src_label'] == record_label]
    controls = [c for c in copies if c['src_space'] == 'Cuda' and c['dst_space'] == 'Host'
                and c['src_label'] == 'interface_front_control']
    device_prefixes = [c for c in copies if c['src_space'] == c['dst_space'] == 'Cuda'
                       and c['src_label'] == 'interface_resident_particles'
                       and c['dst_label'] == 'terrain_step_input']
    particle_stride = 112  # beta5 EmParticleState; also checked against allocation/input totals.
    uploaded = sum(c['bytes'] for c in inputs) // particle_stride
    directional = {('Host', 'Cuda'): 1, ('Cuda', 'Host'): 2, ('Cuda', 'Cuda'): 8}
    # Kokkos also emits a logical deep-copy callback for scalar fills, labeled
    # "(none)". These may be cudaMemset/kernel operations, not H2D memcpy.
    expected_copies = Counter((directional[(c['src_space'], c['dst_space'])], c['bytes'])
                              for c in copies if c['bytes'] and c['src_label']!='(none)'
                              and (c['src_space'], c['dst_space']) in directional)
    actual_copies = Counter((c['kind'], c['bytes']) for c in groups['gpu_copy'])
    after_first = [c for c in groups['gpu_copy'] if c['start'] >= first]
    count_kernels = [k for k in kernels if 'InterfaceQueue' in k['name']
                     and ('5Count' in k['name'] or 'ValidateCount' in k['name'])]
    scatter_kernels = [k for k in kernels if 'InterfaceQueue' in k['name'] and '7Scatter' in k['name']]
    scan_kernels = [k for k in kernels if 'CountScanFunctor' in k['name'] or 'ControlScan' in k['name']]
    queue_allocations = [a for a in groups['allocate'] if a['label'] == 'interface_resident_particles']
    queue_live, queue_peak = 0, 0
    for a in sorted(groups['allocate'] + groups['deallocate'], key=lambda a: a['time']):
        if a['label'] == 'interface_resident_particles' and a['space'] == 'Cuda':
            queue_live += a['bytes'] if a['event'] == 'allocate' else -a['bytes']
            queue_peak = max(queue_peak, queue_live)
    begin_names = Counter(e['name'] for e in groups['for_begin'])
    resident = row['variant'] == 'resident'
    expected_particles = accelerator['cpu_uploaded_particles'] if resident else accelerator['particles_advanced']
    expected_batches = accelerator['cpu_upload_batches'] if resident else batches
    immutable_after_first = [c for c in copies if c['src_space'] == 'Host' and c['dst_space'] == 'Cuda'
                            and c['src_label']!='(none)' and c['dst_label'] != 'terrain_step_input' and c['time'] >= first]
    checks = {
        'accepted_physics_output': all(row['checks'].values()),
        'actual_transport_kernel_per_wavefront': len(steps) == batches,
        'named_transport_launches_match': begin_names['interface_material_em_step'] == batches,
        'valid_device_timestamps': all(k['start'] > 0 and k['end'] >= k['start'] and k['device'] == 0 for k in kernels),
        'particle_upload_bytes_match': sum(c['bytes'] for c in inputs) == expected_particles * particle_stride,
        'particle_upload_calls_match': len(inputs) == expected_batches,
        'copies_present_in_actual_cuda_activity': all(actual_copies[key] >= count for key, count in expected_copies.items()),
        'diagnostic_return_at_expected_checkpoint': len(records) == (accelerator['record_downloads'] if cascade else batches),
        'immutable_tables_not_uploaded_again': not immutable_after_first,
        'trace_complete': len(groups['finalize']) == 1 and not groups['error']
            and not groups['finalize'][0]['failed'] and groups['finalize'][0]['dropped_records'] == 0,
    }
    if resident:
        capacity_bytes = accelerator['resident_capacity'] * particle_stride
        checks.update({
            'queue_allocated_in_cuda': bool(queue_allocations) and all(a['space'] == 'Cuda' and a['bytes'] == capacity_bytes for a in queue_allocations),
            'bounded_queue_compaction_memory': queue_live == 0 and capacity_bytes <= queue_peak <= 2 * capacity_bytes,
            'device_prefix_every_wavefront': len(device_prefixes) == batches
                and sum(c['bytes'] for c in device_prefixes) == accelerator['particles_advanced'] * particle_stride,
            'count_and_scatter_execute_on_gpu': len(count_kernels) == len(scatter_kernels) == batches,
            'scan_executes_on_gpu': len(scan_kernels) >= batches,
            'no_host_queue_download': not any(c['src_label'] == 'interface_resident_particles' and c['dst_space'] == 'Host' for c in copies),
        })
        if row['case'] == 'photon_up':
            checks['zero_h2d_after_initial_particle'] = not any(c['kind'] == 1 for c in after_first)
    if cascade:
        checks.update({
            'one_small_control_per_wavefront': len(controls)==batches and sum(c['bytes'] for c in controls)==16*batches,
            'multi_wavefront_cascade_calls': 0<accelerator['resident_cascade_calls']<batches
                and accelerator['maximum_call_wavefronts']>1,
            'full_records_only_at_cascade_checkpoints': len(records)==accelerator['resident_cascade_calls']<batches
                and not any(c['src_label']=='terrain_step_output' and c['dst_space']=='Host' for c in copies),
        })
    sync_names = Counter(a['name'] for a in groups['cuda_api'] if 'Synchronize' in a['name'])
    return dict(case=row['case'], variant=row['variant'], checks=checks,
                steps=accelerator['particles_advanced'], batches=batches,
                uploaded_particles=uploaded, particle_upload=totals(inputs),
                diagnostic_return=totals(records), device_prefix=totals(device_prefixes),
                front_control_return=totals(controls), cascade_calls=accelerator.get('resident_cascade_calls'),
                maximum_call_wavefronts=accelerator.get('maximum_call_wavefronts'),
                actual_gpu_kernels=dict(transport=len(steps), count=len(count_kernels),
                                        scan=len(scan_kernels), scatter=len(scatter_kernels), total=len(kernels)),
                actual_gpu_copies_after_first_step={name: totals([c for c in after_first if c['kind'] == kind])
                                                   for name, kind in [('H2D', 1), ('D2H', 2), ('D2D', 8)]},
                queue_allocations=len(queue_allocations), peak_queue_allocation_bytes=queue_peak,
                queue_net_bytes_after_finalize=queue_live,
                host_cuda_synchronization_calls=dict(sync_names),
                transport_kernel_time_ms=sum(k['end'] - k['start'] for k in steps) / 1e6,
                first_to_last_transport_kernel_ms=(last-first) / 1e6,
                specified_cpu_fallbacks=accelerator['specified_cpu_fallbacks'],
                actual_copy_totals={name: totals([c for c in groups['gpu_copy'] if c['kind'] == kind])
                                    for name, kind in [('H2D', 1), ('D2H', 2), ('D2D', 8)]})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profiles', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected, rejected = {}, []
    for path in args.profiles:
        for row in json.loads(path.read_text()):
            key = (row['case'], row['variant'])
            if all(row['checks'].values()):
                selected[key] = row
            else:
                rejected.append(dict(manifest=str(path), case=row['case'], variant=row['variant'], checks=row['checks']))
    rows = [analyze(row) for row in selected.values()]
    cases = sorted({r['case'] for r in rows})
    passed = bool(rows) and all(all(r['checks'].values()) for r in rows)
    paired = all((case, 'resident') in selected and (case, 'batched') in selected for case in cases)
    report = dict(passed=passed and paired, matched_pairs=paired, cases=rows, excluded_attempts=rejected,
                  scope='Particle residency, actual GPU kernels and measured control/output checkpoints; host launches and synchronization remain.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        print(row['case'], row['variant'], 'PASS' if all(row['checks'].values()) else row['checks'],
              'uploaded=', row['uploaded_particles'], 'steps=', row['steps'])
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
