#!/usr/bin/env python3
"""Summarize bounded same-condition pilots; never promote one seed to acceptance."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    reference = None
    for path in args.timing:
        results = json.loads(path.read_text())
        for label, result in results.items():
            monitor = result['monitor']
            assert monitor['pass'] and result['returncode'] == 0
            command = list(monitor['command'][1:])
            # Only implementation, execution endpoint, and output path differ.
            for flag in ('-f', '--kokkos-execution', '--kokkos-num-threads'):
                i = command.index(flag)
                del command[i:i + 2]
            if reference is None:
                reference = command
            assert command == reference, 'different physics/observer conditions'
            stats = result['physical_statistics']
            assert stats['queue_overflows'] == 0
            assert stats['profile']['fixed_point_overflows'] == 0
            assert stats['radio']['fixed_point_overflows'] == 0
            telemetry = [json.loads(line) for line in
                         (path.parent / (label + '-guard/resources.jsonl')).read_text().splitlines()]
            cpu = [x for x in telemetry if 'cpu_seconds' in x]
            cpu_equivalent = ((cpu[-1]['cpu_seconds'] - cpu[0]['cpu_seconds']) /
                              (cpu[-1]['elapsed_s'] - cpu[0]['elapsed_s']))
            c = stats['accelerator'].get('cooperative', {})
            rows.append(dict(
                run=path.parent.name + '/' + label,
                binary_sha256=result['binary_sha256'],
                shower_s=result['shower_wall_s'], process_s=monitor['elapsed_s'],
                peak_RSS_MiB=monitor['peak_tree_rss_bytes'] / 2**20,
                peak_device_used_MiB=max(x.get('device_used_mib', 0) for x in telemetry),
                process_cpu_core_equivalent=cpu_equivalent,
                cooperative=c, gpu_final_states=stats['gpu_final_states'],
                energy_ledger_complete_coverage=stats['energy_ledger']['complete_coverage'],
                energy_ledger_accepted=stats['energy_ledger']['accepted']))
    report = dict(
        physics_arguments=reference, runs=rows,
        performance_acceptance=False, large_sample_physics_acceptance=False,
        notes=[
            'One seed per candidate; dynamically different shower trees. Not a median speed-up benchmark.',
            'CPU core equivalent includes the driver, scheduling and waits; it is not the number of OpenMP workers.',
            'Device memory is device-wide telemetry, not exclusive process allocation.',
            'The accelerator energy ledger lacks full scalar/hadronic coverage in these pilots; no global closure claim.',
        ])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps([{k: row[k] for k in ('run', 'shower_s', 'process_cpu_core_equivalent',
                                          'peak_RSS_MiB')} for row in rows], indent=2))


if __name__ == '__main__':
    main()
