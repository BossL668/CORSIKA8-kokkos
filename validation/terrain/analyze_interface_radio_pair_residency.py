#!/usr/bin/env python3
"""Verify paired radio storage, kernels and copies from PSR CUPTI traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import yaml

GRIDS = ['interface_radio_moments', 'interface_radio_coreas_regularized_moments',
         'interface_radio_zhs_moments']
TRACKS = ['interface_radio_transport_tracks', 'interface_radio_device_pending',
          'interface_radio_cpu_pending']
OPTICS = ['interface_radio_observers', 'interface_radio_index_table',
          'interface_radio_optical_breaks']


def without_directories(value):
    if isinstance(value, dict):
        return {k: without_directories(v) for k, v in value.items() if k != 'directory'}
    return value


def inspect(trace, root, reference, batches, cpu_tracks, fixture=False):
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    kernels = {algorithm: [e for e in events if e['event'] == 'gpu_kernel'
               and marker in e['name']] for algorithm, marker in
               [('CoREAS', 'CoreasEndpointKernel'), ('ZHS', 'AccumulateKernel')]}
    both = kernels['CoREAS'] + kernels['ZHS']
    first = min((e['start'] for e in both), default=0)
    last = max((e['end'] for e in both), default=0)
    labels = Counter(e['name'] for e in events if e['event'] == 'for_begin')
    copies = [e for e in events if e['event'] == 'copy_begin'
              and e['src_label'] != '(none)' and e['bytes']]
    downloads = [e for e in copies if e['src_label'] in GRIDS and e['dst_space'] == 'Host']
    cpu = [e for e in copies if e['dst_label'] == 'interface_radio_cpu_tracks'
           and e['src_space'] == 'Host' and e['dst_space'] == 'Cuda']
    allocations = Counter(e['label'] for e in events
                          if e['event'] == 'allocate' and e['space'] == 'Cuda')
    config = json.loads((root / 'CoREAS/config.json').read_text())
    expected_bytes = config['samples'] * len(config['observers']) * 6 * (config['moment_order'] + 1) * 8
    roundtrips = [e for e in copies if e['src_space'] != e['dst_space']
                  and any(label in [e['src_label'], e['dst_label']] for label in TRACKS)]
    # Only the synthetic fixture seeds its initial input array from the host.
    allowed_seed = lambda e: (fixture and e['dst_label'] == TRACKS[0]
                             and e['src_space'] == 'Host' and e['dst_space'] == 'Cuda'
                             and e['time'] < first)
    finals = [e for e in events if e['event'] == 'finalize']
    checks = dict(
        both_cuda_kernels=all(len(v) == batches > 0 for v in kernels.values()),
        labeled_launches=labels['interface_radio_coreas_endpoints'] == batches
            and labels['interface_radio_zhs_intervals'] == batches,
        one_shared_input_allocation=all(allocations[label] == 1 for label in TRACKS
            + ['interface_radio_cpu_tracks', 'interface_radio_observers']),
        independent_grids_and_counters=all(allocations[label] == 1 for label in GRIDS
            + ['interface_radio_counters', 'interface_radio_zhs_counters']),
        no_device_tracks_roundtrip=all(allowed_seed(e) for e in roundtrips),
        device_to_device_append=any(e['src_label'] == TRACKS[0] and e['dst_label'] == TRACKS[1]
            and e['src_space'] == e['dst_space'] == 'Cuda' for e in copies),
        three_final_downloads=len(downloads) == 3 and set(e['src_label'] for e in downloads) == set(GRIDS)
            and all(e['bytes'] == expected_bytes and e['time'] > last for e in downloads),
        no_moment_reupload=not any(e['dst_label'] in GRIDS and e['src_space'] == 'Host' for e in copies),
        immutable_optics=not any(e['dst_label'] in OPTICS and e['src_space'] == 'Host'
            and e['time'] > first for e in copies),
        cpu_sources_uploaded_once=sum(e['bytes'] for e in cpu) == cpu_tracks * 104,
        trace_complete=len(finals) == 1 and not finals[0]['failed'] and finals[0]['dropped_records'] == 0
            and not any(e['event'] == 'error' for e in events))
    # KokkosPropagation constructs one view per medium, including zero-extent
    # views (Kokkos still reports their small allocation headers). Count views,
    # then independently account for the actual nonempty payload uploads.
    checks['shared_optical_tables'] = all(allocations[label] == len(config['media'])
        and sum(e['bytes'] for e in copies if e['dst_label'] == label
                and e['src_space'] == 'Host' and e['dst_space'] == 'Cuda')
            == sum(len(m[key]) * size for m in config['media'])
        for label, key, size in [('interface_radio_index_table', 'radial_index', 16),
                                 ('interface_radio_optical_breaks', 'radial_integration_breaks_m', 8)])
    actual = Counter((e['kind'], e['bytes']) for e in events if e['event'] == 'gpu_copy')
    checks['actual_cupti_grid_copies'] = actual[(2, expected_bytes)] >= 3
    errors = {}
    for algorithm in ['CoREAS', 'ZHS']:
        checks[algorithm + '_same_config'] = ((root / algorithm / 'config.json').read_bytes()
            == (reference / algorithm / 'config.json').read_bytes())
        for filename in ['moments.bin'] + (['regularized_moments.bin'] if algorithm == 'CoREAS' else []):
            a = np.fromfile(root / algorithm / filename, dtype=np.float64)
            b = np.fromfile(reference / algorithm / filename, dtype=np.float64)
            errors[algorithm + '/' + filename] = (float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1.e-100))
                if a.shape == b.shape else float('inf'))
    checks['three_moments_unchanged'] = all(e < 1.e-10 for e in errors.values())
    return dict(checks=checks, trace=str(trace), batches_per_algorithm=batches,
        gpu_kernels={k: len(v) for k, v in kernels.items()},
        gpu_time_ms={k: sum(e['end']-e['start'] for e in v)/1.e6 for k, v in kernels.items()},
        moment_download_bytes=3*expected_bytes, cpu_upload_bytes=sum(e['bytes'] for e in cpu),
        initial_fixture_upload_bytes=sum(e['bytes'] for e in roundtrips if allowed_seed(e)),
        moment_relative_l2=errors)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--acceptance', type=Path, required=True)
    p.add_argument('--fixture-output', type=Path, required=True)
    p.add_argument('--fixture-reference', type=Path, required=True)
    p.add_argument('--fixture-trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    accepted = json.loads(args.acceptance.read_text()); rows = []
    for run in json.loads(args.profiles.read_text()):
        command = run['command']; root = Path(command[command.index('--output')+1])
        radio = yaml.safe_load((root / 'terrain_run.yaml').read_text())['radio_result']
        previous = next(r for r in accepted if (r['case'], r['variant']) == (run['case'], run['variant']))
        row = inspect(Path(run['trace']), root / 'radio', Path(previous['radio']['directory']),
                      radio['wavefronts'], radio['cpu_tracks'])
        row['case'] = run['case']; row['checks']['accepted_transport'] = all(run['checks'].values())
        row['checks']['same_radio_counters'] = without_directories(radio) == without_directories(previous['radio'])
        rows.append(row)
    fixture = json.loads((args.fixture_output / 'pair.json').read_text())
    row = inspect(args.fixture_trace, args.fixture_output, args.fixture_reference,
                  fixture['batches_per_algorithm'], fixture['cpu_tracks'], fixture=True)
    row['case'] = 'interleaved_cpu_device_plane_fixture'
    row['checks']['fixture_passed'] = fixture['passed']; rows.append(row)
    report = dict(passed=bool(rows) and all(all(r['checks'].values()) for r in rows), runs=rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
