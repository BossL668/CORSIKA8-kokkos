#!/usr/bin/env python3
"""Audit bounded curved-terrain regressions, keeping the strict trace gate visible."""
import argparse
import json
from pathlib import Path
import yaml

from analyze_reference_runs import analyze
from compare_multimaterial_runs import compare
from summarize_indexed_transport import digest, first_numeric_difference


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--previous', required=True, type=Path)
    p.add_argument('--check-only', action='store_true', help='Repeat checks without replacing summary.json')
    args = p.parse_args()
    root, previous = args.root, args.previous
    cases = ['photon_up', 'photon_down', 'electron_rock', 'positron_rock',
             'nue_natural', 'nue_forced']
    groups = {mode+'_igrf': cases for mode in ['cpu', 'openmp', 'cuda']}
    groups.update({mode+'_zero': ['photon_up', 'electron_rock']
                   for mode in ['cpu', 'openmp', 'cuda']})
    groups['cuda_repeat'] = ['photon_up', 'nue_forced']
    runs, resources, unchanged_zero = [], [], {}
    input_fields = ['material_tables', 'mesh_sha256', 'air_magnetic_field_enu_T',
                    'rock_magnetic_field_enu_T', 'primary', 'energy_GeV', 'seed',
                    'position_m', 'direction', 'emcut_GeV', 'hadcut_GeV', 'mucut_GeV', 'emthin']
    for group, names in groups.items():
        driver = json.loads((root/group/'acceptance.json').read_text())
        for name in names:
            path = root/group/name
            result = analyze(path)
            meta = yaml.safe_load((path/'terrain_run.yaml').read_text())
            old = yaml.safe_load((previous/group/name/'terrain_run.yaml').read_text())
            if any(meta.get(field) != old.get(field) for field in input_fields):
                raise ValueError(f'{group}/{name}: input or physics bank changed')
            if meta['seed'] != 67101 or meta['rock_magnetic_field_enu_T'] != [0., 0., 0.]:
                raise ValueError('seed or rock field changed')
            if meta.get('accelerator', {}).get('pending_particles', 0):
                raise ValueError(f'{path}: particles not drained')
            runs.append(result)
            item = next(x for x in driver if x['case'] == name)
            if item['returncode'] or item['stop_reason'] or not item['complete']:
                raise ValueError(f'{path}: incomplete or guarded stop')
            samples = item['samples']
            resources.append(dict(group=group, case=name, binary_sha256=item['binary_sha256'],
                peak_rss_kib=max(x.get('rss_kib', 0) for x in samples),
                minimum_available_kib=min(x['available_kib'] for x in samples),
                peak_device_increment_mib=max(x.get('gpu_delta_mib', 0) for x in samples),
                wall_seconds=item['wall_seconds']))
            if group.endswith('_zero'):
                unchanged_zero[group+'/'+name] = (digest(path/'terrain/tracks.csv') ==
                    digest(previous/group/name/'terrain/tracks.csv'))
    comparisons = []
    for name in cases:
        a, b = root/'openmp_igrf'/name, root/'cuda_igrf'/name
        result = compare(a, b)
        result['first_numeric_difference'] = first_numeric_difference(
            a/'terrain/tracks.csv', b/'terrain/tracks.csv')
        comparisons.append(result)
    repeats = {name: digest(root/'cuda_repeat'/name/'terrain/tracks.csv') ==
               digest(root/'cuda_igrf'/name/'terrain/tracks.csv')
               for name in groups['cuda_repeat']}
    geometry = {name: yaml.safe_load((root/file).read_text()) for name, file in
                [('openmp', 'curved_openmp_v2.yaml'), ('cuda', 'curved_cuda.yaml')]}
    for mode in ('host', 'openmp', 'cuda'):
        last = (root/f'unit_{mode}.log').read_text().splitlines()[-1]
        if 'failures=0 host_device_exact_mismatches=0' not in last or 'checks=2000038' not in last:
            raise ValueError(f'{mode}: curved regression failed/incomplete')
    if not all(repeats.values()) or not all(unchanged_zero.values()) or not all(
            x['passed'] for x in geometry.values()):
        raise ValueError('geometry, repeat, or zero-field regression failed')
    gates = json.loads((root/'magnetic_gates.json').read_text())
    if not gates['passed']:
        raise ValueError('application gate failure')
    report = dict(scope='curved boundary and bounded transport integrity, NOT production/ensemble/radio acceptance',
        boundary_integrity_passed=True, production_ready=False,
        distinct_configurations=24, cuda_repeats=2, geometry=geometry,
        zero_field_track_bytes_unchanged=unchanged_zero, cuda_repeat_track_bytes_equal=repeats,
        strict_cross_backend_trace_passed=all(x['passed'] for x in comparisons),
        runs=runs, resources=resources, comparisons=comparisons,
        remaining=['full cross-backend floating-point trace gate',
                   'real DEM cross-interface CoREAS/ZHS',
                   'rest-mass-aware energy closure and multi-seed physics acceptance',
                   'adversarial curved edge/corner conditions beyond tested fixtures'])
    if not args.check_only:
        with (root/'summary.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print('Boundary/integrity PASS, 24 configurations + 2 repeats; strict trace:',
          report['strict_cross_backend_trace_passed'])


if __name__ == '__main__':
    main()
