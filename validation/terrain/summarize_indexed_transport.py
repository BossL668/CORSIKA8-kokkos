#!/usr/bin/env python3
"""Summarize this bounded terrain regression without claiming ensemble/radio acceptance."""
import argparse
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import yaml

from analyze_reference_runs import analyze
from compare_multimaterial_runs import compare


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            value.update(data)
    return value.hexdigest()


def first_numeric_difference(left, right):
    with left.open() as a, right.open() as b:
        for i, (x, y) in enumerate(zip_longest(csv.DictReader(a), csv.DictReader(b)), 1):
            if x is None or y is None:
                return dict(row=i, field='row_count')
            for key in x:
                equal = x[key] == y[key] if key == 'medium' else float(x[key]).hex() == float(y[key]).hex()
                if not equal:
                    return dict(row=i, field=key, left=x[key], right=y[key])
    return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Repeat checks without rewriting an existing report')
    args = parser.parse_args()
    root = args.root
    cases = ['photon_up','photon_down','electron_rock','positron_rock','nue_natural','nue_forced']
    groups = {mode+'_igrf': cases for mode in ['cpu','openmp','cuda']}
    groups.update({mode+'_zero': ['photon_up','electron_rock'] for mode in ['cpu','openmp','cuda']})
    groups['cuda_repeat'] = ['photon_up','nue_forced']
    runs, resources = [], []
    for group, names in groups.items():
        driver = json.loads((root/group/'acceptance.json').read_text())
        for name in names:
            path = root/group/name
            runs.append(analyze(path))
            meta = yaml.safe_load((path/'terrain_run.yaml').read_text())
            if meta['seed'] != 67101 or meta['rock_magnetic_field_enu_T'] != [0.,0.,0.]:
                raise ValueError('seed or rock-field mismatch')
            if meta.get('accelerator',{}).get('pending_particles',0) != 0:
                raise ValueError(f'{path}: pending particles after completion')
            item = next(d for d in driver if d['case'] == name)
            if item['returncode'] or item['stop_reason'] or not item['complete']:
                raise ValueError(f'{path}: driver did not complete normally')
            samples = item['samples']
            resources.append(dict(group=group,case=name,binary_sha256=item['binary_sha256'],
                peak_rss_kib=max(s.get('rss_kib',0) for s in samples),
                minimum_available_kib=min(s['available_kib'] for s in samples),
                peak_device_increment_mib=max(s.get('gpu_delta_mib',0) for s in samples)))
    pairs = []
    for name in cases:
        a,b = root/'openmp_igrf'/name,root/'cuda_igrf'/name
        result = compare(a,b)
        result['first_numeric_difference'] = first_numeric_difference(a/'terrain/tracks.csv',b/'terrain/tracks.csv')
        result['csv_sha256_equal'] = digest(a/'terrain/tracks.csv') == digest(b/'terrain/tracks.csv')
        ma,mb = [yaml.safe_load((p/'terrain_run.yaml').read_text()) for p in (a,b)]
        for field in ('material_tables','mesh_sha256','air_magnetic_field_enu_T','primary','energy_GeV',
                      'position_m','direction','emcut_GeV','hadcut_GeV','mucut_GeV','emthin'):
            if ma.get(field) != mb.get(field):
                raise ValueError(f'{name}: input/bank mismatch: {field}')
        pairs.append(result)
    repeat = {name: digest(root/'cuda_igrf'/name/'terrain/tracks.csv') ==
                   digest(root/'cuda_repeat'/name/'terrain/tracks.csv') for name in groups['cuda_repeat']}
    if not all(repeat.values()):
        raise ValueError('same CUDA seed does not reproduce track bytes')
    geometry = {mode: yaml.safe_load((root/f'dem_{mode}_exact.yaml').read_text()) for mode in ['openmp','cuda']}
    if not all(g['passed'] and g['indexed_host_device_exact_mismatches']==0 for g in geometry.values()):
        raise ValueError('indexed host/device geometry failed')
    report = dict(scope='terrain straight-boundary integration and bounded transport integrity; NOT production/radio/ensemble acceptance',
        integrity_passed=True, distinct_test_configurations=24, repeated_cuda_cases=2,
        geometry=geometry, runs=runs, resources=resources, cuda_repeat_track_bytes_equal=repeat,
        openmp_cuda_existing_strict_trace_gate_passed=all(p['passed'] for p in pairs), comparisons=pairs,
        remaining=['curved shared-feature ownership', 'full cross-backend floating-point trace gate',
                   'cross-interface CoREAS/ZHS', 'rest-mass-aware energy closure and multi-seed physics acceptance'])
    if not args.check_only:
        with (root/'summary.json').open('x') as output:
            json.dump(report,output,indent=2)
            output.write('\n')
    print('integrity PASS: 24 cases + 2 repeats; strict trace:',report['openmp_cuda_existing_strict_trace_gate_passed'])


if __name__ == '__main__':
    main()
