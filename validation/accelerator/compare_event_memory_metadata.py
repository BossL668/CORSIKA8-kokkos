#!/usr/bin/env python3
"""Compare memory-refactor summaries without dropping physical flight times.

Only measured wall-clock durations and the rebuilt project revision may differ.
Schema, ordering, physical counters, lifecycle, hashes and energy ledger remain
strict comparisons. Parquet/NPZ arrays are checked separately by the runner.
"""
import argparse
import json
import math
from pathlib import Path

import yaml


def walk(a, b, path=()):
    if type(a) is not type(b):
        yield path, a, b
    elif isinstance(a, dict):
        if list(a) != list(b):
            yield (*path, '<keys/order>'), list(a), list(b)
        for k in a.keys() & b.keys():
            yield from walk(a[k], b[k], (*path, str(k)))
    elif isinstance(a, list):
        if len(a) != len(b):
            yield (*path, '<length>'), len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, (*path, str(i)))
    elif a != b and not (isinstance(a, float) and math.isnan(a) and math.isnan(b)):
        yield path, a, b


def allowed(path, a, b):
    if '<keys/order>' in path or '<length>' in path:
        return False
    if path[-2:] == ('accelerator', 'project_revision'):
        return True
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    # Physical track time is seconds, not measured wall-clock milliseconds.
    # Do not exclude fields merely because they contain the word "time".
    blocks = {'hybrid_timing_ms', 'scalar_stepper_time_by_pdg_ms',
              'scalar_phase_timing_by_pdg_ms', 'lepton_pipeline_timing'}
    if blocks.intersection(path):
        return True
    key = path[-1]
    if key == 'standard_deviation_ms' and {
            'hadronic_work_classes', 'hadronic_final_state_classes'}.intersection(path):
        return True
    if path[-2:] == ('hadronic_worker_oracle', 'predicted_kernel_speedup'):
        return True
    return (key.endswith('_time_ms') or key in {
        'one_time_initialization_ms', 'wall_time_ms', 'wall_time_ms_partial',
        'host_wait_upper_bound_ms', 'actual_generator_load_imbalance_ms',
        'predicted_makespan_ms', 'load_imbalance_ms', 'estimated_load_ms'})


def compare(left, right):
    files = {p.relative_to(left) for p in left.glob('*/summary.yaml')}
    if files != {p.relative_to(right) for p in right.glob('*/summary.yaml')}:
        raise RuntimeError('Summary file sets differ')
    accepted, failures, equal = [], [], []
    for name in sorted(files):
        a = yaml.load((left / name).read_text(), Loader=yaml.CSafeLoader)
        b = yaml.load((right / name).read_text(), Loader=yaml.CSafeLoader)
        differences = list(walk(a, b))
        if not differences:
            equal.append(str(name))
        for path, x, y in differences:
            item = dict(file=str(name), path='.'.join(path), before=x, after=y)
            (accepted if allowed(path, x, y) else failures).append(item)
    return dict(pass_=not failures, exact_files=equal, files=len(files),
                permitted_differences=len(accepted), unexpected=failures,
                permitted_paths=sorted({x['file'] + ':' +
                    '.'.join(x['path'].split('.')[1:]) for x in accepted}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    report = json.loads((args.root / 'regression.json').read_text())
    results = {}
    for label in report['runs']:
        if '_before_' not in label:
            continue
        other = label.replace('_before_', '_after_')
        if other in report['runs']:
            results[label.replace('_before_', '_')] = compare(args.root / label,
                                                               args.root / other)
    destination = args.root / 'metadata_comparison.json'
    destination.write_text(json.dumps(results, indent=2, default=str) + '\n')
    print(json.dumps(results, indent=2, default=str))
    return 0 if results and all(r['pass_'] for r in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
