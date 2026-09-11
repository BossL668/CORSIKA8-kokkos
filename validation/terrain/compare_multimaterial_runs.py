#!/usr/bin/env python3
"""Compare complete OpenMP/CUDA terrain traces without loading a shower in RAM.

This is an implementation regression test, not a CPU/GPU ensemble test. The
scalar CPU uses another RNG/schedule and must not be required to match rows.
"""
import argparse
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path

from analyze_reference_runs import analyze


def compare(left, right):
    a, b = analyze(left), analyze(right)
    maximum_absolute = {}
    maximum_relative = {}
    first_difference = None
    first_discrete_difference = None
    weights_exactly_equal = True
    rows = 0
    with (left / 'terrain/tracks.csv').open() as fa, (right / 'terrain/tracks.csv').open() as fb:
        for rows, (ra, rb) in enumerate(zip_longest(csv.DictReader(fa), csv.DictReader(fb)), 1):
            if ra is None or rb is None:
                first_difference = first_difference or {'row': rows, 'field': 'row_count'}
                first_discrete_difference = {'row': rows, 'field': 'row_count'}
                break
            for key in ra:
                if key in ('step', 'pdg', 'medium'):
                    equal = ra[key] == rb[key]
                else:
                    x, y = float(ra[key]), float(rb[key])
                    error = abs(x - y)
                    maximum_absolute[key] = max(maximum_absolute.get(key, 0.), error)
                    maximum_relative[key] = max(maximum_relative.get(key, 0.),
                                                error / max(abs(x), abs(y), 1.e-300))
                    equal = math.isclose(x, y, rel_tol=1.e-10, abs_tol=1.e-12)
                if not equal and first_difference is None:
                    first_difference = {'row': rows, 'field': key, 'left': ra[key], 'right': rb[key]}
                if key == 'weight' and ra[key] != rb[key]:
                    weights_exactly_equal = False
                if key in ('step', 'pdg', 'medium') and ra[key] != rb[key] and first_discrete_difference is None:
                    first_discrete_difference = {'row': rows, 'field': key, 'left': ra[key], 'right': rb[key]}
    diagnostics_equal = a['diagnostics'] == b['diagnostics']
    return {'left': str(left.resolve()), 'right': str(right.resolve()), 'rows': rows,
            'passed': first_difference is None and diagnostics_equal,
            'first_difference': first_difference, 'diagnostics_exactly_equal': diagnostics_equal,
            'discrete_trace_equal': first_discrete_difference is None,
            'first_discrete_difference': first_discrete_difference,
            'weights_exactly_equal': weights_exactly_equal,
            'max_absolute_difference': maximum_absolute, 'max_relative_difference': maximum_relative,
            'left_integrity': a, 'right_integrity': b}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pair', nargs=2, type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [compare(*pair) for pair in args.pair]
    result = {'passed': all(r['passed'] for r in results), 'track_integrity_passed': True,
              'scope': 'single-seed bounded multi-material OpenMP/CUDA trace regression, not ensemble physics acceptance',
              'relative_tolerance': 1.e-10, 'absolute_tolerance': 1.e-12, 'pairs': results}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    for r in results:
        print(r['left'], r['rows'], r['passed'], r['first_difference'])
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
