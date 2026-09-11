#!/usr/bin/env python3
"""Streaming validation of complete terrain tracks; no physics-equivalence claim."""
import argparse
import csv
import math
from pathlib import Path
import yaml


def analyze(root):
    summary = yaml.safe_load((root / 'terrain_run.yaml').read_text())
    if not summary['complete'] or summary['diagnostics']['csv_truncated']:
        raise ValueError(f'{root}: incomplete or truncated run')
    counts = {'rock': 0, 'air': 0}
    max_norm_error = 0.
    with (root / 'terrain/tracks.csv').open() as stream:
        for count, row in enumerate(csv.DictReader(stream), 1):
            values = {key: float(value) for key, value in row.items() if key != 'medium'}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f'{root}: nonfinite track')
            if values['step'] != count or values['weight'] < 0. or min(values['E0_GeV'], values['E1_GeV']) < 0.:
                raise ValueError(f'{root}: invalid step/energy/weight')
            if values['t1_s'] < values['t0_s'] - 1.e-15:
                raise ValueError(f'{root}: backwards flight time')
            for end in (0, 1):
                error = abs(sum(values[f'n{axis}{end}']**2 for axis in 'xyz') - 1.)
                max_norm_error = max(max_norm_error, error)
            counts[row['medium']] += 1
    diag = summary['diagnostics']
    if (sum(counts.values()) != diag['steps'] or counts['rock'] != diag['rock_steps'] or
            counts['air'] != diag['air_steps'] or diag['material_mismatches'] or max_norm_error > 1.e-10):
        raise ValueError(f'{root}: count/material/direction validation failure')
    return {'directory': str(root.resolve()), 'passed': True, 'primary': summary['primary'],
            'energy_GeV': summary['energy_GeV'], 'emthin': summary['emthin'],
            'direction_max_norm2_error': max_norm_error, 'diagnostics': diag,
            'reported_shower_seconds': summary['shower_seconds']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'passed': True, 'scope': 'real-terrain track integrity and material bookkeeping only',
              'runs': [analyze(run) for run in args.runs]}
    with args.output.open('x') as file:
        yaml.safe_dump(result, file, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == '__main__':
    main()
