#!/usr/bin/env python3
"""Compare beta5's native terrain scene with the original mountain reference."""
import argparse
from pathlib import Path
import math
import yaml


def compare(beta, original):
    a = {o['name']: o for o in beta['observers']}
    b = {o['name']: o for o in original['observer_medium_checks']}
    if a.keys() != b.keys() or not a:
        raise ValueError('observer identities differ or are empty')
    errors = {'position_m': 0., 'altitude_m': 0., 'density_kg_m3': 0., 'refractive_index': 0.}
    for name in a:
        if a[name]['inside_rock'] or b[name]['rock']:
            raise ValueError(f'{name}: exterior array contains an in-rock observer')
        for key, field in [('altitude_m', 'altitude_asl_m'), ('density_kg_m3', 'density_kg_m3'), ('refractive_index', 'refractive_index')]:
            diff = abs(a[name][field] - b[name][field])
            if not math.isfinite(diff):
                raise ValueError('nonfinite environmental comparison')
            errors[key] = max(errors[key], diff)
        for x, y in zip(a[name]['enu_m'], b[name]['enu_m']):
            if not math.isfinite(x-y):
                raise ValueError('nonfinite ENU comparison')
            errors['position_m'] = max(errors['position_m'], abs(x-y))
    errors['mesh_minimum_altitude_m'] = abs(beta['minimum_altitude_asl_m'] - original['mesh_minimum_altitude_asl_m'])
    errors['mesh_maximum_altitude_m'] = abs(beta['maximum_altitude_asl_m'] - original['mesh_maximum_altitude_asl_m'])
    errors['origin_altitude_m'] = abs(beta['origin_altitude_asl_m'] - original['origin_altitude_asl_m'])
    passed = beta['mesh_vertices'] == original['mesh_vertices_checked'] and all(
        math.isfinite(error) and error <= (1.e-13 if key in ('density_kg_m3', 'refractive_index') else 1.e-9)
        for key, error in errors.items())
    return {'passed': passed, 'observers': len(a), 'mesh_vertices': beta['mesh_vertices'],
            'maximum_absolute_differences': errors,
            'scope': 'geometry, ASL/ellipsoid handling, antennas and atmosphere; not shower/radio acceptance'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--beta5', type=Path, required=True)
    parser.add_argument('--mountain', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(yaml.safe_load(args.beta5.read_text()), yaml.safe_load(args.mountain.read_text()))
    result['inputs'] = {'beta5': str(args.beta5.resolve()), 'mountain': str(args.mountain.resolve())}
    with args.output.open('x') as file:
        yaml.safe_dump(result, file, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
