#!/usr/bin/env python3
"""Version/hash/height/material/observer gates; no PROPOSAL or device setup."""
import argparse
import copy
from pathlib import Path
import subprocess
import tempfile
import yaml


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    args = parser.parse_args()
    base = yaml.safe_load(args.scene.read_text())
    base['geometry']['mesh_path'] = str((args.scene.resolve().parent / base['geometry']['mesh_path']).resolve())
    cases = []

    def case(name, section, key, value, message):
        cfg = copy.deepcopy(base)
        cfg[section][key] = value
        cases.append((name, cfg, message))

    case('wrong frame', 'site', 'coordinate_frame', 'flat_approximation', 'versioned ENU')
    case('height datum mismatch', 'site', 'geoid_undulation_m', 0., 'height mismatch')
    case('mesh hash', 'provenance', 'mesh_sha256', '0'*64, 'SHA-256 mismatch')
    case('wrong material', 'geometry', 'material', 'water', 'versioned ENU')
    case('wrong atmosphere', 'atmosphere', 'model', 'uniform', 'versioned ENU')
    case('invalid latitude', 'site', 'latitude_deg', 91., 'geographic origin')
    case('invalid padding', 'geometry', 'boundary_padding_m', -1., 'padding')
    case('invalid density', 'geometry', 'rock_density_g_cm3', -1., 'rock material')
    case('empty observers', 'radio', 'observers', [], 'observer positions')
    case('duplicate observers', 'radio', 'observers', [base['radio']['observers'][0]]*2, 'duplicate observer')
    case('in-rock observer', 'radio', 'observers', [{'name': 'bad', 'position_enu_m': base['geometry']['rock_reference_enu_m']}], 'inside terrain')
    results = []
    with tempfile.TemporaryDirectory(prefix='c8-scene-gates-') as directory:
        root = Path(directory)
        for i, (name, cfg, expected) in enumerate(cases):
            source, target = root/f'{i}.yaml', root/f'{i}_result.yaml'
            source.write_text(yaml.safe_dump(cfg))
            result = subprocess.run([str(args.binary.resolve()), '--config', str(source), '--output', str(target)],
                                    text=True, capture_output=True, timeout=60)
            if result.returncode == 0 or expected not in result.stderr or target.exists():
                raise RuntimeError((name, result.returncode, result.stderr))
            results.append({'name': name, 'passed': True})
    print(yaml.safe_dump({'passed': True, 'cases': results}, sort_keys=False))


if __name__ == '__main__':
    main()
