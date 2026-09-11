#!/usr/bin/env python3
"""Fail-closed application tests; no device or shower starts are requested.

Uses a private temporary directory. A COF fixture deliberately lacks the 2025
epoch, so an older model cannot silently be labelled IGRF14. All rejected cases
must leave the proposed output absent, except the pre-existing-output case.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='c8-terrain-magnetic-gates-') as temp:
        root = Path(temp)
        old = root / 'old.COF'
        old.write_text('IGRF2020 2020.00\n')
        existing = root / 'existing'
        existing.mkdir()
        sentinel = existing / 'untouched.txt'
        sentinel.write_text('do not overwrite\n')
        cases = [
            ('expired-year', ['--magnetic-year', '2031'], '2030'),
            ('missing-coefficients', ['--igrf-file', str(root / 'missing.COF')], 'does not exist'),
            ('old-coefficients', ['--igrf-file', str(old)], '2025 reference epoch'),
            ('zero-direction', ['--direction', '0', '0', '0'], 'zero direction'),
            ('forced-photon', ['--force-vertex-cc'], 'requires a neutrino vertex'),
            ('forced-air-neutrino', ['--primary', 'nu_e', '--energy-GeV', '10000',
                                      '--force-vertex-cc'], 'strictly inside rock'),
            ('preserve-output', [], 'output exists'),
        ]
        for name, extra, expected in cases:
            output = existing if name == 'preserve-output' else root / name
            command = [str(args.binary.resolve()), '--scene', str(args.scene.resolve()),
                       '--output', str(output), '--position-m', '0', '0', '0.01'] + extra
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            message = run.stdout + run.stderr
            passed = (run.returncode != 0 and expected in message and
                      (output == existing or not output.exists()) and
                      sentinel.read_text() == 'do not overwrite\n')
            results.append(dict(name=name, passed=passed, returncode=run.returncode,
                                expected=expected, message=message[-2000:]))
        report = {'passed': all(r['passed'] for r in results), 'cases': results}
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(report, indent=2))
        if not report['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
