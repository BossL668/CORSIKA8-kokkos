#!/usr/bin/env python3
"""Compare FLUKA interfaces in fresh processes with identical inputs and seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--production', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--material', choices=['air', 'ice', 'firn', 'rock'], required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='fluka-final-state-', dir=args.directory)).resolve()
    report = {'passed': False, 'material': args.material, 'comparisons': [],
              'scope': 'fixed-input final states; not a complete shower',
              'flupro': os.environ.get('FLUPRO'), 'directory': str(root)}
    programs = [('candidate-stack', args.candidate, 'stack'),
                ('candidate-direct', args.candidate, 'direct')]
    if args.production:
        programs.extend([('production-stack', args.production, 'stack'),
                         ('production-direct', args.production, 'direct')])
    if args.baseline:
        programs.append(('baseline-stack', args.baseline, 'stack'))
    try:
        for seed in [17, 103, 2026]:
            outputs = []
            for name, executable, mode in programs:
                work = root / (name + '-' + str(seed))
                work.mkdir()
                output = work / 'final-states.txt'
                command = [str(executable.resolve()), args.material, mode, str(seed), str(output)]
                with (work / 'run.log').open('w') as log:
                    result = subprocess.run(command, cwd=work, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=600)
                record = {'name': name, 'seed': seed, 'command': command,
                          'exit_code': result.returncode, 'path': str(output)}
                report['comparisons'].append(record)
                if result.returncode != 0:
                    raise RuntimeError('FLUKA probe failed: ' + str(work / 'run.log'))
                data = output.read_bytes()
                lines = data.decode().splitlines()
                if not lines or not lines[-1].startswith('complete '):
                    raise RuntimeError('Missing complete final-state output')
                _, events, secondaries, random_values, random_hash = lines[-1].split()
                expected_events = (4 if args.material == 'air' else 2) * 4 * 3 * 2 * 2
                if int(events) != expected_events or int(secondaries) == 0 or int(random_values) == 0:
                    raise RuntimeError('Incomplete physical-event coverage')
                if sum(line.startswith('rejected ') for line in lines) != 2:
                    raise RuntimeError('Missing invalid-reaction coverage')
                record.update(events=int(events), secondaries=int(secondaries),
                              random_values=int(random_values), random_hash=int(random_hash),
                              sha256=hashlib.sha256(data).hexdigest())
                if outputs and data != outputs[0]:
                    raise RuntimeError('Final states or random histories differ: ' + name)
                outputs.append(data)
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        (root / 'RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
        print(root, flush=True)


if __name__ == '__main__':
    main()
