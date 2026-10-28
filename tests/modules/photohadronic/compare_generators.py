#!/usr/bin/env python3
"""Fresh-process production/reference comparison; preserves raw failed evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', required=True, type=Path)
    parser.add_argument('--migrated', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # CTest may rerun in an existing build tree. Keep every attempt, including
    # failures, instead of overwriting evidence or refusing the next invocation.
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=args.output))
    (args.output / 'LATEST.json').write_text(json.dumps({'directory': output.name}) + '\n')
    records = []
    for name, executable in (('production', args.production), ('migrated', args.migrated)):
        directory = output / name
        directory.mkdir()
        raw = directory / 'generator.txt'
        completed = subprocess.run([str(executable.resolve()), str(raw.resolve())], cwd=directory,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        (directory / 'run.log').write_bytes(completed.stdout)
        records.append(dict(name=name, executable=str(executable), exit_code=completed.returncode,
                            raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None))
        (output / 'RUNS.json').write_text(json.dumps(records, indent=2) + '\n')
        completed.check_returncode()
        assert b'passed actual SOPHIA/threshold generator checks=' in completed.stdout
    expected = (output / 'production/generator.txt').read_bytes()
    actual = (output / 'migrated/generator.txt').read_bytes()
    assert actual == expected and expected, 'Ordered generator output differs; raw evidence retained'
    (output / 'RESULT.json').write_text(json.dumps(dict(passed=True, runs=records,
        scope='Same seed, initialization and ordered call history in fresh processes; not a full shower'), indent=2) + '\n')
    print('passed production versus migrated real SOPHIA and threshold final states')
    print('evidence=' + str(output))


if __name__ == '__main__':
    main()
