"""Retain the production finite adjacent-repeat and persistent-pool self-tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe', required=True, type=Path)
parser.add_argument('--reference', action='store_true')
parser.add_argument('--directory', required=True, type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='fluka-production-fixture-', dir=args.directory)).resolve()
result = {'passed': False, 'count': 96, 'initial_seed': 1, 'directory': str(root), 'runs': [],
          'scope': 'unchanged production fixture generator and exact repeated-result checks'}


def run(mode, original=False):
    work = root / (('production-' if original else 'candidate-') + mode)
    work.mkdir()
    output = work / 'responses.txt'
    env = os.environ.copy()
    env['CORSIKA_TEST_FLUKA_MATERIAL'] = 'air'
    # Match the pinned worker main; inherited by every exec'ed pool worker.
    env['CORSIKA_TEST_FLUKA_INITIAL_SEED'] = '1'
    env.pop('CORSIKA_TEST_ORIGINAL_FLUKA_WORKER', None)
    command_mode = mode
    if original:
        if mode == 'self-adjacent': command_mode = 'self-reference-adjacent'
        else: env['CORSIKA_TEST_ORIGINAL_FLUKA_WORKER'] = '1'
    command = [str(args.probe.resolve()), command_mode, str(output)]
    with (work / 'run.log').open('w') as log:
        completed = subprocess.run(command, cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=1200)
    record = {'mode': mode, 'production': original, 'command': command,
              'exit_code': completed.returncode, 'output': str(output)}
    result['runs'].append(record)
    assert completed.returncode == 0, str(work / 'run.log')
    data = output.read_bytes()
    lines = data.decode().splitlines()
    assert lines[-1] == 'complete 96 2'
    starts = [i for i,line in enumerate(lines) if line.startswith('event ')]
    assert len(starts) == 192
    starts.append(len(lines)-1)
    events = [lines[starts[i]:starts[i+1]] for i in range(192)]
    if mode == 'self-adjacent':
        assert [int(e[0].split()[1]) for e in events] == [i for i in range(96) for _ in range(2)]
        assert all(events[2*i] == events[2*i+1] for i in range(96))
    else:
        workers = int(mode.removeprefix('self-pool'))
        batches = min(max(workers * 4, 1), 96)
        expected = [i for b in range(batches) for i in range(96) if i % batches == b]
        assert [int(e[0].split()[1]) for e in events] == expected * 2
        assert events[:96] == events[96:]
    record['sha256'] = hashlib.sha256(data).hexdigest()
    return data


try:
    for mode in ['self-adjacent', 'self-pool1', 'self-pool2', 'self-pool4']:
        candidate = run(mode)
        if args.reference:
            assert candidate == run(mode, original=True), 'Production finite fixture changed'
    result['passed'] = True
finally:
    (root / 'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(root,flush=True)
