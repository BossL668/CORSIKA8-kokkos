"""Compare pool results with the same ordered history in each independent worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe', required=True, type=Path)
parser.add_argument('--plain', type=Path)
parser.add_argument('--reference', action='store_true')
parser.add_argument('--material', choices=['air', 'ice', 'firn', 'rock'], required=True)
parser.add_argument('--directory', required=True, type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='fluka-worker-', dir=args.directory)).resolve()
count = 288 if args.material == 'air' else 144
result = {'passed': False, 'material': args.material, 'directory': str(root), 'runs': [],
          'pool_checks': [], 'cross_history_invariance_claimed': False,
          'scope': 'same per-worker history, assignments, RNG keys and ordered merge'}
environment = os.environ.copy()
environment['CORSIKA_TEST_FLUKA_MATERIAL'] = args.material
environment['CORSIKA_TEST_FLUKA_INITIAL_SEED'] = '2026'
environment.pop('CORSIKA_TEST_ORIGINAL_FLUKA_WORKER', None)


def run(label, program, mode, expected):
    work = root / label
    work.mkdir()
    output = work / 'responses.txt'
    command = [str(program.resolve()), mode, str(output)]
    with (work / 'run.log').open('w') as log:
        completed = subprocess.run(command, cwd=work, env=environment,
            stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    record = {'label': label, 'mode': mode, 'command': command,
              'exit_code': completed.returncode, 'output': str(output)}
    result['runs'].append(record)
    assert completed.returncode == 0, str(work / 'run.log')
    data = output.read_bytes()
    lines = data.decode().splitlines()
    assert lines[-1] == f'complete {len(expected)} 2'
    starts = [i for i, line in enumerate(lines) if line.startswith('event ')]
    assert len(starts) == 2 * len(expected)
    starts.append(len(lines) - 1)
    events = [lines[starts[i]:starts[i+1]] for i in range(len(starts)-1)]
    assert [int(e[0].split()[1]) for e in events] == expected * 2
    for event in events:
        assert int(event[0].split()[3]) == len(event) - 1
    record.update(sha256=hashlib.sha256(data).hexdigest(), requests=len(expected), passes=2)
    return data, events


try:
    sequence = list(range(count))
    direct, _ = run('direct', args.probe, 'direct', sequence)
    fresh_direct, _ = run('direct-fresh-repeat', args.probe, 'direct', sequence)
    assert direct == fresh_direct, 'Fresh direct process with the same history changed'
    if args.reference:
        reference, _ = run('reference', args.probe, 'reference', sequence)
        assert direct == reference, 'Same-history direct execution differs from production'
    if args.plain:
        plain, _ = run('plain', args.plain, 'direct', sequence)
        assert direct == plain, 'Plain build differs from reference-enabled build'
    for workers in [1, 2, 4]:
        pool_bytes, pooled = run(f'pool{workers}', args.probe, f'pool{workers}', sequence)
        repeated, _ = run(f'pool{workers}-fresh-repeat', args.probe, f'pool{workers}', sequence)
        assert pool_bytes == repeated, 'Fresh process with the same configuration changed'
        ordered = {}
        worker_sequences = []
        for worker in range(workers):
            assigned = [i for i in sequence if (i // 13) % workers == worker]
            worker_sequences.append(assigned)
            _, events = run(f'history-{workers}-{worker}', args.probe,
                            f'direct-history:{workers}:{worker}', assigned)
            for pass_id in range(2):
                for j, identifier in enumerate(assigned):
                    key = (pass_id, identifier)
                    assert key not in ordered, 'A request was assigned more than once'
                    ordered[key] = events[pass_id * len(assigned) + j]
        assert len(ordered) == 2 * count
        merged = [ordered[pass_id, identifier] for pass_id in range(2) for identifier in sequence]
        assert pooled == merged, 'IPC pool changed a worker history or its ordered output'
        result['pool_checks'].append({'workers': workers, 'batch_size': 13,
            'worker_sequences': worker_sequences, 'fresh_process_repeat_equal': True,
            'same_worker_history_equal': True, 'requests_per_pass': count})
    result['passed'] = True
except Exception as error:
    result['error'] = repr(error)
    raise
finally:
    (root / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(root, flush=True)
