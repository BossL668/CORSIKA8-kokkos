"""Exercise both real C++ coordinators on identical closed inputs; no shower claim."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--production', type=Path, required=True)
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--mode', choices=['io', 'cli', 'process', 'firn', 'prefix'], required=True)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
parent = args.work
args.work = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
(parent / 'LATEST.json').write_text(json.dumps({'directory': args.work.name}) + '\n')
checks = 0
records = []


def require(condition, message):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(message)


def call(binary, operation, success=True, **fields):
    request = dict(operation=operation, **fields)
    result = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    reply = json.loads(result.stdout)
    require(reply['ok'] == success and result.returncode == (0 if success else 2),
            (request, result.returncode, result.stdout, result.stderr))
    require('AddressSanitizer' not in result.stderr and 'runtime error:' not in result.stderr,
            result.stderr)
    records.append(dict(binary=str(binary), request=request, result=reply))
    return reply.get('value') if success else reply['error']


def compare(operation, **fields):
    old = call(args.production, operation, **fields)
    new = call(args.candidate, operation, **fields)
    require(new == old, (operation, old, new))
    return new


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(history, energy, columns=15):
    fields = ['11', str(history), '0', '0', '0', str(energy), '2', '0', '0', '0', '0', '0', '1', '0']
    return ' '.join(fields + ['0'] * (columns - 14))


def spool(path, count):
    # Independently encode the production's documented little-endian wire format.
    header = b'C8_RADIO_TRACKS\0' + struct.pack('<IIQ', 1, 140, count)
    require(len(header) == 32, 'Spool header length')
    track = struct.pack('<17di', *([0.] * 12 + [0., 1., 1., .9, 1.]), 11)
    path.write_bytes(header + track * count)


def io_tests():
    defaults = compare('defaults')
    require(defaults == dict(backend='kokkos-proposal', egs4_stepfc=1, seed=1, events=1,
                            prefix_workers=0, split_energy=0, energy=0, memory=.5,
                            frontier_energy=0, timeout=7200, worker='', output='',
                            devices=[], physics=[]), defaults)
    for seed, index in [(1, 0), (1, 1), (91, 1024), (4294967299, 15), (9223372036854775806, 254)]:
        value = compare('seed', seed=seed, index=index)
        expected = int.from_bytes(hashlib.sha256(f'{seed}:{index}'.encode()).digest()[:4], 'little') or 1
        require(value == expected, 'Independent little-endian SHA seed')
    data = args.work / 'hash.dat'
    data.write_bytes(bytes(range(256)) * 1027)
    require(compare('digest', path=str(data)) == sha(data), 'File SHA256')
    headers = [('C8_STATIC_FRONTIER_V1', 14), ('C8_STATIC_FRONTIER_V2 fixture', 15),
               ('C8_STATIC_FRONTIER_V3 fixture', 15), ('C8_CPU_FRONTIER_V1 fixture', 16)]
    for version, (header, columns) in enumerate(headers):
        source = args.work / f'frontier-{version}.txt'
        lines = [row(1, 3, columns), row(2, 5, columns), row(3, 5, columns), row(4, 1, columns)]
        source.write_text(header + '\n' + '\n'.join(lines) + '\n')
        reports, shards = [], []
        for tag, binary in [('old', args.production), ('new', args.candidate)]:
            paths = [args.work / f'{version}-{tag}-{i}.txt' for i in range(3)]
            reports.append(call(binary, 'partition', source=str(source), destinations=list(map(str, paths))))
            shards.append([p.read_bytes() for p in paths])
        require(reports[0] == reports[1] and shards[0] == shards[1], 'Full partition output')
        require(reports[0]['counts'] == [1, 1, 2] and reports[0]['cost_energy_GeV'] == [5, 5, 4], reports)
        require(reports[0]['weighted_energy_GeV'] == 28, 'Weights preserved')
        require(shards[0][0].decode().splitlines()[1] == lines[2], 'Energy/history tie ordering')
        for bad, replacement in [('duplicate', lines + [lines[0]]), ('nan', [row(10, 'nan', columns)]),
                                  ('negative', [row(10, -1, columns)]), ('columns', ['11 1'])]:
            source.write_text(header + '\n' + '\n'.join(replacement) + '\n')
            for tag, binary in [('old', args.production), ('new', args.candidate)]:
                call(binary, 'partition', success=False, source=str(source),
                     destinations=[str(args.work / f'bad-{version}-{bad}-{tag}')])
    empty = args.work / 'empty.txt'
    empty.write_text('C8_STATIC_FRONTIER_V1\n')
    for tag, binary in [('old', args.production), ('new', args.candidate)]:
        result = call(binary, 'partition', source=str(empty), destinations=[str(args.work / f'empty-{tag}')])
        require(result['roots'] == 0 and result['counts'] == [0], 'Empty shard preserved')
    sources = []
    for i in range(2):
        path = args.work / f'join-{i}'
        path.write_text('C8_STATIC_FRONTIER_V3 tree\n' + row(i + 1, i + 5) + '\n')
        sources.append(str(path))
    joined = []
    for tag, binary in [('old', args.production), ('new', args.candidate)]:
        target = args.work / f'joined-{tag}'
        report = call(binary, 'join', sources=sources, destination=str(target))
        require(report['roots'] == 2 and report['joined_sha256'] == sha(target), report)
        joined.append(target.read_bytes())
        call(binary, 'join', success=False, sources=sources + sources,
             destination=str(args.work / f'join-duplicate-{tag}'))
    require(joined[0] == joined[1], 'Joined frontier bytes')
    spools = []
    for i, count in enumerate([4, 2, 4, 0]):
        path = args.work / f'tracks-{i}.bin'
        spool(path, count)
        spools.append(str(path))
    audit = compare('radio', sources=spools, workers=[2, 7])
    require(audit['worker_records'] == [6, 4] and audit['records'] == 10, audit)
    require([s['worker'] for s in audit['sources']] == [2, 7, 2, None], 'Stable largest-first spool order')
    for binary in [args.production, args.candidate]:
        call(binary, 'radio', success=False, sources=spools, workers=[])
        call(binary, 'radio', success=False, sources=spools, workers=[2, 2])
        call(binary, 'radio', success=False, sources=spools + spools[:1], workers=[2])
        incomplete = args.work / 'incomplete.bin'
        incomplete.write_bytes(b'C8_RADIO_TRACKS\0' + struct.pack('<IIQ', 1, 140, 2**64-1))
        call(binary, 'radio', success=False, sources=[str(incomplete)], workers=[2])
    parts = []
    for i, value in enumerate([3., -2., 4.]):
        part = args.work / f'part-{i}'
        call(args.candidate, 'fixture', root=str(part), value=value)
        compare('validate', root=str(part))
        parts.append(str(part))
    products = []
    for tag, binary in [('old', args.production), ('new', args.candidate)]:
        destination = args.work / f'merged-{tag}'
        products.append(call(binary, 'merge', parts=parts, destination=str(destination)))
        compare('validate', root=str(destination))
        field = call(binary, 'read', path=str(destination / 'CoREAS/observers.parquet'))
        require(field['value'] == [5., 10.], 'Sum fields, never powers')
        particle = call(binary, 'read', path=str(destination / 'particles/particles.parquet'))
        interaction = call(binary, 'read', path=str(destination / 'interactions/interactions.parquet'))
        require(particle['value'] == [3., -2., 4.] and interaction['value'] == [3.], 'Record/first interaction ordering')
        marker = args.work / f'metadata-{tag}.json'
        call(binary, 'save', path=str(marker), value={'a': 3})
        first = marker.read_bytes()
        call(binary, 'save', success=False, path=str(marker), value={'a': 4})
        require(marker.read_bytes() == first, 'Closed metadata may not be overwritten')
    require(products[0] == products[1], 'All seven real Parquet output SHA256 values')
    bad = args.work / 'bad-grid'
    call(args.candidate, 'fixture', root=str(bad), value=1., shift=.5)
    for tag, binary in [('old', args.production), ('new', args.candidate)]:
        call(binary, 'merge', success=False, parts=parts + [str(bad)],
             destination=str(args.work / f'grid-error-{tag}'))
    (bad / 'profile/config.yaml').write_text('fixture: incompatible\n')
    for tag, binary in [('old', args.production), ('new', args.candidate)]:
        call(binary, 'merge', success=False, parts=parts + [str(bad)],
             destination=str(args.work / f'config-error-{tag}'))


def firn_tests():
    parts = [args.work / 'prefix', args.work / 'worker']
    for part, value, execution in zip(parts, [2., -1.], ['CPU', 'Cuda']):
        call(args.candidate, 'fixture', root=str(part), value=value, firn=True, execution=execution)
        call(args.candidate, 'validate', root=str(part))
    target = args.work / 'merged'
    call(args.candidate, 'merge', parts=list(map(str, parts)), destination=str(target))
    for filename in ['radio_waveforms.parquet', 'radio_spectra.parquet']:
        result = call(args.candidate, 'read', path=str(target / 'firn_radio' / filename))
        require(result['field'] == [1., 2., -1., -2.], 'Sum signed fields, not powers')
        require(result['observer'] == [0., 0., 1., 1.], 'Preserve observer IDs')
    for label, options in [('observer', dict(observer_shift=.5)),
                           ('reflection', dict(reflection=False))]:
        bad = args.work / ('bad-' + label)
        call(args.candidate, 'fixture', root=str(bad), value=1., firn=True, **options)
        call(args.candidate, 'merge', success=False, parts=[str(parts[0]), str(bad)],
             destination=str(args.work / ('rejected-' + label)))

    replay = parts[0] / 'firn_radio' / 'summary.yaml'
    call(args.candidate, 'validate_firn_replay', success=False, root=str(parts[0]), expected=3)
    replay.write_text('shower_0:\n  replayed_prefix_segments: 3\n')
    call(args.candidate, 'validate_firn_replay', root=str(parts[0]), expected=3)
    call(args.candidate, 'validate_firn_replay', success=False, root=str(parts[0]), expected=4)


def prefix_tests():
    # CPU splitting depends on CPU workers, independently of GPU handoff settings.
    for energy in [1.e4, 1.e8]:
        for workers in [1, 5, 16, 64]:
            for devices in [['0'], ['0', '1', '2', '3']]:
                for frontier in [0., 12345.]:
                    for explicit in [0., 321.]:
                        value = call(args.candidate, 'cpu_split_energy', energy=energy,
                                     workers=workers, devices=devices,
                                     frontier_energy=frontier, split_energy=explicit)
                        expected = explicit or energy / (96 * workers)
                        require(value == expected, ('CPU split energy', value, expected))
    for explicit in [0., 321.]:
        value = call(args.candidate, 'cpu_split_energy', energy=1.e8, workers=0,
                     devices=['0', '1'], split_energy=explicit)
        require(value == 0., 'Serial prefix does not use a CPU splitting threshold')


def cli_tests():
    for flags, expected in [([], False), (['--device', '0'], False),
                            (['--device', '0,1'], True), (['--devices', '0', '1'], True),
                            (['--prefix-workers', '0'], True), (['--prefix-workers=2'], True)]:
        require(compare('requested', args=['probe', *flags]) == expected, flags)
    base = ['--device', '0,1', '-E', '10000', '-f', str(args.work / 'unused')]
    cases = [['--help'], base + ['--gpu-memory-fraction', '1'],
             base + ['--em-backend', 'proposal'], base + ['--radio-backend', 'cpu'],
             base + ['--kokkos-execution', 'openmp'], base + ['--prefix-workers', '255'],
             base + ['--egs4-stepfc', '.2'], base + ['--frontier-in', 'x'],
             base + ['--hadronic-workers', '2']]
    for flags in cases:
        results = []
        for binary in [args.production, args.candidate]:
            result = subprocess.run([str(binary), 'entry', *flags], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            require(result.returncode == (0 if flags == ['--help'] else 1), (flags, result.stderr))
            require('runtime error:' not in result.stderr and 'AddressSanitizer' not in result.stderr,
                    result.stderr)
            results.append((result.returncode, result.stdout, result.stderr))
        if flags == ['--help']:
            old_help = ' '.join(results[0][1].split())
            new_help = ' '.join(results[1][1].split())
            old_cap = 'Static CPU split energy cap [GeV]; 0 uses the EM handoff cap'
            new_cap = 'CPU split energy [GeV]; 0 uses primary energy / (96 * prefix workers)'
            require(old_cap in old_help and new_cap in new_help, 'Document CPU default')
            require(old_help.replace(old_cap, new_cap) == new_help, 'Other help unchanged')
            require(results[0][2] == results[1][2], 'Help stderr unchanged')
        else:
            require(results[0] == results[1], ('Original CLI behavior', flags, results))
    require(not (args.work / 'unused').exists(), 'Rejected CLI must not launch a shower')


def process_tests():
    # Actual fork/exec/wait/cancellation with native C++ coordination. The child
    # deliberately does not simulate physics; full shower integration is separate.
    sleeper = args.candidate.parent / 'coordinator_sleep_child'
    sentinel = subprocess.Popen(['/usr/bin/sleep', '30'])
    try:
        for kind, worker in [('failure', Path('/usr/bin/false')), ('timeout', sleeper)]:
            for tag, binary in [('old', args.production), ('new', args.candidate)]:
                output = args.work / f'{kind}-{tag}'
                start = time.monotonic()
                result = subprocess.run([str(binary), 'entry', '--device', '0,1', '-E', '10000',
                    '-f', str(output), '--multigpu-worker', str(worker), '--multigpu-timeout', '.2'],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                require(result.returncode == 1, (result.returncode, result.stderr))
                require('runtime error:' not in result.stderr and 'AddressSanitizer' not in result.stderr, result.stderr)
                require((output / 'INCOMPLETE.json').is_file() and not (output / 'COMPLETE.json').exists(), 'Failure publication')
                message = json.loads((output / 'INCOMPLETE.json').read_text())['error']
                require(('Process failed:' if kind == 'failure' else 'Process deadline exceeded:') in message, message)
                owned = json.loads((output / 'prefix/OWNED_PROCESS.json').read_text())['pid']
                require(not Path(f'/proc/{owned}').exists(), 'Owned child reaped')
                require(sentinel.poll() is None, 'Unrelated process survives cleanup')
                require(time.monotonic() - start < 10, 'Bounded cleanup')
                records.append(dict(binary=str(binary), kind=kind, message=message))
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


try:
    tests = {'io': io_tests, 'cli': cli_tests, 'process': process_tests,
             'firn': firn_tests, 'prefix': prefix_tests}
    tests[args.mode]()
    (args.work / 'RESULTS.json').write_text(json.dumps(dict(passed=True, checks=checks, records=records), indent=2) + '\n')
    print(f'CHECKS={checks}')
    print('evidence=' + str(args.work))
finally:
    (args.work / 'CALLS.json').write_text(json.dumps(records, indent=2) + '\n')
