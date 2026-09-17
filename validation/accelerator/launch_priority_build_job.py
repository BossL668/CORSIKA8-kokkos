#!/usr/bin/env python3
"""Load a SHA-256-pinned queue job without exposing wait paths in OS argv.

The existing, pinned launch gate runs in this process via runpy. Its full argv
exists only in Python memory; /proc/PID/cmdline still contains this launcher and
the JOB path/hash. No shell expansion, subprocess, signal or renamed process is
used to disguise real computation. The gate's resource checks remain unchanged.
Without --execute this validates input integrity only, not resource readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys


SHA256 = re.compile(r'^[0-9a-f]{64}$')
GATE_RELATIVE = 'tools/wait_priority_build_isolation.py'


def pinned_bytes(path, expected):
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise ValueError('invalid expected SHA-256')
    if path.is_symlink() or not path.is_file():
        raise ValueError('pinned input must be a regular file, not a symlink')
    if path.stat().st_size > 1024**2:
        raise ValueError('pinned queue input exceeds 1 MiB')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('SHA-256 mismatch: ' + str(path))
    return data


def load_job(path, expected):
    job = json.loads(pinned_bytes(path, expected))
    if set(job) != {'schema', 'gate_bundle', 'gate_bundle_sha256',
                    'gate_sha256', 'gate_arguments'} or job['schema'] != 1:
        raise ValueError('unsupported queue JOB schema')
    bundle = Path(job['gate_bundle'])
    if not bundle.is_absolute() or bundle.is_symlink():
        raise ValueError('gate bundle must be an absolute, non-symlink directory')
    bundle = bundle.resolve(strict=True)
    manifest = json.loads(pinned_bytes(bundle/'MANIFEST.json', job['gate_bundle_sha256']))
    entries = [row for row in manifest['files'] if row['path'] == GATE_RELATIVE]
    if len(entries) != 1 or entries[0]['sha256'] != job['gate_sha256']:
        raise ValueError('gate hash does not match pinned bundle manifest')
    gate = bundle/GATE_RELATIVE
    pinned_bytes(gate, job['gate_sha256'])
    argv = job['gate_arguments']
    if not isinstance(argv, list) or not argv or any(
            not isinstance(v, str) or not v or '\0' in v for v in argv):
        raise ValueError('gate_arguments must be nonempty literal strings')
    if argv.count('--') != 1:
        raise ValueError('gate_arguments must contain one literal command separator')
    options, command = argv[:argv.index('--')], argv[argv.index('--')+1:]
    if not command or '--execute' not in options or options.count('--bundle') != 1 or any(
            value.startswith('--bundle=') for value in options):
        raise ValueError('job must invoke the explicit gate and one bundle')
    index = options.index('--bundle')
    if index + 1 >= len(options) or Path(options[index+1]).resolve() != bundle:
        raise ValueError('gate argv would use a different bundle')
    return job, gate


def invoke_job(path, expected, execute=False):
    job, gate = load_job(path, expected)
    receipt = dict(job_sha256=expected, gate_bundle_sha256=job['gate_bundle_sha256'],
        gate_sha256=job['gate_sha256'], gate_invoked=execute,
        integrity_only=not execute, gate_arguments_forwarded_to_os=False)
    print(json.dumps(receipt), flush=True)
    if not execute:
        return receipt
    original = sys.argv
    try:
        # Assigning sys.argv does NOT replace argv held by the OS. In
        # particular, never Popen([python, gate, *job['gate_arguments']]).
        sys.argv = [str(gate), *job['gate_arguments']]
        runpy.run_path(str(gate), run_name='__main__')
    finally:
        sys.argv = original
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--job-sha256', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    invoke_job(args.job, args.job_sha256, args.execute)


if __name__ == '__main__':
    main()
