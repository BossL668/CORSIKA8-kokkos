#!/usr/bin/env python3
"""Replay accepted CUDA commands with an independent Kokkos Tools/CUPTI plugin.

Preserves full activity traces and verifies that instrumentation leaves CSVs,
material tables, diagnostics and queue counters unchanged. Run on the GPU host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--guard', type=Path, default=Path(__file__).with_name('run_guarded_diagnostic.py'))
    parser.add_argument('--cases', nargs='+', default=['photon_up', 'nue_forced'])
    parser.add_argument('--variants', nargs='+', choices=['batched','resident'], default=['batched','resident'])
    parser.add_argument('--timeout', type=float, default=900.)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    accepted = json.loads(args.acceptance.read_text())
    report = []
    for case in args.cases:
        for variant in args.variants:
            previous = next(r for r in accepted if r['case'] == case and r['variant'] == variant)
            assert previous['accelerator']['execution_space'] == 'Cuda'
            tag = case + '_' + variant
            command = list(previous['command'])
            output = root / tag
            command[command.index('--output') + 1] = str(output)
            binary = Path(command[0])
            if digest(binary) != previous['binary_sha256']:
                raise RuntimeError('accepted CUDA binary has changed')
            work = root / (tag + '_work')
            work.mkdir()
            trace = root / (tag + '.jsonl')
            env = dict(os.environ, KOKKOS_TOOLS_LIBS=str(args.plugin.resolve()),
                       C8_INTERFACE_TRACE_FILE=str(trace), OMP_NUM_THREADS='256',
                       OMP_PROC_BIND='spread', OMP_PLACES='threads', OPENBLAS_NUM_THREADS='1')
            wrapped = [sys.executable, str(args.guard.resolve()), '--gpu',
                       '--timeout', str(args.timeout), '--output', str(root / (tag + '_guard')),
                       '--'] + command
            print('PROFILE', tag, flush=True)
            process = subprocess.run(wrapped, env=env, cwd=work)
            summary_path = output / 'terrain_run.yaml'
            summary = yaml.safe_load(summary_path.read_text()) if summary_path.exists() else {}
            # The accepted manifest is JSON: YAML integer PDG mapping keys were
            # serialized as strings. Apply the same representation, preserving
            # every diagnostic value and its exact numeric comparison.
            summary = json.loads(json.dumps(summary))
            hashes = {p.name: digest(p) for p in (output / 'terrain').glob('*.csv')}
            errors, finalized, kernels, copies = [], [], 0, 0
            if trace.exists():
                with trace.open() as stream:
                    for line in stream:
                        event = json.loads(line)
                        if event['event'] == 'error':
                            errors.append(event)
                        if event['event'] == 'finalize':
                            finalized.append(event)
                        kernels += event['event'] == 'gpu_kernel'
                        copies += event['event'] == 'gpu_copy'
            checks = {
                'completed': process.returncode == 0 and summary.get('complete') is True,
                'binary_unchanged': digest(binary) == previous['binary_sha256'],
                'exact_csv': hashes == previous['csv_sha256'],
                'exact_material_tables': summary.get('material_tables') == previous['material_tables'],
                'exact_diagnostics': summary.get('diagnostics') == previous['diagnostics'],
                'exact_accelerator_counters': summary.get('accelerator') == previous['accelerator'],
                'actual_gpu_kernels': kernels > 0,
                'actual_gpu_copies': copies > 0,
                'complete_trace': len(finalized) == 1 and not errors
                    and not finalized[0]['failed'] and finalized[0]['dropped_records'] == 0,
            }
            report.append(dict(case=case, variant=variant, command=command, checks=checks,
                               csv_sha256=hashes, trace=str(trace), errors=errors,
                               gpu_kernels=kernels, gpu_copies=copies,
                               accelerator=summary.get('accelerator')))
            (root / 'profile_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
            if not all(checks.values()):
                raise RuntimeError(f'{tag}: {checks}; errors={errors}')
            print('PASS', tag, 'kernels=', kernels, 'copies=', copies, flush=True)


if __name__ == '__main__':
    main()
