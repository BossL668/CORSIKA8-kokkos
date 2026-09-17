#!/usr/bin/env python3
"""Compare frozen terrain, stateless reference and resident queues on one backend.

Keeps exact CSV hashes and native table hashes; never normalizes or relaxes
floating-point comparisons. Diagnostic records remain streamed to disk.
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
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--aux-cache', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backend', choices=['openmp', 'cuda'], required=True)
    parser.add_argument('--cases', nargs='+', default=['photon_up', 'electron_rock', 'nue_forced'])
    parser.add_argument('--timeout', type=float, default=480.)
    parser.add_argument('--threads', type=int, default=256)
    parser.add_argument('--require-cascade', action='store_true')
    parser.add_argument('--previous-acceptance', type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    previous_runs = json.loads(args.previous_acceptance.read_text()) if args.previous_acceptance else []
    guard = Path(__file__).with_name('run_guarded_diagnostic.py').resolve()
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND='spread',
               OMP_PLACES='threads', OPENBLAS_NUM_THREADS='1')
    cases = {
        'photon_up': ('photon', 1., -.01, 1, 1.e-6, 'none', False),
        'photon_down': ('photon', 1., .01, -1, 1.e-6, 'igrf14', False),
        'electron_rock': ('electron', 1., -.01, 1, 1.e-6, 'igrf14', False),
        'positron_rock': ('positron', 1., -.01, 1, 1.e-6, 'igrf14', False),
        'nue_forced': ('nu_e', 10000., -.01, 1, .1, 'none', True),
    }
    rows = []
    for case in args.cases:
        primary, energy, height, direction, thin, field, force = cases[case]
        previous = None
        variants = [('batched', args.binary), ('resident', args.binary)]
        if args.baseline:
            variants.insert(0, ('baseline', args.baseline))
        for variant, binary in variants:
            tag = case + '_' + variant
            output = args.root / tag
            work = args.root / (tag + '_work')
            work.mkdir()
            cmd = [str(binary.resolve()), '--scene', str(args.scene.resolve()), '--output', str(output),
                   '--primary', primary, '--energy-GeV', str(energy), '--seed', '67101',
                   '--position-m', '0', '0', str(height), '--direction', '0', '0', str(direction),
                   '--emthin', str(thin), '--em-backend', 'kokkos', '--threads', str(args.threads), '--batch', '64',
                   '--device-memory-MiB', '128', '--aux-cache', str(args.aux_cache.resolve()),
                   '--track-row-limit', '300000', '--magnetic-field', field]
            if force:
                cmd.append('--force-vertex-cc')
            if variant != 'baseline':
                cmd.extend(['--em-scheduler', variant, '--resident-capacity', '65536'])
            command = [sys.executable, str(guard), '--output', str(args.root / (tag + '_guard')),
                       '--timeout', str(args.timeout)]
            if args.backend == 'cuda':
                command.append('--gpu')
            print('RUN', args.backend, tag, flush=True)
            completed = subprocess.run(command + ['--'] + cmd, env=env, cwd=work)
            summary_path = output / 'terrain_run.yaml'
            summary = yaml.safe_load(summary_path.read_text()) if summary_path.exists() else {}
            # Archived JSON uses string keys for the YAML PDG dictionaries.
            summary = json.loads(json.dumps(summary))
            accelerator = summary.get('accelerator', {})
            diagnostics = summary.get('diagnostics', {})
            hashes = {p.name: digest(p) for p in (output / 'terrain').glob('*.csv')}
            checks = {
                'completed': completed.returncode == 0 and summary.get('complete') is True,
                'no_pending': accelerator.get('pending_particles') == 0,
                'backend': accelerator.get('execution_space') == ('Cuda' if args.backend == 'cuda' else 'OpenMP'),
                'material_correct': diagnostics.get('material_mismatches') == 0,
                'full_tracks': diagnostics.get('csv_truncated') is False and bool(hashes),
            }
            if previous:
                checks['exact_csv'] = hashes == previous['csv_sha256']
                checks['exact_material_tables'] = summary.get('material_tables') == previous['material_tables']
                checks['exact_diagnostics'] = diagnostics == previous['diagnostics']
                checks['exact_fallback_count'] = accelerator.get('specified_cpu_fallbacks') == previous['accelerator'].get('specified_cpu_fallbacks')
            if variant == 'resident':
                if args.backend == 'openmp':
                    checks['requested_threads'] = accelerator.get('execution_concurrency') == args.threads
                checks['retains_particles'] = 0 < accelerator.get('cpu_uploaded_particles', 0) < accelerator.get('particles_advanced', 0)
                checks['device_successors'] = accelerator.get('device_enqueued_particles', 0) > 0
                checks['bounded_workspace'] = accelerator.get('projected_peak_device_bytes', 2**63) <= 128 * 1024**2
                checks['bounded_host_staging'] = accelerator.get('peak_host_staging_particles', 2**63) <= 64
                if args.require_cascade:
                    waves=accelerator.get('batches',0)
                    calls=accelerator.get('resident_cascade_calls',0)
                    checks['multi_wavefront_backend'] = 0<calls<waves and accelerator.get('maximum_call_wavefronts',0)>1
                    checks['small_control_per_wavefront'] = accelerator.get('control_downloads')==waves
                    checks['buffered_output_checkpoints'] = accelerator.get('record_downloads')==calls
                    checks['all_records_delivered'] = accelerator.get('downloaded_records')==accelerator.get('particles_advanced')
                    checks['bounded_record_ledger'] = accelerator.get('peak_buffered_records',2**63)<=4096
                    if case!='nue_forced':
                        checks['complete_em_cascade_in_one_call'] = calls==1
            if previous_runs and variant!='baseline':
                old=next(r for r in previous_runs if r['case']==case and r['variant']==variant)
                checks['frozen_csv_unchanged'] = hashes==old['csv_sha256']
                checks['frozen_material_tables_unchanged'] = summary.get('material_tables')==old['material_tables']
                checks['frozen_diagnostics_unchanged'] = diagnostics==old['diagnostics']
                checks['frozen_fallbacks_unchanged'] = accelerator.get('specified_cpu_fallbacks')==old['accelerator'].get('specified_cpu_fallbacks')
            if case == 'nue_forced':
                checks['exercised_cpu_fallback'] = accelerator.get('specified_cpu_fallbacks', 0) > 0
            row = dict(case=case, variant=variant, command=cmd, binary_sha256=digest(binary.resolve()),
                       checks=checks, csv_sha256=hashes, diagnostics=diagnostics,
                       accelerator=accelerator, material_tables=summary.get('material_tables'))
            rows.append(row)
            (args.root / 'acceptance.json').write_text(json.dumps(rows, indent=2) + '\n')
            if not all(checks.values()):
                raise RuntimeError(f'{tag}: {checks}; error={summary.get("error")}')
            previous = row
            print('PASS', tag, 'steps=', accelerator['particles_advanced'],
                  'CPU_uploads=', accelerator.get('cpu_uploaded_particles'), flush=True)
    print('PASS', args.backend, len(rows), 'runs', flush=True)


if __name__ == '__main__':
    main()
