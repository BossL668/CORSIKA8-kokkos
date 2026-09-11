#!/usr/bin/env python3
"""Unforced nu_tau pilot; keep preselected regression seeds explicitly separate.

No changes to cross sections, decay lifetimes, geometry, or production binaries.
Each independent event has its own working directory and resource guard.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_nutau_ppt_demo import sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2, choices=[1, 2])
    p.add_argument('--threads', type=int, default=2, choices=[1, 2, 4])
    p.add_argument('--timeout', type=float, default=900.)
    p.add_argument('--thick-reference', type=Path,
                   help='Previously geometry-validated real DEM chord directory')
    p.add_argument('--seed-scan',type=Path,help='100 PeV diagnostic prefilter CSV; run first three interior candidates')
    a = p.parse_args()
    root, ref, binary = a.output.resolve(), a.reference.resolve(), a.binary.resolve()
    cache = binary.parents[1] / 'CMakeCache.txt'
    if 'CORSIKA_KOKKOS_BACKEND:STRING=OPENMP\n' not in cache.read_text():
        raise RuntimeError('OpenMP-only binary required; do not use the production GPU')
    spec = yaml.safe_load((ref / 'pilot_manifest.yaml').read_text())
    scene = ref / 'original_expanded_scene.yaml'
    scene_data = yaml.safe_load(scene.read_text())
    if sha256(Path(scene_data['geometry']['mesh_path'])) != spec['mesh_sha256']:
        raise RuntimeError('DEM hash mismatch')
    ray = next(j for j in spec['jobs'] if j['name'] == 'nutau10tev_unselected_s79101')
    if a.thick_reference:
        thick = yaml.safe_load((a.thick_reference / 'manifest.yaml').read_text())
        gate = yaml.safe_load((a.thick_reference / 'geometry_summary.yaml').read_text())
        if not gate['passed'] or gate['mesh_sha256'] != spec['mesh_sha256']:
            raise RuntimeError('Thick chord geometry provenance mismatch')
        spec['chord'] = thick['chord']
        direction = thick['chord']['direction']
        ray = dict(direction=direction, position=[x - 50*d for x, d in
                   zip(thick['chord']['entry'], direction)])
    jobs = []
    for index, energy in enumerate([1.e6, 3.e6, 1.e7]):
        for seed in ([910100 + 10 * index, 910101 + 10 * index]
                     if a.thick_reference else [910100 + 10 * index, 910101 + 10 * index, 515061]):
            group = 'previously_selected_control' if seed == 515061 else 'unselected'
            jobs.append(dict(name=f'nu_tau_{energy:g}GeV_s{seed}', energy_GeV=energy,
                             seed=seed, group=group))
    if a.seed_scan:
        if not a.thick_reference: raise ValueError('scan requires validated thick chord')
        with a.seed_scan.open() as f: candidates=list(csv.DictReader(f))
        selected=[r for r in candidates if .1*spec['chord']['length_m']<float(r['depth_m'])<.7*spec['chord']['length_m']][:3]
        jobs=[dict(name=f"nu_tau_1e8GeV_s{r['seed']}",energy_GeV=1e8,seed=int(r['seed']),
                   group='optical_depth_prefilter_NOT_unbiased',predicted_depth_m=float(r['depth_m'])) for r in selected]
        assert len(jobs)==3
    root.mkdir(parents=True, exist_ok=False)
    (root / 'runs').mkdir()
    manifest = dict(binary=str(binary), binary_sha256=sha256(binary), scene=str(scene),
                    mesh_sha256=spec['mesh_sha256'], chord=spec['chord'],
                    position=ray['position'], direction=ray['direction'], jobs=jobs,
                    forced_interaction=False, forced_decay=False, radio=False,
                    primary='nu_tau', channels='cc+nc', window_ns=100000.,
                    emcut_GeV=.01, hadcut_GeV=.3, mucut_GeV=.3, emthin=.001,
                    max_weight=1000., max_step_m=1., threads=a.threads,
                    concurrent_events=a.workers, gpu_used=False,
                    selected_seed_warning='515061 was preselected in the original mountain pilot; NOT an unbiased sample',
                    scope='natural primary flight, CTW high-energy CC/NC, native tau decay; limited weak/spin models')
    (root / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    if a.seed_scan:
        manifest.update(seed_scan=str(a.seed_scan.resolve()),seed_scan_sha256=sha256(a.seed_scan),
            scanned_seed_range=[1,1000000],selection='first three seeds with predicted depth in 10%-70% of first chord, before final states',
            selected_seed_warning='Optical-depth conditioned sample, not unbiased shower-rate measurement; verify predicted depths against full transport',
            emcut_GeV=.1,hadcut_GeV=10.,mucut_GeV=.3,emthin=.01,max_weight=50000.,
            scope='coarse-cut full-transport topology pilot, not converged shower or radio acceptance')
        (root / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OMP_PROC_BIND='false',
               OPENBLAS_NUM_THREADS='1')

    def run(job):
        name = job['name']
        destination = root / 'runs' / name
        work = root / 'runs' / (name + '_work')
        work.mkdir()
        command = [str(binary), '--scene', str(scene), '--output', str(destination),
                   '--primary', 'nu_tau', '--energy-GeV', str(job['energy_GeV']),
                   '--seed', str(job['seed']), '--position-m', *map(str, ray['position']),
                   '--direction', *map(str, ray['direction']), '--neutrino-channels', 'cc+nc',
                   '--tau-decay-model', 'tauola', '--emcut-GeV', '.01', '--hadcut-GeV', '.3',
                   '--mucut-GeV', '.3', '--emthin', '.001', '--max-weight', '1000',
                   '--max-step-m', '1', '--transport-window-ns', '100000',
                   '--magnetic-field', 'igrf14', '--magnetic-year', '2027',
                   '--em-backend', 'kokkos', '--threads', str(a.threads), '--batch', '64',
                   '--device-memory-MiB', '192', '--track-row-limit', '2000000',
                   '--aux-cache', str(ref / 'auxiliary')]
        for flag,key in [('--emcut-GeV','emcut_GeV'),('--hadcut-GeV','hadcut_GeV'),
                         ('--emthin','emthin'),('--max-weight','max_weight')]:
            command[command.index(flag)+1]=str(manifest[key])
        assert not any('force' in arg for arg in command)
        (root / 'runs' / (name + '_command.json')).write_text(json.dumps(
            dict(command=command, group=job['group'], binary_sha256=manifest['binary_sha256']), indent=2))
        guard = [sys.executable, str(Path(__file__).with_name('run_guarded_diagnostic.py')),
                 '--output', str(root / 'runs' / (name + '_guard')),
                 '--timeout', str(a.timeout), '--']
        print('RUN', name, job['group'], flush=True)
        completed = subprocess.run(guard + command, cwd=work, env=env, check=False)
        result = dict(**job, returncode=completed.returncode, complete=False)
        if (destination / 'terrain_run.yaml').exists():
            report = yaml.safe_load((destination / 'terrain_run.yaml').read_text())
            result.update(complete=report.get('complete', False),
                          interactions=report.get('neutrino', {}).get('interaction_count'),
                          tau_decays=report.get('tau', {}).get('tau_decay_count'))
        (root / 'runs' / (name + '_status.json')).write_text(json.dumps(result, indent=2))
        print('DONE', json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run, jobs))
    (root / 'run_status.json').write_text(json.dumps(results, indent=2))
    if any(not r['complete'] or r['returncode'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
