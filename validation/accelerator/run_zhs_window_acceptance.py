#!/usr/bin/env python3
"""Small end-to-end ZHS window regression; never replaces production binaries.

Compare frozen-before and rebuilt-after N>1 runs, then compare the corrected
window with a four-ns wider recording. Memory guards stop only diagnostic
children. Large ensemble data are neither modified nor implicitly accepted.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

from run_nmulti_memory_acceptance import monitor, save, sha256
from compare_event_memory_metadata import compare as compare_metadata


def read_yaml(path):
    with path.open() as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def radio(path, count):
    data = pq.read_table(path / 'observers.parquet', use_threads=False)
    cfg = read_yaml(path / 'config.yaml')
    observers = list(cfg['observers'].values())
    bins = int(round(observers[0]['number of bins']))
    nobs = len(observers)
    fields = np.column_stack([data[c].to_numpy() for c in ('Ex', 'Ey', 'Ez')])
    assert len(data) == count * nobs * bins and np.isfinite(fields).all()
    ids = data['shower'].to_numpy().reshape(count, nobs, bins)
    assert np.array_equal(ids[:, 0, 0], np.arange(count))
    return fields.reshape(count, nobs, bins, 3), cfg


def compare(root, backend, primary, count, reference_root=None):
    name = f'{backend}_{primary}'
    before = (reference_root or root) / f'{name}_before'
    after, wide = (root / f'{name}_{stage}' for stage in ('after', 'wide'))
    result = dict(nonradio={}, radio={})
    files = sorted(p.relative_to(before) for p in before.rglob('*.parquet'))
    assert files == sorted(p.relative_to(after) for p in after.rglob('*.parquet'))
    for rel in files:
        if rel.parts[0] in ('ZHS', 'CoREAS'):
            continue
        a = pq.read_table(before / rel, use_threads=False)
        b = pq.read_table(after / rel, use_threads=False)
        c = pq.read_table(wide / rel, use_threads=False)
        equal = a.equals(b, check_metadata=False) and b.equals(c, check_metadata=False)
        result['nonradio'][str(rel)] = dict(equal=equal, rows=len(a))
        if not equal:
            raise AssertionError(f'{name}: nonradio output changed: {rel}')
    archives = sorted(p.relative_to(before) for p in before.rglob('*.npz'))
    assert archives == sorted(p.relative_to(after) for p in after.rglob('*.npz'))
    for rel in archives:
        with np.load(before / rel) as a, np.load(after / rel) as b, np.load(wide / rel) as c:
            equal = set(a.files) == set(b.files) == set(c.files) and all(
                np.array_equal(a[k], b[k]) and np.array_equal(b[k], c[k]) for k in a.files)
        result['nonradio'][str(rel)] = dict(equal=equal)
        assert equal, f'{name}: interaction histogram changed: {rel}'
    for alg in ('CoREAS', 'ZHS'):
        a, cfg_a = radio(before / alg, count)
        b, cfg_b = radio(after / alg, count)
        w, cfg_w = radio(wide / alg, count)
        assert cfg_a == cfg_b, 'Output schema/window changed'
        # This application's time column is relative to EACH window's start,
        # so the wider window's index 2 corresponds to narrow index 0.
        obs_b, obs_w = list(cfg_b['observers'].values()), list(cfg_w['observers'].values())
        for x, y in zip(obs_b, obs_w):
            assert abs((x['start time'] - y['start time']) - 2.) < 1e-7
            assert x['sampling frequency'] == y['sampling frequency'] == 1.
        crop = w[:, :, 2:-2, :]
        assert b.shape == crop.shape
        diff = np.abs(a - b)
        record = dict(shape=list(b.shape), unchanged_samples=int(np.count_nonzero(a == b)),
                      changed_samples=int(np.count_nonzero(a != b)),
                      max_change_Vpm=float(diff.max()),
                      interior_max_change_Vpm=float(diff[:, :, 1:-1].max()),
                      wider_window_max_difference_Vpm=float(np.abs(b - crop).max()))
        if alg == 'CoREAS':
            assert np.array_equal(a, b), 'CoREAS must be unchanged by this fix'
        else:
            assert np.array_equal(a[:, :, 1:-1], b[:, :, 1:-1]), 'Interior ZHS changed'
            scale = max(float(np.abs(crop).max()), 1e-20)
            record['wider_window_peak_normalized_difference'] = float(np.abs(b-crop).max()/scale)
            # Different window origins involve absolute-time subtraction and
            # may change last-bit weights. Compare with an absolute floor too.
            assert np.abs(b-crop).max() <= 4e-18 + 2e-8 * scale, 'ZHS window invariance failed'
        result['radio'][alg] = record
    if backend != 'proposal':
        summary = read_yaml(after / 'gpu_em/summary.yaml')
        assert set(summary) == {f'shower_{i}' for i in range(count)}
        for event in summary.values():
            assert event['complete']
            assert event['statistics']['radio']['fixed_point_overflows'] == 0
        result['all_events_complete'] = True
    metadata = compare_metadata(before, after)
    # Accept ONLY the intentional extra ZHS contributions in complete edge
    # bins. All other physics counters, table hashes, energy ledgers, summary
    # schema/order and backend-reuse flags remain strict comparisons.
    edge_counts, timing_oracle_counts, unexpected = [], [], []
    old_gpu = read_yaml(before / 'gpu_em/summary.yaml') if backend != 'proposal' else {}
    new_gpu = read_yaml(after / 'gpu_em/summary.yaml') if backend != 'proposal' else {}
    for item in metadata['unexpected']:
        path = item['path'].split('.')
        is_edge_count = (item['file'] == 'gpu_em/summary.yaml' and len(path) == 4
                         and path[0].startswith('shower_')
                         and path[1:] == ['statistics', 'radio', 'zhs_contributions']
                         and type(item['before']) is int and type(item['after']) is int
                         and item['after'] >= item['before'])
        # KokkosShowerReport stages measured final_state_time_ms into a
        # retrospective HadronicWorkQueue AFTER transport. Its simulated batch
        # count can change with wall time; it never drives this scalar campaign.
        is_oracle_batch = (item['file'] == 'gpu_em/summary.yaml'
                           and path[1:3] == ['statistics', 'hadronic_worker_oracle']
                           and ((len(path) == 4 and path[-1] == 'batches')
                                or (len(path) == 6 and path[3] == 'worker_loads'
                                    and path[4].isdigit() and path[5] == 'batches'))
                           and type(item['before']) is int and type(item['after']) is int
                           and item['before'] >= 0 and item['after'] >= 0
                           and all(m.get(path[0], {}).get('statistics', {})
                                   .get('hadronic_worker_oracle', {}).get('mode')
                                   == 'retrospective_measured_final_state_only'
                                   for m in (old_gpu, new_gpu)))
        (edge_counts if is_edge_count else timing_oracle_counts if is_oracle_batch
         else unexpected).append(item)
    metadata['intentional_zhs_edge_contributions'] = edge_counts
    metadata['walltime_dependent_retrospective_batch_counts'] = timing_oracle_counts
    metadata['unexpected'] = unexpected
    metadata['pass_'] = not unexpected
    result['metadata'] = metadata
    assert not unexpected, f'{name}: unrelated metadata changed: {unexpected}'
    result['pass'] = True
    save(root / f'{name}_comparison.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stage', choices=('before', 'after', 'compare'), required=True)
    ap.add_argument('--before', type=Path)
    ap.add_argument('--cuda-executable', type=Path)
    ap.add_argument('--openmp-executable', type=Path)
    ap.add_argument('--reference-root', type=Path,
                    help='Reuse frozen *_before data from this root in compare mode')
    ap.add_argument('--backends', nargs='+', choices=('proposal', 'openmp', 'cuda'),
                    default=['proposal', 'openmp', 'cuda'])
    args = ap.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[3]
    source = Path(__file__).resolve().parents[2]
    data = project / 'install/cuda-openmp/share/corsika/data'
    antennas = Path('/home/yuhanglu/21CMA/data/antennas_nwu_coordinates_test.txt')
    results = {}
    for backend in args.backends:
        for primary, pdg, energy, count in [('photon', '22', '1', 4), ('proton', '2212', '10', 2)]:
            if args.stage == 'compare':
                results[f'{backend}_{primary}'] = compare(
                    root, backend, primary, count, args.reference_root)
                continue
            versions = ['before'] if args.stage == 'before' else ['after', 'wide']
            for version in versions:
                label = f'{backend}_{primary}_{version}'
                executable = (args.before if version == 'before' else
                              args.cuda_executable if backend == 'cuda' else args.openmp_executable)
                assert executable is not None and executable.is_file()
                executable = executable.resolve()
                threads = 4 if backend == 'openmp' else 1
                env = dict(os.environ, CORSIKA_DATA=str(data), OMP_NUM_THREADS=str(threads),
                           OMP_THREAD_LIMIT=str(threads), OMP_PROC_BIND='spread', OMP_PLACES='cores',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           FLUPRO='/home/yuhanglu/fluka',
                           LD_LIBRARY_PATH=str(project / 'install/cuda-openmp/lib/corsika')
                               + ':/home/yuhanglu/miniconda3/envs/corsika_venv/lib')
                cmd = [str(executable), '-p', pdg, '-E', energy, '-N', str(count),
                       '-s', '26090729', '-z', '0', '-a', '0', '--emthin', '1e-6',
                       '--antenna-file', str(antennas), '--geomagnetic-model', 'IGRF14',
                       '--geomagnetic-year', '2027', '--verbosity', 'info', '-f', str(root / label),
                       '--radio-window-duration-ns', '404' if version == 'wide' else '400',
                       '--radio-pretrigger-ns', '12' if version == 'wide' else '10']
                if backend == 'proposal':
                    cmd += ['--em-backend', 'proposal', '--radio-backend', 'cpu']
                else:
                    cmd += ['--em-backend', 'kokkos-proposal', '--radio-backend', 'kokkos',
                            '--kokkos-execution', backend, '--kokkos-num-threads', str(threads),
                            '--gpu-min-batch', '16', '--gpu-resident-batch-limit', '4096',
                            '--gpu-memory-fraction', '.15']
                done = root / f'{label}_provenance.json'
                digest = sha256(executable)
                if done.exists():
                    old = json.loads(done.read_text())
                    assert old['command'] == cmd and old['binary_sha256'] == digest
                    assert old['returncode'] == 0 and old['failure'] is None
                    print('REUSE', label, flush=True)
                    continue
                assert not (root / label).exists(), f'Incomplete output preserved: {label}'
                print('RUN', label, flush=True)
                result = monitor(cmd, root, env, label, timeout=600)
                result['binary_sha256'] = digest
                result['source_script_sha256'] = sha256(Path(__file__))
                save(done, result)
    if args.stage == 'compare':
        save(root / 'shower_comparison.json', results)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
