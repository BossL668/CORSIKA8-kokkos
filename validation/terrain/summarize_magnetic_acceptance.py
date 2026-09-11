#!/usr/bin/env python3
"""Summarize bounded terrain magnetic tests, without promoting them to ensemble acceptance.

Reads streamed traces and complete manifests. Reports strict comparison failures
instead of hiding them behind matching step counts or relaxed tolerances.
"""
import argparse
import json
from pathlib import Path
import yaml

from analyze_reference_runs import analyze
from compare_multimaterial_runs import compare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--suffix', default='final')
    p.add_argument('--cpu-suffix', help='Reuse a validated scalar run when only device admission changed')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {'scope': 'bounded magnetic terrain integration; NOT production physics certification',
              'integration_passed': True, 'strict_trace_passed': True,
              'radio_validated': False, 'ensemble_validated': False, 'cases': []}
    modes = ('cpu', 'openmp', 'cuda')
    directories = {m: args.root / f'magnetic_{m}_{args.cpu_suffix if m == "cpu" and args.cpu_suffix else args.suffix}'
                   for m in modes}
    result['input_directories'] = {m: str(path.resolve()) for m, path in directories.items()}
    manifests = {m: json.loads((directories[m] / 'acceptance.json').read_text())
                 for m in modes}
    cases = [row['case'] for row in manifests['cpu']]
    for mode, rows in manifests.items():
        if [r['case'] for r in rows] != cases:
            raise ValueError(f'{mode}: missing or differently ordered cases')
        if not all(r['complete'] and r['returncode'] == 0 and not r['stop_reason'] for r in rows):
            raise ValueError(f'{mode}: incomplete test')
    for case in cases:
        paths = {m: directories[m] / case for m in modes}
        summaries = {m: yaml.safe_load((path / 'terrain_run.yaml').read_text()) for m, path in paths.items()}
        for mode, s in summaries.items():
            for key in ('seed', 'energy_GeV', 'primary', 'direction', 'position_m', 'emthin',
                        'emcut_GeV', 'hadcut_GeV', 'mucut_GeV', 'mesh_sha256',
                        'air_magnetic_field_enu_T', 'rock_magnetic_field_enu_T',
                        'magnetic_height_reference', 'hadronic_models'):
                if s[key] != summaries['cpu'][key]:
                    raise ValueError(f'{case}/{mode}: configuration mismatch: {key}')
            if not any(s['air_magnetic_field_enu_T']) or any(s['rock_magnetic_field_enu_T']):
                raise ValueError(f'{case}/{mode}: material magnetic-field gate failed')
            if mode != 'cpu' and s['accelerator']['pending_particles'] != 0:
                raise ValueError('accelerator queue not drained')
        if summaries['openmp']['material_tables'] != summaries['cuda']['material_tables']:
            raise ValueError(f'{case}: physics-table identities differ')
        comparison = compare(paths['openmp'], paths['cuda'])
        result['strict_trace_passed'] &= comparison['passed']
        result['cases'].append({'name': case,
                                'integrity': {m: analyze(path) for m, path in paths.items()},
                                'cpu_direction_roundoff_canonicalizations': {
                                    m: summaries[m].get('accelerator', {}).get(
                                        'cpu_direction_roundoff_canonicalizations', 0)
                                    for m in modes},
                                'openmp_cuda': comparison})
    result['resources'] = {}
    for mode, manifest in manifests.items():
        samples = [s for row in manifest for s in row['samples']]
        result['resources'][mode] = {
            'binary_sha256': sorted(set(row['binary_sha256'] for row in manifest)),
            'peak_test_rss_KiB': max(s.get('rss_kib', 0) for s in samples),
            'peak_global_vram_increase_MiB': max(s.get('gpu_delta_mib', 0) for s in samples),
            'wall_seconds': [r['wall_seconds'] for r in manifest],
            'timing_scope': 'startup+cache/export+shower+CSV; concurrent validation, not a speed benchmark'}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'summary.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    lines = ['# 山体空气磁场集成验收', '',
             '本报告仅验收受控小事例的几何、输运与记录完整性；不代表跨界射电或系综物理验收。', '',
             '| 事例 | CPU 步数 | OpenMP 步数 | CUDA 步数 | 离散轨迹序列一致 | 严格浮点比较 |',
             '|---|---:|---:|---:|---|---|']
    for row in result['cases']:
        n = [row['integrity'][m]['diagnostics']['steps'] for m in modes]
        comparison = row['openmp_cuda']
        lines.append(f"| {row['name']} | {n[0]} | {n[1]} | {n[2]} | {comparison['discrete_trace_equal']} | {comparison['passed']} |")
    lines += ['', '严格比较保持原门限 rtol=1e-10、atol=1e-12，没有为通过而放宽。',
              'CPU 与加速调度使用不同随机流顺序，不能把单事例步数差异当作系综偏差。',
              '第一次分叉、各字段最大误差、完整性、表哈希与资源记录见 summary.json。',
              '本轮数据不得用于宣称强制 CC 的事件率、全强子能量闭合、射电一致性或生产加速比。']
    with (args.output / 'REPORT_CN.md').open('x') as file:
        file.write('\n'.join(lines) + '\n')
    print(json.dumps({'integration_passed': True, 'strict_trace_passed': result['strict_trace_passed'],
                      'cases': len(cases), 'resources': result['resources']}, indent=2))


if __name__ == '__main__':
    main()
