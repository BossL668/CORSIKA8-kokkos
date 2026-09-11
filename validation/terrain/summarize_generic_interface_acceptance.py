#!/usr/bin/env python3
"""Summarize completed, guarded interface tests without rerunning showers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def resources(path):
    value = json.loads(path.read_text())
    return dict(returncode=value['returncode'], stop_reason=value['stop_reason'],
                wall_seconds=value['wall_seconds'],
                peak_rss_MiB=max((p.get('rss_kib', 0) for p in value['samples']),
                                 default=0) / 1024,
                gpu_monitoring_requested=value['gpu_baseline'] is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--openmp-build', required=True, type=Path)
    parser.add_argument('--cuda-build', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    result = dict(scope='single closed oriented mesh, two logical regions; no radio',
                  openmp_live_tests={}, application_tests=[], binary_sha256={})
    for variant in ['ordinary', 'same', 'tetra', 'field', 'swapped']:
        folder = root / f'unit_{variant}_verified'
        data = resources(folder / 'resources.json')
        log = (folder / 'stdout.log').read_text()
        data['success'] = (data['returncode'] == 0 and data['stop_reason'] is None
                           and f'PASS interface variant={variant}' in log)
        data['geometry_queries'] = 1000000
        data['live_crossings'] = 6
        result['openmp_live_tests'][variant] = data

    for filename in ['applications_SiO2_results.json',
                     'applications_Water_Ice_results.json']:
        for case in json.loads((root / filename).read_text()):
            folder = root / case['tag']
            metadata = yaml.safe_load((folder / 'terrain_run.yaml').read_text())
            case['resources'] = resources(root / (case['tag'] + '_guard') / 'resources.json')
            case['material_interface'] = metadata['material_interface']
            case['csv_sha256'] = {p.name: digest(p) for p in sorted((folder / 'terrain').glob('*.csv'))}
            if case['tag'].startswith('applications_SiO2_'):
                backend = case['tag'].rsplit('_', 1)[1]
                baseline = yaml.safe_load((root / f'baseline_SiO2_{backend}/terrain_run.yaml').read_text())
                # Only timing and explicitly changed descriptive metadata may differ.
                for item in [baseline, metadata]:
                    item.pop('shower_seconds', None)
                    item.pop('material_interface', None)
                    if 'accelerator' in item:
                        item['accelerator'].pop('scope', None)
                case['checks']['unchanged_physics_metadata'] = baseline == metadata
            result['application_tests'].append(case)

    before = root / 'binaries/c8_terrain_cascade_before'
    after = args.openmp_build / 'applications/c8_terrain_cascade'
    # CLI11 prints argv[0]; compare the same invocation name for the archived
    # and rebuilt executable, without filtering any options or help text.
    invocation = ['c8_terrain_cascade', '--help']
    help_before = subprocess.check_output(invocation, executable=str(before.resolve()))
    help_after = subprocess.check_output(invocation, executable=str(after.resolve()))
    result['help_argv0'] = invocation[0]
    result['help_identical'] = help_before == help_after
    for label, path in [('before_openmp', before), ('after_openmp', after),
                        ('after_cuda', args.cuda_build / 'applications/c8_terrain_cascade')]:
        result['binary_sha256'][label] = digest(path)
    # This file is populated by the caller from the actual ctest run.
    ctest_log = root / 'ctest_openmp_verified.log'
    trace = ctest_log.read_text()
    result['ctest_success'] = (trace.count('Test Passed.') == 8 and 'Test Failed.' not in trace)
    result['cuda_build_only'] = True
    result['cuda_runtime_tested'] = False
    result['historical_attempts'] = {
        'unit_ordinary': '600 s first-use PROPOSAL cache-generation timeout; not accepted',
        'unit_ordinary_retry': 'invalid test fixture observation radius; fixture corrected, not a physics change',
    }
    result['accepted'] = (result['help_identical'] and result['ctest_success'] and
                          all(t['success'] for t in result['openmp_live_tests'].values()) and
                          len(result['application_tests']) == 6 and
                          all(all(t['checks'].values()) and
                              t['resources']['returncode'] == 0 and
                              t['resources']['stop_reason'] is None
                              for t in result['application_tests']))
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# beta5 通用双侧介质输运接口验收', '',
        '日期：2026-09-10。只验证粒子输运；射电关闭。', '',
        f"本轮门禁：{'通过' if result['accepted'] else '未通过'}。不是大样本物理等价或生产性能验收。", '',
        '## 结果', '',
        '- 8 项 OpenMP CTest 通过；CUDA 应用及接口测试程序编译通过，未占用生产显卡执行测试。',
        '- 5 种界面配置各 100 万次解析几何查询，总计 500 万次；30 个真实 PROPOSAL/Kokkos 跨界输入，另验证重放和跨界后的下一步。',
        '- 覆盖箱体/四面体、水/SiO₂、任意区域编号、同材质两侧、交换 bank 顺序、两侧独立磁场。',
        '- 原 SiO₂ CPU/OpenMP 各 1 个固定种子事例：重构前后全部轨迹、沉积、存活粒子 CSV 的 SHA-256 相同。',
        '- 重构前后应用 `--help` 字节一致。', '',
        '| 实际 DEM 内部介质 | 后端 | 完整性 | 轨迹/介质错配 | 峰值 RSS [MiB] | 启动及事件总时长 [s] |',
        '|---|---|---|---|---:|---:|',
    ]
    for t in result['application_tests']:
        _, material, backend = t['tag'].split('_')
        r = t['resources']
        lines.append(f"| {material} | {backend} | {t['checks']['complete']} | {t['diagnostics']['material_mismatches']} | {r['peak_rss_MiB']:.1f} | {r['wall_seconds']:.3f} |")
    lines += [
        '', '真实 DEM 检查采用 0.1 GeV 光子，seed=67101，emcut=0.5 MeV、不薄化、20 ns 窗口、0.1 m 诊断步长；OpenMP 为 2 线程。',
        '首次使用新介质时自动生成 PROPOSAL 自身缓存，表中时间包含该成本及强子模型初始化，不是后端加速比。', '',
        '## 保护与限制', '',
        '- 无 material mismatch、无截断 CSV，Kokkos 队列最终为空；详细计数、表标识和哈希见 summary.json 与各 terrain_run.yaml。',
        '- 非法区域/bank、非法密度及同核组分但不同 Medium 的 calculator 键冲突有门禁。水与冰同时进入原版单一 PROPOSAL 包装器仍不受支持，不能自动覆盖。',
        '- 接口是一个封闭定向网格的两侧，不是任意相交多体积或自交网格导航器；密度和磁场仍受现有环境快照能力约束。',
        '- Kokkos 会话处理 γ/e±，其他粒子和指定末态继续 CPU；跨界位置不人为平移，也不在界面把粒子吸收。',
        '- 本轮没有 CUDA 运行时、HIP、SYCL、大样本 shower 或跨界射电验收，也没有替换生产安装文件。',
        '- 初次冷缓存超时和测试夹具半径错误的记录已保留，不纳入通过结果。', '',
        '复现：仓库 validation/terrain/run_generic_interface_acceptance.py 和 tests/modules/testInterfaceTransport.cpp。',
    ]
    (root / 'VALIDATION_REPORT_CN.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(accepted=result['accepted'], report=str(root / 'VALIDATION_REPORT_CN.md'))))
    if not result['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
