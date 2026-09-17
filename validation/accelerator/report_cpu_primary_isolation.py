#!/usr/bin/env python3
"""Independently re-read guard logs and exact-output checks for isolation tests."""
import argparse
import json
from pathlib import Path
import statistics


def checked_run(root):
    state = json.loads((root/'STATUS.json').read_text())
    args = state['arguments']
    modes = args['modes']
    expected = {(radio, order, mode) for radio in (0, 1) for order in (0, 1) for mode in modes}
    if not state['complete'] or state.get('error') or len(state['records']) != len(expected):
        raise RuntimeError('incomplete '+str(root))
    seen, references = set(), {}
    rows = []
    for record in state['records']:
        key = record['radio'], record['order'], record['mode']
        if key not in expected or key in seen:
            raise RuntimeError('duplicate/unexpected run '+str(key))
        seen.add(key)
        directory = root/record['label']
        guard = json.loads((directory/'summary.json').read_text())
        if not guard['pass'] or guard['returncode'] != 0 or guard['failure'] or \
                guard['foreign_gpu_pids'] or \
                guard['minimum_available_bytes'] < 4*2**30:
            raise RuntimeError('invalid guard '+str(directory))
        if not guard['performance_valid'] and not guard['gpu_observation_failures']:
            raise RuntimeError('unexplained invalid performance guard '+str(directory))
        reports = [json.loads(line[len('ISOLATION_RESULT '):]) for line in
                   (directory/'command.log').read_text().splitlines() if line.startswith('ISOLATION_RESULT ')]
        if len(reports)!=1 or reports[0] != record['result']:
            raise RuntimeError('raw fixture report mismatch')
        result = reports[0]
        if not result['complete'] or result['gpu_transport_steps'] or result['threads'] != args['threads'] or \
                result['mode'] != record['mode'] or len(result['runs']) != 3:
            raise RuntimeError('incorrect fixture')
        for i, event in enumerate(result['runs']):
            if event['repeat'] != i or event['warm'] != (i>0) or event['energy_relative_residual']>1e-4:
                raise RuntimeError('incorrect repeat/energy')
            identity = {k: event[k] for k in ('input_sha256', 'physics_arrays_terminal_sha256',
                        'call_sequence_sha256', 'steps', 'photon_waves', 'lepton_waves', 'terminal_count',
                        'photon_calls', 'lepton_calls', 'energy_relative_residual')}
            reference = references.setdefault(key[0], identity)
            if identity != reference:
                raise RuntimeError('exact fixed-input/call/output mismatch')
            if i:
                rows.append(dict(radio=key[0], mode=key[2], seconds=event['driver_seconds'],
                                 performance_valid=guard['performance_valid']))
    groups = []
    for radio in (0, 1):
        baseline = statistics.median(r['seconds'] for r in rows if r['radio']==radio and r['mode']=='direct')
        for mode in modes:
            values = [r['seconds'] for r in rows if r['radio']==radio and r['mode']==mode]
            median = statistics.median(values)
            groups.append(dict(radio=radio, mode=mode, median_s=median, samples=values,
                               valid_warm_samples=sum(r['performance_valid'] for r in rows if r['radio']==radio and r['mode']==mode),
                               ratio_to_direct=median/baseline))
    return dict(directory=str(root), threads=args['threads'], count=args['count'],
                energy_mev=args['energy_mev'], complete=True, groups=groups, references=references,
                peak_tree_rss_gib=max(r['guard']['peak_tree_rss_bytes'] for r in state['records'])/2**30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    names = ['local20', 'psr130', 'local20-pump', 'psr130-pump']
    reports = [checked_run(a.root/name) for name in names]
    summary = dict(complete=True, scope='backend/workspace and below-reserve pump isolation, not whole-shower speed',
                   groups=reports, strict_performance_certified=False)
    (a.root/'ISOLATION_AUDIT.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# CPU 优先 OpenMP 固定输入隔离检查', '',
             '2026-09-12。仅新增验证可执行程序，链接冻结 v2 的后端对象；生产源码、安装和物理公式未改。', '',
             '## 检查范围', '',
             '- 工作区检查：32768 个 e±，总能量逐粒子为 10.0–10.9 MeV；普通/预分配/借用双端 runtime/空闲 CUDA 工作区。',
             '- 实际调度器检查：1024 个 e±，1.5–2.4 MeV，低于现有 CPU 保留门槛；断言 GPU 分配、提交、完成均为零。没有修改调度算法来强行达成。',
             '- 原生 PROPOSAL 同一表和辅助缓存；同一初始身份/随机种子；光子16轮、轻子1024轮、minimum=4096。',
             '- 分别无射电及 CoREAS＋ZHS（4天线）。每模式正/反顺序各一次进程，每进程1次预热＋2次相同输入；表中为4个热重复中位数。',
             '- 本地20线程；PSR130线程，绑定逻辑382–511。所有运行有4GiB可用内存下限及RSS上限。', '',
             '## 实际结果', '',
             '| 设备/输入 | 射电 | 模式 | 热中位 / s | 相对纯 OpenMP | 完整监控热样本 |',
             '|---|---|---|---:|---:|---:|']
    for name, report in zip(names,reports):
        for g in report['groups']:
            lines.append('| %s | %s | %s | %.6f | %.3f | %d/4 |' %
                         (name, '开' if g['radio'] else '关', g['mode'], g['median_s'], g['ratio_to_direct'], g['valid_warm_samples']))
    lines += ['', '## 结论与边界', '',
              '每组所有重复的输入摘要、host调用顺序/计数/history区间摘要、输运步数、波前计数、profile/射电数组及终态摘要一致；能量通量门禁通过。', '',
              '表格保留全部完成事件，不剔除较慢或监控不完整样本。本地大输入全射电 direct 的一次进程出现 NVML PID 无法验证；其2个热重复计时不获严格认证，原始错误保留。PSR各项监控完整。', '',
              '这些固定输入检查没有复现之前完整 Fe shower 的数倍 OpenMP 吞吐退化。因此，不能将生产退化直接归咎于“初始化了 CUDA”或“多了一层虚接口”。', '',
              '**仍未解决完整生产性能问题。** 零GPU的实际pump检查只覆盖低于分配门槛的小输入；不覆盖真实双端划分后的大队列、迁移、scalar fallback、完整强子链和设备同时负载。短重复计时有波动，不用于认证3%性能门禁。', '',
              '下一项应在真实 CPU 优先队列中检查分配前后有效波前、低批量返回及 scalar 重新组批；先保住 CPU 主端批量，再衡量辅助 GPU 的净收益。', '',
              '各子目录保留 STATUS.json、guard summary 和原始 command.log。PSR大资源日志仍在服务器。', '']
    (a.root/'ISOLATION_REPORT_CN.md').write_text('\n'.join(lines))
    print(a.root/'ISOLATION_REPORT_CN.md')


if __name__ == '__main__':
    main()
