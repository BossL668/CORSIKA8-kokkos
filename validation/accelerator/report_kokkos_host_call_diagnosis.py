#!/usr/bin/env python3
"""Check diagnostic perturbation and summarize sampled HOST call costs.

Read completed, isolated diagnoses only. Never pool instrumented runs into
production speed measurements; CUDA callback intervals are not kernel times.
"""
import argparse
import json
import math
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def estimate(row, probability):
    require(0 < probability <= 1, 'Invalid sampling probability')
    n, total = row['calls'], row['host_inclusive_s']
    require(n > 0 and math.isfinite(total) and total >= 0, 'Invalid timing row')
    squares = row.get('host_seconds_squared_sum', 0.)
    require(math.isfinite(squares) and squares >= 0, 'Invalid squared durations')
    if probability != 1:
        require('host_seconds_squared_sum' in row, 'Sampled profile lacks variance input')
        require(squares + max(1.e-24, squares*1.e-12) >= total*total/n,
                'Inconsistent duration sums')
    # Bernoulli/Poisson-inclusion approximation for a fixed sequence of costs.
    # Tool-only pseudorandom sampling and schedule perturbation mean this is
    # a diagnostic sampling SE, NOT a CI on production performance.
    stderr = math.sqrt((1-probability)*squares)/probability
    return dict(kind=row['kind'], execution=row['execution_type'], label=row['label'],
                sampled_calls=n, estimated_calls=n/probability,
                sampled_mean_ms=1000*total/n, estimated_host_s=total/probability,
                approximate_sampling_se_s=stderr, sampled_max_s=row['host_max_s'],
                sparse=n < 30)


def analyze(pilot, diagnosis):
    state = read(diagnosis/'STATUS.json')
    require(state['complete'], 'Diagnostic workflow has not completed')
    require(read(pilot/'PERFORMANCE_RESULT.json')['complete'], 'Incomplete reference')
    config = read(pilot/'CONFIG.json')
    binary = config['arguments']['after']
    require(state['binary_sha256'] == config['hashes'][binary], 'Binary mismatch')
    result = dict(diagnosis=str(diagnosis), binary_sha256=state['binary_sha256'],
                  production_speed_measurement=False, rows=[])
    for record in state['records']:
        label = record['label']
        base = read(pilot/(label+'.json'))
        guard = read(diagnosis/(label+'-guard')/'summary.json')
        require(base['complete'] and guard['pass'] and guard['returncode'] == 0,
                'Incomplete simulation')
        require(guard['performance_valid'] and not guard['foreign_gpu_pids'],
                'External GPU contention')
        expected = list(base['command'])
        expected[expected.index('-f')+1] = str(diagnosis/label)
        require(guard['command'] == expected, 'Profiled physics command changed')
        profile = read(diagnosis/(label+'.host_calls.json'))
        require(profile['complete'] and profile['errors'] == profile['active_at_finalize'] == 0,
                'Invalid callback profile')
        require(profile['settings_callback_seen'] and not profile['requires_global_fencing'],
                'Profile requires global fencing')
        probability = profile.get('timing_sample_probability', 1.)
        require(probability == state.get('timing_sample_probability', 1.),
                'Profile/launcher sampling mismatch')
        base_stats = yaml.safe_load((pilot/label/'gpu_em/summary.yaml').read_text())['shower_0']['statistics']
        stats = yaml.safe_load((diagnosis/label/'gpu_em/summary.yaml').read_text())['shower_0']['statistics']
        exact = None
        if label.endswith('-openmp'):
            exact = read(diagnosis/(label+'.same_output.json'))['pass']
            require(exact, 'Standalone physics changed under profiling')
        endpoints = {}
        for name, data in (('reference', base_stats), ('profiled', stats)):
            c = data['accelerator'].get('cooperative', {})
            if c:
                endpoints[name] = {e: dict(
                    steps=sum(c['adaptive'][e][k]['transport_records'] for k in ('photon','lepton')),
                    waves=sum(c['adaptive'][e][k]['resident_wavefronts'] for k in ('photon','lepton')))
                    for e in ('cuda','openmp')}
            else:
                endpoints[name] = dict(openmp=dict(steps=data['gpu_particles'],
                    waves=data['resident_photon_wavefronts']+data['resident_lepton_wavefronts']))
        ranked = [estimate(row, probability) for row in profile['rows']
                  if row['kind'] not in ('allocate','deallocate') and row['calls']]
        ranked.sort(key=lambda row: row['estimated_host_s'], reverse=True)
        # Rank distinct APIs but do not sum nested or simultaneous intervals.
        result['rows'].append(dict(label=label, probability=probability,
            reference_s=base['process_s'], profiled_s=guard['elapsed_s'],
            perturbation_percent=100*(guard['elapsed_s']/base['process_s']-1),
            standalone_physics_exact=exact, endpoints=endpoints, host_calls=ranked))
    require(len(result['rows']) == 4, 'Expected two complete pairs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--diagnosis', required=True, type=Path, action='append')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = [analyze(args.pilot.resolve(), path.resolve()) for path in args.diagnosis]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'HOST_CALL_DIAGNOSIS.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    text = ['# 双端主机调用诊断：先检查探针扰动', '',
            '同一模拟二进制和物理参数；这是诊断，不是新的生产加速统计。', '',
            'OpenMP同步调用区间包含计算/屏障；CUDA调用区间不是设备kernel时间。',
            '嵌套或重叠区间不求和成总时间。抽样总量为估计，SE仅描述固定调用序列的',
            '近似抽样误差，不包含系统噪声或探针导致的调度变化；小于30次的行标为稀疏。', '']
    for result in results:
        text += ['## '+Path(result['diagnosis']).name, '',
                 '| 事例 | 原始/s | 插桩/s | 时间变化 | OpenMP波前 原始→插桩 |',
                 '|---|---:|---:|---:|---:|']
        for row in result['rows']:
            endpoints = row['endpoints']
            text.append('| {} | {:.3f} | {:.3f} | {:+.2f}% | {} → {} |'.format(
                row['label'],row['reference_s'],row['profiled_s'],row['perturbation_percent'],
                endpoints['reference']['openmp']['waves'],endpoints['profiled']['openmp']['waves']))
        for row in result['rows']:
            text += ['', '### '+row['label'], '', '抽样概率：{:.8f}；单OpenMP物理输出严格一致：{}。'.format(
                row['probability'], row['standalone_physics_exact'] if row['standalone_physics_exact'] is not None else '双端不要求同树'), '',
                '| OpenMP调用 | 抽中次数 | 抽样平均/ms | 估计主机区间/s | 抽样SE/s |',
                '|---|---:|---:|---:|---:|']
            for call in [c for c in row['host_calls'] if c['execution']=='OpenMP'][:15]:
                text.append('| `{}`{} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    call['label'], '（稀疏）' if call['sparse'] else '', call['sampled_calls'],
                    call['sampled_mean_ms'],call['estimated_host_s'],call['approximate_sampling_se_s']))
    (args.output/'HOST_CALL_DIAGNOSIS_CN.md').write_text('\n'.join(text)+'\n')
    print(args.output/'HOST_CALL_DIAGNOSIS_CN.md')


if __name__ == '__main__':
    main()
