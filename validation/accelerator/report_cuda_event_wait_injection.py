#!/usr/bin/env python3
"""Read-only analysis of an injected event timer; writes only a fresh report.

No GPU, simulation, service, baseline modification or performance certification.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_log(path):
    rows = {}
    results = []
    timer = []
    summaries = []
    for line in path.read_text().splitlines():
        if line.startswith('CONTENTION_REPEAT '):
            row = json.loads(line.partition(' ')[2])
            key = (row['repeat'], row['mode'])
            if key in rows:
                raise ValueError('duplicate fixed-work row: '+str(key))
            rows[key] = row
        elif line.startswith('CONTENTION_RESULT '):
            results.append(json.loads(line.partition(' ')[2]))
        elif line.startswith('{'):
            row = json.loads(line)
            if row.get('kind') == 'C8_CUDA_EVENT_TIMER':
                timer.append(row)
            elif row.get('kind') == 'C8_CUDA_EVENT_TIMER_SUMMARY':
                summaries.append(row)
    expected = {(r,m) for r in range(6) for m in ('helper-off','default-fence','blocking-event')}
    if set(rows) != expected or len(results) != 1 or not results[0]['complete']:
        raise ValueError('require complete 18-row fixed-work diagnostics; no row selection')
    return rows, results[0], timer, summaries


def build(injected, baseline):
    rows, result, timer, summaries = read_log(injected/'command.log')
    base, before, _, _ = read_log(baseline/'command.log')
    guard = json.loads((injected/'summary.json').read_text())
    if not guard.get('pass') or guard.get('returncode') != 0:
        raise ValueError('injected diagnostic did not complete successfully')
    if len(summaries) != 1 or not summaries[0]['coverage_complete']:
        raise ValueError('incomplete or missing injected API coverage')
    if any(not r['timing_valid'] or r['retval'] != 0 for r in timer):
        raise ValueError('timer contains failed calls or invalid timing')
    if sum(r['calls'] for r in timer) != summaries[0]['observed_calls']:
        raise ValueError('API aggregate count does not close')
    configuration_keys = ('schema','threads','cpu_roots','gpu_roots','energy_MeV','radio',
                          'resident_capacity_limit','gpu_memory_fraction','call_shapes')
    if any(result[k] != before[k] for k in configuration_keys):
        raise ValueError('injected and uninstrumented configurations differ')
    checked = []
    for key in sorted(rows):
        a, b = rows[key], base[key]
        same = a['cpu']['hashes'] == b['cpu']['hashes'] and (
            (a['gpu'] is None and b['gpu'] is None) or
            (a['gpu'] is not None and b['gpu'] is not None and a['gpu']['hashes'] == b['gpu']['hashes']))
        checked.append(dict(repeat=key[0], mode=key[1], warmup=a['warmup'],
                            cpu_hashes=a['cpu']['hashes'],
                            gpu_hashes=a['gpu']['hashes'] if a['gpu'] else None, exact=same))
        if not same:
            raise ValueError('physical/input/call hashes changed: '+str(key))
    blocking = [r['gpu'] for key,r in rows.items() if key[1]=='blocking-event']
    sync = [r for r in timer if r['api']=='cudaEventSynchronize' and r['flags'] is not None and r['flags'] & 1]
    recorded = [r for r in timer if r['api']=='cudaEventRecord' and r['flags'] is not None and r['flags'] & 1]
    def aggregate(values):
        return dict(calls=sum(v['calls'] for v in values),
            wall_seconds=sum(v['wall_ns'] for v in values)/1e9,
            thread_cpu_seconds=sum(v['thread_cpu_ns'] for v in values)/1e9,
            tids=sorted({v['tid'] for v in values}), flags=sorted({v['flags'] for v in values}))
    sync, recorded = aggregate(sync), aggregate(recorded)
    if not sync['calls'] or sync['calls'] != sum(b['blocking_wait_calls'] for b in blocking):
        raise ValueError('blocking API count does not match all six backend rows')
    if sync['tids'] != recorded['tids'] or len(sync['tids']) != 1:
        raise ValueError('unexpected event submitter identity')
    driver_cpu = sum(b['driver_thread_cpu_seconds'] for b in blocking)
    sources = {name: digest(path) for name,path in {
        'injected_log':injected/'command.log', 'injected_guard':injected/'summary.json',
        'baseline_log':baseline/'command.log', 'baseline_guard':baseline/'summary.json'}.items()}
    return dict(schema=1, scope='injected event API attribution, NOT a production speedup test',
        complete=True, injected=True, performance_certification=False,
        rows_compared=18, exact_all_hashes=True,
        configuration={k:result[k] for k in configuration_keys}, sources=sources,
        injection_guard=guard, timer_summary=summaries[0], timer_rows=timer,
        blocking_mode=dict(warmups_included=True, repeats=6, driver_thread_cpu_seconds=driver_cpu,
            record=recorded, synchronize=sync,
            synchronize_cpu_to_wall_ratio=sync['thread_cpu_seconds']/sync['wall_seconds'],
            synchronize_fraction_of_driver_cpu=sync['thread_cpu_seconds']/driver_cpu,
            record_fraction_of_driver_cpu=recorded['thread_cpu_seconds']/driver_cpu),
        row_hash_comparisons=checked,
        limitations=[
            'API clocks include measurement overhead; injection perturbs host execution outside clocks too.',
            'Six blocking rows include one warmup; API aggregates cannot split the warmup from measured rows.',
            'Only observed event runtime APIs are covered; default-fence/stream/driver waits are not intercepted.',
            'API wall time includes GPU work and waiting; it is not kernel time or all-worker CPU time.',
            'This cannot distinguish driver spin, system calls, scheduling or other internal CUDA code.',
            'Outputs agree for this no-radio fixed-work accelerator boundary, not full-shower energy coverage or 1% statistics.',
            'Uninstrumented a5 reports and timing remain unchanged.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--injected-guard',type=Path,required=True)
    p.add_argument('--baseline-guard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); report=build(a.injected_guard,a.baseline_guard)
    b=report['blocking_mode']; s=b['synchronize']; r=b['record']
    lines=['# CUDA event 阻塞等待：注入诊断', '',
        '结论：本次剩余 CUDA driver 核时主要发生在 `cudaEventSynchronize` 内部，而不是 Record。', '',
        f"18/18 行输入、调用序列及 CPU/GPU 物理哈希与无注入 a5 完全一致。配置：{report['configuration']['threads']} 线程，"
        f"每端 {report['configuration']['cpu_roots']} 初始粒子、基准 {report['configuration']['energy_MeV']} MeV，无射电。", '',
        '| 阻塞事件 API | flags | 调用数 | 主机墙钟 / s | 主机线程核时 / CPU s |',
        '|---|---|---:|---:|---:|',
        f"| Record | {r['flags']} | {r['calls']} | {r['wall_seconds']:.6f} | {r['thread_cpu_seconds']:.6f} |",
        f"| Synchronize | {s['flags']} | {s['calls']} | {s['wall_seconds']:.6f} | {s['thread_cpu_seconds']:.6f} |", '',
        f"flags=3 是 BlockingSync(1) 与 DisableTiming(2) 的组合。Sync 核时/墙钟为 {b['synchronize_cpu_to_wall_ratio']:.2%}。",
        f"六轮（包含一次预热）GPU driver 总核时 {b['driver_thread_cpu_seconds']:.6f} s；"
        f"Sync 占 {b['synchronize_fraction_of_driver_cpu']:.2%}，Record 占 {b['record_fraction_of_driver_cpu']:.2%}。", '',
        '这说明“请求阻塞事件”并不等于驱动实现零核时；本测试尚不能把内部核时进一步归因于自旋、系统调用或唤醒。',
        '它也不证明更改调度器能消除这部分时间，不能据此修改物理算法。', '',
        '## 证据与限制', '',
        '事件/聚合表没有溢出，没有未知 flags 或计时失败，正常退出时无活动事件。',
        '此次是注入诊断，不能拿注入前后的墙钟差值宣称提速；正式无注入报告保持原样。', '']
    lines.extend('- '+v for v in report['limitations'])
    lines += ['', '完整原始 API 行、18 行哈希及输入文件 SHA-256 见 RESULT.json。', '']
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output/'REPORT_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(a.output), exact_all_hashes=True,blocking_mode=b)))


if __name__=='__main__':main()
