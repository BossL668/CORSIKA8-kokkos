#!/usr/bin/env python3
"""Audit a finished fixed-workload contention log; never touch running jobs.

Reads run_overlap_guarded.py's command.log/summary.json. All logged repeats,
including invalid or duplicate ones, remain in the report. Timing eligibility
is distinct from process success, output agreement and shower-level acceptance.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


MODES = ('helper-off', 'default-fence', 'blocking-event')
HASH_KEYS = frozenset(('input', 'profile_integer_and_float', 'radio_integer_and_float',
    'terminal_particles', 'fallback_with_random_provenance', 'call_sequence',
    'physical_counts', 'table', 'auxiliary'))
SHA256 = re.compile(r'^[0-9a-f]{64}$')
MAX_LOG_BYTES = 256 * 1024**2
LIMITATIONS = [
    '这里只测固定真实 EM 输入的端点计算及辅助 CUDA 线程竞争，不是完整 shower 或生产调度器验收。',
    'OpenMP 各模式必须保持同一输入、调用序列和输出；CUDA 两种等待模式单独核对，不要求 CPU 与 GPU 哈希相同。',
    '实际输运工作量使用 photon_transport_records 与 lepton_transport_records；endpoint 的旧路由汇总字段 particles_advanced 可以为 0，不能据此推断粒子丢失。',
    'GPU driver 核时来自 CLOCK_THREAD_CPUTIME_ID，是主机提交线程实际耗用的 CPU 秒，不是 GPU kernel 时间。',
    'blocking_wait_host_seconds 是 event 创建/记录/等待的主机墙时，包含睡眠及唤醒延迟，与 OpenMP 运算重叠，不能相加为总耗时。',
    'overlap 是两个端点主机调用时间窗的交集，不是 CUPTI/Nsight kernel 时间线，也不能证明 GPU 持续满载。',
    'CPU 主线程核时不包含 OpenMP worker；joint_process_cpu_seconds 才包含进程各线程的 CPU 时间。',
    '最终 profile/radio 下载与哈希在两端输运都结束后进行，单独列示，不纳入输运计时。',
    'schema 3 的浮点数组来自同一次定点快照的主机解码，不是独立普通下载路径的交叉验证。',
    'guard 的 GPU 监控有效性不等于整个 CPU 服务器独占；其他用户负载仍可能影响测量。',
    'fallback 在本探针中只作为返回加速器边界的能量通量记录，没有执行完整标量 fallback 或强子级联。',
    '五次中位数仅提供这一固定工作量的描述性比较，不据此声明统计显著性、1% 物理等价或整体 shower 加速。',
]


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def fingerprint(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0))


def mapping(value):
    return value if isinstance(value, dict) else {}


def parse_log(path):
    repeats, results, issues = [], [], []
    tail = deque(maxlen=8)
    digest = hashlib.sha256()
    before = fingerprint(path)
    if before[2] > MAX_LOG_BYTES:
        raise ValueError('command.log exceeds the bounded 256 MiB input limit')
    consumed = 0
    with path.open('rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            consumed += len(raw)
            if consumed > MAX_LOG_BYTES:
                raise ValueError('command.log grew beyond the bounded input limit')
            digest.update(raw)
            clean = raw.strip()
            recognized = False
            for prefix, records in ((b'CONTENTION_REPEAT ', repeats),
                                    (b'CONTENTION_RESULT ', results)):
                if not clean.startswith(prefix):
                    continue
                recognized = True
                try:
                    value = json.loads(clean[len(prefix):])
                    if not isinstance(value, dict):
                        raise ValueError('record must be an object')
                    records.append(dict(line=line_number, record=value))
                except (ValueError, UnicodeError) as error:
                    issues.append('line {}: invalid {}: {}'.format(
                        line_number, prefix.decode().strip(), error))
            if clean and not recognized:
                tail.append(dict(line=line_number, text=clean.decode('utf-8', errors='replace')[:4096]))
    after = fingerprint(path)
    stable = before == after
    if not stable:
        issues.append('command.log changed while being read; timing is not certified')
    return repeats, results, issues, dict(path=str(path), sha256=digest.hexdigest(),
        bytes_read=consumed, stable=stable, nonrecord_tail=list(tail))


def read_guard(path):
    try:
        before = fingerprint(path)
        if before[2] > 8 * 1024**2:
            raise ValueError('guard summary exceeds 8 MiB')
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('guard summary must be an object')
        stable = before == fingerprint(path)
        return value, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                          bytes_read=len(raw), stable=stable), [] if stable else [
                              'guard summary changed while being read']
    except (OSError, ValueError) as error:
        return {}, dict(path=str(path), stable=False), ['guard summary: '+str(error)]


def read_known_interference(path):
    """Presence invalidates timing only, including unreadable/malformed markers.

    NVML exclusivity cannot detect unrelated CPU work. Do not reinterpret an
    interference declaration as a physics failure or remove affected repeats.
    """
    source = dict(path=str(path), stable=False)
    info = dict(present=True, manifest=None, issues=[],
                reason='KNOWN_INTERFERENCE.json 存在；本轮严格计时无效。')
    try:
        # lstat distinguishes an absent marker from a broken symlink marker.
        path.lstat()
    except FileNotFoundError:
        return dict(present=False, manifest=None, issues=[], reason=None), None
    except OSError as error:
        info['issues'].append('cannot determine interference marker state: '+str(error))
        return info, source
    try:
        before = fingerprint(path)
        if before[2] > 8 * 1024**2:
            raise ValueError('interference manifest exceeds 8 MiB')
        raw = path.read_bytes()
        source.update(sha256=hashlib.sha256(raw).hexdigest(), bytes_read=len(raw),
                      stable=before == fingerprint(path))
        if not source['stable']:
            info['issues'].append('interference manifest changed while being read')
        def invalid_constant(value):
            raise ValueError('nonfinite JSON constant: '+value)
        value = json.loads(raw, parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ValueError('interference manifest must be an object')
        info['manifest'] = value
        reason = value.get('reason')
        reasons = value.get('reasons')
        if isinstance(reason, str) and reason.strip():
            info['reason'] = reason
        elif isinstance(reasons, list) and reasons and all(isinstance(v, str) for v in reasons):
            info['reason'] = '; '.join(reasons)
    except (OSError, ValueError, UnicodeError) as error:
        info['issues'].append('known interference manifest: '+str(error))
    return info, source


def hashes_valid(endpoint):
    hashes = endpoint.get('hashes') if isinstance(endpoint, dict) else None
    return (isinstance(hashes, dict) and set(hashes) == HASH_KEYS and
            all(isinstance(v, str) and SHA256.fullmatch(v) for v in hashes.values()))


def audit_endpoint(endpoint, name, issues):
    if not isinstance(endpoint, dict):
        issues.append(name+': missing endpoint object')
        return
    if not hashes_valid(endpoint):
        issues.append(name+': incomplete or invalid SHA-256 dictionary')
    counts = endpoint.get('physical_counts')
    if not isinstance(counts, dict) or not counts or not all(
            type(v) is int and v >= 0 for v in counts.values()):
        issues.append(name+': physical_counts must contain nonnegative integer counters')
    elif mapping(endpoint.get('hashes')).get('physical_counts') != hashlib.sha256(
            canonical_bytes(counts)).hexdigest():
        issues.append(name+': physical_counts hash does not match the recorded counters')
    for key in ('transport_seconds', 'driver_thread_cpu_seconds',
                'transport_driver_thread_cpu_seconds', 'output_download_seconds',
                'blocking_wait_host_seconds'):
        if not finite(endpoint.get(key), positive=(key == 'transport_seconds')):
            issues.append(name+': invalid or missing '+key)
    if type(endpoint.get('blocking_wait_calls')) is not int or endpoint['blocking_wait_calls'] < 0:
        issues.append(name+': invalid blocking_wait_calls')
    for key in ('photon_transport_records', 'lepton_transport_records'):
        if (type(endpoint.get(key)) is not int or endpoint[key] < 0 or
                mapping(counts).get(key) != endpoint[key]):
            issues.append(name+': invalid or inconsistent actual '+key)


def median_value(rows, endpoint, key):
    values = [r.get(endpoint, {}).get(key) if isinstance(r.get(endpoint), dict) else None
              for r in rows]
    return statistics.median(values) if values and all(finite(v) for v in values) else None


def numeric_median(rows, key):
    values = [r.get(key) for r in rows]
    return statistics.median(values) if values and all(finite(v) for v in values) else None


def build_report(guard_dir):
    guard_dir = Path(guard_dir).resolve()
    try:
        repeats, results, issues, log_source = parse_log(guard_dir/'command.log')
    except (OSError, ValueError) as error:
        repeats, results = [], []
        issues = ['command log: '+str(error)]
        log_source = dict(path=str(guard_dir/'command.log'), stable=False)
    guard, guard_source, guard_issues = read_guard(guard_dir/'summary.json')
    interference, interference_source = read_known_interference(guard_dir/'KNOWN_INTERFERENCE.json')
    rows = [r['record'] for r in repeats]
    result = results[0]['record'] if len(results) == 1 else {}
    if len(results) != 1:
        issues.append('expected exactly one CONTENTION_RESULT; found {}'.format(len(results)))
    if result.get('schema') not in (2, 3) or result.get('complete') is not True:
        issues.append('unique schema-2/3 result is missing or report.complete is not true')
    for field in ('same_cpu_workload_and_outputs_exact', 'same_gpu_workload_and_outputs_exact',
                  'wait_mode_and_lifecycle_tests_passed'):
        if result.get(field) is not True:
            issues.append('result assertion missing/false: '+field)
    if result.get('warmups_per_mode') != 1 or result.get('measured_repeats_per_mode') != 5:
        issues.append('result must specify one warmup and five measured repeats per mode')
    if result.get('runs') != rows:
        issues.append('final result.runs does not exactly match all logged repeat records')
    if results and repeats and results[0]['line'] <= repeats[-1]['line']:
        issues.append('repeat record occurs after the final result')
    pairs = [(r.get('mode'), r.get('repeat')) for r in rows]
    # Counter keys below use strings even for malformed JSON values.
    pair_counts = Counter(str(p) for p in pairs)
    duplicates = {key: count for key, count in pair_counts.items() if count > 1}
    expected_pairs = [(m, n) for n in range(6) for m in MODES]
    matrix_complete = (len(rows) == 18 and all(type(r.get('repeat')) is int and
                       r.get('mode') in MODES for r in rows) and
                       sorted(pairs) == sorted(expected_pairs))
    if not matrix_complete:
        issues.append('repeat matrix is incomplete/duplicate: require each mode × repeats 0..5 exactly once')
    if [(r.get('repeat'), r.get('order')) for r in rows] != [
            (repeat, order) for repeat in range(6) for order in range(3)]:
        issues.append('logged repeat/order sequence does not match the rotated execution schedule')
    for index, row in enumerate(rows):
        name = 'repeat line {}'.format(repeats[index]['line'])
        mode, repeat, order = row.get('mode'), row.get('repeat'), row.get('order')
        if type(repeat) is not int or repeat not in range(6):
            issues.append(name+': invalid repeat index')
        elif row.get('warmup') is not (repeat == 0):
            issues.append(name+': warmup flag does not match repeat 0')
        if (type(order) is not int or order not in range(3) or type(repeat) is not int or
                mode != MODES[(order+repeat) % 3]):
            issues.append(name+': rotated run order is invalid')
        audit_endpoint(row.get('cpu'), name+'/cpu', issues)
        cpu = mapping(row.get('cpu'))
        if cpu.get('blocking_wait_enabled') is not False or cpu.get('blocking_wait_calls') != 0:
            issues.append(name+': OpenMP endpoint reports CUDA blocking work')
        if mode == 'helper-off':
            if row.get('gpu') is not None:
                issues.append(name+': disabled helper has GPU work')
        elif mode in MODES:
            audit_endpoint(row.get('gpu'), name+'/gpu', issues)
            gpu = mapping(row.get('gpu'))
            blocking = mode == 'blocking-event'
            calls = gpu.get('blocking_wait_calls')
            if (gpu.get('blocking_wait_enabled') is not blocking or type(calls) is not int or
                    (calls <= 0 if blocking else calls != 0)):
                issues.append(name+': GPU wait mode/counter does not match requested mode')
            overlap = row.get('transport_host_call_window_overlap_seconds')
            if (not finite(overlap) or not finite(cpu.get('transport_seconds'), True) or
                    not finite(gpu.get('transport_seconds'), True) or
                    overlap > min(cpu['transport_seconds'], gpu['transport_seconds']) + 1.e-9):
                issues.append(name+': invalid host-call window overlap')
        for key in ('rss_bytes', 'available_memory_bytes', 'joint_wall_seconds',
                    'joint_process_cpu_seconds'):
            if not finite(row.get(key)):
                issues.append(name+': invalid or missing '+key)
    cpu_hashes = [r['cpu']['hashes'] for r in rows if hashes_valid(r.get('cpu'))]
    gpu_rows = [r for r in rows if r.get('mode') in MODES[1:]]
    gpu_hashes = [r['gpu']['hashes'] for r in gpu_rows if hashes_valid(r.get('gpu'))]
    cpu_exact = len(cpu_hashes) == 18 and all(h == cpu_hashes[0] for h in cpu_hashes)
    gpu_exact = len(gpu_hashes) == 12 and all(h == gpu_hashes[0] for h in gpu_hashes)
    if not cpu_exact:
        issues.append('OpenMP output/input/call hashes are missing or differ across all 18 runs')
    if not gpu_exact:
        issues.append('CUDA output/input/call hashes are missing or differ across both helper modes')
    summaries = {}
    for mode in MODES:
        selected = [r for r in rows if r.get('mode') == mode]
        measured = [r for r in selected if r.get('warmup') is False]
        usable = (len(selected) == 6 and sorted(r.get('repeat', -1) for r in selected
                    if type(r.get('repeat')) is int) == list(range(6)) and
                  len(measured) == 5 and all(type(r.get('repeat')) is int and
                                           1 <= r['repeat'] <= 5 for r in measured))
        summary = dict(logged_runs=len(selected), warmups=sum(r.get('warmup') is True for r in selected),
                       measured_repeats=len(measured), sample_set_complete=usable)
        summary['cpu_transport_median_seconds'] = median_value(measured, 'cpu', 'transport_seconds') if usable else None
        summary['cpu_photon_transport_records'] = median_value(measured, 'cpu', 'photon_transport_records') if usable else None
        summary['cpu_lepton_transport_records'] = median_value(measured, 'cpu', 'lepton_transport_records') if usable else None
        summary['joint_wall_median_seconds'] = numeric_median(measured, 'joint_wall_seconds') if usable else None
        summary['rss_peak_bytes'] = max((r['rss_bytes'] for r in selected if finite(r.get('rss_bytes'))), default=None)
        if mode != 'helper-off':
            for key in ('transport_seconds', 'driver_thread_cpu_seconds',
                        'transport_driver_thread_cpu_seconds', 'output_download_seconds',
                        'blocking_wait_calls', 'blocking_wait_host_seconds'):
                summary['gpu_'+key+'_median'] = median_value(measured, 'gpu', key) if usable else None
            summary['host_window_overlap_median_seconds'] = numeric_median(
                measured, 'transport_host_call_window_overlap_seconds') if usable else None
        summaries[mode] = summary
    baseline = summaries['helper-off']['cpu_transport_median_seconds']
    for mode, summary in summaries.items():
        duration = summary['cpu_transport_median_seconds']
        ratio = duration/baseline if finite(duration, True) and finite(baseline, True) else None
        summary['cpu_transport_ratio_to_helper_off'] = ratio
        summary['cpu_transport_change_percent'] = 100*(ratio-1) if ratio is not None else None
        advertised = mapping(mapping(result.get('medians')).get(mode))
        expected = {'cpu_transport_seconds': duration, 'cpu_time_ratio_to_helper_off': ratio}
        if mode != 'helper-off':
            expected['gpu_driver_thread_cpu_seconds'] = summary.get('gpu_driver_thread_cpu_seconds_median')
        for field, value in expected.items():
            actual = advertised.get(field)
            if value is None or not finite(actual) or not math.isclose(value, actual, rel_tol=1.e-12, abs_tol=1.e-12):
                issues.append('reported median disagrees with all five repeats: '+mode+'/'+field)
    guard_passed = (guard.get('pass') is True and type(guard.get('returncode')) is int and
                    guard['returncode'] == 0 and guard.get('failure') is None)
    monitor_valid = (guard.get('performance_valid') is True and
                     guard.get('exclusive_gpu_required') is True and
                     guard.get('gpu_observation_failures') == 0 and
                     guard.get('foreign_gpu_pids') == [] and finite(guard.get('gpu_checks'), True))
    correctness = not issues and matrix_complete and cpu_exact and gpu_exact
    timing_valid = (correctness and guard_passed and monitor_valid and not guard_issues
                    and not interference['present'])
    sources = dict(command_log=log_source, guard_summary=guard_source)
    if interference_source is not None:
        sources['known_interference'] = interference_source
    return dict(schema=1, generated_utc=datetime.now(timezone.utc).isoformat(), guard_dir=str(guard_dir),
        scope='fixed real-EM CPU workload contention diagnostic; not whole-shower speedup',
        sources=sources,
        checks=dict(unique_result=len(results) == 1, result_complete=result.get('complete') is True,
                    repeat_matrix_complete=matrix_complete, cpu_hashes_exact=cpu_exact,
                    gpu_hashes_exact=gpu_exact, correctness_passed=correctness,
                    guard_passed=guard_passed, guard_performance_valid=guard.get('performance_valid'),
                    monitor_valid=monitor_valid, known_interference_present=interference['present'],
                    performance_evidence_valid=timing_valid,
                    whole_shower_performance_accepted=False),
        issues=issues, guard_issues=guard_issues, known_interference=interference,
        duplicates=duplicates, guard=guard,
        probe_settings={k: v for k, v in result.items() if k not in ('runs', 'medians')},
        mode_summaries=summaries, expected_cpu_hashes=cpu_hashes[0] if cpu_hashes else None,
        expected_gpu_hashes=gpu_hashes[0] if gpu_hashes else None,
        repeats=repeats, results=results, limitations=LIMITATIONS)


def fmt(value, precision=3):
    return '{:.{}f}'.format(value, precision) if finite(value) else '—'


def markdown(report, result_sha256=None):
    c = report['checks']
    lines = ['# CPU 优先辅助 CUDA 等待诊断', '',
        '这是固定输入的 EM 端点竞争测试，不是完整 shower 的加速验收。', '',
        '- 完整记录矩阵：{}；CPU 哈希一致：{}；GPU 两种等待模式哈希一致：{}。'.format(
            c['repeat_matrix_complete'], c['cpu_hashes_exact'], c['gpu_hashes_exact']),
        '- 正确性记录核验：{}；guard.pass：{}；guard.performance_valid：{}。'.format(
            c['correctness_passed'], c['guard_passed'], c['guard_performance_valid']),
        '- 这次固定工作量计时证据有效：{}；完整 shower 性能验收：未进行。'.format(c['performance_evidence_valid']), '',
        '- guard 返回码：{}；失败原因：{}；监控记录的进程树 RSS 峰值：{} GiB。'.format(
            report['guard'].get('returncode'), report['guard'].get('failure'),
            fmt(report['guard']['peak_tree_rss_bytes']/2**30
                if finite(report['guard'].get('peak_tree_rss_bytes')) else None)), '',
        '各模式使用全部五次正式重复，预热单独保留；不删除慢样本。比值大于 1 表示 OpenMP 输运比关闭 helper 更慢。', '',
        '| 模式 | 预热/正式次数 | CPU 输运中位数 [s] | 相对 helper-off | GPU driver 核时中位数 [CPU s] | blocking 次数中位数 | blocking 主机墙时中位数 [s] | RSS 峰值 [GiB] |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, row in report['mode_summaries'].items():
        lines.append('| {} | {}/{} | {} | {} | {} | {} | {} | {} |'.format(mode,
            row['warmups'], row['measured_repeats'], fmt(row['cpu_transport_median_seconds']),
            fmt(row['cpu_transport_ratio_to_helper_off'], 4), fmt(row.get('gpu_driver_thread_cpu_seconds_median')),
            fmt(row.get('gpu_blocking_wait_calls_median'), 0), fmt(row.get('gpu_blocking_wait_host_seconds_median')),
            fmt(row['rss_peak_bytes']/2**30 if row['rss_peak_bytes'] is not None else None)))
    lines += ['', '## 逐次记录（包含预热）', '',
        '| 行号 | repeat | 模式 | 预热 | CPU 输运 [s] | GPU 输运 [s] | GPU driver 核时 [CPU s] | 主机调用重叠窗 [s] |',
        '|---:|---:|---|---|---:|---:|---:|---:|']
    for entry in report['repeats']:
        row = entry['record']
        cpu = row.get('cpu') if isinstance(row.get('cpu'), dict) else {}
        gpu = row.get('gpu') if isinstance(row.get('gpu'), dict) else {}
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(entry['line'], row.get('repeat'),
            row.get('mode'), row.get('warmup'), fmt(cpu.get('transport_seconds')),
            fmt(gpu.get('transport_seconds')), fmt(gpu.get('driver_thread_cpu_seconds')),
            fmt(row.get('transport_host_call_window_overlap_seconds'))))
    lines += ['', '## 限制与解释', ''] + ['- '+v for v in report['limitations']]
    interference = report['known_interference']
    if interference['present']:
        lines += ['', '## 已知外部干扰：严格计时无效', '',
            interference['reason'], '',
            '即使 guard.performance_valid 为 true，NVML 也不能排除该 CPU 干扰；因此不认证这轮性能。'
            '全部 18 次记录（若已完成）、输出哈希和正确性结果原样保留，不删除可能受影响的慢样本。']
        if interference['issues']:
            lines += ['', '清单无法完整验证，同样按计时无效处理：', ''] + [
                '- '+p for p in interference['issues']]
    problems = report['issues'] + report['guard_issues']
    if problems:
        lines += ['', '## 未通过项', ''] + ['- '+p for p in problems]
    if not c['guard_passed'] and report['sources']['command_log'].get('nonrecord_tail'):
        lines += ['', '### 原始日志末尾（非结果行，保留失败原因）', '', '```text'] + [
            '{}: {}'.format(r['line'], r['text']) for r in report['sources']['command_log']['nonrecord_tail']] + ['```']
    if not c['performance_evidence_valid']:
        lines += ['', '监控、完整性或外部干扰门禁未通过：上面的时间只能作为描述性记录，不能声称性能通过。']
    lines += ['', '## 数据与输出校验', '', '| 来源 | SHA-256 |', '|---|---|']
    for name, source in report['sources'].items():
        lines.append('| {} | {} |'.format(name, source.get('sha256', '缺失')))
    if result_sha256:
        lines.append('| RESULT.json | {} |'.format(result_sha256))
    lines += ['', 'CPU 与 GPU 各自的输入、调用序列、profile、radio、粒子、fallback、计数、表及辅助缓存 SHA-256：', '',
              '```json', json.dumps(dict(cpu=report['expected_cpu_hashes'], gpu=report['expected_gpu_hashes']),
                                  ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('guard_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Fresh report directory; never overwritten')
    args = parser.parse_args()
    report = build_report(args.guard_dir)
    result = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    result_sha = hashlib.sha256(result).hexdigest()
    document = markdown(report, result_sha).encode('utf-8')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'RESULT.json').open('xb') as stream:
        stream.write(result)
    with (args.output/'REPORT_CN.md').open('xb') as stream:
        stream.write(document)
    print(json.dumps(dict(output=str(args.output), result_sha256=result_sha,
        markdown_sha256=hashlib.sha256(document).hexdigest(), checks=report['checks'])))
    return 0 if report['checks']['performance_evidence_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
