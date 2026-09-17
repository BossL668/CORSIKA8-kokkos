#!/usr/bin/env python3
"""Read-only acceptance of completed campaign cases; calculate only on PSR."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import socket

os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import yaml


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'psrpku2025':
        raise RuntimeError('PSR only; no local numerical verification')
    root = args.root
    manifest = json.loads((root / 'campaign.json').read_text())
    changed = [p['path'] for p in manifest['files'] + manifest['runtime_libraries'] if sha(p['path']) != p['sha256']]
    assert not changed, 'Archived physics inputs/runtime changed: ' + str(changed)
    supplied = yaml.safe_load((root / 'bundle/materials.yaml').read_text())
    cards = {p['id']: p for p in supplied['material_cards']}
    previous_scene = yaml.safe_load((root / 'bundle/previous_scene.yaml').read_text())
    report_rows = {row['tag']: row for row in json.loads((root / 'report/results.json').read_text())}
    accepted, pending = [], []
    for case in manifest['cases']:
        folder = root / 'runs' / case['tag']
        if not (folder / 'material_checks.json').exists() or case['tag'] not in report_rows:
            pending.append(case['tag'])
            continue
        status = json.loads((folder / 'status.json').read_text())
        saved_checks = json.loads((folder / 'checks.json').read_text())
        s = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
        report = report_rows[case['tag']]
        c = dict(complete=status['complete'] and s['complete'] and status['returncode'] == 0,
                 automatic_checks=all(saved_checks['checks'].values()),
                 unchanged_command=status['command'] == case['command'],
                 unchanged_binary=status['binary_sha256'] == manifest['binary_sha256'] and status['binary_unchanged'],
                 unchanged_scene=status['scene_sha256'] == sha(case['command'][case['command'].index('--scene') + 1]))
        ref = json.loads(pathlib.Path(case['reference']).read_text())['command']
        allowed = {3} | {ref.index(k) + 1 for k in ['--scene', '--output', '--aux-cache']}
        c['original_physics_arguments'] = len(ref) == len(case['command']) and all(
            a == b or i in allowed for i, (a, b) in enumerate(zip(ref, case['command'])))
        material = s['resolved_material']
        card = cards[material['id']]
        actual_mix = {(p['Z'], p['A']): p['number_fraction'] for p in material['transport']['nuclei']}
        expected_mix = {(p['Z'], p['A']): p['number_fraction'] for p in card['transport']['nuclei']}
        c['supplied_composition'] = actual_mix.keys() == expected_mix.keys() and all(
            abs(actual_mix[k] - expected_mix[k]) < 2e-15 for k in actual_mix)
        c['supplied_density'] = material['transport']['density_kg_m3'] == card['transport']['density_kg_m3']
        c['supplied_radio'] = all(abs(material['radio'][k] - card['radio'][k]) < 2e-13
                                 for k in ['refractive_index', 'field_attenuation_length_m'])
        scene = s['scene']
        c['original_atmosphere_and_observers'] = scene['atmosphere'] == previous_scene['atmosphere'] and all(
            scene['radio'][k] == v for k, v in previous_scene['radio'].items() if k != 'rock_attenuation_length_m')
        c['native_LPM'] = all(p['photon_pair_LPM'] for p in s['proposal_materials'])
        native = next(p for p in s['proposal_materials'] if p['name'] == material['id'])
        c['native_material'] = abs(native['density_g_cm3'] * 1000 - material['transport']['density_kg_m3']) < 1e-9 and native['I_eV'] == material['transport']['ionisation']['I_eV']
        a = s['accelerator']
        affinity = json.loads((folder / 'affinity.json').read_text())
        workers = list(affinity['workers'].values())
        c['actual_256_core_binding'] = all(len(v) == 1 for v in workers) and {v[0] for v in workers} == set(range(256))
        c['OpenMP_resident_queue'] = a['execution_space'] == 'OpenMP' and a['execution_concurrency'] == 256 and a['resident_capacity'] == 262144 and a['pending_particles'] == 0
        r = s['radio_result']
        c['paired_radio_complete'] = r['complete'] and r['execution_space'] == 'OpenMP' and r['algorithms'] == ['CoREAS', 'ZHS'] and r['errors'] == r['out_of_window'] == 0
        c['radio_finalization'] = r['downloads'] == 1 and r['moment_arrays'] == 3
        c['transport_inventory'] = all(r[alg]['device_tracks'] + r[alg]['cpu_tracks'] == saved_checks['charged_tracks'] for alg in ['CoREAS', 'ZHS'])
        c['no_output_truncation'] = not s['diagnostics']['csv_truncated'] and not s['diagnostics']['deposition_csv_truncated']
        ledger = s['energy_ledger']
        residual = (ledger['recorded_offset_GeV'] - ledger['predicted_offset_GeV']) / ledger['initial_GeV']
        c['ledger_recomputed'] = abs(residual - ledger['unexplained_over_initial']) < 1e-16 and abs(residual) < 1e-8
        spectra, fields = [], []
        for alg in ['CoREAS', 'ZHS']:
            radio_dir = folder / 'output/radio' / alg
            config = json.loads((radio_dir / 'config.json').read_text())
            c[alg + '_optics'] = config['media'][1]['index'] == material['radio']['refractive_index'] and config['media'][1]['attenuation_length_m'] == material['radio']['field_attenuation_length_m']
            spectrum = pd.read_csv(radio_dir / 'spectrum.csv', float_precision='round_trip')
            values = np.stack([spectrum[k + '_real'].to_numpy() + 1j * spectrum[k + '_imag'].to_numpy() for k in ['Ex', 'Ey', 'Ez']], axis=-1)
            spectra.append(values)
            frame = pd.read_csv(radio_dir / 'field.csv', float_precision='round_trip',
                                usecols=['observer', 'time_s', 'Ex_V_m', 'Ey_V_m', 'Ez_V_m'])
            times = frame['time_s'].to_numpy().reshape(3, config['samples'])
            target = np.arange(config['samples']) / config['sample_rate_Hz'] + config['start_time_s']
            c[alg + '_sampling'] = np.allclose(times, target, rtol=0, atol=1e-18) and np.array_equal(frame['observer'].to_numpy(), np.repeat(np.arange(3), config['samples']))
            fields.append(frame[['Ex_V_m', 'Ey_V_m', 'Ez_V_m']].to_numpy().reshape(3, config['samples'], 3))
        per_station = []
        frequencies = np.fft.rfftfreq(config['samples'], 1 / config['sample_rate_Hz'])
        for i, observer in enumerate(config['observers']):
            select = spectrum['observer'].to_numpy() == i
            spectrum_l2 = relative(spectra[0][select], spectra[1][select])
            field_l2 = relative(fields[0][i], fields[1][i])
            transformed = np.fft.rfft(fields[1][i], axis=0)
            transformed[(frequencies < 50e6) | (frequencies > 100e6)] = 0
            band_field = np.fft.irfft(transformed, n=config['samples'], axis=0)
            norm = np.linalg.norm(band_field, axis=1)
            peak_index = int(np.argmax(norm))
            peak = float(norm[peak_index])
            peak_time = float(target[peak_index] * 1e6)
            reported = next(p for p in report['peaks'] if p['observer'] == observer['name'])
            c[observer['name'] + '_reported_peak'] = abs(peak - reported['peak_V_m']) < 1e-12 * max(peak, 1e-100) and abs(peak_time - reported['peak_time_us']) < 1e-10
            per_station.append(dict(observer=observer['name'], spectrum_relative_l2=spectrum_l2,
                                    raw_field_relative_l2=field_l2, band_peak_V_m=peak, peak_time_us=peak_time))
        total_l2 = relative(spectra[0], spectra[1])
        c['spectrum_recomputed'] = abs(total_l2 - saved_checks['coreas_zhs_relative_l2']) < 1e-13
        c['each_station_algorithms'] = all(p['spectrum_relative_l2'] < 1e-5 and p['raw_field_relative_l2'] < 1e-5 for p in per_station)
        c['finite_radio'] = all(np.isfinite(v).all() for v in fields + spectra)
        if not all(c.values()):
            raise RuntimeError(case['tag'] + ': ' + str([k for k, v in c.items() if not v]))
        accepted.append(dict(tag=case['tag'], label=case['label'], seed=case['seed'], checks=c,
            ledger_relative=residual, spectrum_relative_l2=total_l2, stations=per_station,
            steps=s['diagnostics']['steps'], wall_minutes=status['wall_s'] / 60, decays=report['decays']))
        print('ACCEPTED', case['tag'], 'spectrum L2', total_l2, flush=True)
    result = dict(evaluated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        accepted_count=len(accepted), total=len(manifest['cases']), pending=pending,
        input_and_runtime_hashes_unchanged=True, accepted=accepted,
        scope='Independent completed-output audit; no new shower simulations. Existing full-track checks retained. Algebraic ledger closure is not a validation of all physical models.')
    (root / 'report/acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    local_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    md = ['# 100 PeV ντ 不同介质计算：验收记录', '',
          '验收时间：%s（北京时间）。**%d / %d 组完成并通过本次验收。**' % (local_time, len(accepted), len(manifest['cases'])), '',
          '每组保留此前的入射条件与随机种子，使用 OpenMP 256 个物理核、常驻队列及 CoREAS/ZHS 双算法。', '']
    if pending:
        md += ['尚未纳入本次验收：`' + '`、`'.join(pending) + '`。以服务器 `progress.json` 为实时状态；此记录不是九组全部完成的声明。', '']
    if accepted:
        spectra_l2 = [p['spectrum_relative_l2'] for p in accepted]
        station_l2 = [s['spectrum_relative_l2'] for p in accepted for s in p['stations']]
        field_l2 = [s['raw_field_relative_l2'] for p in accepted for s in p['stations']]
        md += ['本次直接读取原始 CSV 复核，未重跑 shower：', '',
               '- 初始物理参数、程序、运行库和材料输入散列一致；组分、密度、折射率和衰减长度符合提供的材料文件。',
               '- 实际 256 核绑定、OpenMP 并发数、LPM 开关、材料输运与射电接线通过；队列最终清空，无轨迹或沉积输出截断。',
               '- 能量账本最大相对闭合残差：%.3g。该值已计入账本中的生成器交换项、静质量处理和 thinning 权重跳变，不等同于全物理模型精度。' % max(abs(p['ledger_relative']) for p in accepted),
               '- 每事件 CoREAS/ZHS 全频复数谱相对 L2 差异范围：%.3g–%.3g；逐站最大 %.3g。' % (min(spectra_l2), max(spectra_l2), max(station_l2)),
               '- 原始时域电场逐站最大相对 L2 差异：%.3g；全部采样时刻和 50–100 MHz 峰值与报告一致。' % max(field_l2),
               '- 射电计算无错误、无接收窗外丢弃，两种算法共用完整带电轨迹来源；累计网格在最终阶段统一导出。', '',
               '| 材料 | seed | 输运步数 | 计算时间/min | 本次 τ 衰变 |', '|---|---:|---:|---:|---|']
        for row in accepted:
            category = '; '.join(p['channel'] + ' / ' + p['medium'] for p in row['decays']) or 'No decay in window'
            md.append('| %s | %d | %d | %.1f | %s |' % (row['label'], row['seed'], row['steps'], row['wall_minutes'], category))
    md += ['', '[查看 shower 与射电诊断图](SLIDES_CN.md)。图片均为英文标注；完整数值在 [acceptance.json](acceptance.json)。', '',
           '图中的材料差异同时包含 shower 随机发展、τ 衰变位置和传播参数变化，不能只解释为衰减长度变化。保留了原来的有限输运窗、截断和 thinning，本次通过数值与调度验收，尚不是这些近似的收敛结论或完整物理的独立验证。', '',
           '旧绘图日志中的 NumExpr 线程上限提示已在分析脚本中修正；本次图表重新生成，原始数据不受改动。模拟及数值验收均在 PSR 完成。', '']
    (root / 'report/ACCEPTANCE_CN.md').write_text('\n'.join(md))
    print(json.dumps(dict(accepted=len(accepted), pending=pending)), flush=True)


if __name__ == '__main__':
    main()
