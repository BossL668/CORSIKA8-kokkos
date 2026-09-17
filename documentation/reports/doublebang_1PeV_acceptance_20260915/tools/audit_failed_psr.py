#!/usr/bin/env python3
"""Inspect a failed event on PSR. Never classify its saved prefix as complete."""
import argparse
from pathlib import Path
import csv
import datetime
import gzip
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
campaign = json.loads((root/'campaign.json').read_text())
progress = json.loads((root/'progress.json').read_text())
tag = progress['current']
run = root/'runs'/tag
out = root/'report/failure_acceptance_20260915'
out.mkdir(exist_ok=True)
status = json.loads((run/'status.json').read_text())
timing = json.loads((run/'timing.json').read_text())
affinity = json.loads((run/'affinity.json').read_text())
scene = yaml.safe_load((run/'output/terrain_run.yaml').read_text())
summary = yaml.safe_load((run/'output/terrain/summary.yaml').read_text())
command = json.loads((run/'command.json').read_text())
source = np.array(scene['position_m'], dtype=float)
direction = np.array(scene['direction'], dtype=float)
direction /= np.linalg.norm(direction)
started = time.monotonic()
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
binary_ok = sha(command[3]) == campaign['binary_sha256'] == status['binary_sha256']
library_checks = [{**r, 'matches': sha(r['path']) == r['sha256']}
                  for r in campaign['runtime_libraries']]
input_checks = [{**r, 'matches': sha(r['path']) == r['sha256']}
                for r in campaign['files']]
checks = []
profiles = {}
non_em = []
last_step = 0
primary_tail = None
for name in ['tracks.csv.gz', 'deposits.csv.gz', 'domain_exits.csv.gz', 'window_survivors.csv.gz']:
    path = run/'output/terrain'/name
    row_count = 0
    nonfinite = 0
    total_energy = []
    medium_counts = {}
    pdg_counts = {}
    step_gaps = 0
    for chunk in pd.read_csv(path, chunksize=100000):
        row_count += len(chunk)
        numeric = chunk.select_dtypes(include=[np.number]).to_numpy()
        nonfinite += int((~np.isfinite(numeric)).sum())
        if 'pdg' in chunk:
            for key, val in chunk['pdg'].value_counts().items():
                pdg_counts[str(int(key))] = pdg_counts.get(str(int(key)), 0) + int(val)
        if name == 'tracks.csv.gz':
            steps = chunk['step'].to_numpy()
            if len(steps):
                step_gaps += int(steps[0] != last_step+1) + int((np.diff(steps) != 1).sum())
                last_step = int(steps[-1])
            for key, val in chunk['medium'].value_counts().items():
                medium_counts[key] = medium_counts.get(key, 0)+int(val)
            subset = chunk[~chunk['pdg'].isin([-11, 11, 22])]
            if len(subset): non_em.append(subset.copy())
            prim = chunk[chunk['history_id'] == 1]
            if len(prim): primary_tail = json.loads(prim.tail(1).to_json(orient='records'))[0]
        elif name == 'deposits.csv.gz':
            energy = chunk['weighted_deposited_GeV'].to_numpy()
            total_energy.append(float(energy.sum()))
            mid = .5*(chunk[['x0_m','y0_m','z0_m']].to_numpy()+chunk[['x1_m','y1_m','z1_m']].to_numpy())
            distance = (mid-source) @ direction
            keys = np.floor(distance/25.).astype(np.int64)
            keys_unique, reverse = np.unique(keys, return_inverse=True)
            sums = np.bincount(reverse, weights=energy)
            for key, value in zip(keys_unique, sums):
                profiles[int(key)] = profiles.get(int(key), 0.)+float(value)
        elif name == 'domain_exits.csv.gz':
            total_energy.append(float((chunk['weight']*chunk['total_GeV']).sum()))
        elif name == 'window_survivors.csv.gz':
            total_energy.append(float((chunk['weight']*chunk['total_GeV']).sum()))
    expected = {'tracks.csv.gz':summary['steps'], 'deposits.csv.gz':summary['deposition_rows'],
                'domain_exits.csv.gz':summary['domain_exits'],
                'window_survivors.csv.gz':summary['finite_window_survivors']}[name]
    item = dict(file=name, rows=row_count, summary_rows=expected, row_count_matches=row_count == expected,
                gzip_read_to_eof=True, nonfinite_numeric_values=nonfinite, compressed_bytes=path.stat().st_size)
    if total_energy: item['weighted_energy_sum_GeV'] = math.fsum(total_energy)
    if name == 'tracks.csv.gz':
        item.update(last_step=last_step, step_gaps=step_gaps, medium_counts=medium_counts, pdg_counts=pdg_counts)
    checks.append(item)
    print(json.dumps(item), flush=True)

cpu = pd.concat(non_em, ignore_index=True) if non_em else pd.DataFrame()
tau = cpu[cpu['pdg'].abs() == 15] if len(cpu) else cpu
if len(tau): tau.to_csv(out/'partial_tau_tracks.csv', index=False)
prefix_ok = all(c['row_count_matches'] and c['nonfinite_numeric_values'] == 0 for c in checks) and checks[0]['step_gaps'] == 0
deposited = checks[1]['weighted_energy_sum_GeV']
deposit_difference = deposited-summary['deposited_energy_GeV']
deposit_ok = abs(deposit_difference) <= 1.e-8*max(1.,abs(deposited))
actual_scene = yaml.safe_load(Path(command[command.index('--scene')+1]).read_text())
observers = actual_scene['radio']['observers']
observer_names = [x['name'] for x in observers]
expected_names = {p+('%02d'%i) for p in ['E','W','N','S'] for i in range(1,21)}
station_audit = json.loads((root/'bundle/stations_80_audit.json').read_text())
stations_ok = len(observer_names) == 80 and set(observer_names) == expected_names and len(set(observer_names)) == 80
station_bundle_ok = sha(root/'bundle/observers.yaml') == station_audit['source_sha256']['observers.yaml']
bundled_observers = yaml.safe_load((root/'bundle/observers.yaml').read_text())['radio']['observers']
station_positions_ok = ({x['name']: x['position_enu_m'] for x in observers} ==
                        {x['name']: x['position_enu_m'] for x in bundled_observers})
radio_files = [str(f.relative_to(run)) for f in run.rglob('*') if f.name in
               ['field.csv','spectrum.csv','moments.bin','regularized_moments.bin']]
resources = pd.read_json(run/'resources.jsonl', lines=True)
last_resource = json.loads(resources.tail(1).to_json(orient='records'))[0]
queue_capacity = int(command[command.index('--resident-capacity')+1])
profile = pd.DataFrame([{'axis_start_m':25*k, 'axis_center_m':25*k+12.5,
                         'weighted_deposited_GeV':v, 'dE_daxis_TeV_per_m':v/1000./25.,
                         'scope':'INCOMPLETE_SAVED_PREFIX'} for k,v in sorted(profiles.items())])
profile.to_csv(out/'partial_deposition_profile.csv', index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig, ax = plt.subplots(1,2,figsize=(12.5,4.1),constrained_layout=True)
ax[0].plot(resources['elapsed_s']/3600.,resources['recorded_steps_prefix']/1.e6)
ax[0].scatter([status['wall_s']/3600.],[summary['steps']/1.e6],color='#c83232',label='Final saved prefix')
ax[0].set(xlabel='Wall time since process launch (h)',ylabel='Recorded transport steps (millions)',title='Failed at the resident queue capacity limit')
ax[0].legend(fontsize=9)
ax[1].plot(resources['elapsed_s']/3600.,resources['rss_GiB'],label='Process RSS')
ax[1].plot(resources['elapsed_s']/3600.,resources['available_GiB'],label='System available RAM')
ax[1].set(xlabel='Wall time since process launch (h)',ylabel='Memory (GiB)',title='No memory-guard termination')
ax[1].legend(fontsize=9)
for a in ax:a.grid(alpha=.2)
fig.savefig(out/'failure_resources.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
if len(profile):ax.bar(profile['axis_center_m'],profile['dE_daxis_TeV_per_m'],width=25,color='#387ca3')
ax.set(xlabel='Distance along primary axis from injection (m)',ylabel='Recorded deposition (TeV/m)',
       title='INCOMPLETE EVENT: saved deposition prefix only, 25 m bins')
ax.text(.98,.96,'Not a complete shower profile; Xmax cannot be inferred',transform=ax.transAxes,
        ha='right',va='top',fontsize=10,color='#a52222')
ax.grid(alpha=.2);fig.savefig(out/'partial_profile_INCOMPLETE.png',dpi=170);plt.close(fig)
verdict = dict(audit_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=tag,
    passed=False,classification='INCOMPLETE_QUEUE_CAPACITY_FAILURE',simulation_complete=bool(scene['complete']),
    status=status,timing=timing,affinity=affinity,queue_capacity=queue_capacity,
    saved_prefix_readable=prefix_ok and deposit_ok,files=checks,
    deposited_energy_summary_GeV=summary['deposited_energy_GeV'],deposit_csv_minus_summary_GeV=deposit_difference,
    tau_track_rows_in_saved_prefix=len(tau),last_primary_track=primary_tail,
    topology='not accepted: complete interaction/decay diagnostics unavailable',
    full_energy_closure='not available: live queues and incomplete buffered records are not a final energy ledger',
    radio_output_files=radio_files,radio_validation='unavailable: final spectra/waveforms were not produced',
    observer_input_valid=stations_ok and station_bundle_ok and station_positions_ok and station_audit['names_complete_unique'],
    observer_coordinates_match_audited_bundle=station_positions_ok,
    station_input_audit=station_audit,station_geometry_note='Prior validated coordinates; DEM+1m model heights, not measured phase centers',
    binary_matches_campaign=binary_ok,runtime_libraries=library_checks,campaign_file_checks=input_checks,last_resource=last_resource,
    production_changes='none; no simulation restarted; failed data preserved',
    audit_wall_seconds=time.monotonic()-started)
(out/'acceptance.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+'\n')
hours=int(status['wall_s']//3600);minutes=int(status['wall_s']%3600//60);seconds=status['wall_s']%60
text=f'''# 1 PeV double bang 验收：未通过，队列容量导致中断

事件 `{tag}`。2026-09-15 15:02（北京时间）结束，退出码 {status['returncode']}；`complete=false`。成功完成及通过验收均为 **0 例**。种子扫描在首个候选失败后停止，后续候选没有启动。

## 运行与失败原因

- 进程总耗时 **{hours} 小时 {minutes} 分 {seconds:.1f} 秒**，包括初始化到报错退出；不是完整 shower 的耗时。
- OpenMP **256 个物理核**绑定通过；峰值 RSS **{status['peak_rss_GiB']:.3f} GiB**。
- 固定常驻队列上限 **{queue_capacity:,} 个活跃 EM 粒子**。某批处理后，未处理粒子与新后继粒子合计将超过上限，因此抛出 `interface resident successors exceed capacity; input retained`。
- 最后一次资源采样仍有约 **{last_resource['available_GiB']:.1f} GiB 可用内存、{last_resource['free_disk_GiB']:.1f} GiB 空闲磁盘**；监控器没有因资源阈值终止任务。
- `input retained` 指异常发生前尚未消费该批内存中的输入；当前程序没有把它保存为可恢复检查点。异常退出后，不能从这些 CSV 直接续跑。

队列的容量检查发生在消费原批次之前，避免静默丢弃后继粒子。但当前实现没有在满载时自动扩容或转入可继续运行的溢出处理。容量上限不足是运行配置与调度处理的限制，不能靠加强 thinning 或提高粒子截止来掩盖。

![Failure resources and saved steps](failure_resources.png)

左图是已写出记录数随耗时变化，红点为最终保存数；右图是进程内存和系统可用内存。曲线终点表示失败停止。

## 已保存数据的完整性

- 轨迹 **{checks[0]['rows']:,} 行**；沉积 **{checks[1]['rows']:,} 行**。4 个 gzip 文件均读取到末尾并通过解压校验，行数与 summary 对应，数值无非有限项，step 连续：**{prefix_ok}**。
- CSV 沉积总和 **{deposited:.9f} GeV**，与 summary 的差为 **{deposit_difference:.3g} GeV**，保存前缀内部核对：**{deposit_ok}**。
- 已保存 τ 运输记录 **{len(tau)} 行**。完整的 CC/τ 衰变顶点及 double bang 选图验收未完成，不能仅凭候选种子或任务名确认事件合格。
- 这些是已完成输出检查点的前缀。抛异常的常驻调用可能还有未写出的缓冲记录，以及未输运的活粒子；`csv_truncated=false` 只表示没有触发 CSV 行数上限。
- 缺少完整终态和活粒子账本，因此**无法验收整事件的能量闭合**。不能把输入能量减去目前沉积，解释成能量丢失；也不能把沉积占比当作完成百分比。

![Incomplete deposition profile](partial_profile_INCOMPLETE.png)

本图只展示已保存的沉积前缀，数据见 [25 m 分箱](partial_deposition_profile.csv)。它不是完整 shower profile，不能据此判断最终 Xmax。事件内不同粒子的处理顺序也不是全局时间排序。

## 射电与观测站

- 输入包含 E/W/N/S 四臂各 20 个站，名称唯一且齐全；实际场景中每站的三维坐标与已核对的输入包逐项一致：**{stations_ok and station_bundle_ok and station_positions_ok}**。
- 经纬度沿用已验收输入；高度为 **DEM+1 m 的模型高度**，不是重新测得的天线相位中心。
- 本次没有生成最终 `field.csv`、`spectrum.csv` 或矩文件；射电结束后的下载、重建及输出未执行。因此 **80 站信号与 CoREAS/ZHS 一致性均无法验收**。

## 验收结论与后续条件

**未通过。** 物理设置仍为 1 PeV ντ、SiO₂、emthin=1e-6、空气默认截止与 Wmax=0.5；该 Wmax 下单位权重粒子不实际薄化。此次失败首先暴露常驻队列容量处理不足，不能据此评判完整 shower 或射电物理结果。

重跑前应处理队列满载：在既定内存预算内安全扩容，或实现保持粒子及后继状态的溢出调度，并验证恰好一次消费、随机数状态、记录输出和能量账本。简单增大固定上限只能推迟下一次满载，不能作为通用修复。失败时还应输出队列水位及所需后继数，当前记录未保存这些精确值。

本次只进行了验收、已有文件读取与诊断绘图，均在 PSR；没有重启模拟、修改生产程序或删除失败数据。本地仅同步小型报告。二进制校验：**{binary_ok}**；运行库校验：**{all(x['matches'] for x in library_checks)}**；已记录输入与运行脚本的 SHA256 校验：**{all(x['matches'] for x in input_checks)}**。

[机器可读验收](acceptance.json) · [完整输入、退出及耗时证据](evidence/)
'''
(out/'README_CN.md').write_text(text)
evidence=out/'evidence';evidence.mkdir(exist_ok=True)
import shutil
for name in ['status.json','timing.json','affinity.json','command.json','STARTED.json','run.log']:
    shutil.copy2(run/name,evidence/name)
for name in ['terrain_run.yaml','summary.yaml']:
    shutil.copy2(run/'output'/name,evidence/name)
shutil.copy2(run/'output/terrain/summary.yaml',evidence/'terrain_summary.yaml')
shutil.copy2(root/'progress.json',evidence/'campaign_progress.json')
shutil.copy2(root/'bundle/stations_80_audit.json',evidence/'stations_80_audit.json')
shutil.copy2(root/'bundle/stations_80_audit.csv',evidence/'stations_80_audit.csv')
shutil.copy2(run/'resources.jsonl',evidence/'resources.jsonl')
shutil.copy2(root/'campaign.json',evidence/'campaign.json')
shutil.copy2(Path(command[command.index('--scene')+1]),evidence/'scene.yaml')
shutil.copy2(root/'bundle/silica_SiO2.yaml',evidence/'silica_SiO2.yaml')
print(json.dumps({'passed':False,'saved_prefix_readable':prefix_ok and deposit_ok,
                  'tau_track_rows':len(tau),'radio_files':radio_files,
                  'audit_wall_seconds':time.monotonic()-started,'report':str(out)}),flush=True)
