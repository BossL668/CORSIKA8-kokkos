#!/usr/bin/env python3
"""Switch this PSR campaign to one shower+radio run per first-vertex candidate."""
import ast
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import yaml

ROOT=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_100PeV_ne_sw_130cores_20260916')
REV=ROOT/'revisions/20260916_direct_radio'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)


README_FUNCTION = '''def write_readme(root):
    p=read(root/'progress.json');m=read(root/'campaign.json')
    lines=['# 100 PeV SiO₂：首顶点筛选后直接计算 shower＋射电', '',
           f'状态：`{p["state"]}`；更新时间：{p["updated_utc"]}。',
           f'当前：`{p.get("current", "尚未启动")}`。', '',
           '1. 首顶点快筛已完成：30 个深度候选中有 23 CC、7 NC，19 个满足岩石 CC、τ 和首个强子系统各 ≥1 PeV；不是已确认 double bang。[候选与验证](report/first_vertex/README_CN.md)。',
           '2. 初筛通过后直接以 emthin=1e-3、Wmax=50000 运行完整 shower，同时计算全部 80 站 CoREAS 和 ZHS。没有单独的无射电完整输运复核阶段。',
           '3. 从同一次计算验收实际 CC→τ→空气中电子/强子衰变、两次可见能量、shower profile、能量账本、完整记录和射电。两个顶点各 ≥1 PeV 且间距 ≥100 m 才通过本轮 double bang 条件；否则保留结果并处理下一首顶点候选。得到一个通过的完整事件后停止。', '',
           '首例 seed 14251：100 PeV ντ，在出口前 59.08 m 发生天然 CC，产生 62.68 PeV τ 和 37.32 PeV 强子系统；τ 的出山与衰变以完整模拟为准。首次顶点未强制，筛选样本不能直接用于无偏事件率。', '',
           'SiO₂ 基准：2650 kg/m³、n=√5、场衰减长度 100 m。保留 LPM、IGRF14（2027）、天然 CC+NC、TAUOLA、DEM 覆盖边界。EM 截止 0.5 MeV，强子/μ/τ 截止 0.3 GeV；输运窗 500 μs。', '',
           'PSR OpenMP 130 个物理核，沿用既有 CPU 配置；其他任务亲和性保持原状。常驻队列 4,194,304，波前 4096，记录缓冲 65536，射电批次 8192。内存和磁盘保护保持启用。[CPU 配置](report/cpu_selection.json)。', '',
           '80 站坐标和 DEM+1 m 模型高度沿用已核对版本。CoREAS/ZHS 同时计算，1 GHz 采样（1 ns），1,048,576 点、1048.576 μs 接收窗；保存未经带通的波形，主图 50–250 MHz，附 50–300 MHz。[射电配置验证](report/radio_alignment/README_CN.md)。', '',
           '几何：东北→西南，方位角 225°、向下 6°，穿岩 874.69 m 后沿轴连续 7.062 km 为空气。朝向北臂，在 N10 水平位置经过天线上方 250 m。[几何与透射图](report/geometry/FIGURES_CN.md)。', '',
           '此前无射电复核已按用户要求停止；未完成事件不能用于物理结论。本次从原始初态启动带射电计算，随后所有候选都只安排一次完整模拟。[流程切换](report/direct_radio/README_CN.md)。', '',
           f'已完成带射电事件 {len(p["completed"])} 例；通过 double bang 条件 {len(p["qualified"])} 例。']
    if p.get('latest'):
        q=p['latest'];lines += ['',f'当前进程耗时 {q["elapsed_s"]/3600:.3f} h，已读取轨迹记录前缀 {q["recorded_steps_prefix"]:,}；RSS {q["rss_GiB"]:.2f} GiB。这不是完成百分比。']
    if p.get('error'):lines += ['', '停止原因：'+p['error']]
    for stage in p.get('stages',[]):
        lines += [f'- [{stage["tag"]}](report/{stage["tag"]}/README_CN.md)：double bang={stage["qualified"]}，耗时 {stage["wall_s"]:.2f} s。']
    lines += ['', '[输入与命令](campaign.json) · [状态](progress.json)', '', 'PSR：`'+str(root)+'`。']
    (root/'README_CN.md').write_text('\\n'.join(lines)+'\\n')


'''


def main():
    assert socket.gethostname()=='psrpku2025'
    assert not (REV/'applied.json').exists()
    interrupted=read(REV/'interrupted.json')
    for key in ['controller_pid','simulation_pid']:
        p=Path('/proc')/str(interrupted[key])
        assert not p.exists() or (p/'stat').read_text().split(') ',1)[1][0]=='Z'
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(ROOT/'campaign.json');before=copy.deepcopy(m)
    for row in m['files']+m['runtime_libraries']:
        assert sha(Path(row['path']))==row['sha256'],row['path']
    assert read(ROOT/'report/first_vertex/regression.json')['passed']
    assert read(ROOT/'report/first_vertex/new_geometry_prefix_check.json')['passed']
    alignment=read(ROOT/'report/radio_alignment/acceptance.json')
    assert alignment['passed'] and alignment['validation']['passed']
    radio_scene=yaml.safe_load((ROOT/'scenes/radio.yaml').read_text())
    screen_scene=yaml.safe_load((ROOT/'scenes/screen.yaml').read_text())
    radio=radio_scene['radio']
    assert radio['enabled'] and len(radio['observers'])==80
    assert radio['observers']==screen_scene['radio']['observers']
    assert radio['sample_rate_GHz']==1. and radio['samples']==1048576
    assert sha(ROOT/'bundle/c8_terrain_cascade_radio_1GHz')==alignment['validation']['new_binary_sha256']
    m['stage_order']=['radio']
    m['cases']=[c for c in m['cases'] if c['stage']=='radio']
    assert len(m['cases'])==len(m['seeds'])==19 and m['seeds'][0]==14251
    for c in m['cases']:
        cmd=c['command'];old=next(x for x in before['cases'] if x['tag']==c['tag'])['command']
        assert c['emthin']==1e-3 and cmd==old
        assert not (ROOT/'runs'/c['tag']).exists()
        assert Path(cmd[cmd.index('--scene')+1])==ROOT/'scenes/radio.yaml'
        assert cmd[cmd.index('--threads')+1]=='130'
        assert float(cmd[cmd.index('--emcut-GeV')+1])==.0005
        assert float(cmd[cmd.index('--hadcut-GeV')+1])==.3
        assert float(cmd[cmd.index('--mucut-GeV')+1])==.3
        # Explicit in the command as well as enabled in the scene card.
        if '--radio' not in cmd:cmd.append('--radio')
        assert cmd==old+['--radio']
    source=ROOT/'code/run_psr.py';shutil.copy2(source,REV/'driver_before.py')
    text=source.read_text();start=text.index('def write_readme(root):');end=text.index('def prepare(root):',start)
    text=text[:start]+README_FUNCTION+text[end:]
    text=text.replace("for stage in m.get('stage_order', ['coarse','confirm','radio']):", "for stage in m['stage_order']:")
    ast.parse(text,filename=str(source));source.write_text(text)
    shutil.copy2(__file__,ROOT/'code/direct_radio_after_vertex_psr.py')
    report=ROOT/'report/direct_radio';report.mkdir(exist_ok=True)
    save(report/'interrupted_confirmation.json',interrupted)
    preflight=dict(passed=True,checked_utc=now(),stage_order=m['stage_order'],candidates=len(m['seeds']),
                   all_commands_have_explicit_radio=True,scene_radio_enabled=True,station_count=80,
                   production_command_changes=['append explicit --radio; original scene already enabled radio'],
                   stage_removed='confirm (complete shower without radio)',
                   physics_binaries_unchanged=True,radio_binary_matches_validated_1GHz_build=True,
                   geometry_and_station_positions_unchanged=True,air_default_cuts_unchanged=True,
                   emthin=1e-3,threads=130,python_syntax_passed=True)
    save(report/'preflight.json',preflight)
    (report/'README_CN.md').write_text('''# 首顶点快筛后直接计算 shower＋射电

按用户要求，去掉无射电的完整输运复核。首顶点快筛通过后，每个候选只安排一次完整的 shower＋CoREAS/ZHS 计算；从同一次输出验收实际 double bang 和射电。

原无射电 seed 14251 任务已经停止，未完成结果不算正式完成事例。带射电任务使用原始 100 PeV 初态重新开始；原先未计算的射电不能在进程中途补齐。

保持 seed 14251、SiO₂、东北→西南入射、emthin=1e-3、空气默认低截止、130 核、全部 80 站。使用已有验证通过的 1 GHz 射电二进制和配置，并在命令行显式添加 `--radio`；两个算法默认同时计算。

首顶点候选不是已确认 double bang。最终在这一次带射电模拟中确认 τ 的出山、空气衰变、两次 shower、profile、能量账本和波形；若不满足 double bang 选图条件，再处理下一候选。

[配置检查](preflight.json) · [停止记录](interrupted_confirmation.json) · [当前状态](../../progress.json)
''')
    p=read(ROOT/'progress.json');assert not p['stages'] and not p['completed'] and not p['qualified']
    p.update(state='prepared_direct_radio',current=m['cases'][0]['tag'],updated_utc=now())
    p.pop('latest',None)
    p['interrupted_runs'].append(str(ROOT/'runs'/interrupted['previous_progress']['current']))
    save(ROOT/'progress.json',p)
    m['production_flow']=dict(applied_utc=now(),rule='First vertex candidates -> one complete shower+radio run -> audit same output',
                             separate_radio_disabled_confirmation=False,first_candidate=14251)
    for row in m['files']:row['sha256']=sha(Path(row['path']))
    for f in [ROOT/'code/direct_radio_after_vertex_psr.py',report/'preflight.json']:
        m['files'].append(dict(path=str(f),sha256=sha(f)))
    save(ROOT/'campaign.json',m)
    for f in [ROOT/'report/first_vertex/README_CN.md',ROOT/'report/radio_alignment/README_CN.md']:
        shutil.copy2(f,REV/(f.parent.name+'_README_before.md'))
        note='\n\n2026-09-16 流程更正：首顶点快筛后直接运行一次完整 shower＋80 站 CoREAS/ZHS，从同一次输出验收 double bang；取消无射电完整复核。[当前流程](../direct_radio/README_CN.md)。\n'
        f.write_text(f.read_text()+note)
    spec=importlib.util.spec_from_file_location('driver',source);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.CPU_LIST=m['cpu_list'];driver.write_readme(ROOT)
    (ROOT/'STOP_AFTER_CURRENT').unlink()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    with (ROOT/'driver.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(source),'--root',str(ROOT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'driver.pid').write_text(str(child.pid)+'\n')
    receipt=dict(applied_utc=now(),driver_pid=child.pid,first_case=m['cases'][0]['tag'],stage_order=m['stage_order'])
    save(REV/'applied.json',receipt);save(report/'applied.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
