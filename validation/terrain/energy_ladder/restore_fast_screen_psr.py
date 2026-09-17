#!/usr/bin/env python3
"""Restore documented historical screening cuts; retain production cuts and radio."""
import argparse
import ast
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
ROOT = BASE/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916'
OLD = BASE/'beta5_nutau100PeV_openmp120_20260910_v1'
RADIO = ROOT/'revisions/20260916_air_radio_1GHz'
REV = ROOT/'revisions/20260916_historical_screen_cuts'
PARAMS = {'--emcut-GeV': '.1', '--hadcut-GeV': '10', '--mucut-GeV': '.3',
          '--emthin': '.01', '--max-weight': '50000', '--transport-window-ns': '100000'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def arg(cmd, key):
    return cmd[cmd.index(key)+1]


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def stop(pid, required, group=False):
    proc = Path('/proc')/str(pid)
    if not proc.exists():
        return
    command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert required in command, (pid, command)
    if group:
        assert os.getpgid(pid) == pid
    send = (lambda sig: os.killpg(pid, sig)) if group else (lambda sig: os.kill(pid, sig))
    send(signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or (proc/'stat').read_text().split(') ', 1)[1][0] == 'Z':
            return
        time.sleep(.1)
    send(signal.SIGKILL)
    for _ in range(50):
        if not proc.exists() or (proc/'stat').read_text().split(') ', 1)[1][0] == 'Z':
            return
        time.sleep(.1)
    raise RuntimeError(f'Process {pid} did not stop')


def prepare():
    assert not (REV/'preflight.json').exists() and not (REV/'applied.json').exists()
    p = read(ROOT/'progress.json')
    assert p['state'] == 'running' and p['current'] == 'coarse_SiO2_100PeV_seed946'
    assert not p['stages'] and not p['qualified'] and not p['completed']
    assert read(RADIO/'handoff.json')['state'] == 'waiting_for_complete_stage'
    assert sha(ROOT/'campaign.json') == sha(RADIO/'before/campaign.json')
    old = read(OLD/'command_946.json')
    for key, value in PARAMS.items():
        assert float(arg(old, key)) == float(value)
    admission = read(RADIO/'admission.json')
    assert admission['validation']['passed']
    assert sha(ROOT/'bundle/c8_terrain_cascade_radio_1GHz') == admission['validation']['new_binary_sha256']
    for row in read(ROOT/'campaign.json')['files']:
        assert sha(Path(row['path'])) == row['sha256'], row['path']
    relatives = ['scenes/radio.yaml', 'code/run_psr.py', 'code/audit_psr.py', 'code/event_runner.py']
    for relative in relatives:
        assert sha(ROOT/relative) == sha(RADIO/'before'/relative), relative
        for side, source in [('before', ROOT), ('after', RADIO/'after')]:
            dest = REV/side/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, dest)
    for name in ['campaign.json', 'progress.json', 'README_CN.md']:
        shutil.copy2(ROOT/name, REV/'before'/name)
    shutil.copy2(OLD/'command_946.json', REV/'historical_command_946.json')
    shutil.copy2(OLD/'status.json', REV/'historical_status.json')
    m = read(RADIO/'after/campaign.json')
    for case in m['cases']:
        if case['stage'] != 'coarse':
            continue
        case['tag'] = case['tag'].replace('coarse_', 'coarse_legacycuts_', 1)
        cmd = case['command']
        for key, value in {**PARAMS, '--output': str(ROOT/'runs'/case['tag']/'output')}.items():
            if key in cmd:
                cmd[cmd.index(key)+1] = value
            else:
                cmd.extend([key, value])
        case['screening_only'] = True
        case['historical_physics_options'] = PARAMS
    source = REV/'after/code/run_psr.py'
    text = source.read_text()
    text = replace_once(text,
        '1. `emthin=1e-2`、自动 Wmax=500000，不带射电，完成真实输运和谱系筛选。',
        '1. 旧参数快速筛选：EM 100 MeV、强子 10 GeV、μ/τ 0.3 GeV，`emthin=1e-2`、显式 Wmax=50000、输运窗 100 μs，不带射电。完整输运后核对实际谱系。')
    text = replace_once(text,
        '以及 EM 0.5 MeV / 强子和 μ、τ 0.3 GeV 截止。',
        '复核及最终射电使用 EM 0.5 MeV / 强子和 μ、τ 0.3 GeV 截止；只有粗筛采用上列旧截止。')
    text = replace_once(text,
        '保留完整轨迹、沉积、出界末态和每阶段耗时。输运窗 500 μs；',
        '保留完整轨迹、沉积、出界末态和每阶段耗时。粗筛输运窗 100 μs，复核与最终射电为 500 μs；')
    text = replace_once(text,
        '粗筛/确认使用原二进制；射电阶段使用已验证的逐数组释放输出版本，物理公式与输运不变。',
        '输运二进制保持当前已修正版本，射电使用已验证的 1 GHz 版本。旧记录中 seed 158/946/3605 分别用时 15.2/22.3/44.3 分钟，当前耗时仍需实测。')
    source.write_text(text)
    runner = REV/'after/code/event_runner.py'
    runner.write_text(runner.read_text().replace('256 physical-core binding not verified', '130 physical-core binding not verified'))
    for path in (REV/'after/code').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    original = read(RADIO/'after/campaign.json')
    for a, b in zip(original['cases'], m['cases']):
        if a['stage'] != 'coarse':
            assert a == b, 'Production configuration changed'
        else:
            assert all(float(arg(b['command'], k)) == float(v) for k, v in PARAMS.items())
            assert b['command'][2] == m['cpu_list']
            assert int(arg(b['command'], '--threads')) == 130
    m['screening_revision'] = dict(created_utc=now(), reference=str(OLD), options=PARAMS,
        reason='User requested restoring the documented approximately 15-minute historical screening setup.',
        scope='Only coarse screening cuts, weight cap and time window restored; current corrected physics, material, DEM boundary, resident scheduler and CPU allocation retained. Confirmation and radio stay at air cuts / emthin=1e-3 / 500 us.',
        incomplete_previous_run_not_a_candidate=True)
    m['radio_revision'] = admission
    m['binary_sha256_by_stage'] = {'coarse': m['binary_sha256'], 'confirm': m['binary_sha256'],
        'radio': admission['validation']['new_binary_sha256']}
    save(REV/'after/campaign.json', m)
    save(REV/'preflight.json', dict(passed=True, checked_utc=now(), old_command_matches=True,
        confirmation_and_radio_commands_unchanged=True, python_syntax_passed=True,
        old_runs=read(OLD/'status.json'), live_input_hashes_verified=True))
    print('PREPARED', REV, flush=True)


def apply():
    assert read(REV/'preflight.json')['passed']
    assert sha(ROOT/'campaign.json') == sha(REV/'before/campaign.json')
    p = read(ROOT/'progress.json')
    assert p['current'] == 'coarse_SiO2_100PeV_seed946' and not p['stages']
    run = ROOT/'runs'/p['current']
    process = int((run/'pid').read_text())
    controller = int((ROOT/'driver.pid').read_text())
    handoff = read(RADIO/'handoff.json')['pid']
    save(REV/'interrupted.json', dict(stopped_utc=now(), reason='User requested historical screening settings',
        complete=False, classification='Not an accepted event', progress=p,
        simulation_pid=process, controller_pid=controller, handoff_pid=handoff))
    stop(handoff, 'stage_air_radio_psr.py --handoff')
    stop(controller, str(ROOT/'code/run_psr.py'))
    stop(process, str(run/'output'), group=True)
    lock = (ROOT/'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = ROOT/'abandoned'/run.name
    target.parent.mkdir(exist_ok=True)
    assert not target.exists()
    run.rename(target)
    save(target/'INTERRUPTED.json', read(REV/'interrupted.json'))
    files = [dict(path=str(p.relative_to(target)), bytes=p.stat().st_size)
             for p in target.rglob('*') if p.is_file()]
    save(REV/'retained_partial_files.json', dict(files=files, total_bytes=sum(p['bytes'] for p in files), deleted=False))
    for relative in ['scenes/radio.yaml', 'code/run_psr.py', 'code/audit_psr.py', 'code/event_runner.py']:
        assert sha(ROOT/relative) == sha(REV/'before'/relative)
        shutil.copy2(REV/'after'/relative, ROOT/relative)
    m = read(REV/'after/campaign.json')
    for row in m['files']:
        row['sha256'] = sha(Path(row['path']))
    binary = ROOT/'bundle/c8_terrain_cascade_radio_1GHz'
    if str(binary) not in [row['path'] for row in m['files']]:
        m['files'].append(dict(path=str(binary), sha256=sha(binary)))
    save(ROOT/'campaign.json', m)
    save(ROOT/'report/radio_alignment_pending.json', dict(state='applied', revision=str(RADIO),
        admission=m['radio_revision'], reason='Applied together with user-requested coarse-screen restart.'))
    save(RADIO/'handoff.json', dict(state='applied_by_screening_revision', revision=str(REV), updated_utc=now()))
    (ROOT/'STOP_AFTER_CURRENT').unlink()
    p = dict(state='prepared_historical_screen', completed=[], qualified=[], stages=[], updated_utc=now(),
        interrupted_runs=[str(target)], current=m['cases'][0]['tag'])
    save(ROOT/'progress.json', p)
    report = ROOT/'report/screening_revision'
    report.mkdir(exist_ok=True)
    for name in ['preflight.json', 'interrupted.json', 'historical_command_946.json', 'historical_status.json', 'retained_partial_files.json']:
        shutil.copy2(REV/name, report/name)
    (report/'README_CN.md').write_text('旧参数快速筛选已恢复。\n\n历史 100 PeV / 120 线程记录：seed 158 为 911.06 s（15.2 min），946 为 1336.54 s（22.3 min），3605 为 2657.99 s（44.3 min）。15 分钟是其中一例，不能当作每例固定耗时。\n\n| 参数 | 被替换的粗筛 | 恢复后的粗筛 | 后续复核及射电 |\n|---|---:|---:|---:|\n| EM 截止 | 0.5 MeV | 100 MeV | 0.5 MeV |\n| 强子截止 | 0.3 GeV | 10 GeV | 0.3 GeV |\n| μ/τ 截止 | 0.3 GeV | 0.3 GeV | 0.3 GeV |\n| emthin | 1e-2 | 1e-2 | 1e-3 |\n| Wmax | 500000 | 50000 | 50000 |\n| 输运窗 | 500 μs | 100 μs | 500 μs |\n\n维持当前已修正的输运、LPM、磁场、SiO₂ 材料、DEM 出界模块和已选定的 130 个物理核；不回退旧程序。粗筛不计算射电，完整轨迹和真实 CC→τ→衰变谱系仍需通过检查。筛选参数会影响随机历史，后续低截止及 1e-3 薄化必须重新确认 double bang。100 μs 筛选窗会漏选较晚的衰变，因此本流程只找展示候选，不估计无偏事件率。\n\n旧的低截止粗筛已停止，未完成输出保留在 abandoned，不能当作成功或失败的物理事件。1 GHz 射电配置已同时安装，最终使用 80 站、CoREAS/ZHS、50–250 MHz 主图，并附未经带通和 50–300 MHz。\n\n[历史耗时](historical_status.json) · [历史命令](historical_command_946.json) · [切换检查](preflight.json) · [中断记录](interrupted.json)\n')
    spec = importlib.util.spec_from_file_location('driver', ROOT/'code/run_psr.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.CPU_LIST = m['cpu_list']
    driver.write_readme(ROOT)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    with (ROOT/'driver.log').open('a') as log:
        child = subprocess.Popen([sys.executable, str(ROOT/'code/run_psr.py'), '--root', str(ROOT)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT/'driver.pid').write_text(str(child.pid)+'\n')
    save(REV/'applied.json', dict(applied_utc=now(), driver_pid=child.pid,
        first_case=m['cases'][0]['tag'], old_partial_data_retained=str(target)))
    print('STARTED', child.pid, m['cases'][0]['tag'], flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'psrpku2025', 'Run only on PSR'
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not (REV/'preflight.json').exists():
        prepare()
    if args.apply:
        apply()
