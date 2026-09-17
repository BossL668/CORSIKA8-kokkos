#!/usr/bin/env python3
"""Restore the user-selected mountain 0.1 / 1e6 transport screening settings."""
import ast
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_100PeV_ne_sw_130cores_20260916')
REV=ROOT/'revisions/20260916_replace_initial_screen'
PARAMS={'--emcut-GeV':'.01','--hadcut-GeV':'10','--mucut-GeV':'.3','--emthin':'.1','--max-weight':'1000000'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arg(cmd,key):return cmd[cmd.index(key)+1]
def replace(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)

assert socket.gethostname()=='psrpku2025'
assert not (REV/'applied.json').exists()
progress=read(ROOT/'progress.json');assert progress['state']=='paused_for_screening_revision'
assert not progress['stages'] and not progress['completed'] and not progress['qualified']
lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
interrupted=read(REV/'interrupted.json')
for key in ['simulation_pid','controller_pid']:
    p=Path('/proc')/str(interrupted[key]);assert not p.exists() or (p/'stat').read_text().split(') ',1)[1][0]=='Z'

# Read the actual old mountain source without importing or running it.
reference=REV/'validate_thick_terrain_nutau.py';tree=ast.parse(reference.read_text())
legacy=None
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and any(k.arg=='thinning_fraction' and isinstance(k.value,ast.Constant) and k.value.value==.1 for k in node.keywords):
        legacy={k.arg:ast.literal_eval(k.value) for k in node.keywords}
assert legacy==dict(em_cut_eV=1e7,hadron_cut_eV=1e10,muon_cut_eV=3e8,thinning_fraction=.1,max_weight=1e6)
assert float(PARAMS['--emcut-GeV'])==legacy['em_cut_eV']/1e9
assert float(PARAMS['--hadcut-GeV'])==legacy['hadron_cut_eV']/1e9
assert float(PARAMS['--mucut-GeV'])==legacy['muon_cut_eV']/1e9
assert float(PARAMS['--emthin'])==legacy['thinning_fraction']
assert float(PARAMS['--max-weight'])==legacy['max_weight']
m=read(ROOT/'campaign.json');before=copy.deepcopy(m)
for row in m['files']+m['runtime_libraries']:assert sha(Path(row['path']))==row['sha256'],row['path']
for relative in ['campaign.json','progress.json','README_CN.md','code/run_psr.py']:
    p=REV/'before'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,p)
for case in m['cases']:
    if case['stage']!='coarse':continue
    case['tag']=case['tag'].replace('coarse_ne_sw_','coarse_fast01_ne_sw_',1)
    cmd=case['command']
    for key,value in {**PARAMS,'--output':str(ROOT/'runs'/case['tag']/'output')}.items():cmd[cmd.index(key)+1]=value
    case['emthin']=.1;case['historical_physics_options']=PARAMS.copy()
    case['screening_only']=True
    assert '--radio-algorithm' not in cmd and float(arg(cmd,'--transport-window-ns'))==100000
for a,b in zip(before['cases'],m['cases']):
    if a['stage']!='coarse':assert a==b,'Confirmation/radio changed'
    else:
        normalized=list(b['command'])
        for key in [*PARAMS,'--output']:normalized[normalized.index(key)+1]=arg(a['command'],key)
        assert normalized==a['command']
        assert all(float(arg(b['command'],k))==float(v) for k,v in PARAMS.items())

source=ROOT/'code/run_psr.py';text=source.read_text()
text=replace(text,
    '1. 旧参数快速筛选：EM 100 MeV、强子 10 GeV、μ/τ 0.3 GeV，`emthin=1e-2`、显式 Wmax=50000、输运窗 100 μs，不带射电。完整输运后核对实际谱系。',
    '1. 旧 mountain 高薄化输运筛选：EM 10 MeV、强子 10 GeV、μ/τ 0.3 GeV，`emthin=0.1`、显式 Wmax=1000000，不带射电。保留当前 100 μs 输运窗，完成后核对实际 CC→τ→空气中非 μ 衰变谱系。')
text=replace(text,
    '旧记录中 seed 158/946/3605 分别用时 15.2/22.3/44.3 分钟，当前耗时仍需实测。',
    '旧 mountain 的 0.1/1e6 粗筛来源及本次切换见 report/screening_fast01；首例实测耗时完成后记录。此前 15.2/22.3/44.3 分钟记录属于另一套 1e-2 参数，不能当成本轮计时。')
ast.parse(text,filename=str(source));source.write_text(text)
shutil.copy2(__file__,ROOT/'code/restore_mountain_fast01_psr.py')
m['screening_revision']=dict(applied_utc=now(),reference_source=str(reference),reference_sha256=sha(reference),
    options=PARAMS,transport_window_ns=100000,
    retained_current_physics=['100 PeV','NE-SW geometry','CTW CC+NC','TAUOLA','LPM','IGRF14 2027','SiO2 baseline','DEM boundary','130-core resident transport'],
    user_selection='旧 mountain 的 0.1 高薄化输运筛选',
    note='Restore old thinning, maximum weight and particle cuts only. Keep current 100 us coarse window; do not copy the old 12 us/zero-field/CC-only prototype. Confirmation and radio stay at 1e-3 and air-default cuts.')
for row in m['files']:row['sha256']=sha(Path(row['path']))
extra=ROOT/'code/restore_mountain_fast01_psr.py'
m['files'].append(dict(path=str(extra),sha256=sha(extra)))
save(ROOT/'campaign.json',m)
run=ROOT/'runs'/progress['current']
save(run/'status.json',dict(complete=False,interrupted_utc=interrupted['stopped_utc'],reason=interrupted['reason'],not_a_physics_rejection=True))
save(ROOT/'report/screening_fast01/preflight.json',dict(passed=True,checked_utc=now(),legacy_source_parameters=legacy,
    current_coarse_parameters=PARAMS,legacy_source_verified_by_ast=True,python_syntax_passed=True,
    confirmation_and_radio_case_dictionaries_unchanged=True,geometry_and_station_positions_unchanged=True,
    binaries_and_physics_libraries_unchanged=True,all_30_coarse_commands_verified=True,
    current_100us_window_retained=True))
report=ROOT/'report/screening_fast01'
for name in ['validate_thick_terrain_nutau.py','TERRAIN_THICK_NUTAU_RADIO_PILOT_CN.md','interrupted.json']:shutil.copy2(REV/name,report/name)
(report/'README_CN.md').write_text('已按用户确认改用旧 mountain 的 0.1 高薄化输运筛选。\n\n| 参数 | 停止的 1e-2 粗筛 | 当前快速粗筛 | 后续复核 / 完整射电 |\n|---|---:|---:|---:|\n| emthin | 1e-2 | 0.1 | 1e-3 |\n| Wmax | 50000 | 1000000 | 50000 |\n| EM 截止 | 100 MeV | 10 MeV | 0.5 MeV |\n| 强子截止 | 10 GeV | 10 GeV | 0.3 GeV |\n| μ/τ 截止 | 0.3 GeV | 0.3 GeV | 0.3 GeV |\n| 输运窗 | 100 μs | 100 μs | 500 μs |\n\n旧参数来自 `validate_thick_terrain_nutau.py` 的 `retry_high(late=True)`。这里只恢复其粗筛 thinning、权重与粒子截止，保留当前 100 PeV、新东北→西南几何、SiO₂、CC+NC、TAUOLA、LPM、IGRF14、DEM 边界与 130 个物理核。旧原型的 12 μs 窗口和零磁场不用于本轮。\n\n初筛不计算射电。它需完整运行声明的输运窗口并核对真实谱系，不强制 CC、τ 寿命或衰变道；只将满足岩石 CC→实际 τ→空气中非 μ 衰变及可见能量门限的事例列为候选。粗筛并不替代低薄化验证；1e-3 复核和最终全 80 站 CoREAS/ZHS 均重新验收 double bang。\n\n首个候选仍为 seed 14251，但重新运行新粗筛参数，使用独立输出目录。停止的 1e-2 运行及日志保留，不能算作完整事件；后续记录每例实际耗时，不沿用旧计时估算。\n\n[配置检查](preflight.json) · [旧源码](validate_thick_terrain_nutau.py) · [停止记录](interrupted.json)\n')
progress=dict(state='prepared_fast01_screen',completed=[],qualified=[],stages=[],current=m['cases'][0]['tag'],
    updated_utc=now(),interrupted_runs=[str(run)])
save(ROOT/'progress.json',progress)
spec=importlib.util.spec_from_file_location('driver',source);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
driver.CPU_LIST=m['cpu_list'];driver.write_readme(ROOT)
(ROOT/'STOP_AFTER_CURRENT').unlink()
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (ROOT/'driver.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(source),'--root',str(ROOT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'driver.pid').write_text(str(child.pid)+'\n')
receipt=dict(applied_utc=now(),driver_pid=child.pid,first_case=m['cases'][0]['tag'],parameters=PARAMS,
    partial_previous_run_retained=str(run))
save(REV/'applied.json',receipt);save(report/'applied.json',receipt)
print('STARTED',child.pid,m['cases'][0]['tag'],flush=True)
