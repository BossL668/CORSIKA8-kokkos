#!/usr/bin/env python3
"""Start one complete shower+radio run for each validated first-vertex candidate.

For the already-applied 20260916 campaign, direct_radio_after_vertex_psr.py
records the migration from the superseded separate-confirmation workflow.
"""
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

ROOT=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_100PeV_ne_sw_130cores_20260916')
REV=ROOT/'revisions/20260916_first_vertex_screen'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)


def main():
    assert socket.gethostname()=='psrpku2025'
    assert not (REV/'applied.json').exists()
    report=ROOT/'report/first_vertex'
    assert read(report/'regression.json')['passed']
    selection=read(report/'selection.json')
    seeds=[r['seed'] for r in selection['candidates']]
    assert seeds and selection['confirmed_doublebangs']==0
    interrupted=read(ROOT/'first_vertex/interrupted_coarse.json')
    for key in ['simulation_pid','controller_pid']:
        p=Path('/proc')/str(interrupted[key])
        assert not p.exists() or (p/'stat').read_text().split(') ',1)[1][0]=='Z'
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(ROOT/'campaign.json');before=copy.deepcopy(m)
    for row in m['files']+m['runtime_libraries']:
        assert sha(Path(row['path']))==row['sha256'],row['path']
    REV.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'campaign.json',REV/'campaign_before.json')
    shutil.copy2(ROOT/'progress.json',REV/'progress_before.json')
    source=ROOT/'code/run_psr.py';shutil.copy2(source,REV/'driver_before.py')
    text=source.read_text()
    old="for stage in ['coarse','confirm','radio']:"
    assert text.count(old)==1
    text=text.replace(old,"for stage in m['stage_order']:")
    # Replace the visible stage description along with the actual scheduling rule.
    oldline=next(line for line in text.splitlines() if "'1. 旧 mountain 高薄化" in line)
    text=text.replace(oldline,"             '1. 首个中微子反应快筛已完成：天然 CC+NC、实际地形、相同随机流和顶点生成器；顶点后立即停止，不计算 shower、不使用 thinning。只有岩石 CC 且 τ/强子系统各 ≥1 PeV 的种子进入下一步。[候选与核对记录](report/first_vertex/README_CN.md)。',")
    text=text.replace('2. 候选改为 `emthin=1e-3`、自动 Wmax=50000，不带射电，重新核对完整谱系。',
                      '2. 初筛候选直接以 `emthin=1e-3`、自动 Wmax=50000 完整运行 shower＋80 站 CoREAS/ZHS。')
    text=text.replace('3. 通过后保持 `emthin=1e-3`，计算 21CMA 全部 80 站 CoREAS/ZHS；最终再验收谱系、全部记录、能量账本和射电。若拓扑改变则继续下一候选。',
                      '3. 从同一次带射电输出验收 double bang、完整 profile、能量账本和射电；如不满足选图条件则处理下一候选。')
    text=text.replace('最多顺序考察 30 个候选','按首顶点筛出的候选顺序考察')
    text=text.replace('；只有粗筛采用上列旧截止','')
    text=text.replace('粗筛输运窗 100 μs，复核与最终射电为 500 μs','首顶点快筛不输运 shower；复核与最终射电的输运窗为 500 μs')
    text=text.replace('旧 mountain 的 0.1/1e6 粗筛来源及本次切换见 report/screening_fast01；首例实测耗时完成后记录。此前 15.2/22.3/44.3 分钟记录属于另一套 1e-2 参数，不能当成本轮计时。',
                      '原 0.1 全 shower 粗筛已停止并保留未完成记录；首顶点快筛的实测时间和与完整事件的核对见 report/first_vertex。')
    ast.parse(text);source.write_text(text)
    m['seeds']=seeds
    m['stage_order']=['radio']
    m['cases']=[c for c in m['cases'] if c['seed'] in seeds and c['stage'] in m['stage_order']]
    assert all(c==next(x for x in before['cases'] if x['tag']==c['tag']) for c in m['cases'])
    assert len(m['cases'])==len(seeds)
    m['screening_revision']=dict(applied_utc=now(),method='first_natural_neutrino_vertex',
                               report=str(report/'selection.json'),qualified_first_vertices=len(seeds),
                               confirmed_doublebangs=0,production_cases_unchanged=True)
    shutil.copy2(__file__,ROOT/'code/use_first_vertex_candidates_psr.py')
    for row in m['files']:row['sha256']=sha(Path(row['path']))
    p=ROOT/'code/use_first_vertex_candidates_psr.py';m['files'].append(dict(path=str(p),sha256=sha(p)))
    for p in [report/'selection.json',report/'regression.json',ROOT/'first_vertex/first_vertex',ROOT/'first_vertex/source/validation/terrain/energy_ladder/FirstNeutrinoVertex.cpp']:
        m['files'].append(dict(path=str(p),sha256=sha(p)))
    save(ROOT/'campaign.json',m)
    oldprogress=read(ROOT/'progress.json')
    p=dict(state='prepared_after_first_vertex_screen',completed=[],qualified=[],stages=[],
           first_vertex_candidate_count=len(seeds),current=next(c['tag'] for c in m['cases'] if c['seed']==seeds[0] and c['stage']=='radio'),
           updated_utc=now(),interrupted_runs=oldprogress.get('interrupted_runs',[])+[str(ROOT/'runs'/interrupted['previous_progress']['current'])])
    save(ROOT/'progress.json',p)
    spec=importlib.util.spec_from_file_location('driver',source);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.CPU_LIST=m['cpu_list'];driver.write_readme(ROOT)
    save(REV/'preflight.json',dict(passed=True,production_case_dictionaries_unchanged=True,
                                  first_vertex_regression_passed=True,selected_seeds=seeds,
                                  cpu_list=m['cpu_list'],threads=130,production_binary_unchanged=True))
    (ROOT/'STOP_AFTER_CURRENT').unlink()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    with (ROOT/'driver.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(source),'--root',str(ROOT)],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'driver.pid').write_text(str(child.pid)+'\n')
    receipt=dict(applied_utc=now(),driver_pid=child.pid,first_seed=seeds[0],stage_order=m['stage_order'])
    save(REV/'applied.json',receipt);save(report/'handoff.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
