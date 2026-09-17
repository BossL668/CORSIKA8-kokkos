#!/usr/bin/env python3
"""Freeze the audited NE-SW geometry and switch the 130-core campaign on PSR."""
import argparse
import ast
import copy
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import yaml

BASE=Path('/data/yhlu/CorsikaData/corsika_validation_results')
OLD=BASE/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916'
ROOT=BASE/'beta5_doublebang_100PeV_ne_sw_130cores_20260916'
GEO=BASE/'beta5_ne_sw_geometry_20260916'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def setarg(cmd,key,values):
    values=values if isinstance(values,list) else [values]
    i=cmd.index(key)+1;cmd[i:i+len(values)]=list(map(str,values))
def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def prepare():
    assert not (ROOT/'campaign.json').exists()
    checks=read(GEO/'report/checks.json');assert checks['passed'] and checks['valid_station_count_at_10m']==40
    g=read(GEO/'selected.json');old=read(OLD/'campaign.json')
    for row in old['files']+old['runtime_libraries']:assert sha(Path(row['path']))==row['sha256'],row['path']
    ROOT.mkdir(exist_ok=True)
    for d in ['bundle','code','scenes']:
        shutil.copytree(OLD/d,ROOT/d,ignore=shutil.ignore_patterns('__pycache__'))
    for d in ['runs','report']:(ROOT/d).mkdir(exist_ok=True)
    shutil.copytree(GEO/'report',ROOT/'report/geometry')
    shutil.copy2(OLD/'cpu_policy.json',ROOT/'cpu_policy.json')
    shutil.copy2(OLD/'report/cpu_selection.json',ROOT/'report/cpu_selection.json')
    shutil.copy2(OLD/'campaign.json',ROOT/'report/previous_campaign.json')
    shutil.copy2(__file__,ROOT/'code/start_ne_sw_campaign_psr.py')
    for p in (ROOT/'scenes').glob('*.yaml'):
        s=yaml.safe_load(p.read_text().replace(str(OLD),str(ROOT)))
        s['provenance']['campaign_note']='100 PeV NE-to-SW geometry: azimuth 225 deg, elevation -6 deg; 874.69 m rock and 7.062 km continuous air; 130 OpenMP cores.'
        p.write_text(yaml.safe_dump(s,sort_keys=False))
        assert len(s['radio']['observers'])==80
        previous=yaml.safe_load((OLD/'scenes'/p.name).read_text())
        assert s['radio']==previous['radio']
    p=ROOT/'code/run_psr.py';text=p.read_text()
    text=replace(text,'原入射位置/方向','东北→西南的新入射位置/方向（方位角 225°、向下 6°）')
    text=replace(text,'自然 CC→实际输运的 τ→电子/强子衰变','自然 CC→实际输运的 τ→空气中的电子/强子衰变')
    text=replace(text,"lines += ['', '[输入与完整命令]", "lines += ['', '入射 ENU (2207.376, 3905.012, 650.281) m；方向 (-0.703233, -0.703233, -0.104528)。注入点在入山前 100 m，穿岩 874.69 m，随后沿轴连续 7.062 km 为空气；在 N10 水平位置经过天线上方 250 m。几何预检的岩石测试源对 40/80 站有有效透射，这不是完整 shower 信号。[几何与光路图解](report/geometry/FIGURES_CN.md)。', '', '[输入与完整命令]")
    p.write_text(text)
    p=ROOT/'code/classify_psr.py';text=p.read_text()
    text=replace(text,"genealogy_connected=True, nonmuonic_second_shower=mode in ['electronic', 'hadronic'],", "genealogy_connected=True, nonmuonic_second_shower=mode in ['electronic', 'hadronic'],\n                tau_decay_in_air=decay['medium']=='air',")
    text=text.replace('to a nonmuonic decay;', 'to a nonmuonic decay in air;')
    p.write_text(text)
    for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def cross_section(E):
        x=math.log(math.log10(E)+1.826)
        return 10**(-17.31-6.406*x+1.431*x*x-17.91/x)+10**(-17.31-6.448*x+1.431*x*x-18.61/x)
    ratio=cross_section(3e8)/cross_section(1e8)
    candidates=[]
    for row in csv.DictReader((ROOT/'bundle/seed_depths_300PeV_reference.csv').open()):
        depth=float(row['depth_m'])*ratio;remaining=g['rock_length_m']-depth
        if 10<remaining<200:
            candidates.append(dict(seed=int(row['seed']),approximate_depth_m=depth,approximate_remaining_rock_m=remaining))
    candidates.sort(key=lambda x:(abs(x['approximate_remaining_rock_m']-60),x['seed']))
    assert len(candidates)>=30,len(candidates)
    seeds=[x['seed'] for x in candidates[:30]]
    save(ROOT/'bundle/seed_selection.json',dict(seeds=seeds,approximate_candidates=candidates,
        scope='Existing natural Philox exponential-depth candidates rescaled with the same CTW cross sections to 100 PeV; 10-200 m of the new rock chord remaining, ranked near 60 m. CC, decay, air medium and complete genealogy must be rechecked in every stage; no forced interaction or decay; not an unbiased event-rate sample.'))
    save(ROOT/'bundle/selected_geometry.json',g)
    cases=[]
    for seed in seeds:
        for stage in ['coarse','confirm','radio']:
            before=next(x for x in old['cases'] if x['stage']==stage)
            case=copy.deepcopy(before);tag=f'{stage}_ne_sw_SiO2_100PeV_seed{seed}'
            cmd=[x.replace(str(OLD),str(ROOT)) if isinstance(x,str) else x for x in before['command']]
            for k,val in {'--seed':seed,'--output':str(ROOT/'runs'/tag/'output'),'--position-m':g['position_m'],'--direction':g['direction']}.items():setarg(cmd,k,val)
            # Normalize only authorized geometry, seed and path changes to verify all physics/scheduler arguments.
            normalized=[x.replace(str(ROOT),str(OLD)) if isinstance(x,str) else x for x in cmd]
            for k,n in [('--seed',1),('--output',1),('--position-m',3),('--direction',3)]:
                i=normalized.index(k)+1;j=before['command'].index(k)+1
                normalized[i:i+n]=before['command'][j:j+n]
            assert normalized==before['command']
            case.update(tag=tag,seed=seed,command=cmd,geometry='NE-to-SW 225 deg, elevation -6 deg')
            cases.append(case)
    m=copy.deepcopy(old);m.update(created_utc=now(),cases=cases,seeds=seeds,
        cpu_policy_file=str(ROOT/'cpu_policy.json'),previous_geometry_campaign=str(OLD),
        geometry=g,geometry_preflight=checks,seed_selection_file=str(ROOT/'bundle/seed_selection.json'),
        selection_revision='Require an actual nonmuonic tau decay in air; all previous genealogy, energy, separation and completion cuts retained.')
    m['files']=[dict(path=str(p),sha256=sha(p)) for d in ['bundle','code','scenes'] for p in sorted((ROOT/d).iterdir()) if p.is_file()]
    mesh=Path(yaml.safe_load((ROOT/'scenes/screen.yaml').read_text())['geometry']['mesh_path'])
    m['files'].append(dict(path=str(mesh),sha256=sha(mesh)))
    save(ROOT/'campaign.json',m)
    save(ROOT/'progress.json',dict(state='prepared_geometry',completed=[],qualified=[],stages=[],current=cases[0]['tag'],updated_utc=now()))
    # Exercise the added selection on a recorded complete natural CC -> tau -> air decay.
    reference=OLD/'runs/coarse_legacycuts_SiO2_100PeV_seed3605'
    summary=yaml.safe_load((reference/'output/terrain_run.yaml').read_text())
    classifier=module('ne_sw_classifier',ROOT/'code/classify_psr.py')
    regression=[]
    for medium,expected in [('air',True),('rock',False)]:
        folder=ROOT/'report'/('selection_check_'+medium);folder.mkdir()
        for name in ['tau_tracks.json','vertex_daughter_tracks.json','command.json']:shutil.copy2(reference/name,folder/name)
        sample=copy.deepcopy(summary)
        for d in sample['tau']['decays']:d['medium']=medium
        result=classifier.classify(folder,sample)
        assert result['qualified_doublebang']==expected
        regression.append(dict(medium=medium,expected=expected,actual=result['qualified_doublebang'],
            scope='Recorded event reused for selection regression; rock-labelled variant is synthetic, not a simulated event.'))
    save(ROOT/'report/preflight.json',dict(passed=True,checked_utc=now(),geometry=checks,
        python_syntax_passed=True,commands_changed_only_in_geometry_seed_and_paths=True,
        binaries_unchanged=True,station_coordinates_unchanged=True,radio_configuration_unchanged=True,
        selected_seed_count=len(seeds),selection_regression=regression))
    driver=module('ne_sw_driver',ROOT/'code/run_psr.py');driver.CPU_LIST=m['cpu_list'];driver.write_readme(ROOT)
    print('PREPARED',ROOT,'first seed',seeds[0],flush=True)

def stop(pid,required,group=False):
    path=Path('/proc')/str(pid)
    if not path.exists():return
    cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert required in cmd,(pid,cmd)
    if group:assert os.getpgid(pid)==pid
    send=(lambda sig:os.killpg(pid,sig)) if group else (lambda sig:os.kill(pid,sig))
    send(signal.SIGTERM)
    for _ in range(50):
        if not path.exists() or (path/'stat').read_text().split(') ',1)[1][0]=='Z':return
        time.sleep(.1)
    send(signal.SIGKILL)
    for _ in range(50):
        if not path.exists() or (path/'stat').read_text().split(') ',1)[1][0]=='Z':return
        time.sleep(.1)
    raise RuntimeError('Process did not stop')

def start():
    assert read(ROOT/'report/preflight.json')['passed'] and not (ROOT/'launch.json').exists()
    p=read(OLD/'progress.json');assert p['state']=='running'
    run=OLD/'runs'/p['current'];controller=int((OLD/'driver.pid').read_text());simulation=int((run/'pid').read_text())
    record=dict(stopped_utc=now(),reason='User requested NE-to-SW geometry pointing toward 21CMA with a clear air section after the mountain.',
        previous_progress=p,simulation_pid=simulation,controller_pid=controller,complete=False,
        partial_data_retained=True,partial_directory=str(run),next_campaign=str(ROOT))
    save(ROOT/'report/previous_geometry_interrupted.json',record)
    (OLD/'STOP_AFTER_CURRENT').write_text('Superseded by the user-requested NE-SW geometry.\n')
    stop(controller,str(OLD/'code/run_psr.py'))
    stop(simulation,str(run/'output'),group=True)
    lock=(OLD/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(run/'INTERRUPTED.json',record)
    save(run/'status.json',dict(complete=False,stop_reason=record['reason'],interrupted_utc=record['stopped_utc'],not_a_physics_rejection=True))
    p.update(state='superseded',updated_utc=now(),next_campaign=str(ROOT),interruption=record['reason'])
    save(OLD/'progress.json',p)
    driver=module('old_driver',OLD/'code/run_psr.py');driver.CPU_LIST=read(OLD/'campaign.json')['cpu_list'];driver.write_readme(OLD)
    with (OLD/'README_CN.md').open('a') as f:f.write('\n已按用户要求切换为东北→西南的新几何。当前低薄化复核被中止，不能作为完整事件；此前两例已完成粗筛和本例未完成原始数据均保留。新任务：`'+str(ROOT)+'`。\n')
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    with (ROOT/'driver.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'code/run_psr.py'),'--root',str(ROOT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'driver.pid').write_text(str(child.pid)+'\n')
    save(ROOT/'launch.json',dict(pid=child.pid,started_utc=now(),previous_simulation_stopped=simulation,
        same_130_cpu_list=read(ROOT/'campaign.json')['cpu_list']))
    print('STARTED',child.pid,flush=True)

if __name__=='__main__':
    assert socket.gethostname()=='psrpku2025'
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');args=p.parse_args()
    if not (ROOT/'report/preflight.json').exists():prepare()
    if args.start:start()
