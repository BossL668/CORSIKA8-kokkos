#!/usr/bin/env python3
"""Sequential complete 1 PeV pilots, frozen inputs, timings and automatic audit."""
import argparse
import csv
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
import traceback
import yaml

BASE=Path('/data/yhlu/CorsikaData/corsika_validation_results')
PREVIOUS=BASE/'beta5_doublebang_21CMA80_dem_openmp256_20260914'
ACCEPTED=BASE/'beta5_interface_batch_performance_20260915'
GIB=1024**3
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def setarg(cmd,key,value):
    if key in cmd:cmd[cmd.index(key)+1]=str(value)
    else:cmd.extend([key,str(value)])
def available():return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
def legacy(root):
    spec=importlib.util.spec_from_file_location('legacy_monitor',root/'code/legacy_driver.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
def readme(root):
    p=json.loads((root/'progress.json').read_text());m=json.loads((root/'campaign.json').read_text())
    lines=['# 1 PeV double bang：逐档测试起点','',
        f'状态：`{p["state"]}`；完整验收 {len(p["completed"])} 例。更新时间：{p["updated_utc"]}。',
        '当前：`'+p.get('current','尚未开始')+'`。','',
        'ντ 1 PeV；原入射位置和方向、SiO₂、真实 DEM 侧边界、IGRF14（2027）。PSR OpenMP 256 个物理核，常驻 EM 队列；全部 80 站同时计算 CoREAS/ZHS。本地空气任务不受影响。','',
        'EM 动能截止 0.5 MeV，强子/μ/τ 截止 0.3 GeV，emthin=1e-6；Wmax=0.5，单位权重粒子实际不触发 thinning。LPM 和其余物理沿用已验收版本。CPU 暂存/波前 4096，记录缓冲 65536，射电源缓冲 8192。','',
        '自然 CC/NC 和 TAUOLA；不强制顶点、衰变道或 τ 寿命。先用廉价的自然相互作用深度预筛选择接近首段山体出口的候选，优先检查是否可形成第二个空气 shower；这不是无偏事件率样本。实际顶点及谱系以完整模拟为准。最多顺序测试 8 个候选，得到首个通过选图条件的 double bang 后停止；本阶段不自动启动更高能量。','',
        '双顶点拓扑与选图条件分别报告。1 PeV 选图条件：两个顶点各有 ≥10 TeV 可见 shower 次级且实际输运，顶点相距 ≥10 m；不等于接收站已分辨双脉冲。旧的“各 ≥1 PeV、距离 ≥100 m”门槛单独保留为参考，不用于拒绝本档事件。','',
        '保留完整 tracks、deposits、逃逸末态、能量账本、ντ/τ 顶点谱系、80 站原始矩和波形。输运窗 500 μs；射电窗 2048 μs、256 MHz。所有截止和逃逸能量分别记账。记录安全上限 100 亿，达到上限会明确判为未完成，绝不静默丢行。','',
        '进程总耗时、shower 阶段、输运循环、射电尾批/下载/重建/输出、验收绘图及无损归档分别计时；保留资源采样和已写出步数。未完成时只显示已耗时，不把它当作完整事件耗时。','',
        '80 站沿用已核对经纬度和 DEM+1 m 的模型高度；并非实测天线相位中心。完成后自动全量验收、绘制英文图并验证无损 gzip 归档。所有计算在 PSR，本地仅同步报告。','']
    if p.get('latest'):
        x=p['latest'];lines += [f'当前已运行 {x["elapsed_s"]/3600:.2f} 小时；已读取的记录前缀 {x["recorded_steps_prefix"]:,} 步；RSS {x["rss_GiB"]:.1f} GiB。','']
    for tag in p['completed']:lines.append(f'- [{tag}](report/{tag}/README_CN.md) · [耗时](report/{tag}/timing.json)')
    if p.get('error'):lines += ['', '停止原因：'+p['error']]
    lines += ['', '[输入与命令](campaign.json) · [当前状态](progress.json)','', 'PSR 目录：`'+str(root)+'`。']
    (root/'README_CN.md').write_text('\n'.join(lines)+'\n')

def prepare(root):
    assert not (root/'campaign.json').exists()
    accepted=json.loads((ACCEPTED/'report/acceptance.json').read_text());assert accepted['passed']
    prior=json.loads((PREVIOUS/'campaign.json').read_text())
    for name in ['bundle','scenes','runs','report']:(root/name).mkdir(exist_ok=False)
    for name in ['silica_SiO2.yaml','silica_SiO2_expected.yaml','stations_80_audit.json','stations_80_audit.csv','observers.yaml','terrain_grid.npz','terrain_manifest.yaml','LPM_to_1EeV.json']:
        shutil.copy2(PREVIOUS/'bundle'/name,root/'bundle'/name)
    binary=root/'bundle/c8_terrain_cascade';shutil.copy2(root/'build-openmp/c8_terrain_cascade',binary)
    # Only two CLI admission ceilings differ from the frozen, accepted program.
    before=(ACCEPTED/'source/applications/c8_terrain_cascade.cpp').read_text()
    after=(root/'source/applications/c8_terrain_cascade.cpp').read_text()
    assert before.count('CLI::Range(1LL, 1000000000LL)')==2
    assert after==before.replace('CLI::Range(1LL, 1000000000LL)','CLI::Range(1LL, 10000000000LL)')
    for name in ['libCORSIKA8InterfaceEm.a','libCORSIKA8InterfaceRadio.a']:
        assert digest(root/'build-openmp'/name)==digest(ACCEPTED/'build-openmp'/name)
    scene=yaml.safe_load((ACCEPTED/'scenes/production_reference.yaml').read_text())
    scene['geometry']['material_file']=str(root/'bundle/silica_SiO2.yaml')
    scene['radio'].update(enabled=True,samples=524288,memory_MiB=81920)
    scene['provenance'].update(campaign_note='1 PeV energy-ladder start, air-default cuts and emthin=1e-6; natural near-exit seed candidates; actual genealogy required.',station_audit=str(root/'bundle/stations_80_audit.json'))
    assert len(scene['radio']['observers'])==80
    scene_file=root/'scenes/silica_SiO2_21CMA80.yaml';scene_file.write_text(yaml.safe_dump(scene,sort_keys=False))
    seeds=list(csv.DictReader((root/'preflight/seeds_1PeV.csv').open()))
    candidates=[row for row in seeds if 1287.4258149368304<float(row['depth_m'])<1382.4258149368304][:8]
    assert len(candidates)==8
    cases=[]
    for row in candidates:
        seed=int(row['seed']);tag=f'openmp_SiO2_21CMA80_1PeV_seed{seed}'
        cmd=list(prior['cases'][0]['command']);cmd[3]=str(binary)
        for key,value in {'--scene':scene_file,'--output':root/'runs'/tag/'output','--energy-GeV':1000000,'--seed':seed,
            '--emthin':1e-6,'--max-weight':.5,'--track-row-limit':10000000000,'--transport-step-limit':10000000000,
            '--resident-wavefront-capacity':4096,'--resident-record-capacity':65536,'--radio-batch':8192,'--device-output-threads':32}.items():setarg(cmd,key,value)
        cases.append(dict(tag=tag,energy_GeV=1000000,seed=seed,command=cmd,prefilter=row))
    physical=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    core_ids=[tuple(map(int,x.split(',')))[1:] for x in physical.splitlines() if not x.startswith('#') and int(x.split(',')[0])<256]
    assert len(core_ids)==256 and len(set(core_ids))==256
    for lib in prior['runtime_libraries']:assert digest(Path(lib['path']))==lib['sha256']
    save(root/'campaign.json',dict(created_utc=now(),cases=cases,energy_GeV=1000000,binary_sha256=digest(binary),
        accepted_reference=str(ACCEPTED),changes_from_reference='Only CLI row/step ceilings 1e9 -> 1e10; interface libraries identical.',
        cpu_data_directory=prior['cpu_data_directory'],runtime_libraries=prior['runtime_libraries'],
        physical_cores=256,observer_count=80,target_doublebangs=1,maximum_candidates=8,higher_energies_scheduled=False,
        files=[dict(path=str(p),sha256=digest(p)) for folder in ['bundle','code','scenes','preflight'] for p in sorted((root/folder).iterdir()) if p.is_file()],
        memory=dict(minimum_available_before_start_GiB=190,rss_guard_GiB=220,radio_limit_GiB=80,combined_limit_GiB=96),
        seed_selection='Ascending seeds among approximate first-chord depths with 5-100 m remaining; no forced interactions/decays; not an event-rate sample.'))
    save(root/'progress.json',dict(state='prepared',completed=[],qualified=[],current=cases[0]['tag'],updated_utc=now()))
    readme(root)

def run_case(root,case,manifest,progress,helper):
    assert available()>190*GIB,'Insufficient available RAM before event'
    assert shutil.disk_usage(root).free>180*GIB,'Insufficient disk before event'
    f=root/'runs'/case['tag'];f.mkdir(exist_ok=False);(f/'work').mkdir()
    cmd=case['command'];save(f/'command.json',cmd)
    env=dict(os.environ,CORSIKA_DATA=manifest['cpu_data_directory'],OMP_NUM_THREADS='256',OMP_PROC_BIND='spread',OMP_PLACES='threads',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for key in ['KOKKOS_TOOLS_LIBS','C8_INTERFACE_TRACE_FILE']:env.pop(key,None)
    tail=helper.TrackTail(f/'output/terrain/tracks.csv.gz');peak=0;reason=None;binding={'verified':False}
    with (f/'run.log').open('x') as log,(f/'resources.jsonl').open('x') as resources:
        start=time.monotonic();started_utc=now()
        child=subprocess.Popen(cmd,cwd=f/'work',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (f/'pid').write_text(str(child.pid));save(f/'STARTED.json',dict(pid=child.pid,started_utc=started_utc,binary_sha256=manifest['binary_sha256']))
        try:
            while child.poll() is None:
                try:status=Path(f'/proc/{child.pid}/status').read_text()
                except FileNotFoundError:break
                rss=int(next((s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:')),'0'))*1024;peak=max(peak,rss)
                if not binding['verified']:binding=helper.affinity(child.pid);save(f/'affinity.json',binding)
                sample=dict(elapsed_s=time.monotonic()-start,rss_GiB=rss/GIB,available_GiB=available()/GIB,
                    free_disk_GiB=shutil.disk_usage(root).free/GIB,recorded_steps_prefix=tail.sample(),pid=child.pid)
                resources.write(json.dumps(sample)+'\n');resources.flush()
                progress.update(state='running',current=case['tag'],latest=sample,updated_utc=now());save(root/'progress.json',progress);readme(root)
                if rss>220*GIB:reason='RSS exceeded 220 GiB'
                if sample['available_GiB']<16:reason='Available RAM below 16 GiB'
                if sample['free_disk_GiB']<30:reason='Available disk below 30 GiB'
                if reason:break
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        elapsed=time.monotonic()-start
    path=f/'output/terrain_run.yaml';s=yaml.safe_load(path.read_text()) if path.exists() else {}
    complete=child.returncode==0 and not reason and bool(s.get('complete'))
    timing=dict(complete=complete,started_utc=started_utc,finished_utc=now(),process_wall_seconds=elapsed,
        shower_seconds=s.get('shower_seconds'),execution_timing=s.get('execution_timing'),peak_rss_GiB=peak/GIB,
        wall_scope='Process launch through exit, including initialization and all simulation/radio file output; excludes later audit and archive.')
    save(f/'timing.json',timing);save(f/'status.json',dict(complete=complete,returncode=child.returncode,stop_reason=reason,wall_s=elapsed,
        peak_rss_GiB=peak/GIB,error=s.get('error'),binary_sha256=digest(Path(cmd[3]))))
    assert complete,'Incomplete event: '+str(reason or s.get('error') or child.returncode)
    assert binding['verified'],'256 physical-core binding not verified'
    return f,timing

def execute(root):
    lock=(root/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((root/'campaign.json').read_text());p=json.loads((root/'progress.json').read_text());helper=legacy(root)
    for item in m['files']+m['runtime_libraries']:assert digest(Path(item['path']))==item['sha256'],item['path']
    try:
        for case in m['cases']:
            if case['tag'] in p['completed']:continue
            if p['qualified'] or (root/'STOP_AFTER_CURRENT').exists():break
            print('START',case['tag'],now(),flush=True);pipeline=time.monotonic()
            f,timing=run_case(root,case,m,p,helper)
            p.update(state='auditing',updated_utc=now());save(root/'progress.json',p);readme(root)
            start=time.monotonic()
            with (f/'audit.log').open('x') as log:subprocess.run([sys.executable,str(root/'code/audit_psr.py'),'--root',str(root),'--folder',str(f)],stdout=log,stderr=subprocess.STDOUT,check=True)
            timing['validation_and_figures_seconds']=time.monotonic()-start
            out=root/'report'/case['tag'];verdict=json.loads((out/'acceptance.json').read_text());assert verdict['passed']
            p['completed'].append(case['tag'])
            if verdict['classification']['qualified_doublebang']:p['qualified'].append(case['tag'])
            save(out/'timing.json',timing);save(f/'timing.json',timing)
            report=out/'README_CN.md'
            report.write_text(report.read_text()+f'\n完整模拟进程耗时 {timing["process_wall_seconds"]:.3f} s；shower 阶段 {timing["shower_seconds"]:.3f} s。[全部阶段耗时](timing.json)。\n')
            p.update(state='archiving',updated_utc=now());save(root/'progress.json',p);readme(root)
            start=time.monotonic()
            with (f/'archive.log').open('x') as log:subprocess.run([sys.executable,str(root/'code/archive_psr.py'),'--folder',str(f)],stdout=log,stderr=subprocess.STDOUT,check=True)
            timing['lossless_archive_seconds']=time.monotonic()-start;timing['total_pipeline_seconds']=time.monotonic()-pipeline
            save(out/'timing.json',timing);save(f/'timing.json',timing)
            print('ACCEPTED',case['tag'],timing,flush=True)
        p.update(state='complete' if p['qualified'] else 'stopped_after_current' if (root/'STOP_AFTER_CURRENT').exists() else 'candidate_limit_reached',updated_utc=now())
    except BaseException as e:
        p.update(state='failed',error=str(e),traceback=traceback.format_exc(),updated_utc=now());raise
    finally:save(root/'progress.json',p);readme(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    assert socket.gethostname()=='psrpku2025'
    prepare(args.root) if args.prepare else execute(args.root)
