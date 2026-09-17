#!/usr/bin/env python3
"""100 PeV: coarse genealogy selection, target-thinning check, one full radio event.

Preparation and all numerical work run on PSR. No production physics is changed.
CPU sharing is disabled unless explicitly authorized in the campaign policy.
"""
import argparse
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
import socket
import subprocess
import sys
import time
import traceback

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
PREVIOUS = BASE/'beta5_doublebang_1PeV_thin1e4_queue4M_20260915'
SEED_SOURCE = BASE/'beta5_SiO2_high_energy_aircuts_thin1e-5_20260914/build/seeds_300000000.csv'
CPU_LIST = '382-511'
CPUS = set(range(382, 512))
THREADS = 130
GIB = 1024**3


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cpu_set(value):
    out = set()
    for part in value.split(','):
        bounds = list(map(int, part.split('-')))
        out.update(range(bounds[0], bounds[-1]+1))
    return out


def topology():
    lines = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True)
    rows = [tuple(map(int, x.split(','))) for x in lines.splitlines() if not x.startswith('#')]
    mapping = {cpu:(core, socket) for cpu, core, socket in rows}
    assert CPUS <= mapping.keys()
    cores = {mapping[c] for c in CPUS}
    assert len(cores) == THREADS
    siblings = {cpu for cpu, pair in mapping.items() if pair in cores}
    return mapping, siblings


def conflicts():
    """Inspect external compute jobs, including sibling logical CPUs."""
    _, siblings = topology()
    result = []
    lines = subprocess.check_output(['ps', '-eo', 'pid,pcpu,comm,args', '--no-headers'], text=True)
    for line in lines.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, usage, name, command = parts
        if int(pid) == os.getpid() or float(usage) < 10:
            continue
        # Inspect compute processes, rather than the short-lived SSH/status commands.
        if not (name.startswith(('python', 'c8_', 'java', 'matlab', 'R', 'mpirun')) or float(usage) > 200):
            continue
        try:
            allowed = os.sched_getaffinity(int(pid))
            if allowed & siblings:
                result.append(dict(pid=int(pid), cpu_percent=float(usage), name=name,
                                   overlapping_logical_cpus=sorted(allowed & siblings), command=command))
        except ProcessLookupError:
            pass
    return result


def affinity(pid):
    singleton = set()
    tasks = list(Path(f'/proc/{pid}/task').glob('*/status'))
    for path in tasks:
        try:
            text = path.read_text()
            allowed = cpu_set(next(x.split(':',1)[1].strip() for x in text.splitlines() if x.startswith('Cpus_allowed_list:')))
        except (FileNotFoundError, StopIteration):
            continue
        if not allowed <= CPUS:
            return dict(verified=False, reason='Thread outside '+CPU_LIST)
        if len(allowed) == 1:
            singleton |= allowed
    return dict(verified=singleton==CPUS, threads_inspected=len(tasks),
                singleton_physical_cores=len(singleton), allowed_cpu_list=CPU_LIST)


def write_readme(root):
    p = read(root/'progress.json')
    policy = read(root/'cpu_policy.json')
    sharing = ('用户已要求利用空闲算力开始计算；按实测负载选择 130 个不同物理核上的逻辑 CPU。现有任务仍可在重叠范围迁移，这不是独占预留。'
               if policy['allow_shared'] else '检查现有计算任务是否与这些物理核重叠，重叠时等待。')
    lines = ['# 100 PeV SiO₂：筛选后完成一个 double bang', '',
             f'状态：`{p["state"]}`；更新时间：{p["updated_utc"]}。',
             f'当前：`{p.get("current", "尚未启动计算")}`。', '',
             '目标：自然 CC→实际输运的 τ→电子/强子衰变，两顶点各注入至少 1 PeV 可见 shower 次级、间隔至少 100 m。得到一个完整且通过该条件的 80 站事件后停止；不把谱系双顶点直接称为可分辨射电双脉冲。', '',
             '1. `emthin=1e-2`、自动 Wmax=500000，不带射电，完成真实输运和谱系筛选。',
             '2. 候选改为 `emthin=1e-3`、自动 Wmax=50000，不带射电，重新核对完整谱系。',
             '3. 通过后保持 `emthin=1e-3`，计算 21CMA 全部 80 站 CoREAS/ZHS；最终再验收谱系、全部记录、能量账本和射电。若拓扑改变则继续下一候选。', '',
             '改变 thinning 或执行条件可能改变随机历史；不会只凭粗筛种子认定最终事件是 double bang。最多顺序考察 30 个候选，达到资源保护条件时停止并保留原因。完整筛选结果均保留，筛选不是无偏事件率样本。', '',
             f'PSR OpenMP {THREADS} 线程，CPU `{CPU_LIST}`；这 130 个逻辑 CPU 分别属于 130 个物理核，另有对应超线程兄弟。'+sharing+'其他任务的亲和性保持原状。', '',
             '沿用 SiO₂（密度 2650 kg/m³、n=√5、场衰减长度 100 m）、原入射位置/方向、DEM 覆盖边界、IGRF14（2027）、LPM，以及 EM 0.5 MeV / 强子和 μ、τ 0.3 GeV 截止。常驻粒子容量 4,194,304，波前 4096，记录缓冲 65536，射电源批次 8192。', '',
             '保留完整轨迹、沉积、出界末态和每阶段耗时。输运窗 500 μs；完整射电仍用 2048 μs、256 MHz 采样，图中分析 50–100 MHz；并不覆盖完整 50–300 MHz。80 站位置沿用已核对坐标和 DEM+1 m 模型高度。', '',
             '计算、验收和绘图均在 PSR；本地仅维护脚本并同步结果。生产二进制与已经完成的 1 PeV 事件一致。', '',
             f'已完成阶段 {len(p.get("stages", []))} 个；完整射电验收 {len(p["completed"])} 例；合格 double bang {len(p["qualified"])} 例。']
    if p.get('latest'):
        q=p['latest'];lines += ['',f'当前阶段已耗时 {q["elapsed_s"]/3600:.3f} h，轨迹记录前缀 {q["recorded_steps_prefix"]:,}，RSS {q["rss_GiB"]:.2f} GiB；不是完成百分比。']
    for stage in p.get('stages', []):
        lines.append(f'- [{stage["tag"]}](report/{stage["tag"]}/README_CN.md)：double bang={stage["qualified"]}，进程 {stage["wall_s"]:.2f} s。')
    if p.get('error'):
        lines += ['', '停止原因：'+p['error']]
    lines += ['', '[输入与完整命令](campaign.json) · [状态](progress.json) · [CPU 冲突检查](report/cpu_conflicts.json)', '', 'PSR：`'+str(root)+'`。']
    (root/'README_CN.md').write_text('\n'.join(lines)+'\n')


def prepare(root):
    import yaml
    assert not (root/'campaign.json').exists()
    prior=read(PREVIOUS/'campaign.json')
    old=read(PREVIOUS/'report/openmp_SiO2_21CMA80_1PeV_seed22309/acceptance.json')
    assert old['passed'] and all(old['gates'].values())
    for d in ['bundle','scenes','runs','report']:
        (root/d).mkdir(exist_ok=False)
    shutil.copytree(PREVIOUS/'bundle',root/'bundle',dirs_exist_ok=True)
    shutil.copy2(SEED_SOURCE,root/'bundle/seed_depths_300PeV_reference.csv')
    for name in ['legacy_driver.py','legacy_classifier.py','archive_psr.py']:
        shutil.copy2(PREVIOUS/'code'/name,root/'code'/name)
    runner=(PREVIOUS/'code/reference_runner.py').read_text()
    assert runner.count("OMP_NUM_THREADS='256'")==1
    runner=runner.replace("OMP_NUM_THREADS='256'", "OMP_NUM_THREADS='130'")
    runner=runner.replace("available()>190*GIB", "available()>(190 if case['stage']=='radio' else 32)*GIB")
    runner=runner.replace("shutil.disk_usage(root).free>180*GIB", "shutil.disk_usage(root).free>(180 if case['stage']=='radio' else 30)*GIB")
    runner=runner.replace("try:child.wait(timeout=10)\n                except", "try:child.wait(timeout=1 if not binding['verified'] else 10)\n                except")
    (root/'code/event_runner.py').write_text(runner)
    classifier=(PREVIOUS/'code/legacy_classifier.py').read_text()
    assert 'new_seed = seed not in [158, 946, 3605]' in classifier
    classifier=classifier.replace('new_seed = seed not in [158, 946, 3605]', 'new_seed = True  # Historical seeds are eligible in this pilot.')
    classifier=classifier.replace('Require three qualifying NEW seeds per energy; original seeds 158/946/3605 are retained as reference events and never fill the new-seed quota.', 'One complete 100 PeV event at emthin=1e-3; all candidate seeds are eligible.')
    (root/'code/classify_psr.py').write_text(classifier)
    audit=(PREVIOUS/'code/audit_psr.py').read_text()
    for before,after in [("s['emthin']==1e-4", "s['emthin']==1e-3"),
                         (".5e-4*s['energy_GeV']", ".5e-3*s['energy_GeV']"),
                         ("a['execution_concurrency']==256", "a['execution_concurrency']==130")]:
        assert audit.count(before)==1
        audit=audit.replace(before,after)
    (root/'code/audit_psr.py').write_text(audit)
    # The same complete CSV/profile scanner also serves radio-disabled screening.
    screen=audit.replace("r=summary['radio_result'];terrain=", "r=summary.get('radio_result',{});terrain=")
    old_block="    for algorithm in ['CoREAS','ZHS']:\n        assert r[algorithm]['device_tracks']+r[algorithm]['cpu_tracks']==charged\n        assert r[algorithm]['track_observer_pairs']==charged*80"
    assert screen.count(old_block)==1
    screen=screen.replace(old_block,"    if all(key in r for key in ['CoREAS','ZHS']):\n"+'\n'.join('    '+line for line in old_block.splitlines()))
    (root/'code/screen_audit_helpers.py').write_text(screen)
    scene=yaml.safe_load((PREVIOUS/'scenes/silica_SiO2_21CMA80.yaml').read_text())
    scene['geometry']['material_file']=str(root/'bundle/silica_SiO2.yaml')
    scene['provenance'].update(station_audit=str(root/'bundle/stations_80_audit.json'),campaign_note='100 PeV coarse 1e-2 selection, 1e-3 transport confirmation, 1e-3 80-station radio; OpenMP 130 threads.')
    for enabled,name in [(False,'screen'),(True,'radio')]:
        scene['radio']['enabled']=enabled
        scene['provenance']['radio_disabled']=not enabled
        (root/'scenes'/(name+'.yaml')).write_text(yaml.safe_dump(scene,sort_keys=False))
    def cross_section(energy):
        x=math.log(math.log10(energy)+1.826)
        return 10**(-17.31-6.406*x+1.431*x*x-17.91/x)+10**(-17.31-6.448*x+1.431*x*x-18.61/x)
    ratio=cross_section(3e8)/cross_section(1e8)
    rows=list(csv.DictReader(SEED_SOURCE.open()))
    possible=[dict(seed=int(x['seed']),approx_depth_m=float(x['depth_m'])*ratio) for x in rows if 0<float(x['depth_m'])*ratio<1387.4258149368304]
    possible.sort(key=lambda x:(abs(x['approx_depth_m']-1250),x['seed']))
    seeds=[946,3605,158]
    for row in possible:
        if row['seed'] not in seeds:
            seeds.append(row['seed'])
        if len(seeds)==30:
            break
    assert len(seeds)==30
    save(root/'bundle/seed_selection.json',dict(seeds=seeds,approximate_candidates=possible,scope='Historical candidates first, then approximate depth prefilter rescaled with the same CTW rates; does not predict CC or decay and does not force either.'))
    cases=[]
    for seed in seeds:
        for stage,thin in [('coarse',1e-2),('confirm',1e-3),('radio',1e-3)]:
            tag=f'{stage}_SiO2_100PeV_seed{seed}'
            cmd=list(prior['cases'][0]['command']);cmd[2]=CPU_LIST;cmd[3]=str(root/'bundle/c8_terrain_cascade')
            for key,value in {'--scene':root/'scenes'/('radio.yaml' if stage=='radio' else 'screen.yaml'),
                              '--output':root/'runs'/tag/'output','--energy-GeV':100000000,
                              '--seed':seed,'--emthin':thin,'--threads':THREADS}.items():
                cmd[cmd.index(key)+1]=str(value)
            assert '--max-weight' not in cmd
            cases.append(dict(tag=tag,seed=seed,stage=stage,energy_GeV=100000000,emthin=thin,command=cmd))
    mapping,siblings=topology()
    save(root/'bundle/cpu_topology.json',dict(cpu_list=CPU_LIST,threads=THREADS,physical_cores=[mapping[c] for c in sorted(CPUS)],all_siblings=sorted(siblings)))
    save(root/'cpu_policy.json',dict(allow_shared=False,reason='User requested no conflict with other jobs; wait for the selected physical cores.'))
    manifest=dict(created_utc=now(),energy_GeV=100000000,cases=cases,seeds=seeds,target_doublebangs=1,
                  cpu_list=CPU_LIST,threads=THREADS,physical_cores=THREADS,observer_count=80,
                  binary_sha256=digest(root/'bundle/c8_terrain_cascade'),cpu_data_directory=prior['cpu_data_directory'],
                  runtime_environment=prior['runtime_environment'],runtime_libraries=prior['runtime_libraries'],
                  previous_completed_campaign=str(PREVIOUS),production_physics_unchanged=True,
                  cpu_policy_file=str(root/'cpu_policy.json'))
    assert manifest['binary_sha256']==prior['binary_sha256']
    files=[p for d in ['bundle','code','scenes'] for p in sorted((root/d).iterdir()) if p.is_file()]
    files.append(Path(scene['geometry']['mesh_path']))
    manifest['files']=[dict(path=str(p),sha256=digest(p)) for p in files]
    save(root/'campaign.json',manifest)
    save(root/'progress.json',dict(state='prepared_waiting_cpu',completed=[],qualified=[],stages=[],updated_utc=now()))
    save(root/'report/cpu_conflicts.json',dict(checked_utc=now(),conflicts=conflicts()))
    write_readme(root)
    print('PREPARED',len(seeds),'candidate seeds; physical binary unchanged',flush=True)


def screen_accept(root,folder):
    import yaml
    s=yaml.safe_load((folder/'output/terrain_run.yaml').read_text())
    d=s['diagnostics'];a=s['accelerator']
    assert s['complete'] and not d['csv_truncated'] and not d['deposition_csv_truncated'] and not d['domain_exit_rows_truncated']
    assert d['material_mismatches']==0 and a['pending_particles']==0
    assert a['execution_space']=='OpenMP' and a['execution_concurrency']==THREADS
    assert abs(s['energy_ledger']['unexplained_over_initial'])<1e-8
    assert not s['neutrino']['force_interaction_called'] and not s['tau']['force_decay_called']
    out=root/'report'/folder.name;out.mkdir(exist_ok=False)
    helper=module('screen_helpers',root/'code/screen_audit_helpers.py')
    records,tau=helper.tracks_and_profile(folder,s,out)
    result=module('classifier',root/'code/classify_psr.py').classify(folder,s)
    save(out/'acceptance.json',dict(passed=True,scope='Complete radio-disabled transport and genealogy; not a radio result.',records=records,classification=result,energy_ledger=s['energy_ledger']))
    save(out/'classification.json',result)
    shutil.copy2(folder/'timing.json',out/'timing.json')
    (out/'README_CN.md').write_text('# '+folder.name+'\n\n完整输运和记录检查通过；double bang：'+str(result['qualified_doublebang'])+'。本阶段不带射电。\n\n[实际谱系](classification.json) · [完整性与账本](acceptance.json) · [沉积 profile](profile.npz) · [耗时](timing.json)\n')
    return result


def wait_resources(root,p,stage):
    while True:
        if (root/'STOP_AFTER_CURRENT').exists():
            return False
        overlap=conflicts()
        policy=read(root/'cpu_policy.json')
        save(root/'report/cpu_conflicts.json',dict(checked_utc=now(),allow_shared=policy['allow_shared'],conflicts=overlap))
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
        disk=shutil.disk_usage(root).free/GIB
        memory_needed=190 if stage=='radio' else 32
        disk_needed=180 if stage=='radio' else 30
        if (not overlap or policy['allow_shared']) and mem>memory_needed and disk>disk_needed:
            return True
        p.update(state='waiting_cpu' if overlap and not policy['allow_shared'] else 'waiting_resources',updated_utc=now())
        p.pop('latest',None)
        save(root/'progress.json',p);write_readme(root)
        time.sleep(30)


def execute(root):
    global CPU_LIST, CPUS
    lock=(root/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(root/'campaign.json');p=read(root/'progress.json')
    CPU_LIST=m['cpu_list'];CPUS=cpu_set(CPU_LIST)
    assert len(CPUS)==THREADS
    topology()
    os.environ.update(m['runtime_environment'])
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=str(THREADS),MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    os.sched_setaffinity(0,CPUS)
    for row in m['files']+m['runtime_libraries']:
        assert digest(Path(row['path']))==row['sha256'],row['path']
    runner=module('runner',root/'code/event_runner.py');runner.readme=write_readme
    helper=module('monitor',root/'code/legacy_driver.py');helper.affinity=affinity
    try:
        for seed in m['seeds']:
            if p['qualified'] or (root/'STOP_AFTER_CURRENT').exists():
                break
            for stage in ['coarse','confirm','radio']:
                case=next(c for c in m['cases'] if c['seed']==seed and c['stage']==stage)
                existing=next((x for x in p['stages'] if x['tag']==case['tag']),None)
                if existing:
                    if not existing['qualified']:
                        break
                    continue
                if not wait_resources(root,p,stage):
                    break
                p.pop('latest',None)
                p.update(state='starting_'+stage,current=case['tag'],updated_utc=now())
                save(root/'progress.json',p);write_readme(root)
                print('START',case['tag'],now(),flush=True)
                folder,timing=runner.run_case(root,case,m,p,helper)
                p.pop('latest',None)
                p.update(state='auditing_'+stage,updated_utc=now());save(root/'progress.json',p);write_readme(root)
                start=time.monotonic()
                if stage=='radio':
                    with (folder/'audit.log').open('x') as log:
                        subprocess.run([sys.executable,str(root/'code/audit_psr.py'),'--root',str(root),'--folder',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    verdict=read(root/'report'/folder.name/'acceptance.json');assert verdict['passed']
                    classification=verdict['classification'];p['completed'].append(folder.name)
                else:
                    classification=screen_accept(root,folder)
                timing['validation_and_figures_seconds']=time.monotonic()-start
                if stage=='radio':
                    start=time.monotonic()
                    with (folder/'archive.log').open('x') as log:
                        subprocess.run([sys.executable,str(root/'code/archive_psr.py'),'--folder',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    timing['lossless_archive_seconds']=time.monotonic()-start
                    if classification['qualified_doublebang']:
                        p['qualified'].append(folder.name)
                save(folder/'timing.json',timing);save(root/'report'/folder.name/'timing.json',timing)
                p['stages'].append(dict(tag=folder.name,stage=stage,seed=seed,qualified=classification['qualified_doublebang'],wall_s=timing['process_wall_seconds']))
                p.update(updated_utc=now());save(root/'progress.json',p);write_readme(root)
                print('ACCEPTED_STAGE',p['stages'][-1],flush=True)
                if not classification['qualified_doublebang']:
                    break
        p.update(state='complete' if p['qualified'] else 'stopped_after_current' if (root/'STOP_AFTER_CURRENT').exists() else 'candidate_limit_reached',updated_utc=now())
    except BaseException as e:
        p.update(state='failed',error=str(e),traceback=traceback.format_exc(),updated_utc=now())
        raise
    finally:
        save(root/'progress.json',p);write_readme(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    assert socket.gethostname()=='psrpku2025'
    prepare(args.root) if args.prepare else execute(args.root)
