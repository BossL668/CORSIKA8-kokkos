#!/usr/bin/env python3
"""Isolated 80-station double-bang campaign; only runs on PSR."""
import argparse
import csv
import datetime
import fcntl
import hashlib
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
import zlib
import yaml

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
BOUNDARY = BASE / 'beta5_dem_boundary_20260914'
PREVIOUS = BASE / 'beta5_SiO2_high_energy_aircuts_thin1e-5_20260914'
GIB = 1024**3


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def setarg(cmd, key, value):
    cmd[cmd.index(key) + 1] = str(value)


def mem_available():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                    if s.startswith('MemAvailable:'))) * 1024


def prepare(root):
    if (root / 'campaign.json').exists():
        raise RuntimeError('Campaign inputs already exist; use their recorded manifest')
    acceptance = json.loads((BOUNDARY / 'report/acceptance.json').read_text())
    assert acceptance['passed'] and acceptance['stations']['count'] == 80
    assert (PREVIOUS / 'preflight/PASSED').exists()
    control=root/'control_check/report/air_80_control/acceptance.json'
    assert json.loads(control.read_text())['passed']
    assert json.loads((root/'control_check/profile_intervals.json').read_text())['passed']
    for name in ['bundle', 'scenes', 'report', 'runs']:
        (root / name).mkdir(exist_ok=False)
    binary = root / 'bundle/c8_terrain_cascade'
    shutil.copy2(BOUNDARY / 'build-openmp/c8_terrain_cascade', binary)
    assert digest(binary) == acceptance['binaries']['openmp']
    for name in ['acceptance.json', 'stations_80_audit.json', 'stations_80_audit.csv', 'source_manifest.json']:
        shutil.copy2(BOUNDARY / 'report' / name, root / 'bundle' / name)
    for name in ['observers.yaml', 'terrain_grid.npz', 'terrain_manifest.yaml']:
        shutil.copy2(BOUNDARY / 'stations' / name, root / 'bundle' / name)
    for name in ['silica_SiO2.yaml', 'silica_SiO2_expected.yaml']:
        shutil.copy2(PREVIOUS / 'bundle' / name, root / 'bundle' / name)
    shutil.copy2(PREVIOUS / 'preflight/result.json', root / 'bundle/LPM_to_1EeV.json')
    shutil.copy2(control,root/'bundle/streaming_audit_control.json')
    shutil.copy2(root/'control_check/profile_intervals.json',root/'bundle/profile_intervals.json')
    old = json.loads((PREVIOUS / 'campaign.json').read_text())
    scene = yaml.safe_load((PREVIOUS / 'scenes/silica_SiO2.yaml').read_text())
    scene['geometry']['material_file'] = str(root / 'bundle/silica_SiO2.yaml')
    scene['geometry']['transport_boundary'] = {'type': 'dem_coverage'}
    scene['radio']['observers'] = yaml.safe_load((root / 'bundle/observers.yaml').read_text())['radio']['observers']
    scene['radio'].update(enabled=True, samples=524288, memory_MiB=81920)
    scene['provenance'].update(
        campaign_note='100 PeV regression, then 300 PeV and 1 EeV new seeds; actual DEM coverage boundary; 80 stations; natural CC/NC and TAUOLA.',
        radio_note='All 80 E/W/N/S stations; preserve surveyed longitude/latitude and existing DEM+1m model height. Model heights differ from measured heights.',
        radio_disabled=False, station_audit=str(root / 'bundle/stations_80_audit.json'))
    scene_file = root / 'scenes/silica_SiO2_21CMA80_dem_coverage.yaml'
    scene_file.write_text(yaml.safe_dump(scene, sort_keys=False))
    moments = 3 * 524288 * 80 * 6 * 13 * 8
    # Device/host moment copies coexist at finalization, plus rendered fields,
    # optical tables, queues and native physics. Do not retain the old 32 GiB RSS guard.
    budget = dict(moment_arrays_bytes=moments, moment_arrays_GiB=moments/GIB,
        radio_limit_GiB=80, transport_and_radio_limit_GiB=96,
        estimated_finalization_RSS_GiB=170, process_RSS_guard_GiB=220,
        minimum_host_available_GiB=16, available_before_prepare_GiB=mem_available()/GIB,
        free_disk_before_prepare_GiB=shutil.disk_usage(root).free/GIB)
    assert budget['available_before_prepare_GiB'] > 190
    assert budget['free_disk_before_prepare_GiB'] > 180
    cases = []
    def add(energy, seed, role):
        cmd = list(old['cases'][0]['command'])
        tag = 'openmp_SiO2_21CMA80_%dPeV_seed%d' % (energy//1000000, seed)
        cmd[3] = str(binary)
        for key, value in {'--scene': scene_file, '--output': root/'runs'/tag/'output',
            '--energy-GeV': energy, '--seed': seed, '--max-weight': .5e-5*energy,
            '--device-memory-MiB': 98304, '--track-row-limit': 1000000000,
            '--transport-step-limit': 1000000000}.items():
            setarg(cmd, key, value)
        assert cmd[cmd.index('--emthin')+1] == '1e-05'
        cases.append(dict(tag=tag, energy_GeV=energy, seed=seed, role=role, command=cmd))
    add(100000000, 946, 'reference_regression')
    # Carry forward all 27 prefiltered NEW candidates per energy. Original
    # seeds do not fill the new-seed quota and do not predict a new decay mode.
    candidates = {energy: [c for c in old['cases'] if c['energy_GeV']==energy and c['seed'] not in [158,946,3605]]
                  for energy in [300000000,1000000000]}
    for rank in range(max(map(len,candidates.values()))):
        for energy, rows in candidates.items():
            if rank < len(rows):
                add(energy, rows[rank]['seed'], 'new_seed_search')
    files = [p for directory in ['bundle','scenes','code'] for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.append(Path(scene['geometry']['mesh_path']))
    save(root/'campaign.json', dict(created_utc=now(), host=socket.gethostname(), cases=cases,
        files=[dict(path=str(p),sha256=digest(p)) for p in files],
        runtime_libraries=old['runtime_libraries'], binary_sha256=digest(binary),
        boundary_acceptance_root=str(BOUNDARY), previous_campaign=str(PREVIOUS),
        physical_cores=256, observer_count=80, transport_window_ns=500000,
        radio_samples=524288, radio_sample_rate_Hz=256e6, memory=budget,
        target_new_doublebangs_per_energy=3, maximum_candidates_per_energy=27,
        seed_policy='Approximate interaction-depth prefilter only; actual genealogy decides double bang. Not an unbiased event-rate sample.',
        cpu_data_directory=old['cases'][0]['cache']))
    save(root/'progress.json', dict(state='prepared', completed=[], qualified={}, total=len(cases),updated_utc=now()))
    write_readme(root)
    print('PREPARED',len(cases),'cases;',moments/GIB,'GiB resident moments',flush=True)


def write_readme(root):
    p=json.loads((root/'progress.json').read_text())
    m=json.loads((root/'campaign.json').read_text())
    lines=['**SiO₂ · DEM 边界 · 21CMA 80 站 double-bang 重跑**','',
        '状态：`%s`；已完整验收 %d 例。服务器更新时间：%s。' % (p['state'],len(p['completed']),p['updated_utc']),
        '当前：`%s`。' % p.get('current','尚未开始'),'',
        '先跑 100 PeV / seed 946 回归例，再交替运行 300 PeV 和 1 EeV 的新种子，每档目标 3 个实际 double bang，最多 27 个新候选。旧种子不抵扣新种子目标；没有强制相互作用或 τ 衰变。','',
        'OpenMP 256 个物理核；设备常驻 EM 队列；CoREAS 与 ZHS 同算全部 80 站。保留原入射位置、方向、IGRF14（2027）和 SiO₂ 基准。EM 截断 0.5 MeV，强子/μ/τ 为 0.3 GeV；emthin=1e-5，最大权重分别为 500、1500、5000；LPM 开启。','',
        '输运窗 500 μs；接收窗 2048 μs、256 MHz。三个射电矩数组常驻约 73.125 GiB，射电预算 80 GiB，总设备预算 96 GiB，包含最后拷贝及渲染的进程内存另行监测。每次只跑一例。','',
        '80 站经纬度来自已核对的绝对坐标，高度仍采用 DEM+1 m，与原始测点高程相差约 −1.00～+14.09 m。这里不是实测天线相位中心模型。','',
        'DEM 覆盖侧边界处停止输运，剩余能量单列为逃逸；检查域内全部 profile、出界末态唯一性、能量账本和射电源段。域外 shower 尾部被有意截断。','',
        '每例完成后才判定 double bang：自然岩石 CC，经实际 τ 谱系连接到电子/强子衰变，两顶点可见子粒子均实际输运、能量各 ≥1 PeV、间隔 ≥100 m。此拓扑选择不等于接收站一定能分辨两个射电脉冲。未通过拓扑筛选的完整事件也保留。','',
        '模拟、验收、绘图全部在 PSR；本地只同步文件。完成事件自动生成全部 80 站波形、山体/大气源分量、几何和沉积 profile。大文件验收后无损 gzip 归档，验证解压散列后才去掉等价的未压缩副本。','']
    for tag in p['completed']:
        lines.append('- [%s](report/%s/README_CN.md)' % (tag,tag))
    if p.get('error'): lines += ['', '停止原因：'+p['error']]
    lines += ['', '运行目录：`'+str(root)+'`。输入见 [campaign.json](campaign.json)，状态见 [progress.json](progress.json)。']
    (root/'README_CN.md').write_text('\n'.join(lines)+'\n')


class TrackTail:
    """Read only a bounded prefix of newly compressed bytes for progress."""
    def __init__(self,path):
        self.path=path;self.offset=0;self.decoder=zlib.decompressobj(31);self.pending=b'';self.step=0
    def sample(self):
        if not self.path.exists():return self.step
        with self.path.open('rb') as f:
            f.seek(self.offset)
            # Monitoring must not monopolize a core reading a huge backlog.
            for _ in range(8):
                data=f.read(262144)
                if not data:break
                self.offset+=len(data);self.pending+=self.decoder.decompress(data)
                k=self.pending.rfind(b'\n')
                if k>=0:
                    last=self.pending[:k].rsplit(b'\n',1)[-1]
                    try:self.step=int(last.split(b',',1)[0])
                    except ValueError:pass
                    self.pending=self.pending[k+1:]
        return self.step


def affinity(pid):
    tasks=list(Path('/proc/%d/task'%pid).glob('*/status'))
    sets=[]
    for path in tasks:
        try:
            value=next(line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('Cpus_allowed_list:'))
        except (FileNotFoundError,StopIteration):continue
        allowed=set()
        for item in value.split(','):
            bounds=list(map(int,item.split('-')))
            allowed.update(range(bounds[0],bounds[-1]+1))
        if not allowed<=set(range(256)):return dict(verified=False,reason='thread outside cores 0-255')
        if len(allowed)==1:sets.append(next(iter(allowed)))
    return dict(verified=len(set(sets))==256,threads_inspected=len(tasks),singleton_physical_cores=len(set(sets)))


def run_event(root, case, manifest, progress):
    folder=root/'runs'/case['tag'];folder.mkdir(exist_ok=False)
    (folder/'work').mkdir()
    command=case['command'];save(folder/'command.json',command)
    env=dict(os.environ,CORSIKA_DATA=manifest['cpu_data_directory'],OMP_NUM_THREADS='256',
        OMP_PROC_BIND='spread',OMP_PLACES='threads',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for key in ['KOKKOS_TOOLS_LIBS','C8_INTERFACE_TRACE_FILE']:env.pop(key,None)
    start=time.monotonic();reason=None;peak=0;binding={'verified':False}
    tail=TrackTail(folder/'output/terrain/tracks.csv.gz')
    with (folder/'run.log').open('x') as log,(folder/'resources.jsonl').open('x') as resources:
        child=subprocess.Popen(command,cwd=folder/'work',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/'pid').write_text(str(child.pid))
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-start
                try:status=Path('/proc/%d/status'%child.pid).read_text()
                except FileNotFoundError:continue
                rss=int(next((s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:')),'0'))*1024
                peak=max(peak,rss)
                if not binding['verified']:
                    binding=affinity(child.pid);save(folder/'affinity.json',binding)
                sample=dict(elapsed_s=elapsed,rss_GiB=rss/GIB,available_GiB=mem_available()/GIB,
                    free_disk_GiB=shutil.disk_usage(root).free/GIB,recorded_steps_prefix=tail.sample(),pid=child.pid)
                resources.write(json.dumps(sample)+'\n');resources.flush()
                progress.update(state='running',current=case['tag'],latest=sample,updated_utc=now())
                save(root/'progress.json',progress);write_readme(root)
                if elapsed>7*24*3600:reason='Owned event exceeded 7-day limit'
                if rss>220*GIB:reason='Owned event RSS exceeded 220 GiB'
                if sample['available_GiB']<16:reason='Available RAM below 16 GiB'
                if sample['free_disk_GiB']<30:reason='Free disk below 30 GiB'
                if (root/'STOP_AFTER_CURRENT').exists():progress['stop_after_current_requested']=True
                if reason:break
                time.sleep(10)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    output=folder/'output/terrain_run.yaml'
    summary=yaml.safe_load(output.read_text()) if output.exists() else {}
    result=dict(complete=bool(summary.get('complete')),returncode=child.returncode,stop_reason=reason,
        wall_s=time.monotonic()-start,peak_rss_GiB=peak/GIB,binary_sha256=digest(Path(command[3])),
        scene_sha256=digest(Path(command[command.index('--scene')+1])),error=summary.get('error'))
    save(folder/'status.json',result)
    if reason or child.returncode or not result['complete']:raise RuntimeError('Incomplete '+case['tag']+': '+str(reason or summary.get('error') or child.returncode))
    assert binding['verified'] and result['binary_sha256']==manifest['binary_sha256']
    return folder


def execute(root):
    lock=(root/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'campaign.json').read_text())
    assert manifest.get('memory_budget_admission',{}).get('passed'), 'Validate the enlarged CLI budget with memory_budget_psr.py before production'
    for row in manifest['files']+manifest['runtime_libraries']:
        assert digest(Path(row['path']))==row['sha256'],row['path']
    # CPU identifiers 0..255 must denote distinct physical cores on this host.
    topology=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    physical=[tuple(map(int,line.split(',')))[1:] for line in topology.splitlines() if not line.startswith('#') and int(line.split(',')[0])<256]
    assert len(physical)==256 and len(set(physical))==256
    progress=json.loads((root/'progress.json').read_text())
    qualified={str(e):progress.get('qualified',{}).get(str(e),[]) for e in [300000000,1000000000]}
    progress['qualified']=qualified
    try:
        for case in manifest['cases']:
            tag=case['tag'];energy=str(case['energy_GeV'])
            if tag in progress['completed']:continue
            if case['role']=='new_seed_search' and len(qualified[energy])>=3:continue
            if shutil.disk_usage(root).free<180*GIB:raise RuntimeError('Less than 180 GiB free before next event; completed outputs retained')
            if mem_available()<190*GIB:raise RuntimeError('Less than 190 GiB RAM available before next event')
            print('START',tag,now(),flush=True)
            folder=run_event(root,case,manifest,progress)
            progress.update(state='auditing',current=tag,updated_utc=now());save(root/'progress.json',progress);write_readme(root)
            subprocess.run([sys.executable,str(root/'code/audit_psr.py'),'--root',str(root),'--folder',str(folder)],check=True)
            verdict=json.loads((root/'report'/tag/'acceptance.json').read_text())
            assert verdict['passed']
            if case['role']=='new_seed_search' and verdict['classification']['qualified_doublebang']:
                qualified[energy].append(tag)
            progress['completed'].append(tag)
            progress.update(state='archiving',updated_utc=now());save(root/'progress.json',progress);write_readme(root)
            subprocess.run([sys.executable,str(root/'code/archive_psr.py'),'--folder',str(folder)],check=True)
            if progress.get('stop_after_current_requested'):break
        met=all(len(q)>=3 for q in qualified.values())
        progress.update(state='complete' if met else 'stopped_after_current' if progress.get('stop_after_current_requested') else 'candidate_limit_reached',target_met=met,updated_utc=now())
    except BaseException as error:
        progress.update(state='failed',error=str(error),traceback=traceback.format_exc(),updated_utc=now())
        raise
    finally:save(root/'progress.json',progress);write_readme(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    assert socket.gethostname()=='psrpku2025','PSR only; no local simulations or checks'
    (prepare if args.prepare else execute)(args.root.resolve())
