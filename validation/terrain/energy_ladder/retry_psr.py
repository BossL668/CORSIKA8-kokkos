#!/usr/bin/env python3
"""Frozen-binary PSR retry with effective thinning and a larger resident queue."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import yaml

def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_runner(root):
    spec = importlib.util.spec_from_file_location('frozen_runner', root/'code/reference_runner.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    runner.readme = readme
    return runner

def readme(root):
    p = json.loads((root/'progress.json').read_text())
    m = json.loads((root/'campaign.json').read_text())
    lines = ['# 1 PeV SiO₂ double bang：1e-4 薄化与 4M 常驻队列', '',
        f'状态：`{p["state"]}`；完整验收 {len(p["completed"])} 例，double bang 选图通过 {len(p["qualified"])} 例。',
        f'更新时间：{p["updated_utc"]}；当前：`{p.get("current", "尚未开始")}`。', '',
        'PSR OpenMP 256 个物理核；SiO₂、原入射位置与方向、IGRF14（2027）、DEM 覆盖边界、全部 80 站同时计算 CoREAS/ZHS。站点经纬度沿用已核对输入，高度仍是 DEM+1 m 模型值。', '',
        '**emthin=1e-4，阈值 100 GeV；按空气默认公式得到 Wmax=50。** 原 1e-6 设置得到 Wmax=0.5，会阻止单位权重粒子启动薄化。本次保留 EM 动能截止 0.5 MeV、强子/μ/τ 截止 0.3 GeV 和 LPM。', '',
        '**常驻粒子容量 4,194,304，是原容量的 16 倍。** 波前 4096、记录缓冲 65536、射电源缓冲 8192；队列仍有固定上限，溢出仍会明确报错。', '',
        '输运窗 500 μs；80 站射电窗 2048 μs、256 MHz，设备总预算 96 GiB。进程内存、可用内存与磁盘持续监测。', '',
        '本轮只试跑 seed 22309 这一例；不自动扫描其他种子或启动更高能量。先记录完整结果与耗时，再判断 double bang 是否实际发生。', '',
        '自然 CC/NC 与 TAUOLA，未强制相互作用或衰变。真实顶点谱系决定 double bang；本档要求两顶点可见 shower 次级各 ≥10 TeV、实际输运且顶点间距 ≥10 m。改变薄化后须重新确认事件，候选名不等于 double bang 已发生。', '',
        '完成后验收完整轨迹、加权沉积 profile、出界末态、包含随机薄化跳变的能量账本和 80 站射电。记录总耗时及各阶段耗时，绘制英文图，随后无损归档。薄化保留统计贡献，但会增加单事件波形和 profile 的抽样噪声。', '',
        '[输入与完整命令](campaign.json) · [状态](progress.json) · [队列容量准入检查](report/preflight/README_CN.md)', '']
    if p.get('latest'):
        x=p['latest']; lines += [f'已耗时 {x["elapsed_s"]/3600:.3f} h；已写出记录前缀 {x["recorded_steps_prefix"]:,} 步；RSS {x["rss_GiB"]:.2f} GiB。这不是完整事件耗时或完成百分比。', '']
    for tag in p['completed']: lines += [f'- [{tag}](report/{tag}/README_CN.md)']
    if p.get('error'): lines += ['', '停止原因：'+p['error']]
    lines += ['', '本次仅改变运行参数，生产二进制与上轮相同。模拟、检查和绘图均在 PSR，本地仅同步报告。', '', 'PSR 目录：`'+str(root)+'`。']
    (root/'README_CN.md').write_text('\n'.join(lines)+'\n')

def prepare(root, reference):
    assert not (root/'campaign.json').exists()
    prior=json.loads((reference/'campaign.json').read_text())
    assert json.loads((reference/'progress.json').read_text())['state']=='failed'
    for directory in ['bundle','scenes','runs','report','preflight']:
        (root/directory).mkdir(exist_ok=False)
    shutil.copytree(reference/'bundle',root/'bundle',dirs_exist_ok=True)
    for name in ['legacy_driver.py','classify_psr.py','legacy_classifier.py','archive_psr.py']:
        shutil.copy2(reference/'code'/name,root/'code'/name)
    shutil.copy2(reference/'code/run_psr.py',root/'code/reference_runner.py')
    audit=(reference/'code/audit_psr.py').read_text()
    assert audit.count("s['emthin']==1e-6")==1 and audit.count(".5e-6*s['energy_GeV']")==1
    audit=audit.replace("s['emthin']==1e-6", "s['emthin']==1e-4")
    audit=audit.replace(".5e-6*s['energy_GeV']", ".5e-4*s['energy_GeV']")
    (root/'code/audit_psr.py').write_text(audit)
    scene_path=Path(prior['cases'][0]['command'][prior['cases'][0]['command'].index('--scene')+1])
    scene=yaml.safe_load(scene_path.read_text())
    scene['geometry']['material_file']=str(root/'bundle/silica_SiO2.yaml')
    scene['provenance'].update(station_audit=str(root/'bundle/stations_80_audit.json'),
        campaign_note='User requested emthin=1e-4; air default Wmax=50; resident capacity 4194304; rerun natural candidates.')
    (root/'scenes/silica_SiO2_21CMA80.yaml').write_text(yaml.safe_dump(scene,sort_keys=False))
    cases=[]
    for old in prior['cases'][:1]:
        case=dict(old); cmd=list(old['command']); case['command']=cmd
        cmd[3]=str(root/'bundle/c8_terrain_cascade')
        for key,value in {'--scene':root/'scenes/silica_SiO2_21CMA80.yaml',
            '--output':root/'runs'/case['tag']/'output','--emthin':1e-4,
            '--resident-capacity':4194304}.items(): cmd[cmd.index(key)+1]=str(value)
        # Let the same air-default formula compute 50, rather than carrying 0.5 forward.
        i=cmd.index('--max-weight');del cmd[i:i+2]
        cases.append(case)
    manifest={**prior,'created_utc':now(),'cases':cases,'previous_failed_campaign':str(reference),
        'changes_from_reference':'No binary changes; emthin 1e-6 -> 1e-4, automatic Wmax 0.5 -> 50, resident capacity 262144 -> 4194304.',
        'thinning':dict(emthin=1e-4,threshold_GeV=100,maximum_weight=50,automatic_maximum_weight=True),
        'resident_capacity':4194304,'maximum_candidates':1,
        'seed_selection':'Single pilot: rerun the first prior naturally prefiltered candidate, seed 22309; no automatic seed scan.',
        'user_scope':'Try one 1 PeV event with emthin=1e-4, automatic air-default Wmax, and an enlarged resident queue.'}
    runtime_dirs=list(dict.fromkeys(str(Path(r['path']).parent) for r in prior['runtime_libraries']
        if Path(r['path']).name in ['libgfortran.so.5','libcudart.so.12']))
    manifest['runtime_environment']={'LD_LIBRARY_PATH':':'.join(runtime_dirs),
        'FLUPRO':'/home/yuhanglu/fluka','PYTHONUNBUFFERED':'1'}
    manifest['files']=[dict(path=str(p),sha256=sha(p)) for d in ['bundle','scenes','code']
        for p in sorted((root/d).iterdir()) if p.is_file()]
    mesh=Path(scene['geometry']['mesh_path'])
    assert sha(mesh)==scene['provenance']['mesh_sha256']
    manifest['files'].append(dict(path=str(mesh),sha256=sha(mesh)))
    assert sha(root/'bundle/c8_terrain_cascade')==prior['binary_sha256']
    save(root/'campaign.json',manifest)
    save(root/'progress.json',dict(state='prepared',completed=[],qualified=[],current=cases[0]['tag'],updated_utc=now()))
    readme(root)

def preflight(root):
    m=json.loads((root/'campaign.json').read_text());reference=Path(m['previous_failed_campaign'])
    base=root.parent/'beta5_material_models_20260913/build-openmp'
    target=base/'tests/accelerator/CMakeFiles/testKokkosInterfaceQueue.dir'
    out=root/'report/preflight';out.mkdir(exist_ok=False)
    flags={k:shlex.split(v) for line in (target/'flags.make').read_text().splitlines()
           if ' = ' in line for k,v in [line.split(' = ',1)]}
    obj=root/'preflight/QueueCapacityAdmission.o';binary=root/'preflight/QueueCapacityAdmission'
    cxx='/home/yuhanglu/miniconda/envs/corsika_venv/bin/g++'
    nvcc='/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/bin/nvcc'
    cmd=[nvcc,'-ccbin='+cxx]+flags['CUDA_DEFINES']+['-I'+str(reference/'source')]+flags['CUDA_INCLUDES']+flags['CUDA_FLAGS']
    cmd+=['-x','cu','-c',str(root/'code/QueueCapacityAdmission.cpp'),'-o',str(obj)]
    link=shlex.split((target/'link.txt').read_text())
    i=next(i for i,x in enumerate(link) if x.endswith('testKokkosInterfaceQueue.cpp.o'));link[i]=str(obj)
    link[link.index('-o')+1]=str(binary)
    commands=[]
    for label,command in [('compile',cmd),('link',link)]:
        commands.append(dict(label=label,argv=command,cwd=str(target.parent.parent)))
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(command,cwd=target.parent.parent,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    env=dict(os.environ,OMP_NUM_THREADS='256',OMP_PROC_BIND='spread',OMP_PLACES='threads',OPENBLAS_NUM_THREADS='1')
    cuda_lib='/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/targets/x86_64-linux/lib'
    env['LD_LIBRARY_PATH']=cuda_lib+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    command=['taskset','-c','0-255',str(binary)]
    save(out/'commands.json',commands+[dict(label='run',argv=command,environment={k:env[k]
        for k in ['OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH']})])
    start=time.monotonic()
    result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
    (out/'run.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    proof=json.loads(next(x for x in result.stdout.splitlines() if x.startswith('{')))
    proof.update(wall_seconds=time.monotonic()-start,production_binary_unchanged=sha(root/'bundle/c8_terrain_cascade')==m['binary_sha256'],
        production_memory_gate='The unchanged production initialization checks combined transport and radio allocations against 96 GiB.',
        scope='Full-capacity FIFO/compaction and overflow preservation; deterministic thinning activation. This is not a complete shower or radio validation.')
    save(out/'acceptance.json',proof)
    (out/'README_CN.md').write_text('# 扩大队列的准入检查\n\nPSR OpenMP 256 核检查通过：填满 4,194,304 个粒子后执行波前提交与队列整理，逐个核对全部粒子的谱系、随机数步号、种类、介质、权重和能量；满载溢出仍保留原输入。\n\n每粒子 %d 字节，队列工作区约 %.3f GiB，整理时峰值约 %.3f GiB；不含射电等其他模块。生产程序仍执行 96 GiB 总预算检查。\n\n验证 Wmax=0.5 会跳过单位权重粒子的薄化，而本次阈值 100 GeV、Wmax=50 可启动薄化。此次检查不替代完整事件及射电验收。\n\n[数值](acceptance.json) · [运行日志](run.log) · [编译及执行命令](commands.json)\n'%(proof['particle_bytes'],proof['queue_workspace_bytes']/1024**3,proof['peak_queue_workspace_bytes']/1024**3))
    print(json.dumps(proof),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path);parser.add_argument('--prepare',action='store_true');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();assert socket.gethostname()=='psrpku2025'
    if args.prepare: prepare(args.root,args.reference)
    elif args.preflight: preflight(args.root)
    else:
        assert json.loads((args.root/'report/preflight/acceptance.json').read_text())['passed']
        manifest=json.loads((args.root/'campaign.json').read_text())
        os.environ.update(manifest['runtime_environment'])
        load_runner(args.root).execute(args.root)
