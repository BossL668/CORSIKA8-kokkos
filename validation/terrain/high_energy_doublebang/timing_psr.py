#!/usr/bin/env python3
"""PSR-only low-energy timing ladder, retaining physical settings and radio."""
import argparse
import csv
import datetime
import importlib.util
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import types
import zlib

import yaml


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def write(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class LiveTracks:
    """Incremental read of gzip bytes already written; never edits a live file."""
    def __init__(self, folder):
        self.path=folder/'output/terrain/tracks.csv.gz'
        self.offset=0;self.decoder=zlib.decompressobj(31);self.pending=b''
        self.last_step=0;self.started=None;self.samples=[]

    def sample(self, elapsed):
        if not self.path.exists():return
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                data=stream.read(262144)
                if not data:break
                self.offset+=len(data)
                self.pending+=self.decoder.decompress(data)
                index=self.pending.rfind(b'\n')
                if index>=0:
                    lines=self.pending[:index].split(b'\n')
                    last=lines[-1]
                    try:self.last_step=int(last.split(b',',1)[0])
                    except ValueError:pass
                    self.pending=self.pending[index+1:]
        if self.last_step and self.started is None:self.started=elapsed
        self.samples.append(dict(elapsed_s=elapsed,steps=self.last_step,compressed_bytes=self.offset))


def report(root, rows):
    out=root/'report';out.mkdir(exist_ok=True)
    write(out/'timing_results.json',dict(updated_utc=now(),rows=rows,
        scope='Natural nu_tau timing probes with actual CC checks; not high-energy double-bang acceptance or a detector sensitivity calculation.'))
    md=['# 低能计时与高能任务状态','','300 PeV / 1 EeV 正式事件尚未启动。先按照用户要求做低能计时。','',
        'SiO₂；EM 截断 0.5 MeV，强子/μ/τ 0.3 GeV，emthin=1e-5；最大权重仍按 0.5×emthin×E/GeV。原 DEM、入射方向和三个站点，500 μs 输运，2048 μs 射电接收，CoREAS/ZHS 同算，OpenMP 256 核。','',
        '表中“shower 阶段”来自程序 shower_seconds，包含该阶段的设备/表格上传、输运、射电收尾和输出；不是单独 GPU/CPU 内核计时。“总进程时间”还包含原生物理初始化。gzip 完整轨迹检查另计。','',
        '| 能量/TeV | seed | 完成 | CC 数 | τ 衰变数 | 步数 | 总进程/min | shower阶段/min | 验证/min |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append('| %g | %d | %s | %s | %s | %d | %.2f | %s | %.2f |' %
            (r['energy_GeV']/1000,r['seed'],'是' if r['complete'] else '否',r.get('CC_count','—'),r.get('tau_decay_count','—'),
             r['steps'],r['wall_s']/60,'%.2f'%(r['shower_seconds']/60) if r.get('shower_seconds') is not None else '—',r.get('audit_s',0)/60))
    eligible=[r for r in rows if r['complete'] and r.get('CC_count',0)>0]
    if eligible:
        rates=[r['steps']/r['shower_seconds'] for r in eligible]
        md+=['','已完成 CC 样本的平均速度范围：%.0f–%.0f 个记录步/秒（含上述阶段开销）。' % (min(rates),max(rates)), '',
             '如果高能事件分别产生 10⁸ 或 10⁹ 个记录步，在相同平均速度下，阶段耗时约为 %.2f–%.2f 小时、%.2f–%.2f 小时。这只是给定步数的条件换算，尚不是 300 PeV / 1 EeV 的预测。' % (1e8/max(rates)/3600,1e8/min(rates)/3600,1e9/max(rates)/3600,1e9/min(rates)/3600),'']
    if any(not r['complete'] for r in rows):
        md+=['','有试跑达到计时上限或失败；它只提供已运行时间及已生成步数，不能作为完整 shower 耗时。后续更高能量没有自动启动。CC/τ 列的“—”表示没有最终统计，不表示没有发生相互作用或衰变；轨迹前缀的实际证据见下方链接。','']
    md+=['','低能量的最大权重为 0.05、0.5、5；前两档从单位权重出发不会触发 thinning。高能时最大权重随能量增大，不能用能量倍数直接乘低能耗时。LPM、CC 非弹性、τ 衰变位置、CPU 强子输运和射电可见光路也会改变成本。','',
         '此前启动失败是误用了射电字节相等检查：解压后的全部轨迹、沉积和逃逸 CSV 完全一致；原射电数组只差约 10⁻¹⁶。现保留 CSV 精确比较，并对射电 moment、时域场及频谱用 10⁻¹² 数值容差。','',
         '旧粗 thinning 孤立进程已确认终止，计时前检查 CPU 无其他计算任务。每个耗时都是实测，不把原初始化建表的冷启动时间混成能量标度。','']
    md+=['[计时曲线、条件耗时换算及实际 CC/τ 轨迹证据](LIVE_TIMING_CN.md)','']
    (out/'TIMING_CN.md').write_text('\n'.join(md))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=pathlib.Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    assert socket.gethostname()=='psrpku2025'
    assert (root/'preflight/PASSED').exists(), 'Corrected preflight must pass first'
    timing=root/'timing';timing.mkdir(exist_ok=True)
    manifest=json.loads((root/'campaign.json').read_text())
    helper=module('material_driver',root/'bundle/material_driver.py')
    owned=module('owned_run',root/'bundle/owned_run.py');owned.ROOT=timing
    classifier=module('selector',root/'code/classify_psr.py')
    monitors={}
    class TimedProcess(helper.AuditedProcess):
        def __init__(self,command,**kwargs):
            self.start=time.monotonic();self.last_sample=-100
            super().__init__(command,**kwargs)
            self.monitor=LiveTracks(self.folder);monitors[self.folder.name]=self.monitor
            self.stream=(self.folder/'live_steps.jsonl').open('x')
        def poll(self):
            result=super().poll();elapsed=time.monotonic()-self.start
            if hasattr(self,'monitor') and elapsed-self.last_sample>=5:
                self.monitor.sample(elapsed);self.last_sample=elapsed
                if self.monitor.samples:
                    self.stream.write(json.dumps(self.monitor.samples[-1])+'\n');self.stream.flush()
            return result
    owned.subprocess=types.SimpleNamespace(Popen=TimedProcess,check_output=subprocess.check_output,
        STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)
    os.environ['CORSIKA_DATA']=manifest['cases'][0]['cache']
    cases=[]
    for energy in [10000,100000,1000000]:
        path=timing/('seeds_%d.csv'%energy)
        subprocess.run([str(root/'build/scan_seeds'),str(path),str(energy)],check=True)
        with path.open() as stream:seeds=list(csv.DictReader(stream))[:3]
        assert seeds
        for candidate in seeds:
            seed=int(candidate['seed']);tag='timing_nutau_SiO2_%dGeV_seed%d'%(energy,seed)
            cmd=list(manifest['cases'][0]['command'])
            for key,value in {'--output':timing/'runs'/tag/'output','--energy-GeV':energy,
                              '--seed':seed,'--max-weight':.5e-5*energy}.items():
                cmd[cmd.index(key)+1]=str(value)
            cases.append(dict(tag=tag,energy_GeV=energy,seed=seed,command=cmd,
                              approximate_prefilter_depth_m=float(candidate['depth_m'])))
    write(timing/'campaign.json',dict(created_utc=now(),cases=cases,settings_reference=str(root/'campaign.json'),
        policy='First completed actual-CC sample per energy; at most three prefiltered seeds per energy. Stop ladder on any incomplete run; never force a vertex or coarsen physical settings.',
        simulation_timeout_s=1800,binary_sha256=manifest['binary_sha256']))
    rows=[];finished=set()
    write(root/'progress.json',dict(state='timing_pilot',current='preparing',completed=[],total=60,
         qualified={},updated_utc=now(),note='High-energy production deferred for user-requested timing ladder'))
    report(root,rows)
    for case in cases:
        if case['energy_GeV'] in finished:continue
        write(timing/'progress.json',dict(state='running',current=case['tag'],completed=[r['tag'] for r in rows],updated_utc=now()))
        write(root/'progress.json',dict(state='timing_pilot',current=case['tag'],completed=[],total=60,
             timing_completed=[r['tag'] for r in rows],qualified={},updated_utc=now()))
        print('START',case['tag'],now(),flush=True)
        try:
            folder,s=owned.run(case['tag'],case['command'],'openmp',1800,reuse_complete=True)
            audit=time.monotonic();check=owned.checks(folder,s,True);audit=time.monotonic()-audit
            assert s['emcut_GeV']==.0005 and s['hadcut_GeV']==s['mucut_GeV']==.3 and s['emthin']==1e-5
            assert s['max_weight']==.5e-5*case['energy_GeV']
            assert s['accelerator']['execution_space']=='OpenMP' and s['accelerator']['execution_concurrency']==256
            assert json.loads((folder/'affinity.json').read_text())['verified']
            assert s['radio_result']['errors']==s['radio_result']['out_of_window']==0
            assert s['diagnostics']['material_mismatches']==0
            status=json.loads((folder/'status.json').read_text())
            cc=[v for v in s['neutrino']['interactions'] if v['current']=='CC']
            tau=s['tau'].get('decays',[])
            row=dict(tag=case['tag'],energy_GeV=case['energy_GeV'],seed=case['seed'],complete=True,
                wall_s=status['wall_s'],shower_seconds=s['shower_seconds'],audit_s=audit,
                startup_and_teardown_s=status['wall_s']-s['shower_seconds'],steps=s['diagnostics']['steps'],
                CC_count=len(cc),tau_decay_count=len(tau),CC_vertices=cc,tau_decays=tau,
                deposited_GeV=s['diagnostics']['deposited_energy_GeV'],maximum_weight=s['max_weight'],
                unit_weight_thinning_can_activate=s['max_weight']>1,
                gzip_bytes=sum(p.stat().st_size for p in (folder/'output/terrain').glob('*.gz')),
                checks=check['checks'],algorithm_relative_l2=check['coreas_zhs_relative_l2'])
            rows.append(row)
            if cc:finished.add(case['energy_GeV'])
            print('FINISHED',case['tag'],'CC',len(cc),'steps',row['steps'],'wall_s',row['wall_s'],flush=True)
        except BaseException as error:
            folder=timing/'runs'/case['tag'];status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            monitor=monitors.get(case['tag'])
            rows.append(dict(tag=case['tag'],energy_GeV=case['energy_GeV'],seed=case['seed'],complete=False,
                wall_s=status.get('wall_s',0),steps=monitor.last_step if monitor else 0,
                error=str(error),stop_reason=status.get('stop_reason'),status=status))
            report(root,rows)
            write(timing/'progress.json',dict(state='incomplete',current=case['tag'],updated_utc=now(),error=str(error)))
            write(root/'progress.json',dict(state='timing_pilot_incomplete',current=case['tag'],completed=[],total=60,
                qualified={},updated_utc=now(),note='High-energy production deferred; timing probe incomplete',error=str(error)))
            return
        report(root,rows)
    write(timing/'progress.json',dict(state='complete',completed=[r['tag'] for r in rows],CC_energies_GeV=sorted(finished),updated_utc=now()))
    write(root/'progress.json',dict(state='timing_pilot_complete',current='timing_results_ready',completed=[],total=60,
        qualified={},updated_utc=now(),note='Low-energy timing finished; high-energy production has not run'))


if __name__=='__main__':main()
