#!/usr/bin/env python3
"""Stage a verified radio revision and apply it between complete event stages."""
import datetime
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

assert socket.gethostname() == 'psrpku2025'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
ROOT = BASE/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916'
BUILD = BASE/'beta5_radio_air_alignment_20260916'
REV = ROOT/'revisions/20260916_air_radio_1GHz'

def read(path): return json.loads(path.read_text())
def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()
def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)

def prepare():
    import numpy as np
    import yaml
    verdict = read(BUILD/'acceptance.json')
    assert verdict['passed']
    lifetime = read(BUILD/'finalize_lifetime.json')
    assert lifetime['passed'] and lifetime['released_grids_before_accumulator_destruction']==3
    assert sha(BUILD/'build/c8_terrain_cascade') == verdict['new_binary_sha256']
    p = read(ROOT/'progress.json')
    assert p['state'] == 'running' and p['current'].startswith('coarse_')
    REV.mkdir(exist_ok=False)
    (REV/'before').mkdir(); (REV/'after').mkdir()
    targets = ['campaign.json', 'scenes/radio.yaml', 'code/run_psr.py', 'code/audit_psr.py', 'code/event_runner.py']
    for name in targets:
        dest = REV/'before'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        dest = REV/'after'/name; dest.parent.mkdir(parents=True, exist_ok=True)
    m = read(ROOT/'campaign.json')
    scene = yaml.safe_load((ROOT/'scenes/radio.yaml').read_text())
    scene['radio'].update(sample_rate_GHz=1., samples=1048576, subdivision_frequency_GHz=1., memory_MiB=163840)
    scene['provenance']['campaign_note'] += ' Radio revision: air-default 1 GHz sampling, 1048.576 us common window, 50-250 and 50-300 MHz analysis; unchanged transport.'
    (REV/'after/scenes/radio.yaml').write_text(yaml.safe_dump(scene, sort_keys=False))
    new_binary = ROOT/'bundle/c8_terrain_cascade_radio_1GHz'
    shutil.copy2(BUILD/'build/c8_terrain_cascade', new_binary)
    for case in m['cases']:
        if case['stage'] == 'radio':
            case['command'][3] = str(new_binary)
            case['command'][case['command'].index('--device-memory-MiB')+1] = '180224'
    driver = (ROOT/'code/run_psr.py').read_text()
    driver = replace(driver, '完整射电仍用 2048 μs、256 MHz 采样，图中分析 50–100 MHz；并不覆盖完整 50–300 MHz。',
                     '完整射电改为与空气默认一致的 1 GHz 采样（1 ns），共 1,048,576 点、1048.576 μs；保留未经带通输出，主图 50–250 MHz，附 50–300 MHz。空气的 400 ns 窄窗不直接用于 double bang。')
    driver = replace(driver, '生产二进制与已经完成的 1 PeV 事件一致。',
                     '粗筛/确认使用原二进制；射电阶段使用已验证的逐数组释放输出版本，物理公式与输运不变。采样与接收窗准入见 report/radio_alignment。')
    driver = replace(driver, "memory_needed=190 if stage=='radio' else 32", "memory_needed=212 if stage=='radio' else 32")
    driver = replace(driver, "disk_needed=180 if stage=='radio' else 30", "disk_needed=220 if stage=='radio' else 30")
    (REV/'after/code/run_psr.py').write_text(driver)
    runner = (ROOT/'code/event_runner.py').read_text()
    runner = replace(runner, "(190 if case['stage']=='radio' else 32)", "(212 if case['stage']=='radio' else 32)")
    runner = replace(runner, "(180 if case['stage']=='radio' else 30)", "(220 if case['stage']=='radio' else 30)")
    runner = replace(runner, "started_utc=started_utc,binary_sha256=manifest['binary_sha256']", "started_utc=started_utc,binary_sha256=digest(Path(cmd[3]))")
    (REV/'after/code/event_runner.py').write_text(runner)
    audit = (ROOT/'code/audit_psr.py').read_text()
    audit = replace(audit, "c['samples']==524288 and c['sample_rate_Hz']==256e6", "c['samples']==1048576 and c['sample_rate_Hz']==1e9 and c['subdivision_frequency_Hz']>=1e9")
    audit = audit.replace('50_100MHz', '50_250MHz').replace('50-100 MHz', '50-250 MHz').replace('50–100 MHz', '50–250 MHz')
    audit = replace(audit, '<=100e6)', '<=250e6)')
    # Produce both requested bands from the saved full-band fields, once per station.
    audit = replace(audit, 'x,z=filtered;vector=', '''wide_band=(np.fft.rfftfreq(n,1/rate)>=50e6)&(np.fft.rfftfreq(n,1/rate)<=300e6)
        wide=[np.fft.irfft(np.fft.rfft(a,axis=0)*wide_band[:,None],n=n,axis=0) for a in arrays]
        x,z=filtered;vector=''')
    audit = replace(audit, "raw_peak_V_m=float(np.max(np.linalg.norm(arrays[1][:,:3],axis=1))),", "raw_peak_V_m=float(np.max(np.linalg.norm(arrays[1][:,:3],axis=1))),\n            peak_50_300MHz_V_m=float(np.linalg.norm(wide[1][:,:3],axis=1).max()),\n            relative_l2_50_300MHz=float(np.linalg.norm(wide[0][:,:3]-wide[1][:,:3])/np.linalg.norm(wide[1][:,:3])) if np.linalg.norm(wide[1][:,:3]) else None,")
    audit = replace(audit, "print('RADIO',name,flush=True)", '''if plots:
            fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
            for k,(pair,label) in enumerate([(arrays,'Unfiltered stored band: 0-500 MHz'),(wide,'50-300 MHz')]):
                v=np.linalg.norm(pair[1][:,:3],axis=1);pk=int(np.argmax(v));component=int(np.argmax(abs(pair[1][pk,:3])))
                near=slice(max(0,pk-200),min(n,pk+201));block=128;size=n//block*block
                y=pair[1][:size,component].reshape(-1,block)*1e6
                tt=times[:size].reshape(-1,block).mean(axis=1)*1e6
                axes[k,0].fill_between(tt,y.min(axis=1),y.max(axis=1),label='ZHS; min/max envelope')
                for values,alg,style in [(pair[1],'ZHS','-'),(pair[0],'CoREAS','--')]:
                    axes[k,1].plot((times[near]-times[pk])*1e9,values[near,component]*1e6,style,label=alg)
                axes[k,0].set(xlabel='Arrival time [us]',ylabel='Electric field [uV/m]',title=label)
                axes[k,1].set(xlabel='Time from vector peak [ns]',ylabel='Electric field [uV/m]',title=['East','North','Up'][component]+' component')
            for axis_plot in axes.flat:axis_plot.grid(alpha=.2);axis_plot.legend(fontsize=8)
            fig.suptitle(name+' | Saved full band and 50-300 MHz | '+folder.name)
            fig.savefig(out/'figures'/('waveform_bands_'+name+'.png'),dpi=140);plt.close(fig)
        print('RADIO',name,flush=True)''')
    audit = replace(audit, "signals=radio(folder,out,not args.control)", "signals=radio(folder,out,not args.control)\n    assert all(row['relative_l2_50_300MHz'] is None or row['relative_l2_50_300MHz']<1e-5 for row in signals['stations'])")
    audit = replace(audit, "if len(active):axes[1].set_xlim(centres[active[0]]/1000-.1,centres[active[-1]]/1000+.1)", "if len(active):\n        cumulative=np.cumsum(np.maximum(energy,0)); bounds=np.searchsorted(cumulative,[.00001,.99999]*cumulative[-1]);axes[1].set_xlim(centres[bounds[0]]/1000-.05,centres[bounds[1]]/1000+.05)")
    audit = audit.replace("[.00001,.99999]*cumulative[-1]", "np.array([.00001,.99999])*cumulative[-1]")
    audit = replace(audit, "for row in signals['stations']:lines.append('- [%s](figures/waveform_%s.png)'%(row['station'],row['station']))", "lines += ['', '保留 1 GHz 原始波形；主图 50–250 MHz，另附未经带通 0–500 MHz 与 50–300 MHz。Profile 主视图自动放大主要沉积区，完整数值仍保存在 profile.npz。', '']\n    for row in signals['stations']:lines.append('- [%s: 50–250 MHz](figures/waveform_%s.png) · [未经带通 / 50–300 MHz](figures/waveform_bands_%s.png)'%(row['station'],row['station'],row['station']))")
    (REV/'after/code/audit_psr.py').write_text(audit)
    for path in (REV/'after/code').glob('*.py'): compile(path.read_text(), str(path), 'exec')

    # Bound arrival times before choosing the smaller, faster-sampled common window.
    previous = BASE/'beta5_doublebang_1PeV_thin1e4_queue4M_20260915'
    c = read(previous/'runs/openmp_SiO2_21CMA80_1PeV_seed22309/output/radio/ZHS/config.json')
    air = c['media'][0]; nmax = max([air['index']]+[x[1] for x in air['radial_index']])
    origin = np.array([2516.535431729868,4294.03815963489,500.52949201660203])
    obs = np.array([o['position_m'] for o in c['observers']])
    assert np.array_equal(obs,np.array([o['position_enu_m'] for o in scene['radio']['observers']]))
    distance = float(np.linalg.norm(obs-origin,axis=1).max()); T=.0005; light=299792458.
    direct_bound = T+nmax*(T+distance/light)
    # Use ALL mesh vertices, including bottom and side faces, for the bound.
    with Path(scene['geometry']['mesh_path']).open('rb') as stream:
        header=[]
        while True:
            line=stream.readline().decode('ascii').strip();header.append(line)
            if line=='end_header':break
        assert 'format binary_little_endian 1.0' in header
        count=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        vertices=np.fromfile(stream,dtype='<f8',count=count*3).reshape(count,3)
    corners=np.array(list(itertools.product(*zip(vertices.min(axis=0),vertices.max(axis=0)))))
    mesh_diameter=float(np.linalg.norm(vertices.max(axis=0)-vertices.min(axis=0)))
    destination=float(np.linalg.norm(corners[:,None,:]-obs[None,:,:],axis=2).max())
    transmitted_bound=T+(c['media'][1]['index']*mesh_diameter+nmax*destination)/light
    end=scene['radio']['start_ns']*1e-9+scene['radio']['samples']/1e9
    assert end>max(direct_bound,transmitted_bound)+10e-6
    grid_bytes=1048576*80*6*13*8
    admission=dict(passed=True, sample_rate_Hz=1e9, time_step_ns=1., nyquist_MHz=500., samples=1048576,
        duration_us=1048.576, start_us=-1., end_us=end*1e6, transport_window_us=500.,
        analysis_bands_MHz=[[50,250],[50,300]], moment_order=12, subdivision_frequency_Hz=1e9,
        direct_path_arrival_bound_us=direct_bound*1e6, transmitted_path_arrival_bound_us=transmitted_bound*1e6,
        minimum_arrival_margin_us=(end-max(direct_bound,transmitted_bound))*1e6,
        time_bound_scope='Endpoint geometric bound: causal tracks start at injection, t<=500 us; maximum native-air table index and finite mesh AABB. Numerical deposition still requires errors=out_of_window=0; no clipping accepted.',
        air_max_index=nmax, maximum_injection_observer_distance_m=distance, mesh_aabb_diagonal_m=mesh_diameter,
        resident_moment_GiB=3*grid_bytes/1024**3, consuming_download_moment_peak_GiB=4*grid_bytes/1024**3,
        previous_nonconsuming_download_moment_peak_GiB=6*grid_bytes/1024**3,
        minimum_available_memory_GiB=212, minimum_free_disk_GiB=220,
        unchanged='Transport seeds, cuts, thinning, 130 selected CPUs, 80 station coordinates, propagation equations and moment order; only future radio sampling/subdivision/output memory lifecycle change.',
        validation=verdict,allocation_lifetime_validation=lifetime)
    save(REV/'admission.json',admission)
    save(REV/'after/campaign.json',m)
    note='等待当前不带射电的粗筛完整结束并验收；随后自动切换到已验证的 1 GHz 后续射电配置。当前进程不终止，不丢弃已有数据。'
    save(ROOT/'report/radio_alignment_pending.json',dict(state='ready_for_stage_boundary',revision=str(REV),note=note,admission=admission))
    (ROOT/'STOP_AFTER_CURRENT').write_text('Radio 1 GHz revision: automatic handoff after complete current stage; '+str(REV)+'\n')
    print(json.dumps(admission),flush=True)

def handoff():
    assert read(BUILD/'postprocess_acceptance.json')['passed']
    assert read(BUILD/'finalize_lifetime.json')['passed']
    assert sha(ROOT/'bundle/c8_terrain_cascade_radio_1GHz')==read(BUILD/'acceptance.json')['new_binary_sha256']
    save(REV/'handoff.json',dict(state='waiting_for_complete_stage',pid=os.getpid()))
    while True:
        p=read(ROOT/'progress.json')
        if p['state']=='stopped_after_current': break
        if p['state'] in ['failed','complete','candidate_limit_reached']:
            save(REV/'handoff.json',dict(state='not_applied',reason=p['state']));return
        time.sleep(30)
    lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    assert read(ROOT/'progress.json')['state']=='stopped_after_current'
    before=read(REV/'before/campaign.json')
    assert sha(ROOT/'campaign.json')==sha(REV/'before/campaign.json')
    assert (ROOT/'STOP_AFTER_CURRENT').read_text().startswith('Radio 1 GHz revision:')
    for relative in ['scenes/radio.yaml','code/run_psr.py','code/audit_psr.py','code/event_runner.py']:
        assert sha(ROOT/relative)==sha(REV/'before'/relative)
    # All preconditions are checked before touching live inputs.
    for relative in ['scenes/radio.yaml','code/run_psr.py','code/audit_psr.py','code/event_runner.py']:
        shutil.copy2(REV/'after'/relative,ROOT/relative)
    m=read(REV/'after/campaign.json')
    m['radio_revision']=read(REV/'admission.json')
    m['binary_sha256_by_stage']={stage:(m['radio_revision']['validation']['new_binary_sha256'] if stage=='radio' else m['binary_sha256']) for stage in ['coarse','confirm','radio']}
    for row in m['files']:row['sha256']=sha(Path(row['path']))
    binary=ROOT/'bundle/c8_terrain_cascade_radio_1GHz'
    m['files'].append(dict(path=str(binary),sha256=sha(binary)))
    save(ROOT/'campaign.json',m)
    report=ROOT/'report/radio_alignment';report.mkdir(exist_ok=True)
    shutil.copy2(REV/'admission.json',report/'acceptance.json')
    shutil.copy2(BUILD/'acceptance.json',report/'kernel_equivalence.json')
    shutil.copy2(BUILD/'postprocess_acceptance.json',report/'postprocess_acceptance.json')
    shutil.copy2(REV/'README_CN.md',report/'README_CN.md')
    save(ROOT/'report/radio_alignment_pending.json',dict(state='applied',revision=str(REV),admission=m['radio_revision']))
    (ROOT/'STOP_AFTER_CURRENT').unlink()
    p=read(ROOT/'progress.json');p.update(state='resuming_with_1GHz_radio',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(ROOT/'progress.json',p)
    save(REV/'handoff.json',dict(state='applied_and_resuming',pid=os.getpid(),completed_stages=len(p['stages'])))
    (ROOT/'driver.pid').write_text(str(os.getpid())+'\n')
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    os.execv(sys.executable,[sys.executable,str(ROOT/'code/run_psr.py'),'--root',str(ROOT)])

if __name__=='__main__':
    if '--handoff' in sys.argv:handoff()
    else:prepare()
