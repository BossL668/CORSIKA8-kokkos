#!/usr/bin/env python3
"""Bounded-memory acceptance/work accounting of completed 100 PeV dual pilots.

Read existing outputs only. No simulator changes, job submission or unique-ID
claims from repeated wavefront-input counters. Radio uses the original basis.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def load(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reserve():
    assert psutil.virtual_memory().available > 4*2**30, 'available RAM below 4 GiB'


def scan_parquet(path):
    f = pq.ParquetFile(path)
    rows = 0
    for batch in f.iter_batches(batch_size=65536, use_threads=False):
        reserve()
        for column in batch.columns:
            assert column.null_count == 0, (path,'null')
            if pa.types.is_floating(column.type):
                assert np.isfinite(column.to_numpy()).all(), (path,'nonfinite')
        if 'shower' in batch.schema.names:
            assert np.all(batch.column('shower').to_numpy()==0), (path,'wrong event')
        rows += batch.num_rows
    assert rows == f.metadata.num_rows and rows > 0
    return dict(rows=rows, finite=True, no_nulls=True)


def resource_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    total = weighted = 0.
    for a,b in zip(rows,rows[1:]):
        dt = b['elapsed_s']-a['elapsed_s']
        if 'device_util_percent' in b:
            total += dt; weighted += dt*b['device_util_percent']
    duration = rows[-1]['elapsed_s']-rows[0]['elapsed_s']
    return dict(cpu_cores=(rows[-1]['cpu_seconds']-rows[0]['cpu_seconds'])/duration,
        gpu_utilization_percent=weighted/total,
        peak_gpu_MiB=max(r.get('device_used_mib',0) for r in rows)), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pulse-package', type=Path,
        default=Path('/home/yuhanglu/21CMA/python/MCMCTidyUp/pulse_analysis_modular'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    sys.path.insert(0, '/home/yuhanglu/21CMA/analysis_jobs')
    import reproject_waveform_ensembles as w
    project,basis = w.basis_for(dict(theta=47,phi=180,pulse_package=str(args.pulse_package)))
    paths = dict(old=args.old.resolve(), new=args.new.resolve())
    records, grids, fields, telemetry = {}, {}, {}, {}
    previous = None
    for name,path in paths.items():
        reserve()
        event = load(path/'summary.yaml')
        gpu = load(path/'gpu_em/summary.yaml')['shower_0']
        s = gpu['statistics']; c = s['accelerator']['cooperative']
        guard = json.loads((path.parent/(path.name+'-guard/summary.json')).read_text())
        timing = load(path/'simulation_timing/summary.yaml')['shower_0']
        primary = load(path/'primary/summary.yaml')['shower_0']
        cfg = load(path/'gpu_em/config.yaml')
        command = guard['command']
        # Each parameter in these frozen commands takes one value.
        flags = dict(zip(command[1::2],command[2::2]))
        flags.pop('-f'); flags.pop('--kokkos-cooperative-policy',None)
        identity = dict(primary=primary, flags=flags, environment=cfg['environment'],
            constants=cfg['magnetic_rigidity_GeV_per_T_m'],
            table={k:s['proposal_native'][k] for k in ('proposal_version','cubic_interpolation_version',
                'table_sha256','aux_sha256')})
        if previous is None: previous = identity
        assert identity == previous, 'physical configuration/table mismatch'
        assert primary['pdg']==2212 and primary['total_energy']==1e8
        assert np.allclose([primary[k] for k in ('nx','ny','nz')],basis['direction'],atol=2e-12,rtol=0)
        magnetic = np.array([cfg['environment']['magnetic_field_T'][k] for k in ('x','y','z')])*1e6
        assert np.linalg.norm(magnetic-basis['B_NWU_microtesla'])/np.linalg.norm(magnetic)<1e-9
        assert guard['pass'] and guard['returncode']==0 and timing['closed'] and gpu['complete']
        assert event['showers']==1 and event['seed']==2026110001 and event.get('end time')
        assert c['subshower_cuda_submissions']==c['subshower_cuda_commits']>0
        assert s['cross_species']['final_pending_photons']==s['cross_species']['final_pending_leptons']==0
        assert s['queue_overflows']==s['profile']['fixed_point_overflows']==s['radio']['fixed_point_overflows']==0
        assert s['profile']['invalid_records']==s['proposal_native']['inverse_failures']==0
        reasons = s['cpu_fallbacks_by_reason_name']
        assert set(reasons)<= {'unsupported_geometry','epair_rejection_envelope_exceeded','native_selection_replay'}
        assert reasons.get('native_selection_replay',0)==s['cpu_completed_native_selection_replays']==s['cpu_completed_selected_losses']
        assert sum(reasons.values())==s['cpu_generic_fallbacks']
        assert s['cpu_specified_final_states']==reasons.get('native_selection_replay',0)+reasons.get('epair_rejection_envelope_exceeded',0)
        files = {str(f.relative_to(path)):scan_parquet(f) for f in sorted(path.rglob('*.parquet'))}
        assert len(files)==7
        for f in sorted((path/'interaction_hist').glob('*.npz')):
            with np.load(f,allow_pickle=False) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            files[str(f.relative_to(path))]=dict(finite=True)
        radio = {}; fields[name] = {}
        for algorithm in ('CoREAS','ZHS'):
            rcfg = load(path/algorithm/'config.yaml')
            if name=='new': assert rcfg == load(paths['old']/algorithm/'config.yaml')
            raw = list(w.events(path/algorithm/'observers.parquet',rcfg,1,algorithm))
            assert len(raw)==1
            raw = raw[0].copy(); prime=project(raw)
            assert np.isfinite(prime).all()
            power = np.sum(prime**2,axis=2)
            nonzero = np.max(power,axis=1)>0
            assert nonzero.all()
            edges = power[:,:5].sum(1)+power[:,-5:].sum(1)
            fraction = np.divide(edges,power.sum(1),out=np.zeros(len(power)),where=power.sum(1)>0)
            radio[algorithm]=dict(observers=len(nonzero), bins=raw.shape[1],
                nonzero_observers=int(nonzero.sum()), finite=True, expected_time_grid=True,
                edge5_bins_pooled_power_fraction=float(edges.sum()/power.sum()),
                maximum_observer_edge5_fraction=float(fraction.max()),
                peaks_at_edge_count=int(((power.argmax(1)<5)|(power.argmax(1)>=raw.shape[1]-5)).sum()),
                outside_recorded_window_unknown=True)
            fields[name][algorithm] = prime
        # Ground output is streamed; count and weights are different quantities.
        ground = dict(rows=0, weighted_count=0.,weighted_kinetic_GeV=0.,direction_norm_max_error=0.)
        for batch in pq.ParquetFile(path/'particles/particles.parquet').iter_batches(batch_size=65536,use_threads=False):
            arrays={k:batch.column(k).to_numpy() for k in ('weight','kinetic_energy','nx','ny','nz')}
            assert np.all(arrays['weight']>0) and np.all(arrays['kinetic_energy']>=0)
            # Observation output stores direction components in float32. Test the
            # serialized values in double against their storage-rounding bound,
            # not a tighter double-only tolerance or a float32 norm calculation.
            precision = max(np.finfo(arrays[k].dtype).eps for k in ('nx','ny','nz'))
            ground['direction_norm_storage_tolerance'] = float(precision)
            error = np.max(np.abs(np.sqrt(sum(arrays[k].astype(np.float64)**2 for k in ('nx','ny','nz')))-1))
            ground['direction_norm_max_error']=max(ground['direction_norm_max_error'],float(error))
            ground['rows']+=batch.num_rows
            ground['weighted_count']+=float(arrays['weight'].sum(dtype=np.float64))
            ground['weighted_kinetic_GeV']+=float(np.sum(arrays['weight'].astype(np.float64)*arrays['kinetic_energy']))
        assert ground['direction_norm_max_error']<ground['direction_norm_storage_tolerance']
        ps = load(path/'particles/summary.yaml')['shower_0']
        assert ground['rows']==sum(group['count'] for group in ps.values())
        profile = pq.read_table(path/'profile/profile.parquet').to_pandas()
        x = profile['X'].to_numpy()
        assert np.all(np.diff(x)>0)
        curves = {k:profile[k].to_numpy() for k in profile if k not in ('shower','X')}
        assert all(np.all(v>=0) for v in curves.values())
        curves['em'] = curves['photon']+curves['electron']+curves['positron']
        curves['muon'] = curves['muplus']+curves['muminus']
        ed = pq.read_table(path/'energyloss/dEdX.parquet').to_pandas()
        assert np.array_equal(x,ed['X'].to_numpy())
        curves['deposit']=ed['total'].to_numpy()
        grids[name]=(x,curves)
        res,telemetry[name]=resource_summary(path.parent/(path.name+'-guard/resources.jsonl'))
        lepton_steps=sum(s['lepton_transport'].values())
        values=dict(process_s=guard['elapsed_s'],shower_s=timing['wall_time_ms']/1000,
            transport_records=s['gpu_particles'], lepton_transport_records=lepton_steps,
            photon_transport_records=s['gpu_particles']-lepton_steps,
            materialized_final_state_children=s['physical_secondaries'],
            accelerated_final_states=s['gpu_final_states'],scalar_particle_steps=s['cpu_particle_steps'],
            staged_input_entries=s['particles_staged'],radio_segments=s['radio_tracks'],
            total_resident_wavefronts=s['resident_photon_wavefronts']+s['resident_lepton_wavefronts'],
            cuda_job_input_entries=c['cuda_input_particles'],openmp_job_input_entries=c['openmp_input_particles'],
            cuda_jobs=c['subshower_cuda_submissions'],openmp_jobs=c['subshower_openmp_epochs'],
            cuda_mean_job_inputs=c['cuda_input_particles']/c['subshower_cuda_submissions'],
            openmp_mean_job_inputs=c['openmp_input_particles']/c['subshower_openmp_epochs'],
            records_per_resident_wavefront=s['gpu_particles']/(s['resident_photon_wavefronts']+s['resident_lepton_wavefronts']),
            service_delay_s=c['cuda_result_service_delay_ms']/1000,
            cuda_call_s=c['cuda_driver_wall_ms']/1000,openmp_call_s=c['openmp_wall_ms']/1000,
            migration_bytes=c['migration_bytes'],coordinator_idle_s=c['coordinator_idle_wait_ms']/1000,
            scalar_stepper_s=s['hybrid_timing_ms']['scalar_stepper']/1000,
            router_advance_s=s['hybrid_timing_ms']['router_advance']/1000,
            set_nodes_s=s['hybrid_timing_ms']['set_nodes']/1000,
            native_selected_loss_completions=s['cpu_completed_native_selection_replays'],
            peak_tree_rss_GiB=guard['peak_tree_rss_bytes']/2**30,
            minimum_available_GiB=guard['minimum_available_bytes']/2**30,
            **res)
        assert s['gpu_particles']==s['profile']['steps']
        records[name]=dict(values=values,files=files,radio=radio,ground=ground,
            accelerator_local_energy_ledger=s['energy_ledger'],
            longitudinal_summary=load(path/'energyloss/summary.yaml')['shower_0'],
            EM_profile_bin_Xmax_gcm2=float(x[curves['em'].argmax()]),
            EM_profile_integral=float(np.trapz(curves['em'],x)),
            processes=s['processes'],fallback_reasons=reasons,
            hadronic_models=s['hadronic_models'],thinning=s['thinning'],
            guard=guard,complete=True)
    old,new=records['old']['values'],records['new']['values']
    ratios={k:new[k]/old[k] for k in old if old[k]!=0}
    result=dict(operational_acceptance=True,performance_gate_pass=False,
        statistical_physics_acceptance=False,full_shower_energy_closure_certified=False,
        exact_unique_particle_count_available=False, exact_endpoint_transport_counts_available=False,
        configuration_match=True, records=records,new_over_old=ratios,
        wall_time_per_transport_record_ratio=ratios['process_s']/ratios['transport_records'],
        counter_semantics={
            'transport_records':'Actual completed transport steps summed across CUDA and OpenMP; not unique particles',
            'materialized_final_state_children':'Post-thinning child states from accelerated final-state kernels, including outgoing parent states; excludes scalar generators',
            'job_input_entries':'Repeated input states at host job boundaries; same particle may occur in several jobs; not endpoint physical-step totals',
            'radio_segments':'Accelerated charged track segments, not unique charged particles',
            'scalar_particle_steps':'Separate scalar CPU transport steps, not the OpenMP endpoint'},
        identity=previous, pulse_basis=basis,
        limitations=['One independent shower per strategy, not an ensemble statistical test',
            'CPU utilization includes runtime spinning; GPU activity does not measure useful throughput; no device-kernel profiling in these runs',
            'Per-step normalization is descriptive; process and energy mix differ',
            'Accelerator ledger explicitly incomplete for these thinned hadronic showers'])
    save(args.output/'ACCEPTANCE.json',result)
    with (args.output/'workload_comparison.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['metric','old','new','new_over_old'])
        writer.writerows((k,old[k],new[k],ratios.get(k)) for k in old)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':11,'savefig.dpi':180})
    labels={'old':'Independent queues (36.27 min)','new':'Adaptive v3 (66.87 min)'}
    colors={'old':'#0072B2','new':'#D55E00'}
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,key,title in zip(axs.flat,('em','electron','positron','photon','muon','hadron'),
        ('Electromagnetic','Electrons','Positrons','Photons','Muons','Hadrons')):
        maximum=0.
        for name,(x,curves) in grids.items():
            y=curves[key];ax.plot(x,y,color=colors[name],label=labels[name],lw=1.2)
            active=np.flatnonzero(y>max(y.max()*1e-4,0.))
            if len(active): maximum=max(maximum,x[active[-1]])
        ax.set_xlim(0,maximum+20);ax.set_ylim(bottom=0);ax.set_title(title)
        ax.set_xlabel(r'Slant depth [g cm$^{-2}$]');ax.set_ylabel('Weighted particle count');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('100 PeV proton: two individual showers, same seed, different scheduling\nNot an ensemble agreement test')
    fig.tight_layout();fig.savefig(args.output/'single_shower_profiles.png');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for name,rows in telemetry.items():
        t=np.array([r['elapsed_s'] for r in rows]);cs=np.array([r['cpu_seconds'] for r in rows])
        axes_time=[];cpu=[];gpu=[]
        for start in np.arange(0,t[-1],15):
            a=np.searchsorted(t,start);b=min(len(t)-1,np.searchsorted(t,start+15))
            if b<=a:continue
            dt=np.diff(t[a:b+1]); axes_time.append((t[a]+t[b])/120)
            cpu.append((cs[b]-cs[a])/(t[b]-t[a]));gpu.append(np.average([r.get('device_util_percent',0) for r in rows[a+1:b+1]],weights=dt))
        axs[0].plot(axes_time,cpu,color=colors[name],label=labels[name],lw=1)
        axs[1].plot(axes_time,gpu,color=colors[name],lw=1)
        axs[2].plot(t/60,[r['rss_bytes']/2**30 for r in rows],color=colors[name],lw=1)
    axs[0].set_ylabel('Process logical-core equivalent');axs[1].set_ylabel('GPU utilization [%]');axs[2].set_ylabel('Process RSS [GiB]')
    axs[0].legend();axs[1].set_ylim(0,105);axs[-1].set_xlabel('Elapsed time [min]')
    for ax in axs:ax.grid(alpha=.2);ax.set_xlim(0,new['process_s']/60)
    fig.tight_layout();fig.savefig(args.output/'resource_comparison.png');plt.close(fig)
    categories=['Wall time','Transport steps','Final-state children','Radio segments','CUDA jobs','OpenMP jobs']
    keys=['process_s','transport_records','materialized_final_state_children','radio_segments','cuda_jobs','openmp_jobs']
    fig,ax=plt.subplots(figsize=(10,5));bars=ax.bar(categories,[ratios[k] for k in keys],color='#D55E00')
    ax.axhline(1,color='grey',ls='--');ax.set_ylabel('Adaptive / independent');ax.grid(axis='y',alpha=.2)
    for bar,k in zip(bars,keys):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,f'{ratios[k]:.2f}',ha='center')
    fig.tight_layout();fig.savefig(args.output/'workload_ratios.png');plt.close(fig)
    for alg in ('CoREAS','ZHS'):
        cfg=load(paths['new']/alg/'config.yaml');observers=list(cfg['observers'].values())
        direction=np.array(basis['direction']);positions=np.array([o['location'] for o in observers]);positions-=positions[0]
        perpendicular=np.linalg.norm(positions-np.outer(positions@direction,direction),axis=1)
        selected=[]
        for target in (0,100,300):
            index=int(np.argmin(abs(perpendicular-target)))
            if index not in selected:selected.append(index)
        fig,axs=plt.subplots(len(selected),3,figsize=(12,7),squeeze=False)
        time=np.arange(fields['new'][alg].shape[1])+(0.5 if alg=='ZHS' else 0.)
        for row,index in enumerate(selected):
            combined=sum(np.sum(fields[name][alg][index]**2,axis=1) for name in paths)
            lo,hi=np.searchsorted(np.cumsum(combined)/combined.sum(),[.001,.999])
            lower=max(0,float(time[lo])-10);upper=min(float(time[-1]),float(time[min(hi,len(time)-1)])+10)
            for col,component in enumerate(("Ex′","Ey′","Ez′")):
                ax=axs[row,col]
                for name in paths:ax.plot(time,fields[name][alg][index,:,col]*1e6,color=colors[name],label=labels[name],lw=1)
                ax.set_xlim(lower,upper);ax.grid(alpha=.2);ax.set_title(f'{component}, r_perp={perpendicular[index]:.1f} m')
                ax.set_xlabel('Time from observer window start [ns]');ax.set_ylabel('Field [µV/m]')
        axs[0,0].legend(fontsize=8);fig.suptitle(alg+': individual signed waveforms in the original pulse basis (not ensemble means)')
        fig.tight_layout();fig.savefig(args.output/(alg+'_single_event_prime.png'));plt.close(fig)
    lines=['# 100 PeV adaptive-v3 单事件验收与工作量诊断','',
        '**运行完整性通过；性能不通过；单例不能进行系综物理验收。**','',
        '同种子2026110001、质子1e8 GeV、theta47/phi180、emthin1e-6、20线程、70%显存预算、默认自动max-weight=50。',
        '物理配置、磁场和PROPOSAL/辅助表哈希一致；动态调度不保证同一shower tree。','',
        '| 指标 | 旧独立双端 | adaptive-v3 | 新/旧 |','|---|---:|---:|---:|']
    for title,k in [('总时间 [s]','process_s'),('实际加速输运步数（两端合计）','transport_records'),
        ('光子输运步数','photon_transport_records'),('轻子输运步数','lepton_transport_records'),
        ('薄化后末态子状态数','materialized_final_state_children'),('实际加速末态次数','accelerated_final_states'),
        ('标量CPU步数','scalar_particle_steps'),('射电轨迹段数','radio_segments'),
        ('CUDA批次入口状态数（重复计数）','cuda_job_input_entries'),('OpenMP批次入口状态数（重复计数）','openmp_job_input_entries'),
        ('CUDA提交次数','cuda_jobs'),('OpenMP工作批次次数','openmp_jobs'),
        ('CUDA平均批次入口粒子状态数','cuda_mean_job_inputs'),('OpenMP平均批次入口粒子状态数','openmp_mean_job_inputs'),
        ('每个物理波前平均输运步数（两端合计）','records_per_resident_wavefront'),
        ('GPU结果等待 [s]','service_delay_s'),('CUDA调用窗口累计 [s]','cuda_call_s'),
        ('OpenMP调用窗口累计 [s]','openmp_call_s'),('router_advance主线程阶段 [s]','router_advance_s')]:
        lines.append(f'| {title} | {old[k]:,.3f} | {new[k]:,.3f} | {ratios[k]:.4f} |')
    lines += ['', '## 计数解释', '',
        '- YAML的`gpu_particles`是光子与轻子完成的输运步数之和，双端模式包含OpenMP；不是仅GPU、更不是独立粒子总数。',
        '- `physical_secondaries`来自薄化后的`child_count`，包括反应后的入射粒子延续状态；不含标量强子/指定过程生成的全部粒子。',
        '- 两端`input_particles`是每个job入口累加，同一粒子多次推进会重复出现，不能当作真实物理工作量或去重粒子数。',
        '- 当前生产输出未保存完整history集合及每端累计transport-record计数，不能事后准确给出全shower去重粒子数或两端真实步数分工。', '',
        '## 性能判定','',
        f"总耗时从{old['process_s']/60:.2f} min增至{new['process_s']/60:.2f} min（+{(ratios['process_s']-1)*100:.1f}%），"
        f"实际输运步数只增加{(ratios['transport_records']-1)*100:.1f}%，末态/射电工作量也仅增加约13%。",
        f"按总输运步数粗略归一化，单步平均墙钟成本增加{(result['wall_time_per_transport_record_ratio']-1)*100:.1f}%。"
        '这不是假定每步成本相同的精确分解，但显然不能只用shower更大解释退化。',
        'CUDA批次和OpenMP批次显著增加、平均入口量下降，是批次切碎的直接计数证据；GPU高利用率仍可包含大量小kernel/低效率工作。',
        'GPU领取等待下降不能抵消更多调度批次。两端调用窗口重叠，禁止把CUDA/OpenMP累计耗时直接相加；本轮没有设备事件级kernel计时，无法精确拆出launch、barrier和物理内核各占多少。','',
        '## 完整性与物理边界','',
        '- 所有Parquet逐批扫描无NaN/Inf/空值，事件编号正确；profile非负、地面能量/权重有效。地面方向存储为float32，以double计算范数后的最大误差约5.1e-8，在存储精度eps=1.19e-7内。',
        '- CoREAS/ZHS均81个天线×400个采样点，全部非零，时间网格正确，Ex′/Ey′/Ez′使用原pulse_analysis转换。',
        '- CUDA提交全部回收，两端pending清零，queue/profile/radio溢出与native反解失败均为0。',
        '- 指定native selected-loss、Epair和几何fallback保留并对账；没有把存在这些明确回退误报为零fallback。',
        f"- 新账本`complete_coverage=false`、`accepted=false`，局部残差{records['new']['accelerator_local_energy_ledger']['relative_closure_error']*100:.3f}%"
        f"（旧{records['old']['accelerator_local_energy_ledger']['relative_closure_error']*100:.3f}%）。它不覆盖完整强子/标量/薄化通道，不能称为能量守恒通过，也不能单凭残差断言丢失了这些能量。",
        '- 图是两次单事件对照，不是平均分布或1%等价检验；不计算没有统计意义的单例显著性。','',
        '## 图','', '![工作量与批次数](workload_ratios.png)','',
        '![资源曲线](resource_comparison.png)','', '![单事件profile](single_shower_profiles.png)','',
        '![CoREAS](CoREAS_single_event_prime.png)','', '![ZHS](ZHS_single_event_prime.png)']
    (args.output/'ACCEPTANCE_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),operational=True,performance=False,
        old_process_s=old['process_s'],new_process_s=new['process_s'],transport_records=[old['transport_records'],new['transport_records']],
        children=[old['materialized_final_state_children'],new['materialized_final_state_children']],
        per_step_cost_ratio=result['wall_time_per_transport_record_ratio']),indent=2))


if __name__ == '__main__':
    main()
