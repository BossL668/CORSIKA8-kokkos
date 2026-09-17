#!/usr/bin/env python3
"""PSR-only diagnostics; plots use English and real oracle/shower outputs."""
import csv, hashlib, json, pathlib
from collections import Counter
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
REPORT=ROOT/'report'; FIG=REPORT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
labels=['silica','limestone','granite']
titles=['Silica (SiO2)', 'Limestone / calcite (CaCO3)',
        'Granite (H, C, O, Na, Mg, Al, Si, K, Ca, Fe)']
for backend in ['openmp','cuda']:
    if not (ROOT/('MATERIAL_RUNS_'+backend.upper()+'_PASSED')).exists(): raise RuntimeError('Incomplete '+backend)

result={'backends':{},'lpm':{},'cross_backend_table_identity':True}
result['materials']={label:yaml.safe_load((ROOT/(label+'_resolved.yaml')).read_text()) for label in labels}
result['fluka_target_comparison']=json.loads((ROOT/'fluka-validation.json').read_text())
if not result['fluka_target_comparison']['exact_output_identity']:
    raise RuntimeError('FLUKA material setup comparison failed')
result['cuda_residency']={}
grids={'interface_radio_moments','interface_radio_coreas_regularized_moments','interface_radio_zhs_moments'}
for label in labels:
    folder=ROOT/'runs'/('cuda_'+label+'_on')
    summary=yaml.safe_load((folder/'output/terrain_run.yaml').read_text())
    accelerator=summary['accelerator']; counts=Counter(); downloads=Counter(); allocations=set()
    for line in (folder/'cupti.jsonl').open():
        e=json.loads(line)
        if e['event']=='gpu_kernel':
            for key,pattern in [('transport','InterfaceStepKernel'),('CoREAS','CoreasEndpointKernel'),('ZHS','AccumulateKernel')]:
                if pattern in e['name']: counts[key]+=1
        if e['event']=='allocate' and e.get('space')=='Cuda': allocations.add(e.get('label'))
        if e['event']=='copy_begin' and e.get('src_label') in grids and e.get('dst_space')=='Host' and e.get('bytes',0)>0:
            downloads[e['src_label']]+=1
    checks=dict(cuda_execution=accelerator['execution_space']=='Cuda',
                resident_queue=accelerator['resident_capacity']>0 and accelerator['device_enqueued_particles']>0,
                gpu_transport=counts['transport']>0,gpu_coreas=counts['CoREAS']>0,gpu_zhs=counts['ZHS']>0,
                resident_particle_allocation='interface_resident_particles' in allocations,
                one_final_copy_per_radio_grid=all(downloads[g]==1 for g in grids))
    if not all(checks.values()): raise RuntimeError('CUDA material residency checks: '+str(checks))
    result['cuda_residency'][label]=dict(checks=checks,kernels=dict(counts),grid_downloads=dict(downloads),
        uploaded_particles=accelerator['cpu_uploaded_particles'],device_enqueued_particles=accelerator['device_enqueued_particles'],
        control_downloads=accelerator['control_downloads'],record_downloads=accelerator['record_downloads'])
for backend in ['openmp','cuda']:
    rows=json.loads((ROOT/('materials_'+backend+'.json')).read_text())
    if len(rows)!=len(labels): raise RuntimeError('Incomplete material inventory: '+backend)
    result['backends'][backend]=[dict(material=r['material'],radio_preserves_transport=r['radio_does_not_change_transport'],
        coreas_zhs_relative_l2=r['pair'][1]['checks']['coreas_zhs_relative_l2'],
        proposal_material=r['pair'][1]['proposal_material'],tables=r['pair'][1]['material_tables']) for r in rows]
for a,b in zip(result['backends']['openmp'],result['backends']['cuda']):
    if a['material']!=b['material'] or len(a['tables'])!=2 or len(b['tables'])!=2:
        raise RuntimeError('Material inventory differs by backend')
    for ta,tb in zip(a['tables'],b['tables']):
        if ta['hash']!=tb['hash'] or ta['auxiliary_hash']!=tb['auxiliary_hash']:
            result['cross_backend_table_identity']=False
if not result['cross_backend_table_identity']: raise RuntimeError('Material table identity differs by backend')

fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for label,title in zip(labels,titles):
    a=np.genfromtxt(str(ROOT/(label+'_lpm.csv')),delimiter=',',names=True,dtype=None,encoding='utf-8')
    maximum=float(np.max(a['relative_error'])); result['lpm'][label]=dict(points=len(a),maximum_relative_error=maximum)
    for ax,process in zip(axes,['pair','brems']):
        select=(a['process']==process)&(a['Z']==8)&(a['fraction']==.5)&(a['density_scale']==1.)
        data=a[select]; order=np.argsort(data['energy_MeV']); data=data[order]
        ax.semilogx(data['energy_MeV']*1.e6,data['native'],'o-',label=title,markerfacecolor='none',markersize=7)
        ax.semilogx(data['energy_MeV']*1.e6,data['portable'],'x',color='black',markersize=5)
        ax.set(xlabel='Incident energy [eV]',ylabel='LPM suppression factor',title=('Photon pair creation' if process=='pair' else 'Electron bremsstrahlung'))
        ax.grid(alpha=.25); ax.set_ylim(0,1.05)
axes[0].legend(fontsize=8)
fig.suptitle('Sampled O-16 target, energy fraction = 0.5; circles: native PROPOSAL, crosses: portable snapshots',fontsize=10)
fig.savefig(FIG/'material_lpm.png',dpi=170); plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(9,8),constrained_layout=True)
for ax,label,title in zip(axes,labels,titles):
    arrays=[]
    for algorithm in ['CoREAS','ZHS']:
        path=ROOT/'runs'/('cuda_'+label+'_on')/'output/radio'/algorithm/'field.csv'
        a=np.genfromtxt(str(path),delimiter=',',names=True); a=a[a['observer']==0]
        arrays.append(a)
    z=arrays[1]; components=['Ex_V_m','Ey_V_m','Ez_V_m']
    name=max(components,key=lambda k:np.max(np.abs(z[k])))
    peak=int(np.argmax(np.abs(z[name]))); centre=z['time_s'][peak]*1.e9
    for a,algorithm,style in zip(arrays,['CoREAS','ZHS'],['-','--']):
        ax.plot(a['time_s']*1.e9,a[name]*1.e6,style,label=algorithm,lw=1.2)
    ax.set(xlabel='Arrival time [ns]',ylabel=name[:2]+' [microvolt/m]',title=title+' | 10 GeV electron, CUDA resident queue')
    ax.set_xlim(centre-160,centre+300); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.suptitle('Raw simulation waveforms at diagnostic_overhead; no normalization or time shift',fontsize=11)
fig.savefig(FIG/'material_waveforms.png',dpi=170); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,3.8),constrained_layout=True)
distance=np.linspace(0,300,400)
for label,title in zip(labels,titles):
    model=yaml.safe_load((ROOT/(label+'_resolved.yaml')).read_text())
    length=model['radio']['field_attenuation_length_m']
    ax.semilogy(distance,np.exp(-distance/length),label=title+' (L_E=%.2f m)'%length)
ax.set(xlabel='Path length inside material [m]',ylabel='Electric-field attenuation E/E0',title='Selected constant attenuation models (not measured site curves)',ylim=(1.e-5,1))
ax.grid(alpha=.25); ax.legend(fontsize=8); fig.savefig(FIG/'material_attenuation.png',dpi=170); plt.close(fig)

statuses=[]
for p in sorted((ROOT/'runs').glob('*/status.json')):
    s=json.loads(p.read_text()); record={k:s[k] for k in ['tag','complete','returncode','wall_s','binary_sha256']}
    run=yaml.safe_load((p.parent/'output/terrain_run.yaml').read_text())
    record['energy_ledger_relative_residual']=run['energy_ledger']['unexplained_over_initial']
    record['material_mismatches']=run['diagnostics']['material_mismatches']
    record['finite_window_survivors']=run.get('accelerator',{}).get('finite_window_survivors')
    if abs(record['energy_ledger_relative_residual'])>=1.e-8 or record['material_mismatches']!=0:
        raise RuntimeError('Material transport audit failed: '+s['tag'])
    checks=p.parent/'checks.json'
    if checks.exists():
        record['checks']=json.loads(checks.read_text())['checks']
        if not all(record['checks'].values()): raise RuntimeError('Failed transport/radio check: '+s['tag'])
    elif s['tag']!='cpu_limestone': raise RuntimeError('Missing case checks: '+s['tag'])
    statuses.append(record)
if not all(s['complete'] and s['returncode']==0 for s in statuses): raise RuntimeError('Incomplete run in final inventory')
expected={backend+'_'+label+'_'+mode for backend in ['openmp','cuda'] for label in labels for mode in ['off','on','proton']}|{'cpu_limestone'}
if {s['tag'] for s in statuses}!=expected: raise RuntimeError('Final inventory must contain all 19 material runs')
result['runs']=statuses
result['warning']='Integration/oracle checks, not mineral sample validation or full shower convergence.'
(REPORT/'validation.json').write_text(json.dumps(result,indent=2))
for label in labels:
    (REPORT/(label+'_resolved.yaml')).write_bytes((ROOT/(label+'_resolved.yaml')).read_bytes())
print(json.dumps(result,indent=2))
