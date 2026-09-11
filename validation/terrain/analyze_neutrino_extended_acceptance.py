#!/usr/bin/env python3
"""Plot measured NC/decay/stack controls and audit bounded terrain histories.

No event-rate interpretation for forced vertices; no full-physics pass flag.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=a.output;figures=root/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':True,
                         'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})
    pol=pd.read_csv(root/'polarization_v4.csv');rates=pd.read_csv(root/'rates_v4.csv')
    regen=pd.read_csv(root/'regeneration_v4.csv');branch=pd.read_csv(root/'branching_v4.csv')
    summary={'scope':'conditional vertices and prescribed longitudinal tau decay controls',
             'full_neutrino_physics_validated':False,
             'outstanding':['below-10-TeV CC/NC transport','CC tau spin density matrix',
                 'spin evolution/depolarization in matter','consistent inclusive/differential DIS model and low-Q2 validation',
                 'natural regeneration flux comparison against independent transport reference'],
             'local_tests':{},'integration':[]}
    summary['local_tests']['log_passed']='All tests passed' in (root/'extended_tests_v4.log').read_text()
    summary['local_tests']['legacy_CC_passed']='All tests passed' in (root/'legacy_tests_final.log').read_text()
    summary['local_tests']['channel_selection_draws']=int(rates.draws.sum())
    summary['local_tests']['pion_decays']=len(pol)
    summary['local_tests']['inclusive_decays']=int(branch.n.sum())
    summary['local_tests']['conditional_regeneration_cycles']=len(regen)//4
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for pdg,label,marker in [(16,r'$\nu_\tau$','o'),(-16,r'$\bar\nu_\tau$','s')]:
        q=rates[rates.pdg==pdg]
        axes[0].loglog(q.energy_GeV,q.cc_cm2,marker=marker,label=label+' CC')
        axes[0].loglog(q.energy_GeV,q.nc_cm2,marker=marker,ls='--',label=label+' NC')
        p=q.nc_probability.to_numpy();observed=q.nc_count/q.draws
        axes[1].errorbar(q.energy_GeV,observed-p,yerr=2*np.sqrt(p*(1-p)/q.draws),fmt=marker,label=label)
    axes[0].set(xlabel='Neutrino energy [GeV]',ylabel=r'$\sigma_N$ [cm$^2$]',title='CTW2011 fit (not a new cross-section oracle)')
    axes[1].set(xscale='log',xlabel='Neutrino energy [GeV]',ylabel='Observed NC fraction minus rate prediction',title=r'$10^6$ draws per point; $2\sigma$ binomial bars')
    axes[1].axhline(0,color='black',lw=.8)
    for ax in axes:ax.legend()
    save(fig,figures/'01_CC_NC_competition')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    bins=np.linspace(-1,1,26);x=np.linspace(-1,1,300)
    pol_summary=[]
    for col,pdg in enumerate([15,-15]):
        for P,color in [(-1,'tab:blue'),(0,'tab:gray'),(1,'tab:red')]:
            q=pol[(pol.pdg==pdg)&(pol.tau_minus_P==P)]
            counts,_=np.histogram(q.cos_theta_pi,bins);density=counts/len(q)/np.diff(bins)
            mids=(bins[1:]+bins[:-1])/2
            err=np.sqrt(counts*(1-counts/len(q)))/len(q)/np.diff(bins)
            axes[0,col].plot(mids,density,color=color,label=f'prescribed $P_{{\\tau^-}}={P}$')
            axes[0,col].fill_between(mids,density-err,density+err,color=color,alpha=.16)
            axes[0,col].plot(x,(1+P*x)/2,color=color,ls='--',lw=1)
            axes[1,col].hist(q.nu_energy_fraction,bins=30,density=True,histtype='step',color=color)
            mean=q.cos_theta_pi.mean();se=q.cos_theta_pi.std(ddof=1)/np.sqrt(len(q))
            pol_summary.append(dict(pdg=pdg,P=P,n=len(q),mean_cos=float(mean),expected=P/3.,standard_error=float(se),z=float((mean-P/3.)/se)))
        axes[0,col].set(xlabel=r'$\cos\theta_\pi$ in tau rest frame',ylabel='Probability density',title=r'$\tau^-\to\pi^-\nu_\tau$' if pdg==15 else r'$\tau^+\to\pi^+\bar\nu_\tau$')
        axes[0,col].legend(fontsize=8)
        axes[1,col].set(xlabel=r'$E_{\nu_\tau}/E_\tau$ (10 GeV tau)',ylabel='Probability density')
    fig.suptitle('Prescribed-spin decay oracle: dashed analytic density; shaded sampling SEM')
    save(fig,figures/'02_polarized_tau_pion_and_neutrino')
    summary['polarization']=pol_summary
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for pdg,label in [(16,r'$\nu_\tau$'),(-16,r'$\bar\nu_\tau$')]:
        q=regen[regen.primary_pdg==pdg]
        axes[0].semilogy(np.arange(len(q)),q.energy_GeV,'o-',label=label)
    axes[0].set_xticks(range(8),['in','CC $\\tau$','decay $\\nu$','NC $\\nu$']*2,rotation=30)
    axes[0].set(ylabel='Actual particle energy [GeV]',title='Two conditional regeneration cycles; no energy reset')
    axes[0].axhline(1e4,color='gray',ls=':',label='CTW lower validity edge');axes[0].legend()
    for offset,channel,color in [(-.08,'e','tab:blue'),(.08,'mu','tab:red')]:
        x=np.arange(len(branch))+offset;observed=branch[f'observed_{channel}_BR'];expected=branch[f'expected_{channel}_BR']
        axes[1].errorbar(x,observed-expected,yerr=2*np.sqrt(expected*(1-expected)/branch.n),fmt='o',color=color,label=channel)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(len(branch)),[f'{r.pdg:+g}, P={r.tau_minus_P:g}' for r in branch.itertuples()],rotation=35)
    axes[1].set(ylabel='Observed minus configured branching fraction',title='All decay channels enabled; 5000 decays per point')
    axes[1].legend();save(fig,figures/'03_regeneration_and_inclusive_tau_decays')
    event_data={}
    for backend in ['cpu','openmp','cuda']:
        for name in ['nu_cc','antinu_cc','nu_nc','antinu_nc']:
            path=root/backend/name;run=yaml.safe_load((path/'terrain_run.yaml').read_text())
            tr=pd.read_csv(path/'terrain/tracks.csv');dep=pd.read_csv(path/'terrain/deposits.csv')
            guard=json.loads((root/backend/(name+'_guard')/'resources.json').read_text())
            audit=run['neutrino']['interactions'][0]['vertex_audit']
            wanted=16 if name.startswith('nu_') else -16
            first=run['neutrino']['interactions'][0]
            charge_lepton=15 if wanted>0 else -15
            current=name.rsplit('_',1)[1].upper()
            checks=dict(complete=bool(run['complete']),resource_guard=guard['returncode']==0 and guard['stop_reason'] is None,
                material_correct=run['diagnostics']['material_mismatches']==0,
                all_tracks=not run['diagnostics']['csv_truncated'],all_deposits=not run['diagnostics']['deposition_csv_truncated'],
                radio_off=run['radio']=='disabled',correct_current=first['current']==current,
                outgoing_flavor=audit['selected_outgoing_lepton_pdg']==(charge_lepton if current=='CC' else wanted),
                charge_conserved=audit['initial_charge_e']==audit['final_charge_e'],
                vertex_P4=audit['conservation']['corsika_max_component_relative_residual']<1e-4,
                not_full_physics_claim=not run['neutrino']['full_neutrino_physics_validated'])
            q=tr[tr.pdg==wanted]
            # Outgoing NC neutrino must actually undergo propagation, not merely exist in a vertex list.
            if current=='NC':
                ids=[v['history_id'] for v in first['daughters'] if v['pdg']==wanted]
                checks['NC_neutrino_propagates']=bool(q.history_id.isin(ids).any())
            tau_ids=tr[tr.pdg.abs()==15].history_id.unique()
            endpoint=[]
            for decay in run['tau'].get('decays',[]):
                z=tr[tr.history_id==decay['history_id']]
                endpoint.append(float(np.linalg.norm(z[[f'{c}1_m' for c in 'xyz']].to_numpy()-decay['position_enu_m'],axis=1).min()) if len(z) else float('inf'))
                checks['tau_flavor']=checks.get('tau_flavor',True) and any(v['pdg']==(16 if decay['parent_pdg']==15 else -16) for v in decay['daughters'])
                checks['tau_P4']=checks.get('tau_P4',True) and max(abs(decay['relative_energy_residual']),abs(decay['relative_momentum_residual']))<1e-3
            if current=='CC':checks['tau_propagates_and_decays']=len(tau_ids)>0 and bool(endpoint) and max(endpoint)<1e-5
            checks={k:bool(v) for k,v in checks.items()}
            row=dict(backend=backend,case=name,passed=all(checks.values()),checks=checks,steps=len(tr),tau_decays=len(endpoint),
                     elapsed_s=guard['wall_seconds'],peak_rss_MiB=max((r.get('rss_kib',0)/1024 for r in guard['samples']),default=0),
                     peak_gpu_increment_MiB=max((r.get('gpu_delta_mib',0) for r in guard['samples']),default=0))
            # Explicitly expose the regenerated-neutrino part outside the CTW fit.
            children=[v for d in run['tau'].get('decays',[]) for v in d['daughters'] if abs(v['pdg']) in [12,14,16]]
            row['tau_neutrinos_below_10TeV']=sum(v['energy_GeV']<1e4 for v in children)
            summary['integration'].append(row);event_data[backend,name]=(run,tr,dep)
    # Compare the FIRST vertex only: after it, independent transport scheduling draws may diverge.
    summary['first_vertex_comparison']={}
    for name in ['nu_cc','antinu_cc','nu_nc','antinu_nc']:
        ref=event_data['cpu',name][0]['neutrino']['interactions'][0]['vertex_audit']
        summary['first_vertex_comparison'][name]={b:event_data[b,name][0]['neutrino']['interactions'][0]['vertex_audit']==ref for b in ['openmp','cuda']}
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    colors={'cpu':'tab:blue','openmp':'tab:orange','cuda':'tab:green'}
    for ax,name in zip(axes.flat,['nu_cc','antinu_cc','nu_nc','antinu_nc']):
        inset=ax.inset_axes([.55,.1,.4,.25]) if name.endswith('_cc') else None
        for backend in colors:
            run,tr,dep=event_data[backend,name];origin=np.array(run['position_m']);direction=np.array(run['direction'])
            q=tr[tr.pdg.abs().isin([15,16])]
            for pid,style in [(16,'-'),(15,'--')]:
                for j,(history,track) in enumerate(q[q.pdg.abs()==pid].groupby('history_id')):
                    start=(track[[f'{c}0_m' for c in 'xyz']].to_numpy()-origin)@direction
                    end=(track[[f'{c}1_m' for c in 'xyz']].to_numpy()-origin)@direction
                    sx=np.column_stack([start,end,np.full(len(track),np.nan)]).ravel()
                    sy=np.column_stack([track.E0_GeV,track.E1_GeV,np.full(len(track),np.nan)]).ravel()
                    ax.plot(sx,sy,style,color=colors[backend],lw=1,alpha=.8,
                            label=backend+(' $\\nu_\\tau$' if pid==16 else ' $\\tau$') if j==0 else None)
                    if pid==15 and inset is not None:
                        inset.plot(sx,float(track.E0_GeV.iloc[0])-sy,style,color=colors[backend],lw=1.2,marker='.',ms=3)
        ax.set(xlabel='Distance projected along injection direction [m]',ylabel='Total energy [GeV]',yscale='log',title=name+' (single conditional history)')
        ax.legend(fontsize=7,ncol=2,loc='upper right')
        if inset is not None:
            inset.set_title(r'$\tau$ energy loss [GeV]',fontsize=7)
            inset.tick_params(labelsize=6);inset.set_xlabel('m',fontsize=6)
    save(fig,figures/'04_actual_terrain_neutrino_tau_tracks')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    rebuild=[]
    for backend,name in [('cpu','nu_nc'),('openmp','nu_cc'),('cuda','nu_cc')]:
        new=root/'final_build_checks'/backend/name;old=root/backend/name
        if not (new/'terrain_run.yaml').exists():continue
        before=yaml.safe_load((old/'terrain_run.yaml').read_text())
        after=yaml.safe_load((new/'terrain_run.yaml').read_text())
        guard=json.loads((new.parent/(name+'_guard')/'resources.json').read_text())
        checks={k:before[k]==after[k] for k in ['diagnostics','neutrino','tau']}
        for f in ['tracks.csv','deposits.csv','window_survivors.csv']:
            checks[f]=digest(old/'terrain'/f)==digest(new/'terrain'/f)
        checks['complete']=bool(after['complete'])
        checks['resource_guard']=guard['returncode']==0 and guard['stop_reason'] is None
        rebuild.append(dict(backend=backend,case=name,passed=all(checks.values()),checks=checks))
    summary['final_rebuild_checks']=rebuild
    gates=root/'startup_gates_v2/gates.json'
    summary['startup_gates']=json.loads(gates.read_text()) if gates.exists() else []
    summary['local_and_integration_controls_passed']=bool(summary['local_tests']['log_passed'] and summary['local_tests']['legacy_CC_passed'] and all(x['passed'] for x in summary['integration']) and all(all(v.values()) for v in summary['first_vertex_comparison'].values()))
    summary['final_build_and_gate_checks_passed']=len(rebuild)==3 and all(x['passed'] for x in rebuild) and len(summary['startup_gates'])==7 and all(x['passed'] for x in summary['startup_gates'])
    (root/'extended_acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 中微子 CC/NC、再生链与极化控制验收','',
        '**这不是完整中微子物理的验收通过声明。** 当前实现仍缺低能弱反应、CC 自旋密度矩阵及退极化。','',
        f"本轮局部/集成控制通过：{summary['local_and_integration_controls_passed']}；完整物理通过：False。",'',
        '测试包括 600 万次通道竞争抽样、18 个 NC 顶点、6 万次 πν 极化衰变、3 万次全通道衰变，',
        '以及真实 SecondaryView 上的正反 ντ 各两轮条件 CC→τ→ντ→NC 连接。',
        '山体集成使用 IGRF14 空气场、岩内零磁场，12 个单事例；不计算射电。','',
        '|后端|事例|通过|轨迹段|τ 衰变|总耗时 s|新增显存 MiB|','|---|---|---|---:|---:|---:|---:|']
    for r in summary['integration']:lines.append(f"|{r['backend']}|{r['case']}|{r['passed']}|{r['steps']}|{r['tau_decays']}|{r['elapsed_s']:.1f}|{r['peak_gpu_increment_MiB']:.0f}|")
    lines+=['','时间包含初始化，且与其他测试/生产并行，不能作为加速比。','',
        '图 1：CC/NC 率及通道抽样；图 2：极化角分布和再生 ν 能量；',
        '图 3：两轮条件再生链及全通道分支比；图 4：山体中真实 ν/τ 轨迹。','',
        '同一首顶点不同后端对齐，不要求整棵 shower 树一致。指定极化不是从 CC 自动推导的极化。',
        '条件链没有飞行距离和自然概率含义；山体样本不足以测量多轮自然再生概率。','',
        f"最终重建复测及启动门禁通过：{summary['final_build_and_gate_checks_passed']}；三后端各补测一例并对比物理 CSV SHA-256、neutrino/tau/diagnostics；7 项启动拒绝测试。",'',
        '详细数值和失败项见 extended_acceptance.json。旧失败日志保留，带 CSV 的局部测试见 extended_tests_v4.log，最终构建再次复测日志为 final_extended_test.log。']
    (root/'VALIDATION_REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'controls_passed':summary['local_and_integration_controls_passed'],'full_physics':False,'integration_events':len(summary['integration'])}))
    if not summary['local_and_integration_controls_passed'] or not summary['final_build_and_gate_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
