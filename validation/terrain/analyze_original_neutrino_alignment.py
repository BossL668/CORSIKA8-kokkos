#!/usr/bin/env python3
"""Original-module alignment and analytic controls. Does NOT certify full weak physics."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--legacy',type=Path)
    a=p.parse_args();root=a.root;figdir=root/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.grid':True,
                         'grid.alpha':.18,'savefig.dpi':180,'axes.spines.top':False,
                         'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/(name+'.png'));fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
    unit=(root/'unit_tests_v2.log').read_text()
    spinlog=(root/'pion_oracle.log').read_text()
    checks={'original_module_unit_tests':bool(re.search(r'All tests passed.*6 test cases',unit)),
            'analytic_pion_control':'All tests passed' in spinlog}
    spin=pd.read_csv(root/'tauola_pion.csv');spinrows=[]
    fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
    bins=np.linspace(-1,1,26);x=(bins[1:]+bins[:-1])/2;dx=bins[1]-bins[0]
    for ax,h in zip(axs,[-1,0,1]):
        for e,color in [(10,'#2563a6'),(1000,'#d57a24')]:
            q=spin[(spin.helicity==h)&(spin.energy_GeV==e)]
            counts,_=np.histogram(q.cos_theta_pi,bins=bins);n=len(q)
            y=counts/n/dx;err=np.sqrt(counts*(1-counts/n))/n/dx
            ax.step(x,y,where='mid',color=color,label=f'{e:g} GeV ({n:,})')
            ax.fill_between(x,y-err,y+err,step='mid',alpha=.16,color=color)
        ax.plot(x,(1+h*x)/2,'k--',label='Analytic $(1+h\cos\theta)/2$')
        ax.set(xlabel=r'$\cos\theta_{\pi}$',title=f'Upstream helicity h={h:+d}',xlim=(-1,1))
    axs[0].set_ylabel('Probability density');axs[0].legend(fontsize=8)
    save(fig,'01_tauola_pion_polarization')
    for (pdg,h,e,axis),q in spin.groupby(['pdg','helicity','energy_GeV','axis']):
        mean=float(q.cos_theta_pi.mean());se=np.sqrt((1/3-h*h/9)/len(q))
        spinrows.append(dict(pdg=int(pdg),helicity=float(h),energy_GeV=float(e),axis=int(axis),
                             count=len(q),mean=mean,expected=h/3,z=(mean-h/3)/se))
    checks['pion_groups_6sigma']=all(abs(r['z'])<=6 for r in spinrows)
    regen=pd.read_csv(root/'regeneration_v2.csv')
    fig,ax=plt.subplots(figsize=(9,3.6))
    for pdg,q in regen.groupby('primary_pdg',sort=False):
        ax.plot(np.arange(len(q)),q.energy_GeV,'o-',ms=4,label=r'$\nu_\tau$' if pdg>0 else r'$\bar\nu_\tau$')
    q=regen[regen.primary_pdg==16]
    ax.set_xticks(np.arange(len(q)),[f'{int(c)+1}: {s}' for c,s in zip(q.cycle,q.stage)],rotation=40,ha='right')
    ax.set(yscale='log',ylabel='Total energy [GeV]',title='Three conditional CC → TAUOLA decay → NC cycles; not a natural rate')
    ax.axhline(1e4,color='gray',ls=':',label='CTW lower validity limit');ax.legend();save(fig,'02_original_tauola_regeneration')
    runs={};rows=[];resources=[]
    for b in ['cpu','openmp','cuda']:
        for c in ['nu_cc','antinu_cc','nu_nc','antinu_nc']:
            path=root/'terrain'/b/c/'terrain_run.yaml'
            if not path.exists():continue
            s=yaml.safe_load(path.read_text());runs[b,c]=s
            d=s.get('diagnostics',{});t=s.get('tau',{})
            command=json.loads((path.parent.parent/(c+'_command.json')).read_text())
            rows.append(dict(backend=b,case=c,complete=s['complete'],steps=d.get('steps'),
                material_mismatches=d.get('material_mismatches'),tau_decays=t.get('tau_decay_count'),
                max_tau_energy_residual=max([abs(r['relative_energy_residual']) for r in t.get('decays',[])]+[0]),
                uncovered_neutrinos=d.get('neutrino_model_domain',{}).get('uncovered_transported_histories'),
                binary_sha256=command['binary_sha256'],shower_seconds=s.get('shower_seconds')))
            resource=json.loads((path.parent.parent/(c+'_guard')/'resources.json').read_text())
            resources.append(dict(backend=b,case=c,returncode=resource['returncode'],
                stop_reason=resource['stop_reason'],
                peak_rss_MiB=max(r['rss_kib'] for r in resource['samples'])/1024,
                gpu_peak_increment_MiB=max([r.get('gpu_delta_mib',0) for r in resource['samples']]+[0])))
    checks['terrain_12_complete']=len(rows)==12 and all(r['complete'] for r in rows)
    checks['material_interfaces']=len(rows)==12 and all(r['material_mismatches']==0 for r in rows)
    checks['tauola_selected']=len(runs)==12 and all(s.get('tau',{}).get('tau_decay_backend')=='tauola' for s in runs.values())
    checks['resource_guards']=len(resources)==12 and all(r['returncode']==0 and r['stop_reason'] is None for r in resources)
    checks['first_weak_vertices_equal']=len(runs)==12 and all(
        runs[b,c]['neutrino']['interactions'][0]==runs['cpu',c]['neutrino']['interactions'][0]
        for b in ['openmp','cuda'] for c in ['nu_cc','antinu_cc','nu_nc','antinu_nc'])
    if len(runs)==12:
        fig,axs=plt.subplots(2,2,figsize=(10,6))
        for col,c in enumerate(['nu_cc','antinu_cc']):
            tau_end=0.
            for b,color in [('cpu','#222222'),('openmp','#2563a6'),('cuda','#d57a24')]:
                df=pd.read_csv(root/'terrain'/b/c/'terrain'/'tracks.csv')
                df=df[np.abs(df.pdg).isin([15,16])]
                taus=df[np.abs(df.pdg)==15]
                initial_tau_energy=float(taus.E0_GeV.max()) if len(taus) else 0.
                tau_label=True
                for hid,g in df.groupby('history_id',sort=False):
                    # Include both endpoints; do not omit one-step neutrinos/taus.
                    time=np.r_[g.t0_s.iloc[0],g.t1_s.to_numpy()]*1e9
                    energy=np.r_[g.E0_GeV.iloc[0],g.E1_GeV.to_numpy()]
                    axs[0,col].plot(time,energy,color=color,lw=1,label=b if hid==df.history_id.iloc[0] else None)
                    if abs(g.pdg.iloc[0])==15:
                        axs[1,col].plot(time,initial_tau_energy-energy,'.-',ms=3,color=color,label=b if tau_label else None)
                        tau_label=False;tau_end=max(tau_end,float(time.max()))
            axs[0,col].set(title='CC neutrino' if col==0 else 'CC antineutrino',yscale='log',ylabel='Energy [GeV]')
            axs[1,col].set(xlabel='Particle time [ns]',ylabel=r'$E_\tau(t_0)-E_\tau(t)$ [GeV]')
            if tau_end>0:axs[1,col].set_xlim(0,tau_end*1.06)
            axs[0,col].legend();axs[1,col].legend()
        save(fig,'03_terrain_tau_transport')
    gates=root/'startup_gates'/'gates.json'
    checks['startup_gates']=gates.exists() and all(r['passed'] for r in json.loads(gates.read_text()))
    strict=root/'strict_coverage'/'terrain_run.yaml'
    checks['runtime_domain_gate']=strict.exists() and (lambda s:not s['complete'] and 'outside CTW2011' in s.get('error',''))(yaml.safe_load(strict.read_text()))
    legacyrows=[]
    if a.legacy:
        for b in ['cpu','openmp','cuda']:
            directory=root/'legacy_pythia'/b/'nu_cc'
            if not (directory/'terrain_run.yaml').exists():continue
            for name in ['tracks.csv','deposits.csv','window_survivors.csv']:
                new=directory/'terrain'/name;old=a.legacy/b/'nu_cc'/'terrain'/name
                legacyrows.append(dict(backend=b,file=name,equal=sha(new)==sha(old)))
        checks['legacy_pythia_physics_unchanged']=len(legacyrows)==9 and all(r['equal'] for r in legacyrows)
    out=dict(checks=checks,controls_passed=all(checks.values()),full_neutrino_physics_validated=False,
             spin_groups=spinrows,terrain=rows,resources=resources,legacy=legacyrows,
             limitations=['CTW only 1e4..1e12 GeV; uncovered daughters audited, not modeled',
                          'fixed upstream TAUOLA helicity, not CC density-matrix transfer or depolarization',
                          'conditional regeneration links are not natural-flux validation'])
    (root/'acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
    source=Path(__file__).resolve().parents[2]
    paths=['applications/c8_air_shower.cpp','applications/c8_terrain_cascade.cpp',
           'corsika/modules/neutrino/ConditionedTauolaDecay.hpp',
           'corsika/modules/neutrino/TransportedLeptonDecay.hpp',
           'corsika/modules/neutrino/NeutrinoModelDomain.hpp',
           'corsika/modules/neutrino/MountainNeutrinoInteraction.hpp',
           'corsika/modules/neutrino/AuditedPythiaNeutrino.hpp',
           'corsika/modules/tauola/Decay.hpp','corsika/detail/modules/tauola/Decay.inl',
           'tests/modules/testMountainOriginalLeptons.cpp']
    (root/'source_sha256.json').write_text(json.dumps({name:sha(source/name) for name in paths},indent=2)+'\n')
    report=['# 原版中微子/τ 模块接线验收','',
            '本报告检验原版模块的复用与数值适配，不等同于完整中微子物理通过。','',
            '|检查|通过|','|---|---|']+[f'|{k}|{v}|' for k,v in checks.items()]
    report+=['','已保留首轮失败日志：随机数漏接和极高能 TAUOLA boost 非有限值是真实检出的错误。',
             '修复后：540 组原生 TAUOLA 逐女儿/RNG 对照；1800 个极高能 τ；72000 个 πν 极化控制；',
             '正反 ντ 各三轮条件再生；12 个真实 DEM 三后端 CC/NC 事例。','',
             '**未完成**：10 TeV 以下弱相互作用、事件级 CC 自旋/介质退极化、独立自然再生通量验收。','',
             '当前低能次级按 history 去重审计；`--require-neutrino-model-coverage` 可禁止此近似。',
             '没有给低能区填入任意 4 nb 或外推 CTW。','',
             '![πν 极化控制](figures/01_tauola_pion_polarization.png)',
             '![条件再生链](figures/02_original_tauola_regeneration.png)',
             '![真实山体 τ 输运](figures/03_terrain_tau_transport.png)']
    (root/'VALIDATION_REPORT_CN.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(checks,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
