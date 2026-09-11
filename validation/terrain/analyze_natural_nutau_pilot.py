#!/usr/bin/env python3
"""Audit real unforced histories; never label selected seeds as unbiased trials."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from analyze_nutau_ppt_demo import save, xyz, segment_histogram, _terrain_interpolator


def sigma(energy, cc):
    # Diagnostic-only transcription of MountainNeutrinoInteraction.hpp CTW fit.
    l = np.log(np.log10(energy) + 1.826)
    return 10.**(-17.31 + (-6.406 if cc else -6.448)*l + 1.431*l*l
                  + (-17.91 if cc else -18.61)/l)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    root = a.root
    manifest = yaml.safe_load((root/'manifest.yaml').read_text())
    out = root/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':11,
                         'axes.grid':True, 'grid.alpha':.18,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'legend.frameon':False})
    entry = np.array(manifest['chord']['entry'])
    direction = np.array(manifest['direction'])
    length = manifest['chord']['length_m']
    scene = yaml.safe_load(Path(manifest['scene']).read_text())
    with np.load(Path(scene['geometry']['mesh_path']).parent/'terrain_grid.npz') as grid:
        surface = _terrain_interpolator(*(grid[k] for k in ('east_m','north_m','up_m')))
    records = []
    control_rock_lengths = []
    tau_mass, tau_lifetime = None, None
    for job in manifest['jobs']:
        status_path = root/'runs'/(job['name']+'_status.json')
        if not status_path.exists():
            records.append(dict(**job, state='interrupted_or_not_started' if (root/'STOPPED_CN.md').exists() else 'pending'))
            continue
        status = json.loads(status_path.read_text())
        if not status['complete'] or status['returncode']:
            records.append(dict(**job, state='failed', returncode=status['returncode']))
            continue
        run = root/'runs'/job['name']
        s = yaml.safe_load((run/'terrain_run.yaml').read_text())
        t = pd.read_csv(run/'terrain/tracks.csv', dtype={'history_id':'uint64','parent_history_id':'uint64'})
        dep = pd.read_csv(run/'terrain/deposits.csv')
        d = s['diagnostics']
        vertices = s['neutrino'].get('interactions', []) or []
        decays = s['tau'].get('decays', []) or []
        tau_mass = s['tau']['tau_mass_GeV']; tau_lifetime = s['tau']['tau_rest_lifetime_s']
        checks = dict(complete=s['complete'], unforced=not s['neutrino']['force_interaction_called'],
                      decay_unforced=not s['tau']['force_decay_called'],
                      natural_sampler=s['neutrino']['sampling_mode']=='natural_exponential_grammage',
                      no_material_mismatch=d['material_mismatches']==0,
                      no_truncation=not d['csv_truncated'] and not d['deposition_csv_truncated'],
                      rows=len(t)==d['steps'], queue_empty=s['accelerator']['pending_particles']==0,
                      openmp=s['accelerator']['execution_space']=='OpenMP', radio_off=s['radio']=='disabled',
                      increasing_time=bool((t.t1_s>=t.t0_s).all()),
                      window=bool((t.t1_s<=manifest['window_ns']*1e-9+1e-14).all()),
                      deposits=bool(np.isclose(dep.weighted_deposited_GeV.sum(), d['deposited_energy_GeV'], rtol=1e-10, atol=1e-8)))
        primary = t[t.history_id==1]
        rock = primary[primary.medium=='rock']
        rock_length = float(np.linalg.norm(xyz(rock,1)-xyz(rock,0), axis=1).sum())
        if not vertices:
            first_rock_length=float(np.linalg.norm(xyz(rock,1)[0]-xyz(rock,0)[0]))
            checks['first_chord_length']=bool(np.isclose(first_rock_length,length,rtol=0,atol=1e-5))
            checks['first_entry']=bool(np.allclose(xyz(rock,0)[0],entry,rtol=0,atol=1e-5))
            checks['first_exit']=bool(np.allclose(xyz(rock,1)[0],manifest['chord']['exit'],rtol=0,atol=1e-5))
            checks['flight_speed']=bool(np.allclose(np.linalg.norm(xyz(primary,1)-xyz(primary,0),axis=1),
                (primary.t1_s-primary.t0_s)*299792458.,rtol=1e-10,atol=1e-8))
            checks['transparent_energy'] = bool((primary.E0_GeV==job['energy_GeV']).all() and
                                                (primary.E1_GeV==job['energy_GeV']).all())
            checks['no_false_deposit'] = d['deposited_energy_GeV']==0
            checks['energy_budget']=bool(np.isclose(d['finite_window_escaped_total_GeV'],job['energy_GeV'],rtol=1e-12))
            checks['single_survivor']=d['finite_window_survivors']==1
            control_rock_lengths.append(rock_length)
            if len(control_rock_lengths)==1:
                fig,axes=plt.subplots(1,2,figsize=(12,4.5))
                distance=np.linspace(-50,length+200,1600)
                points=entry+distance[:,None]*direction
                height=surface(points[:,0],points[:,1])
                floor=min(float(np.nanmin(height)),entry[2])-20
                axes[0].fill_between(distance,floor,height,color='#be9760',alpha=.8,label='DEM rock section')
                axes[0].plot(distance,np.full_like(distance,entry[2]),'--',color='#12817a',label='Incident axis')
                axes[0].scatter([0,length],[entry[2]]*2,c='black',s=25,label='Entry / exit')
                axes[0].set(xlim=(-50,length+200),ylim=(floor,float(np.nanmax(height))+20),
                            xlabel='Distance from first entry [m]',ylabel='ENU Up [m]',title=f'First natural chord: {length:.2f} m')
                axes[0].legend(fontsize=8)
                s0=(xyz(primary,0)-entry)@direction;s1=(xyz(primary,1)-entry)@direction
                for start,end,medium in zip(s0,s1,primary.medium):
                    if medium=='rock':axes[1].axvspan(start/1000,end/1000,color='#be9760',alpha=.35)
                axes[1].plot(np.r_[s0[0],s1]/1000,np.r_[primary.E0_GeV.iloc[0],primary.E1_GeV]/job['energy_GeV'],c='#12817a')
                axes[1].set(xlim=(s0[0]/1000,s1[-1]/1000),ylim=(.95,1.05),xlabel='Distance from first entry [km]',
                            ylabel='Neutrino E / initial E',title=f'Actual trajectory: {rock_length:.2f} m total rock')
                save(fig,out,'natural_flight_geometry','Uninteracted control; shaded intervals are recorded rock segments. No shower or double bang is implied.')
        chains = []
        if 'predicted_depth_m' in job:
            checks['prefilter_reproduced']=bool(vertices and abs(float(np.dot(
                np.array(vertices[0]['position_enu_m'])-entry,direction))-job['predicted_depth_m'])<1e-3)
        for v in vertices:
            if v['current']!='CC': continue
            for daughter in v['daughters']:
                if abs(daughter['pdg'])!=15: continue
                for decay in decays:
                    if daughter['history_id']!=decay['history_id']: continue
                    dt = decay['time_ns']*1e-9-v['time_s']
                    separation = float(np.linalg.norm(np.array(decay['position_enu_m'])-v['position_enu_m']))
                    parent = t[t.history_id==v['history_id']]
                    nearest = int(np.argmin(np.linalg.norm(xyz(parent,1)-v['position_enu_m'],axis=1)))
                    cc_medium = str(parent.iloc[nearest].medium)
                    visible = sum(c['energy_GeV'] for c in decay['daughters']
                                  if abs(c['pdg']) not in [12,13,14,15,16])
                    first_visible = sum(c['energy_GeV'] for c in v['daughters']
                                        if abs(c['pdg']) not in [12,13,14,15,16])
                    chains.append(dict(tau_history_id=daughter['history_id'],
                        cc_position=v['position_enu_m'], decay_position=decay['position_enu_m'],
                        separation_m=separation, flight_ns=dt*1e9,
                        cc_medium=cc_medium, decay_medium=decay['medium'],
                        first_visible_energy_GeV=first_visible, decay_visible_energy_GeV=visible,
                        double_cascade_topology=(dt>0 and separation>0 and visible>0 and first_visible>0),
                        resolved_double_bang_certified=False))
                    checks['causal_tau'] = checks.get('causal_tau',True) and dt>0 and separation<=dt*299792458.*(1+1e-5)
        resource=json.loads((root/'runs'/(job['name']+'_guard')/'resources.json').read_text())
        row=dict(**job,state='complete',checks=checks,interactions=len(vertices),
                 CC=sum(v['current']=='CC' for v in vertices), NC=sum(v['current']=='NC' for v in vertices),
                 tau_decays=len(decays), chains=chains, primary_rock_length_m=rock_length,
                 steps=len(t), deposited_GeV=d['deposited_energy_GeV'],
                 wall_seconds=resource['wall_seconds'], shower_seconds=s['shower_seconds'])
        records.append(row)
        if vertices:
            fig, axes=plt.subplots(2,2,figsize=(13,8))
            extent=max(300.,max((np.dot(np.array(c['decay_position'])-entry,direction)+50 for c in chains),default=300.))
            axis=np.linspace(-50,extent,1000); points=entry+axis[:,None]*direction
            heights=surface(points[:,0],points[:,1])
            ax=axes[0,0]; ax.fill_between(axis,entry[2]-30,heights,color='#be9760',alpha=.7,label='Rock surface section')
            for pid,label,color in [(16,r'$\nu_\tau$','#12817a'),(15,r'$\tau$','#b835ab')]:
                part=t[t.pdg.abs()==pid]; start,end=xyz(part,0),xyz(part,1)
                seg=np.stack([np.c_[(start-entry)@direction,start[:,2]],np.c_[(end-entry)@direction,end[:,2]]],axis=1)
                ax.add_collection(LineCollection(seg,colors=color,lw=1.5));ax.plot([],[],c=color,label=label)
            ax.set(xlim=(-50,extent),ylim=(entry[2]-25,np.nanmax(heights)+10),
                   xlabel='Distance from first rock entry [m]',ylabel='ENU Up [m]',title='Recorded natural flight')
            ax.legend(fontsize=9)
            ax=axes[0,1]
            for pid,label,color in [(16,r'$\nu_\tau$','#12817a'),(15,r'$\tau$','#b835ab')]:
                part=t[t.pdg.abs()==pid]; start,end=xyz(part,0),xyz(part,1)
                seg=np.stack([np.c_[(start-entry)@direction,part.E0_GeV],np.c_[(end-entry)@direction,part.E1_GeV]],axis=1)
                ax.add_collection(LineCollection(seg,colors=color,lw=1.1));ax.plot([],[],c=color,label=label)
            ax.set(xlim=(-50,extent),ylim=(max(1.,job['energy_GeV']*1e-5),job['energy_GeV']*1.5),yscale='log',
                   xlabel='Distance from first rock entry [m]',ylabel='Total energy [GeV]',title='Neutrino / tau tracks, not a species sum')
            ax.legend(fontsize=9)
            positions=[vertices[0]['position_enu_m'],chains[0]['decay_position'] if chains else None]
            for ax,vertex,title in zip(axes[1],positions,['First weak vertex','Matched tau decay']):
                if vertex is None:
                    ax.text(.5,.5,'No matched tau decay',ha='center',transform=ax.transAxes);continue
                da=(xyz(dep,0)-vertex)@direction;db=(xyz(dep,1)-vertex)@direction
                edges=np.linspace(-20,80,201)
                hist=segment_histogram(da,db,dep.weighted_deposited_GeV.to_numpy(),edges)
                ax.stairs(hist/np.diff(edges),edges,c='#356ba5');ax.axvline(0,c='black',ls='--')
                ax.set(xlabel='Distance from vertex [m]',ylabel='Recorded deposition [GeV/m]',title=title+' (local window)')
            fig.suptitle(f"{job['energy_GeV']/1e6:g} PeV nu_tau, seed {job['seed']}: {job['group']}")
            save(fig,out,job['name'],'No forced vertex/decay. Deposits: segment-split total, not ancestry-separated; local windows may omit energy.')
        del t,dep;gc.collect()
    energy=np.array(sorted({j['energy_GeV'] for j in manifest['jobs']})); column=length*100*2.65
    probability=-np.expm1(-column/1.66053906660e-24*(sigma(energy,True)+sigma(energy,False)))
    model=[dict(energy_GeV=float(e), first_ridge_any_interaction_probability=float(p)) for e,p in zip(energy,probability)]
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.loglog(energy/1e6,probability,'o-',label=f'CTW CC+NC, first {length:.2f} m rock chord')
    ax.set(xlabel='Incident neutrino energy [PeV]',ylabel='First-interaction probability in this chord',
           title='Model expectation, not the selected-seed success fraction');ax.legend(fontsize=9)
    save(fig,out,'interaction_probability','First ridge only; free-nucleon CTW approximation. No forced interactions used in runs.')
    completed=[r for r in records if r['state']=='complete']
    result=dict(records=records,model_expectation=model,
                all_completed=len(completed)==len(records),
                completed_checks_pass=all(all(r['checks'].values()) for r in completed),
                unselected_completed=sum(r['group']=='unselected' for r in completed),
                unselected_interactions=sum(r['interactions'] for r in completed if r['group']=='unselected'),
                control_rock_lengths_m=control_rock_lengths,
                tau_mass_GeV=tau_mass,tau_rest_lifetime_s=tau_lifetime)
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=['# ντ 自然穿山：PeV 小样本测试','',
          f"全部事例完成：{result['all_completed']}；已完成事例的输运检查通过：{result['completed_checks_pass']}。",
          '检查覆盖自然取样/非强制标记、介质错配、CSV完整性、队列清空、时间窗、沉积记录；无反应事例另核对首段入口/出口/长度、飞行速度及窗口存活能量。此结论不包含未发生的CC/NC末态或τ衰变验证。','',
          '强制 CC、强制 NC、强制衰变均关闭；射电关闭，OpenMP-only。',
          '未筛选种子与历史预选回归种子分组统计；预选种子不能用来估算无偏概率。','',
          '| 能量 [PeV] | seed | 组别 | 状态 | CC / NC | τ 衰变 | 实际运行 [s] |',
          '|---:|---:|---|---|---|---:|---:|']
    for r in records:
        text.append(f"| {r['energy_GeV']/1e6:g} | {r['seed']} | {r['group']} | {r['state']} | {r.get('CC','—')} / {r.get('NC','—')} | {r.get('tau_decays','—')} | {r.get('wall_seconds',0):.1f} |")
    text+=['','## 是否形成 double bang','']
    for r in completed:
        for c in r['chains']:
            text.append(f"- {r['energy_GeV']/1e6:g} PeV，seed {r['seed']}：同一 τ history 的 CC→衰变距离 {c['separation_m']:.6g} m、飞行 {c['flight_ns']:.6g} ns，{c['cc_medium']}→{c['decay_medium']}。双级联拓扑候选：{c['double_cascade_topology']}。")
    if not any(r['chains'] for r in completed):text.append('已完成样本中尚未记录到匹配的 CC→τ→自然衰变链。')
    text+=['','两处真实顶点和可见次级只建立拓扑候选；未证明两个沉积峰可分辨，更没有证明射电/探测器能分辨。',
           '不能仅凭能量沉积的两个峰认定 double bang。τ→μ 道也不自动归为经典双级联。','',
           '## 概率、配置与限制','',
           f"未筛选组已完成 {result['unselected_completed']} 例，记录 {result['unselected_interactions']} 次弱反应。样本太小，不能检验稀有反应发生率。",'']
    for m in model:text.append(f"- 首段 {length:.2f} m 岩石中，{m['energy_GeV']/1e6:g} PeV 的 CTW CC+NC 首次反应概率估计为 {m['first_ridge_any_interaction_probability']:.3g}。不包含后续山脊及空气柱深。")
    text+=['',f"初级从空气进入真实 DEM。时间窗 {manifest['window_ns']} ns；EM/hadron/muon cut={manifest['emcut_GeV']}/{manifest['hadcut_GeV']}/{manifest['mucut_GeV']} GeV；emthin={manifest['emthin']}、max-weight={manifest['max_weight']}。配置与条件筛选规则以 manifest.yaml 为准。",
           '保留原生寿命和 τ 传播，不加大截面、不缩短 τ 寿命；所有种子及失败记录保留。未发生反应的入射中微子不会凭空产生 shower。',
           '低于 CTW 10 TeV 下限的次级弱反应、完整事件级 CC 极化和介质退极化仍非完整模型，本结果不是探测效率或全物理验收。','',
           '[CTW 截面模型](https://arxiv.org/abs/1102.0691)；[Double-bang 原始论文](https://arxiv.org/abs/hep-ph/9408296)。','',
           '![相互作用概率](figures/interaction_probability.png)','']
    if control_rock_lengths:
        text += [f'无反应对照累计岩石路径：{control_rock_lengths[0]:.6f} m；它包括首段之后的其他岩石段，不等同于首座山体厚度。',
                 '![真实地形与自然飞行](figures/natural_flight_geometry.png)','']
    for r in completed:
        if r['interactions']:text+= [f"![{r['name']}](figures/{r['name']}.png)",'']
    (root/'REPORT_CN.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','model_expectation']},ensure_ascii=False))
    if not result['completed_checks_pass']:raise SystemExit(1)


if __name__=='__main__':main()
