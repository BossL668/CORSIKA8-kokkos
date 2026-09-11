#!/usr/bin/env python3
"""Audit the bounded CC-only pilot and plot *actual* shower tracks, no radio.

Track-length histograms are labelled as such, not as longitudinal particle
counts. Samples are single independent histories, not ensemble mean profiles.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

COLORS={16:'teal',15:'magenta',13:'royalblue',11:'firebrick',22:'goldenrod'}
LABELS={16:r'$\nu_\tau$',15:r'$\tau^\pm$',13:r'$\mu^\pm$',11:r'$e^\pm$',22:r'$\gamma$'}
STYLES=['-','--',':']


def backend_label(group):
    return {'cpu':'CPU','openmp':'Kokkos OpenMP','cuda':'Kokkos CUDA'}.get(group.split('_')[0],group)


def load(path):return yaml.safe_load(path.read_text())


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def audit(path,job):
    s=load(path/'terrain_run.yaml');tr=pd.read_csv(path/'terrain/tracks.csv')
    dep=pd.read_csv(path/'terrain/deposits.csv');escape=pd.read_csv(path/'terrain/window_survivors.csv')
    p=tr[[f'{c}0_m' for c in 'xyz']].to_numpy();q=tr[[f'{c}1_m' for c in 'xyz']].to_numpy()
    tau=tr.pdg.abs()==15
    d=s['diagnostics'];td=s['tau'];nu=s['neutrino']
    check=dict(complete=s['complete'],radio_disabled=s['radio']=='disabled',
        no_material_mismatches=d['material_mismatches']==0,
        all_tracks_recorded=not d['csv_truncated'] and len(tr)==d['steps'],
        all_deposits_recorded=not d['deposition_csv_truncated'],
        finite=np.isfinite(tr.drop(columns=['medium']).to_numpy()).all().item(),
        finite_window=bool((tr.t1_s<=job['window_ns']*1e-9+1e-14).all()),
        no_forced_vertex=not s['conditional_forced_CC'],
        no_forced_decay=not td['force_decay_called'],
        survivors_match_counter=len(escape)==d['finite_window_survivors'],
        survivors_match_energy=abs(float((escape.weight*escape.total_GeV).sum())-d['finite_window_escaped_total_GeV'])<1e-9*job['energy_GeV'],
        deposits_match_counter=abs(dep.weighted_deposited_GeV.sum()-d['deposited_energy_GeV'])<1e-9*job['energy_GeV'])
    tau_length=float(np.linalg.norm(q[tau]-p[tau],axis=1).sum())
    endpoint_errors=[]
    for decay in td.get('decays',[]):
        rows=tr.history_id==decay['history_id']
        endpoint_errors.append(float(np.min(np.linalg.norm(q[rows]-decay['position_enu_m'],axis=1))) if rows.any() else float('inf'))
        check['tau_P4']=check.get('tau_P4',True) and max(abs(decay['relative_energy_residual']),abs(decay['relative_momentum_residual']))<1e-3
        check['tau_daughter_identity']=check.get('tau_daughter_identity',True) and all(c['parent_history_id']==decay['history_id'] for c in decay['daughters'])
        check['tau_has_nutau']=check.get('tau_has_nutau',True) and any(abs(c['pdg'])==16 for c in decay['daughters'])
    if td['tau_decay_count']:
        check['finite_tau_flight']=tau_length>0.
        check['decay_endpoint_continuous']=max(endpoint_errors)<1e-5
    role=job['role']
    if role=='unselected_neutrino':
        materials=tr.medium.to_numpy();sequence=materials[np.r_[True,materials[1:]!=materials[:-1]]].tolist()
        check['transparent_neutrino']=nu['interaction_count']==0 and sequence[:3]==['air','rock','air']
        check['unchanged_neutrino_energy']=bool(np.allclose(tr.E0_GeV,job['energy_GeV'],rtol=1e-12))
        check['no_false_deposit']=d['deposited_energy_GeV']==0.
    elif role=='conditioned_natural_CC':
        check['one_CC_one_tau_decay']=nu['interaction_count']==1 and td['tau_decay_count']==1
        check['decay_in_rock']=all(t['medium']=='rock' for t in td['decays'])
        check['CC_keeps_tau']=all(v['vertex_audit']['final_state_pdg_counts'].get(15,0)>=1 for v in nu.get('interactions',[]))
    elif role=='primary_tau_control':
        check['rock_tau_decay']=td['tau_decay_count']==1 and td['decays'][0]['medium']=='rock'
    elif role=='primary_tau_near_exit_control':
        check['tau_emerges_then_decays']=td['tau_decay_count']==1 and td['decays'][0]['medium']=='air' and {'air','rock'}.issubset(set(tr.medium[tau]))
    resources=json.loads((path.parent/(path.name+'_guard')/'resources.json').read_text())
    check['resource_guard']=resources['returncode']==0 and resources['stop_reason'] is None
    check={k:bool(v) for k,v in check.items()}
    result=dict(name=job['name'],run=str(path),passed=all(check.values()),checks=check,
        steps=d['steps'],CC_count=nu['interaction_count'],tau_decay_count=td['tau_decay_count'],
        tau_length_m=tau_length,decay_endpoint_errors_m=endpoint_errors,
        deposits_GeV=d['deposited_energy_GeV'],escape_GeV=d['finite_window_escaped_total_GeV'],
        elapsed_s=resources['wall_seconds'],
        peak_rss_MiB=max((x.get('rss_kib',0)/1024 for x in resources['samples']),default=0),
        peak_gpu_increment_MiB=max((x.get('gpu_delta_mib',0) for x in resources['samples']),default=0))
    return result,(s,tr,p,q,dep,escape)


def plots(root,spec,data,rows):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':True,
                         'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})
    out=root/'figures';out.mkdir(exist_ok=True)
    name='nutau10tev_selected_cc'
    selected=[(key,d) for key,d in data.items() if key[1]==name and
              d[0]['neutrino']['interaction_count'] and d[0]['tau']['tau_decay_count']]
    if not selected:return
    (group,_),(s,tr,p,q,dep,esc)=selected[0]
    entry=np.array(spec['chord']['entry']);direction=np.array(spec['chord']['direction'])
    vertex=np.array(s['neutrino']['interactions'][0]['position_enu_m'])
    decay=np.array(s['tau']['decays'][0]['position_enu_m'])
    grid=np.load(Path(s['scene']['geometry']['mesh_path']).parent/'terrain_grid.npz')
    fig=plt.figure(figsize=(17,5.5))
    for k,(span,stride,title) in enumerate([(9000,12,'Expanded DEM and 80 stations'),(180,1,'Rock CC and subsequent tracks'),(6,1,'Actual shower near CC')]):
        ax=fig.add_subplot(1,3,k+1,projection='3d')
        if k<2:
            e,n,z=[grid[c][::stride,::stride] for c in ('east_m','north_m','up_m')]
            mask=(abs(e-vertex[0])<span)&(abs(n-vertex[1])<span)
            ax.plot_trisurf(e[mask],n[mask],z[mask],cmap='terrain',alpha=.25,linewidth=0)
        for pdg,color in COLORS.items():
            ids=np.flatnonzero(tr.pdg.abs().to_numpy()==pdg)
            if k:ids=ids[(np.max(abs(p[ids]-vertex),axis=1)<span)&(np.max(abs(q[ids]-vertex),axis=1)<span)]
            ids=ids[::max(1,len(ids)//1800)]
            if ids.size:ax.add_collection3d(Line3DCollection(np.stack([p[ids],q[ids]],axis=1),colors=color,lw=1.2 if pdg in (15,16) else .4))
            ax.plot([],[],[],c=color,label=LABELS[pdg])
        ax.scatter(*vertex,c='black',marker='*',s=60);ax.scatter(*decay,c='magenta',marker='x',s=45)
        if k==0:
            obs=np.array([o['position_enu_m'] for o in s['scene']['radio']['observers']])
            ax.scatter(*obs.T,c='navy',s=3,label='Stations (radio off)')
        else:
            ax.set(xlim=(vertex[0]-span,vertex[0]+span),ylim=(vertex[1]-span,vertex[1]+span),zlim=(vertex[2]-span/2,vertex[2]+span/2))
        ax.set(xlabel='East [m]',ylabel='North [m]',zlabel='ENU Up [m]',title=title)
        if k==2:
            # mplot3d does not reliably include the rightmost z label in its
            # tight bounding box. Use an in-axes unit label for this panel.
            ax.set_zlabel('')
            ax.text2D(.85,.88,'Up [m]',transform=ax.transAxes,fontsize=9)
        ax.view_init(25,-45)
        if k==1:ax.legend(fontsize=8)
    fig.suptitle(f'{group}: 10 TeV neutrino, conditioned natural CC (single event)')
    save(fig,out/'01_nutau_terrain_3d.png')
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for pdg,color in COLORS.items():
        mask=tr.pdg.abs().to_numpy()==pdg
        seg=np.stack([np.column_stack(((p[mask]-entry)@direction,tr.E0_GeV[mask])),
                      np.column_stack(((q[mask]-entry)@direction,tr.E1_GeV[mask]))],axis=1)
        axs[0,0].add_collection(LineCollection(seg,colors=color,lw=.7,label=LABELS[pdg]))
    axs[0,0].set(xlim=(-50,280),ylim=(.009,12000),yscale='log',xlabel='Distance from rock entry [m]',ylabel='Total energy [GeV]',title='Actual tracks, not a fitted cascade')
    axs[0,0].axvspan(0,spec['chord']['length_m'],alpha=.12,color='saddlebrown');axs[0,0].legend(fontsize=8)
    for j,((key,_),d) in enumerate(selected):
        ss,tt,pp,qq,_,_=d;mask=tt.pdg.abs().to_numpy()==15
        origin=np.array(ss['neutrino']['interactions'][0]['position_enu_m'])
        axs[0,1].plot(np.r_[0,((qq[mask]-origin)@direction)*1000],np.r_[tt.E0_GeV[mask].iloc[0],tt.E1_GeV[mask]],
                      linestyle=STYLES[j%3],marker=['o','s','x'][j%3],ms=4,zorder=10-j,label=backend_label(key))
    axs[0,1].set(xlabel='Distance from CC along incident axis [mm]',ylabel='Tau total energy [GeV]',title='Resolved tau flight before decay');axs[0,1].legend()
    pdgs=list(dict.fromkeys(r['pdg'] for _,d in selected for r in d[0]['tau']['decays'][0]['daughters']))
    idx=np.arange(len(pdgs));width=.8/len(selected)
    for j,((key,_),d) in enumerate(selected):
        daughters=d[0]['tau']['decays'][0]['daughters']
        totals=[sum(r['energy_GeV'] for r in daughters if r['pdg']==pdg) for pdg in pdgs]
        axs[1,0].bar(idx+(j-(len(selected)-1)/2)*width,totals,width,label=backend_label(key),alpha=.8)
    axs[1,0].set_xticks(idx,list(map(str,pdgs)));axs[1,0].set(xlabel='Daughter PDG',ylabel='Total energy [GeV]',title='Tau daughters before thinning and cuts')
    axs[1,0].legend(fontsize=8)
    for j,((key,_),d) in enumerate(selected):
        _,tt,pp,qq,_,_=d;mu=(tt.pdg.abs()==13)&(tt.E0_GeV>100)
        label=backend_label(key)+(' (none >100 GeV)' if not mu.any() else '')
        axs[1,1].plot(((pp[mu]-entry)@direction),tt.E0_GeV[mu],linestyle='none',
                      marker=['o','s','x'][j%3],ms=2,zorder=10-j,label=label)
    axs[1,1].axvspan(0,spec['chord']['length_m'],alpha=.12,color='saddlebrown')
    axs[1,1].set(xlabel='Distance from entry [m]',ylabel='Muon total energy [GeV]',title='Observed high-energy muon continuation');axs[1,1].legend()
    save(fig,out/'02_nutau_CC_tau_decay_chain.png')
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    edges=np.linspace(-.5,6,66)
    for j,((key,_),(ss,tt,pp,qq,dd,_)) in enumerate(selected):
        v=np.array(ss['neutrino']['interactions'][0]['position_enu_m']);distance=((pp+qq)/2-v)@direction
        length=np.linalg.norm(qq-pp,axis=1)*tt.weight.to_numpy()
        for pid,ax in [(11,axs[0,0]),(22,axs[0,1])]:
            mask=tt.pdg.abs().to_numpy()==pid
            counts=np.histogram(distance[mask],edges,weights=length[mask])[0]
            ax.stairs(counts/np.diff(edges),edges,label=backend_label(key),linestyle=STYLES[j%3],lw=1.5,zorder=10+j)
            ax.set(xlabel='Distance from CC [m]',ylabel='Weighted track length / bin width',title=LABELS[pid]+' development (midpoint-binned)')
            ax.legend(fontsize=8)
        midpoint=np.column_stack([.5*(dd[f'{c}0_m']+dd[f'{c}1_m']) for c in 'xyz'])
        counts=np.histogram((midpoint-v)@direction,edges,weights=dd.weighted_deposited_GeV)[0]
        axs[1,0].stairs(counts/np.diff(edges),edges,label=backend_label(key),linestyle=STYLES[j%3],lw=1.5,zorder=10+j)
        part=tt.loc[tt.pdg.abs().isin([15,16,13,11,22])].copy()
        groups=part.groupby(part.pdg.abs()).apply(lambda x:(x.weight*np.linalg.norm(qq[x.index]-pp[x.index],axis=1)).sum())
        axs[1,1].plot([LABELS[p] for p in COLORS],[groups.get(p,0) for p in COLORS],
                      marker=['o','s','x'][j%3],linestyle=STYLES[j%3],label=backend_label(key),zorder=10-j)
    axs[1,0].set(xlabel='Distance from CC [m]',ylabel='Deposited energy [GeV/m]',title='Deposits: midpoint diagnostic, not exact dE/dX')
    axs[1,1].set(ylabel='Weighted track length [m]',yscale='log',title='Components in finite time window')
    axs[1,0].legend();axs[1,1].legend()
    fig.suptitle('Single-event shower development; not an ensemble/statistical equivalence test')
    save(fig,out/'03_shower_development.png')
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    chosen=[r for r in rows if r['name']==name]
    labels=[backend_label(r['group']) for r in chosen];x=np.arange(len(chosen))
    axs[0].bar(x-.18,[r['deposits_GeV'] for r in chosen],.36,label='Deposit');axs[0].bar(x+.18,[r['escape_GeV'] for r in chosen],.36,label='Window survivors')
    axs[0].set(ylabel='Weighted energy [GeV]',title='Partial accounting; not full energy closure');axs[0].legend()
    axs[1].bar(x,[r['elapsed_s'] for r in chosen]);axs[1].set(ylabel='Wall time [s]',title='Cold/warm initialization included')
    axs[2].bar(x,[r['peak_rss_MiB'] for r in chosen]);axs[2].set(ylabel='Peak RSS [MiB]',title='Bounded-memory diagnostics')
    for ax in axs:ax.set_xticks(x,labels,rotation=20)
    save(fig,out/'04_energy_and_resources.png')
    # Independent display of longitudinal plane crossings, with an explicit
    # straight-segment reconstruction convention (not a native writer export).
    planes=np.linspace(0,6,121)
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    categories=[('Electrons/positrons',lambda p:abs(p)==11),
                ('Photons',lambda p:p==22),('Muons',lambda p:abs(p)==13),
                ('Taus',lambda p:abs(p)==15),
                ('Hadrons',lambda p:~np.isin(abs(p),[11,12,13,14,15,16,22])),
                ('Neutrinos',lambda p:np.isin(abs(p),[12,14,16]))]
    for j,((key,_),(ss,tt,pp,qq,_,_)) in enumerate(selected):
        origin=np.array(ss['neutrino']['interactions'][0]['position_enu_m'])
        aa=(pp-origin)@direction;bb=(qq-origin)@direction
        lo=np.searchsorted(planes,np.minimum(aa,bb),side='left')
        hi=np.searchsorted(planes,np.maximum(aa,bb),side='left')
        for ax,(label,predicate) in zip(axs.flat,categories):
            ids=predicate(tt.pdg.to_numpy())&(lo<hi)
            delta=np.zeros(len(planes)+1)
            np.add.at(delta,lo[ids],tt.weight.to_numpy()[ids]);np.add.at(delta,hi[ids],-tt.weight.to_numpy()[ids])
            ax.plot(planes,np.cumsum(delta)[:-1],label=backend_label(key),lw=1.5,
                    linestyle=STYLES[j%3],zorder=10+j,marker='x' if j%3==2 else None,ms=3,markevery=10)
            ax.set(title=label,xlabel='Distance from CC [m]',ylabel='Weighted plane crossings')
            ax.legend(fontsize=8)
    fig.suptitle('All components: single-event profiles reconstructed from recorded step endpoints')
    save(fig,out/'05_longitudinal_components.png')


def local_cross_backend_controls(data,rows):
    """Compare only histories before scheduler-dependent secondary evolution.

    This does not require identical showers or decay channels after a CC.
    """
    result=[]
    selected=[(g,values[0]) for (g,name),values in data.items() if name=='nutau10tev_selected_cc']
    if len(selected)>1 and all(s['neutrino']['interaction_count']==1 for _,s in selected):
        reference=selected[0][1]['neutrino']['interactions'][0]
        for group,s in selected[1:]:
            v=s['neutrino']['interactions'][0]
            position_error=float(np.linalg.norm(np.array(v['position_enu_m'])-reference['position_enu_m']))
            p4=np.array(v['vertex_audit']['four_momentum_GeV']['outgoing_charged_lepton'])
            ref=np.array(reference['vertex_audit']['four_momentum_GeV']['outgoing_charged_lepton'])
            p4_error=float(np.max(abs(p4-ref))/ref[0])
            check=(v['target_pdg']==reference['target_pdg'] and position_error<1e-8 and p4_error<1e-10)
            result.append(dict(control='first_CC_before_secondary_evolution',reference=selected[0][0],
                group=group,passed=bool(check),vertex_error_m=position_error,outgoing_tau_P4_relative_error=p4_error))
    for name in ['tau100_rock_control','tau1tev_exit_control']:
        controls=[r for r in rows if r['name']==name]
        if len(controls)>1:
            ref=controls[0]
            for r in controls[1:]:
                delta=abs(r['tau_length_m']-ref['tau_length_m'])
                result.append(dict(control=name+'_first_flight',reference=ref['group'],group=r['group'],
                                   passed=delta<1e-8,length_error_m=delta))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',nargs='+',required=True)
    p.add_argument('--cases',nargs='+',help='Explicit subset; recorded in the report')
    p.add_argument('--report-dir',type=Path,help='Keep subset/field plots separate')
    a=p.parse_args();spec=load(a.output/'pilot_manifest.yaml')
    if a.cases and set(a.cases)-{j['name'] for j in spec['jobs']}:
        p.error('unknown case in --cases')
    report=a.report_dir or a.output;report.mkdir(parents=True,exist_ok=True)
    rows=[];data={};missing=[]
    for group in a.groups:
        for job in spec['jobs']:
            if a.cases and job['name'] not in a.cases:continue
            path=a.output/group/job['name']
            if not (path/'terrain_run.yaml').exists():missing.append(str(path));continue
            row,values=audit(path,job);row['group']=group;rows.append(row);data[group,job['name']]=values
    local=local_cross_backend_controls(data,rows)
    result=dict(passed=not missing and bool(rows) and all(r['passed'] for r in rows) and all(r['passed'] for r in local),
        scope='bounded CC-only transport/decay pilot, no radio, not a shower-ensemble acceptance',
        groups=a.groups,cases=a.cases or [j['name'] for j in spec['jobs']],
        missing=missing,events=rows,local_cross_backend_controls=local,limitations=spec['limitations'])
    (report/'particle_support_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    plots(report,spec,data,rows)
    print(json.dumps(dict(passed=result['passed'],events=len(rows),missing=len(missing))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
