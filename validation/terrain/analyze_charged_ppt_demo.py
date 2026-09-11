#!/usr/bin/env python3
"""Charged terrain pictures plus independent per-step kinematic checks.

All curves use actual recorded segments. The Lorentz check tests positions,
not post-scattering directions or the whole stochastic shower distribution.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml
from analyze_nutau_ppt_demo import save, xyz, plt, LineCollection, Line3DCollection, _terrain_interpolator


def read(root, job):
    p=root/'runs'/job['name'];s=yaml.safe_load((p/'terrain_run.yaml').read_text())
    t=pd.read_csv(p/'terrain/tracks.csv');d=s['diagnostics']
    checks=dict(complete=s['complete'],openmp=s['accelerator']['execution_space']=='OpenMP',
                radio_off=s['radio']=='disabled',no_material_error=d['material_mismatches']==0,
                complete_tracks=not d['csv_truncated'] and len(t)==d['steps'],
                finite=bool(np.isfinite(t.drop(columns='medium').to_numpy()).all()),
                time_monotone=bool((t.t1_s>=t.t0_s).all()),
                window=bool((t.t1_s<=1.1e-6+1e-14).all()),
                field=s['magnetic_field_model']==job['field'],
                rock_zero_field=bool(np.all(np.array(s['rock_magnetic_field_enu_T'])==0)))
    r=json.loads((root/'runs'/(job['name']+'_guard/resources.json')).read_text())
    checks['resource_guard']=r['returncode']==0 and r['stop_reason'] is None
    return dict(s=s,t=t,checks=checks,resource=r)


def lorentz(t, s):
    # CORSIKA transport masses; not PROPOSAL cross-section masses.
    t=t[t.pdg.abs().isin([11,13])].copy()
    mass=np.where(t.pdg.abs()==11,.0005109989,.1056584)
    E=t.E0_GeV.to_numpy();momentum=np.sqrt((E-mass)*(E+mass))
    L=(t.t1_s-t.t0_s).to_numpy()*299792458.*momentum/E
    direction=t[['nx0','ny0','nz0']].to_numpy()
    field=np.array(s['air_magnetic_field_enu_T'])[None,:]*(t.medium.to_numpy()=='air')[:,None]
    cross=np.cross(direction,field);norm=np.linalg.norm(cross,axis=1)
    # q/e = -sign(PDG) for electrons/positrons and muons/antimuons.
    prediction=-np.sign(t.pdg.to_numpy())[:,None]*.2997925146834389/(2*momentum[:,None])*L[:,None]**2*cross
    measured=xyz(t,1)-xyz(t,0)-L[:,None]*direction
    unit=np.divide(cross,norm[:,None],out=np.zeros_like(cross),where=norm[:,None]>0)
    return pd.DataFrame(dict(step=t.step.to_numpy(),pdg=t.pdg.to_numpy(),medium=t.medium.to_numpy(),
        L_m=L,predicted_signed_m=np.sum(prediction*unit,axis=1),measured_signed_m=np.sum(measured*unit,axis=1),
        residual_m=np.linalg.norm(measured-prediction,axis=1),
        measured_bend_m=np.linalg.norm(measured,axis=1)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--neutrino-results',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();out=root/'figures';out.mkdir(exist_ok=True)
    tab=root/'plot_data';tab.mkdir(exist_ok=True)
    spec=yaml.safe_load((root/'manifest.yaml').read_text());data={};rows=[];oracles={}
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.grid':True,'grid.alpha':.18,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    for job in spec['jobs']:
        event=read(root,job);data[job['name']]=event;oracle=lorentz(event['t'],event['s']);oracles[job['name']]=oracle
        oracle.to_csv(tab/(job['name']+'_local_lorentz.csv'),index=False)
        event['checks']['local_position_oracle']=bool((oracle.residual_m<1e-8).all())
        rows.append(dict(name=job['name'],checks=event['checks'],steps=len(event['t']),
            maximum_local_position_residual_m=float(oracle.residual_m.max()),
            local_charged_steps=len(oracle),wall_s=event['resource']['wall_seconds'],
            peak_RSS_MiB=max(r.get('rss_kib',0)/1024 for r in event['resource']['samples'])))
    mu=data['mu100_crossing'];m=mu['t'][mu['t'].pdg==13].sort_values('t0_s').reset_index(drop=True)
    # Verify this is one continuation chain, not unrelated muons joined together.
    assert all(m.history_id.iloc[i]==m.history_id.iloc[i-1] or
               m.parent_history_id.iloc[i]==m.history_id.iloc[i-1] for i in range(1,len(m)))
    entry=np.array(spec['chord']['entry']);direction=np.array(spec['direction'])
    aa,bb=xyz(m,0),xyz(m,1);s0=(aa-entry)@direction;s1=(bb-entry)@direction
    changes=np.flatnonzero(m.medium.to_numpy()[1:]!=m.medium.to_numpy()[:-1])+1
    boundary=[]
    for i in changes:
        boundary.append(dict(from_medium=m.medium.iloc[i-1],to_medium=m.medium.iloc[i],distance_m=float(s0[i]),
            position_gap_m=float(np.linalg.norm(aa[i]-bb[i-1])),
            time_gap_s=float(abs(m.t0_s.iloc[i]-m.t1_s.iloc[i-1])),
            energy_gap_GeV=float(abs(m.E0_GeV.iloc[i]-m.E1_GeV.iloc[i-1])),
            same_history=bool(m.history_id.iloc[i]==m.history_id.iloc[i-1]),
            same_weight=bool(m.weight.iloc[i]==m.weight.iloc[i-1])))
    assert len(boundary)==2 and [v['to_medium'] for v in boundary]==['rock','air']
    boundary_pass=all(v['position_gap_m']<1e-7 and v['time_gap_s']<1e-14 and v['energy_gap_GeV']<1e-8
                      and v['same_history'] and v['same_weight'] for v in boundary)
    result=dict(passed=all(all(r['checks'].values()) for r in rows) and boundary_pass,
                scope='charged local transport and one muon crossing; not ensemble or radio certification',
                runs=rows,boundaries=boundary,muon_exit_energy_GeV=float(m.E0_GeV.iloc[changes[1]]),
                neutrino_figure_source=str(a.neutrino_results.resolve()))
    (root/'charged_acceptance.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('See charged_acceptance.json; do not claim acceptance')
    gridpath=Path(mu['s']['scene']['geometry']['mesh_path']).parent/'terrain_grid.npz'
    with np.load(gridpath) as g:surface=_terrain_interpolator(g['east_m'],g['north_m'],g['up_m'])
    colors={'rock':'#ab8353','air':'#328abe'}
    fig,axs=plt.subplots(1,3,figsize=(16,5.2))
    distance=np.linspace(-50,280,700);q=entry+distance[:,None]*direction;h=np.asarray(surface(q[:,0],q[:,1]))
    axs[0].fill_between(distance,h,h.min()-5,color='#ccb38e',alpha=.7,label='Rock cross-section at incident axis')
    for medium in ['air','rock']:
        mask=m.medium==medium
        axs[0].add_collection(LineCollection(np.stack([np.column_stack((s0[mask],aa[mask,2])),np.column_stack((s1[mask],bb[mask,2]))],axis=1),colors=colors[medium],lw=2,label='Muon in '+medium))
        axs[1].add_collection(LineCollection(np.stack([np.column_stack((s0[mask],m.E0_GeV[mask])),np.column_stack((s1[mask],m.E1_GeV[mask]))],axis=1),colors=colors[medium],lw=1.8,label=medium))
        transverse=np.array([-direction[1],direction[0],0.])
        axs[2].add_collection(LineCollection(np.stack([np.column_stack((s0[mask],(aa[mask]-entry)@transverse)),np.column_stack((s1[mask],(bb[mask]-entry)@transverse))],axis=1),colors=colors[medium],lw=2,label=medium))
    axs[0].set(xlim=(-50,280),ylim=(h.min()-5,h.max()+4),ylabel='ENU Up [m]',title=r'100 GeV $\mu^-$: air → rock → air');axs[0].legend(fontsize=8)
    axs[1].set(xlim=(-50,280),ylim=(0,105),ylabel='Muon total energy [GeV]',title='Continuous loss + stochastic losses');axs[1].legend()
    axs[2].autoscale();axs[2].set(xlim=(-50,280),ylabel='Horizontal transverse displacement [m]',title='Rock can scatter even when B = 0')
    for ax in axs:
        ax.set_xlabel('Distance along incident axis [m]')
        for crossing in boundary:ax.axvline(crossing['distance_m'],ls=':',c='black',lw=.8)
    save(fig,out,'07_muon_rock_air_transport','Actual continuous muon lineage; interface positions from transport. Terrain cross-section is a projection, not the off-axis surface.')

    fig,axs=plt.subplots(1,3,figsize=(16,5.2));B=np.array(data['electron_air_B']['s']['air_magnetic_field_enu_T'])
    u=np.cross(direction,B);u/=np.linalg.norm(u)
    for ax,(name,label) in zip(axs,[('electron_air_B',r'$e^-$ primary, IGRF14'),('positron_air_B',r'$e^+$ primary, IGRF14'),('electron_air_zero',r'$e^-$ primary, B = 0')]):
        t=data[name]['t'];origin=np.array(next(j['position'] for j in spec['jobs'] if j['name']==name))
        for pdg,color,lab in [(11,'#c43e39',r'$e^-$'),(-11,'#396eae',r'$e^+$'),(22,'#b69939',r'$\gamma$')]:
            selected=t[t.pdg==pdg];aa2,bb2=xyz(selected,0)-origin,xyz(selected,1)-origin
            seg=np.stack([np.column_stack((aa2@direction,aa2@u)),np.column_stack((bb2@direction,bb2@u))],axis=1)
            ax.add_collection(LineCollection(seg,colors=color,lw=.7,alpha=.7,label=lab))
        ax.autoscale();ax.margins(.06);ax.set(xlabel='Distance along incident axis [m]',ylabel=r'Displacement along $\hat n_0\times\hat B$ [m]',title=label);ax.legend(fontsize=9)
    xlimits=[ax.get_xlim() for ax in axs];ylimits=[ax.get_ylim() for ax in axs]
    for ax in axs:
        ax.set_xlim(min(v[0] for v in xlimits),max(v[1] for v in xlimits))
        ax.set_ylim(min(v[0] for v in ylimits),max(v[1] for v in ylimits))
    fig.suptitle('1 GeV electromagnetic showers in air: actual tracks, no thinning')
    save(fig,out,'08_air_charged_tracks','Different shower trees are allowed. Scattering remains when B = 0; visual separation alone does not isolate the Lorentz force.')

    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    allair=pd.concat([oracles['electron_air_B'],oracles['positron_air_B']])
    for pid,label,color in [(11,r'$e^-$','#c43e39'),(-11,r'$e^+$','#396eae')]:
        v=allair[(allair.pdg==pid)&(abs(allair.predicted_signed_m)>1e-9)]
        axs[0].scatter(v.predicted_signed_m*1e6,v.measured_signed_m*1e6,s=9,alpha=.5,c=color,label=label)
    low,high=axs[0].get_xlim();axs[0].plot([low,high],[low,high],'k--',lw=1,label='identity')
    axs[0].set(xlabel='Lorentz / leapfrog prediction [µm]',ylabel='Measured local transverse displacement [µm]',title='Charge sign and magnetic curvature');axs[0].legend()
    for name,label in [('electron_air_B','Air, IGRF14'),('electron_air_zero','Air, zero field'),('mu100_crossing','Muon + EM, rock / air')]:
        residual=np.sort(oracles[name].residual_m.to_numpy())
        axs[1].semilogx(np.maximum(residual,1e-17),np.arange(1,len(residual)+1)/len(residual),label=label)
    axs[1].set(xlabel='Per-step position residual [m]',ylabel='Fraction of tested steps',title='Recorded endpoint − independent formula');axs[1].legend(fontsize=9)
    save(fig,out,'09_local_magnetic_oracle',r'$\Delta r-\beta c\Delta t\,n_0=q\,\kappa(\beta c\Delta t)^2(n_0\times B)/(2p)$; positions only, not post-scattering directions.')

    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for col,i in enumerate(changes):
        start,stop=max(0,i-3),min(len(m),i+3)
        centre=aa[i];near=np.arange(start,stop)
        for j in near:
            st=np.array([s0[j],s1[j]])-s0[i]
            height=np.array([aa[j,2],bb[j,2]])-centre[2]
            axs[0,col].plot(st,height*1000,c=colors[m.medium.iloc[j]],lw=2)
            axs[1,col].plot(st,[m.E0_GeV.iloc[j],m.E1_GeV.iloc[j]],c=colors[m.medium.iloc[j]],lw=2)
        for ax in axs[:,col]:ax.axvline(0,c='black',ls=':',lw=1);ax.set_xlabel('Distance from actual crossing [m]')
        axs[0,col].scatter(0,0,c='black',s=35);axs[0,col].set(ylabel='Up relative to boundary point [mm]',title=boundary[col]['from_medium']+' → '+boundary[col]['to_medium'])
        axs[1,col].set_ylabel('Muon total energy [GeV]');axs[1,col].ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('Interface continuity: blue = air, brown = rock')
    save(fig,out,'10_boundary_continuity','No artificial absorption, energy jump or identity reset at the material switch. Slopes may change; stochastic interactions are separate.')
    for f in (a.neutrino_results/'figures').glob('*'):
        if f.suffix in ('.png','.pdf'):shutil.copy2(f,out/f.name)
    # Archive the referenced acceptance, not a fabricated new neutrino test.
    shutil.copy2(a.neutrino_results/'acceptance.json',root/'neutrino_acceptance_reused.json')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
