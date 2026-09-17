#!/usr/bin/env python3
"""PSR-only profile, ledger and all-station radio diagnostics; English figures."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert socket.gethostname()=='psrpku2025'
    out=root/'report';runs={m:json.loads((out/('runs_'+m+'.json')).read_text()) for m in ['openmp','cuda']}
    station=json.loads((out/'stations_80_audit.json').read_text())
    extra=json.loads((out/'extra_checks.json').read_text())
    assert station['count']==80 and station['names_complete_unique'] and extra['outside_injection_rejected']['passed']
    for mode in runs:
        for tag in ['real_air_electron_80stations','real_rock_electron_80stations']:
            assert runs[mode][tag]['complete'] and len(runs[mode][tag]['observer_names'])==80
        for name in ['', 'cpu_'] if mode=='openmp' else ['']:
            assert json.loads((out/(mode+'_'+name+'contained_profile.json')).read_text())['exact_decompressed_csv_match']
    def load(mode,tag,name):return pd.read_csv(root/'runs'/mode/tag/'output/terrain'/(name+'.csv.gz'))
    def profile(frame,bins):
        value=np.zeros(len(bins)-1)
        for row in frame.itertuples(index=False):
            x0,x1=float(row.x0_m),float(row.x1_m);energy=float(row.weighted_deposited_GeV)
            lo,hi=sorted([x0,x1])
            if hi-lo<1e-14:
                index=np.searchsorted(bins,.5*(lo+hi),side='right')-1
                if 0<=index<len(value):value[index]+=energy
            else:value+=energy*np.maximum(0,np.minimum(bins[1:],hi)-np.maximum(bins[:-1],lo))/(hi-lo)
        return value
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for column,(backend,prefix) in enumerate([('Kokkos OpenMP',''),('Scalar CPU','cpu_')]):
        on=load('openmp',prefix+'contained_on','tracks');off=load('openmp',prefix+'contained_off','tracks')
        dep=load('openmp',prefix+'contained_on','deposits');dep_off=load('openmp',prefix+'contained_off','deposits')
        lo=min(on.x0_m.min(),on.x1_m.min(),dep.x0_m.min(),dep.x1_m.min());hi=max(on.x0_m.max(),on.x1_m.max(),dep.x0_m.max(),dep.x1_m.max())
        bins=np.linspace(lo-1e-10,hi+1e-10,100);centres=.5*(bins[:-1]+bins[1:]);counts=[]
        for frame in [on,off]:
            count=np.zeros(len(centres))
            for row in frame.itertuples(index=False):
                if abs(row.pdg)!=11:continue
                low,high=sorted([row.x0_m,row.x1_m]);count+=row.weight*((centres>=low)&(centres<high))
            counts.append(count)
        for i,(label,style) in enumerate([('Boundary on','-'),('Boundary off','--')]):
            axes[0,column].plot(centres*265,counts[i],style,label=label,lw=2)
            values=profile(dep if i==0 else dep_off,bins)
            axes[1,column].plot(centres*265,values*1000,style,label=label,lw=2)
        assert abs(profile(dep,bins).sum()-dep.weighted_deposited_GeV.sum())<1e-10
        axes[0,column].set(title=backend+': contained 50 MeV electron',ylabel='Weighted electron/positron crossings')
        axes[1,column].set(xlabel='Longitudinal coordinate in SiO2 [g/cm2]',ylabel='Deposited energy per bin [MeV]')
        for ax in axes[:,column]:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'03_complete_profiles.png',dpi=180);plt.close(fig)
    tracks=load('openmp','profile_crossing','tracks');exits=load('openmp','profile_crossing','domain_exits')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for row in tracks.itertuples(index=False):
        axes[0].plot((np.array([row.x0_m,row.x1_m])-10)*1000,np.array([row.z0_m,row.z1_m])+1,'-',lw=1.5)
    axes[0].axvline(0,color='k',ls='--',label='DEM coverage boundary')
    axes[0].scatter((exits.x_m-10)*1000,exits.z_m+1,c='red',marker='x',s=65,label='Recorded terminal states')
    axes[0].set(xlabel='Distance from boundary [mm]',ylabel='Vertical displacement [m]',title='100 MeV electron: every interior segment retained');axes[0].legend(fontsize=8)
    tags=['rock_electron_exit','profile_crossing','contained_on','cpu_tau_minus','cpu_nu_tau']
    residuals=[max(abs(runs['openmp'][t]['unexplained_over_initial']),1e-18) for t in tags]
    axes[1].bar(range(len(tags)),residuals,color='steelblue');axes[1].set_yscale('log');axes[1].axhline(1e-9,color='red',ls='--',label='Acceptance tolerance')
    axes[1].set_xticks(range(len(tags)));axes[1].set_xticklabels(['Rock e exit','Crossing shower','Contained shower','Tau exit','Neutrino exit'],rotation=20,ha='right')
    axes[1].set(ylabel='Unexplained weighted energy / initial energy',title='Energy ledger includes remaining energy at exit');axes[1].legend(fontsize=8)
    fig.savefig(out/'04_boundary_tracks_and_energy.png',dpi=180);plt.close(fig)
    radio={};peak_rows=[]
    for mode in ['openmp','cuda']:
        for tag in ['real_air_electron_80stations','real_rock_electron_80stations']:
            folder=root/'runs'/mode/tag/'output/radio'
            c=pd.read_csv(folder/'CoREAS/field.csv',usecols=['observer','time_s','Ex_V_m','Ey_V_m','Ez_V_m'])
            z=pd.read_csv(folder/'ZHS/field.csv',usecols=['observer','time_s','Ex_V_m','Ey_V_m','Ez_V_m'])
            config=json.loads((folder/'ZHS/config.json').read_text())
            names=runs[mode][tag]['observer_names'];n=len(c)//80
            assert len(c)==len(z)==80*n and set(c.observer)==set(range(80)) and set(z.observer)==set(range(80))
            ca=c[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy();za=z[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy()
            assert np.isfinite(ca).all() and np.isfinite(za).all()
            norm=np.linalg.norm(za);relative=float(np.linalg.norm(ca-za)/norm) if norm else None
            if relative is not None:assert relative<1e-5,(mode,tag,relative)
            peaks=[]
            for i,name in enumerate(names):
                a=ca[c.observer.to_numpy()==i];b=za[z.observer.to_numpy()==i]
                peak=float(np.max(np.linalg.norm(b,axis=1)));peaks.append(peak)
                peak_rows.append(dict(backend=mode,case=tag,station=name,peak_ZHS_V_m=peak,peak_CoREAS_V_m=float(np.max(np.linalg.norm(a,axis=1)))))
            radio[mode+'_'+tag]=dict(observer_count=80,rows_per_algorithm=len(c),samples_per_observer=n,relative_l2=relative,
                nonzero_stations=int(np.count_nonzero(peaks)),maximum_peak_V_m=max(peaks),all_finite=True)
            if mode=='openmp':
                best=int(np.argmax(peaks));mask=c.observer.to_numpy()==best;t=c.time_s.to_numpy()[mask]*1e6
                a=ca[mask];b=za[mask];axis=int(np.argmax(np.max(abs(b),axis=0)));peak=int(np.argmax(np.linalg.norm(b,axis=1)))
                lo=max(0,peak-18);hi=min(n,peak+19)
                fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
                axes[0].plot(t[lo:hi],b[lo:hi,axis],label='ZHS',lw=2)
                axes[0].plot(t[lo:hi],a[lo:hi,axis],'--',label='CoREAS',lw=1.6)
                axes[0].set(xlabel='Arrival time [us]',ylabel='E'+str('xyz'[axis])+' [V/m]',title=names[best]+': strongest station in this boundary test');axes[0].legend();axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                if max(peaks)==0:
                    axes[0].set_title('No accepted top-surface transmission path')
                    axes[0].text(.5,.7,'Zero field is a path-acceptance result,\nnot an amplitude-agreement test.',transform=axes[0].transAxes,ha='center',fontsize=9)
                axes[1].bar(range(80),peaks,color=['crimson' if s.startswith('E') else 'darkorange' if s.startswith('W') else 'navy' if s.startswith('N') else 'purple' for s in names])
                axes[1].set_xticks(np.arange(0,80,4));axes[1].set_xticklabels(names[::4],rotation=90,fontsize=7)
                axes[1].set(ylabel='Peak vector field [V/m]',title='All 80 stations: no observer subset');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                kind='air' if 'air_' in tag else 'rock';fig.suptitle('Real DEM, '+kind+' source: short boundary validation track')
                fig.savefig(out/('05_radio_80_'+kind+'.png'),dpi=180);plt.close(fig)
    folder=root/'runs/openmp/rock_electron_exit/output/radio'
    c=pd.read_csv(folder/'CoREAS/field.csv');z=pd.read_csv(folder/'ZHS/field.csv')
    ca=c[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy();za=z[['Ex_V_m','Ey_V_m','Ez_V_m']].to_numpy()
    positive_relative=float(np.linalg.norm(ca-za)/np.linalg.norm(za));assert np.max(abs(za))>0 and positive_relative<1e-10
    axis=int(np.argmax(np.max(abs(za),axis=0)));peak=int(np.argmax(abs(za[:,axis])));lo=max(0,peak-18);hi=min(len(z),peak+19)
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.plot(z.time_s.to_numpy()[lo:hi]*1e9,za[lo:hi,axis],label='ZHS',lw=2)
    ax.plot(c.time_s.to_numpy()[lo:hi]*1e9,ca[lo:hi,axis],'--',label='CoREAS')
    ax.set(xlabel='Arrival time [ns]',ylabel='Electric field [V/m]',title='Nonzero radio control: 20 MeV electron in flat-top SiO2')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'06_nonzero_rock_radio_control.png',dpi=180);plt.close(fig)
    (out/'nonzero_radio_control.json').write_text(json.dumps(dict(relative_l2=positive_relative,peak_V_m=float(np.max(abs(za))),observer_count=1,scope='diagnostic observer, not a 21CMA station')))
    with (out/'radio_80_station_peaks.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(peak_rows[0]));writer.writeheader();writer.writerows(peak_rows)
    (out/'radio_80_acceptance.json').write_text(json.dumps(radio,indent=2)+'\n')
    # Explicitly paired resident/batched full source ledgers.
    for mode in ['openmp','cuda']:
        for name in ['tracks.csv.gz','deposits.csv.gz','domain_exits.csv.gz']:
            paths=[root/'runs'/mode/t/'output/terrain'/name for t in ['rock_electron_exit','rock_electron_exit_batched']]
            assert gzip.open(paths[0],'rb').read()==gzip.open(paths[1],'rb').read()
    result=dict(passed=True,complete_boundary_cases=sum(len(v) for v in runs.values()),
        maximum_unexplained_over_initial=max(abs(r['unexplained_over_initial']) for v in runs.values() for r in v.values()),
        exact_contained_profiles=True,exact_resident_batched_terminal_ledgers=True,original_binary_replay=extra,
        stations=station,radio=radio,binaries={m:hashlib.sha256((root/('build-'+m)/'c8_terrain_cascade').read_bytes()).hexdigest() for m in runs})
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACCEPTED',result['complete_boundary_cases'],'cases; 80 stations; boundary profiles and ledgers complete',flush=True)

if __name__=='__main__':main()
