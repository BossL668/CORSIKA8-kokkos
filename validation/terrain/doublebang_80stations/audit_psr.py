#!/usr/bin/env python3
"""Stream all tracks and all 80 radio stations; never sample the acceptance data."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HISTORY={'history_id':'uint64','parent_history_id':'uint64'}
FIELDS=['Ex_V_m','Ey_V_m','Ez_V_m','Ex_outside','Ey_outside','Ez_outside','Ex_inside','Ey_inside','Ez_inside']


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def data_path(path):
    return path if path.exists() else path.with_suffix(path.suffix+'.gz')


def json_rows(frame):
    # JSON round-trip preserves uint64 history identifiers as integers.
    return json.loads(frame.to_json(orient='records',double_precision=15))


def distribute(a,b,energy,edges):
    """Integrate constant dE/dx over every crossed bin, retaining all energy."""
    lo=np.minimum(a,b);hi=np.maximum(a,b);n=len(edges)-1
    assert np.all(lo>=edges[0]) and np.all(hi<=edges[-1])
    i=np.minimum(np.searchsorted(edges,lo,side='right')-1,n-1)
    j=np.minimum(np.searchsorted(edges,hi,side='right')-1,n-1)
    same=i==j
    out=np.bincount(i[same],weights=energy[same],minlength=n)
    take=~same;i=i[take];j=j[take];lo=lo[take];hi=hi[take]
    rate=energy[take]/(hi-lo)
    out+=np.bincount(i,weights=rate*(edges[i+1]-lo),minlength=n)
    out+=np.bincount(j,weights=rate*(hi-edges[j]),minlength=n)
    diff=np.bincount(i+1,weights=rate,minlength=n+1)-np.bincount(j,weights=rate,minlength=n+1)
    out+=np.cumsum(diff[:-1])*np.diff(edges)
    return out


def tracks_and_profile(folder,summary,out):
    d=summary['diagnostics'];r=summary['radio_result'];terrain=folder/'output/terrain'
    exits=pd.read_csv(data_path(terrain/'domain_exits.csv'),dtype=HISTORY)
    assert len(exits)==d['domain_exits'] and exits.history_id.is_unique
    exits['exit_row']=np.arange(len(exits));matched=np.zeros(len(exits),dtype=np.int64)
    wanted={int(k['history_id']) for v in summary['neutrino'].get('interactions',[])+summary['tau'].get('decays',[]) for k in v['daughters']}
    witnessed={};tau=[];count=0;charged=0
    physical_charges=[11,13,15,211,321,2212,3222,3112,3312,3334]
    for frame in pd.read_csv(data_path(terrain/'tracks.csv'),dtype=HISTORY,chunksize=150000):
        size=len(frame)
        assert np.array_equal(frame.step.to_numpy(),np.arange(count+1,count+size+1))
        count+=size
        p=frame[['x0_m','y0_m','z0_m']].to_numpy();q=frame[['x1_m','y1_m','z1_m']].to_numpy()
        dt=frame.t1_s.to_numpy()-frame.t0_s.to_numpy();w=frame.weight.to_numpy()
        vals=frame[['E0_GeV','E1_GeV','t0_s','t1_s']].to_numpy()
        assert np.isfinite(vals).all() and np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(w).all()
        assert np.all(w>0) and np.all(dt>=0) and np.all(frame.E1_GeV>=0)
        assert np.all(np.linalg.norm(q-p,axis=1)<=299792458*dt*(1+1e-8)+1e-6)
        pid=np.abs(frame.pdg.to_numpy());nucleus=(pid>=1000000000)&(((pid//10000)%1000)>0)
        moving=np.any(p!=q,axis=1)&(dt>0)
        charged+=int(np.count_nonzero((np.isin(pid,physical_charges)|nucleus)&moving))
        tau.extend(json_rows(frame[pid==15]))
        selected=frame[frame.history_id.isin(wanted)]
        for history,group in selected.groupby('history_id',sort=False):
            key=str(int(history))
            if key not in witnessed:witnessed[key]=dict(first=json_rows(group.iloc[:1])[0],steps=0)
            witnessed[key]['steps']+=len(group)
        if len(exits):
            selected=frame[frame.history_id.isin(exits.history_id)]
            if len(selected):
                pairs=selected.groupby('history_id',sort=False).tail(1).merge(exits,on='history_id',suffixes=('_track','_exit'),validate='one_to_one')
                assert np.all(pairs.t1_s.to_numpy()<=pairs.time_s.to_numpy()+1e-16),'track continued after terminal exit'
                valid=np.abs(pairs.t1_s.to_numpy()-pairs.time_s.to_numpy())<=1e-16
                for axis in 'xyz':valid&=np.abs(pairs[axis+'1_m'].to_numpy()-pairs[axis+'_m'].to_numpy())<1e-8
                valid&=np.isclose(pairs.E1_GeV.to_numpy(),pairs.total_GeV.to_numpy(),rtol=1e-12,atol=1e-10)
                for key in ['weight','pdg','parent_history_id','medium']:
                    valid&=pairs[key+'_track'].to_numpy()==pairs[key+'_exit'].to_numpy()
                # Later chunks replace earlier states of the same history.
                # Validate its final recorded segment once, after the full scan.
                matched[pairs.exit_row.to_numpy()]=valid
    assert count==d['steps'] and np.all(matched==1)
    for algorithm in ['CoREAS','ZHS']:
        assert r[algorithm]['device_tracks']+r[algorithm]['cpu_tracks']==charged
        assert r[algorithm]['track_observer_pairs']==charged*80
    initial=summary['energy_GeV']
    escape=float(np.sum(exits.weight.to_numpy(dtype=np.longdouble)*exits.total_GeV.to_numpy(dtype=np.longdouble)))
    assert abs(escape-d['domain_escaped_total_GeV'])<1e-9*max(initial,1)
    assert abs(escape-summary['energy_ledger']['domain_escape_GeV'])<1e-9*max(initial,1)
    origin=np.asarray(summary['position_m']);direction=np.asarray(summary['direction']);direction/=np.linalg.norm(direction)
    maximum=299792458*summary['transport_window_ns']*1e-9+100
    limit=math.ceil(maximum/25)*25;edges=np.arange(-limit,limit+25,25.)
    histogram=np.zeros(len(edges)-1);total=np.longdouble(0);deposit_count=0
    for frame in pd.read_csv(data_path(terrain/'deposits.csv'),chunksize=150000):
        a=(frame[['x0_m','y0_m','z0_m']].to_numpy()-origin)@direction
        b=(frame[['x1_m','y1_m','z1_m']].to_numpy()-origin)@direction
        energy=frame.weighted_deposited_GeV.to_numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(energy).all()
        histogram+=distribute(a,b,energy,edges)
        total+=np.sum(energy,dtype=np.longdouble);deposit_count+=len(frame)
    assert deposit_count==d['deposition_rows']
    assert abs(float(total)-d['deposited_energy_GeV'])<1e-8*max(initial,1)
    assert abs(histogram.sum()-float(total))<1e-8*max(initial,1)
    np.savez_compressed(out/'profile.npz',edges_m=edges,deposited_GeV=histogram)
    save(folder/'tau_tracks.json',tau);save(folder/'vertex_daughter_tracks.json',witnessed)
    return dict(all_tracks=count,all_deposits=deposit_count,charged_source_tracks=charged,domain_exits=len(exits),
        domain_escape_GeV=escape,deposited_GeV=float(total),terminal_tracks_complete=True),tau


def radio(folder,out,plots):
    config=[json.loads((folder/'output/radio'/alg/'config.json').read_text()) for alg in ['CoREAS','ZHS']]
    c=config[0];n=c['samples'];rate=c['sample_rate_Hz'];names=[o['name'] for o in c['observers']]
    assert len(names)==80 and set(names)=={arm+('%02d'%i) for arm in 'EWNS' for i in range(1,21)}
    for other in config:
        assert other['samples']==n and other['sample_rate_Hz']==rate and other['observers']==c['observers'] and other['dem_coverage_boundary']
    readers=[pd.read_csv(data_path(folder/'output/radio'/alg/'field.csv'),chunksize=n,float_precision='round_trip') for alg in ['CoREAS','ZHS']]
    times=c['start_time_s']+np.arange(n)/rate
    band=(np.fft.rfftfreq(n,1/rate)>=50e6)&(np.fft.rfftfreq(n,1/rate)<=100e6)
    rows=[];num=np.zeros(3);den=np.zeros(3)
    for index,name in enumerate(names):
        frames=[next(reader) for reader in readers]
        arrays=[f[FIELDS].to_numpy() for f in frames]
        for frame,array in zip(frames,arrays):
            assert len(frame)==n and np.all(frame.observer==index) and np.isfinite(array).all()
            assert np.allclose(frame.time_s.to_numpy(),times,rtol=1e-14,atol=1e-16)
            assert np.linalg.norm(array[:,0:3]-array[:,3:6]-array[:,6:9])<=1e-13*max(np.linalg.norm(array),1e-100)
        x,z=arrays
        for group in range(3):
            sl=slice(group*3,group*3+3)
            num[group]+=np.sum((x[:,sl]-z[:,sl])**2);den[group]+=np.sum(z[:,sl]**2)
        filtered=[np.fft.irfft(np.fft.rfft(a,axis=0)*band[:,None],n=n,axis=0) for a in arrays]
        x,z=filtered;vector=np.linalg.norm(z[:,:3],axis=1);peak=int(np.argmax(vector));axis=int(np.argmax(abs(z[peak,:3])))
        qair=float(np.sum(z[:,3:6]**2)/rate);qrock=float(np.sum(z[:,6:9]**2)/rate);qtotal=float(np.sum(z[:,:3]**2)/rate)
        cross=float(2*np.sum(z[:,3:6]*z[:,6:9])/rate)
        assert abs(qtotal-qair-qrock-cross)<1e-12*max(qair+qrock+qtotal,1e-100)
        norm=np.linalg.norm(z[:,:3]);delta=np.linalg.norm(x[:,:3]-z[:,:3])
        rows.append(dict(station=name,peak_50_100MHz_V_m=float(vector[peak]),peak_time_us=float(times[peak]*1e6),
            air_Q=qair,rock_Q=qrock,total_Q=qtotal,interference_Q=cross,
            rock_self_fraction=qrock/(qair+qrock) if qair+qrock else None,
            raw_peak_V_m=float(np.max(np.linalg.norm(arrays[1][:,:3],axis=1))),
            filtered_algorithm_relative_l2=float(delta/norm) if norm else None))
        if plots:
            fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
            stride=max(1,n//8000);zoom=slice(max(0,peak-100),min(n,peak+101))
            # Full-window drawing is downsampled only for display; peak and all
            # acceptance metrics above use every sample. Zoom uses full resolution.
            display=np.unique(np.r_[np.arange(0,n,stride),peak])
            for a,label,style in [(z,'ZHS','-'),(x,'CoREAS','--')]:
                axes[0,0].plot(times[display]*1e6,a[display,axis]*1e6,style,label=label,lw=.9)
                axes[0,1].plot(times[zoom]*1e6,a[zoom,axis]*1e6,style,label=label,lw=1.2)
            for start,label,color in [(3,'Air source','steelblue'),(6,'Rock source','darkorange'),(0,'Coherent total','black')]:
                axes[1,0].plot(times[display]*1e6,np.linalg.norm(z[display,start:start+3],axis=1)*1e6,label=label,color=color,lw=.9)
                axes[1,1].plot(times[zoom]*1e6,z[zoom,start+axis]*1e6,label=label,color=color,lw=1)
            for ax in axes.flat:ax.set(xlabel='Arrival time [us]',ylabel='Electric field [uV/m]');ax.legend(fontsize=8);ax.grid(alpha=.2)
            axes[0,0].set_title('Full window (display decimated)');axes[0,1].set_title('Strongest pulse (all samples)')
            axes[1,0].set_ylabel('Vector field magnitude [uV/m]')
            fig.suptitle(name+' | 50-100 MHz | '+folder.name)
            fig.savefig(out/'figures'/('waveform_'+name+'.png'),dpi=130);plt.close(fig)
        print('RADIO',name,flush=True)
    for reader in readers:
        try:next(reader)
        except StopIteration:continue
        raise AssertionError('Extra observer rows')
    relative=np.sqrt(np.divide(num,den,out=np.zeros(3),where=den>0))
    assert np.all(relative<1e-5) and np.all(num[den==0]==0)
    pd.DataFrame(rows).to_csv(out/'station_signals.csv',index=False)
    return dict(observer_count=80,rows_per_algorithm=80*n,relative_l2_by_source={k:float(v) for k,v in zip(['total','air','rock'],relative)},stations=rows)


def figures(root,folder,out,summary,tau,signals):
    with np.load(root/'bundle/terrain_grid.npz') as grid:
        e=grid['east_m'][::3,::3];n=grid['north_m'][::3,::3];z=grid['up_m'][::3,::3]
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    im=axes[0].pcolormesh(e/1000,n/1000,z,cmap='terrain',shading='auto');fig.colorbar(im,ax=axes[0],label='Terrain ENU Up [m]')
    station=summary['scene']['radio']['observers'];xyz=np.array([s['position_enu_m'] for s in station])
    axes[0].scatter(xyz[:,0]/1000,xyz[:,1]/1000,s=7,c='purple',label='All 80 stations')
    p=summary['position_m'];d=summary['direction'];axes[0].scatter(p[0]/1000,p[1]/1000,c='red',marker='*',s=90,label='Injection')
    axes[0].arrow(p[0]/1000,p[1]/1000,d[0],d[1],width=.025,color='red')
    for vertices,label,color in [(summary['neutrino'].get('interactions',[]),'Neutrino vertex','black'),(summary['tau'].get('decays',[]),'Tau decay','cyan')]:
        points=np.asarray([v['position_enu_m'] for v in vertices])
        if len(points):axes[0].scatter(points[:,0]/1000,points[:,1]/1000,s=45,marker='x',color=color,label=label)
    axes[0].set(xlabel='East [km]',ylabel='North [km]',title='Real DEM and recorded vertices',aspect='equal');axes[0].legend(fontsize=7)
    with np.load(out/'profile.npz') as profile:
        edges=profile['edges_m'];energy=profile['deposited_GeV'];centres=.5*(edges[:-1]+edges[1:])
    active=np.flatnonzero(energy>0)
    axes[1].plot(centres/1000,energy/np.diff(edges)*1000)
    if len(active):axes[1].set_xlim(centres[active[0]]/1000-.1,centres[active[-1]]/1000+.1)
    axes[1].set(xlabel='Distance along injection axis [km]',ylabel='Deposited energy [GeV/km]',title='All interior deposits; 25 m bins');axes[1].grid(alpha=.2)
    fig.suptitle(folder.name);fig.savefig(out/'figures/geometry_profile.png',dpi=160);plt.close(fig)
    rows=signals['stations'];peak=[r['peak_50_100MHz_V_m'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax in axes:ax.set(xlabel='East [km]',ylabel='North [km]',aspect='equal')
    im=axes[0].scatter(xyz[:,0]/1000,xyz[:,1]/1000,c=peak,s=35);fig.colorbar(im,ax=axes[0],label='Peak 50-100 MHz field [V/m]');axes[0].set_title('All 80 stations')
    fraction=[r['rock_self_fraction'] if r['rock_self_fraction'] is not None else np.nan for r in rows]
    im=axes[1].scatter(xyz[:,0]/1000,xyz[:,1]/1000,c=fraction,s=35,vmin=0,vmax=1);fig.colorbar(im,ax=axes[1],label='Rock Q / (air Q + rock Q)');axes[1].set_title('Emission-medium contribution')
    fig.savefig(out/'figures/array_radio.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--control',action='store_true');args=p.parse_args()
    assert socket.gethostname()=='psrpku2025'
    root=args.root;folder=args.folder;out=root/'report'/folder.name;out.mkdir(exist_ok=False);(out/'figures').mkdir()
    s=yaml.safe_load((folder/'output/terrain_run.yaml').read_text());d=s['diagnostics'];r=s['radio_result'];a=s['accelerator']
    gates=dict(complete=s['complete'],boundary=d['dem_coverage_boundary'],all_tracks=not d['csv_truncated'],all_deposits=not d['deposition_csv_truncated'],
        all_exits=not d['domain_exit_rows_truncated'],pending_empty=a['pending_particles']==0,medium=d['material_mismatches']==0,
        energy=abs(s['energy_ledger']['unexplained_over_initial'])<1e-8,
        radio=r['complete'] and r['errors']==0 and r['out_of_window']==0,
        paired=r['algorithms']==['CoREAS','ZHS'] and r['downloads']==1 and r['moment_arrays']==3,
        openmp=a['execution_space']=='OpenMP' and a['execution_concurrency']==256 and r['execution_space']=='OpenMP',
        natural=not s['tau']['force_decay_called'] and not s['neutrino']['forced_vertex_is_conditional'])
    for v in s['neutrino'].get('interactions',[]):
        assert v['vertex_audit']['conservation']['corsika_max_component_relative_residual']<1e-8
        assert v['vertex_audit']['initial_charge_e']==v['vertex_audit']['final_charge_e']
    for v in s['tau'].get('decays',[]):
        assert abs(v['relative_energy_residual'])<2e-6 and v['relative_momentum_residual']<2e-6
        assert abs(sum(x['energy_GeV'] for x in v['daughters'])/v['energy_GeV']-1)<2e-6
    assert all(gates.values()),gates
    if not args.control:
        expected=yaml.safe_load((root/'bundle/silica_SiO2_expected.yaml').read_text())
        assert s['resolved_material']['transport']==expected['transport'] and s['resolved_material']['radio']==expected['radio']
        assert s['resolved_material']['magnetic_field_enu_T']==expected['magnetic_field_enu_T']
        assert s['emcut_GeV']==.0005 and s['hadcut_GeV']==s['mucut_GeV']==.3 and s['emthin']==1e-5
        assert math.isclose(s['max_weight'],.5e-5*s['energy_GeV'],rel_tol=1e-14)
        assert s['transport_window_ns']==500000 and all(v['photon_pair_LPM'] for v in s['proposal_materials'])
        c=json.loads((folder/'output/radio/ZHS/config.json').read_text());assert c['samples']==524288 and c['sample_rate_Hz']==256e6
    records,tau=tracks_and_profile(folder,s,out)
    signals=radio(folder,out,not args.control)
    spec=importlib.util.spec_from_file_location('doublebang_classifier',root/'code/classify_psr.py');classifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(classifier)
    classification=classifier.classify(folder,s)
    if not args.control:figures(root,folder,out,s,tau,signals)
    save(out/'acceptance.json',dict(passed=True,gates=gates,records=records,radio=signals,classification=classification,energy_ledger=s['energy_ledger']))
    lines=['**'+folder.name+'**','',
        '完整事件验收通过；double-bang 拓扑：'+('通过' if classification['topology_passed'] else '未通过')+'。拓扑通过不等于已确认接收站双脉冲。','',
        '![](figures/geometry_profile.png)','','左图看真实 DEM、全部站点和实际顶点，右图看所有域内沉积的纵向 profile。','',
        '![](figures/array_radio.png)','','左图比较 80 站 50–100 MHz 峰值，右图比较山体源自项占比。干涉项见数值表；山体/大气源不等同于第一/第二个 bang。','',
        '[全部站点数值](station_signals.csv) · [验收与顶点谱系](acceptance.json)','','每个站的图：上排比较 CoREAS/ZHS，下排比较山体源、大气源与相干总场。全窗绘图降低显示点数，峰值、验收和峰值局部图保留全部采样。','']
    for row in signals['stations']:lines.append('- [%s](figures/waveform_%s.png)'%(row['station'],row['station']))
    (out/'README_CN.md').write_text('\n'.join(lines)+'\n')
    print('ACCEPTED',folder.name,records,flush=True)


if __name__=='__main__':main()
