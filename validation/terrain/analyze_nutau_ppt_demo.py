#!/usr/bin/env python3
"""PPT figures from actual beta5 terrain steps; conditional CC/NC, no radio.

Do not join different histories into invented tracks. Use all recorded rows
for numerical diagnostics, with display-only decimation for 3-D drawings.
Longitudinal profiles are endpoint-reconstructed unsigned plane crossings,
not native writer output. Deposits are split uniformly over each recorded
segment in projected distance, not claimed to be exact microscopic dE/dX.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from corsika_terrain.terrain_validation import _terrain_interpolator

GROUPS = [
    ('Electrons / positrons', r'$e^\pm$', '#bd3939', lambda p: abs(p) == 11),
    ('Photons', r'$\gamma$', '#c48c14', lambda p: p == 22),
    ('Muons', r'$\mu^\pm$', '#3669ab', lambda p: abs(p) == 13),
    ('Taus', r'$\tau^\pm$', '#b835ab', lambda p: abs(p) == 15),
    ('Hadrons', 'Hadrons', '#5b536a', lambda p: ~np.isin(abs(p), [11, 12, 13, 14, 15, 16, 22])),
    ('Neutrinos', r'$\nu,\bar\nu$', '#12817a', lambda p: np.isin(abs(p), [12, 14, 16]))]
PARTICLES = {16:r'$\nu_\tau$', -16:r'$\bar\nu_\tau$', 12:r'$\nu_e$', -12:r'$\bar\nu_e$',
             14:r'$\nu_\mu$', -14:r'$\bar\nu_\mu$', 11:r'$e^-$', -11:r'$e^+$',
             13:r'$\mu^-$', -13:r'$\mu^+$', 15:r'$\tau^-$', -15:r'$\tau^+$',
             211:r'$\pi^+$', -211:r'$\pi^-$', 111:r'$\pi^0$', 22:r'$\gamma$'}


def load(path):
    return yaml.safe_load(path.read_text())


def save(fig, out, name, footer=None):
    if footer:
        fig.text(.5, .008, footer, ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .035 if footer else 0, 1, .97))
    fig.savefig(out / (name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def xyz(df, end):
    return df[[f'{c}{end}_m' for c in 'xyz']].to_numpy()


def audit(root, job):
    run = root / 'runs' / job['name']
    s = load(run / 'terrain_run.yaml')
    t = pd.read_csv(run / 'terrain/tracks.csv')
    dep = pd.read_csv(run / 'terrain/deposits.csv')
    esc = pd.read_csv(run / 'terrain/window_survivors.csv')
    resource = json.loads((root / 'runs' / (job['name'] + '_guard/resources.json')).read_text())
    a, b = xyz(t, 0), xyz(t, 1)
    d = s['diagnostics']
    checks = dict(complete=s['complete'], radio_off=s['radio'] == 'disabled',
                  openmp_only=s['accelerator']['execution_space'] == 'OpenMP',
                  no_material_mismatch=d['material_mismatches'] == 0,
                  no_track_truncation=not d['csv_truncated'] and len(t) == d['steps'],
                  no_deposit_truncation=not d['deposition_csv_truncated'],
                  finite_tracks=bool(np.isfinite(t.drop(columns='medium').to_numpy()).all()),
                  finite_deposits=bool(np.isfinite(dep.to_numpy(dtype=float)).all()),
                  finite_survivors=bool(np.isfinite(esc.to_numpy(dtype=float)).all()),
                  nondecreasing_time=bool((t.t1_s >= t.t0_s).all()),
                  window=bool((t.t1_s <= 1100e-9 + 1e-14).all()),
                  deposits_match=abs(dep.weighted_deposited_GeV.sum()-d['deposited_energy_GeV']) < 1e-5,
                  escapes_match=abs((esc.total_GeV*esc.weight).sum()-d['finite_window_escaped_total_GeV']) < 1e-5,
                  escape_count=len(esc) == d['finite_window_survivors'],
                  resource_guard=resource['returncode'] == 0 and resource['stop_reason'] is None,
                  tau_not_forced=not s['tau']['force_decay_called'])
    endpoint_errors = []
    for decay in s['tau'].get('decays', []):
        mask = t.history_id == decay['history_id']
        endpoint_errors.append(float(np.linalg.norm(b[mask]-decay['position_enu_m'], axis=1).min())
                               if mask.any() else float('inf'))
        checks['tau_P4'] = checks.get('tau_P4', True) and max(
            abs(decay['relative_energy_residual']), abs(decay['relative_momentum_residual'])) < 1e-3
        checks['tau_daughters_parent'] = checks.get('tau_daughters_parent', True) and all(
            c['parent_history_id'] == decay['history_id'] for c in decay['daughters'])
    if endpoint_errors:
        checks['tau_endpoint'] = max(endpoint_errors) < 1e-5
    interactions = s['neutrino'].get('interactions', [])
    if job['current']:
        checks['conditioned_first_current'] = bool(interactions) and interactions[0]['current'] == job['current'].upper()
        checks['forced_vertex_marked'] = s['neutrino']['force_interaction_called']
        checks['vertex_P4'] = all(v['vertex_audit']['conservation']['corsika_max_component_relative_residual'] < 1e-3 for v in interactions)
        if job['current'] == 'cc':
            checks['tau_decayed'] = s['tau']['tau_decay_count'] >= 1
        if job['current'] == 'nc':
            checks['nc_keeps_nutau'] = any(v['pdg'] == 16 for v in interactions[0]['daughters'])
    else:
        seq = t.medium.to_numpy()
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]].tolist()
        checks['transparent_control'] = not interactions and seq == ['air', 'rock', 'air']
        checks['unforced_control'] = not s['neutrino']['force_interaction_called']
        checks['transparent_energy'] = bool(np.allclose(t.E0_GeV, 10000., atol=0, rtol=1e-12) and
                                            np.allclose(t.E1_GeV, 10000., atol=0, rtol=1e-12))
        checks['transparent_no_deposit'] = d['deposited_energy_GeV'] == 0
    tau_mask = t.pdg.abs() == 15
    row = dict(name=job['name'], seed=job['seed'], checks={k:bool(v) for k,v in checks.items()},
               passed=all(checks.values()), steps=len(t), rock_steps=d['rock_steps'], air_steps=d['air_steps'],
               accelerated_steps=d['accelerated_steps'], CC_NC_count=len(interactions),
               tau_decay_count=s['tau']['tau_decay_count'],
               tau_track_length_m=float(np.linalg.norm(b[tau_mask]-a[tau_mask], axis=1).sum()),
               tau_endpoint_errors_m=endpoint_errors,
               deposited_GeV=d['deposited_energy_GeV'], window_survivor_GeV=d['finite_window_escaped_total_GeV'],
               unmodeled_neutrino_histories=d['neutrino_model_domain']['uncovered_transported_histories'],
               wall_s=resource['wall_seconds'], shower_s=s['shower_seconds'],
               peak_rss_MiB=max(x.get('rss_kib',0)/1024 for x in resource['samples']))
    return row, dict(s=s, t=t, a=a, b=b, dep=dep, esc=esc, resource=resource)


def crossing_profile(a, b, weights, planes):
    """Half-open unsigned crossings; include lower endpoint, exclude upper.

    This is a diagnostic projection of recorded segments, not exact integration
    of curved paths. Adjacent collinear steps do not double-count a plane.
    """
    lo = np.searchsorted(planes, np.minimum(a, b), side='left')
    hi = np.searchsorted(planes, np.maximum(a, b), side='left')
    delta = np.zeros(len(planes)+1)
    np.add.at(delta, lo, weights)
    np.add.at(delta, hi, -weights)
    return np.cumsum(delta)[:-1]


def segment_histogram(a, b, weights, edges):
    low, high = np.minimum(a,b), np.maximum(a,b)
    length = high-low
    point = length < 1e-12
    hist = np.histogram((a[point]+b[point])*.5, bins=edges, weights=weights[point])[0]
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        overlap = np.maximum(0., np.minimum(high[~point], right)-np.maximum(low[~point], left))
        hist[i] += np.sum(weights[~point] * overlap / length[~point])
    return hist


def display_tracks(ax, data, origin, span, three_d=True):
    a, b, t = data['a']-origin, data['b']-origin, data['t']
    inside = (np.maximum(np.abs(a), np.abs(b)) < np.asarray(span)).all(axis=1)
    # Endpoint-inside subset is deliberately not extrapolated across crop edges.
    for _, label, color, predicate in GROUPS:
        ids = np.flatnonzero(predicate(t.pdg.to_numpy()) & inside)
        ids = ids[::max(1, int(np.ceil(len(ids)/2000)))]
        if len(ids):
            seg = np.stack([a[ids], b[ids]], axis=1)
            lw = 1.6 if label in (r'$\tau^\pm$', r'$\nu,\bar\nu$') else .45
            if three_d:
                ax.add_collection3d(Line3DCollection(seg, colors=color, lw=lw, alpha=.8))
                ax.plot([],[],[], color=color, label=label)
            else:
                ax.add_collection(LineCollection(seg[:,:,[0,2]], colors=color, lw=lw, alpha=.7))
                ax.plot([],[], color=color, label=label)


def figures(root, spec, data, rows):
    out = root/'figures'; out.mkdir(exist_ok=True)
    tables = root/'plot_data'; tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':11,
                         'axes.grid':True, 'grid.alpha':.18, 'axes.spines.top':False,
                         'axes.spines.right':False, 'legend.frameon':False})
    cc, nc, control = (data[k] for k in ('conditional_cc','conditional_nc','natural_throughgoing'))
    summary = cc['s']; scene = summary['scene']; direction = np.array(spec['direction'])
    entry = np.array(spec['chord']['entry']); exit = np.array(spec['chord']['exit'])
    vertex = np.array(summary['neutrino']['interactions'][0]['position_enu_m'])
    decay = summary['tau']['decays'][0]; decay_pos = np.array(decay['position_enu_m'])
    grid_path = Path(scene['geometry']['mesh_path']).parent/'terrain_grid.npz'
    with np.load(grid_path) as grid:
        e,n,z = [grid[k] for k in ('east_m','north_m','up_m')]
    # Reuse the exact mesh diagonal convention, not a fresh Delaunay surface.
    surface = _terrain_interpolator(e,n,z)
    h0 = scene['atmosphere']['origin_altitude_asl_m']
    stations = np.array([o['position_enu_m'] for o in scene['radio']['observers']])

    fig = plt.figure(figsize=(14,6))
    ax=fig.add_subplot(121)
    mesh=ax.contourf(e[::5,::5]/1000,n[::5,::5]/1000,(z[::5,::5]+h0), levels=45, cmap='terrain')
    ax.scatter(stations[:,0]/1000,stations[:,1]/1000,c='#1e355a',s=9,label=f'{len(stations)} station centres')
    ax.scatter(vertex[0]/1000,vertex[1]/1000,c='crimson',marker='*',s=130,label='Illustration site')
    ax.set(xlabel='East [km]',ylabel='North [km]',title='21CMA region: terrain and array');ax.set_aspect('equal');ax.legend(loc='lower right',fontsize=9)
    fig.colorbar(mesh,ax=ax,label='Local Up + origin ASL [m]',shrink=.72)
    ax=fig.add_subplot(122,projection='3d')
    mask=(abs(e-vertex[0])<180)&(abs(n-vertex[1])<180)
    ax.plot_trisurf(e[mask]-vertex[0],n[mask]-vertex[1],z[mask]-vertex[2],cmap='terrain',alpha=.30,linewidth=.15)
    display_tracks(ax,cc,vertex,[180,180,65])
    incident=np.vstack([entry-50*direction,vertex])-vertex
    ax.plot(*incident.T,ls='--',c='black',lw=1.4,label='Incoming axis (not a recorded CC track)')
    ax.scatter(0,0,0,c='black',marker='*',s=85)
    ax.scatter(*(decay_pos-vertex),c='#b835ab',marker='x',s=50)
    ax.set(xlim=(-170,170),ylim=(-170,170),zlim=(-55,55),xlabel='East from CC [m]',ylabel='North from CC [m]',title='Rock ridge and actual secondary tracks')
    ax.text2D(.84,.83,'Up from CC [m]',transform=ax.transAxes,fontsize=9)
    ax.view_init(24,-55);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle(r'10 TeV $\nu_\tau$: a conditional rock-CC illustration | beta5 OpenMP')
    save(fig,out,'01_terrain_and_array','Full-resolution DEM in transport; terrain decimated only for display. Radio is off.')

    fig=plt.figure(figsize=(15,5.5)); ax=fig.add_subplot(131,projection='3d')
    display_tracks(ax,cc,vertex,[6,6,3]); ax.scatter(0,0,0,c='black',marker='*',s=80,label='CC vertex')
    ax.scatter(*(decay_pos-vertex),c='#b835ab',marker='x',s=60,label=r'$\tau$ decay')
    ax.set(xlim=(-1,6),ylim=(-1,6),zlim=(-3,3),xlabel='East from CC [m]',ylabel='North from CC [m]',title='Shower tracks inside rock')
    ax.text2D(.82,.89,'Up [m]',transform=ax.transAxes,fontsize=9);ax.view_init(25,-45);ax.legend(fontsize=8,loc='upper left')
    ax=fig.add_subplot(132)
    # Use shower-axis and vertical coordinates for a physically meaningful side view.
    for _,label,color,predicate in GROUPS:
        ids=np.flatnonzero(predicate(cc['t'].pdg.to_numpy()))
        ids=ids[::max(1,int(np.ceil(len(ids)/2000)))]
        a,b=cc['a'][ids]-vertex,cc['b'][ids]-vertex
        ax.add_collection(LineCollection(np.stack([np.column_stack((a@direction,a[:,2])),np.column_stack((b@direction,b[:,2]))],axis=1),colors=color,lw=.5,alpha=.65))
    ax.axhline(0,c='black',ls=':',lw=.7);ax.scatter(0,0,c='black',marker='*',s=70)
    ax.set(xlim=(-.5,8),ylim=(-2,2),xlabel='Distance along incident axis [m]',ylabel='Up from CC [m]',title='Longitudinal / transverse structure')
    ax=fig.add_subplot(133)
    tau=cc['t'].pdg.abs().to_numpy()==15
    aa=(cc['a'][tau]-vertex)*1000;bb=(cc['b'][tau]-vertex)*1000
    ax.add_collection(LineCollection(np.stack([aa[:,:2],bb[:,:2]],axis=1),colors='#b835ab',lw=2))
    ax.scatter(0,0,c='black',marker='*',s=90,label='CC');ax.scatter(*((decay_pos-vertex)*1000)[:2],c='#b835ab',s=50,label='Decay')
    ax.autoscale();ax.margins(.16);ax.set_aspect('equal',adjustable='datalim')
    ax.set(xlabel='East from CC [mm]',ylabel='North from CC [mm]',title='Actual tau flight (equal spatial scale)');ax.legend(fontsize=9)
    fig.suptitle('Recorded trajectories: no stretched tau path or synthetic shower cloud')
    save(fig,out,'02_shower_tracks_and_tau_zoom','At most 2,000 segments per species in drawings; every recorded row is used for diagnostics.')

    fig,axs=plt.subplots(2,2,figsize=(13,8.5))
    distance=np.linspace(-50,280,800);path=entry+distance[:,None]*direction
    heights=np.asarray(surface(path[:,0],path[:,1]))
    ax=axs[0,0];ax.fill_between(distance,heights,heights.min()-8,color='#c4a378',alpha=.9,label='Rock')
    ax.plot(distance,path[:,2],'--',color='teal',label='Incident axis')
    ax.scatter([(vertex-entry)@direction],[vertex[2]],c='black',marker='*',s=100,label='Conditioned CC')
    ax.set(xlim=(-50,280),ylim=(heights.min()-8,heights.max()+5),xlabel='Distance from rock entry [m]',ylabel='ENU Up [m]',title=f'Natural slope chord: {spec["chord"]["length_m"]:.2f} m of rock');ax.legend(fontsize=9)
    ax=axs[0,1]
    for pdgs,label,color in [([15,-15],r'$\tau$', '#b835ab'),([16,-16],r'$\nu_\tau$', '#12817a'),([12,-12,14,-14],'Other neutrinos','#73852d'),([13,-13],r'$\mu^\pm$','#3669ab')]:
        mask=cc['t'].pdg.isin(pdgs).to_numpy()
        seg=np.stack([np.column_stack(((cc['a'][mask]-entry)@direction,cc['t'].E0_GeV[mask])),np.column_stack(((cc['b'][mask]-entry)@direction,cc['t'].E1_GeV[mask]))],axis=1)
        ax.add_collection(LineCollection(seg,colors=color,lw=1.2,label=label))
    ax.axvspan(0,spec['chord']['length_m'],color='#c4a378',alpha=.2)
    ax.set(xlim=(55,400),ylim=(.1,12000),yscale='log',xlabel='Distance from rock entry [m]',ylabel='Total energy [GeV]',title='Lepton propagation after the CC vertex');ax.legend(fontsize=9)
    ax=axs[1,0];td=decay['daughters'];x=np.arange(len(td));en=[r['energy_GeV'] for r in td]
    ax.bar(x,en,color=['#12817a' if abs(r['pdg']) in (12,14,16) else '#bd3939' for r in td])
    ax.set_xticks(x,[PARTICLES.get(r['pdg'],str(r['pdg'])) for r in td])
    ax.set(ylabel='Total energy [GeV]',title='TAUOLA daughters before thinning / cuts')
    for i,energy in enumerate(en):ax.text(i,energy,f'{energy:.1f}',ha='center',va='bottom',fontsize=9)
    ax.margins(y=.2)
    ax=axs[1,1];ta=cc['t'].loc[tau].sort_values('t0_s')
    times=np.stack([ta.t0_s,ta.t1_s],axis=1)*1e9
    energies=np.stack([ta.E0_GeV,ta.E1_GeV],axis=1)
    ax.add_collection(LineCollection(np.stack([times,energies],axis=2),colors='#b835ab',lw=2))
    ax.autoscale();ax.margins(.12);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.set(xlabel='Time since conditioned CC [ns]',ylabel='Tau total energy [GeV]',title='Transport before natural tau decay')
    fig.suptitle(r'$\nu_\tau + N \rightarrow \tau^- + X$: first vertex conditioned, decay not forced')
    save(fig,out,'03_CC_tau_decay_chain','IGRF14 (2027) in air; magnetic field zero in rock. No full low-energy neutrino-regeneration claim.')

    planes=np.linspace(0,12,241);profile=pd.DataFrame({'distance_from_vertex_m':planes})
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for name,event,style in [('CC',cc,'-'),('NC',nc,'--')]:
        origin=np.array(event['s']['neutrino']['interactions'][0]['position_enu_m'])
        aa=(event['a']-origin)@direction;bb=(event['b']-origin)@direction
        for ax,(group,label,color,predicate) in zip(axs.flat,GROUPS):
            mask=predicate(event['t'].pdg.to_numpy())
            counts=crossing_profile(aa[mask],bb[mask],event['t'].weight.to_numpy()[mask],planes)
            profile[name+'_'+group]=counts
            ax.plot(planes,counts,ls=style,color=color,lw=1.5,label=name+' single event')
            ax.set(xlim=(0,12),ylim=(0,None),title=group,xlabel='Distance from conditioned vertex [m]',ylabel='Weighted plane crossings');ax.legend(fontsize=9)
    profile.to_csv(tables/'longitudinal_plane_crossings.csv',index=False)
    # Limits must include BOTH curves; setting ylim after the first curve freezes
    # autoscaling and can hide a larger second-event maximum.
    for ax in axs.flat:
        maximum=max(float(np.max(line.get_ydata())) for line in ax.lines)
        ax.set_ylim(0.,max(1.,maximum*1.12))
        assert all(np.max(line.get_ydata()) <= ax.get_ylim()[1] for line in ax.lines)
    fig.suptitle('Single-event shower development: all particle components')
    save(fig,out,'04_longitudinal_all_components','Unsigned plane crossings reconstructed from complete step endpoints; not ensemble means or particle production counts.')

    fig,axs=plt.subplots(1,3,figsize=(16,5.5))
    edges=np.linspace(-1,12,131);histdata=pd.DataFrame({'left_m':edges[:-1],'right_m':edges[1:]})
    for name,event,style in [('CC',cc,'-'),('NC',nc,'--')]:
        origin=np.array(event['s']['neutrino']['interactions'][0]['position_enu_m'])
        da,db=xyz(event['dep'],0)-origin,xyz(event['dep'],1)-origin
        weights=event['dep'].weighted_deposited_GeV.to_numpy()
        h=segment_histogram(da@direction,db@direction,weights,edges)
        histdata[name+'_GeV']=h
        axs[0].stairs(h/np.diff(edges),edges,label=f'{name}: {h.sum()/weights.sum()*100:.1f}% of deposit in view',ls=style)
    histdata.to_csv(tables/'deposition_profile.csv',index=False)
    axs[0].set(xlim=(-1,12),xlabel='Distance from conditioned vertex [m]',ylabel='Deposited energy [GeV/m]',title='Longitudinal energy deposition');axs[0].legend(fontsize=9)
    dp=(xyz(cc['dep'],0)+xyz(cc['dep'],1))*.5-vertex;dw=cc['dep'].weighted_deposited_GeV.to_numpy()
    good=dw>0
    h,xedge,yedge=np.histogram2d((dp@direction)[good],dp[:,2][good],bins=[np.linspace(-1,12,100),np.linspace(-2,2,60)],weights=dw[good])
    nonzero=h[h>0];plot=axs[1].pcolormesh(xedge,yedge,np.ma.masked_less_equal(h.T,0),cmap='magma',norm=LogNorm(vmin=nonzero.min(),vmax=nonzero.max()),rasterized=True)
    fig.colorbar(plot,ax=axs[1],label='Deposited energy per pixel [GeV]',shrink=.8)
    axs[1].set(xlabel='Distance from CC [m]',ylabel='Up from CC [m]',title='CC deposit map (segment midpoints)')
    esc=cc['esc'];groups=(esc.weight*esc.total_GeV).groupby(esc.pdg).sum()
    labels=[PARTICLES.get(int(p),str(p)) for p in groups.index]
    axs[2].bar(np.arange(len(groups)),groups.to_numpy(),color='#12817a')
    axs[2].set_xticks(np.arange(len(groups)),labels);axs[2].set(yscale='log',ylabel='Weighted total energy [GeV]',title='Survivors at the 1,100 ns time limit')
    for i,energy in enumerate(groups):
        axs[2].text(i,energy,f'{energy:.2f}',ha='center',va='bottom',fontsize=9)
    axs[2].margins(y=.18)
    save(fig,out,'05_energy_deposition_and_survivors','Deposits use recorded weights; segment-split 1-D and midpoint 2-D diagnostics. Survivor energy is NOT deposition.')

    fig,axs=plt.subplots(2,2,figsize=(13,8.5))
    t=control['t'];aa=(control['a']-entry)@direction;bb=(control['b']-entry)@direction
    for i,r in t.iterrows():
        color='#9a7044' if r.medium=='rock' else '#3d8abb'
        axs[0,0].plot([aa[i],bb[i]],[r.E0_GeV,r.E1_GeV],color=color,lw=3)
        axs[0,0].text((aa[i]+bb[i])*.5,10000.5,r.medium,ha='center',fontsize=10)
    axs[0,0].set(xlim=(-50,280),ylim=(9998,10002),xlabel='Distance from entry [m]',ylabel='Neutrino total energy [GeV]',title='Unforced control: air → rock → air')
    axs[0,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    labels=['CC','NC','Transparent'];x=np.arange(3)
    axs[0,1].bar(x-.18,[r['rock_steps'] for r in rows],.36,label='Rock',color='#aa885f')
    axs[0,1].bar(x+.18,[r['air_steps'] for r in rows],.36,label='Air',color='#5696bf')
    axs[0,1].set_xticks(x,labels);axs[0,1].set(yscale='log',ylabel='Recorded transport steps',title='Material diagnostics: zero mismatches');axs[0,1].legend()
    for name,event,color in [('CC',cc,'#bd3939'),('NC',nc,'#3669ab'),('Transparent',control,'#12817a')]:
        samples=event['resource']['samples']
        axs[1,0].plot([r['elapsed_s'] for r in samples],[r.get('rss_kib',0)/1024 for r in samples],label=name,color=color)
    axs[1,0].set(xlabel='Wall time since launch [s]',ylabel='RSS [MiB]',title='OpenMP resource monitor (no GPU)');axs[1,0].legend()
    axs[1,1].bar(x-.18,[r['wall_s'] for r in rows],.36,label='Whole process',color='#7f8eb2')
    axs[1,1].bar(x+.18,[r['shower_s'] for r in rows],.36,label='Shower loop',color='#c7a061')
    axs[1,1].set_xticks(x,labels);axs[1,1].set(ylabel='Time [s]',title='Separate geometry / model initialization');axs[1,1].legend()
    save(fig,out,'06_boundary_and_runtime_diagnostics','Four OpenMP threads requested; CPU hadronic / neutrino modules remain scalar. Not a GPU performance benchmark.')
    return dict(terrain_grid=str(grid_path),figure_count=6,
                display_only_decimation=True,statistics_use_all_rows=True)


def report(root, spec, rows, result):
    lines=['# beta5 穿山 ντ 示例：PPT 图与诊断', '',
           '## 1. 本轮是什么', '',
           '使用 **Kokkos OpenMP，4 线程，不使用显卡**。沿用原 mountain 程序的实际 DEM、'
           'ENU 原点、标准大气与 80 个天线站中心；天线仅作位置展示，不计算射电。', '',
           '三个预先指定的 10 TeV ντ 事例：岩石中条件化首个 CC、条件化首个 NC、'
           '从空气入射的不强制相互作用对照。CC/NC 图不是天然发生概率，也不是系综平均。'
           '不强制 τ 衰变，不挑选衰变道，画出的粒子段均来自本次 CSV。', '',
           '## 2. 运行与验收', '',
           '| 事例 | seed | 步数 | τ 衰变数 | 全进程 / shower 用时 [s] | 峰值 RSS [MiB] |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["seed"]} | {r["steps"]:,} | {r["tau_decay_count"]} | '
                     f'{r["wall_s"]:.2f} / {r["shower_s"]:.2f} | {r["peak_rss_MiB"]:.1f} |')
    cc=rows[0]
    lines += ['', f'全部工程门禁通过：**{result["passed"]}**。逐事件门禁见 `acceptance.json`。'
              '检查了完整结束、实际 OpenMP 后端、有限值、记录无截断、材质匹配、'
              '边界穿透、沉积/存活粒子计数、顶点四动量残差和 τ 轨迹到衰变点的连续性。', '',
              f'CC 事例 τ 的记录总飞行长度为 **{cc["tau_track_length_m"]:.6g} m**，'
              '并非跨越整座山；毫米尺度小图按真实比例显示。', '',
              '参数：EM cut 10 MeV；强子/μ/τ kinetic cut 0.3 GeV；'
              'thinning 10⁻³、max-weight 50；1 m 诊断步长上限；1,100 ns 有限模拟窗口。'
              '这是快速展示参数，未做 cut/thinning 收敛测试。岩石为 SiO₂、2.65 g/cm³；'
              'FLUKA + QGSJet-II、PROPOSAL-native、TAUOLA；空气 IGRF14/2027，岩石零磁场。', '',
              '## 3. 可直接用于 PPT 的图', '']
    captions=[('01_terrain_and_array','真实地形、阵列位置与山脊局部：虚线只是入射轴，不是补造的粒子轨迹。'),
              ('02_shower_tracks_and_tau_zoom','实际次级粒子轨迹、横向展开、毫米尺度 τ 飞行；3D 图仅显示部分轨迹段。'),
              ('03_CC_tau_decay_chain','CC → τ → 次级粒子，真实飞行时间、能量与 TAUOLA 衰变末态。'),
              ('04_longitudinal_all_components','全部组分纵向发展：由完整轨迹端点重建的带权无符号平面穿越数。CC/NC 各仅一个独立事例。'),
              ('05_energy_deposition_and_survivors','能量沉积的纵向/空间诊断与有限时间窗口末端的存活粒子能量；不是完整能量闭合。'),
              ('06_boundary_and_runtime_diagnostics','自然穿透对照、介质步数、内存与运行时间；几何/模型初始化与 shower 用时分开。')]
    for name,caption in captions:
        lines += [f'### {name[:2]}. {caption}', '', f'![{caption}](figures/{name}.png)', '']
    lines += ['## 4. 展示时必须说明的物理范围', '',
              '- 首个 CC/NC 是在真实岩石内指定位置条件化的，不附天然相互作用概率或探测率权重。',
              '- CTW 总截面适用 10⁴–10¹² GeV。低于该范围的再生 ν 继续传播但没有已验证的弱再相互作用；'
              '本轮实际遇到的范围外 history 数已记录。不能称为完整 ντ 再生链验收。',
              '- 使用原版 TAUOLA 固定 helicity 约定，不代表已验证 CC 自旋密度矩阵传递、核效应或低 Q² 末态。',
              '- 飞行时间窗口之外不继续追踪；末端能量计作 survivor，不作为沉积。'
              '靶核/静质量/薄化完整能量台账未在本轮认证，不能把 E₀ 减沉积再减存活能量解释为数值误差。',
              '- 纵向计数采用端点直线重建；沉积一维图按投影段长分配、二维图按中点放置。'
              '图示没有宣称微观精确 dE/dX。用于数字诊断的 CSV 未做轨迹抽样。', '',
              '## 5. 文件与复现', '',
              '`figures/`：PNG + PDF；`plot_data/`：绘图表；`runs/`：原始轨迹、YAML、命令与资源日志。', '',
              '以下命令在 beta5 源码根目录、`corsika_venv` 环境执行；输出目录必须不存在：', '',
              '```bash',
              'python validation/terrain/run_nutau_ppt_demo.py \\',
              f'  --reference {spec["reference"]} \\',
              f'  --binary {spec["binary"]} --threads 4 \\',
              '  --output /mnt/d/CorsikaData/corsika_validation_results/NEW_OPENMP_DEMO',
              'python validation/terrain/analyze_nutau_ppt_demo.py \\',
              '  --output /mnt/d/CorsikaData/corsika_validation_results/NEW_OPENMP_DEMO',
              '```', '', f'二进制 SHA-256：`{spec["binary_sha256"]}`。', '',
              f'DEM SHA-256：`{spec["mesh_sha256"]}`。', '']
    (root/'PPT_REPORT_CN.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();spec=load(root/'manifest.yaml')
    rows=[];data={}
    for job in spec['jobs']:
        row,event=audit(root,job);rows.append(row);data[job['name']]=event
    result=dict(passed=all(r['passed'] for r in rows),full_neutrino_physics_validated=False,
                backend='OpenMP',radio=False,events=rows)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError('Inspect acceptance.json before drawing accepted illustrations')
    # Tiny exact tests of diagnostic reconstruction, not physics tests.
    assert np.allclose(crossing_profile(np.array([0.,1.]),np.array([1.,2.]),np.ones(2),np.array([0.,1.,2.])),[1,1,0])
    assert np.allclose(segment_histogram(np.array([0.,.5]),np.array([2.,.5]),np.array([2.,3.]),np.array([0.,1.,2.])),[4.,1.])
    result['plotting']=figures(root,spec,data,rows)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    report(root,spec,rows,result)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
