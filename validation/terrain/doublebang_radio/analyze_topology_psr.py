#!/usr/bin/env python3
"""Plot actual high-energy vertices and transported tau segments on the input DEM."""
import json, pathlib
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')

def surface(path):
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline().decode('ascii').strip();lines.append(line)
            if line=='end_header': break
        if 'format binary_little_endian 1.0' not in lines: raise RuntimeError('Unexpected PLY format')
        nv=int(next(x.split()[-1] for x in lines if x.startswith('element vertex ')))
        nf=int(next(x.split()[-1] for x in lines if x.startswith('element face ')))
        vertices=np.fromfile(f,dtype='<f8',count=nv*3).reshape(nv,3)
        faces=np.fromfile(f,dtype=np.dtype([('count','u1'),('indices','<u4',(3,))]),count=nf)
    if not np.all(faces['count']==3): raise RuntimeError('Nontriangular PLY')
    indices=faces['indices'].astype(np.int64)
    cross=np.cross(vertices[indices[:,1]]-vertices[indices[:,0]],vertices[indices[:,2]]-vertices[indices[:,0]])
    top=indices[cross[:,2]>1e-10]
    used=np.unique(top); reduced=np.searchsorted(used,top); v=vertices[used]
    tri=mtri.Triangulation(v[:,0],v[:,1],triangles=reduced)
    return mtri.LinearTriInterpolator(tri,v[:,2])

def main():
    if not ROOT.is_dir(): raise RuntimeError('Run on PSR only')
    scene=yaml.safe_load((ROOT/'scene_radio.yaml').read_text())
    terrain=surface(pathlib.Path(scene['geometry']['mesh_path']))
    out=ROOT/'report';(out/'figures').mkdir(parents=True,exist_ok=True)
    summaries=[]
    for folder in sorted((ROOT/'runs').glob('*_seed*')):
        if not (folder/'checks.json').exists() or not (folder/'tau_tracks.json').exists(): continue
        checked=json.loads((folder/'checks.json').read_text())
        if not all(checked['checks'].values()): continue
        run=yaml.safe_load((folder/'output/terrain_run.yaml').read_text())
        tracks=json.loads((folder/'tau_tracks.json').read_text())
        cc=[v for v in run['neutrino'].get('interactions',[]) if v['current']=='CC' and any(abs(d['pdg'])==15 for d in v['daughters'])]
        decays=run['tau'].get('decays',[])
        if not cc or not decays or not tracks: raise RuntimeError('No transported CC-tau-decay topology: '+folder.name)
        # The projection is explicitly a fixed-East section, not a 3-D ray plot.
        origin=np.array(cc[0]['position_enu_m']); x=origin[0]
        north=np.linspace(min(d['position_enu_m'][1] for d in decays)-1500,origin[1]+700,1600)
        height=terrain(np.full_like(north,x),north)
        distance=(origin[1]-north)/1000
        fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
        axes[0].fill_between(distance,-1.,height/1000,color='#a99979',alpha=.65,label='Rock')
        axes[0].plot(distance,height/1000,color='#665541',lw=1,label='Input DEM surface')
        for t in tracks:
            sy=[(origin[1]-float(t['y0_m']))/1000,(origin[1]-float(t['y1_m']))/1000]
            z=[float(t['z0_m'])/1000,float(t['z1_m'])/1000]
            axes[0].plot(sy,z,color='#184e9e',lw=1.5)
            axes[1].plot(sy,[float(t['E0_GeV'])/1e6,float(t['E1_GeV'])/1e6],color='#665541' if t['medium']=='rock' else '#168ca6',lw=1)
        for i,v in enumerate(cc):
            p=np.array(v['position_enu_m']); axes[0].scatter([(origin[1]-p[1])/1000],[p[2]/1000],marker='*',s=140,color='#bd3c33',label='CC vertex' if i==0 else None,zorder=5)
        ds=[]
        for i,d in enumerate(decays):
            p=np.array(d['position_enu_m']);pdgs=[abs(v['pdg']) for v in d['daughters']]
            mode='muonic control' if 13 in pdgs else 'electronic cascade' if 11 in pdgs else 'hadronic cascade'
            axes[0].scatter([(origin[1]-p[1])/1000],[p[2]/1000],marker='D',s=55,color='#743e99',label='Tau decay' if i==0 else None,zorder=5)
            axes[0].annotate(mode,((origin[1]-p[1])/1000,p[2]/1000),xytext=(5,14+16*i),textcoords='offset points',fontsize=9)
            ds.append(dict(mode=mode,medium=d['medium'],cc_to_decay_distance_km=float(np.linalg.norm(p-origin)/1000),cc_to_decay_time_us=float(d['time_ns']/1000-cc[0]['time_s']*1e6),tau_energy_PeV=d['energy_GeV']/1e6))
        axes[0].set(xlabel='Southward distance from first CC vertex (km)',ylabel='ENU height (km)',ylim=(max(-1.,float(np.ma.min(height))/1000-.1),max(1.,max(d['position_enu_m'][2] for d in decays)/1000+.2)),title='Actual transported tau and vertices | fixed East = %.3f km'%(x/1000))
        axes[0].legend(loc='lower left');axes[0].grid(alpha=.15)
        axes[1].plot([],[],color='#665541',label='Tau in rock');axes[1].plot([],[],color='#168ca6',label='Tau in air')
        axes[1].set(xlabel='Southward distance from first CC vertex (km)',ylabel='Tau total energy (PeV)',title='Actual transport energy loss; no fitted decay position')
        axes[1].grid(alpha=.2);axes[1].legend()
        fig.suptitle(folder.name+' | 100 PeV nu_tau | vertices do not establish resolved radio double pulses')
        for ext in ['png','pdf']:fig.savefig(out/'figures'/(folder.name+'_topology.'+ext),dpi=180)
        plt.close(fig)
        summaries.append(dict(run=folder.name,decays=ds,tau_segments=len(tracks),maximum_east_projection_offset_m=max(abs(float(t[k])-x) for t in tracks for k in ['x0_m','x1_m'])))
    (out/'topology.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(dict(topologies=len(summaries))))
if __name__=='__main__':main()
