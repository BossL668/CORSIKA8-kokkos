#!/usr/bin/env python3
"""Independent NumPy Snell/barycentric/triangle-segment DEM visibility oracle."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_mesh(path):
    with path.open('rb') as stream:
        nv=nf=0
        while True:
            line=stream.readline().decode('ascii').strip()
            if line.startswith('element vertex'):nv=int(line.split()[-1])
            if line.startswith('element face'):nf=int(line.split()[-1])
            if line=='end_header':break
        vertices=np.fromfile(stream,dtype='<f8',count=3*nv).reshape(nv,3)
        faces=np.fromfile(stream,dtype=np.dtype([('size','u1'),('v','<u4',(3,))]),count=nf)
        assert np.all(faces['size']==3)
        return vertices,faces['v']

def clear(a,b,tri):
    delta=b-a;length=np.linalg.norm(delta);direction=delta/length
    edge1=tri[:,1]-tri[:,0];edge2=tri[:,2]-tri[:,0]
    p=np.cross(direction,edge2);det=np.einsum('ij,ij->i',edge1,p)
    good=np.abs(det)>1.e-13
    det=np.where(good,det,1.);t=a-tri[:,0]
    u=np.einsum('ij,ij->i',t,p)/det;q=np.cross(t,edge1)
    v=q@direction/det;distance=np.einsum('ij,ij->i',edge2,q)/det
    hits=good&(u>=-1.e-10)&(v>=-1.e-10)&(u+v<=1.+1.e-10)&(distance>1.e-7)&(distance<length-1.e-7)
    return not bool(np.any(hits))

def solve(source,observer,tri,config):
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    ds=np.einsum('ij,ij->i',tri[:,0]-source,normal)
    do=np.einsum('ij,ij->i',observer-tri[:,0],normal)
    ids=np.flatnonzero((ds>1.e-10)&(do>1.e-10));n=normal[ids];ds=ds[ids];do=do[ids]
    a=source+ds[:,None]*n;b=observer-do[:,None]*n;delta=b-a;distance=np.linalg.norm(delta,axis=1)
    tangent=delta/np.maximum(distance,1.e-30)[:,None]
    m=config['media'][0];table=np.asarray(m['radial_index']);ns=config['media'][1]['index']
    def index(p):
        h=np.linalg.norm(p-np.asarray(m['center_m']),axis=-1)-m['reference_radius_m']
        return np.interp(h,table[:,0],table[:,1])
    no=index(tri[ids,0]);point=a.copy()
    for _ in range(8):
        lo=np.zeros_like(distance);hi=distance.copy()
        for _ in range(60):
            x=(lo+hi)*.5;positive=ns*x/np.hypot(ds,x)>no*(distance-x)/np.hypot(do,distance-x)
            hi=np.where(positive,x,hi);lo=np.where(positive,lo,x)
        point=a+tangent*((lo+hi)*.5)[:,None];no=index(point)
    edge1=tri[ids,1]-tri[ids,0];edge2=tri[ids,2]-tri[ids,0];q=point-tri[ids,0]
    dot=lambda a,b:np.einsum('ij,ij->i',a,b)
    aa=dot(edge1,edge1);ab=dot(edge1,edge2);bb=dot(edge2,edge2);qa=dot(q,edge1);qb=dot(q,edge2)
    det=aa*bb-ab*ab;u=(bb*qa-ab*qb)/det;v=(aa*qb-ab*qa)/det
    retained=np.flatnonzero((u>=-1.e-9)&(v>=-1.e-9)&(u+v<=1.+1.e-9))
    accepted=[];blocked=0;blocked_points=[]
    for j in retained:
        if clear(source,point[j],tri) and clear(point[j],observer,tri):
            if not any(np.linalg.norm(point[j]-np.asarray(r['interface_m']))<1.e-7 for r in accepted):
                accepted.append(dict(face=int(ids[j]),interface_m=point[j].tolist()))
        else:
            blocked+=1
            if len(blocked_points)<3:blocked_points.append(point[j].tolist())
    return dict(finite_face_candidates=len(retained),blocked=blocked,accepted=accepted,blocked_interface_m=blocked_points)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scene=yaml.safe_load(args.scene.read_text());mesh=Path(scene['geometry']['mesh_path'])
    if not mesh.is_absolute():mesh=args.scene.parent/mesh
    vertices,faces=read_mesh(mesh);tri=vertices[faces]
    config=json.loads((args.run/'radio/config.json').read_text())
    tracks=[]
    with (args.run/'terrain/tracks.csv').open() as stream:
        for r in csv.DictReader(stream):
            if abs(int(r['pdg']))==11 and r['medium']=='rock':tracks.append(r)
    picked=[tracks[0],tracks[len(tracks)//2],tracks[-1]]
    observers=[('overhead',np.array([0.,0.,1000.])),('real_station',np.asarray(scene['radio']['observers'][-1]['position_enu_m']))]
    # For older validation scenes all observers were real stations.
    if not scene['radio']['observers'][0]['name'].startswith('diagnostic'):
        observers[1]=('real_station',np.asarray(scene['radio']['observers'][0]['position_enu_m']))
    rows=[];fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    near=np.flatnonzero(np.all(np.abs(tri.mean(axis=1)[:,:2])<2500.,axis=1)&(normals[:,2]>0.))
    ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces[near],cmap='terrain',alpha=.75,linewidth=0)
    for i,r in enumerate(picked):
        source=np.array([(float(r[a])+float(r[b]))*.5 for a,b in [('x0_m','x1_m'),('y0_m','y1_m'),('z0_m','z1_m')]])
        for name,observer in observers:
            result=solve(source,observer,tri,config);rows.append(dict(track=i,source_m=source.tolist(),observer=name,observer_m=observer.tolist(),**result))
            ax.scatter(*observer,color='tab:orange' if name=='overhead' else 'tab:red')
            for ray in result['accepted']:
                line=np.stack([source,ray['interface_m'],observer]);ax.plot(*line.T,color='tab:blue')
            for point in result['blocked_interface_m']:
                line=np.stack([source,point,observer]);ax.plot(*line.T,color='tab:red',ls='--')
    ax.set(xlabel='East (m)',ylabel='North (m)',zlabel='Up (m)',title='Independent DEM rays: blue = transmitted, red dashed = blocked',
           xlim=(-2500,2500),ylim=(-2500,2500),zlim=(-500,1100))
    fig.savefig(args.output/'dem_rays.png',dpi=170);fig.savefig(args.output/'dem_rays.pdf');plt.close(fig)
    report=dict(rows=rows,overhead_visible_tracks=sum(bool(r['accepted']) for r in rows if r['observer']=='overhead'),
                all_overhead_visible=all(bool(r['accepted']) for r in rows if r['observer']=='overhead'),
                real_station_shadowed=all(not r['accepted'] for r in rows if r['observer']=='real_station'),
                scope='Three actual charged rock segments; independent vectorized Snell solve and triangle intersection, straight local-index optical model.')
    (args.output/'dem_visibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['overhead_visible_tracks'] or not report['real_station_shadowed']:raise SystemExit(1)

if __name__=='__main__':main()
