#!/usr/bin/env python3
"""Independent NumPy all-face Snell bisection and segment intersections on PSR."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import json
import socket
import time
import argparse
import numpy as np

assert socket.gethostname()=='psrpku2025'
parser=argparse.ArgumentParser()
parser.add_argument('--controls',action='store_true')
parser.add_argument('--root',type=Path,default=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_1PeV_transmission_diagnostic_20260916'))
parser.add_argument('--query-file',type=Path)
parser.add_argument('--all',action='store_true')
args=parser.parse_args()
r=args.root
c=json.loads((r/'config.json').read_text())
mesh=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_nutau100PeV_openmp120_20260910_v1/bundle/terrain_enu.ply')
with mesh.open('rb') as stream:
    header=[]
    while True:
        line=stream.readline().decode().strip();header.append(line)
        if line=='end_header':break
    nv=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
    nf=int(next(x.split()[-1] for x in header if x.startswith('element face ')))
    vertices=np.fromfile(stream,dtype='<f8',count=nv*3).reshape(nv,3)
    faces=np.fromfile(stream,dtype=np.dtype([('count','u1'),('indices','<u4',(3,))]),count=nf)['indices']
all_tri=vertices[faces]
normals=np.cross(all_tri[:,1]-all_tri[:,0],all_tri[:,2]-all_tri[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
ids=np.flatnonzero(normals[:,2]>0);tri=all_tri[ids];normal=normals[ids]
e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
table=np.asarray(c['media'][0]['radial_index']);earth=np.asarray(c['media'][0]['center_m']);radius=c['media'][0]['reference_radius_m']
dot=lambda a,b:np.einsum('ij,ij->i',a,b)
def index(p):return np.interp(np.linalg.norm(p-earth,axis=-1)-radius,table[:,0],table[:,1])

def first_hit(a,b):
    delta=b-a;length=np.linalg.norm(delta);direction=delta/length
    h=np.cross(direction,e2);det=dot(e1,h);good=abs(det)>1e-13
    safe=np.where(good,det,1);relative=a-tri[:,0]
    u=dot(relative,h)/safe;q=np.cross(relative,e1);v=q@direction/safe
    distance=dot(e2,q)/safe
    hits=np.flatnonzero(good&(u>=0)&(v>=0)&(u+v<=1)&(distance>1e-7)&(distance<length-1e-7))
    if not len(hits):return None
    j=hits[np.argmin(distance[hits])]
    return dict(face=int(ids[j]),distance_m=float(distance[j]),point_m=(a+direction*distance[j]).tolist())

def solve(q):
    source=np.asarray(q['source_m']);observer=np.asarray(q['observer_m'])
    ds=dot(tri[:,0]-source,normal);do=dot(observer-tri[:,0],normal)
    good=np.flatnonzero((ds>1e-10)&(do>1e-10));ds=ds[good];do=do[good];n=normal[good]
    a=source+ds[:,None]*n;b=observer-do[:,None]*n
    tangent=b-a;tangent-=dot(tangent,n)[:,None]*n
    lateral=np.linalg.norm(tangent,axis=1);tangent/=np.maximum(lateral,1e-300)[:,None]
    n1=float(c['media'][1]['index']);n2=index(tri[good,0])
    # Bisection of whichever lateral coordinate is smaller avoids subtracting
    # nearly equal kilometre coordinates for near-boundary endpoints.
    converged=False
    for iteration in range(24):
        mid=.5*lateral
        reverse=n1*mid/np.hypot(ds,mid)<n2*mid/np.hypot(do,mid)
        dA=np.where(reverse,do,ds);dB=np.where(reverse,ds,do)
        nA=np.where(reverse,n2,n1);nB=np.where(reverse,n1,n2)
        low=np.zeros_like(lateral);high=mid.copy()
        for _ in range(70):
            t=.5*(low+high)
            positive=nA*t/np.hypot(dA,t)>nB*(lateral-t)/np.hypot(dB,lateral-t)
            high=np.where(positive,t,high);low=np.where(positive,low,t)
        t=.5*(low+high)
        x=np.where(reverse,lateral-t,t);y=np.where(reverse,t,lateral-t)
        point=np.where(reverse[:,None],b-y[:,None]*tangent,a+x[:,None]*tangent)
        next2=index(point)
        if np.max(abs(next2-n2),initial=0)<1e-13:converged=True;n2=next2;break
        n2=next2
    u1=e1[good];u2=e2[good];rel=point-tri[good,0]
    aa=dot(u1,u1);ab=dot(u1,u2);bb=dot(u2,u2);qa=dot(rel,u1);qb=dot(rel,u2)
    determinant=aa*bb-ab*ab
    u=(bb*qa-ab*qb)/determinant;v=(aa*qb-ab*qa)/determinant
    tol=1e-7/np.maximum(1,np.maximum(np.sqrt(aa),np.sqrt(bb)))
    finite=np.flatnonzero((u>=-tol)&(v>=-tol)&(u+v<=1+tol))
    rays=[]
    for j in finite:
        p=point[j];source_hit=first_hit(source,p);dest_hit=first_hit(p,observer)
        residual=n1*x[j]/np.hypot(ds[j],x[j])-n2[j]*y[j]/np.hypot(do[j],y[j])
        rays.append(dict(face=int(ids[good[j]]),interface_m=p.tolist(),
            snell_residual=float(residual),source_hit=source_hit,destination_hit=dest_hit,
            valid=source_hit is None and dest_hit is None,
            incidence_angle_deg=float(np.degrees(np.arctan2(x[j],ds[j]))),
            transmission_angle_deg=float(np.degrees(np.arctan2(y[j],do[j]))),
            critical_angle_deg=float(np.degrees(np.arcsin(n2[j]/n1)))))
    return dict(**q,side_eligible_faces=len(good),finite_face_roots=len(rays),
        accepted=sum(x['valid'] for x in rays),blocked=sum(not x['valid'] for x in rays),
        index_iteration_converged=converged,stationary_rays=rays)

labels=['axis_1.7m_E18','axis_1.7m_E20','axis_1.7m_N10','axis_1.7m_overhead_control',
        'axis_18.4m_S10','max_east_E18','old_seed946_E20','old_seed3605_E20']
queries=json.loads((args.query_file or r/('control_queries.json' if args.controls else 'queries.json')).read_text())
if args.controls or args.all:labels=[q['label'] for q in queries]
rows=[];start=time.monotonic()
for label in labels:
    q=next(x for x in queries if x['label']==label);rows.append(solve(q))
    print(label,'finite',rows[-1]['finite_face_roots'],'valid',rows[-1]['accepted'],'blocked',rows[-1]['blocked'],flush=True)
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,
        elapsed_s=time.monotonic()-start,scope='Independent bisection over every positive-z DEM face, independent barycentric inclusion and NumPy triangle intersections; same single-interface/straight-leg optical model; no native BVH or candidate-face filter.')
    (r/('independent_controls.json' if args.controls else 'independent.json')).write_text(json.dumps(result,indent=2)+'\n')
