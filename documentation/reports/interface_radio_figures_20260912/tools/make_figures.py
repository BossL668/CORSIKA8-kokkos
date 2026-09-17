#!/usr/bin/env python3
"""Independent continuous-ray check, real DEM sections and simple figures. PSR only."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';F=ROOT/'figures';F.mkdir(exist_ok=True)
STAGE=ROOT.parent;OLD=STAGE/'interface-radio-report-20260912'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
 'savefig.dpi':190,'pdf.fonttype':42})
BLUE='#1666ab';ORANGE='#de7825';ROCK='#cfb493';GREEN='#198579';C=299792458.
selected=json.loads((D/'selected_ray.json').read_text())
rp=list(csv.DictReader((D/'radiopropa_summary.csv').open()))
cfg=json.loads((STAGE/'interface-radio-pair-20260912/app-cuda-p1/electron_rock_resident/radio/CoREAS/config.json').read_text())
med=cfg['media'][0];table=np.array(med['radial_index']);center=np.array(med['center_m']);radius=med['reference_radius_m']
def unit(v):return v/np.linalg.norm(v)
def vec(r,p):return np.array([float(r[p+a]) for a in 'xyz'])
def index_grad(p):
 r=np.linalg.norm(p-center);h=r-radius
 j=np.clip(np.searchsorted(table[:,0],h,side='right')-1,0,len(table)-2)
 a,b=table[j],table[j+1];g=(b[1]-a[1])/(b[0]-a[0]);return a[1]+g*(h-a[0]),g*(p-center)/r
def loadpath(case,mode):
 a=np.genfromtxt(D/(case+'_'+mode+'.csv'),delimiter=',',names=True)
 return np.column_stack([a[k] for k in ['x','y','z']]),a['time_s']
def reference(case):
 r=selected[case];S,O,P,N,w=[vec(r,p) for p in ['s','o','p','n','e']]
 rock=int(r['region']);screen=unit(O-P);u=unit(np.cross(w,[0,0,1] if abs(w[2])<.9 else [0,1,0]));v=unit(np.cross(w,u))
 def trace(a,b,dense=False):
  launch=unit(w+a*u+b*v);Q=S.copy();time=0.;direction=launch
  if rock:
   L=np.dot(P-S,N)/np.dot(launch,N);Q=S+launch*L
   n,_=index_grad(Q);tangent=2/n*(launch-N*np.dot(launch,N))
   direction=tangent+N*np.sqrt(1-np.dot(tangent,tangent));time=2*L/C
  def rhs(s,y):
   n,g=index_grad(y[:3]);d=unit(y[3:6])
   return np.r_[d,(g-d*np.dot(g,d))/n,n/C]
  def event(s,y):return np.dot(y[:3]-O,screen)
  event.terminal=True;event.direction=1
  sol=solve_ivp(rhs,[0,np.linalg.norm(O-Q)*1.1],np.r_[Q,direction,time],
                method='DOP853',rtol=2.e-12,atol=1.e-12,max_step=10.,events=event,dense_output=dense)
  assert len(sol.t_events[0])==1
  return sol,Q,launch
 a=b=0.
 for it in range(8):
  sol,Q,launch=trace(a,b);err=sol.y[:3,-1]-O
  if np.linalg.norm(err)<1.e-8:break
  h=1.e-6
  da=(trace(a+h,b)[0].y[:3,-1]-trace(a-h,b)[0].y[:3,-1])/(2*h)
  db=(trace(a,b+h)[0].y[:3,-1]-trace(a,b-h)[0].y[:3,-1])/(2*h)
  dx=np.linalg.solve([[da@u,db@u],[da@v,db@v]],[err@u,err@v]);a-=dx[0];b-=dx[1]
 sol,Q,launch=trace(a,b,True);grid=np.linspace(0,sol.t[-1],5001);y=sol.sol(grid)
 assert np.linalg.norm(y[:3,-1]-O)<2.e-8
 points=y[:3].T
 if rock:points=np.vstack([S,points])
 np.savetxt(D/(case+'_scipy_reference.csv'),np.column_stack([y[:3].T,y[6]]),delimiter=',',header='x,y,z,time_s',comments='')
 return dict(points=points,flight_s=float(y[6,-1]),miss_m=float(np.linalg.norm(y[:3,-1]-O)),interface=Q,launch=launch)
refs={case:reference(case) for case in selected}
metrics={'case':'1 GeV electron; seed 67101; actual saved CUDA shower','rays':{}}
for case,r in selected.items():
 S,O,P,N,w=[vec(r,p) for p in ['s','o','p','n','e']];screen=unit(O-P)
 points,_=loadpath(case,'native_shoot');base=(points-P)@screen
 transverse=points-(P+base[:,None]*screen)
 if int(r['region']):base=base[1:];transverse=transverse[1:];points=points[1:]
 reference_points=refs[case]['points'][int(r['region']):]
 reference_s=(reference_points-P)@screen
 matching=np.column_stack([np.interp(base,reference_s,reference_points[:,k]) for k in range(3)])
 reference_difference=np.max(np.linalg.norm(points-matching,axis=1))
 frozen=next(x for x in rp if x['case']==case and x['mode']=='frozen_forward' and x['max_step_m']=='10')
 forward=next(x for x in rp if x['case']==case and x['mode']=='native_forward' and x['max_step_m']=='10')
 shot=next(x for x in rp if x['case']==case and x['mode']=='native_shoot' and x['max_step_m']=='10')
 # Source and receiver are fixed for the main comparison. Forward miss is separate.
 metrics['rays'][case]=dict(track_step=int(r['step']),observer=r['observer'],
  beta5_flight_s=float(r['flight_s']),beta5_arrival_s=float(r['arrival_s']),
  maximum_path_deviation_m=float(np.max(np.linalg.norm(transverse,axis=1))),
  radiopropa_scipy_max_position_difference_m=float(reference_difference),
  frozen_endpoint_miss_m=float(frozen['miss_m']),native_same_launch_endpoint_miss_m=float(forward['miss_m']),
  radiopropa_shoot_endpoint_miss_m=float(shot['miss_m']),scipy_endpoint_miss_m=refs[case]['miss_m'],
  radiopropa_flight_s=float(shot['flight_s']),scipy_flight_s=refs[case]['flight_s'])
 assert reference_difference<1.e-5,(case,reference_difference)
 assert float(frozen['miss_m'])<1.e-8

# Slice the original triangle mesh, without fitting/smoothing the mountain outline.
mesh=STAGE/'psr_inputs/terrain_enu.ply'
with mesh.open('rb') as f:
 while True:
  line=f.readline().decode().strip()
  if line.startswith('element vertex'):nv=int(line.split()[-1])
  if line.startswith('element face'):nf=int(line.split()[-1])
  if line=='end_header':break
 vertices=np.fromfile(f,dtype='<f8',count=nv*3).reshape(nv,3)
 faces=np.fromfile(f,dtype=np.dtype([('count','u1'),('v','<u4',(3,))]),count=nf)['v']
tri=vertices[faces];normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);top=tri[normals[:,2]>0]
def section(S,O,lo,hi):
 horizontal=unit(np.r_[O[:2]-S[:2],0]);side=np.cross(horizontal,[0,0,1])
 distance=(top-S)@side
 use=(distance.min(axis=1)<=0)&(distance.max(axis=1)>=0)
 segments=[]
 for triangle,dist in zip(top[use],distance[use]):
  points=[]
  for j in range(3):
   k=(j+1)%3
   if dist[j]*dist[k]<0:points.append(triangle[j]+(triangle[k]-triangle[j])*(-dist[j])/(dist[k]-dist[j]))
   elif abs(dist[j])<1.e-10:points.append(triangle[j])
  if len(points)>=2:
   x=np.array([(pt-S)@horizontal for pt in points]);z=np.array([pt[2] for pt in points])
   ids=np.argsort(x);segments.append([x[ids[0]],z[ids[0]],x[ids[-1]],z[ids[-1]]])
 grid=np.linspace(lo,hi,4001);height=np.full_like(grid,np.nan)
 for x0,z0,x1,z1 in segments:
  if x1-x0<1.e-9:continue
  mask=(grid>=x0-1.e-8)&(grid<=x1+1.e-8)
  height[mask]=z0+(z1-z0)*(grid[mask]-x0)/(x1-x0)
 assert np.isfinite(height).all()
 return horizontal,grid,height,np.array(segments)
def save(fig,name):
 fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
# 1. The real full mountain cross-section and accepted E01 path.
r=selected['air_to_E01'];S,O,P=[vec(r,p) for p in ['s','o','p']]
distance=np.linalg.norm(O[:2]-S[:2]);horizontal,x,z,segments=section(S,O,-250,distance+300)
fig,ax=plt.subplots(figsize=(12.6,5.4),constrained_layout=True)
ax.set_facecolor('#eff7fc');bottom=min(z)-170
ax.fill_between(x,z,bottom,color=ROCK,label='Original DEM terrain');ax.plot(x,z,color='#866347',lw=1.4)
ax.plot([0,distance],[S[2],O[2]],color=BLUE,lw=3,label='beta5 ray')
path,_=loadpath('air_to_E01','native_shoot')
ax.plot((path-S)@horizontal,path[:,2],color=ORANGE,lw=1.8,ls='--',label='RadioPropa: actual n(h)')
ax.scatter([0,distance],[S[2],O[2]],s=[90,100],c=[GREEN,'#a42d30'],zorder=5,marker='o')
ax.annotate('Emission point S: electron track 746',xy=(0,S[2]),xytext=(90,S[2]-80),arrowprops={'arrowstyle':'-','color':GREEN},color=GREEN,fontsize=12)
ax.annotate('Observer E01',xy=(distance,O[2]),xytext=(distance-290,O[2]+190),arrowprops={'arrowstyle':'->','color':'#a42d30'},color='#a42d30',fontsize=13)
ax.text(distance*.48,S[2]*.60,'Path length 2.363 km\nTravel time 7.886 μs',ha='center',fontsize=13,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
ax.set(xlabel='Horizontal distance toward E01 (m)',ylabel='Local ENU height (m)',
       title='Actual DEM cross-section and a ray to E01',xlim=(x[0],x[-1]),ylim=(bottom,max(S[2]+180,max(z)+80)))
ax.legend(loc='upper right',frameon=True);ax.set_aspect('equal',adjustable='box')
save(fig,'01_real_mountain_path')
np.savetxt(D/'terrain_section_E01.csv',np.c_[x,z],delimiter=',',header='horizontal_m,terrain_height_enu_m',comments='')

# 2. Same real shower's rock-to-air transmitted branch, all scales explicitly shown.
r=selected['rock_to_offset'];S,O,P,N,E,R=[vec(r,p) for p in ['s','o','p','n','e','r']]
h,x,z,segments=section(S,O,-650,650)
fig,axes=plt.subplots(1,2,figsize=(12.6,5.6),gridspec_kw={'width_ratios':[1.15,1]},constrained_layout=True)
ax=axes[0];bottom=min(z)-100;ax.set_facecolor('#eff7fc');ax.fill_between(x,z,bottom,color=ROCK);ax.plot(x,z,color='#866347')
ax.plot([0,(P-S)@h,(O-S)@h],[S[2],P[2],O[2]],color=BLUE,lw=2.5)
ax.scatter([0,100],[S[2],O[2]],s=65,c=[GREEN,'#a42d30'],zorder=5)
ax.annotate('Diagnostic observer\n(100, 0, 1000) m',xy=(100,1000),xytext=(210,790),arrowprops={'arrowstyle':'->'},fontsize=11)
ax.annotate('Emission and exit points\noverlap at this scale',xy=(0,0),xytext=(-540,380),arrowprops={'arrowstyle':'->'},fontsize=11)
ax.set(title='Full path: vertical terrain section',xlabel='Horizontal distance (m)',ylabel='Local ENU height (m)',xlim=(-650,650),ylim=(bottom,1100))
ax.set_aspect('equal',adjustable='box')
ax=axes[1];tangent=unit(E-N*np.dot(E,N));sx=(S-P)@tangent*1000;sy=(S-P)@N*1000
ax.axhspan(-12,0,color=ROCK);ax.axhspan(0,14,color='#eff7fc');ax.axhline(0,color='#866347',lw=2,label='DEM face')
ax.axvline(0,color='gray',ls=':',lw=1.3,label='Surface normal')
ax.plot([sx,0,R@tangent*13/(R@N)],[sy,0,13],color=BLUE,lw=3,label='beta5')
rr=next(x for x in rp if x['case']=='rock_to_offset' and x['mode']=='frozen_forward' and x['max_step_m']=='10')
rpdir=vec(rr,'r');ax.plot([0,rpdir@tangent*13/(rpdir@N)],[0,13],color=ORANGE,lw=2,ls='--',label='RadioPropa')
ax.scatter([sx,0],[sy,0],s=65,c=[GREEN,ORANGE],zorder=5)
ax.annotate('Emission point S',xy=(sx,sy),xytext=(-7.5,-10.6),fontsize=11,arrowprops={'arrowstyle':'-'})
ax.text(.5,.6,'Exit point P',fontsize=11)
inc=float(np.degrees(np.arccos(E@N)));trans=float(np.degrees(np.arccos(R@N)))
depth=-sy
ax.text(-7.6,10.5,'Incidence: %.2f°\nRefraction: %.2f°'%(inc,trans),fontsize=12,bbox={'facecolor':'white','edgecolor':'none'})
ax.text(-7.6,-4,'Rock: n = 2',fontsize=11);ax.text(3,7,'Air',fontsize=12)
ax.set(title='Interface close-up: plane of incidence',xlabel='Tangential distance from exit point (mm)',ylabel='Distance along outward normal (mm)',xlim=(-8,8),ylim=(-12,14))
ax.set_aspect('equal',adjustable='box');ax.legend(loc='lower right',fontsize=9)
save(fig,'02_real_refraction')
metrics['rays']['rock_to_offset'].update(normal_depth_mm=float(depth),incidence_deg=inc,transmission_deg=trans,
 section_projection_offset_mm=float(abs(P[1])*1000),selected_face=int(r['face']))
np.savetxt(D/'terrain_section_rock.csv',np.c_[x,z],delimiter=',',header='horizontal_m,terrain_height_enu_m',comments='')

# 3. Both algorithms connect exactly the same source and observer.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),constrained_layout=True)
for ax,(case,r) in zip(axes,selected.items()):
 P,O=[vec(r,p) for p in ['p','o']];w=unit(O-P)
 points,_=loadpath(case,'native_shoot');points=points[int(r['region']):]
 along=(points-P)@w;error=np.linalg.norm(points-(P+along[:,None]*w),axis=1)
 ref=refs[case]['points'][int(r['region']):];rx=(ref-P)@w;re=np.linalg.norm(ref-(P+rx[:,None]*w),axis=1)
 ax.plot([0,np.linalg.norm(O-P)],[0,0],color=BLUE,lw=2,label='beta5 straight ray')
 ax.plot(along,error*1000,color=ORANGE,lw=2.5,label='RadioPropa curved ray')
 ix=np.linspace(0,len(rx)-1,18).astype(int);ax.plot(rx[ix],re[ix]*1000,'o',color='#333333',mfc='none',ms=4,label='Independent SciPy ODE')
 ax.set(xlabel='Distance along beta5 air leg (m)',ylabel='Perpendicular distance from beta5 ray (mm)',
 title='To E01: fixed source and observer' if case=='air_to_E01' else 'After rock exit: fixed source and observer')
 ax.text(.04,.92,'Maximum difference: %.3f mm'%(error.max()*1000),transform=ax.transAxes,va='top',fontsize=13)
 ax.set_ylim(-.1*error.max()*1000,error.max()*1000*1.3);ax.legend(loc='upper right',fontsize=10)
save(fig,'03_external_path_difference')

# 4/5. Reuse raw results from this very CUDA event and this very E01 observer.
archive=np.load(OLD/'data/electron_rock_waveforms.npz')
t=archive['time_s'];f=archive['frequency_Hz'];ec=archive['coreas_field'][2];ez=archive['zhs_field'][2]
sc=archive['coreas_spectrum'][2];sz=archive['zhs_spectrum'][2]
peak=np.argmax(np.linalg.norm(ec,axis=0));peak_t=t[peak];arrival=float(selected['air_to_E01']['arrival_s'])
fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),constrained_layout=True)
colors=[BLUE,ORANGE,GREEN]
for k in range(3):
 for ax in axes:ax.plot(t*1e6,ec[k]*1e12,color=colors[k],lw=1.25,label=['East component','North component','Up component'][k])
axes[0].axvline(arrival*1e6,color='#555555',ls=':',label='Selected ray arrival')
axes[0].set(xlim=(0,25),xlabel='Time since primary injection (μs)',ylabel='Electric field (pV/m)',title='Actual three-component waveform at E01')
axes[0].legend(fontsize=10)
axes[1].set(xlim=((peak_t-.06e-6)*1e6,(peak_t+.06e-6)*1e6),xlabel='Time (μs)',ylabel='Electric field (pV/m)',title='Zoom around the largest pulse')
axes[1].legend(fontsize=10)
save(fig,'04_E01_waveform')
np.savez_compressed(D/'E01_waveforms.npz',time_s=t,frequency_Hz=f,coreas_field=ec,zhs_field=ez,coreas_spectrum=sc,zhs_spectrum=sz)
fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),constrained_layout=True)
k=int(np.argmax(np.max(abs(ec),axis=1)))
axes[0].plot(t*1e6,ec[k]*1e12,color=BLUE,lw=2,label='CoREAS')
axes[0].plot(t*1e6,ez[k]*1e12,color=ORANGE,ls='--',lw=1.6,label='ZHS')
axes[0].set(xlim=((peak_t-.06e-6)*1e6,(peak_t+.06e-6)*1e6),xlabel='Time (μs)',ylabel='Electric field (pV/m)',title='E01 pulse: same field component')
axes[0].legend()
total=np.linalg.norm(sc);res=np.linalg.norm(sc-sz,axis=0)/max(total,1.e-100)
axes[1].semilogy(f[1:]/1e6,np.maximum(res[1:],1.e-20),color=GREEN)
l2=float(np.linalg.norm(sc-sz)/total);wave_l2=float(np.linalg.norm(ec-ez)/np.linalg.norm(ec))
axes[1].set(xlabel='Frequency (MHz)',ylabel='Complex spectral difference / total norm',title='Difference between the overlapping curves',xlim=(0,500))
axes[1].text(.05,.9,'Relative complex-spectrum L2 = %.2e'%l2,transform=axes[1].transAxes)
save(fig,'05_E01_coreas_zhs')
metrics['waveform']=dict(observer='E01',peak_time_s=float(peak_t),peak_norm_V_per_m=float(np.linalg.norm(ec[:,peak])),
 coreas_zhs_complex_l2=l2,coreas_zhs_waveform_l2=wave_l2,selected_ray_arrival_s=arrival,
 interpretation='The field coherently sums all tracks; the selected ray is one contribution, not the whole pulse.')
metrics['passed']=True
(D/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
