#!/usr/bin/env python3
"""PSR-only geometry, transmitted-ray checks and English figures."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import socket
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
assert socket.gethostname()=='psrpku2025'
base=Path('/data/yhlu/CorsikaData/corsika_validation_results')
r=base/'beta5_ne_sw_geometry_20260916';out=r/'report';(out/'figures').mkdir(parents=True,exist_ok=True)
g=json.loads((r/'selected.json').read_text());native=json.loads((r/'native.json').read_text())
oracle=json.loads((r/'independent.json').read_text())
assert native['passed'] and len(native['rows'])==95 and len(oracle['rows'])==10
max_error=0.
for row in oracle['rows']:
    n=next(x for x in native['rows'] if x['label']==row['label'])
    assert row['index_iteration_converged']
    assert (row['accepted'],row['blocked'])==(n['unfiltered_valid'],n['unfiltered_blocked'])
    assert {p['face'] for p in row['stationary_rays']}=={p['face'] for p in n['stationary_rays']}
    for p in row['stationary_rays']:
        q=next(x for x in n['stationary_rays'] if x['face']==p['face'])
        max_error=max(max_error,float(np.linalg.norm(np.array(p['interface_m'])-q['interface_m'])))
        assert (p['source_hit'] is None)==q['source_clear_native']
        assert (p['destination_hit'] is None)==q['destination_clear_native']
assert max_error<1e-6
all80=[x for x in native['rows'] if x['source_id']=='axis_before_exit_10m']
accepted={x['observer'] for x in all80 if x['unfiltered_valid']}
assert len(all80)==80 and len(accepted)==40
scene=yaml.safe_load((base/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916/scenes/screen.yaml').read_text())
mesh=Path(scene['geometry']['mesh_path'])
assert hashlib.sha256(mesh.read_bytes()).hexdigest()==g['mesh_sha256']
with mesh.open('rb') as f:
    header=[]
    while True:
        line=f.readline().decode().strip();header.append(line)
        if line=='end_header':break
    nv=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
    nf=int(next(x.split()[-1] for x in header if x.startswith('element face ')))
    vertices=np.fromfile(f,dtype='<f8',count=nv*3).reshape(nv,3)
    faces=np.fromfile(f,dtype=np.dtype([('count','u1'),('indices','<u4',(3,))]),count=nf)['indices']
t=vertices[faces];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);top=faces[normal[:,2]>1e-10]
used=np.unique(top);v=vertices[used]
tri=mtri.Triangulation(v[:,0],v[:,1],triangles=np.searchsorted(used,top));height=mtri.LinearTriInterpolator(tri,v[:,2])
direction=np.array(g['direction']);origin=np.array(g['position_m']);exit=np.array(g['rock_exit_m'])
entry=np.array(g['rock_entry_m']);anchor=np.array(g['anchor_m']);station=next(x for x in scene['radio']['observers'] if x['name']==g['target'])
station=np.array(station['position_enu_m']);end=np.array(g['next_ground_or_coverage_point_m'])
s=np.unique(np.r_[np.linspace(-g['rock_length_m']-100,g['air_after_exit_within_dem_m']+100,8000),-g['rock_length_m'],0.,g['distance_to_array_target_after_exit_m'],g['air_after_exit_within_dem_m']])
p=exit+s[:,None]*direction;terrain=np.asarray(height(p[:,0],p[:,1]))
air=(s>1e-5)&(s<g['air_after_exit_within_dem_m']-1e-5)
rock=(s>-g['rock_length_m']+1e-5)&(s<-1e-5)
assert np.all(p[air,2]>terrain[air]) and np.all(p[rock,2]<terrain[rock])
assert np.max(abs(np.array([float(height(*a[:2]))-a[2] for a in [entry,exit,end]])))<1e-6
checks=dict(passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    native_queries=95,independent_queries=10,maximum_independent_interface_difference_m=max_error,
    valid_station_count_at_10m=40,blocked_station_count_at_10m=40,
    candidate_filters_agree=True,independent_visibility_agrees=True,
    exact_axis_intersections_verified=True,dense_axis_medium_check=True,
    source_scope='Synthetic rock points on the proposed axis; not shower tracks or received signal predictions.')
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'legend.fontsize':9})
fig,axes=plt.subplots(1,2,figsize=(15,7),constrained_layout=True)
xx,yy=np.meshgrid(np.linspace(-4200,4200,420),np.linspace(-3100,5300,420));hh=height(xx.ravel(),yy.ravel()).reshape(xx.shape)
im=axes[0].pcolormesh(xx/1000,yy/1000,hh,cmap='terrain',shading='auto',vmin=-300,vmax=1050)
fig.colorbar(im,ax=axes[0],label='DEM ENU Up [m]',shrink=.8)
axes[0].plot([origin[0]/1000,end[0]/1000],[origin[1]/1000,end[1]/1000],color='crimson',lw=2,label='New NE-to-SW axis')
axes[0].annotate('',xy=anchor[:2]/1000,xytext=exit[:2]/1000,arrowprops=dict(arrowstyle='->',color='crimson',lw=2))
for flag,color,label in [(False,'#a4a4a4','Blocked from rock test source'),(True,'#003db8','Valid rock-to-air ray')]:
    st=np.array([x['position_enu_m'] for x in scene['radio']['observers'] if (x['name'] in accepted)==flag])
    axes[0].scatter(st[:,0]/1000,st[:,1]/1000,s=22,color=color,edgecolors='white',linewidths=.3,label=label,zorder=5)
for point,color,marker,label in [(origin,'purple','*','Injection'),(entry,'black','o','Rock entry'),(exit,'darkorange','D','Rock exit')]:
    axes[0].scatter(point[0]/1000,point[1]/1000,c=color,marker=marker,s=65,zorder=7,label=label)
axes[0].annotate('N10',station[:2]/1000,xytext=(-30,15),textcoords='offset points')
axes[0].set(xlabel='East [km]',ylabel='North [km]',aspect='equal',title='Real DEM and all 80 stations')
axes[0].legend(loc='upper left',fontsize=8)
ax=axes[1];lo=min(terrain.min(),p[:,2].min())-40;hi=max(terrain.max(),p[:,2].max())+60
ax.fill_between(s/1000,lo,terrain,color='#cbb38c',label='Rock / actual DEM')
ax.fill_between(s/1000,terrain,hi,color='#e5f3fa',label='Air')
ax.plot(s/1000,terrain,color='#756046',lw=1)
ax.plot(s/1000,p[:,2],color='crimson',label='Reference axis')
for point,loc,color,marker,label in [(origin,-g['rock_length_m']-100,'purple','*','Injection'),(entry,-g['rock_length_m'],'black','o','Entry'),(exit,0.,'darkorange','D','Exit')]:
    ax.scatter(loc/1000,point[2],c=color,marker=marker,s=65,zorder=5)
    ax.annotate(label,(loc/1000,point[2]),xytext=(6,16 if label!='Entry' else -24),textcoords='offset points',fontsize=9)
target_s=g['distance_to_array_target_after_exit_m'];ax.scatter(target_s/1000,station[2],marker='^',color='blue',s=55,zorder=7,label='N10 antenna')
ax.plot([target_s/1000]*2,[station[2],anchor[2]],':',color='blue')
ax.annotate('Axis 250 m above N10',(target_s/1000,anchor[2]),xytext=(10,18),textcoords='offset points',fontsize=9)
ax.annotate('',xy=(g['air_after_exit_within_dem_m']/1000,420),xytext=(0,420),arrowprops=dict(arrowstyle='<->',color='darkgreen'))
ax.text(3.5,445,'7.062 km continuous air after exit',ha='center',color='darkgreen')
ax.text(.02,.07,'Rock chord: 874.69 m\nInjection: 100 m before rock entry',transform=ax.transAxes)
ax.set(xlabel='Axis distance from mountain exit [km]',ylabel='ENU Up [m]',ylim=(lo,hi),title='Axis azimuth 225 deg, elevation -6 deg')
ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Geometry preflight | 100 PeV neutrino setup | No shower result implied')
for ext in ['png','pdf']:fig.savefig(out/'figures'/('01_ne_sw_axis_and_21cma.'+ext),dpi=160)
plt.close(fig)

row=next(x for x in all80 if x['observer']=='N10');ray=row['stationary_rays'][0]
a=np.array(row['source_m']);b=np.array(ray['interface_m']);c=np.array(row['observer_m'])
L1=np.linalg.norm(b-a);L2=np.linalg.norm(c-b);H1=np.linalg.norm((b-a)[:2]);H2=np.linalg.norm((c-b)[:2])
t1=np.linspace(0,1,300);t2=np.linspace(0,1,2500)
path=np.r_[a+t1[:,None]*(b-a),b+t2[:,None]*(c-b)];x=np.r_[t1*H1,H1+t2*H2]
h=np.asarray(height(path[:,0],path[:,1]))
fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
lo=min(c[2],h.min())-30;hi=max(a[2],b[2],h.max())+60
ax.fill_between(x/1000,lo,h,color='#cbb38c',label='Rock / actual DEM')
ax.fill_between(x/1000,h,hi,color='#e5f3fa',label='Air')
ax.plot(x/1000,h,color='#756046',lw=1)
ax.plot([0,H1/1000],[a[2],b[2]],color='royalblue',lw=2,label='Solved rock leg')
ax.plot([H1/1000,(H1+H2)/1000],[b[2],c[2]],color='darkorange',lw=2,label='Accepted air leg to N10')
ax.scatter(0,a[2],marker='*',color='purple',s=85,zorder=5,label='Synthetic source 10 m before axis exit')
ax.scatter((H1+H2)/1000,c[2],marker='^',color='blue',s=60,zorder=5,label='Real station N10')
ax.text(.04,.12,f'Rock ray length: {L1:.3f} m\nField absorption factor: {ray["attenuation"]:.4f}\nNo obstruction on either leg',transform=ax.transAxes)
ax.set(xlabel='Horizontal distance along projected ray path [km]',ylabel='ENU Up [m]',ylim=(lo,hi),
    title='An accepted rock-to-air ray in the new geometry | Native and independent solvers agree')
ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
for ext in ['png','pdf']:fig.savefig(out/'figures'/('02_ne_sw_valid_transmission_N10.'+ext),dpi=160)
plt.close(fig)

text='''---
marp: true
paginate: true
---

东北入山，西南出山，朝向 21CMA

![w:1100](figures/01_ne_sw_axis_and_21cma.png)

左图红箭头沿粒子运动方向，方位角从北顺时针计为 **225°**；向下 **6°**。注入点位于入山前 100 m 的空气中，穿岩 **874.69 m**，随后沿轴连续 **7.062 km** 为空气，之后才再次触地。入射 ENU 位置为 (2207.376, 3905.012, 650.281) m，方向 (-0.703233, -0.703233, -0.104528)。

轴线朝向北臂，在 N10 的水平位置经过天线 **250 m 上方**；这不是把轴线直接打到天线上。保留全部 80 站原坐标和高度。蓝色站点表示出口前 10 m 的岩石测试源存在有效折射路径，灰色站点仍被地形遮挡。

---

这次能从岩石透射到真实站点吗？

![w:1100](figures/02_ne_sw_valid_transmission_N10.png)

这条路径从轴线上出口前 10 m 的岩石测试源出发，折射后到达真实 N10，岩石段和空气段均无遮挡。对同一个测试源检查 80 站：**40 站有效透射，40 站有面内根但被遮挡**。95 组正常筛选/逐面/关闭预筛检查一致；10 组独立求根和相交检查一致。

这是几何预检，测试源不是实际 shower 轨迹，也不代表预测接收幅度。初始方向改变后必须重新筛选自然 CC→τ→空气中非 μ 衰变，并在 1e-3 薄化和最终射电阶段分别验收，不能沿用旧方向种子的 double bang 结论。
'''
(out/'FIGURES_CN.md').write_text(text)
(out/'README_CN.md').write_text('入射几何已完成真实 DEM 和射电传播预检。\n\n[两页 PPT 图解](FIGURES_CN.md) · [精确坐标与交点](selected.json) · [数值检查](checks.json)\n\n从东北向西南（方位角 225°），向下 6°。连续岩石 874.69 m，出山后连续空气 7.062 km；轴线在 N10 上方 250 m 经过。出口前 10 m 的测试岩石源对真实 80 站有 40 站有效透射。全部站点位置和介质物性保持原样，仅调整入射几何。\n\n单界面分段直线几何光学模型；未包含绕射、多次穿界面、反射。这里检验的是预设几何源点，不是完整 shower 的接收信号或 double bang 结果。完整模拟使用新随机数候选和实际谱系重新验收。\n')
for name in ['selected.json','candidates.json','preferred_candidates.json','queries.json','independent_queries.json','config.json','native.json','independent.json','transmission_independent_psr.py']:
    shutil.copy2(r/name,out/name)
shutil.copy2(base/'select_ne_sw_geometry_psr.py',out/'select_ne_sw_geometry_psr.py')
shutil.copy2(__file__,out/Path(__file__).name)
(out/'SHA256SUMS').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(out).as_posix() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')+'\n')
print(json.dumps(checks),flush=True)
