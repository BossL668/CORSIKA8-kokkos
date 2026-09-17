#!/usr/bin/env python3
"""PSR-only longitudinal profiles, aligned with the actual DEM and axis grammage."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np


class CrossingProfile:
    """Collect weighted forward/backward plane crossings during the existing audit."""
    def __init__(self, summary):
        self.origin=np.array(summary['position_m']);self.direction=np.array(summary['direction'])
        self.direction/=np.linalg.norm(self.direction)
        limit=np.ceil(299792458*summary['transport_window_ns']*1e-9+100)
        vertices=summary['neutrino'].get('interactions',[])+summary['tau'].get('decays',[])
        near=[(np.array(v['position_enu_m'])-self.origin)@self.direction+np.arange(-20.,100.,.1) for v in vertices]
        self.planes=np.unique(np.concatenate([np.arange(-limit,limit+10,10.)]+near))
        self.delta=np.zeros((2,4,len(self.planes)+1),dtype=np.longdouble)
        self.count=0;self.maximum_weight=0.

    def add(self, frame):
        a=(frame[['x0_m','y0_m','z0_m']].to_numpy()-self.origin)@self.direction
        b=(frame[['x1_m','y1_m','z1_m']].to_numpy()-self.origin)@self.direction
        w=frame.weight.to_numpy();pid=abs(frame.pdg.to_numpy())
        groups=[pid==11,pid==22,pid==13,np.isin(pid,[211,321,2212,3222,3112,3312,3334])|
                ((pid>=1000000000)&((pid//10000)%1000>0))]
        for k,(lo,hi,moving,side) in enumerate([(a,b,b>a,'right'),(b,a,b<a,'left')]):
            left=np.searchsorted(self.planes,lo,side=side);right=np.searchsorted(self.planes,hi,side=side)
            for j,group in enumerate(groups):
                take=moving&group
                self.delta[k,j]+=np.bincount(left[take],weights=w[take],minlength=len(self.planes)+1)
                self.delta[k,j]-=np.bincount(right[take],weights=w[take],minlength=len(self.planes)+1)
        self.count+=len(frame);self.maximum_weight=max(self.maximum_weight,float(w.max()))

    def save(self, directory):
        values=np.cumsum(self.delta[:,:,:-1],axis=2).astype(float)
        residue=float(max(0.,-values.min()))
        assert residue<1e-7*max(1.,values.max())
        np.savez_compressed(Path(directory)/'particle_crossing_profile.npz',planes_m=self.planes,
            weighted_crossings=values,labels=np.array(['electrons_positrons','photons','muons','charged_hadrons']),
            tracks_scanned=self.count,negative_roundoff=residue,maximum_weight=self.maximum_weight)


def generate(root, tag, output=None):
    assert socket.gethostname()=='psrpku2025','Numerical analysis and plotting run only on PSR'
    import yaml
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    root=Path(root);run=root/'runs'/tag;report=root/'report'/tag
    out=Path(output) if output else report/'geometry_depth';(out/'figures').mkdir(parents=True,exist_ok=True)
    s=yaml.safe_load((run/'output/terrain_run.yaml').read_text())
    assert s['complete'] and not s['diagnostics']['csv_truncated'] and not s['diagnostics']['deposition_csv_truncated']
    origin=np.array(s['position_m']);direction=np.array(s['direction']);direction/=np.linalg.norm(direction)
    first=s['neutrino']['interactions'][0];sv=float((np.array(first['position_enu_m'])-origin)@direction)
    rho_rock=float(s['resolved_material']['transport']['density_kg_m3'])
    material=s['scene']['provenance'].get('material_description','Rock')
    mesh=Path(s['scene']['geometry']['mesh_path']);assert hashlib.sha256(mesh.read_bytes()).hexdigest()==s['mesh_sha256']
    with mesh.open('rb') as stream:
        header=[]
        while True:
            line=stream.readline().decode('ascii').strip();header.append(line)
            if line=='end_header':break
        assert 'format binary_little_endian 1.0' in header
        nv=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        nf=int(next(x.split()[-1] for x in header if x.startswith('element face ')))
        vertices=np.fromfile(stream,dtype='<f8',count=nv*3).reshape(nv,3)
        faces=np.fromfile(stream,dtype=np.dtype([('count','u1'),('indices','<u4',(3,))]),count=nf)
    assert np.all(faces['count']==3);faces=faces['indices'].astype(np.int64)
    p=vertices[faces[:,0]];e1=vertices[faces[:,1]]-p;e2=vertices[faces[:,2]]-p
    normals=np.cross(e1,e2);top=faces[normals[:,2]>1e-10];used=np.unique(top);v=vertices[used]
    tri=mtri.Triangulation(v[:,0],v[:,1],triangles=np.searchsorted(used,top))
    height=mtri.LinearTriInterpolator(tri,v[:,2])
    projected=(v-origin)@direction
    rough=np.linspace(0,float(projected.max()),30000)
    sampled=height(origin[0]+rough*direction[0],origin[1]+rough*direction[1])
    valid=~np.ma.getmaskarray(sampled)
    assert valid[0] and np.all(valid[:np.flatnonzero(valid)[-1]+1]),'Axis must stay inside a connected DEM footprint'
    last=float(np.floor(rough[np.flatnonzero(valid)[-1]]))
    assert float(height(origin[0],origin[1]))<origin[2],'This event starts in air'

    # Intersect the reference line with the complete input triangle mesh.
    h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h)
    take=abs(det)>1e-12;p=p[take];e1=e1[take];e2=e2[take];h=h[take];det=det[take]
    rel=origin-p;u=np.einsum('ij,ij->i',rel,h)/det;q=np.cross(rel,e1)
    b=q@direction/det;dist=np.einsum('ij,ij->i',e2,q)/det
    raw=np.sort(dist[(u>=-1e-10)&(b>=-1e-10)&(u+b<=1+1e-10)&(dist>0)&(dist<last)])
    crossings=raw[np.r_[True,np.diff(raw)>1e-6]] if len(raw) else raw
    assert len(crossings)>=2
    def rock_at(position):return np.searchsorted(crossings,position,side='right')%2==1
    check_s=np.linspace(0,last,20001);xyz=origin+check_s[:,None]*direction
    surface=np.asarray(height(xyz[:,0],xyz[:,1]));away=np.min(abs(check_s[:,None]-crossings),axis=1)>1e-4
    assert np.array_equal(rock_at(check_s[away]),xyz[away,2]<surface[away]),'Ray/triangle and DEM-height medium assignments disagree'

    # Same USStdBK constants and spherical altitude as the transport environment.
    config=json.loads((run/'output/radio/ZHS/config.json').read_text())
    air=config['media'][0];earth=np.array(air['center_m']);radius=air['reference_radius_m']
    boundary=np.array([7000.,11400.,37000.,100000.,112800.])
    B=np.array([1183.6071,1143.0425,1322.9748,655.67307,1.]) # g/cm2
    H=np.array([9542.4834,8000.0534,6295.6893,7375.2177,1e7]) # m
    def air_rho(position):
        points=origin+np.asarray(position)[...,None]*direction
        altitude=np.linalg.norm(points-earth,axis=-1)-radius
        assert np.all((altitude>=0)&(altitude<=112800))
        layer=np.minimum(np.searchsorted(boundary,altitude,side='right'),4)
        return 10*B[layer]/H[layer]*np.where(layer==4,1.,np.exp(-altitude/H[layer]))
    table=np.asarray(air['radial_index']);alt=table[:,0]
    lay=np.minimum(np.searchsorted(boundary,alt,side='right'),4)
    density=10*B[lay]/H[lay]*np.where(lay==4,1.,np.exp(-alt/H[lay]))
    expected=1+(air['index']-1)*density/(10*B[0]/H[0])
    away=np.min(abs(alt[:,None]-boundary),axis=1)>5e-6
    index_error=float(abs(expected[away]-table[away,1]).max());assert index_error<5e-13
    with np.load(report/'profile.npz') as data:edges=data['edges_m'];energy=data['deposited_GeV']
    candidates=[out/'particle_crossing_profile.npz',report/'particle_crossing_profile.npz',report/'figure_review_20260916/particle_crossing_profile.npz']
    particle_file=next(p for p in candidates if p.exists())
    with np.load(particle_file) as data:
        planes=data['planes_m'];counts=data['weighted_crossings'];tracks=int(data['tracks_scanned'])
    assert tracks==s['diagnostics']['steps']
    # Include material interfaces exactly. Add atmosphere-layer crossings as well.
    projection=float((origin-earth)@direction);radial0=float(np.sum((origin-earth)**2))
    breaks=[]
    for altitude in boundary:
        disc=projection**2-radial0+(radius+altitude)**2
        if disc>=0:breaks.extend([-projection-np.sqrt(disc),-projection+np.sqrt(disc)])
    grid=np.unique(np.r_[0.,last,sv,crossings,edges[(edges>0)&(edges<last)],planes[(planes>0)&(planes<last)],np.array(breaks)[(np.array(breaks)>0)&(np.array(breaks)<last)]])
    mid=.5*(grid[:-1]+grid[1:]);width=np.diff(grid);rock=rock_at(mid)
    def integrate_air(order):
        nodes,weights=np.polynomial.legendre.leggauss(order)
        return width*.5*np.sum(air_rho(mid[:,None]+width[:,None]*.5*nodes)*weights,axis=1)
    air8=integrate_air(8);air4=integrate_air(4)
    dX=.1*np.where(rock,rho_rock*width,air8)
    X=np.r_[0.,np.cumsum(dX)];Xv=float(np.interp(sv,grid,X))
    assert np.all(dX>0) and np.all(np.diff(X)>0)
    quadrature_error=float(abs(air8[~rock]-air4[~rock]).sum()*.1);assert quadrature_error<1e-7
    def depth(position):return np.interp(position,grid,X)-Xv
    assert rock_at(sv) and rock_at(sv+1)
    unit_error=abs(float(depth(sv+1)-depth(sv))-.1*rho_rock);assert unit_error<1e-7
    # Reparameterize the existing piecewise-constant 1 m deposition histogram.
    bins=np.searchsorted(edges,mid,side='right')-1
    deposited_density=energy/np.diff(edges)
    mass_energy=deposited_density[bins]*width
    per_depth=mass_energy/dX
    centres=.5*(edges[:-1]+edges[1:]);inside=(centres>=0)&(centres<=last)
    rebin_error=float(mass_energy.sum()-energy[inside].sum())
    assert abs(rebin_error)<1e-8*max(1.,abs(energy).sum())
    excluded=float(energy[~inside].sum())
    significant=energy>1e-8
    assert np.all((centres[significant]>=0)&(centres[significant]<=last))
    have_particles=counts.max(axis=(0,1))>1e-6
    assert np.all((planes[have_particles]>=0)&(planes[have_particles]<=last))
    np.savez_compressed(out/'axis_grammage.npz',distance_m=grid,grammage_g_cm2=X,
        delta_grammage_g_cm2=X-Xv,rock_interval=rock,interval_dX_g_cm2=dX,
        interval_deposited_GeV=mass_energy,dEdX_GeV_per_g_cm2=per_depth,
        interfaces_m=crossings,particle_distance_m=planes[have_particles],
        particle_deltaX_g_cm2=depth(planes[have_particles]),weighted_crossings=counts[:,:,have_particles])
    np.savetxt(out/'axis_grammage.csv',np.c_[grid,X,X-Xv],delimiter=',',header='distance_from_injection_m,X_g_cm2,deltaX_from_first_vertex_g_cm2',comments='')
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'legend.fontsize':8})
    title=f"{s['energy_GeV']/1e6:g} PeV | {material} | seed {s['seed']}"
    names=['Electrons + positrons','Photons','Muons','Charged hadrons / nuclei']
    colors=['tab:blue','tab:orange','tab:green','tab:purple']
    tau_file=run/'tau_tracks.json';tau=json.loads(tau_file.read_text()) if tau_file.exists() else []
    if tau:
        starts=np.array([[t[k] for k in ['x0_m','y0_m','z0_m']] for t in tau])
        ends=np.array([[t[k] for k in ['x1_m','y1_m','z1_m']] for t in tau])
        segments=np.stack([np.c_[(starts-origin)@direction,starts[:,2]],np.c_[(ends-origin)@direction,ends[:,2]]],axis=1)
    def savefig(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/'figures'/(name+'.'+ext),dpi=155)
        plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex='col',constrained_layout=True,gridspec_kw={'height_ratios':[1.15,1,1]})
    for col,(lo,hi) in enumerate([(0,last),(sv-5,sv+40)]):
        distances=np.linspace(lo,hi,2500);points=origin+distances[:,None]*direction
        surf=np.asarray(height(points[:,0],points[:,1]));low=min(float(surf.min()),float(points[:,2].min()))-15;high=max(float(surf.max()),float(points[:,2].max()))+15
        ax=axes[0,col];ax.fill_between(distances,low,surf,color='#cbb38c',label=material)
        ax.fill_between(distances,surf,high,color='#e1f1f9',label='Air');ax.plot(distances,surf,color='#715b3c',lw=1)
        ax.plot(distances,points[:,2],':',color='0.35',label='Reference axis (projection guide)')
        ax.plot([0,sv],[origin[2],np.array(first['position_enu_m'])[2]],color='firebrick',label='Incoming neutrino')
        if tau:ax.add_collection(LineCollection(segments,colors='tab:blue',linewidths=1.5,label='Recorded tau tracks (projection)'))
        for vertex in s['neutrino']['interactions']:
            p=np.array(vertex['position_enu_m']);d=float((p-origin)@direction)
            ax.scatter(d,p[2],marker='*',s=65,color='firebrick',zorder=5)
        for decay in s['tau'].get('decays',[]):
            p=np.array(decay['position_enu_m']);d=float((p-origin)@direction)
            ax.scatter(d,p[2],marker='D',s=35,color='purple',zorder=5)
            for panel in axes[:,col]:panel.axvline(d,color='purple',ls='-.',lw=.8)
        ax.set(ylim=(low,high),ylabel='ENU Up [m]',title='Terrain / profile alignment' if col==0 else 'First shower and nearby rock exit')
        ax.legend(loc='best',fontsize=7,ncol=2)
        selected=(planes>=lo)&(planes<=hi)
        for j,label in enumerate(names):
            y=counts[0,j,selected];axes[1,col].plot(planes[selected],np.where(y>1e-6,y,np.nan),color=colors[j],label=label)
        axes[1,col].set(ylabel='Weighted forward plane crossings',yscale='log',ylim=(.7,max(counts[0].max()*2,2)))
        axes[1,col].legend(fontsize=7,ncol=2)
        positive=np.where(deposited_density>1e-8,deposited_density,np.nan)
        axes[2,col].stairs(positive if col==0 else deposited_density/1000,edges,color='tab:blue')
        axes[2,col].set(ylabel='Deposited energy [GeV/m]' if col==0 else 'Deposited energy [TeV/m]',xlabel='Distance from injection along reference axis [m]')
        if col==0:axes[2,col].set(yscale='log',ylim=(1e-8,deposited_density.max()*3))
        for ax in axes[:,col]:
            ax.set_xlim(lo,hi);ax.axvline(sv,color='firebrick',ls='--',lw=1);ax.grid(alpha=.2)
            for interface in crossings:
                if lo<interface<hi:ax.axvline(interface,color='#68573d',ls=':',lw=.8)
        axes[0,col].annotate(first['current']+' vertex',(sv,np.array(first['position_enu_m'])[2]),xytext=(6,12),textcoords='offset points',fontsize=9)
    fig.suptitle(title+' | Red dashed: first vertex; brown dotted: axis / rock interfaces')
    savefig(fig,'09_terrain_aligned_shower_profile')

    physical_end=max(float(edges[np.flatnonzero(significant)[-1]+1]),float(planes[have_particles].max()))
    ranges=[(float(depth(sv-2)),float(depth(sv+14))),(float(depth(sv-2)),float(depth(physical_end))+1)]
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',constrained_layout=True)
    for col,(lo,hi) in enumerate(ranges):
        for j,label in enumerate(names):
            y=counts[0,j,have_particles];axes[0,col].plot(depth(planes[have_particles]),np.where(y>1e-6,y,np.nan),label=label,color=colors[j])
        axes[0,col].set(ylabel='Weighted forward plane crossings',title='Main shower' if col==0 else 'Whole physical profile, including tails')
        axes[0,col].legend(ncol=2,fontsize=8)
        axes[1,col].stairs(np.where(per_depth>1e-12,per_depth/1000,np.nan),X-Xv,color='tab:blue')
        axes[1,col].set(xlabel=r'Axis column depth from first vertex $\Delta X$ [g/cm$^2$]',ylabel=r'$dE_{dep}/dX$ [TeV / (g/cm$^2$)]')
        if col:axes[0,col].set_yscale('log');axes[1,col].set_yscale('log')
        limits=np.r_[0.,crossings,last]
        for k in range(len(limits)-1):
            a0,b0=depth(limits[k:k+2]);a0=max(lo,float(a0));b0=min(hi,float(b0))
            if b0>a0:
                for ax in axes[:,col]:ax.axvspan(a0,b0,color='#cbb38c' if rock_at(.5*(limits[k]+limits[k+1])) else '#dceef8',alpha=.2,zorder=0)
        for ax in axes[:,col]:ax.set_xlim(lo,hi);ax.axvline(0,color='firebrick',ls='--',lw=.8);ax.grid(alpha=.2)
    fig.suptitle(title+' | Axis grammage: rock + native atmosphere | Brown: rock; blue: air along axis')
    savefig(fig,'10_shower_profile_grammage')
    imax=int(np.argmax(counts[0,0]));peak_s=float(planes[imax])
    proof=dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        density_kg_m3=rho_rock,g_cm2_per_m_in_rock=.1*rho_rock,vertex_distance_m=sv,
        vertex_absolute_X_g_cm2=Xv,axis_interfaces_m=crossings.tolist(),
        first_rock_exit_after_vertex_m=float(crossings[crossings>sv][0]),
        first_rock_exit_deltaX_g_cm2=float(depth(crossings[crossings>sv][0])),
        electron_forward_profile_peak=float(counts[0,0,imax]),electron_peak_distance_from_vertex_m=peak_s-sv,
        electron_peak_deltaX_g_cm2=float(depth(peak_s)),native_air_table_maximum_index_error=index_error,
        air_quadrature_order4_vs8_g_cm2=quadrature_error,one_metre_rock_conversion_error_g_cm2=unit_error,
        transformed_deposition_integral_GeV=float(mass_energy.sum()),reparameterization_error_GeV=rebin_error,
        excluded_outside_axis_coverage_GeV=excluded,all_significant_bins_and_crossings_inside_axis_coverage=True,
        profile_tracks_scanned=tracks,mesh_sha256=s['mesh_sha256'],axis_domain_m=[0,last],
        definition='X(s)=0.1 integral rho_kg_m3(s) ds_m from injection; Delta X subtracts the first actual neutrino vertex. Reference-axis grammage, not individual curved-track accumulated grammage. N is weighted forward plane crossings. dE/dX uses dE divided by the transformed bin width.')
    (out/'grammage_checks.json').write_text(json.dumps(proof,indent=2)+'\n')
    text=rf'''

---

距离坐标：与山体剖面上下对齐

![w:1020](figures/09_terrain_aligned_shower_profile.png)

每列三幅图共用距离轴：上面是实际 DEM 剖面，中间是带 thinning 权重的粒子向前穿面数，下面是沉积能量。左列看全景与长尾，右列放大第一次 shower。红虚线贯穿实际 {first['current']} 顶点，棕色点线是参考轴穿过岩石表面的位置；灰虚线只是投影参考轴。

本例顶点位于注入后 {sv:.3f} m；其后参考轴再前进 {crossings[crossings>sv][0]-sv:.3f} m 就离开这一段山体。粒子数和沉积使用完整记录；剖面中的介质颜色说明参考轴所在介质，不代表每个离轴粒子的实际介质。

---

柱深坐标：山体同样可以使用 g/cm²

![w:1020](figures/10_shower_profile_grammage.png)

$X(s)=\int_0^s\rho(s')\,ds'$，图中使用 $\Delta X=X(s)-X(s_{{vertex}})$，把实际首个顶点设为零。当前材料每米对应 {.1*rho_rock:g} g/cm²；跨入空气后用同一 USStdBK 大气密度积分，不能继续乘岩石系数。

电子/正电子向前穿面数峰在顶点后 {peak_s-sv:.2f} m，即 $\Delta X\approx {depth(peak_s):.1f}$ g/cm²。沉积图同时把纵轴改为 $dE_{{dep}}/dX$，转换前后积分保持约 {mass_energy.sum()/1000:.3f} TeV。这是参考轴柱深；不同组分即使柱深相同，辐射长度等物性也可能不同。

棕底表示参考轴位于岩石内，蓝底为空气段。左列看主峰，右列用对数轴保留长尾。
'''
    deck=out/'FIGURES_CN.md';previous=deck.read_text() if deck.exists() else '---\nmarp: true\npaginate: true\n---\n'
    marker='\n\n---\n\n距离坐标：与山体剖面上下对齐'
    if marker in previous:previous=previous.split(marker)[0]
    previous=previous.replace('不是粒子数，也不能直接称为以 g/cm² 表示的 $X_{max}$。',
        '不是粒子数，其峰位不能直接称为粒子数最大位置 $X_{max}$；后两页补上剖面对齐和 g/cm² 柱深坐标。')
    deck.write_text(previous.rstrip()+text+'\n')
    (out/'GRAMMAGE_README_CN.md').write_text(r'''山体可以使用柱深 g/cm²。先看 [剖面对齐与柱深图](FIGURES_CN.md) 的最后两页。

此前“不能直接当作 Xmax”指的是不能把沉积峰直接当成粒子数峰，并非山体不能定义柱深。两者是不同的 profile；本次均给出柱深坐标。

定义 $X(s)=\int_0^s\rho(s')\,ds'$，图中取 $\Delta X=X(s)-X(s_{vertex})$。从注入点累计的 X 包含中微子相互作用前穿过的物质；研究 shower 发展时，用实际产生 shower 的顶点作为零点更直观。

本例 SiO₂ 密度 2.65 g/cm³，因此 1 m 岩石对应 265 g/cm²。穿越山体表面后使用输运程序相同的 USStdBK 大气和球面高度积分。粒子数 N 不因横坐标转换改变；沉积纵轴按每个转换后箱宽计算 $\Delta E/\Delta X$，不能只改横轴标签。

当前图重用完整轨迹的穿面统计和既有沉积箱：主峰附近粒子面间距 0.1 m，在岩石中对应 26.5 g/cm²；原沉积箱宽 1 m，对应 265 g/cm²。没有把旧数据插值当作空气程序默认 10 g/cm² 分箱。界面处按既有箱内常数沉积密度分段转换，转换不增加原始空间分辨率。

这是入射参考轴的柱深，不是每条散射轨迹自己走过的累计柱深。图中棕底和蓝底分别表示参考轴在岩石内和空气内，不能据此给全部离轴粒子判定介质。不同材料在相同柱深下仍可能有不同辐射长度、相互作用长度等。

数值分析和绘图均在 PSR 完成。网格交点与 DEM 高度判定一致，空气密度重建与原生折射率表一致，积分阶数对照和沉积积分守恒均通过。坐标范围外仅有约 8.34e-8 GeV 的原直方图浮点残差，已记录；原始 profile 文件未修改。本例是完整的 NC 单次 shower，没有 τ 衰变，不能称作 double bang。

[转换检查](grammage_checks.json) · [距离—柱深数值表](axis_grammage.csv) · [完整转换数组](axis_grammage.npz) · [绘图源码](geometry_depth_psr.py)
''')
    print(json.dumps(proof),flush=True)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tag',required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();generate(args.root,args.tag,args.output)
