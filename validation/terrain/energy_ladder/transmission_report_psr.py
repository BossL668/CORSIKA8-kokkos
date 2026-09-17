#!/usr/bin/env python3
"""Check independent transmission diagnostics and plot actual obstructed rays."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

assert socket.gethostname()=='psrpku2025'
base=Path('/data/yhlu/CorsikaData/corsika_validation_results')
r=base/'beta5_1PeV_transmission_diagnostic_20260916'
report=r/'report';(report/'figures').mkdir(parents=True,exist_ok=True)
read=lambda name:json.loads((r/name).read_text())
native=read('native.json');controls=read('native_controls.json')
assert native['passed'] and controls['passed'] and len(native['rows'])==147 and len(controls['rows'])==2
native_rows=native['rows']+controls['rows']
oracle=read('independent.json')['rows']+read('independent_controls.json')['rows']
point_error=0.;hit_error=0.;residual=0.
for row in oracle:
    ref=next(x for x in native_rows if x['label']==row['label'])
    assert row['index_iteration_converged']
    assert row['accepted']==ref['unfiltered_valid'] and row['blocked']==ref['unfiltered_blocked']
    assert {p['face'] for p in row['stationary_rays']}=={p['face'] for p in ref['stationary_rays']}
    for path in row['stationary_rays']:
        target=next(x for x in ref['stationary_rays'] if x['face']==path['face'])
        point_error=max(point_error,float(np.linalg.norm(np.array(path['interface_m'])-target['interface_m'])))
        residual=max(residual,abs(path['snell_residual']))
        for leg in ['source','destination']:
            a=path[leg+'_hit'];b=target[leg+'_first_hit_independent']
            assert (a is None)==(b is None)
            if a is not None:
                assert a['face']==b['face']
                hit_error=max(hit_error,float(np.linalg.norm(np.array(a['point_m'])-b['point_m'])))
assert point_error<1e-6 and hit_error<1e-6 and residual<1e-10
main=[x for x in native['rows'] if x['source_id']=='axis_1.7m' and x['observer']!='overhead_control']
assert len(main)==80
real=[x for x in native['rows'] if not x['source_id'].startswith('old_') and x['observer']!='overhead_control']
historical=read('historical_transmission.json')
assert all(x['transmitted_paths']>0 and x['complete'] and x['errors']==0 for x in historical)
checks=dict(passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    historical_cases_with_transmission=len(historical),native_queries=len(native_rows),
    current_real_source_observer_pairs=len(real),current_real_valid_paths=sum(x['unfiltered_valid'] for x in real),
    current_real_blocked_paths=sum(x['unfiltered_blocked'] for x in real),
    current_real_pairs_without_finite_face_root=sum(x['unfiltered_valid']+x['unfiltered_blocked']==0 for x in real),
    main_source_stations=80,main_source_accepted=0,
    main_source_blocked=sum(x['unfiltered_blocked']>0 for x in main),
    main_source_without_finite_face_root=sum(x['unfiltered_valid']+x['unfiltered_blocked']==0 for x in main),
    candidate_filter_disagreements=sum(not x['candidate_filters_agree'] for x in native_rows),
    independent_visibility_disagreements=sum(x['independent_visibility_mismatches'] for x in native_rows),
    invalid_or_unconverged=sum(sum(x['invalid_or_unconverged']) for x in native_rows),
    independent_snell_queries=len(oracle),maximum_interface_point_difference_m=point_error,
    maximum_first_blocker_difference_m=hit_error,maximum_independent_snell_residual=residual,
    positive_control_valid=[x['unfiltered_valid'] for x in controls['rows']],
    scope='10 representative actual current rock electron/positron midpoints, including main shower and extrema, plus 2 historical midpoints; 125 current source/station pairs. One source tested against all 80 stations. Not all 14.9 million charged rock segments. Same single-interface, straight-leg geometric-optics model; does not validate diffraction, repeated interfaces or reflection.')
row=next(x for x in oracle if x['label']=='axis_1.7m_E18');ray=row['stationary_rays'][0]
S=np.array(row['source_m']);P=np.array(ray['interface_m']);O=np.array(row['observer_m']);B=np.array(ray['destination_hit']['point_m'])
V=P+100*(O-P)/np.linalg.norm(O-P)
checks['E18_example']=dict(source_m=S.tolist(),interface_m=P.tolist(),first_blocker_m=B.tolist(),observer_m=O.tolist(),
    blocker_distance_after_exit_m=ray['destination_hit']['distance_m'],incidence_angle_deg=ray['incidence_angle_deg'],
    critical_angle_deg=ray['critical_angle_deg'],rock_path_length_m=float(np.linalg.norm(P-S)),
    rock_absorption_factor=float(np.exp(-np.linalg.norm(P-S)/100)),snell_residual=ray['snell_residual'])
(report/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
mesh=Path(read('provenance.json')['mesh'])
assert hashlib.sha256(mesh.read_bytes()).hexdigest()==read('provenance.json')['mesh_sha256']
with mesh.open('rb') as f:
    header=[]
    while True:
        line=f.readline().decode().strip();header.append(line)
        if line=='end_header':break
    nv=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
    nf=int(next(x.split()[-1] for x in header if x.startswith('element face ')))
    vertices=np.fromfile(f,dtype='<f8',count=nv*3).reshape(nv,3)
    faces=np.fromfile(f,dtype=np.dtype([('count','u1'),('index','<u4',(3,))]),count=nf)['index']
tri=vertices[faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);top=faces[n[:,2]>0];used=np.unique(top)
xyz=vertices[used];terrain=mtri.LinearTriInterpolator(mtri.Triangulation(xyz[:,0],xyz[:,1],np.searchsorted(used,top)),xyz[:,2])
L0=np.linalg.norm(P[:2]-S[:2]);L1=np.linalg.norm(O[:2]-P[:2]);total=L0+L1
xb=L0+np.linalg.norm(B[:2]-P[:2]);xv=L0+np.linalg.norm(V[:2]-P[:2])
def path_xy(x):
    x=np.asarray(x);first=x<=L0
    a=S[:2]+x[:,None]/L0*(P[:2]-S[:2])
    b=P[:2]+(x-L0)[:,None]/L1*(O[:2]-P[:2])
    return np.where(first[:,None],a,b)
def height(x):
    xy=path_xy(x);z=terrain(xy[:,0],xy[:,1]);assert not np.ma.getmaskarray(z).any();return np.asarray(z)
assert abs(float(height(np.array([L0]))[0])-P[2])<1e-6
assert abs(float(height(np.array([xb]))[0])-B[2])<1e-6
plt.rcParams.update({'font.size':11,'legend.fontsize':8})
fig,axes=plt.subplot_mosaic([['full','full'],['exit','ridge']],figsize=(14,9),constrained_layout=True)
for key,(lo,hi,low,high) in {'full':(0,total,-200,600),'exit':(0,25,495,512),'ridge':(xb-100,xb+100,B[2]-30,B[2]+30)}.items():
    ax=axes[key];x=np.linspace(lo,hi,3500);ground=height(x)
    ax.set_facecolor('#edf7fc');ax.fill_between(x,low,ground,color='#cbb38c',label='Rock / DEM');ax.plot(x,ground,color='#685139',lw=1)
    ax.plot([0,L0],[S[2],P[2]],color='tab:blue',lw=2,label='Solved rock leg')
    ax.plot([L0,xb],[P[2],B[2]],color='tab:orange',lw=2,label='Air leg before first obstruction')
    ax.plot([xb,total],[B[2],O[2]],'--',color='crimson',lw=1.5,label='Rejected continuation')
    for xx,p,marker,color,label in [(0,S,'*','blue','Recorded rock source'),(L0,P,'o','black','Solved refraction point'),
        (xb,B,'X','crimson','First terrain obstruction'),(total,O,'^','purple','Real station E18'),(xv,V,'v','teal','Virtual receiver: accepted')]:
        if lo<=xx<=hi:ax.scatter(xx,p[2],marker=marker,c=color,s=55,zorder=7,label=label)
    ax.set(xlim=(lo,hi),ylim=(low,high),xlabel='Horizontal distance along projected ray path [m]',ylabel='ENU Up [m]');ax.grid(alpha=.2)
axes['full'].set_title('Real source to E18: a Snell solution exists, but its air leg intersects another ridge')
axes['full'].legend(ncol=3,loc='upper right')
axes['full'].annotate('First blocker: 797.4 m after exit',(xb,B[2]),xytext=(xb+500,B[2]+70),arrowprops={'arrowstyle':'->'},fontsize=9)
axes['exit'].set_title(r'Exit detail: $\theta_i=26.09^\circ < \theta_c=26.57^\circ$')
axes['exit'].text(.03,.94,f'Rock length = {np.linalg.norm(P-S):.3f} m\nAbsorption factor = {np.exp(-np.linalg.norm(P-S)/100):.3f}',transform=axes['exit'].transAxes,va='top')
axes['ridge'].set_title('First blocking ridge: independent triangle intersection')
fig.suptitle('1 PeV SiO2, seed 22309 | Actual DEM sampled underneath both ray legs')
for ext in ['png','pdf']:fig.savefig(report/'figures'/('12_actual_transmission_shadow.'+ext),dpi=155)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
old=[next(x for x in historical if x['tag']==f'openmp_silica_SiO2_seed{seed}') for seed in [158,946,3605]]
values=[x['transmitted_paths'] for x in old];axes[0].bar(['seed 158','seed 946','seed 3605'],values,color='steelblue')
axes[0].set(yscale='log',ylim=(1e3,3e7),ylabel='Accepted transmitted subsegment-observer contributions',title='Historical 100 PeV SiO2 events: transmission was nonzero')
for i,x in enumerate(values):axes[0].text(i,x*1.1,f'{x:,}',ha='center')
labels=['Normal BVH\nand face filters','All faces\n(no BVH selection)','All faces\n(no projected-face filter)']
axes[1].bar(labels,[61]*3,color='firebrick',label='Root in a face, then terrain obstruction')
axes[1].bar(labels,[19]*3,bottom=61,color='#d8a735',label='No stationary root inside a finite face')
axes[1].set(ylim=(0,92),ylabel='Number of real stations (one actual source)',title='Current main-shower source: same result for all 80 stations')
for i in range(3):axes[1].text(i,30.5,'61',ha='center',color='white');axes[1].text(i,70.5,'19',ha='center')
axes[1].legend(loc='upper center',fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('149 native source-observer checks | 0 filter disagreements | 0 solver failures\n10 independent Snell checks agree; both near-side receiver controls transmit')
for ext in ['png','pdf']:fig.savefig(report/'figures'/('13_transmission_filter_comparison.'+ext),dpi=155)
plt.close(fig)
note='''之前的九组介质测试都有非零岩石→空气透射。下面先看本次真实光路为什么被拒绝，再看筛选对照。

![](figures/12_actual_transmission_shadow.png)

上排沿两段光路的水平投影取真实 DEM 剖面。蓝线是源点到折射点的岩石段，橙线是出山后的空气段；红色叉号是第一次撞上后续地形的位置。红虚线是被拒绝的延长线，并非已经模拟了穿过第二座山的传播。左下放大出射位置，右下放大挡光山脊。

这是 1 PeV / SiO₂ / seed 22309 中 step 9194659 的真实电子段中点，距 NC 顶点轴向约 1.7 m。到 E18 的折射根位于实际面内，入射角 26.090°，小于临界角 26.572°，Snell 残差约 2.22e-16。岩石内长度 6.205 m，电场吸收因子约 0.940。源点到出射点无遮挡，但之后 797.408 m 处与另一段地形相交；E20 对应路径的首次遮挡距出射点 627.098 m。因此这些具体光路不是因全反射、吸收或不收敛而被拒绝。

绿色虚拟接收点沿同一出射方向放在出射后 100 m、第一座挡光山脊之前。原生程序和独立程序都找到有效透射；它只是检验求解器的对照站，不属于真实 80 站结果。

![](figures/13_transmission_filter_comparison.png)

左图为三例历史 SiO₂ 事件的有效透射计数，分别为 184990、28190、5903597；九组介质均非零。这些计数是射电子段—站点贡献，不能当作独立物理光线条数。右图对同一个当前真实源点检查全部 80 站：61 站有面内折射根但之后被山体挡住，19 站没有落在实际三角面内的单界面驻相根；三个计算方式给出相同分类。

正常候选面 BVH、遍历全部 511488 个三角面，以及关闭投影预筛并穷举共面归属的计算，共完成 149 组查询，未出现筛选结果差异或 Invalid/Unconverged。所有找到的面内光路再以 long double 逐三角形相交验证，遮挡判定无差异。另用独立 NumPy 二分法在全部真实地表面上求 Snell 根并做相交判断，10 组查询的根、面编号及分类均一致，其中包含两个成功透射的正对照。

诊断范围包括本事件 10 个真实岩石电子/正电子源点（主 shower、沿轴位置和空间极值）、2 个历史源点、上方控制站与两个山脊前控制站。本事件共有 125 个真实源点—真实站点组合被抽查，均无可见透射；其中 96 个有面内根但出射后的路径被挡，29 个无面内根。没有独立重算全部约 1490 万条岩石带电步段，因此这是有明确覆盖范围的诊断，不能声称对所有源逐条证明。

结论：本次抽查未发现候选面筛选漏解或 Snell 数值求解失败，零接收透射与地形可见性及有限面内解条件一致。结论限于当前单界面、直线分段几何光学模型；绕射、再次入山再出山的多界面路径、反射并未包含，不能据此断言真实电磁场严格为零。

[机器可读检查](checks.json) · [原生逐面结果](native.json) · [独立求解](independent.json) · [正对照](native_controls.json) · [历史九例计数](historical_transmission.json) · [实际源点与轨迹编号](sources.json)
'''
(report/'README_CN.md').write_text(note)
for name in ['native.json','independent.json','native_controls.json','independent_controls.json','historical_transmission.json','sources.json','sampling.json','queries.json','control_queries.json','provenance.json','build_commands.json','TransmissionAudit.cpp','transmission_audit_psr.py','transmission_independent_psr.py']:
    shutil.copy2(r/name,report/name)
shutil.copy2(__file__,report/'transmission_report_psr.py')
original=base/'beta5_doublebang_1PeV_thin1e4_queue4M_20260915/report/openmp_SiO2_21CMA80_1PeV_seed22309'
out=original/'figure_review_20260916'
for p in (report/'figures').iterdir():shutil.copy2(p,out/'figures'/p.name)
shutil.copy2(report/'checks.json',out/'transmission_checks.json')
deck=out/'FIGURES_CN.md';previous=deck.read_text();marker='\n\n---\n\n真实光路：折射能解出，后面的山脊挡住了'
if marker in previous:previous=previous.split(marker)[0]
previous=previous.replace('零透射不能单凭此图归因于吸收、全反射或地形遮挡；具体拒绝原因尚未独立分解。本例不能用来证明山体透射物理正确。',
    '后续逐面和独立求解诊断见第 12–13 页：抽查的面内折射解被后续地形遮挡；未发现候选筛选漏解。结论限于当前单界面几何光学模型。')
slides='''

---

真实光路：折射能解出，后面的山脊挡住了

![w:1020](figures/12_actual_transmission_shadow.png)

蓝线：真实岩石源点到出射点。橙线：出山后走向 E18；797.4 m 后撞上另一段地形。红虚线只是被拒绝的延长线。左下看出射点，右下看首次遮挡位置。

入射角 26.09° 小于临界角 26.57°，岩石吸收因子约 0.94。这条路径因后段遮挡被拒绝。把虚拟站放到出山后 100 m，原生与独立求解都得到有效透射。

---

有没有被筛选器误删？

![w:1020](figures/13_transmission_filter_comparison.png)

左图：历史事例确实有非零透射。右图：同一真实源点检查 80 站，61 站有折射解但被后续山脊挡住，19 站无实际面内的单界面解。关闭候选筛选后分类不变。

共 149 组原生查询无筛选差异、无求解失败；独立求根的 10 组查询也一致。抽查覆盖当前 10 个真实源点，未重算所有源；结论限于单界面直线分段模型，不包含绕射、多次穿山或反射。[数值检查](transmission_checks.json)
'''
deck.write_text(previous.rstrip()+slides+'\n')
p=out/'README_CN.md';p.write_text(p.read_text().replace('十一页','十三页')+'\n第 12–13 页补充实际折射光路被地形遮挡的位置，以及逐面/关闭筛选/独立求解的对照。\n')
p=out/'SOURCE_MEDIUM_README_CN.md';p.write_text(p.read_text()+'\n后续补充：透射独立诊断已完成，见 [第 12–13 页](FIGURES_CN.md) 和 [检查结果](transmission_checks.json)。抽查未发现筛选漏解，当前样本有面内折射根但被后续地形阻挡，或无实际面内单界面根。此结论没有覆盖所有源，也不覆盖绕射、多次穿界面和反射。\n')
for directory in [report,out,original]:
    rows=[]
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS':rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(directory).as_posix())
    (directory/'SHA256SUMS').write_text('\n'.join(rows)+'\n')
print(json.dumps(checks),flush=True)
