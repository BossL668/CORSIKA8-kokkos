#!/usr/bin/env python3
"""English s/p figures and independent TMM checks. Run on PSR only."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Arc
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';F=ROOT/'figures'
OLD=ROOT.parent/'interface-radio-report-20260912'
sys.path.insert(0,str(OLD/'external/python'))
import tmm
from importlib.metadata import version
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
 'savefig.dpi':190,'pdf.fonttype':42})
BLUE='#1666ab';ORANGE='#de7825';S_COLOR='#168579';P_COLOR='#a23b91';ROCK='#cfb493'
r=np.genfromtxt(D/'sp_selected.csv',delimiter=',',names=True)
a=np.genfromtxt(D/'sp_fresnel_scan.csv',delimiter=',',names=True);a=np.sort(a,order='angle_deg')
n1,n2=float(r['n1']),float(r['n2']);inc=float(r['incidence_deg']);trans=float(r['transmission_deg'])
brew=float(r['brewster_deg']);crit=float(r['critical_deg']);valid=a['transmitted']>0
refs={}
for pol in ['s','p']:
 vals=[];Ts=[];Rs=[]
 for angle in a['angle_deg']:
  ti=np.deg2rad(angle);tt=tmm.snell(n1,n2,ti)
  t=tmm.interface_t(pol,n1,n2,ti,tt);rr=tmm.interface_r(pol,n1,n2,ti,tt)
  vals.append(t);Ts.append(tmm.T_from_t(pol,t,n1,n2,ti,tt));Rs.append(tmm.R_from_r(rr))
 refs['t_'+pol]=np.array(vals);refs['T_'+pol]=np.array(Ts);refs['R_'+pol]=np.array(Rs)
error_t=max(float(np.max(abs(a['t_'+p][valid]/refs['t_'+p][valid]-1))) for p in ['s','p'])
error_T=max(float(np.max(abs(a['T_'+p]-refs['T_'+p]))) for p in ['s','p'])
energy=max(float(np.max(abs(refs['T_'+p]+refs['R_'+p]-1))) for p in ['s','p'])
common=float(r['common_per_m'])
matrix=np.array([[r['m_ss'],r['m_sp']],[r['m_ps'],r['m_pp']]])/common
expected=np.diag([r['t_s'],r['t_p']])
matrix_error=float(np.max(abs(matrix-expected)))
input_components=np.array([r['in_s'],r['in_p']])
output=np.array([r['out_s'],r['out_p']])/common
linear_error=float(np.max(abs(output-expected@input_components)))
assert error_t<1.e-9 and error_T<1.e-9 and energy<1.e-12
assert matrix_error<1.e-12 and linear_error<1.e-12
assert abs(float(r['longitudinal_per_m']))<1.e-15
def save(fig,name):
 fig.savefig(F/(name+'.png'),bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def arrow(ax,start,end,color,label=None,offset=(0,0)):
 ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',color=color,lw=2.2,mutation_scale=15))
 if label:ax.text(end[0]+offset[0],end[1]+offset[1],label,color=color,fontsize=13)
# 6. Plane-of-incidence definition, using the actual replay angles.
fig,axes=plt.subplots(1,2,figsize=(12.6,5.2),constrained_layout=True)
ax=axes[0];ax.set_facecolor('#eff7fc');ax.axhspan(-6,0,color=ROCK,alpha=.75);ax.axhline(0,color='#866347',lw=2)
ax.axvline(0,color='#777777',ls=':',lw=1.3)
ki=np.array([np.sin(np.deg2rad(inc)),np.cos(np.deg2rad(inc))])
kt=np.array([np.sin(np.deg2rad(trans)),np.cos(np.deg2rad(trans))])
arrow(ax,-5.2*ki,np.zeros(2),BLUE)
arrow(ax,np.zeros(2),5.2*kt,ORANGE)
ax.text(-1.6,-4.6,r'$k_i$',color=BLUE,fontsize=16);ax.text(1.6,4.6,r'$k_t$',color=ORANGE,fontsize=16)
for k,position,label in [(ki,-3.5*ki,'i'),(kt,3.5*kt,'t')]:
 p=np.array([-k[1],k[0]])
 arrow(ax,position,position+1.9*p,P_COLOR,r'$p_'+label+'$',offset=(-.35,.12))
 ax.text(position[0],position[1],r'$\odot$',color=S_COLOR,ha='center',va='center',fontsize=23)
 ax.text(position[0]+.38,position[1]-.4,r'$s_'+label+'$',color=S_COLOR,fontsize=14)
ax.text(.15,5.45,r'$N$ (outward normal)',fontsize=11,color='#555555')
ax.text(-4.0,4.7,'Air: n = %.6f'%n2,fontsize=11);ax.text(-4.0,-5.5,'Rock: n = 2',fontsize=11)
ax.text(.35,-.6,'Interface',fontsize=11)
ax.text(1.7,-3.4,'Incidence: %.2f°\nRefraction: %.2f°'%(inc,trans),fontsize=11)
ax.set(title='Side view: the incidence plane is this page',xlim=(-4.4,4.4),ylim=(-6,6),xticks=[],yticks=[])
ax.set_aspect('equal',adjustable='box')
ax.text(.5,-.04,r'$\odot$ = out of page; arrow lengths are schematic',transform=ax.transAxes,ha='center',fontsize=10)
ax=axes[1];ax.set_aspect('equal',adjustable='box')
arrow(ax,(0,0),(1.25,0),S_COLOR,r'$\hat{s}$',offset=(.02,-.02))
arrow(ax,(0,0),(0,1.25),P_COLOR,r'$\hat{p}$',offset=(-.06,.03))
e=np.array([.85,.65]);arrow(ax,(0,0),e,'#222222',r'$\mathbf{E}$',offset=(.04,.02))
ax.plot([e[0],e[0]],[0,e[1]],ls=':',color=P_COLOR);ax.plot([0,e[0]],[e[1],e[1]],ls=':',color=S_COLOR)
ax.text(.38,-.11,r'$E_s$',color=S_COLOR,fontsize=15);ax.text(-.15,.32,r'$E_p$',color=P_COLOR,fontsize=15)
ax.text(-.01,-.025,r'$\otimes$',ha='center',va='center',color='#555555',fontsize=22)
ax.text(.05,-.26,r'$k$ points into the page',fontsize=11)
ax.text(.12,1.10,r'$\mathbf{E}=E_s\hat{s}+E_p\hat{p}$',fontsize=17)
ax.set(title='View along the ray: illustrative E decomposition',xlim=(-.28,1.6),ylim=(-.35,1.46),xticks=[],yticks=[])
ax.text(.5,-.04,'Both s and p are perpendicular to k',transform=ax.transAxes,ha='center',fontsize=11)
save(fig,'06_sp_geometry')

# 7. Actual local indices; angle scan is a controlled diagnostic, not more showers.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.9),constrained_layout=True)
for ax in axes:
 ax.axvline(inc,color='#777777',ls='--',lw=1, label='Selected ray: %.2f°'%inc)
 ax.axvline(brew,color='#999999',ls=':',lw=1)
 ax.axvline(crit,color='#555555',ls='-.',lw=1)
 ax.axvspan(crit,34,color='#dce1e5',alpha=.8)
 ax.set(xlabel='Incidence angle from normal (deg)',xlim=(0,34))
for pol,color in [('s',S_COLOR),('p',P_COLOR)]:
 axes[0].plot(a['angle_deg'][valid],a['t_'+pol][valid],color=color,lw=2,label=r'$t_'+pol+'$ (beta5)')
 indices=np.flatnonzero(valid)[::35]
 axes[0].plot(a['angle_deg'][indices],refs['t_'+pol][indices].real,'o',ms=4,mfc='none',color=color,label='TMM '+pol)
 axes[0].plot(inc,float(r['t_'+pol]),'o',color=color,ms=7)
 axes[1].plot(a['angle_deg'],a['T_'+pol],color=color,lw=2,label=r'$T_'+pol+'$ (beta5)')
 axes[1].plot(a['angle_deg'][indices],refs['T_'+pol][indices],'o',ms=4,mfc='none',color=color,label='TMM '+pol)
axes[0].set(title='Field-amplitude transmission',ylabel=r'$t=E_{\rm transmitted}/E_{\rm incident}$',ylim=(0,4.25))
axes[0].text(.035,.60,'At the selected ray:\n'+r'$t_s=%.5f,\quad t_p=%.5f$'%(r['t_s'],r['t_p']),transform=axes[0].transAxes,fontsize=11)
axes[0].text(32,2.5,'TIR',ha='center',rotation=90,color='#555555')
axes[0].legend(fontsize=9,loc='upper left',ncol=2)
axes[1].plot(a['angle_deg'],refs['R_p'],color='#555555',ls=':',lw=1.6,label=r'$R_p$ (TMM reference)')
axes[1].set(title='Power transmission across the interface',ylabel='Normal power-flux fraction',ylim=(-.04,1.22))
axes[1].annotate('Brewster: %.2f°\n'%brew+r'$R_p=0,\ T_p=1$',xy=(brew,1),xytext=(13.3,1.06),fontsize=10,arrowprops={'arrowstyle':'-','color':'#555555'})
axes[1].text(31.95,.43,'TIR',ha='center',rotation=90,color='#555555')
axes[1].legend(fontsize=8.5,loc='lower left',ncol=2)
fig.suptitle('Real local interface: n1 = 2, n2 = %.6f; critical angle = %.3f°'%(n2,crit),fontsize=13)
save(fig,'07_sp_fresnel')

# 8. The actual selected track's normalized transverse-current source factor.
fig,axes=plt.subplots(1,2,figsize=(12.6,4.9),constrained_layout=True)
ax=axes[0];x=np.arange(2);w=.30
ax.bar(x-w/2,input_components,w,color=BLUE,label='Unit input: transverse-current direction')
ax.bar(x+w/2,output,w,color=ORANGE,label='beta5 output / common factor')
ti=np.deg2rad(inc);tt=tmm.snell(n1,n2,ti)
tmm_output=np.array([tmm.interface_t(pol,n1,n2,ti,tt) for pol in ['s','p']])*input_components
ax.plot(x+w/2,tmm_output.real,'o',mfc='none',color='black',ms=8,label='TMM prediction')
for xx,y in zip(x-w/2,input_components):ax.text(xx,y+.025,'%.4f'%y,ha='center',fontsize=11)
for xx,y in zip(x+w/2,output):ax.text(xx,y+.025,'%.4f'%y,ha='center',fontsize=11)
ax.set(xticks=x,xticklabels=['s component','p component'],ylim=(0,max(output)*1.53),
 ylabel='Component / unit input norm',title='Real track 1: polarization factors')
ax.legend(loc='upper left',fontsize=9)
ax=axes[1];im=ax.imshow(abs(matrix),cmap='Blues',vmin=0,vmax=1.5)
ax.set(xticks=[0,1],xticklabels=['Input s','Input p'],yticks=[0,1],yticklabels=['Output s','Output p'],
 title='beta5 transfer in the local s/p bases')
for i in range(2):
 for j in range(2):
  value=matrix[i,j];txt='%.5f'%value if i==j else '< 1e-15'
  ax.text(j,i,txt,ha='center',va='center',color='white' if i==j else '#333333',fontsize=16)
ax.grid(False)
ax.text(.5,-.13,'Common spreading / flux factor removed',transform=ax.transAxes,ha='center',fontsize=11)
fig.suptitle('Single-track factors only: not the full shower waveform or absolute E',fontsize=13)
save(fig,'08_sp_actual_track')
summary=dict(passed=True,tmm_version=version('tmm'),angle_scan_rows=len(a),
 max_fresnel_amplitude_relative_error=error_t,max_power_absolute_error=error_T,
 max_R_plus_T_error=energy,local_matrix_max_error=matrix_error,source_linear_map_max_error=linear_error,
 selected={name:float(r[name]) for name in r.dtype.names},
 selected_output_after_common_factor=output.tolist(),local_matrix_after_common_factor=matrix.tolist(),
 scope='Real track and face; Fresnel angle scan with fixed real local indices; isotropic lossless nonmagnetic media.',
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT.parent/'source/corsika/modules/radio/interface/Propagation.hpp',D/'selected_ray.json',D/'sp_input.txt']})
(D/'sp_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(D/'sp_tmm_reference.npz',angle_deg=a['angle_deg'],**refs)
print(json.dumps({k:v for k,v in summary.items() if k not in ['selected','source_sha256']},indent=2))
print(json.dumps({k:float(r[k]) for k in ['t_s','t_p','T_s','T_p','in_s','in_p','brewster_deg','critical_deg']}))

