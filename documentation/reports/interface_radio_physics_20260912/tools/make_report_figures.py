#!/usr/bin/env python3
"""Numerical comparisons and publication plots. Run only on PSR."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'external/python'))
import tmm
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]; STAGE=ROOT.parent
DATA=ROOT/'data'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
C4=STAGE/'interface-coreas-20260912'; PAIR=STAGE/'interface-radio-pair-20260912'
C=299792458.; BLUE='#1769aa'; ORANGE='#d55e00'; GREEN='#16826c'
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,
    'legend.fontsize':10,'figure.dpi':110,'savefig.dpi':180,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':.2,'font.family':'DejaVu Sans','pdf.fonttype':42})
manifest=[];summary={}
def save(fig,name,sources):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    manifest.append(dict(figure=name,sources=[str(s) for s in sources]))
def read(name):return np.genfromtxt(DATA/name,delimiter=',',names=True)
def l2(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-100))

# Independent public-library geometry and plane-wave boundary coefficients.
a=read('beta5_plane.csv');rp=read('radiopropa_plane.csv')
assert len(a)==len(rp) and np.array_equal(a['angle_deg'],rp['angle_deg'])
valid=(a['transmitted']>0)&(rp['transmitted']>0)
valid &= np.abs(a['angle_deg']-30)>1.e-8 # singular equality has no transmitted flux
snell=float(np.max(abs(a['sin_receive'][valid]-rp['sin_receive'][valid])))
tir=bool(np.array_equal(a['transmitted'],rp['transmitted']))
fig,ax=plt.subplots(1,2,figsize=(11.8,4),constrained_layout=True)
for n1,color in [(2.,BLUE),(1.0003,ORANGE)]:
    m=valid&(a['n1']==n1);label='2 → 1' if n1==2 else '1.0003 → 2'
    ax[0].plot(a['angle_deg'][m],a['sin_receive'][m],color=color,label='beta5 '+label)
    idx=np.flatnonzero(m)[::12]
    ax[0].plot(rp['angle_deg'][idx],rp['sin_receive'][idx],'o',ms=4,mfc='none',color=color,label='RadioPropa '+label)
    ax[1].semilogy(a['angle_deg'][m],np.maximum(abs(a['sin_receive'][m]-rp['sin_receive'][m]),1.e-17),color=color,label=label)
ax[0].axvline(30,color='gray',ls=':',label='critical angle (2 → 1)')
ax[0].set(xlabel='Incidence angle (deg)',ylabel='sin(transmitted angle)',title='Independent forward ray / receiver-conditioned ray')
ax[1].set(xlabel='Incidence angle (deg)',ylabel='Absolute sin-angle residual',title='Identical TIR classification: 642 / 642')
for x in ax:x.legend()
save(fig,'01_public_snell',[DATA/'beta5_plane.csv',DATA/'radiopropa_plane.csv'])

ts=np.zeros(len(a));tp=ts.copy();rs=ts.copy();rpp=ts.copy();T_s=ts.copy();T_p=ts.copy()
for i in np.flatnonzero(valid):
    r=a[i];ti=np.deg2rad(r['angle_deg']);tt=tmm.snell(r['n1'],r['n2'],ti)
    for pol,arr,refl,T in [('s',ts,rs,T_s),('p',tp,rpp,T_p)]:
        arr[i]=tmm.interface_t(pol,r['n1'],r['n2'],ti,tt)
        refl[i]=tmm.R_from_r(tmm.interface_r(pol,r['n1'],r['n2'],ti,tt))
        T[i]=tmm.T_from_t(pol,arr[i],r['n1'],r['n2'],ti,tt)
fresnel=max(float(np.max(abs(a[k][valid]/ref[valid]-1))) for k,ref in [('t_s',ts),('t_p',tp)])
flux=max(float(np.max(abs(a[k][valid]**2-ref[valid]))) for k,ref in [('flux_s',T_s),('flux_p',T_p)])
fig,ax=plt.subplots(1,3,figsize=(12.4,3.8),constrained_layout=True)
for j,n1 in enumerate([2.,1.0003]):
    m=valid&(a['n1']==n1)
    for pol,color,ref in [('s',BLUE,ts),('p',ORANGE,tp)]:
        ax[j].plot(a['angle_deg'][m],a['t_'+pol][m],color=color,label='beta5 '+pol)
        idx=np.flatnonzero(m)[::10];ax[j].plot(a['angle_deg'][idx],ref[idx],'o',ms=3,mfc='none',color=color,label='TMM '+pol)
    ax[j].set(xlabel='Incidence angle (deg)',ylabel='Electric-field coefficient t',title='n = 2 → 1' if j==0 else 'n = 1.0003 → 2');ax[j].legend(ncol=2)
m=valid&(a['n1']==2.)
for pol,color,R in [('s',BLUE,rs),('p',ORANGE,rpp)]:
    ax[2].plot(a['angle_deg'][m],a['flux_'+pol][m]**2,color=color,label='beta5 transmitted '+pol)
    ax[2].plot(a['angle_deg'][m],R[m],color=color,ls=':',label='TMM reflected '+pol)
ax[2].axhline(1,color='gray',lw=1);ax[2].set(xlabel='Incidence angle (deg)',ylabel='Power fraction',title='R + T = 1; beta5 keeps transmission')
ax[2].legend(fontsize=9)
save(fig,'02_public_fresnel',[DATA/'beta5_plane.csv','tmm 0.2.0 interface_t / T_from_t / interface_r'])

from make_raytube_error_figure import render_raytube_error
residual=render_raytube_error(DATA,FIG/'03_public_raytube')
manifest.append(dict(figure='03_public_raytube',sources=[str(DATA/'radiopropa_beam.csv'),str(DATA/'beta5_plane.csv')]))
beam_error=max(r['relative_error'][-1] for r in residual)

# Deliberately show, rather than calibrate away, the pinned library amplitude mismatch.
m=valid&(a['n1']==2.);amp=abs(rp['amplitude_s'][m]/a['flux_s'][m]-1)
fig,ax=plt.subplots(1,2,figsize=(11.8,4),constrained_layout=True)
ax[0].plot(a['angle_deg'][m],a['flux_s'][m],color=BLUE,label='beta5 / TMM, flux-normalized s')
ax[0].plot(a['angle_deg'][m],rp['amplitude_s'][m],color=ORANGE,ls='--',label='RadioPropa vector amplitude')
ax[0].set(xlabel='Incidence angle (deg)',ylabel='Amplitude factor',title='Pinned revision 544a2d6; pure s input');ax[0].legend()
ax[1].plot(a['angle_deg'][m],100*amp,color=ORANGE)
ax[1].set(xlabel='Incidence angle (deg)',ylabel='Relative amplitude difference (%)',title='Diagnostic discrepancy; not an acceptance reference')
save(fig,'04_public_amplitude_caution',[DATA/'radiopropa_plane.csv',DATA/'beta5_plane.csv'])

# Gradient case: validate the external ray, then quantify beta5's known straight-ray approximation.
raw=[]
with (DATA/'radiopropa_gradient.csv').open() as stream:
    raw=list(csv.DictReader(stream))
gr=[r for r in raw if r['record']=='gradient'];sample=[r for r in raw if r['record']=='path']
z=np.array([float(r['b']) for r in sample]);x=np.array([float(r['c']) for r in sample])
g=read('beta5_gradient.csv')
n=lambda z:2.-2.e-4*z
integral=lambda p:quad(lambda z:p/np.sqrt(n(z)**2-p*p),0,1000,epsabs=1.e-10)[0]
invariant=brentq(lambda p:integral(p)-600,.1,1.79,xtol=1.e-14)
t_bent=quad(lambda z:n(z)**2/np.sqrt(n(z)**2-invariant**2)/C,0,1000,epsabs=1.e-15)[0]
t_straight=1.9*np.hypot(600,1000)/C
rp_best=min(gr,key=lambda r:float(r['a']));rp_time=float(rp_best['c'])
gradient=dict(profile='n(z)=2-2e-4 z',endpoint=[600,0,1000],invariant=invariant,
    radiopropa_time_s=rp_time,analytic_bent_time_s=t_bent,beta5_time_s=float(g['time_s'][-1]),
    analytic_straight_time_s=t_straight,model_time_difference_ns=float((g['time_s'][-1]-rp_time)*1.e9),
    maximum_path_offset_m=float(np.max(abs(x-.6*z))),
    external_time_error_ps=abs(rp_time-t_bent)*1.e12,
    straight_implementation_error_ps=abs(g['time_s'][-1]-t_straight)*1.e12)
fig,ax=plt.subplots(1,2,figsize=(11.8,4),constrained_layout=True)
ax[0].plot(x,z,color=ORANGE,label='RadioPropa curved eikonal ray')
ax[0].plot(g['x_m'],g['z_m'],color=BLUE,ls='--',label='beta5 straight chord')
ax[0].set(xlabel='Horizontal x (m)',ylabel='Height z (m)',title='Same source and receiver; synthetic gradient');ax[0].legend()
ax[1].plot(z,x-.6*z,color=ORANGE);ax[1].axhline(0,color=BLUE,ls='--')
ax[1].set(xlabel='Height z (m)',ylabel='Curved ray x − straight chord x (m)',title='Model difference, not a numerical failure')
ax[1].text(.05,.94,'Travel-time difference: {:.4f} ns\nExternal integration error: {:.4f} ps'.format(gradient['model_time_difference_ns'],gradient['external_time_error_ps']),transform=ax[1].transAxes,va='top')
save(fig,'05_gradient_model_boundary',[DATA/'beta5_gradient.csv',DATA/'radiopropa_gradient.csv'])

att=read('beta5_attenuation.csv');reference=np.exp(-att['distance_m']/100)/att['distance_m']
atten_error=float(np.max(abs(att['amplitude_per_m']/reference-1)))
fig,ax=plt.subplots(1,2,figsize=(11.8,3.8),constrained_layout=True)
ax[0].semilogy(att['distance_m'],att['amplitude_per_m'],color=BLUE,label='beta5 production transfer')
ax[0].semilogy(att['distance_m'][::12],reference[::12],'o',mfc='none',color=ORANGE,label='exp(−R / 100 m) / R')
ax[0].semilogy(att['distance_m'],1/att['distance_m'],color='gray',ls=':',label='Geometric 1/R only')
ax[0].set(xlabel='Distance R (m)',ylabel='Field transfer (1/m)',title='Amplitude attenuation, not power attenuation');ax[0].legend()
ax[1].plot(att['distance_m'],att['time_s']*1.e6,color=BLUE,label='beta5')
ax[1].plot(att['distance_m'][::12],1.33*att['distance_m'][::12]/C*1.e6,'o',mfc='none',color=ORANGE,label='n R/c, n = 1.33')
ax[1].set(xlabel='Distance R (m)',ylabel='Travel time (µs)',title='Direct uniform-medium transfer');ax[1].legend()
save(fig,'06_attenuation',[DATA/'beta5_attenuation.csv'])
summary['external']=dict(radiopropa_commit='544a2d6c4e284e3d724cb741dc481245a0f633d7',tmm_version=version('tmm'),
    rows=len(a),snell_max_absolute=snell,tir_classification_exact=tir,
    fresnel_max_relative=fresnel,flux_power_max_absolute=flux,
    raytube_max_relative_h1e6=beam_error,raytube_convergence=residual,
    radiopropa_amplitude_difference_max=float(np.max(amp)),gradient=gradient,
    attenuation_max_relative=atten_error)
checks=dict(snell=snell<1.e-12,tir=tir,fresnel=fresnel<1.e-11,flux=flux<1.e-11,
    raytube=beam_error<1.e-7,external_gradient=gradient['external_time_error_ps']<.1,
    straight_optical=gradient['straight_implementation_error_ps']<.1,attenuation=atten_error<1.e-12)
checks={k:bool(v) for k,v in checks.items()}
summary['external']['checks']=checks
assert all(checks.values()),checks
np.savez_compressed(DATA/'external_arrays.npz',beta5=a,radiopropa=rp,tmm_ts=ts,tmm_tp=tp,T_s=T_s,T_p=T_p,valid=valid)

# Accepted emission diagnostics, with the fixed-track accuracy and algorithm agreement separated.
oracle=json.loads((C4/'oracle-c4/coreas_summary.json').read_text());assert oracle['passed']
cases=[r for r in oracle['cases'] if r['backend']=='CUDA'];names=[r['case'] for r in cases]
fig,ax=plt.subplots(figsize=(12.2,5.2),constrained_layout=True)
x=np.arange(len(cases))
ax.semilogy(x,[max(r['absolute_complex_l2'],1.e-16) for r in cases],'o-',color=ORANGE,label='CoREAS / independent current integral')
ax.semilogy(x,[max(r['zhs_relative_l2'],1.e-16) for r in cases],'s-',color=BLUE,label='CoREAS / ZHS (shared optical model)')
ax.set_xticks(x);ax.set_xticklabels(names,rotation=45,ha='right',fontsize=10)
ax.set(ylabel='Relative complex spectrum L2',title='Fixed-track tests: external accuracy and internal consistency answer different questions')
ax.legend(loc='upper right');save(fig,'07_emission_validation',[C4/'oracle-c4/coreas_summary.json'])
precision=json.loads((C4/'precision-c4/summary.json').read_text());assert precision['passed']
fig,ax=plt.subplots(figsize=(10.5,4.2),constrained_layout=True)
for r in precision['cases']:
    ax.semilogy([s['order'] for s in r['orders']],[s['complex_relative_l2'] for s in r['orders']],
        'o-' if r['backend']=='CUDA' else 'x--',label=r['backend']+' '+r['case'])
ax.set(xticks=[8,12,16,20],xlabel='Moment order P',ylabel='CoREAS / ZHS complex L2',title='Convergence at fixed tracks and optical subdivision');ax.legend(ncol=2)
save(fig,'08_moment_convergence',[C4/'precision-c4/summary.json'])

# Current paired application outputs: no amplitude normalization or time alignment.
accepted=json.loads((PAIR/'app-cuda-p1/acceptance.json').read_text());summary['showers']=[]
for row in accepted:
    assert all(row['checks'].values())
    root=Path(row['radio']['directory']);cfg=json.loads((root/'CoREAS/config.json').read_text())
    count=len(cfg['observers']);samples=cfg['samples'];nf=samples//2+1
    fields={};spectra={};parts={}
    for algorithm in ['CoREAS','ZHS']:
        dat=np.genfromtxt(root/algorithm/'field.csv',delimiter=',',names=True)
        fields[algorithm]=np.stack([dat[k] for k in ['Ex_V_m','Ey_V_m','Ez_V_m']]).reshape(3,count,samples).transpose(1,0,2)
        time=dat['time_s'].reshape(count,samples)[0]
        parts[algorithm]=np.stack([dat[k] for k in ['Ex_outside','Ey_outside','Ez_outside','Ex_inside','Ey_inside','Ez_inside']]).reshape(2,3,count,samples).transpose(2,0,1,3)
        spec=np.genfromtxt(root/algorithm/'spectrum.csv',delimiter=',',names=True)
        spectra[algorithm]=np.stack([spec[k+'_real']+1j*spec[k+'_imag'] for k in ['Ex','Ey','Ez']]).reshape(3,count,nf).transpose(1,0,2)
    freq=np.fft.rfftfreq(samples,1/cfg['sample_rate_Hz']);f0=fields['CoREAS'][0];f1=fields['ZHS'][0]
    peak=int(np.argmax(np.linalg.norm(f1,axis=0)));sl=slice(max(0,peak-30),min(samples,peak+31))
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,color in enumerate([BLUE,ORANGE,GREEN]):
        ax[0].plot(time[sl]*1.e6,f0[i,sl],color=color,label=['Ex','Ey','Ez'][i]+' CoREAS')
        ax[0].plot(time[sl]*1.e6,f1[i,sl],color=color,ls='--',lw=1.5)
    ax[0].set(xlabel='Absolute time (µs)',ylabel='Signed E (V/m)',title='Solid: CoREAS; dashed: ZHS');ax[0].legend()
    for algo,color,ls in [('CoREAS',BLUE,'-'),('ZHS',ORANGE,'--')]:
        ax[1].semilogy(freq[1:]/1.e6,np.linalg.norm(spectra[algo][0,:,1:],axis=0),color=color,ls=ls,label=algo)
    ax[1].set(xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)',title=cfg['observers'][0]['name']+'; unfiltered');ax[1].legend()
    name='09_electron_waveform' if row['case']=='electron_rock' else '10_neutrino_waveform'
    save(fig,name,[root/'CoREAS/field.csv',root/'ZHS/field.csv',root/'CoREAS/spectrum.csv',root/'ZHS/spectrum.csv'])
    fig,ax=plt.subplots(1,2,figsize=(11.8,3.8),constrained_layout=True)
    delta=fields['CoREAS']-fields['ZHS'];ax[0].plot(time[sl]*1.e6,delta[0,:,sl].T)
    ax[0].set(xlabel='Absolute time (µs)',ylabel='CoREAS − ZHS (V/m)',title='Signed waveform residual; no fitted shift')
    den=np.maximum(np.linalg.norm(spectra['ZHS'][0],axis=0),np.max(np.linalg.norm(spectra['ZHS'][0],axis=0))*1.e-12)
    error=np.linalg.norm(spectra['CoREAS'][0]-spectra['ZHS'][0],axis=0)/den
    ax[1].semilogy(freq[1:]/1.e6,np.maximum(error[1:],1.e-17),color=BLUE)
    ax[1].set(xlabel='Frequency (MHz)',ylabel='Relative complex residual',title='Includes phase; denominator floor = peak × 1e−12')
    save(fig,'11_'+row['case']+'_residual',[root/'CoREAS/field.csv',root/'ZHS/field.csv',root/'CoREAS/spectrum.csv',root/'ZHS/spectrum.csv'])
    dominant=int(np.argmax(np.max(abs(f1),axis=1)))
    if row['case']=='nue_forced':
        fig,ax=plt.subplots(1,2,figsize=(11.8,3.8),constrained_layout=True)
        for values,label,color,ls in [(f0[dominant],'Total',BLUE,'-'),(parts['CoREAS'][0,0,dominant],'Outside source',GREEN,'--'),(parts['CoREAS'][0,1,dominant],'Inside source',ORANGE,':')]:
            ax[0].plot(time[sl]*1.e6,values[sl],color=color,ls=ls,label=label)
        ax[0].set(xlabel='Absolute time (µs)',ylabel='Signed '+['Ex','Ey','Ez'][dominant]+' (V/m)',title='Source-medium contributions add coherently');ax[0].legend()
        ax[1].semilogy(time*1.e6,np.maximum(np.linalg.norm(f0,axis=0),1.e-30),color=BLUE)
        ax[1].set(xlabel='Absolute time (µs)',ylabel='|E(t)| (V/m)',title='Full 1.536 ms recorded window')
        save(fig,'12_source_decomposition',[root/'CoREAS/field.csv'])
    command=row['command'];parameters={}
    for flag in ['--primary','--energy-GeV','--direction','--position-m','--seed','--magnetic-field','--emthin']:
        if flag in command:
            first=command.index(flag)+1
            parameters[flag]=command[first:first+3] if flag in ['--direction','--position-m'] else command[first]
    parameters['--force-vertex-cc']='--force-vertex-cc' in command
    summary['showers'].append(dict(case=row['case'],parameters=parameters,command=command,
        sampling=dict(samples=samples,fs_Hz=cfg['sample_rate_Hz'],observers=cfg['observers']),
        statistics=row['radio'],waveform_l2=l2(fields['CoREAS'],fields['ZHS']),
        spectrum_l2=l2(spectra['CoREAS'],spectra['ZHS']),peak_field_V_m=float(np.max(np.linalg.norm(f0,axis=0))),
        peak_time_s=float(time[peak])))
    np.savez_compressed(DATA/(row['case']+'_waveforms.npz'),time_s=time,frequency_Hz=freq,coreas_field=fields['CoREAS'],zhs_field=fields['ZHS'],
        coreas_spectrum=spectra['CoREAS'],zhs_spectrum=spectra['ZHS'],coreas_source_parts=parts['CoREAS'])

# Reuse recorded PSR plots where they already communicate the relevant physical test.
for source,name in [
    (C4/'oracle-c4/cherenkov.png','13_cherenkov'),
    (C4/'algorithm-comparison-c4/boundary_before_after.png','14_boundary_adaptation'),
    (C4/'native-optical-c4/native_optical_times.png','15_native_optical'),
    (C4/'air-comparison-c4/CUDA_R10000_fs32.png','16_air_reference')]:
    shutil.copy2(source,FIG/(name+'.png'));manifest.append(dict(figure=name,sources=[str(source)]))
residency=json.loads((PAIR/'residency.json').read_text());assert residency['passed']
from plot_residency_panel import plot as plot_gpu
plot_gpu(ROOT)
manifest.append(dict(figure='17_gpu_residency',sources=[str(PAIR/'profile-app/electron_rock_resident.jsonl')]))
summary['fixed_track_max_oracle_l2']=max(r['absolute_complex_l2'] for r in oracle['cases'])
summary['fixed_track_max_coreas_zhs_l2']=max(r['zhs_relative_l2'] for r in oracle['cases'])
summary['passed']=True
(DATA/'report_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
(DATA/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary['external'],indent=2))
print('Created',len(manifest),'diagnostic figures')
