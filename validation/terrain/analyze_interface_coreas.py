#!/usr/bin/env python3
"""Independent common-current oracle and CoREAS/ZHS equivalence; run on PSR.

No fitted phase, time shift or normalization. Plane rays use a bracketed Snell
root and numerical launch-angle Jacobian. A Gauss current integral supplies
one common reference for both independent emission representations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=299792458.; Q=1.6021766208e-19; EPS=8.8541878128e-12

def error(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-100))

def index(m,p):
    if not m['radial_index']:return m['index']
    table=np.asarray(m['radial_index']);h=np.linalg.norm(p-np.asarray(m['center_m']))-m['reference_radius_m']
    return np.interp(h,table[:,0],table[:,1])

def optical(m,a,b,order=64):
    if not m['radial_index']:return np.linalg.norm(b-a)*m['index']/C
    nodes,weights=leggauss(order)
    return np.linalg.norm(b-a)/C*.5*sum(w*index(m,a+(x+1)*.5*(b-a)) for x,w in zip(nodes,weights))

def attenuation(m,length):
    return np.exp(-length/m['attenuation_length_m']) if m['attenuation_length_m'] else 1.

def ray(config,kind,source,region,observer,observer_region):
    ms,mo=config['media'][region],config['media'][observer_region]
    displacement=observer-source;length=np.linalg.norm(displacement)
    if region==observer_region:
        return optical(ms,source,observer),displacement/length,np.eye(3)*attenuation(ms,length)/length,index(ms,source)
    z=1. if kind=='mesh_boundary' else 0.
    normal=np.array([0.,0.,1. if region==1 else -1.])
    depth=np.dot(np.array([0.,0.,z])-source,normal)
    if depth<=0.:raise ValueError("quadrature source must be inside its medium")
    height=np.dot(observer-np.array([0.,0.,z]),normal)
    lateral=displacement-normal*np.dot(displacement,normal);distance=np.linalg.norm(lateral)
    ns,no=ms['index'],mo['index']
    root=brentq(lambda x:ns*x/np.hypot(depth,x)-no*(distance-x)/np.hypot(height,distance-x),0,distance,xtol=1.e-18,rtol=1.e-14) if distance else 0.
    point=source+depth*normal+(root*lateral/distance if distance else 0.)
    l1,l2=np.linalg.norm(point-source),np.linalg.norm(observer-point)
    ki,ko=(point-source)/l1,(observer-point)/l2;ci,ct=np.dot(ki,normal),np.dot(ko,normal)
    theta=np.arccos(np.clip(ci,-1,1));ts,tp=2*ns*ci/(ns*ci+no*ct),2*ns*ci/(no*ci+ns*ct)
    if theta<1.e-7:jac=(depth+ns/no*height)**2
    else:
        step=min(1.e-5,theta*.01,(np.arcsin(min(1.,no/ns))-theta)*.01)
        def horizontal(t):
            p=ns*np.sin(t);return depth*np.tan(t)+height*p/np.sqrt(no*no-p*p)
        derivative=(horizontal(theta+step)-horizontal(theta-step))/(2*step)
        jac=distance*ct*derivative/np.sin(theta)
    factor=np.sqrt(ns/(no*jac))*np.sqrt(no*ct/(ns*ci))*attenuation(ms,l1)*attenuation(mo,l2)
    s=np.cross(ki,normal)
    if np.linalg.norm(s)<1.e-12:s=np.cross(ki,[0.,1.,0.])
    s/=np.linalg.norm(s);ps=np.cross(s,ki);po=np.cross(s,ko)
    return (ns*l1+no*l2)/C,ki,factor*(ts*np.outer(s,s)+tp*np.outer(po,ps)),ns

def oracle(config,kind,tracks,frequencies,quadrature=64):
    # Independent finite-current integral. CoREAS endpoint pairs and ZHS must
    # converge to this same radiation-only far-field current, including each
    # visible interval and each one-sided material segment.
    output=np.zeros((len(config['observers']),3,len(frequencies)),complex);omega=2*np.pi*frequencies
    if kind=='boundary_grazing':quadrature*=16
    nodes,weights=leggauss(quadrature)
    for t in tracks:
        a=np.array([t[x] for x in ('sx','sy','sz')]);b=np.array([t[x] for x in ('ex','ey','ez')]);dt=t['t1']-t['t0'];beta=(b-a)/(dt*C)
        coeff=t['charge']*t['weight']*Q/(4*np.pi*EPS*C)
        intervals=[(0.,7./24.),(17./24.,1.)] if kind=='shadow' else [(0.,1.)]
        for oi,o in enumerate(config['observers']):
            for lower,upper in intervals:
                for x,w in zip(nodes,weights):
                    fraction=lower+(upper-lower)*(x+1)*.5
                    delay,k,m,n=ray(config,kind,a+(b-a)*fraction,int(t['region']),np.asarray(o['position_m']),o['region'])
                    amp=m@(beta-k*np.dot(beta,k))
                    output[oi]+=-1j*omega*coeff*dt*(upper-lower)*.5*w*amp[:,None]*np.exp(-1j*omega*(t['t0']+fraction*dt+delay))
    return output

def reconstruct(path,c):
    n=c['samples'];fs=c['sample_rate_Hz'];f=np.fft.rfftfreq(n,1/fs)
    def moments(name):
        data=np.fromfile(path/name,dtype='float64').reshape(c['moment_order']+1,len(c['observers']),2,3,n)
        transforms=np.fft.rfft(data,axis=-1);out=np.zeros(transforms.shape[1:],complex);power=np.ones(len(f),complex)
        for transform in transforms:out+=transform*power;power*=(-2j*np.pi*f/fs)
        return out.sum(axis=1)
    main=moments('moments.bin')
    field=main-2j*np.pi*f*moments('regularized_moments.bin') if c['algorithm'].startswith('CoREAS') else -2j*np.pi*f*main
    return f,field,field*np.exp(-2j*np.pi*f*c['start_time_s'])

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];cache={};fields={}
    for path in sorted(a.root.glob('*_CoREAS')):
        mode,kind=path.name.split('_',1);kind=kind[:-7];c=json.loads((path/'config.json').read_text());stats=json.loads((path/'statistics.json').read_text())
        tracks=np.atleast_1d(np.genfromtxt(path/'tracks.csv',delimiter=',',names=True));f,relative,field=reconstruct(path,c)
        if kind not in cache:
            fine=oracle(c,kind,tracks,f,128);coarse=oracle(c,kind,tracks,f,64);cache[kind]=(fine,error(coarse,fine))
        expected,convergence=cache[kind];absolute_error=error(field[:,:,:-1],expected[:,:,:-1])
        zhs_path=path.with_name(path.name[:-7]+'_ZHS');zhs_config=json.loads((zhs_path/'config.json').read_text())
        _,_,zhs_field=reconstruct(zhs_path,zhs_config);zhs_difference=error(field,zhs_field)
        spectrum=np.genfromtxt(path/'spectrum.csv',delimiter=',',names=True)
        written=np.stack([spectrum[x+'_real']+1j*spectrum[x+'_imag'] for x in ('Ex','Ey','Ez')],axis=-1).reshape(len(c['observers']),len(f),3).transpose(0,2,1)
        wave=np.genfromtxt(path/'field.csv',delimiter=',',names=True)
        written_wave=np.stack([wave[x+'_V_m'] for x in ('Ex','Ey','Ez')],axis=-1).reshape(len(c['observers']),c['samples'],3).transpose(0,2,1)
        target_wave=np.fft.irfft(relative*c['sample_rate_Hz'],n=c['samples'],axis=-1)
        checks=dict(finite=bool(np.isfinite(field).all()),absolute_current_oracle=absolute_error<5.e-3,oracle_convergence=convergence<1.e-8,renderer_spectrum=error(written,field)<1.e-11,renderer_waveform=error(written_wave,target_wave)<1.e-11)
        if kind in ('cherenkov','vacuum_forward'):
            checks['finite_track_limit']=stats['regularized_pairs']>0
            checks['shared_zhs_limit']=zhs_difference<1.e-12
        else:checks['ordinary_endpoints']=stats['endpoint_contributions']>0
        if kind in ('boundary','matched_boundary','mesh_boundary','boundary_reverse','matched_boundary_reverse','boundary_grazing'):checks['boundary_ownership']=stats['boundary_endpoints']==1
        checks['coreas_zhs_equivalence']=zhs_difference<1.e-9
        checks['zero_frequency_cancellation']=bool(np.linalg.norm(field[:,:,0])<1.e-12*max(np.linalg.norm(field[:,:,1:]),1.e-100))
        rows.append(dict(case=kind,backend=mode,checks=checks,absolute_complex_l2=absolute_error,zhs_relative_l2=zhs_difference,oracle_convergence=convergence,statistics=stats));fields[(mode,kind)]=field
        if mode=='CUDA':
            reference=np.fft.irfft(expected*np.exp(2j*np.pi*f*c['start_time_s'])*c['sample_rate_Hz'],n=c['samples'],axis=-1)
            time=(np.arange(c['samples'])/c['sample_rate_Hz']+c['start_time_s'])*1.e9;peak=int(np.argmax(np.linalg.norm(reference[0],axis=0)));sl=slice(max(0,peak-12),min(c['samples'],peak+13))
            fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            for j,color in enumerate(('tab:blue','tab:orange','tab:green')):
                axes[0].plot(time[sl],reference[0,j,sl],color=color,label='reference '+('Ex','Ey','Ez')[j]);axes[0].plot(time[sl],target_wave[0,j,sl],color=color,ls='--')
            axes[0].set(xlabel='Absolute time (ns)',ylabel='E (V/m)',title=kind+'; dashed = CUDA');axes[0].legend(fontsize=7)
            axes[1].semilogy(f[1:-1]/1.e6,np.linalg.norm(expected[0,:,1:-1],axis=0),label='Independent oracle');axes[1].semilogy(f[1:-1]/1.e6,np.linalg.norm(field[0,:,1:-1],axis=0),'--',label='CoREAS');axes[1].set(xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)',title='Complex L2 = %.3g'%absolute_error);axes[1].legend()
            for ax in axes:ax.grid(alpha=.25)
            fig.savefig(a.output/(kind+'.png'),dpi=150);fig.savefig(a.output/(kind+'.pdf'));plt.close(fig)
            np.savez_compressed(a.output/(kind+'.npz'),frequency_Hz=f,field_spectrum=field,reference_spectrum=expected,zhs_spectrum=zhs_field,time_ns=time,field=target_wave,reference=reference)
    pairs=[dict(case=k,relative_l2=error(v,fields[('OPENMP',k)])) for (m,k),v in fields.items() if m=='CUDA']
    cancellations=[dict(backend=m,direction=suffix or 'forward',relative_l2=error(fields[(m,'matched_boundary'+suffix)],fields[(m,'matched_unsplit'+suffix)])) for m in ('OPENMP','CUDA') for suffix in ('','_reverse')]
    passed=len(rows)==36 and all(all(r['checks'].values()) for r in rows) and len(pairs)==18 and all(r['relative_l2']<1.e-9 for r in pairs) and all(r['relative_l2']<1.e-10 for r in cancellations)
    report=dict(passed=passed,cases=rows,backend_pairs=pairs,matched_medium_cancellation=cancellations,scope='Common independent current oracle for paired CoREAS and ZHS, with absolute complex-spectrum equivalence and zero-frequency cancellation; direct/one-transmission real-ray GO only.')
    (a.output/'coreas_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
