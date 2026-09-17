#!/usr/bin/env python3
"""PSR fixed-track oracle: Gauss current integration vs device interval moments.

No fitted amplitudes, peak shifts, or fitted phases. The independent reference
integrates the current along each track using a bracketed Fermat root and a
numerical ray-tube Jacobian, without the device interval or moment routines.
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

C=299792458.
Q=1.6021766208e-19
EPS=8.8541878128e-12

def unit(v):
    return v/np.linalg.norm(v)

def ray(source,observer,meta):
    displacement=observer-source
    if meta['kind'] in ('uniform','radial','mesh_shadow'):
        length=np.linalg.norm(displacement)
        direction=displacement/length
        index=meta['n0']
        if meta['kind']=='radial':
            nodes,weights=leggauss(32)
            positions=source+(nodes[:,None]+1)*.5*displacement
            index=float(np.dot(weights,1.5+.01*(np.linalg.norm(positions+[0.,0.,100.],axis=1)-100.))*.5)
        return index*length/C,direction,np.eye(3)/length
    ns,no=meta['n1'],meta['n0']
    plane=1. if meta['kind']=='mesh_transmitted' else 0.
    ds,do=plane-source[2],observer[2]-plane
    lateral=displacement.copy();lateral[2]=0
    distance=np.linalg.norm(lateral)
    if distance:
        root=brentq(lambda x:ns*x/np.hypot(ds,x)-no*(distance-x)/np.hypot(do,distance-x),
                    0,distance,xtol=1.e-13,rtol=1.e-14)
        point=source.copy();point[2]=plane;point+=root*lateral/distance
    else:point=np.array([source[0],source[1],plane])
    l1,l2=np.linalg.norm(point-source),np.linalg.norm(observer-point)
    ki,ko=(point-source)/l1,(observer-point)/l2
    ci,ct=ki[2],ko[2]
    theta=np.arccos(np.clip(ci,-1,1))
    ts,tp=2*ns*ci/(ns*ci+no*ct),2*ns*ci/(no*ci+ns*ct)
    if theta<1.e-7:
        jac=(ds+ns/no*do)**2
    else:
        # Independently perturb launch angle and measure receiving area per
        # solid angle. This does not use the production analytic derivative.
        delta=min(1.e-5,theta*.01,(np.arcsin(min(1.,no/ns))-theta)*.01)
        def horizontal(t):
            p=ns*np.sin(t)
            return ds*np.tan(t)+do*p/np.sqrt(no*no-p*p)
        derivative=(horizontal(theta+delta)-horizontal(theta-delta))/(2*delta)
        jac=distance*ct*derivative/np.sin(theta)
    geometric=np.sqrt(ns/(no*jac))
    factor=geometric*np.sqrt(no*ct/(ns*ci))*np.exp(-l1/meta['attenuation1'])
    s=np.cross(ki,[0.,0.,1.])
    if np.linalg.norm(s)<1.e-12:s=np.cross(ki,[0.,1.,0.])
    s=unit(s);ps=unit(np.cross(s,ki));po=unit(np.cross(s,ko))
    matrix=factor*(ts*np.outer(s,s)+tp*np.outer(po,ps))
    return (ns*l1+no*l2)/C,ki,matrix

def reference(meta,tracks,frequencies,quadrature):
    nodes,weights=leggauss(quadrature)
    output=np.zeros((len(meta['observers']),3,len(frequencies)),dtype=complex)
    omega=2*np.pi*frequencies
    for track in tracks:
        a=np.array([track[x] for x in ('x0','y0','z0')]);b=np.array([track[x] for x in ('x1','y1','z1')])
        dt=track['t1']-track['t0'];beta=(b-a)/(dt*C)
        coefficient=track['charge']*track['weight']*Q/(4*np.pi*EPS*C)
        for oi,observer in enumerate(meta['observers']):
            # Analytic cube silhouette: a horizontal source at z=2, receiver
            # at z=-3 is visible for |source_x|>2.5. Integrate the two visible
            # intervals directly; no device visibility routine is reused.
            intervals=[(0.,7./24.),(17./24.,1.)] if meta['kind']=='mesh_shadow' else [(0.,1.)]
            for lower,upper in intervals:
                for node,weight in zip(nodes,weights):
                    fraction=lower+(upper-lower)*(node+1)/2
                    delay,direction,matrix=ray(a+(b-a)*fraction,np.asarray(observer),meta)
                    amplitude=matrix@(beta-direction*np.dot(direction,beta))
                    detection=track['t0']+fraction*dt+delay
                    output[oi]+=coefficient*dt*.5*(upper-lower)*weight*amplitude[:,None]*np.exp(-1j*omega*detection)
    return -1j*omega*output

def render(meta,values):
    n=meta['samples'];fs=meta['sample_rate'];observers=len(meta['observers'])
    frequencies=np.fft.rfftfreq(n,1/fs)
    moments=values.reshape(meta['order']+1,observers,2,3,n)
    transforms=np.fft.rfft(moments,axis=-1)
    potential=np.zeros(transforms.shape[1:],complex)
    factor=np.ones(len(frequencies),complex)
    for p in range(meta['order']+1):
        potential+=transforms[p]*factor
        factor*=(-2j*np.pi*frequencies/fs)
    field=-2j*np.pi*frequencies*potential
    return frequencies,field.sum(axis=1),field

def norm_error(actual,expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1.e-100))

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=float,default=.01)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];spectra={};references={}
    for path in sorted(args.root.glob('*_p*_l*.json')):
        meta=json.loads(path.read_text());tracks=np.atleast_1d(np.genfromtxt(path.with_name(path.stem+'_tracks.csv'),delimiter=',',names=True))
        frequencies,field,parts=render(meta,np.fromfile(path.with_suffix('.bin'),dtype=np.float64))
        key=(meta['kind'],path.with_name(path.stem+'_tracks.csv').read_text())
        if key not in references:
            coarse=reference(meta,tracks,frequencies,64)
            fine=reference(meta,tracks,frequencies,128)
            references[key]=(fine,norm_error(coarse,fine))
        expected,oracle_error=references[key]
        errors=[float(norm_error(field[i,:,1:-1],expected[i,:,1:-1])) for i in range(len(field))]
        unused=parts[:,1 if meta['kind'] in ('uniform','radial','mesh_shadow') else 0]
        checks=dict(oracle_converged=oracle_error<1.e-7,
                    absolute_complex_spectrum=max(errors)<args.limit,
                    no_wrong_region_signal=bool(np.count_nonzero(unused)==0),
                    cpu_device_input=meta['cpu_device_input_relative_difference']<2.e-13)
        renderer={}
        rendered=path.with_name(path.stem+'_rendered')
        if rendered.exists():
            rc=json.loads((rendered/'config.json').read_text())
            raw=np.genfromtxt(rendered/'field.csv',delimiter=',',names=True)
            fft=np.genfromtxt(rendered/'spectrum.csv',delimiter=',',names=True)
            rf=field.copy();rf[:,:,-1]=rf[:,:,-1].real
            target=np.fft.irfft(rf*meta['sample_rate'],n=meta['samples'],axis=-1)
            actual=np.stack([raw[x] for x in ('Ex_V_m','Ey_V_m','Ez_V_m')],axis=-1).reshape(len(field),meta['samples'],3).transpose(0,2,1)
            spectrum=np.stack([fft[x+'_real']+1j*fft[x+'_imag'] for x in ('Ex','Ey','Ez')],axis=-1).reshape(len(field),len(frequencies),3).transpose(0,2,1)
            renderer=dict(waveform_relative_l2=norm_error(actual,target),
                absolute_spectrum_relative_l2=norm_error(spectrum,field*np.exp(-2j*np.pi*frequencies*rc['start_time_s'])),
                json_escape=rc['observers'][0]['name']=='normal\n"quoted"')
            checks['public_renderer']=renderer['waveform_relative_l2']<1.e-12 and renderer['absolute_spectrum_relative_l2']<1.e-11 and renderer['json_escape']
        rows.append(dict(file=path.name,kind=meta['kind'],execution=meta['execution'],checks=checks,
                         order=meta['order'],limit=meta['limit'],
                         per_observer_complex_l2=errors,oracle_quadrature_relative_difference=oracle_error,
                         acceptance_relative_l2=args.limit,paths=meta['paths'],leaves=meta['leaves'],renderer=renderer))
        spectra[(meta['kind'],meta['order'],meta['limit'],meta['execution'])]=field
        f=field.copy();ref=expected.copy();f[:,:,-1]=f[:,:,-1].real;ref[:,:,-1]=ref[:,:,-1].real
        waves=np.fft.irfft(f*meta['sample_rate'],n=meta['samples'],axis=-1)
        target=np.fft.irfft(ref*meta['sample_rate'],n=meta['samples'],axis=-1)
        time=np.arange(meta['samples'])/meta['sample_rate']*1.e9
        fig,axes=plt.subplots(len(field),3,figsize=(14,3.3*len(field)),constrained_layout=True,squeeze=False)
        for i in range(len(field)):
            peak=int(np.argmax(np.linalg.norm(target[i],axis=0)));sl=slice(max(0,peak-20),min(len(time),peak+21))
            for j,color in enumerate(('tab:blue','tab:orange','tab:green')):
                axes[i,0].plot(time[sl],target[i,j,sl],color=color,label='reference '+str(j))
                axes[i,0].plot(time[sl],waves[i,j,sl],color=color,ls='--')
            axes[i,0].set(title=f'Observer {i}: signed E, solid=reference',xlabel='Absolute time (ns)',ylabel='V/m')
            axes[i,1].semilogy(frequencies[1:-1]/1.e6,np.linalg.norm(expected[i,:,1:-1],axis=0),label='Gauss current integral')
            axes[i,1].semilogy(frequencies[1:-1]/1.e6,np.linalg.norm(field[i,:,1:-1],axis=0),'--',label=meta['execution'])
            axes[i,1].set(title=f'Complex-spectrum L2 = {errors[i]:.3g}',xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)')
            axes[i,1].legend(fontsize=8)
            for j in range(3):axes[i,2].plot(time[sl],(waves-target)[i,j,sl])
            axes[i,2].set(title='Absolute residual, no fitted shifts/scales',xlabel='Absolute time (ns)',ylabel='V/m')
        for ax in axes.flat:ax.grid(alpha=.25)
        fig.savefig(args.output/(path.stem+'.png'),dpi=160);fig.savefig(args.output/(path.stem+'.pdf'));plt.close(fig)
        np.savez_compressed(args.output/(path.stem+'.npz'),frequency_Hz=frequencies,field_spectrum=field,reference_spectrum=expected,time_ns=time,field=waves,reference=target)
    pairs=[]
    for key,field in spectra.items():
        if key[-1]!='Cuda':continue
        cpu=spectra.get(key[:-1]+('OpenMP',))
        if cpu is not None:pairs.append(dict(case=list(key[:-1]),relative_l2=float(norm_error(field,cpu))))
    convergence=[];orders=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for kind in sorted({r['kind'] for r in rows}):
        selected=sorted([r for r in rows if r['kind']==kind and r['execution']=='Cuda' and r['order']==16],key=lambda r:r['limit'])
        if len(selected)>1:
            errors=[max(r['per_observer_complex_l2']) for r in selected]
            convergence.append(dict(kind=kind,fine_error=errors[0],coarse_error=errors[-1],improved=errors[0]<errors[-1]))
            axes[0].loglog([r['limit'] for r in selected],errors,'o-',label=kind)
            axes[1].loglog([r['leaves'] for r in selected],errors,'o-',label=kind)
    for key,field in spectra.items():
        if key[1]!=12:continue
        ref=spectra.get((key[0],16,key[2],key[3]))
        if ref is not None:orders.append(dict(case=list(key),relative_l2=norm_error(field,ref)))
    axes[0].set(xlabel='Subdivision control',ylabel='Maximum observer relative complex L2')
    axes[1].set(xlabel='Integration leaves',ylabel='Maximum observer relative complex L2')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(args.output/'convergence.png',dpi=170);fig.savefig(args.output/'convergence.pdf');plt.close(fig)
    passed=len(rows)>=4 and all(all(r['checks'].values()) for r in rows) and len(pairs)>=2 and all(p['relative_l2']<1.e-9 for p in pairs)
    passed=passed and all(r['improved'] for r in convergence) and all(r['relative_l2']<1.e-6 for r in orders)
    report=dict(passed=passed,cases=rows,backend_pairs=pairs,spatial_convergence=convergence,moment_order_pairs=orders,
        scope='Fixed charged tracks, uniform, radial-index, planar transmission and closed-mesh visibility/transmission; independent current integral and signed field with absolute timing; shower acceptance is separate.')
    (args.output/'waveform_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
