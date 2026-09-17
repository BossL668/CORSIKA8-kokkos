#!/usr/bin/env python3
"""Compare actual unchanged air CoREAS with paired interface CoREAS and ZHS.

Uniform-air limit only. No fitted amplitude/phase/time. The original air kernel
uses nearest samples; its analytic endpoint and sampling errors are separated
from the common far-field limit. Run on PSR, not on the busy local host.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_interface_coreas import C,Q,EPS,error,reconstruct

def air_reference(c,tracks,f):
    output=np.zeros((3,len(f)),complex);binned=output.copy();bound=np.zeros(len(f))
    omega=2*np.pi*f;fs=c['sample_rate_Hz'];start=c['start_time_s'];observer=np.asarray(c['observers'][0]['position_m']);n=c['media'][0]['index']
    for t in tracks:
        a=np.array([t[x] for x in ('sx','sy','sz')]);b=np.array([t[x] for x in ('ex','ey','ez')]);beta=(b-a)/((t['t1']-t['t0'])*C)
        coefficient=t['charge']*t['weight']*Q/(4*np.pi*EPS*C)
        for sign,point,time in ((-1.,a,t['t0']),(1.,b,t['t1'])):
            displacement=observer-point;distance=np.linalg.norm(displacement);k=displacement/distance
            impulse=sign*coefficient*(beta-k*np.dot(beta,k))/((1-n*np.dot(beta,k))*distance)
            delay=time+n*distance/C-start;cell=np.floor(delay*fs+.5)
            output+=impulse[:,None]*np.exp(-1j*omega*delay)
            binned+=impulse[:,None]*np.exp(-1j*omega*cell/fs)
            bound+=np.linalg.norm(impulse)*np.minimum(2.,omega/(2*fs))
    phase=np.exp(-1j*omega*start)
    return output*phase,binned*phase,bound

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];fields={}
    for root in sorted(a.root.iterdir()):
        if not root.is_dir():continue
        meta=json.loads((root/'air_reference.json').read_text());c=json.loads((root/'interface_CoREAS/config.json').read_text());z=json.loads((root/'interface_ZHS/config.json').read_text())
        tracks=np.atleast_1d(np.genfromtxt(root/'tracks.csv',delimiter=',',names=True));f,_,coreas=reconstruct(root/'interface_CoREAS',c);_,_,zhs=reconstruct(root/'interface_ZHS',z)
        coreas=coreas[0];zhs=zhs[0]
        data=np.genfromtxt(root/'air_field.csv',delimiter=',',names=True);wave=np.stack([data[x+'_V_m'] for x in ('Ex','Ey','Ez')])
        actual=np.fft.rfft(wave,axis=-1)/c['sample_rate_Hz']*np.exp(-2j*np.pi*f*c['start_time_s'])
        unbinned,binned,bound=air_reference(c,tracks,f);band=(f>=30.e6)&(f<=100.e6)
        norm=max(np.linalg.norm(coreas[:,band]),1.e-100)
        sampling=float(np.linalg.norm(actual[:,band]-unbinned[:,band])/norm)
        sampling_bound=float(np.linalg.norm(bound[band])/norm)
        geometry=float(np.linalg.norm(unbinned[:,band]-coreas[:,band])/norm)
        total=error(actual[:,band],coreas[:,band]);equivalence=error(coreas,zhs)
        checks=dict(actual_air_endpoints=meta['air_endpoint_contributions']==2*len(tracks),
            actual_air_matches_its_samples=error(actual,binned)<1.e-8,
            coreas_zhs_same_field=equivalence<1.e-9,
            air_sampling_bound=sampling<=sampling_bound*(1+1.e-8),
            controlled_total_error=total<=sampling_bound+geometry+1.e-10)
        if meta['distance_m']==10000.:
            checks['uniform_air_far_field_limit']=geometry<5.e-4
            if meta['sample_rate_Hz']==32.e9:checks['high_rate_air_agreement']=total<.02
        row=dict(**meta,checks=checks,coreas_zhs_complex_l2=equivalence,actual_air_sampled_reference_l2=error(actual,binned),
                 air_vs_interface_band_l2=total,air_sampling_band_l2=sampling,air_sampling_bound_l2=sampling_bound,
                 air_independent_endpoints_vs_common_path_l2=geometry,comparison_band_Hz=[30.e6,100.e6])
        rows.append(row);fields[(meta['execution'],meta['distance_m'],meta['sample_rate_Hz'])]=actual
        if meta['execution']=='Cuda':
            fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            for values,label,style in ((zhs,'Interface ZHS','-'),(coreas,'Interface CoREAS','--'),(actual,'Unchanged air CoREAS',':')):
                axes[0].plot(f[band]/1.e6,np.linalg.norm(values[:,band],axis=0),style,label=label)
            axes[0].set(xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)',title='R = %g m, sampling = %g GHz'%(meta['distance_m'],meta['sample_rate_Hz']/1.e9));axes[0].legend(fontsize=8)
            denom=np.maximum(np.linalg.norm(coreas[:,band],axis=0),1.e-100)
            axes[1].plot(f[band]/1.e6,np.linalg.norm(actual[:,band]-coreas[:,band],axis=0)/denom,label='Air vs interface')
            axes[1].plot(f[band]/1.e6,bound[band]/denom,'--',label='Nearest-sample bound')
            axes[1].set(xlabel='Frequency (MHz)',ylabel='Relative complex residual',title='No fitted shifts, phases or normalization');axes[1].legend(fontsize=8)
            for ax in axes:ax.grid(alpha=.25)
            fig.savefig(a.output/(root.name+'.png'),dpi=150);fig.savefig(a.output/(root.name+'.pdf'));plt.close(fig)
            np.savez_compressed(a.output/(root.name+'.npz'),frequency_Hz=f,interface_coreas=coreas,interface_zhs=zhs,air_actual=actual,air_unbinned=unbinned,air_nearest_sample_reference=binned)
    convergence=[]
    for mode in ('OpenMP','Cuda'):
        series=sorted([r for r in rows if r['execution']==mode and r['sample_rate_Hz']==32.e9],key=lambda r:r['distance_m'])
        errors=[r['air_independent_endpoints_vs_common_path_l2'] for r in series]
        convergence.append(dict(execution=mode,distances_m=[r['distance_m'] for r in series],errors=errors,
            passed=len(series)==3 and all(errors[i+1]<errors[i] for i in range(2)) and errors[-1]<errors[0]/50.))
    pairs=[dict(distance_m=r,rate_Hz=fs,relative_l2=error(v,fields[('OpenMP',r,fs)])) for (mode,r,fs),v in fields.items() if mode=='Cuda']
    passed=len(rows)==18 and all(all(r['checks'].values()) for r in rows) and all(r['passed'] for r in convergence) and len(pairs)==9 and all(r['relative_l2']<1.e-9 for r in pairs)
    report=dict(passed=passed,cases=rows,far_field_convergence=convergence,backend_pairs=pairs,
        scope='Actual unmodified beta5 air accumulateCoREAS kernel in homogeneous n=1.0003 air. Independent nearest-sample reference and analytic sampling bound; common-current CoREAS/ZHS equality is tested separately from the finite-distance air endpoint approximation.')
    (a.output/'air_comparison_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
