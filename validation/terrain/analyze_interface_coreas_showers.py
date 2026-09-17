#!/usr/bin/env python3
"""PSR CoREAS shower checks including both moment grids and signed fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-100))

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--acceptance',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[];spectra={};hashes={}
    for manifest in args.acceptance:
        for row in json.loads(manifest.read_text()):
            if not all(row['checks'].values()):raise RuntimeError('unaccepted shower '+str(manifest))
            command=row['command'];root=Path(command[command.index('--output')+1])/'radio'
            c=json.loads((root/'config.json').read_text());n=c['samples'];fs=c['sample_rate_Hz'];no=len(c['observers'])
            frequency=np.fft.rfftfreq(n,1/fs)
            def synthesize(name):
                moments=np.fromfile(root/name,dtype=np.float64).reshape(c['moment_order']+1,no,2,3,n)
                result=np.zeros((no,2,3,len(frequency)),complex);factor=np.ones(len(frequency),complex)
                for order in range(c['moment_order']+1):
                    result+=np.fft.rfft(moments[order],axis=-1)*factor
                    factor*=(-2j*np.pi*frequency/fs)
                return result
            field=synthesize('moments.bin')-2j*np.pi*frequency*synthesize('regularized_moments.bin')
            total=field.sum(axis=1)
            sampled=field.copy();sampled[...,-1]=sampled[...,-1].real
            parts=np.fft.irfft(sampled*fs,n=n,axis=-1);wave=parts.sum(axis=1)
            time=c['start_time_s']+np.arange(n)/fs
            data=np.genfromtxt(root/'field.csv',delimiter=',',names=True)
            actual=np.stack([data[x] for x in ['Ex_V_m','Ey_V_m','Ez_V_m']],axis=-1).reshape(no,n,3).transpose(0,2,1)
            fft=np.genfromtxt(root/'spectrum.csv',delimiter=',',names=True)
            actual_spectrum=np.stack([fft[x+'_real']+1j*fft[x+'_imag'] for x in ['Ex','Ey','Ez']],axis=-1).reshape(no,len(frequency),3).transpose(0,2,1)
            stats=row['radio'];tag=stats['execution_space']+'_'+row['case']+'_'+row['variant']
            e=relative(actual,wave);ef=relative(actual_spectrum,total*np.exp(-2j*np.pi*frequency*c['start_time_s']))
            checks=dict(finite=bool(np.isfinite(wave).all()),nonzero=bool(np.max(np.abs(wave))>0),
                        public_renderer=e<1.e-12 and ef<1.e-10,accepted_sources=True)
            if row['case']=='electron_rock':checks['transmitted_rock_signal']=stats['transmitted_paths']>0 and bool(np.max(np.abs(parts[:,1]))>0)
            if row['case']=='nue_forced':checks['mixed_cpu_sources']=stats['cpu_tracks']>0 and stats['device_tracks']>0
            rows.append(dict(case=row['case'],variant=row['variant'],backend=stats['execution_space'],checks=checks,
                renderer_waveform_relative_l2=e,renderer_spectrum_relative_l2=ef,radio=stats,
                peak_field_V_m=float(np.max(np.linalg.norm(wave,axis=1)))))
            key=(row['case'],stats['execution_space'],row['variant']);spectra[key]=total;hashes[key]=row['csv_sha256']['tracks.csv']
            fig,axes=plt.subplots(no,3,figsize=(14,3.3*no),squeeze=False,constrained_layout=True)
            reference_peak=int(np.argmax(np.sum(np.linalg.norm(wave,axis=1),axis=0)))
            reference_field=max(float(np.max(np.abs(wave))),1.e-100)
            reference_parts=max(float(np.max(np.abs(parts))),1.e-100)
            reference_spectrum=max(float(np.max(np.linalg.norm(total,axis=1))),1.e-100)
            for o in range(no):
                zero=not np.any(wave[o]);peak=reference_peak if zero else int(np.argmax(np.linalg.norm(wave[o],axis=0)))
                sl=slice(max(0,peak-30),min(n,peak+31))
                for a,color in enumerate(['tab:blue','tab:orange','tab:green']):
                    axes[o,0].plot(time[sl]*1.e9,wave[o,a,sl],color=color,label=['Ex','Ey','Ez'][a])
                    axes[o,1].plot(time[sl]*1.e9,parts[o,0,a,sl],color=color,label='outside '+['Ex','Ey','Ez'][a])
                    axes[o,1].plot(time[sl]*1.e9,parts[o,1,a,sl],color=color,ls='--',label='inside '+['Ex','Ey','Ez'][a])
                axes[o,0].set(title=c['observers'][o]['name'],xlabel='Absolute time (ns)',ylabel='Signed E (V/m)')
                axes[o,1].set(title='Source-region contributions',xlabel='Absolute time (ns)',ylabel='Signed E (V/m)')
                if np.any(total[o]):
                    axes[o,2].semilogy(frequency[1:]/1.e6,np.linalg.norm(total[o,:,1:],axis=0))
                else:
                    axes[o,2].plot(frequency/1.e6,np.zeros(len(frequency)))
                    axes[o,2].set_ylim(0.,reference_spectrum*1.05)
                    axes[o,2].text(.5,.5,'Zero total spectrum',ha='center',transform=axes[o,2].transAxes)
                if zero:
                    axes[o,0].set_ylim(-reference_field*1.05,reference_field*1.05)
                    axes[o,0].text(.5,.85,'Zero total field in modeled branches',ha='center',fontsize=8,transform=axes[o,0].transAxes)
                if not np.any(parts[o]):axes[o,1].set_ylim(-reference_parts*1.05,reference_parts*1.05)
                axes[o,2].set(title='Total spectrum',xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)')
                axes[o,0].legend(fontsize=8);axes[o,1].legend(fontsize=7,ncol=2)
            for ax in axes.flat:ax.grid(alpha=.25)
            fig.suptitle(tag);fig.savefig(args.output/(tag+'.png'),dpi=150);fig.savefig(args.output/(tag+'.pdf'));plt.close(fig)
            np.savez_compressed(args.output/(tag+'.npz'),time_s=time,field_parts=parts,frequency_Hz=frequency,spectrum=total)
    pairs=[]
    for (case,backend,variant),spectrum in spectra.items():
        if variant!='resident':continue
        key=(case,backend,'batched')
        if key in spectra:pairs.append(dict(case=case,backend=backend,same_tracks=hashes[key]==hashes[(case,backend,variant)],relative_l2=relative(spectrum,spectra[key])))
    backend_pairs=[]
    for (case,backend,variant),spectrum in spectra.items():
        if backend!='Cuda':continue
        key=(case,'OpenMP',variant)
        if key in spectra:backend_pairs.append(dict(case=case,variant=variant,same_tracks=hashes[key]==hashes[(case,backend,variant)],relative_l2=relative(spectrum,spectra[key])))
    passed=bool(rows) and all(all(r['checks'].values()) for r in rows) and all(r['same_tracks'] and r['relative_l2']<1.e-9 for r in pairs)
    passed=passed and len(rows)==16 and all(not r['same_tracks'] or r['relative_l2']<1.e-9 for r in backend_pairs)
    report=dict(passed=passed,showers=rows,scheduler_pairs=pairs,backend_pairs=backend_pairs,
        scope='Actual DEM showers; fixed-source propagation accuracy is checked separately by the independent waveform oracle.')
    (args.output/'shower_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
