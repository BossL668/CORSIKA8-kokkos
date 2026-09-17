#!/usr/bin/env python3
"""Identical full-shower sources: absolute CoREAS/ZHS field comparison on PSR."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_interface_coreas import error,reconstruct

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--coreas',type=Path,nargs='+',required=True)
    p.add_argument('--zhs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    references={}
    for manifest in a.zhs:
        for r in json.loads(manifest.read_text()):
            references[(r['radio']['execution_space'],r['case'],r['variant'])]=r
    rows=[]
    for manifest in a.coreas:
        for r in json.loads(manifest.read_text()):
            key=(r['radio']['execution_space'],r['case'],r['variant']);z=references[key]
            cr=Path(r['radio']['directory']);zr=Path(z['radio']['directory'])
            c=json.loads((cr/'config.json').read_text());zc=json.loads((zr/'config.json').read_text())
            f,relative,field=reconstruct(cr,c);zf,zrelative,zfield=reconstruct(zr,zc)
            assert np.array_equal(f,zf)
            bands={}
            for name,select in [('full',f>0),('below_10MHz',(f>0)&(f<1.e7)),('below_100MHz',(f>0)&(f<1.e8))]:
                bands[name]=error(field[:,:,select],zfield[:,:,select])
            wave=np.fft.irfft(relative*c['sample_rate_Hz'],n=c['samples'],axis=-1)
            zwave=np.fft.irfft(zrelative*c['sample_rate_Hz'],n=c['samples'],axis=-1)
            wave_error=error(wave,zwave)
            dc=float(np.linalg.norm(field[:,:,0])/max(np.linalg.norm(zfield[:,:,1:]),1.e-100))
            common=['device_tracks','cpu_tracks','track_observer_pairs','leaves','paths','direct_paths','transmitted_paths','blocked_paths','errors','out_of_window']
            # Only emission representation, its threshold, and output units may differ.
            ignored={'algorithm','moment_quantity','moment_units','regularized_moment_units','coreas_cherenkov_threshold'}
            cc={k:v for k,v in c.items() if k not in ignored};zz={k:v for k,v in zc.items() if k not in ignored}
            checks=dict(accepted_inputs=all(r['checks'].values()) and all(z['checks'].values()),
                identical_transport=r['csv_sha256']==z['csv_sha256'],
                identical_optical_and_sampling_config=cc==zz,
                identical_current_paths=all(r['radio'][k]==z['radio'][k] for k in common),
                spectrum_equivalence=all(v<1.e-6 for v in bands.values()),
                waveform_equivalence=wave_error<1.e-6,zero_frequency_cancellation=dc<1.e-12)
            rows.append(dict(backend=key[0],case=key[1],variant=key[2],checks=checks,
                spectrum_relative_l2=bands,waveform_relative_l2=wave_error,dc_relative_norm=dc,
                matched_counters=common,moment_order=c['moment_order']))
            if key[0]=='Cuda' and key[2]=='resident':
                peak=int(np.argmax(np.linalg.norm(zwave[0],axis=0)));sl=slice(max(0,peak-25),min(c['samples'],peak+26))
                time=(c['start_time_s']+np.arange(c['samples'])/c['sample_rate_Hz'])*1.e9
                fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
                for j,color in enumerate(('tab:blue','tab:orange','tab:green')):
                    axes[0].plot(time[sl],zwave[0,j,sl],color=color,label='ZHS '+('Ex','Ey','Ez')[j])
                    axes[0].plot(time[sl],wave[0,j,sl],color=color,ls='--')
                axes[0].set(xlabel='Absolute time (ns)',ylabel='E (V/m)',title=key[1]+'; dashed = CoREAS');axes[0].legend(fontsize=7)
                axes[1].semilogy(f[1:]/1.e6,np.linalg.norm(zfield[0,:,1:],axis=0),label='ZHS')
                axes[1].semilogy(f[1:]/1.e6,np.linalg.norm(field[0,:,1:],axis=0),'--',label='CoREAS')
                axes[1].set(xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)',title='Complex L2 = %.3g'%bands['full']);axes[1].legend()
                for ax in axes:ax.grid(alpha=.25)
                fig.savefig(a.output/(key[1]+'.png'),dpi=150);fig.savefig(a.output/(key[1]+'.pdf'));plt.close(fig)
                np.savez_compressed(a.output/(key[1]+'.npz'),frequency_Hz=f,coreas_spectrum=field,zhs_spectrum=zfield)
    report=dict(passed=len(rows)==16 and all(all(r['checks'].values()) for r in rows),cases=rows,
        scope='All 16 actual DEM showers, identical charged tracks and optical/sampling configuration. Absolute complex fields and signed waveforms; no fitted time, phase or amplitude. P12 tolerance is checked against the separate moment-order convergence experiment.')
    (a.output/'algorithm_shower_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
