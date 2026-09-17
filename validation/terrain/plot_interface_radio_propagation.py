#!/usr/bin/env python3
"""Plot saved analytic-ray evidence. Run on PSR; never invokes a shower."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--openmp',type=Path,required=True)
    parser.add_argument('--cuda',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows={m:np.genfromtxt(getattr(args,m),delimiter=',',names=True) for m in ('openmp','cuda')}
    assert len(rows['openmp'])==len(rows['cuda'])==2056
    a,b=rows['openmp'],rows['cuda']
    inputs=['n_source','n_destination','angle_rad','reverse','source_x','source_z','observer_x','observer_z']
    assert all(np.array_equal(a[k],b[k]) for k in inputs)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    choose=(b['n_source']==2)&(b['n_destination']==1)&(b['reverse']==0)
    r=b[choose]
    for row in r[::32]:
        axes[0,0].plot([row['source_x'],0,row['observer_x']],
                       [row['source_z'],0,row['observer_z']],lw=1)
    axes[0,0].axhline(0,color='black',lw=1)
    axes[0,0].set(title='Prescribed Snell rays: n=2 to n=1',xlabel='x (m)',ylabel='z (m)')
    axes[0,0].set_aspect('equal',adjustable='datalim')
    angle=np.rad2deg(r['angle_rad'])
    axes[0,1].plot(angle,r['t_s'],label='s amplitude')
    axes[0,1].plot(angle,r['t_p'],label='p amplitude')
    axes[0,1].set(title='Fresnel field transmission',xlabel='Source angle (deg)',ylabel='Coefficient')
    axes[0,1].legend()
    for mode,color in [('openmp','tab:blue'),('cuda','tab:orange')]:
        data=rows[mode]
        axes[0,2].semilogy(np.maximum(data['interface_error_m'],1.e-18),'.',ms=2,label=mode,color=color)
        axes[1,0].semilogy(np.maximum(abs(data['time_error_s'])/data['time_s'],1.e-18),'.',ms=2,label=mode,color=color)
        axes[1,1].semilogy(np.maximum(abs(data['snell_residual']),1.e-18),'.',ms=2,label=mode,color=color)
    axes[0,2].set(title='Interface point vs prescribed origin',xlabel='Ray',ylabel='Position error (m)')
    axes[0,2].legend()
    axes[1,0].set(title='Absolute optical delay vs analytic path',xlabel='Ray',ylabel='Relative error')
    axes[1,1].set(title='Snell residual',xlabel='Ray',ylabel='|n1 sin(theta1) - n2 sin(theta2)|')
    for key in ('time_s','spreading_per_m','transfer_s_per_m'):
        delta=abs(a[key]-b[key])/np.maximum(np.maximum(abs(a[key]),abs(b[key])),1.e-30)
        axes[1,2].semilogy(np.maximum(delta,1.e-18),'.',ms=2,label=key)
    axes[1,2].set(title='CUDA / OpenMP comparison, without fitting',xlabel='Ray',ylabel='Relative difference')
    axes[1,2].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.savefig(args.output/'propagation_validation.png',dpi=180)
    fig.savefig(args.output/'propagation_validation.pdf')
    summary={}
    for mode,r in rows.items():
        summary[mode]=dict(rays=len(r),max_interface_error_m=float(max(r['interface_error_m'])),
            max_relative_time_error=float(max(abs(r['time_error_s'])/r['time_s'])),
            max_snell_residual=float(max(abs(r['snell_residual']))))
    summary['scope']='Analytic prescribed rays plus separately logged GPU mesh assertions; not full shower/radio acceptance.'
    (args.output/'propagation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
