#!/usr/bin/env python3
"""Independent USStdBK density/Gladstone-Dale normalization check on PSR."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AtmosphereId::USStdBK, corsika/media/CORSIKA7Atmospheres.hpp.
BOUNDARY=np.array([7000.,11400.,37000.,100000.,112800.])
OFFSET=np.array([1183.6071,1143.0425,1322.9748,655.67307,1.])  # g/cm2
SCALE=np.array([954248.34,800005.34,629568.93,737521.77,1.e9]) # cm

def check(config):
    m=config['media'][0];table=np.asarray(m['radial_index']);height=table[:,0];n=table[:,1]
    layer=np.minimum(np.searchsorted(BOUNDARY,height,side='right'),4)
    density=OFFSET[layer]/SCALE[layer]*np.where(layer==4,1.,np.exp(-height*100./SCALE[layer]))
    expected=1.+(m['index']-1.)*density/(OFFSET[0]/SCALE[0])
    mask=np.min(np.abs(height[:,None]-BOUNDARY),axis=1)>5.e-6
    error=float(np.max(np.abs(n[mask]-expected[mask])))
    dilute=(height>100001.)&(height<112799.)
    ratio=float(np.max(n[dilute]-1.)/(m['index']-1.))
    return dict(passed=error<5.e-13 and ratio<1.e-4,maximum_index_error=error,
        upper_to_sea_level_refractivity_ratio=ratio,points_checked=int(mask.sum())),height,n,expected,mask

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text());report,h,n,expected,mask=check(config)
    m=config['media'][0];layer=np.minimum(np.searchsorted(BOUNDARY,h,side='right'),4)
    legacy=1.+(m['index']-1.)*np.where(layer==4,1.,np.exp(-h*100./SCALE[layer]))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].semilogy(h[mask]/1000.,legacy[mask]-1.,'--',label='Separate per-layer normalization (legacy property)')
    axes[0].semilogy(h[mask]/1000.,n[mask]-1.,label='One sea-level density reference (new radio)')
    axes[0].set(xlabel='Altitude ASL (km)',ylabel='Refractivity n - 1',title='USStdBK + common-reference Gladstone-Dale');axes[0].legend(fontsize=8)
    axes[1].plot(h[mask]/1000.,n[mask]-expected[mask]);axes[1].set(xlabel='Altitude ASL (km)',ylabel='Index residual',title='Exported table vs independent density formula')
    for ax in axes:ax.grid(alpha=.3)
    fig.savefig(args.output/'native_refractivity.png',dpi=170);fig.savefig(args.output/'native_refractivity.pdf');plt.close(fig)
    report['scope']='Actual native density parameters with one sea-level Gladstone-Dale reference; original air modules unchanged.'
    (args.output/'atmosphere_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
