#!/usr/bin/env python3
"""Summarize the bounded tau-spin checks; never certify missing weak physics."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();m=pd.read_csv(root/'angular_matrix.csv')
    results={}
    for name in ['unit','angular','original_regression','neutrino_regression','radio_foundation','kokkos_radio']:
        g=root/(name+'_guard');r=json.loads((g/'resources.json').read_text());log=(g/'stdout.log').read_text()
        marker=('homogeneous nR/c, air-default regression and initialization gates passed'
                if name=='kokkos_radio' else 'All tests passed')
        passed=marker in log and r['returncode']==0 and r['stop_reason'] is None
        results[name]=dict(passed=passed,wall_s=r['wall_seconds'],
            test_summary=re.findall(re.escape(marker)+r'[^\n]*',log),
            peak_RSS_MiB=max(x.get('rss_kib',0)/1024 for x in r['samples']))
    app=json.loads((root/'application/acceptance.json').read_text())
    z=(m.mean_cos-m.expected_cos)/m['sem'];maximum=float(abs(z).max())
    plt.rcParams.update({'font.family':'DejaVu Serif','axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax,pid,title in zip(axs,[15,-15],[r'$\tau^-\to\pi^-\nu_\tau$',r'$\tau^+\to\pi^+\bar\nu_\tau$']):
        for (e,axis),s in m[m.pdg==pid].groupby(['energy_GeV','axis']):
            color={10.:'#2a6c9a',1000.:'#c26c36',1.e8:'#53956b'}[e]
            label=f'{e:g} GeV, axis {axis}'
            ax.errorbar(s.P_physical,s.mean_cos,yerr=s['sem'],fmt='o' if axis==0 else 'x',
                        c=color,ms=4,capsize=2,label=label)
        x=np.linspace(-1,1,50);ax.plot(x,(1 if pid==15 else -1)*x/3,'k--',lw=1,label='analytic')
        ax.set(xlabel='Prescribed physical longitudinal polarization',title=title,ylim=(-.4,.4))
    axs[0].set_ylabel(r'$\langle\cos\theta_\pi^*\rangle$');axs[1].legend(fontsize=7,ncol=2)
    fig.suptitle('TAUOLA decay-interface validation: 60 configurations, 180,000 decays')
    fig.text(.5,.015,'3,000 decays per point; error bars: 1 standard error. Prescribed spin, not CC production or material evolution.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.93));figures=root/'figures';figures.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figures/('tau_longitudinal_angular_oracle.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    out=dict(passed=all(v['passed'] for v in results.values()) and app['passed'] and maximum<6.,
             tests=results,application=app['checks'],angular_configurations=len(m),
             angular_decays=int(m.samples.sum()),maximum_angular_mean_z=maximum,
             full_weak_physics_validated=False,event_CC_spin_implemented=False,
             material_depolarization_implemented=False,gpu_used=False)
    (root/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if not out['passed']:raise SystemExit(1)


if __name__=='__main__':main()
