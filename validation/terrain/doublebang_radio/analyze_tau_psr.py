#!/usr/bin/env python3
"""Summarize the unchanged tau-module controls already run on PSR."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
def main():
    energy=np.genfromtxt(str(ROOT/'tau_energy_current.csv'),delimiter=',',names=True)
    spin=np.genfromtxt(str(ROOT/'tau_spin_current.csv'),delimiter=',',names=True)
    out=ROOT/'report';(out/'figures').mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    groups=[]
    for pdg,color in [(15,'#226b9a'),(-15,'#c66032')]:
        points=[]
        for e in np.unique(energy['energy_GeV']):
            a=energy[(energy['pdg']==pdg)&(energy['energy_GeV']==e)]
            row=dict(pdg=pdg,energy_GeV=float(e),decays=len(a),max_energy_relative_residual=float(np.max(np.abs(a['energy_relative_residual']))),max_momentum_relative_residual=float(np.max(a['momentum_relative_residual'])))
            groups.append(row);points.append(row)
        label='tau minus' if pdg==15 else 'tau plus'
        axes[0].loglog([p['energy_GeV'] for p in points],[p['max_energy_relative_residual'] for p in points],'o-',color=color,label=label+': energy')
        axes[0].loglog([p['energy_GeV'] for p in points],[p['max_momentum_relative_residual'] for p in points],'x--',color=color,label=label+': momentum')
    axes[0].set(xlabel='Parent total energy (GeV)',ylabel='Maximum relative four-momentum residual',title='%d decays, both charges and multiple directions'%len(energy))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    pull=(spin['mean_cos']-spin['expected_cos'])/spin['sem']
    for pdg,color in [(15,'#226b9a'),(-15,'#c66032')]:
        chosen=spin['pdg']==pdg
        axes[1].errorbar(np.arange(len(spin))[chosen],pull[chosen],yerr=1,fmt='o',ms=3,color=color,label='tau minus' if pdg==15 else 'tau plus')
    axes[1].axhline(0,color='black',lw=1)
    for y in [-3,3]:axes[1].axhline(y,color='gray',ls='--',lw=1)
    axes[1].set(xlabel='Charge / polarization / energy / axis configuration',ylabel='(Measured mean cos(theta) - expectation) / SEM',title='Prescribed longitudinal spin: %d pion-channel draws'%int(spin['samples'].sum()))
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('TAUOLA consistency checks | does not validate event-level CC spin transfer')
    for ext in ['png','pdf']:fig.savefig(out/'figures'/('tau_physics_checks.'+ext),dpi=180)
    plt.close(fig)
    summary=dict(groups=groups,spin_configurations=len(spin),pion_draws=int(spin['samples'].sum()),maximum_absolute_spin_pull=float(np.max(np.abs(pull))))
    (out/'tau_physics_checks.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
