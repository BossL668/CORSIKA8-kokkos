#!/usr/bin/env python3
"""Plot the actual resolved source failure and independent helix regressions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')

def main():
    for mode in ['OPENMP','CUDA']:
        assert (ROOT/('TESTS_'+mode+'_PASSED')).exists(),'Current backend tests must pass first'
    original=json.loads((ROOT/'failed_runs/openmp_seed946_magnetic_chord/source_contract_scan.json').read_text())
    tables={mode:np.genfromtxt(str(ROOT/('magnetic-accuracy-'+mode+'.csv')),delimiter=',',names=True)
            for mode in ['OPENMP','CUDA']}
    assert all(len(data)==384 for data in tables.values()),'Stale or incomplete magnetic fixture output'
    rows=[]
    for mode,data in tables.items():
        for angle in [.2,4e-5]:
            d=data[data['angle']==angle]
            rows.append(dict(backend=mode,deflection_parameter_rad=angle,cases=len(d),
                rejected_sources=int((d['source_accepted']==0).sum()),
                max_relative_chord_excess=float(d['chord_excess'].max()),
                max_relative_position_error=float(d['position_error'].max()),
                max_direction_error=float(d['direction_error'].max())))
    out=ROOT/'report';(out/'figures').mkdir(parents=True,exist_ok=True)
    summary=dict(actual_failed_event=original['counts'],
        maximum_actual_excess_m=max(d['excess_m'] for d in original['invalid_tracks']),helix_comparisons=rows,
        note='The magnetic-step cap changes mountain transport, with radio on or off. '
             'Local analytic error bounds do not establish full-shower step-size convergence.')
    (out/'magnetic_accuracy.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    invalid=original['invalid_tracks']
    axes[0].scatter([d['length_m'] for d in invalid],[d['excess_m']*1000 for d in invalid],s=10,alpha=.5,color='#bc583f')
    axes[0].set(xscale='log',yscale='log',xlabel='Recorded chord length (m)',ylabel='Chord length minus c dt (mm)',
        title='Actual 100 PeV air-shower failure: 766 sources')
    axes[0].text(.03,.97,'All are electrons or positrons\nResolved integration errors, not input ULPs',transform=axes[0].transAxes,va='top',fontsize=9)
    selected=[r for r in rows if r['backend']=='CUDA']
    for i,(key,label) in enumerate([('max_relative_chord_excess','Chord excess / length'),('max_relative_position_error','Position error / length')]):
        values=[r[key] for r in selected]
        axes[1].bar(np.array([0,1])+(.18 if i else -.18),values,width=.34,label=label)
    axes[1].set_yscale('log');axes[1].set_xticks([0,1]);axes[1].set_xticklabels(['Original 0.2 rad','Capped 4e-5 rad'])
    axes[1].set(ylabel='Maximum relative local error',title='Independent exact helix: both charge signs / fields / pitches')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Terrain magnetic transport refinement | shared CPU and Kokkos accuracy cap')
    for ext in ['png','pdf']:fig.savefig(out/'figures'/('magnetic_accuracy.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(summary))

if __name__=='__main__':main()
