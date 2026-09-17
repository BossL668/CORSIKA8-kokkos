#!/usr/bin/env python3
"""PSR-only Decimal oracle: no production Snell solver or geometric helpers."""
import csv, json, pathlib
from decimal import Decimal, localcontext
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
def reference(depth):
    with localcontext() as ctx:
        ctx.prec=100
        D=Decimal.from_float
        n1=D(2.); n2=D(1.000327); d1=D(depth); d2=D(2000.)
        lateral=(Decimal(6000)**2+Decimal(3000)**2).sqrt()
        low=Decimal(0); high=lateral
        for _ in range(400):
            x=(low+high)/2; y=lateral-x
            value=n1*x/(d1*d1+x*x).sqrt()-n2*y/(d2*d2+y*y).sqrt()
            if value>0: high=x
            else: low=x
        x=(low+high)/2; y=lateral-x
        r1=(d1*d1+x*x).sqrt(); r2=(d2*d2+y*y).sqrt()
        tangent=[Decimal(6000)/lateral,Decimal(3000)/lateral]
        emit=[t*x/r1 for t in tangent]+[d1/r1]
        receive=[t*y/r2 for t in tangent]+[d2/r2]
        return dict(short_length=r1,time=(n1*r1+n2*r2)/Decimal(299792458),
                    emit=np.array([float(v) for v in emit]),receive=np.array([float(v) for v in receive]))
def main():
    if not ROOT.is_dir(): raise RuntimeError('Run on PSR only')
    rows=[]
    for version in ['baseline','fixed']:
        with (ROOT/('snell_'+version+'.csv')).open() as stream:
            for raw in csv.DictReader(stream):
                d={k:float(v) for k,v in raw.items()}; ref=reference(d['actual_depth_m'])
                reverse=bool(d['reverse']); valid=d['status']==1
                expected_emit=-ref['receive'] if reverse else ref['emit']
                expected_receive=-ref['emit'] if reverse else ref['receive']
                actual_emit=np.array([d['emit_'+k] for k in 'xyz'])
                actual_receive=np.array([d['receive_'+k] for k in 'xyz'])
                row=dict(version=version,offset_m=d['offset'],depth_m=d['depth'],reverse=reverse,valid=valid,
                         short_leg_relative_error=float(abs(Decimal.from_float(d['destination_length_m' if reverse else 'source_length_m'])/ref['short_length']-1)) if valid else None,
                         time_relative_error=float(abs(Decimal.from_float(d['time_s'])/ref['time']-1)) if valid else None,
                         direction_absolute_error=float(max(np.linalg.norm(actual_emit-expected_emit),np.linalg.norm(actual_receive-expected_receive))) if valid else None)
                rows.append(row)
    fixed=[r for r in rows if r['version']=='fixed']
    checks=dict(all_fixed_valid=all(r['valid'] for r in fixed),
                fixed_short_leg=all(r['short_leg_relative_error'] is not None and r['short_leg_relative_error']<1e-12 for r in fixed),
                fixed_direction=all(r['direction_absolute_error'] is not None and r['direction_absolute_error']<1e-12 for r in fixed),
                fixed_time=all(r['time_relative_error'] is not None and r['time_relative_error']<1e-12 for r in fixed))
    out=ROOT/'report'; (out/'figures').mkdir(parents=True,exist_ok=True)
    (out/'snell_oracle.json').write_text(json.dumps(dict(checks=checks,decimal_digits=100,rows=rows),indent=2))
    labels=['%g m / %g um / %s'%(r['offset_m'],r['depth_m']*1e6,'reverse' if r['reverse'] else 'forward') for r in fixed]
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    values=np.array([[int(r['valid']) for r in rows if r['version']==v] for v in ['baseline','fixed']])
    axes[0].imshow(values,aspect='auto',cmap=matplotlib.colors.ListedColormap(['#d86152','#43a68b']),vmin=0,vmax=1)
    axes[0].set(yticks=[0,1],yticklabels=['Before fix','After fix'],xticks=range(12),xticklabels=labels,title='Accepted Snell solutions: red = failed, green = valid')
    axes[0].tick_params(axis='x',rotation=85)
    for v,marker in [('baseline','x'),('fixed','o')]:
        selected=[r for r in rows if r['version']==v]
        ys=[max(r['short_leg_relative_error'],1e-17) if r['valid'] else np.nan for r in selected]
        axes[1].semilogy(range(12),ys,marker,ms=7,label='Before fix' if v=='baseline' else 'After fix')
    axes[1].set(xticks=range(12),xticklabels=labels,ylabel='Relative error in short optical leg',title='Independent 100-digit Decimal reference')
    axes[1].tick_params(axis='x',rotation=85);axes[1].grid(alpha=.2);axes[1].legend()
    axes[1].text(.02,.02,'Errors below 1e-17 are shown at 1e-17',transform=axes[1].transAxes,fontsize=9)
    fig.suptitle('Shallow rock-air transmission | 6 km x 3 km horizontal separation | 2 km air height')
    for ext in ['png','pdf']: fig.savefig(out/'figures'/('snell_stability.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(checks=checks,baseline_valid=int(values[0].sum()),fixed_valid=int(values[1].sum()),max_fixed_short_leg_error=max(r['short_leg_relative_error'] for r in fixed))))
    if not all(checks.values()): raise RuntimeError('Independent Snell reference failed')
if __name__=='__main__': main()
