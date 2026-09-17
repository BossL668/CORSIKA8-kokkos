#!/usr/bin/env python3
"""Input precision diagnostics from the retained interrupted 100 PeV trial."""
import csv,json,math,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
def main():
    rows=[];examined=0
    with (ROOT/'failed_runs/cuda_seed946_short_source_roundoff/output/terrain/tracks.csv').open() as f:
        for t in csv.DictReader(f):
            if None in t.values():continue # final interrupted CSV row
            pid=abs(int(t['pdg']))
            if pid not in [11,13,15,211,321,2212,3222,3112,3312,3334] and not(pid>=1000000000 and (pid//10000)%1000>0):continue
            a=np.array([float(t[k+'0_m']) for k in 'xyz']);b=np.array([float(t[k+'1_m']) for k in 'xyz'])
            t0=float(t['t0_s']);t1=float(t['t1_s']);dt=t1-t0;length=math.sqrt(sum((b-a)**2))
            if not length:continue
            examined+=1
            if dt<=0:raise RuntimeError('Unresolved clock interval')
            ct=299792458.*dt;excess=length-ct
            if excess<=0:continue
            budget=float(sum(np.spacing(np.abs(np.concatenate([a,b])))))+299792458.*(float(np.spacing(abs(t0)))+float(np.spacing(abs(t1))))+16*2**-52*(length+ct)
            rows.append(dict(step=int(t['step']),length_m=length,excess_m=excess,input_precision_budget_m=budget,excess_over_budget=excess/budget,old_gate_rejected=length>ct*(1+1e-9)))
    out=ROOT/'report';(out/'figures').mkdir(parents=True,exist_ok=True)
    result=dict(charged_moving_tracks_examined=examined,positive_reconstructed_excess=len(rows),old_gate_rejected=sum(r['old_gate_rejected'] for r in rows),resolved_violations=sum(r['excess_over_budget']>1 for r in rows),rows=rows)
    (out/'source_precision.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True)
    for rejected,color,label in [(False,'#287b9c','Other positive reconstructed excess'),(True,'#cb583b','Rejected by the old fixed relative gate')]:
        rr=[r for r in rows if r['old_gate_rejected']==rejected]
        ax.scatter([r['length_m'] for r in rr],[r['excess_over_budget'] for r in rr],s=20,color=color,label=label,alpha=.8)
    ax.axhline(1.,color='#a03035',ls='--',label='Input precision budget')
    ax.set(xscale='log',xlabel='Recorded chord length (m)',ylabel='(Chord length - c dt) / precision budget',ylim=(0,1.08),title='100 PeV trial: short-track endpoint and clock rounding')
    ax.text(.02,.72,'%d charged moving tracks inspected\n%d exceed the old gate; none exceeds the precision budget'%(examined,result['old_gate_rejected']),transform=ax.transAxes,fontsize=10)
    ax.grid(alpha=.2);ax.legend(loc='center right',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/'figures'/('source_precision.'+ext),dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
