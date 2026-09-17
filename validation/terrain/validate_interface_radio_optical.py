#!/usr/bin/env python3
"""PSR native n(h) optical-time oracle, including thin layer transitions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def breaks(m):
    if 'radial_integration_breaks_m' in m:return np.asarray(m['radial_integration_breaks_m'])
    table=np.asarray(m['radial_index']);i=np.flatnonzero(np.diff(table[:,0])<1.e-4)
    return np.unique(np.r_[table[i,0],table[i+1,0]])

def integral(a,b,m,count):
    table=np.asarray(m['radial_index']);center=np.asarray(m['center_m']);radius=m['reference_radius_m']
    length=np.linalg.norm(b-a);direction=(b-a)/length;relative=a-center
    parallel=float(relative@direction);perpendicular=float(relative@relative-parallel*parallel)
    cuts=[0.,length]
    if 0.<-parallel<length:cuts.append(-parallel)
    for height in breaks(m):
        disc=(radius+height)**2-perpendicular
        if disc>=0.:
            for s in (-parallel-np.sqrt(disc),-parallel+np.sqrt(disc)):
                if 0.<s<length:cuts.append(float(s))
    cuts=sorted(cuts);result=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if hi<=lo:continue
        sample=lo+(np.arange(count)+.5)*(hi-lo)/count
        points=a+sample[:,None]*direction;h=np.linalg.norm(points-center,axis=1)-radius
        result+=(hi-lo)*float(np.mean(np.interp(h,table[:,0],table[:,1])))
    return result/299792458.

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepare',type=Path);p.add_argument('--rays',type=Path,nargs='+');p.add_argument('--output',type=Path)
    args=p.parse_args();config=json.loads(args.config.read_text());m=config['media'][0]
    if args.prepare:
        table=m['radial_index'];b=breaks(m)
        with args.prepare.open('w') as f:
            f.write(' '.join(map(str,m['center_m']+[m['reference_radius_m'],len(table),len(b)]))+'\n')
            for h,n in table:f.write(f'{h:.17g} {n:.17g}\n')
            for h in b:f.write(f'{h:.17g}\n')
    if not args.rays:return
    args.output.mkdir(parents=True,exist_ok=True);rows=[];fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    references={}
    for path in args.rays:
        data=np.genfromtxt(path,delimiter=',',names=True);errors=[]
        for i,r in enumerate(data):
            a=np.array([r[x] for x in ('sx','sy','sz')]);b=np.array([r[x] for x in ('ox','oy','oz')]);key=tuple(np.r_[a,b])
            if key not in references:
                coarse=integral(a,b,m,32768);fine=integral(a,b,m,65536);references[key]=(fine,abs(coarse-fine)*1.e9)
            expected,uncertainty=references[key];error=(float(r['time_s'])-expected)*1.e9;errors.append(error)
            rows.append(dict(file=path.name,ray=i,time_s=float(r['time_s']),reference_s=expected,error_ns=error,reference_difference_ns=uncertainty))
        ax.plot(np.arange(len(data)),errors,'o-',label=path.stem)
    ax.set(xlabel='Vertical / oblique native atmosphere ray',ylabel='Signed optical time error (ns)',title='Layer-aware device integration vs dense independent midpoint integral')
    ax.grid(alpha=.3);ax.legend();fig.savefig(args.output/'native_optical_times.png',dpi=170);fig.savefig(args.output/'native_optical_times.pdf');plt.close(fig)
    report=dict(passed=bool(rows) and all(abs(r['error_ns'])<.001 and r['reference_difference_ns']<.0001 for r in rows),
        maximum_absolute_error_ns=max(abs(r['error_ns']) for r in rows),rows=rows,
        scope='Actual native refractive-index table; straight optical rays reaching 112799 m, including layer jumps and oblique paths.')
    (args.output/'native_optical_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
