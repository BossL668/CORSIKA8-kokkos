#!/usr/bin/env python3
"""Moment truncation convergence for two representations of identical current."""
import argparse,json
from pathlib import Path
from analyze_interface_coreas import error,reconstruct

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for mode in ('OPENMP','CUDA'):
        for kind in ('uniform','boundary'):
            values=[]
            for order in (8,12,16,20):
                fields=[]
                for algorithm in ('CoREAS','ZHS'):
                    root=a.root/('%s_%s_p%d_%s'%(mode,kind,order,algorithm))
                    c=json.loads((root/'config.json').read_text());fields.append(reconstruct(root,c)[2])
                values.append(dict(order=order,complex_relative_l2=error(*fields)))
            e=[v['complex_relative_l2'] for v in values]
            checks=dict(p12_accuracy=e[1]<1.e-6,p16_accuracy=e[2]<1.e-9,p20_accuracy=e[3]<1.e-11,
                convergence=e[1]<e[0]/100. and e[2]<e[1]/100. and e[3]<max(e[2],1.e-12))
            rows.append(dict(backend=mode,case=kind,checks=checks,orders=values))
    report=dict(passed=all(all(r['checks'].values()) for r in rows),cases=rows,
        scope='Only moment order changes: identical path subdivision and source tracks. CoREAS E-delta expansion and differentiated ZHS A-interval expansion converge to the same field.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
