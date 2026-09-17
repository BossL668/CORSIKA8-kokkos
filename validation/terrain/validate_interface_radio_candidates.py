#!/usr/bin/env python3
"""Compare the optimized real-DEM radio with accepted exhaustive-face runs on PSR."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--buffering',action='store_true',help='Compare accepted reference cases, allowing radio batch count and buffer bytes to change')
    args=p.parse_args();reference=json.loads(args.reference.read_text());candidate=json.loads(args.candidate.read_text());rows=[]
    for r in candidate:
        matches=[x for x in reference if (x['case'],x['variant'])==(r['case'],r['variant']) and all(x['checks'].values())]
        if args.buffering and not matches:continue
        old=matches[0]
        assert all(r['checks'].values()) and all(old['checks'].values())
        a=np.fromfile(Path(r['radio']['directory'])/'moments.bin',dtype=np.float64)
        b=np.fromfile(Path(old['radio']['directory'])/'moments.bin',dtype=np.float64)
        assert a.shape==b.shape
        error=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-100))
        ignored={'directory','wavefronts','device_bytes'} if args.buffering else {'directory'}
        keys=[k for k in old['radio'] if k not in ignored]
        checks=dict(same_full_tracks=r['csv_sha256']==old['csv_sha256'],
            same_radio_counters=all(r['radio'][k]==old['radio'][k] for k in keys),
            same_optical_config=(Path(r['radio']['directory'])/'config.json').read_bytes()==(Path(old['radio']['directory'])/'config.json').read_bytes(),
            same_moments=error<1.e-11)
        rows.append(dict(case=r['case'],variant=r['variant'],checks=checks,moment_relative_l2=error,
            old_radio_batches=old['radio']['wavefronts'],new_radio_batches=r['radio']['wavefronts']))
    if args.buffering:
        assert len(rows)==sum(all(x['checks'].values()) for x in reference),'Missing accepted buffer reference cases'
    report=dict(passed=bool(rows) and all(all(r['checks'].values()) for r in rows),rows=rows,
        reference=str(args.reference),candidate=str(args.candidate),
        scope=('Full real-DEM showers against accepted unbuffered CUDA runs; GPU batching preserves physical outputs and radio moments.' if args.buffering else
            'Full real-DEM showers against accepted exhaustive solving; conservative face rejection and BVH shared-face ownership preserve the radio result.'))
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
