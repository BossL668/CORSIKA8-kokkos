#!/usr/bin/env python3
"""Identify every CSV source violating the current declared precision contract."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
root=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
folder=root/'runs'/a.run
counts=dict(rows=0,charged_moving=0,positive_excess=0,invalid=0,zero_duration=0)
bad=[];maximum=0.
cols=['step','pdg','weight','medium','x0_m','y0_m','z0_m','x1_m','y1_m','z1_m','t0_s','t1_s','E0_GeV','E1_GeV','history_id','parent_history_id']
for data in pd.read_csv(folder/'output/terrain/tracks.csv',usecols=cols,chunksize=200000,float_precision='round_trip'):
    counts['rows']+=len(data)
    pid=data.pdg.abs();chosen=pid.isin([11,13,15,211,321,2212,3222,3112,3312,3334])|((pid>=1000000000)&((pid//10000)%1000>0))
    data=data[chosen & (data.weight>0)]
    xyz=data[['x0_m','y0_m','z0_m','x1_m','y1_m','z1_m']].to_numpy()
    dt=(data.t1_s-data.t0_s).to_numpy();length=np.linalg.norm(xyz[:,3:]-xyz[:,:3],axis=1)
    moving=length>0;counts['charged_moving']+=int(moving.sum());counts['zero_duration']+=int((moving&(dt<=0)).sum())
    ct=299792458.*dt;excess=length-ct
    ulp=np.spacing(np.abs(xyz)).sum(axis=1)+299792458.*(np.spacing(data.t0_s.abs().to_numpy())+np.spacing(data.t1_s.abs().to_numpy()))+16*2**-52*(length+ct)
    declared=ct*1e-9;budget=np.maximum(ulp,declared)
    counts['positive_excess']+=int((moving&(excess>0)).sum())
    ratio=np.divide(excess,budget,out=np.zeros_like(excess),where=budget>0)
    if len(ratio):maximum=max(maximum,float(ratio.max()))
    invalid=moving&((dt<=0)|(excess>budget)|~np.isfinite(excess))
    counts['invalid']+=int(invalid.sum())
    for i in np.flatnonzero(invalid):
        r=data.iloc[i].to_dict();r.update(length_m=float(length[i]),ct_m=float(ct[i]),excess_m=float(excess[i]),ulp_budget_m=float(ulp[i]),declared_budget_m=float(declared[i]),excess_over_budget=float(ratio[i]))
        bad.append(r)
    print(json.dumps(dict(progress=counts,maximum_excess_over_budget=maximum)),flush=True)
result=dict(counts=counts,maximum_excess_over_budget=maximum,invalid_tracks=bad)
(folder/'source_contract_scan.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='invalid_tracks'}))
