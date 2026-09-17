#!/usr/bin/env python3
"""Prepare s/p probe input from the existing real-ray selection."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];D=R/'data'
ray=json.loads((D/'selected_ray.json').read_text())['rock_to_offset']
provenance=json.loads((D/'input_provenance.json').read_text())
track=next(t for t in provenance['track_candidates'] if t['step']==ray['step'])
values=[]
for p in ['e','r','n']:values += [ray[p+a] for a in 'xyz']
values += [ray['n1'],ray['n2'],ray['spreading_per_m']]
for endpoint in ['0','1']:values += [track[a+endpoint+'_m'] for a in 'xyz']
values += [track['t0_s'],track['t1_s'],str(-1 if int(track['pdg'])==11 else 1)]
(D/'sp_input.txt').write_text(' '.join(values)+'\n')

