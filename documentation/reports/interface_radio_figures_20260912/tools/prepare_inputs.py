#!/usr/bin/env python3
"""Read actual saved CUDA shower inputs on PSR; no new shower or synthetic source."""
from pathlib import Path
import csv,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
case=STAGE/'interface-radio-pair-20260912/app-cuda-p1/electron_rock_resident'
config=json.loads((case/'radio/CoREAS/config.json').read_text())
rows=list(csv.DictReader((case/'terrain/tracks.csv').open()))
rock=[r for r in rows if r['medium']=='rock' and abs(int(r['pdg']))==11]
assert len(rock)==2
air=[r for r in rows if r['medium']=='air' and abs(int(r['pdg']))==11 and
  0<math.sqrt(sum((float(r[x+'1_m'])-float(r[x+'0_m']))**2 for x in 'xyz'))<.1]
tracks=rock+sorted(air,key=lambda r:abs(float(r['z0_m'])-1000))
with (DATA/'tracks.txt').open('w') as f:
 f.write(str(len(tracks))+'\n')
 for r in tracks:
  keys=['step','pdg','weight','x0_m','y0_m','z0_m','x1_m','y1_m','z1_m','t0_s','t1_s','history_id']
  f.write(' '.join(r[k] for k in keys)+' '+str(int(r['medium']=='rock'))+'\n')
with (DATA/'observers.txt').open('w') as f:
 f.write(str(len(config['observers']))+'\n')
 for o in config['observers']:f.write(o['name']+' '+' '.join(map(str,o['position_m']))+' '+str(o['region'])+'\n')
with (DATA/'media.txt').open('w') as f:
 for m in config['media']:
  f.write(' '.join(map(str,[m['index'],*m['center_m'],m['reference_radius_m'],len(m['radial_index']),len(m['radial_integration_breaks_m'])]))+'\n')
  for row in m['radial_index']:f.write(' '.join(map(str,row))+'\n')
  for x in m['radial_integration_breaks_m']:f.write(str(x)+'\n')
mesh=STAGE/'psr_inputs/terrain_enu.ply'
files=[case/'terrain/tracks.csv',case/'radio/CoREAS/config.json',mesh,STAGE/'source/corsika/modules/radio/interface/Propagation.hpp']
record=dict(case=str(case),mesh=str(mesh),track_candidates=tracks,radio_config={k:v for k,v in config.items() if k!='media'},
 input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 method='Replay actual recorded charged-track midpoint queries with the production propagation functions; no stored per-ray GPU trace exists.')
(DATA/'input_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('Prepared',len(rock),'real rock tracks,',len(air),'short charged air tracks and',len(config['observers']),'actual observers')
