#!/usr/bin/env python3
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'
rows=list(csv.DictReader((D/'beta5_real_paths.csv').open()))
rock=next(r for r in rows if r['region']=='1' and r['step']=='1' and r['observer']=='diagnostic_offset')
air=min((r for r in rows if r['region']=='0' and r['observer']=='E01'),key=lambda r:abs(float(r['sz'])-1000))
out={'air_to_E01':air,'rock_to_offset':rock}
with (D/'selected_ray.txt').open('w') as f:
 f.write('2\n')
 for name,r in out.items():
  values=[name,r['region']]
  for pre in ['s','o','p','n','e']:values.extend(r[pre+axis] for axis in 'xyz')
  values.extend([r['n1'],r['n2']]);f.write(' '.join(values)+'\n')
(D/'selected_ray.json').write_text(json.dumps(out,indent=2)+'\n')
with (D/'path_check_inputs.txt').open('w') as f:
 f.write('2\n')
 for name,r in out.items():
  f.write(' '.join([name,r['region'],r['face']]+[r[p+a] for p in ['s','o'] for a in 'xyz'])+'\n')
print(json.dumps({k:{a:r[a] for a in ['step','observer','sz','face','flight_s','leaf_length_m']} for k,r in out.items()},indent=2))
