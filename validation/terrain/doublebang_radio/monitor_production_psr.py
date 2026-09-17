#!/usr/bin/env python3
"""Read bounded progress records without scanning the live event files."""
import json
from pathlib import Path

ROOT = Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')

def tail(path, size):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-size))
        return stream.read().decode(errors='replace').splitlines()

rows = []
for folder in sorted((ROOT / 'runs').glob('*_seed*')):
    row = dict(run=folder.name)
    for name in ['status', 'checks']:
        file = folder / (name + '.json')
        if file.exists():
            data = json.loads(file.read_text())
            row[name] = data.get('checks', {k:data.get(k) for k in ['complete', 'returncode', 'error', 'wall_s']})
    file = folder / 'output/terrain/tracks.csv'
    if file.exists():
        lines = tail(file, 2500)
        if len(lines) > 1:
            row['last_complete_step_lower_bound'] = lines[-2].split(',')[0]
    file = folder / 'resources.jsonl'
    if file.exists():
        row['resources'] = json.loads(tail(file, 1000)[-1])
    rows.append(row)
driver=ROOT/'production_magnetic_driver.log'
if not driver.exists():driver=ROOT/'production_precision_driver.log'
print(json.dumps(dict(runs=rows, complete=(ROOT / 'HIGH_ENERGY_COMPLETE').exists(),
    driver_tail='\n'.join(tail(driver, 1500)[-4:]))))
