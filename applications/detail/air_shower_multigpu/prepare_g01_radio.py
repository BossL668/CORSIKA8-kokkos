"""Create a nonempty default-ring radio smoke test, preserving older outputs."""
import json
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--case', choices=['photon', 'proton'], default='photon')
options = parser.parse_args()
work = Path(__file__).resolve().parent
cfg = json.loads((work / 'g01_smoke.json').read_text())
if '--ring' not in cfg['physics_args']:
    cfg['physics_args'] += ['--ring', '1']
# This host needs an explicit compatibility loader. Its brk heap can approach
# the grow-down main stack; prefer mmap for sizable host allocations. This is
# only a deployment setting of experimental children, never a physics change.
cfg['env']['MALLOC_MMAP_THRESHOLD_'] = '65536'
cfg['output'] = '/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/photon10GeV_four_gpu_radio_v5'
filename = 'g01_radio.json'
if options.case == 'proton':
    args = cfg['physics_args']
    args[args.index('-p') + 1] = '2212'
    args[args.index('-E') + 1] = '1000'
    cfg['frontier_max_energy_GeV'] = 31.25
    cfg['seed'] = 26092402
    cfg['output'] = '/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/proton1TeV_four_gpu_radio_v7'
    filename = 'g01_proton.json'
with (work / filename).open('x') as f:
    json.dump(cfg, f, indent=2)
