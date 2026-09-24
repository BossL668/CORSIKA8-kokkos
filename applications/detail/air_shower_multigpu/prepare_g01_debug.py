"""Generate isolated follow-up configurations; never overwrite previous outputs."""
import json
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--four', action='store_true')
parser.add_argument('--mapping', action='store_true')
options = parser.parse_args()
work = Path(__file__).resolve().parent
cfg = json.loads((work / 'g01_smoke.json').read_text())
cfg['worker_debugger'] = ['/usr/bin/gdb', '--batch', '-ex', 'set pagination off',
                  '-ex', 'run', '-ex', 'thread apply all bt',
                  '-ex', 'info proc mappings', '-ex', 'info registers',
                  '-ex', 'x/6i $pc', '-ex', 'x/8gx $rsp', '--args']
if '--ring' not in cfg['physics_args']:
    cfg['physics_args'] += ['--ring', '1']
cfg['devices'] = [0, 1, 2, 3] if options.four else [1]
cfg['output'] = '/data/yuhanglu/CorsikaData/beta5_multigpu_20260924/' + (
    'photon10GeV_debug_four_gpu_v6' if options.mapping else
    'photon10GeV_debug_four_gpu_v4' if options.four else 'photon10GeV_debug_one_gpu_v3')
with (work / ('g01_debug_mapping.json' if options.mapping else
              'g01_debug_four.json' if options.four else 'g01_debug.json')).open('x') as f:
    json.dump(cfg, f, indent=2)
