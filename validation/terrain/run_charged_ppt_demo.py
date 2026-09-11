#!/usr/bin/env python3
"""OpenMP-only charged-track illustrations; leave air production unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from run_nutau_ppt_demo import sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();ref=a.reference.resolve();binary=a.binary.resolve()
    if 'CORSIKA_KOKKOS_BACKEND:STRING=OPENMP\n' not in (binary.parents[1]/'CMakeCache.txt').read_text():
        raise RuntimeError('Only OpenMP-only binary allowed')
    spec=yaml.safe_load((ref/'pilot_manifest.yaml').read_text())
    scene=ref/'original_expanded_scene.yaml';s=yaml.safe_load(scene.read_text())
    if sha256(Path(s['geometry']['mesh_path'])) != s['provenance']['mesh_sha256']:
        raise RuntimeError('DEM hash mismatch')
    root.mkdir(parents=True,exist_ok=False);(root/'runs').mkdir()
    through=next(j for j in spec['jobs'] if j['role']=='unselected_neutrino')
    air=list(through['position']);air[2]+=100. # Explicit elevated-air control, not terrain displacement.
    jobs=[dict(name='mu100_crossing',primary='mu_minus',energy_GeV=100.,field='igrf14',position=through['position']),
          dict(name='electron_air_B',primary='electron',energy_GeV=1.,field='igrf14',position=air),
          dict(name='positron_air_B',primary='positron',energy_GeV=1.,field='igrf14',position=air),
          dict(name='electron_air_zero',primary='electron',energy_GeV=1.,field='none',position=air)]
    manifest=dict(jobs=jobs,seed=909360,binary=str(binary),binary_sha256=sha256(binary),
                  scene=str(scene),mesh_sha256=s['provenance']['mesh_sha256'],direction=through['direction'],
                  chord=spec['chord'],threads=4,backend='OpenMP',gpu_used=False,radio=False)
    (root/'manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    for j in jobs:
        work=root/'runs'/(j['name']+'_work');work.mkdir()
        cmd=[str(binary),'--scene',str(scene),'--output',str(root/'runs'/j['name']),
             '--primary',j['primary'],'--energy-GeV',str(j['energy_GeV']),'--seed','909360',
             '--position-m',*map(str,j['position']),'--direction',*map(str,through['direction']),
             '--emcut-GeV','.01','--hadcut-GeV','.3','--mucut-GeV','.3',
             '--emthin','0','--max-weight','1e30','--max-step-m','1',
             '--transport-window-ns','1100','--magnetic-field',j['field'],'--magnetic-year','2027',
             '--em-backend','kokkos','--threads','4','--batch','64','--device-memory-MiB','192',
             '--track-row-limit','2000000','--aux-cache',str(ref/'auxiliary')]
        (root/'runs'/(j['name']+'_command.json')).write_text(json.dumps(dict(command=cmd),indent=2))
        print('RUN '+j['name'],flush=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_guarded_diagnostic.py')),
                        '--output',str(root/'runs'/(j['name']+'_guard')),'--timeout','600','--']+cmd,
                       check=True,cwd=work,env=dict(os.environ,OMP_NUM_THREADS='4',OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1'))
        result=yaml.safe_load((root/'runs'/j['name']/'terrain_run.yaml').read_text())
        assert result['complete'] and result['accelerator']['execution_space']=='OpenMP'
        assert result['diagnostics']['material_mismatches']==0


if __name__=='__main__':main()
