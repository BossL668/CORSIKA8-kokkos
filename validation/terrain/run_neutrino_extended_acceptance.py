#!/usr/bin/env python3
"""Bounded CC/NC terrain integration and prescribed-spin controls, NO radio.

These conditional vertices do not estimate a natural neutrino interaction rate.
Uses an existing validated DEM and native auxiliary cache, never a new DEM.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','openmp','cuda'],required=True)
    p.add_argument('--tau-decay-model',choices=['pythia','tauola'],default='pythia',
                   help='Legacy prescribed-Pythia control or original CORSIKA TAUOLA assembly')
    p.add_argument('--cases',nargs='+',choices=['nu_cc','antinu_cc','nu_nc','antinu_nc'],
                   default=['nu_cc','antinu_cc','nu_nc','antinu_nc'])
    a=p.parse_args();a.output=a.output.resolve();a.binary=a.binary.resolve();a.reference=a.reference.resolve()
    spec=yaml.safe_load((a.reference/'pilot_manifest.yaml').read_text())
    job=spec['jobs'][0] # in-rock tau control point; do not relocate to force an intersection
    scene=a.reference/'original_expanded_scene.yaml'
    group=a.output/a.backend;group.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    with a.binary.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    binary_hash=digest.hexdigest()
    for name in a.cases:
        out=group/name;guard=group/(name+'_guard');work=group/(name+'_work')
        if out.exists() or guard.exists():raise RuntimeError(f'refusing overwrite {name}')
        work.mkdir(exist_ok=False)
        primary='anti_nu_tau' if name.startswith('antinu') else 'nu_tau'
        current=name.rsplit('_',1)[1]
        cmd=[str(a.binary),'--scene',str(scene),'--output',str(out),
             '--primary',primary,'--energy-GeV','1e4','--seed','909340',
             '--position-m',*map(str,job['position']),'--direction',*map(str,job['direction']),
             '--emcut-GeV','.01','--hadcut-GeV','.3','--mucut-GeV','.3',
             '--emthin','.001','--max-weight','50','--max-step-m','1',
             '--transport-window-ns','1100','--neutrino-channels','cc+nc',
             '--force-vertex-'+current,'--tau-decay-model',a.tau_decay_model,
             '--magnetic-field','igrf14','--em-backend','proposal' if a.backend=='cpu' else 'kokkos',
             '--threads','2' if a.backend=='openmp' else '1','--batch','64',
             '--device-memory-MiB','192','--track-row-limit','2000000',
             '--aux-cache',str(a.reference/'auxiliary')]
        if a.tau_decay_model=='pythia':cmd+=['--tau-minus-polarization','-1']
        (group/(name+'_command.json')).write_text(json.dumps(dict(command=cmd,binary_sha256=binary_hash,
            reference=str(a.reference),conditional=True,tau_decay_model=a.tau_decay_model,
            spin_model='prescribed_not_CC_density_matrix' if a.tau_decay_model=='pythia'
            else 'original_TAUOLA_fixed_left_helicity_not_CC_density_matrix'),indent=2)+'\n')
        wrapper=[sys.executable,str(Path(__file__).with_name('run_guarded_diagnostic.py')),
                 '--output',str(guard),'--timeout','600']
        if a.backend=='cuda':wrapper+=['--gpu','--gpu-increment-MiB','700']
        env=dict(os.environ,OMP_NUM_THREADS='2' if a.backend=='openmp' else '1',
                 OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
        print(f'RUN {a.backend} {name}',flush=True)
        subprocess.run(wrapper+['--']+cmd,cwd=work,env=env,check=True)
        summary=yaml.safe_load((out/'terrain_run.yaml').read_text())
        assert summary['complete'] and summary['radio']=='disabled'
        interactions=summary['neutrino']['interactions']
        assert interactions[0]['current']==current.upper()
        assert summary['diagnostics']['material_mismatches']==0
        assert summary['neutrino']['neutral_current_included']
        assert summary['tau']['tau_decay_backend']==a.tau_decay_model
        assert summary['tau']['prescribed_polarization_enabled']==(a.tau_decay_model=='pythia')
        # A completed diagnostic is NOT certification of the missing physics.
        assert not summary['neutrino']['full_neutrino_physics_validated']


if __name__=='__main__':main()
