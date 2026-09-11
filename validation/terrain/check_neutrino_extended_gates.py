#!/usr/bin/env python3
"""Exercise rejection paths before any shower output/calculator initialization."""
import argparse
import json
from pathlib import Path
import subprocess
import yaml


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=yaml.safe_load((a.reference/'pilot_manifest.yaml').read_text());job=spec['jobs'][0]
    cases=[('both_forced',['--force-vertex-cc','--force-vertex-nc'],'at most one'),
           ('disabled_cc',['--neutrino-channels','nc','--force-vertex-cc'],'disabled'),
           ('disabled_nc',['--neutrino-channels','cc','--force-vertex-nc'],'disabled'),
           ('invalid_p',['--tau-minus-polarization','1.2'],'Range'),
           ('nan_p',['--tau-minus-polarization','nan'],'polarization'),
           ('tauola_with_pythia_spin',['--tau-decay-model','tauola','--tau-minus-polarization','-1'],'requires'),
           ('pythia_with_tauola_spin',['--tau-decay-model','pythia','--tauola-helicity','right'],'requires'),
           ('below_fit',[],'within CTW2011'),
           ('non_nu_forced',['--force-vertex-nc'],'requires a neutrino')]
    rows=[]
    for name,extra,error in cases:
        out=a.output/(name+'_must_not_exist')
        command=[str(a.binary.resolve()),'--scene',str((a.reference/'original_expanded_scene.yaml').resolve()),
                 '--output',str(out),'--primary','photon' if name=='non_nu_forced' else 'nu_tau',
                 '--energy-GeV','9999' if name=='below_fit' else '1e4',
                 '--position-m',*map(str,job['position']),'--direction',*map(str,job['direction'])]+extra
        # Two geometry-dependent gates follow the full DEM admission, which can
        # exceed 30 s on a cold WSL mounted drive. They still precede calculators.
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        (a.output/(name+'.log')).write_text(result.stdout)
        passed=result.returncode!=0 and not out.exists() and error.lower() in result.stdout.lower()
        rows.append(dict(name=name,command=command,returncode=result.returncode,output_created=out.exists(),passed=passed))
    (a.output/'gates.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'{sum(r["passed"] for r in rows)}/{len(rows)} startup gates passed')
    if not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
