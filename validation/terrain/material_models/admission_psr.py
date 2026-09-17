#!/usr/bin/env python3
"""Exercise material_file and conflict rejection through the real scene loader."""
import copy,json,pathlib,subprocess
import yaml
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
case=ROOT/'admission'; case.mkdir(exist_ok=True)
base=yaml.safe_load((ROOT/'silica_off.yaml').read_text())
binary=ROOT/'build-openmp/applications/c8_terrain_environment'
results=[]
for label in ['silica','limestone','granite','custom_example']:
    scene=copy.deepcopy(base)
    del scene['geometry']['material']
    scene['geometry']['material_file']=str(ROOT/'source/configs/mountain/materials'/(label+'.yaml'))
    path=case/(label+'.yaml'); path.write_text(yaml.safe_dump(scene))
    output=case/(label+'-admitted.yaml')
    command=['taskset','-c','0-255',str(binary),'--config',str(path),'--output',str(output)]
    with (case/(label+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    result=yaml.safe_load(output.read_text()); assert result['complete']
    assert not result['legacy_material_overrides']
    results.append(dict(case=label,complete=True,material=result['resolved_material']))
bad=copy.deepcopy(base); bad['geometry']['rock_refractive_index']=2.
path=case/'conflict.yaml'; path.write_text(yaml.safe_dump(bad))
proc=subprocess.run([str(binary),'--config',str(path),'--output',str(case/'rejected.yaml')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
assert proc.returncode!=0 and 'conflicts' in proc.stdout
assert not (case/'rejected.yaml').exists()
results.append(dict(case='legacy_conflict',rejected=True,message=proc.stdout))
(ROOT/'admission.json').write_text(json.dumps(results,indent=2))
print('4 material-file admissions and conflicting legacy parameters checked')
