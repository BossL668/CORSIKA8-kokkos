#!/usr/bin/env python3
"""Record production/public-library provenance and collect supporting PSR evidence."""
from pathlib import Path
import json,hashlib,shutil,platform,tarfile
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent;DATA=ROOT/'data'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
c4=STAGE/'interface-coreas-20260912';pair=STAGE/'interface-radio-pair-20260912'
files=['corsika/modules/radio/interface/'+name for name in ['Types.hpp','Propagation.hpp','KokkosPropagation.hpp',
       'ZhsIntervals.hpp','CoreasEndpoints.hpp','KokkosAccumulator.hpp']]
files+=['src/radio/interface/Output.cpp','src/transport/InterfaceEmSession.cpp',
        'applications/detail/mountain/TerrainRadioConfig.hpp']
current={name:digest(STAGE/'source'/name) for name in files}
old=json.loads((c4/'integration-final.json').read_text())
core=['corsika/modules/radio/interface/CoreasEndpoints.hpp','corsika/modules/radio/interface/ZhsIntervals.hpp','src/radio/interface/Output.cpp']
same={name:old[name]==current[name] for name in core};assert all(same.values())
pair_source=json.loads((pair/'source-psr-final.json').read_text())
same_pair={name:pair_source[name]==current[name] for name in files if name in pair_source};assert all(same_pair.values())
air=json.loads((pair/'protected-air-after.json').read_text())
assert all(digest(STAGE/'source'/name)==expected for name,expected in air.items())
rp=ROOT/'external/RadioPropa-544a2d6c4e284e3d724cb741dc481245a0f633d7'
archive=ROOT/'external/radio-report-public-radiopropa.tar.gz'
public_files_checked=0
with tarfile.open(archive,'r:gz') as public_tar:
 for member in public_tar.getmembers():
  if not member.isfile():continue
  relative=Path(member.name).relative_to(rp.name)
  assert hashlib.sha256(public_tar.extractfile(member).read()).hexdigest()==digest(rp/relative),str(relative)
  public_files_checked+=1
assert public_files_checked>0
provenance=dict(passed=True,host=platform.node(),source_directory=str(STAGE/'source'),
    source_sha256=current,unchanged_c4_physics=same,unchanged_pair_source=same_pair,
    protected_air_files_unchanged=len(air),
    radiopropa=dict(commit='544a2d6c4e284e3d724cb741dc481245a0f633d7',
      archive_url='https://codeload.github.com/nu-radio/RadioPropa/tar.gz/544a2d6c4e284e3d724cb741dc481245a0f633d7',
      archive_sha256=digest(archive),source_modified=False,archive_source_files_verified=public_files_checked,
      discontinuity_sha256=digest(rp/'radiopropa/src/module/Discontinuity.cpp'),
      library_sha256=digest(ROOT/'external/build-radiopropa/libradiopropa.so')),
    probes={name:digest(ROOT/'tools'/name) for name in ['beta5_optics_probe','radiopropa_plane_probe','radiopropa_gradient_probe']},
    scope='Read-only optics probes and report generation. No production physics or air modules changed.')
(DATA/'source_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
history={
    'c4_emission_oracle.json':c4/'oracle-c4/coreas_summary.json',
    'c4_moment_convergence.json':c4/'precision-c4/summary.json',
    'c4_native_optical.json':c4/'native-optical-c4/native_optical_summary.json',
    'pair_acceptance_summary.json':pair/'acceptance-summary.json',
    'pair_residency.json':pair/'residency.json',
    'pair_openmp_acceptance.json':pair/'app-openmp-p1/acceptance.json',
    'pair_cuda_acceptance.json':pair/'app-cuda-p1/acceptance.json',
    'scene_electron.yaml':pair/'app-cuda-p1/scene_cli.yaml',
    'scene_neutrino.yaml':pair/'app-cuda-p1/scene_nue.yaml'}
for name,path in history.items():shutil.copy2(path,DATA/name)
# Improve the descriptive parameter index without recomputing any numerical result.
r=json.loads((DATA/'report_metrics.json').read_text())
for row in r['showers']:
 command=row['command'];parameters={}
 for flag in ['--primary','--energy-GeV','--direction','--position-m','--seed','--magnetic-field','--emthin']:
  if flag in command:
   first=command.index(flag)+1
   parameters[flag]=command[first:first+3] if flag in ['--direction','--position-m'] else command[first]
 parameters['--force-vertex-cc']='--force-vertex-cc' in command;row['parameters']=parameters
(DATA/'report_metrics.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS source provenance; c4 physics / pair implementation unchanged; 244 protected air files unchanged')
