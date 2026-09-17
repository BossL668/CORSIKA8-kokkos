#!/usr/bin/env python3
"""Build and verify a narrow radio-finalization overlay on PSR only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

assert socket.gethostname() == 'psrpku2025'
BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
ROOT = BASE/'beta5_radio_air_alignment_20260916'
OLD = BASE/'beta5_doublebang_energy_ladder_1PeV_20260915'
BATCH = BASE/'beta5_interface_batch_performance_20260915'
MATERIAL = BASE/'beta5_material_models_20260913'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1',
                  C8_INTERFACE_TEST_THREADS='2', OMP_PROC_BIND='spread', OMP_PLACES='threads',
                  LD_LIBRARY_PATH='/home/yuhanglu/miniconda/envs/corsika_venv/lib:/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/targets/x86_64-linux/lib',
                  FLUPRO='/home/yuhanglu/fluka')
commands = []

def verify():
    import numpy as np
    files = [p.relative_to(ROOT/'before') for p in (ROOT/'before').rglob('*') if p.is_file()]
    assert files
    errors=[]
    for p in files:
        before=ROOT/'before'/p;after=ROOT/'after'/p
        if p.suffix=='.json':
            assert read(before)==read(after);continue
        if p.suffix=='.bin':
            x=np.fromfile(before,dtype='float64');y=np.fromfile(after,dtype='float64')
        else:
            x=np.loadtxt(before,delimiter=',',skiprows=1);y=np.loadtxt(after,delimiter=',',skiprows=1)
            assert np.array_equal(x[:,:2],y[:,:2]);x=x[:,2:];y=y[:,2:]
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        norm=np.linalg.norm(x)
        error=float(np.linalg.norm(y-x)/norm) if norm else 0.
        assert error<1e-11 and (norm or np.array_equal(x,y)),(str(p),error)
        errors.append(dict(file=str(p),relative_l2=error))
    proof=dict(passed=True, comparison_files=len(files), maximum_relative_l2=max(r['relative_l2'] for r in errors),
        exact_comparison_files=sum(sha(ROOT/'before'/p)==sha(ROOT/'after'/p) for p in files),
        fixtures=['uniform','transmitted plane'],buffered_capacities=[7,8192,16384],
        checks='All CoREAS/ZHS and regularized moments, full rendered fields/spectra, counters, CPU/device interleaving and tails; finalization guards. Tolerance accounts for OpenMP atomic summation order.',
        sample_rate_Hz=1e9,old_binary_sha256=sha(OLD/'build-openmp/c8_terrain_cascade'),
        new_binary_sha256=sha(ROOT/'build/c8_terrain_cascade'),comparisons=errors)
    (ROOT/'acceptance.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('PASS',len(files),'files; maximum relative L2',proof['maximum_relative_l2'],flush=True)

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(argv, cwd, label):
    commands.append(dict(argv=list(map(str, argv)), cwd=str(cwd), label=label))
    (ROOT/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    with (ROOT/(label+'.log')).open('w') as log:
        subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(label, flush=True)

if '--verify-only' in sys.argv:
    verify()
    sys.exit(0)

build = ROOT/'build'
build.mkdir(exist_ok=True)
overlay = ROOT/'source'
header = Path('corsika/modules/radio/interface/KokkosAccumulator.hpp')
original = BATCH/'source'/header
if not original.exists(): original = MATERIAL/'source'/header
# Keep the exact before/after header and known production application object.
shutil.copy2(original, ROOT/'KokkosAccumulator.before.hpp')
assert sha(OLD/'build-openmp/c8_terrain_cascade') == '6e4d4a0da37bf434a80db782804594dd80443296052746bbce2a48273db14b1d'
assert sha(OLD/'build-openmp/libCORSIKA8InterfaceEm.a') == sha(BATCH/'build-openmp/libCORSIKA8InterfaceEm.a')
old_commands = read(BATCH/'build-openmp/commands_1789433695964987792.json')
entry = next(x for x in old_commands if x['label'] == 'CORSIKA8InterfaceEm')
cmd = entry['argv'].copy()
cmd.insert(1, '-I'+str(overlay))
cmd[cmd.index('-o')+1] = str(build/'InterfaceEm.o')
run(cmd, entry['cwd'], 'compile-radio-overlay')
run(['/usr/bin/ar', 'rcs', str(build/'libCORSIKA8InterfaceEm.a'), str(build/'InterfaceEm.o')], build, 'archive-radio-overlay')
entry = next(x for x in read(OLD/'build-openmp/commands.json') if x['label'] == 'link-application')
cmd = entry['argv'].copy()
cmd = [str(build/'libCORSIKA8InterfaceEm.a') if Path(x).name == 'libCORSIKA8InterfaceEm.a' else x for x in cmd]
cmd[cmd.index('-o')+1] = str(build/'c8_terrain_cascade')
run(cmd, entry['cwd'], 'link-radio-overlay')

# Compare the actual accumulator and public renderer before/after consumption.
checks = read(BATCH/'checks/commands.json')
compile_entry = next(x for x in checks if x['label'] == 'BufferedRadioChecks-compile')
link_entry = next(x for x in checks if x['label'] == 'BufferedRadioChecks-link')
fixture = (BATCH/'source/validation/terrain/batch_performance/BufferedRadioChecks.cpp').read_text()
fixture = fixture.replace('settings.set_num_threads(256)', 'settings.set_num_threads(2)')
fixture = fixture.replace('int main(){try{', 'int main(int argc,char** argv){try{\n  require(argc==2,"output directory required");\n  std::string directory=argv[1];')
needle = 'same(result.coreas,reference.coreas);same(result.zhs,reference.zhs);'
assert fixture.count(needle) == 1
fixture = fixture.replace(needle, needle+'\n      auto tag=directory+"/"+(transmitted?"plane":"uniform")+std::to_string(capacity);\n      ri::writeResult(result.coreas,tag+"_CoREAS");ri::writeResult(result.zhs,tag+"_ZHS");')
(ROOT/'BufferedRadioChecks.cpp').write_text(fixture)
for variant in ['before', 'after']:
    cmd = compile_entry['argv'].copy()
    if variant == 'after': cmd.insert(1, '-I'+str(overlay))
    cmd[cmd.index('-c')+1] = str(ROOT/'BufferedRadioChecks.cpp')
    cmd[cmd.index('-o')+1] = str(build/(variant+'.o'))
    run(cmd, compile_entry['cwd'], 'compile-check-'+variant)
    cmd = link_entry['argv'].copy()
    cmd = [str(build/(variant+'.o')) if x.endswith('BufferedRadioChecks.o') else x for x in cmd]
    cmd[cmd.index('-o')+1] = str(build/('check-'+variant))
    run(cmd, link_entry['cwd'], 'link-check-'+variant)
    out = ROOT/variant; out.mkdir(exist_ok=False)
    run([str(build/('check-'+variant)), str(out)], ROOT, 'run-check-'+variant)
verify()
