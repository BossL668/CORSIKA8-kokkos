#!/usr/bin/env python3
"""PSR-only, separate processes for each FLUKA COMMON-block initialization."""
import hashlib,json,os,pathlib,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=pathlib.Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_models_20260913')
folder=ROOT/'fluka-oracle'; folder.mkdir(exist_ok=False)
binary=ROOT/'build-openmp/tests/modules/testFlukaMaterialOracle'
env=dict(os.environ,FLUPRO='/home/yuhanglu/fluka',OMP_NUM_THREADS='1',
    LD_LIBRARY_PATH='/home/yuhanglu/miniconda/envs/corsika_venv/lib:/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/targets/x86_64-linux/lib')
def run(mode):
    work=folder/mode; work.mkdir()
    with (folder/(mode+'.log')).open('x') as log:
        subprocess.run(['taskset','-c','0' if mode=='stock' else '128',str(binary),mode,str(folder/(mode+'.txt'))],
            cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(run,['stock','expanded']))
a=(folder/'stock.txt').read_bytes();b=(folder/'expanded.txt').read_bytes()
assert a==b and len(a.splitlines())>120
lines=a.decode().splitlines()
result=dict(exact_output_identity=True,cross_section_points=sum(l.startswith('XS ') for l in lines),
            same_seed_events=sum(l.startswith('EVENT ') for l in lines),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
assert result['cross_section_points']==120 and result['same_seed_events']==3
(ROOT/'fluka-validation.json').write_text(json.dumps(result,indent=2))
(ROOT/'FLUKA_TARGETS_PASSED').touch()
print(json.dumps(result,indent=2))
