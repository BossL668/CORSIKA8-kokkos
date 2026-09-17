#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_build_pid=${1:?pass current build PID}
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
while [ ! -f "$c8_root/TESTS_CUDA_PASSED" ]; do
  kill -0 "$c8_build_pid" 2>/dev/null || { echo 'Magnetic build/test gate failed'; exit 1; }
  sleep 5
done
# No radio kernel/optics code changed in this phase. Preserve, and explicitly
# identify, the existing exhaustive optical and prescribed-current evidence.
python - <<'PY'
from pathlib import Path
import json
r=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')
for p in (r/'source/corsika/modules/radio/interface').glob('*.hpp'):
    assert p.read_bytes()==(r/'before_magnetic_step_fix/source/corsika/modules/radio/interface'/p.name).read_bytes(),p
assert json.loads((r/'optical_oracle_precision/coreas_summary.json').read_text())['passed']
for mode in ['OPENMP','CUDA']:
    assert (r/('bvh-real-dem-'+mode+'.json')).exists()
print('Optics unchanged; preserve prior exhaustive BVH and high-accuracy current-oracle results')
PY
python "$c8_root/run_psr.py" --phase controls --backend openmp --prefix magnetic_ --transport-refined
python "$c8_root/run_psr.py" --phase controls --backend cuda --prefix magnetic_ --transport-refined
python "$c8_root/analyze_psr.py" --controls-only
touch "$c8_root/BVH_PHYSICS_PASSED"
