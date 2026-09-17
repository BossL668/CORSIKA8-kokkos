#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_validation_pid=${1:?pass the running validation driver PID}
c8_production_backend=${2:-openmp}
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
while [ ! -f "$c8_root/BVH_PHYSICS_PASSED" ]; do
  kill -0 "$c8_validation_pid" 2>/dev/null || { echo "Validation stopped before all physics gates"; exit 1; }
  sleep 5
done
python "$c8_root/analyze_psr.py" --controls-only
python "$c8_root/run_psr.py" --phase production --backend "$c8_production_backend" --seeds 946 3605 158
python "$c8_root/analyze_psr.py"
python "$c8_root/analyze_topology_psr.py"
touch "$c8_root/HIGH_ENERGY_COMPLETE"
