#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
while [ ! -f "$c8_root/TESTS_CUDA_PASSED" ]; do
  kill -0 551811 2>/dev/null || { echo "Build/test driver stopped before CUDA gates"; exit 1; }
  sleep 5
done
python "$c8_root/run_psr.py" --phase controls --backend openmp
python "$c8_root/run_psr.py" --phase controls --backend cuda

