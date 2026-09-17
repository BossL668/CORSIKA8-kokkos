#!/usr/bin/env bash
set -euo pipefail
campaign_root=$1
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
export FLUPRO=/home/yuhanglu/fluka
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads
export OMP_DISPLAY_ENV=TRUE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset KOKKOS_TOOLS_LIBS C8_INTERFACE_TRACE_FILE
export PYTHONUNBUFFERED=1
exec python "$campaign_root/code/timing_psr.py" --root "$campaign_root"
