#!/usr/bin/env bash
# Isolated snapshot only. Never install or touch a running campaign binary.
set -euo pipefail
test "$(hostname)" = psrpku2025
export C8_ADAPTIVE_STAGE=${C8_ADAPTIVE_STAGE:-/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v11-host-profile-shards-20260912}
bash "$C8_ADAPTIVE_STAGE/source/validation/accelerator/build_adaptive_v5_psr.sh"
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
set +u
conda activate corsika_venv
set -u
export LD_LIBRARY_PATH=/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_cooperative130_100PeV_20260911/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cmake --build "$C8_ADAPTIVE_STAGE/build" --parallel 128 --target testKokkosHostProfileShards
ctest --test-dir "$C8_ADAPTIVE_STAGE/build" --output-on-failure -R '^testKokkosHostProfileShards$'
# CPU-only execution-space initialization even in a combined package.
OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=130 \
  taskset -c 382-511 "$C8_ADAPTIVE_STAGE/build/tests/accelerator/testKokkosHostProfileShards" 130
