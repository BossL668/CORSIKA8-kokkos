#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
c8_deps=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-t400-cooperative-20260911/deps
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
mkdir -p "$c8_root/optical_fixtures" "$c8_root/paired_fixtures"
for c8_mode in OPENMP CUDA; do
  cmake -S "$c8_root/source/validation/terrain/coreas_consumer" -B "$c8_root/consumer-$c8_mode" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/g++" -DCMAKE_CUDA_COMPILER="$c8_cuda/bin/nvcc" \
    -DCORSIKA_BUILD="$c8_root/build-${c8_mode,,}" -DRADIO_EXECUTION="$c8_mode" \
    > "$c8_root/configure-consumer-$c8_mode.log" 2>&1
  cmake --build "$c8_root/consumer-$c8_mode" --parallel 256 --target coreas_fixture coreas_pair_fixture > "$c8_root/build-consumer-$c8_mode.log" 2>&1
done
while [ ! -f "$c8_root/CONTROLS_CUDA_PASSED" ]; do
  kill -0 560539 2>/dev/null || { echo "Controls driver stopped before CUDA gates"; exit 1; }
  sleep 5
done
C8_TAUOLA_ENERGY_CSV="$c8_root/tau_energy_current.csv" "$c8_root/build-openmp/tests/modules/testMountainOriginalLeptons" 'Conditioned TAUOLA remains finite*' > "$c8_root/tau_energy_current.log" 2>&1
C8_TAU_SPIN_MATRIX_CSV="$c8_root/tau_spin_current.csv" "$c8_root/build-openmp/tests/modules/testMountainTauSpin" '[.spinangular]' > "$c8_root/tau_spin_current.log" 2>&1
for c8_mode in OPENMP CUDA; do
  for c8_kind in uniform plane; do
    taskset -c 0-255 "$c8_root/consumer-$c8_mode/coreas_pair_fixture" "$c8_root/paired_fixtures/${c8_mode}_$c8_kind" "$c8_kind"
  done
  for c8_kind in uniform radial negative cherenkov vacuum_forward plane plane_reverse boundary matched_boundary matched_unsplit mesh_boundary boundary_reverse matched_boundary_reverse matched_unsplit_reverse boundary_grazing shadow curved curved_reverse; do
    for c8_algorithm in CoREAS ZHS; do
      taskset -c 0-255 "$c8_root/consumer-$c8_mode/coreas_fixture" "$c8_root/optical_fixtures/${c8_mode}_${c8_kind}_$c8_algorithm" "$c8_kind" "$c8_algorithm"
    done
  done
done
python "$c8_root/source/validation/terrain/analyze_interface_coreas.py" --root "$c8_root/optical_fixtures" --output "$c8_root/optical_oracle" > "$c8_root/optical_oracle.log" 2>&1
touch "$c8_root/EXTRA_PHYSICS_PASSED"

