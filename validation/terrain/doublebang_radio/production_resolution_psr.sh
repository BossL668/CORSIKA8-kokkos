#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
mkdir "$c8_root/optical_fixtures_run_resolution"
for c8_mode in OPENMP CUDA; do
  cmake --build "$c8_root/consumer-$c8_mode" --parallel 8 --target coreas_fixture > "$c8_root/build-consumer-resolution-$c8_mode.log" 2>&1
  for c8_kind in uniform radial negative cherenkov vacuum_forward plane plane_reverse boundary matched_boundary matched_unsplit mesh_boundary boundary_reverse matched_boundary_reverse matched_unsplit_reverse boundary_grazing shadow curved curved_reverse; do
    for c8_algorithm in CoREAS ZHS; do
      C8_RADIO_PRODUCTION_RESOLUTION=1 taskset -c 0-255 "$c8_root/consumer-$c8_mode/coreas_fixture" "$c8_root/optical_fixtures_run_resolution/${c8_mode}_${c8_kind}_$c8_algorithm" "$c8_kind" "$c8_algorithm"
    done
  done
done
python "$c8_root/source/validation/terrain/analyze_interface_coreas.py" --root "$c8_root/optical_fixtures_run_resolution" --output "$c8_root/optical_oracle_run_resolution" > "$c8_root/optical_oracle_run_resolution.log" 2>&1
touch "$c8_root/RUN_RESOLUTION_ORACLE_PASSED"
