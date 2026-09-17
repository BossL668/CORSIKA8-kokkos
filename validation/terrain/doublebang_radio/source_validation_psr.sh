#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_build_pid=${1:?pass the running build/test driver PID}
c8_tag=${2:-precision}
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
while [ ! -f "$c8_root/TESTS_CUDA_PASSED" ]; do
  kill -0 "$c8_build_pid" 2>/dev/null || { echo "Source build/test driver stopped before CUDA gates"; exit 1; }
  sleep 5
done
mkdir "$c8_root/optical_fixtures_$c8_tag"
for c8_mode in OPENMP CUDA; do
  cmake --build "$c8_root/consumer-$c8_mode" --parallel 256 --target coreas_fixture coreas_pair_fixture > "$c8_root/build-consumer-$c8_tag-$c8_mode.log" 2>&1
  for c8_kind in uniform plane; do
    taskset -c 0-255 "$c8_root/consumer-$c8_mode/coreas_pair_fixture" "$c8_root/paired_fixtures/${c8_mode}_${c8_kind}_$c8_tag" "$c8_kind"
  done
  for c8_kind in uniform radial negative cherenkov vacuum_forward plane plane_reverse boundary matched_boundary matched_unsplit mesh_boundary boundary_reverse matched_boundary_reverse matched_unsplit_reverse boundary_grazing shadow curved curved_reverse; do
    for c8_algorithm in CoREAS ZHS; do
      taskset -c 0-255 "$c8_root/consumer-$c8_mode/coreas_fixture" "$c8_root/optical_fixtures_$c8_tag/${c8_mode}_${c8_kind}_$c8_algorithm" "$c8_kind" "$c8_algorithm"
    done
  done
done
python "$c8_root/source/validation/terrain/analyze_interface_coreas.py" --root "$c8_root/optical_fixtures_$c8_tag" --output "$c8_root/optical_oracle_$c8_tag" > "$c8_root/optical_oracle_$c8_tag.log" 2>&1
python "$c8_root/run_psr.py" --phase controls --backend openmp --prefix "${c8_tag}_"
python "$c8_root/run_psr.py" --phase controls --backend cuda --prefix "${c8_tag}_"
touch "$c8_root/BVH_PHYSICS_PASSED"
