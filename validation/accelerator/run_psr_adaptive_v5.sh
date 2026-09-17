#!/usr/bin/env bash
# Gates then short same-binary pilots; do not automatically start hour-long UHE.
set -euo pipefail
test "$(hostname)" = psrpku2025
c8_stage=${C8_ADAPTIVE_STAGE:-/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v5-work-quantum-20260912}
c8_old=${C8_ADAPTIVE_BEFORE:-/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v4-batching-20260911}
c8_assets=/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_cooperative130_100PeV_20260911
c8_out=${C8_ADAPTIVE_OUTPUT:-/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v5_work_quantum130_20260912}
c8_policy=${C8_ADAPTIVE_POLICY:-adaptive-v5-work-quantum}
c8_build_unit=${C8_ADAPTIVE_BUILD_UNIT:-c8-psr-adaptive-v5-build-20260912}
export FLUPRO=/home/yuhanglu/fluka
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec taskset -c 382-511 /home/yuhanglu/miniconda/envs/corsika_venv/bin/python \
  "$c8_stage/source/validation/accelerator/run_adaptive_cooperative_acceptance.py" \
  --before "$c8_old/build/applications/c8_air_shower" \
  --after "$c8_stage/build/applications/c8_air_shower" \
  --fixture "$c8_stage/build/tests/accelerator/testKokkosCooperativeBackend" \
  --output "$c8_out" --data "$c8_assets/data" --libraries "$c8_assets/lib" \
  --antennas "$c8_assets/antennas.txt" --threads 130 --expected-affinity 382-511 \
  --policy-version "$c8_policy" --before-has-policy \
  --sample-threads --require-exclusive-gpu --timing-seeds "${C8_ADAPTIVE_TIMING_SEEDS:-2}" --high-energy-seeds 0 \
  --wait-build-unit "$c8_build_unit"
