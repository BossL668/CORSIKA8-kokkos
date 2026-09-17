#!/usr/bin/env bash
# Persistent server test entry; scalar generators stay single-threaded.
set -euo pipefail
test "$(hostname)" = psrpku2025
c8_stage=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v4-batching-20260911
c8_old=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-cooperative-balanced-20260911
c8_assets=/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_cooperative130_100PeV_20260911
c8_out=/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v4_batching130_20260911
export FLUPRO=/home/yuhanglu/fluka
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Physical cores 126..255, one hardware thread each (logical CPUs 382..511).
exec taskset -c 382-511 /home/yuhanglu/miniconda/envs/corsika_venv/bin/python \
  "$c8_stage/source/validation/accelerator/run_adaptive_cooperative_acceptance.py" \
  --before "$c8_old/build/applications/c8_air_shower" \
  --after "$c8_stage/build/applications/c8_air_shower" \
  --fixture "$c8_stage/build/tests/accelerator/testKokkosCooperativeBackend" \
  --output "$c8_out" --data "$c8_assets/data" --libraries "$c8_assets/lib" \
  --antennas "$c8_assets/antennas.txt" --threads 130 --expected-affinity 382-511 \
  --policy-version adaptive-v4-bounded-batching --before-has-policy \
  --sample-threads --timing-seeds 2 --high-energy-seeds 1 \
  --wait-build-unit c8-psr-adaptive-v4-build-20260911
