#!/usr/bin/env bash
# Isolated laptop build. No install, table generation, or production overwrite.
set -eo pipefail
source /home/yuhanglu/miniconda3/etc/profile.d/conda.sh
conda activate corsika_venv
set -u
c8_source=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5
c8_parent=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5
c8_build=$c8_parent/build/cooperative-adaptive-20260911
c8_deps=$c8_parent/build/cuda-openmp/deps
export FLUPRO=/home/yuhanglu/fluka
export CXXFLAGS="${CXXFLAGS:-} -I$CONDA_PREFIX/targets/x86_64-linux/include"
export LDFLAGS="${LDFLAGS:-} -L$CONDA_PREFIX/targets/x86_64-linux/lib -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs"
cmake -S "$c8_source" -B "$c8_build" \
  -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" -DCONAN_CMAKE_DIR="$c8_deps" \
  -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON -DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP \
  -DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=89 -DWITH_FLUKA=ON \
  -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=OFF -DC8_COOPERATIVE_BACKEND_CUPTI=OFF \
  -DC8_TAUOLA_PREFIX="$c8_parent/build/cuda-openmp/modules/tauola/tauola/install" \
  -DCMAKE_INSTALL_PREFIX="$c8_parent/install/cooperative-adaptive-experimental"
cmake --build "$c8_build" --parallel 2 --target c8_air_shower \
  testIndependentSubshowerPump testAdaptiveSubshowerControl testCooperativeDriverAffinity \
  testKokkosCooperativeBackend
ctest --test-dir "$c8_build" --output-on-failure \
  -R '^(testIndependentSubshowerPump|testAdaptiveSubshowerControl|testCooperativeDriverAffinity)$'
