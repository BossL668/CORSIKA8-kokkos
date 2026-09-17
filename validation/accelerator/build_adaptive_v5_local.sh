#!/usr/bin/env bash
# Isolated local build: no production install, no changes to table caches.
set -eo pipefail
source /home/yuhanglu/miniconda3/etc/profile.d/conda.sh
conda activate corsika_venv
set -u
c8_project=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5
c8_stage=${C8_ADAPTIVE_STAGE:-$c8_project/build/adaptive-v5-work-quantum-20260912}
c8_deps=$c8_project/build/cuda-openmp/deps
c8_previous=$c8_project/build/cooperative-adaptive-20260911
c8_pythia=$c8_previous/modules/pythia8/pythia8/install
export FLUPRO=/home/yuhanglu/fluka
export CUDA_HOME="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CXXFLAGS="${CXXFLAGS:-} -I$CONDA_PREFIX/targets/x86_64-linux/include"
export LDFLAGS="${LDFLAGS:-} -L$CONDA_PREFIX/targets/x86_64-linux/lib -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs"
cmake -S "$c8_stage/source" -B "$c8_stage/build" \
  -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" -DCONAN_CMAKE_DIR="$c8_deps" \
  -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
  -DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP -DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=89 \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=OFF \
  -DC8_COOPERATIVE_BACKEND_CUPTI=OFF \
  -DC8_TAUOLA_PREFIX="$c8_project/build/cuda-openmp/modules/tauola/tauola/install" \
  -DUSE_Pythia8_C8=SYSTEM -DPythia8_PREFIX="$c8_pythia" \
  -DPythia8_Pythia_h_LOC="$c8_pythia/include/corsika_modules/Pythia8/Pythia.h" \
  -DPythia8_INCLUDE_DIR="$c8_pythia/include/corsika_modules" \
  -DPythia8_LIBRARY="$c8_pythia/lib/corsika" \
  -DCMAKE_INSTALL_PREFIX="$c8_stage/install-unused"
cmake --build "$c8_stage/build" --parallel 2 --target c8_air_shower \
  testAdaptiveSubshowerControl testIndependentSubshowerPump \
  testCooperativeDriverAffinity testKokkosCooperativeBackend
ctest --test-dir "$c8_stage/build" --output-on-failure \
  -R '^(testAdaptiveSubshowerControl|testIndependentSubshowerPump|testCooperativeDriverAffinity)$'
