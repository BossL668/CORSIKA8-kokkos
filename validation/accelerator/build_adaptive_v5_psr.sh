#!/usr/bin/env bash
# Compile on PSR. Independent snapshot; no install or production binary changes.
set -eo pipefail
test "$(hostname)" = psrpku2025
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
set -u
c8_project=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5
c8_stage=${C8_ADAPTIVE_STAGE:-$c8_project/build/psr-adaptive-v5-work-quantum-20260912}
c8_deps=$c8_project/build/psr-t400-cooperative-20260911/deps
c8_cuda=$c8_project/build/toolchain-cuda126
c8_previous=$c8_project/build/psr-adaptive-v4-batching-20260911/build
c8_pythia=$c8_previous/modules/pythia8/pythia8/install
export PATH="$CONDA_PREFIX/bin:$c8_cuda/bin:$PATH"
export CC="$CONDA_PREFIX/bin/gcc" CXX="$CONDA_PREFIX/bin/g++"
export CUDAHOSTCXX="$CXX" CUDACXX="$c8_cuda/bin/nvcc" CUDA_HOME="$c8_cuda"
export FLUPRO=/home/yuhanglu/fluka
export CXXFLAGS="-I$c8_cuda/targets/x86_64-linux/include"
export LDFLAGS="-L$c8_cuda/targets/x86_64-linux/lib -L$c8_cuda/targets/x86_64-linux/lib/stubs -L/usr/lib/x86_64-linux-gnu -Wl,--sysroot=/"
cmake -S "$c8_stage/source" -B "$c8_stage/build" \
  -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" -DCONAN_CMAKE_DIR="$c8_deps" \
  -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
  -DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP -DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=75 \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=OFF \
  -DC8_COOPERATIVE_BACKEND_CUPTI=OFF \
  -DC8_TAUOLA_PREFIX="$c8_project/build/psr-cooperative-balanced-20260911/build/modules/tauola/tauola/install" \
  -DUSE_Pythia8_C8=SYSTEM -DPythia8_PREFIX="$c8_pythia" \
  -DPythia8_Pythia_h_LOC="$c8_pythia/include/corsika_modules/Pythia8/Pythia.h" \
  -DPythia8_INCLUDE_DIR="$c8_pythia/include/corsika_modules" \
  -DPythia8_LIBRARY="$c8_pythia/lib/corsika" \
  -DCMAKE_INSTALL_PREFIX="$c8_stage/install-unused" \
  "-DCMAKE_EXE_LINKER_FLAGS=$LDFLAGS -Wl,--enable-new-dtags"
cmake --build "$c8_stage/build" --parallel 128 --target c8_air_shower \
  testAdaptiveSubshowerControl testIndependentSubshowerPump \
  testCooperativeDriverAffinity testKokkosCooperativeBackend
ctest --test-dir "$c8_stage/build" --output-on-failure \
  -R '^(testAdaptiveSubshowerControl|testIndependentSubshowerPump|testCooperativeDriverAffinity)$'
