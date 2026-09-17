#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_root=/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912
c8_previous=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-t400-cooperative-20260911
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export PATH="$CONDA_PREFIX/bin:$c8_cuda/bin:$PATH"
export CC="$CONDA_PREFIX/bin/gcc" CXX="$CONDA_PREFIX/bin/g++"
export CUDAHOSTCXX="$CXX" CUDACXX="$c8_cuda/bin/nvcc" CUDA_HOME="$c8_cuda"
export FLUPRO=/home/yuhanglu/fluka
export CXXFLAGS="-I$c8_cuda/targets/x86_64-linux/include"
export LDFLAGS="-L$c8_cuda/targets/x86_64-linux/lib -L$c8_cuda/targets/x86_64-linux/lib/stubs -L/usr/lib/x86_64-linux-gnu -Wl,--sysroot=/"
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
for c8_mode in OPENMP CUDA; do
  c8_build="$c8_root/build-${c8_mode,,}"
  cmake -S "$c8_root/source" -B "$c8_build" \
    -DCMAKE_TOOLCHAIN_FILE="$c8_previous/deps/conan_toolchain.cmake" -DCONAN_CMAKE_DIR="$c8_previous/deps" \
    -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON -DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP \
    -DCORSIKA_KOKKOS_CUDA_ARCHITECTURES=75 -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON -DWITH_FLUKA=ON \
    -DPYTHIA8="$c8_previous/modules/pythia8/pythia8/install" \
    -DC8_TAUOLA_PREFIX="$c8_previous/modules/tauola/tauola/install" \
    -DC8_INTERFACE_EXECUTION_SPACE="$c8_mode" -DC8_INTERFACE_TEST_THREADS=256 \
    -DCMAKE_INSTALL_PREFIX="$c8_root/install-${c8_mode,,}" \
    "-DCMAKE_EXE_LINKER_FLAGS=$LDFLAGS -Wl,--enable-new-dtags" \
    > "$c8_root/configure-${c8_mode,,}.log" 2>&1
  cmake --build "$c8_build" --parallel 256 --target c8_terrain_cascade testMountainNeutrino testMountainNeutrinoExtended testMountainOriginalLeptons testMountainTauSpin testInterfaceTransport testKokkosInterfaceQueue testKokkosInterfaceRadio testKokkosInterfaceRadioWaveforms > "$c8_root/build-${c8_mode,,}.log" 2>&1
  touch "$c8_root/BUILD_${c8_mode}_PASSED"
  C8_MAGNETIC_ACCURACY_CSV="$c8_root/magnetic-accuracy-$c8_mode.csv" taskset -c 0-255 ctest --test-dir "$c8_build" -R '^(testMountainNeutrino|testMountainNeutrinoExtended|testMountainOriginalLeptons|testMountainTauSpin|InterfaceTransport_|testKokkosInterfaceQueue|testKokkosInterfaceRadio)' --output-on-failure > "$c8_root/tests-${c8_mode,,}.log" 2>&1
  touch "$c8_root/TESTS_${c8_mode}_PASSED"
done
