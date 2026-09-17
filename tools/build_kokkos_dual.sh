#!/usr/bin/env bash
# Experimental CUDA/OpenMP single executable; never replaces independent installs.
set -euo pipefail
c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
c8_project=$(dirname -- "$c8_source")
c8_jobs=${C8_BUILD_JOBS:-1}
c8_profile=${C8_KOKKOS_DUAL_PROFILE:-"$c8_source/dependencies/kokkos/profiles/cuda-openmp-ada89"}
c8_build="$c8_project/build/cuda-openmp"
c8_install="$c8_project/install/cuda-openmp"
[[ $c8_jobs =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid C8_BUILD_JOBS' >&2; exit 2; }
command -v conan >/dev/null
command -v nvcc >/dev/null
conan export "$c8_source/dependencies/kokkos"
conan install "$c8_source" -pr:h "$c8_profile" -pr:b default \
  --output-folder="$c8_build/deps" --build=missing \
  -c:a "tools.build:jobs=$c8_jobs"
cmake -S "$c8_source" -B "$c8_build" \
  -DCMAKE_TOOLCHAIN_FILE="$c8_build/deps/conan_toolchain.cmake" \
  -DCONAN_CMAKE_DIR="$c8_build/deps" \
  -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
  -DCORSIKA_KOKKOS_BACKEND=CUDA_OPENMP \
  -DCMAKE_INSTALL_PREFIX="$c8_install" -DCORSIKA_LAUNCHER_PREFIX= "$@"
# A global install requires all configured installable products, including the
# interface libraries and optional terrain applications. Building only the air
# executable can appear to work in a warm tree but fail in a clean checkout.
# Let CMake own the dependency/optional-target list, as build_kokkos.sh does.
cmake --build "$c8_build" --parallel "$c8_jobs"
cmake --install "$c8_build"
echo "Experimental dual executable: $c8_install/bin/c8_air_shower"
