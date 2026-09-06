#!/usr/bin/env bash
# beta5: one source directory, one build root, one install root.
set -euo pipefail

c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
c8_project=$(dirname -- "$c8_source")
c8_jobs=${C8_BUILD_JOBS:-1}
if [[ $# == 0 || $1 == --help ]]; then
  echo 'Usage: bash tools/build_kokkos.sh openmp|cuda|hip|sycl|openmp,cuda [CMake options]'
  echo 'A list builds independent backends sequentially and installs one shared launcher.'
  echo 'Set C8_KOKKOS_PROFILE for an explicit GPU Conan profile (required for HIP/SYCL).'
  echo 'CUDA otherwise detects a supported local compute capability; no device is guessed.'
  echo 'C8_BUILD_JOBS defaults to 1, including Conan dependency builds.'
  exit 0
fi
[[ $c8_jobs =~ ^[1-9][0-9]*$ ]] || { echo 'C8_BUILD_JOBS must be positive' >&2; exit 2; }
[[ $1 != ,* && $1 != *, && $1 != *,,* ]] || { echo 'Invalid backend list' >&2; exit 2; }
IFS=, read -r -a c8_backends <<< "$1"
shift
declare -A c8_seen=()
declare -A c8_profiles=()
c8_gpu_count=0
for c8_backend in "${c8_backends[@]}"; do
  case "$c8_backend" in
    openmp|cuda|hip|sycl) ;;
    *) echo "Unknown backend: $c8_backend" >&2; exit 2 ;;
  esac
  [[ -z ${c8_seen[$c8_backend]:-} ]] || { echo 'Duplicate backend' >&2; exit 2; }
  c8_seen[$c8_backend]=1
  if [[ $c8_backend != openmp ]]; then c8_gpu_count=$((c8_gpu_count + 1)); fi
done
if (( c8_gpu_count > 1 )); then
  echo 'Build at most one GPU toolchain per invocation; installations may coexist.' >&2
  exit 2
fi

# Resolve every profile before an expensive build. Explicit profiles also allow
# cross-compilation without a local GPU and carry compiler/toolchain settings.
for c8_backend in "${c8_backends[@]}"; do
  c8_profile=${C8_KOKKOS_PROFILE:-}
  if [[ $c8_backend == openmp ]]; then
    c8_profile=${C8_KOKKOS_OPENMP_PROFILE:-openmp}
  elif [[ -z $c8_profile && $c8_backend == cuda ]]; then
    c8_smi=$(command -v nvidia-smi || true)
    if [[ -z $c8_smi && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
      c8_smi=/usr/lib/wsl/lib/nvidia-smi
    fi
    [[ -n $c8_smi ]] || { echo 'No nvidia-smi; set C8_KOKKOS_PROFILE explicitly' >&2; exit 2; }
    c8_cap=$($c8_smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
    case "$c8_cap" in
      7.5) c8_profile=cuda-turing75 ;;
      8.0) c8_profile=cuda-ampere80 ;;
      8.6) c8_profile=cuda-ampere86 ;;
      8.9) c8_profile=cuda-ada89 ;;
      9.0) c8_profile=cuda-hopper90 ;;
      *) echo "Unsupported or mixed CUDA capabilities ($c8_cap); set C8_KOKKOS_PROFILE" >&2; exit 2 ;;
    esac
  elif [[ -z $c8_profile ]]; then
    echo "$c8_backend requires C8_KOKKOS_PROFILE matching the target/compiler; see dependencies/kokkos/profiles" >&2
    exit 2
  fi
  if [[ $c8_profile != /* ]]; then
    c8_profile="$c8_source/dependencies/kokkos/profiles/$c8_profile"
  fi
  [[ -f $c8_profile ]] || { echo "Profile not found: $c8_profile" >&2; exit 2; }
  c8_profiles[$c8_backend]=$c8_profile
done

for c8_backend in "${c8_backends[@]}"; do
  c8_build="$c8_project/build/$c8_backend"
  c8_deps="$c8_build/deps"
  c8_install="$c8_project/install/$c8_backend"
  echo "Building $c8_backend using ${c8_profiles[$c8_backend]}"
  # Activate corsika_venv first. Only generated descriptors live in deps/.
  conan install "$c8_source" \
    -pr:h "${c8_profiles[$c8_backend]}" -pr:b default \
    --output-folder="$c8_deps" --build=missing -c:a "tools.build:jobs=$c8_jobs"
  cmake -S "$c8_source" -B "$c8_build" \
    -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" \
    -DCONAN_CMAKE_DIR="$c8_deps" \
    -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
    -DCORSIKA_KOKKOS_BACKEND="${c8_backend^^}" \
    -DCMAKE_INSTALL_PREFIX="$c8_install" \
    -DCORSIKA_LAUNCHER_PREFIX="$c8_project/install" "$@"
  cmake --build "$c8_build" --parallel "$c8_jobs"
  cmake --install "$c8_build"
done
echo "Unified entry point: $c8_project/install/bin/c8_air_shower"
