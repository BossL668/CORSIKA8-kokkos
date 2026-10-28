#!/usr/bin/env bash
# Build independent native applications with shared library adapters.
set -euo pipefail
c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
c8_workspace=${C8_WORKSPACE:-$(dirname -- "$c8_source")}
c8_jobs=${C8_BUILD_JOBS:-1}
if [[ $# == 0 || $1 == --help ]]; then
  cat <<'EOF'
Usage: tools/build_kokkos.sh openmp|cuda|cuda-openmp|hip|sycl|openmp,cuda [CMake options]
Each backend gets separate build/<backend> and install/<backend> directories.
Run install/<backend>/bin/c8_air_shower, c8_ice_cascade or c8_firn_cascade directly.
C8_BUILD_JOBS defaults to 1. C8_WORKSPACE overrides the source's parent directory.
C8_KOKKOS_PROFILE selects an explicit GPU profile (required for HIP/SYCL).
C8_KOKKOS_OPENMP_PROFILE selects a CPU profile. CUDA otherwise detects the GPU.
Use C8_BUILD_PROFILE for a non-default Conan build profile.
EOF
  exit 0
fi
[[ $c8_jobs =~ ^[1-9][0-9]*$ ]] || { echo 'C8_BUILD_JOBS must be positive' >&2; exit 2; }
[[ $1 != ,* && $1 != *, && $1 != *,,* ]] || { echo 'Invalid backend list' >&2; exit 2; }
IFS=, read -r -a c8_backends <<< "$1"
shift
declare -A c8_seen=() c8_profiles=()
for c8_backend in "${c8_backends[@]}"; do
  case "$c8_backend" in openmp|cuda|cuda-openmp|hip|sycl) ;;
    *) echo "Unknown backend: $c8_backend" >&2; exit 2 ;; esac
  [[ -z ${c8_seen[$c8_backend]:-} ]] || { echo 'Duplicate backend' >&2; exit 2; }
  c8_seen[$c8_backend]=1
  c8_profile=${C8_KOKKOS_PROFILE:-}
  if [[ $c8_backend == openmp ]]; then
    c8_profile=${C8_KOKKOS_OPENMP_PROFILE:-openmp}
  elif [[ -z $c8_profile && $c8_backend == cuda* ]]; then
    c8_smi=$(command -v nvidia-smi || true)
    [[ -n $c8_smi ]] || c8_smi=/usr/lib/wsl/lib/nvidia-smi
    [[ -x $c8_smi ]] || { echo 'Set C8_KOKKOS_PROFILE for the target GPU' >&2; exit 2; }
    c8_cap=$("$c8_smi" --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
    case "$c8_cap" in
      7.0) c8_arch=volta70 ;; 7.5) c8_arch=turing75 ;;
      8.0) c8_arch=ampere80 ;; 8.6) c8_arch=ampere86 ;;
      8.9) c8_arch=ada89 ;; 9.0) c8_arch=hopper90 ;;
      12.0) c8_arch=blackwell120 ;;
      *) echo "Unsupported or mixed GPU capabilities: $c8_cap; set C8_KOKKOS_PROFILE" >&2; exit 2 ;;
    esac
    c8_profile="$c8_backend-$c8_arch"
  elif [[ -z $c8_profile ]]; then
    echo "$c8_backend requires C8_KOKKOS_PROFILE" >&2; exit 2
  fi
  if [[ ! -f $c8_profile ]]; then
    c8_profile="$c8_source/externals/conan/profiles/$c8_profile"
  fi
  [[ -f $c8_profile ]] || { echo "Profile not found: $c8_profile" >&2; exit 2; }
  c8_profiles[$c8_backend]=$(realpath -- "$c8_profile")
done
command -v cmake >/dev/null
command -v conan >/dev/null
bash "$c8_source/tools/prepare_data.sh"
for c8_backend in "${c8_backends[@]}"; do
  c8_build="$c8_workspace/build/$c8_backend"
  c8_install="$c8_workspace/install/$c8_backend"
  c8_deps="$c8_build/deps"
  bash "$c8_source/conan-install.sh" --release --output-folder "$c8_deps" \
    --profile "${c8_profiles[$c8_backend]}" --build-profile "${C8_BUILD_PROFILE:-default}" \
    --with-proposal-export
  c8_runtime=${c8_backend//-/_}
  cmake -S "$c8_source" -B "$c8_build" \
    -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" \
    -DCONAN_CMAKE_DIR="$c8_deps" \
    -DCMAKE_PREFIX_PATH="$c8_deps" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
    -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
    -DCORSIKA_ENABLE_PROPOSAL_EXPORT=ON -DCORSIKA_KOKKOS_BACKEND="${c8_runtime^^}" \
    -DCMAKE_INSTALL_PREFIX="$c8_install" "$@"
  cmake --build "$c8_build" --parallel "$c8_jobs"
  cmake --install "$c8_build"
  printf 'Native applications: %s/bin/c8_{air_shower,ice_cascade,firn_cascade}\n' "$c8_install"
  printf 'Before running: export CORSIKA_DATA="%s/share/corsika/data"\n' "$c8_install"
done
