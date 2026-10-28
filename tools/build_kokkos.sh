#!/usr/bin/env bash
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
# Distributed under the 3-clause BSD license; see LICENSE.
# Build independent native applications with shared library adapters.
set -euo pipefail
c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
c8_workspace=${C8_WORKSPACE:-$(dirname -- "$c8_source")}
c8_jobs=${C8_BUILD_JOBS:-1}
c8_check=false
c8_fluka=ON
c8_fluka_argument=
c8_cmake_args=()
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
require_command() { command -v "$1" >/dev/null || fail "Missing $1; activate the build environment and see README.md."; }
cache_value() {
  if [[ -f $1 ]]; then
    sed -n "s/^$2:[^=]*=//p" "$1" | head -n 1
  fi
}
if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  cat <<'HELP'
Usage: tools/build_kokkos.sh openmp|cuda|cuda-openmp|hip|sycl|openmp,cuda [--check] [CMake options]
Each backend gets separate build/<backend> and install/<backend> directories.
Run install/<backend>/bin/c8_air_shower directly.
FLUKA is ON by default: set FLUPRO to the complete licensed installation.
-DWITH_FLUKA=OFF explicitly selects a build without FLUKA (different physics).
--check checks prerequisites and reports the plan without downloading/building.
C8_BUILD_JOBS defaults to 1. C8_WORKSPACE overrides the source's parent directory.
C8_KOKKOS_PROFILE selects an explicit GPU profile (required for HIP/SYCL).
CUDA otherwise detects the GPU; C8_CUDA_DEVICE selects an nvidia-smi index/UUID
on a mixed-GPU host. This selects the build target, not the runtime device.
C8_KOKKOS_OPENMP_PROFILE selects a CPU profile. C8_BUILD_PROFILE selects Conan's
build profile. Only a missing default profile is detected; none is overwritten.
C8_PYTHON selects the build Python (default: python3 from the active environment).
HELP
  exit 0
fi
[[ $c8_jobs =~ ^[1-9][0-9]*$ ]] || fail 'C8_BUILD_JOBS must be positive'
[[ $1 != ,* && $1 != *, && $1 != *,,* ]] || fail 'Invalid backend list'
IFS=, read -r -a c8_backends <<< "$1"
shift
while (( $# )); do
  if [[ $1 == --check ]]; then
    c8_check=true
    shift
    continue
  fi
  # Normalize both CMake spellings: -DVAR=value and -D VAR=value.
  c8_arg=$1
  shift
  if [[ $c8_arg == -D ]]; then
    (( $# )) || fail 'Missing value after -D'
    c8_arg="-D$1"
    shift
  fi
  c8_cmake_args+=("$c8_arg")
  if [[ $c8_arg =~ ^-DWITH_FLUKA(:[^=]+)?=(.*)$ ]]; then
    c8_fluka=${BASH_REMATCH[2]^^}
    case "$c8_fluka" in
      ON|TRUE|YES|Y|1) c8_fluka=ON ;;
      OFF|FALSE|NO|N|0) c8_fluka=OFF ;;
      *) fail 'WITH_FLUKA must be ON or OFF' ;;
    esac
  elif [[ $c8_arg =~ ^-DC8_FLUKALIB(:[^=]+)?=(.*)$ ]]; then
    c8_fluka_argument=${BASH_REMATCH[2]}
  fi
done
declare -A c8_seen=() c8_profiles=() c8_fluka_libraries=()
for c8_backend in "${c8_backends[@]}"; do
  case "$c8_backend" in openmp|cuda|cuda-openmp|hip|sycl) ;;
    *) fail "Unknown backend: $c8_backend" ;; esac
  [[ -z ${c8_seen[$c8_backend]:-} ]] || fail 'Duplicate backend'
  c8_seen[$c8_backend]=1
done

# Fail before fetching dependencies when a requested physics library is missing.
if [[ $c8_fluka == ON ]]; then
  require_command ar
  for c8_backend in "${c8_backends[@]}"; do
    c8_library=$c8_fluka_argument
    if [[ -z $c8_library && -n ${FLUPRO:-} ]]; then
      c8_library="$FLUPRO/libflukahp.a"
    elif [[ -z $c8_library ]]; then
      c8_library=$(cache_value "$c8_workspace/build/$c8_backend/CMakeCache.txt" C8_FLUKALIB)
    fi
    [[ -f $c8_library && -r $c8_library ]] || fail \
      'FLUKA is ON: set FLUPRO to the complete installation containing libflukahp.a, or pass -DC8_FLUKALIB=/absolute/path/libflukahp.a. To omit FLUKA explicitly, use -DWITH_FLUKA=OFF.'
    c8_members=$(ar t "$c8_library") || fail "Cannot read FLUKA archive: $c8_library"
    grep -qx 'ndmhep.o' <<< "$c8_members" || fail \
      'This FLUKA archive lacks ndmhep.o, required by the C8 interface. See the FLUKA section in README.md.'
    c8_fluka_libraries[$c8_backend]=$(realpath -- "$c8_library")
  done
fi
for c8_tool in cmake conan git make "${CC:-gcc}" "${CXX:-g++}" "${FC:-gfortran}"; do
  require_command "$c8_tool"
done
c8_python=${C8_PYTHON:-python3}
require_command "$c8_python"
c8_python=$(command -v "$c8_python")
"$c8_python" -c 'import particle, hepunits, numpy' || fail \
  'Build Python needs particle==0.25.1, hepunits==2.4.1 and numpy; see the environment setup in README.md.'
c8_cmake_version=$(cmake --version)
c8_conan_version=$(conan --version)
[[ $c8_conan_version == 'Conan version 2.'* ]] || fail 'Conan 2 is required'
c8_version=$(sed -n 's/^cmake version \([0-9.]*\).*/\1/p' <<< "$c8_cmake_version")
[[ $c8_version == 3.* && $(printf '%s\n' 3.24 "$c8_version" | sort -V | head -n 1) == 3.24 ]] || fail 'CMake 3.24 through 3.x is required; install cmake>=3.24,<4 in the build environment'

for c8_backend in "${c8_backends[@]}"; do
  c8_profile=${C8_KOKKOS_PROFILE:-}
  if [[ $c8_backend == openmp ]]; then
    c8_profile=${C8_KOKKOS_OPENMP_PROFILE:-openmp}
  elif [[ -z $c8_profile && $c8_backend == cuda* ]]; then
    c8_smi=$(command -v nvidia-smi || true)
    [[ -n $c8_smi ]] || c8_smi=/usr/lib/wsl/lib/nvidia-smi
    [[ -x $c8_smi ]] || fail 'nvidia-smi is unavailable; set C8_KOKKOS_PROFILE for the target GPU'
    c8_query=(--query-gpu=compute_cap --format=csv,noheader)
    [[ -z ${C8_CUDA_DEVICE:-} ]] || c8_query+=("--id=$C8_CUDA_DEVICE")
    c8_cap=$("$c8_smi" "${c8_query[@]}") || fail \
      'GPU query failed; check the NVIDIA driver or specify C8_KOKKOS_PROFILE for a known target'
    c8_cap=$(tr -d ' \r' <<< "$c8_cap" | sed '/^$/d' | sort -u)
    case "$c8_cap" in
      7.0) c8_arch=volta70 ;; 7.5) c8_arch=turing75 ;;
      8.0) c8_arch=ampere80 ;; 8.6) c8_arch=ampere86 ;;
      8.9) c8_arch=ada89 ;; 9.0) c8_arch=hopper90 ;;
      12.0) c8_arch=blackwell120 ;;
      *) fail "Unsupported or mixed GPU capabilities: $c8_cap. Select C8_CUDA_DEVICE or C8_KOKKOS_PROFILE explicitly." ;;
    esac
    c8_profile="$c8_backend-$c8_arch"
    printf 'Detected CUDA compute capability %s -> %s\n' "$c8_cap" "$c8_profile"
  elif [[ -z $c8_profile ]]; then
    fail "$c8_backend requires C8_KOKKOS_PROFILE"
  fi
  if [[ ! -f $c8_profile ]]; then
    c8_profile="$c8_source/externals/conan/profiles/$c8_profile"
  fi
  [[ -f $c8_profile ]] || fail "Profile not found: $c8_profile"
  c8_profiles[$c8_backend]=$(realpath -- "$c8_profile")
  if [[ $c8_backend == cuda* ]]; then
    require_command nvcc
    # Bundled profiles specify the target here; custom inherited profiles are
    # checked by Conan/CMake when their complete configuration is resolved.
    c8_arch=$(sed -n 's/^&:kokkos_architecture=//p' "$c8_profile")
    c8_sm=
    case "$c8_arch" in
      VOLTA70) c8_sm=70 ;; TURING75) c8_sm=75 ;; AMPERE80) c8_sm=80 ;;
      AMPERE86) c8_sm=86 ;; ADA89) c8_sm=89 ;; HOPPER90) c8_sm=90 ;;
      BLACKWELL120) c8_sm=120 ;;
    esac
    if [[ -n $c8_sm ]]; then
      c8_codes=$(nvcc --list-gpu-code) || fail 'Cannot query nvcc target architectures'
      grep -qx "sm_$c8_sm" <<< "$c8_codes" || fail \
        "This nvcc cannot compile sm_$c8_sm. Select a compatible CUDA toolkit (V100: CUDA 12.x; RTX 50: CUDA 12.8+)."
    fi
  fi
  printf 'Build %s: profile=%s; FLUKA=%s; jobs=%s\n' \
    "$c8_backend" "${c8_profiles[$c8_backend]}" "$c8_fluka" "$c8_jobs"
  printf 'Install: %s/install/%s\n' "$c8_workspace" "$c8_backend"
done
if ! conan profile path default >/dev/null 2>&1; then
  if $c8_check; then
    echo 'No default Conan profile; a normal build will run conan profile detect.'
  else
    conan profile detect
  fi
fi
if [[ ${C8_BUILD_PROFILE:-default} != default ]]; then
  conan profile path "$C8_BUILD_PROFILE" >/dev/null || fail 'Conan build profile not found'
fi
$c8_check && { echo 'Prerequisite checks passed; no dependencies downloaded or builds started.'; exit 0; }
bash "$c8_source/tools/prepare_data.sh"
for c8_backend in "${c8_backends[@]}"; do
  c8_build="$c8_workspace/build/$c8_backend"
  c8_install="$c8_workspace/install/$c8_backend"
  c8_deps="$c8_build/deps"
  bash "$c8_source/conan-install.sh" --release --output-folder "$c8_deps" \
    --profile "${c8_profiles[$c8_backend]}" --build-profile "${C8_BUILD_PROFILE:-default}" \
    --with-proposal-export
  c8_runtime=${c8_backend//-/_}
  c8_physics_args=("-DWITH_FLUKA=$c8_fluka")
  if [[ $c8_fluka == ON ]]; then
    c8_physics_args+=("-DC8_FLUKALIB=${c8_fluka_libraries[$c8_backend]}")
  fi
  cmake -S "$c8_source" -B "$c8_build" \
    -DCMAKE_TOOLCHAIN_FILE="$c8_deps/conan_toolchain.cmake" \
    -DCONAN_CMAKE_DIR="$c8_deps" \
    -DCMAKE_PREFIX_PATH="$c8_deps" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
    -DCMAKE_BUILD_TYPE=Release -DCORSIKA_ENABLE_KOKKOS=ON \
    -DCORSIKA_ENABLE_PROPOSAL_EXPORT=ON -DCORSIKA_KOKKOS_BACKEND="${c8_runtime^^}" \
    -DPYTHON_EXECUTABLE="$c8_python" "${c8_physics_args[@]}" \
    -DCMAKE_INSTALL_PREFIX="$c8_install" "${c8_cmake_args[@]}"
  cmake --build "$c8_build" --parallel "$c8_jobs"
  cmake --install "$c8_build"
  printf 'Native binary: %s/bin/c8_air_shower\n' "$c8_install"
  printf 'Before running: export CORSIKA_DATA="%s/share/corsika/data"\n' "$c8_install"
done
