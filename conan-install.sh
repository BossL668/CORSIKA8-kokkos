#!/usr/bin/env bash
# Prepare an isolated Conan 2 graph without changing existing compiler profiles.
set -euo pipefail
c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
c8_output="$PWD/conan_cmake"
c8_build_type=RelWithDebInfo
c8_host_profile=default
c8_build_profile=default
c8_export=False
c8_jobs=${C8_BUILD_JOBS:-1}
usage() {
  cat <<'EOF'
Usage: conan-install.sh [options]
  -s, --source-directory DIR     Project source (default: script directory)
  -o, --output-folder DIR        Dependency descriptors (default: ./conan_cmake)
  -d, --debug                    Debug dependencies
  -r, --release                  Release dependencies
  -rd, --release-with-debug      RelWithDebInfo dependencies (default)
  --profile FILE                Conan host profile (default: default)
  --build-profile FILE          Conan build profile (default: default)
  --with-proposal-export        Use the read-only native-table export dependencies
  -h, --help                    Show this help
C8_BUILD_JOBS defaults to 1. Existing profiles are never replaced.
EOF
}
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -s|--source-directory|-o|--output-folder|--profile|--build-profile)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        -s|--source-directory) c8_source=$2 ;;
        -o|--output-folder) c8_output=$2 ;;
        --profile) c8_host_profile=$2 ;;
        --build-profile) c8_build_profile=$2 ;;
      esac
      shift 2 ;;
    -d|--debug) c8_build_type=Debug; shift ;;
    -r|--release) c8_build_type=Release; shift ;;
    -rd|--release-with-debug) c8_build_type=RelWithDebInfo; shift ;;
    --with-proposal-export) c8_export=True; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ $c8_jobs =~ ^[1-9][0-9]*$ ]] || { echo 'C8_BUILD_JOBS must be positive' >&2; exit 2; }
c8_source=$(cd -- "$c8_source" && pwd)
[[ -f "$c8_source/conanfile.py" ]] || { echo 'Missing conanfile.py' >&2; exit 2; }
command -v conan >/dev/null
conan profile path "$c8_host_profile" >/dev/null
conan profile path "$c8_build_profile" >/dev/null
for c8_recipe in cubicinterpolation proposal kokkos; do
  conan export "$c8_source/externals/conan/$c8_recipe"
done
conan install "$c8_source" --output-folder="$c8_output" --build=missing \
  -pr:h "$c8_host_profile" -pr:b "$c8_build_profile" \
  -s:h "build_type=$c8_build_type" -s:h compiler.cppstd=17 \
  -o "with_proposal_export=$c8_export" -c:a "tools.build:jobs=$c8_jobs"
printf 'Conan descriptors: %s\n' "$c8_output"
