#!/usr/bin/env bash
# Compatibility entry for the optional combined CUDA/OpenMP installation.
set -euo pipefail
c8_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
  exec bash "$c8_tools/build_kokkos.sh" --help
fi
if [[ -n ${C8_KOKKOS_DUAL_PROFILE:-} ]]; then
  export C8_KOKKOS_PROFILE=$C8_KOKKOS_DUAL_PROFILE
fi
exec bash "$c8_tools/build_kokkos.sh" cuda-openmp "$@"
