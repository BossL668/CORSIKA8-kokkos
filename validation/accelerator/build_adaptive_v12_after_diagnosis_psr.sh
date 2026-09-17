#!/usr/bin/env bash
# Do not compile concurrently with the frozen v11 profiler/timing diagnostic.
set -euo pipefail
test "$(hostname)" = psrpku2025
c8_prior=c8-psr-adaptive-v11-exact-host-profile-r2-20260912
while systemctl --user is-active --quiet "$c8_prior"; do sleep 5; done
test "$(systemctl --user show "$c8_prior" --property=ExecMainStatus --value)" = 0
test "$(systemctl --user show "$c8_prior" --property=Result --value)" = success
export C8_ADAPTIVE_STAGE=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v12-coalesced-fallback-20260912
exec taskset -c 0-125 bash "$C8_ADAPTIVE_STAGE/source/validation/accelerator/build_adaptive_v11_psr.sh"
