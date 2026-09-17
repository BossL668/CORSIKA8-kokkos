#!/usr/bin/env bash
# Do not run a many-core build concurrently with the isolated timed pilot.
set -euo pipefail
test "$(hostname)" = psrpku2025
c8_prior=c8-psr-adaptive-v11-acceptance-20260912
while systemctl --user is-active --quiet "$c8_prior"; do sleep 5; done
c8_code=$(systemctl --user show "$c8_prior" --property=ExecMainStatus --value)
test "$c8_code" = 0
export C8_ADAPTIVE_STAGE=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-adaptive-v11-host-profile-exact-r2-20260912
exec taskset -c 0-125 bash "$C8_ADAPTIVE_STAGE/source/validation/accelerator/build_adaptive_v11_psr.sh"
