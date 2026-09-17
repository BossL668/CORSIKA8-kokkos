#!/usr/bin/env bash
# Queue read-only Kokkos Tools diagnosis after an isolated PSR timing pilot.
# No rebuild, simulator restart, or instrumentation of the timed reference.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo "Usage: $0 PYTHON TOOLS PILOT WAIT_UNIT PLUGIN GUARD OUTPUT" >&2
  exit 2
fi
c8_python=$1
c8_tools=$2
c8_pilot=$3
c8_wait_unit=$4
c8_plugin=$5
c8_guard=$6
c8_output=$7
test "$(hostname)" = psrpku2025
test -x "$c8_python"
test -f "$c8_plugin"
test -f "$c8_guard"
test ! -e "$c8_output"
while :; do
  c8_active=$(systemctl --user show "$c8_wait_unit" --property=ActiveState --value)
  case "$c8_active" in
    active|activating|deactivating) sleep 5 ;;
    inactive) break ;;
    *) echo "Pilot is not successfully terminal: $c8_active" >&2; exit 1 ;;
  esac
done
# The Python tool rechecks unit Result, event completion, binary hashes,
# CPU affinity, resource guards, and GPU exclusivity before any simulation.
taskset -c 382-511 "$c8_python" "$c8_tools/run_kokkos_host_call_diagnosis.py" \
  --pilot "$c8_pilot" --plugin "$c8_plugin" --tools "$c8_tools" \
  --guard-script "$c8_guard" --sample-log2 6 --wait-unit "$c8_wait_unit" \
  --output "$c8_output"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  "$c8_python" "$c8_tools/report_kokkos_host_call_diagnosis.py" \
  --pilot "$c8_pilot" --diagnosis "$c8_output" --output "$c8_output/report"
