#!/usr/bin/env bash
# Called only after the frozen v5 UHE service and GPU have gone idle.
set -euo pipefail
c8_project=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5
c8_stage=$c8_project/build/adaptive-v7-gpu-continuation-20260912
c8_old=$c8_project/build/adaptive-v5-work-quantum-20260912
c8_result=/mnt/d/CorsikaData/corsika_validation_results/beta5_adaptive_v5_work_quantum_20260912/high-energy/proton100PeV-2026110001-adaptive-guard/summary.json
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["pass"] and r["returncode"]==0, "baseline did not finish successfully"' "$c8_result"
test ! -e "$c8_stage/source"
cp -a "$c8_old/source" "$c8_stage/source"
cp -a "$c8_stage/overlay/." "$c8_stage/source/"
export C8_ADAPTIVE_STAGE="$c8_stage"
exec /bin/bash "$c8_stage/source/validation/accelerator/build_adaptive_v5_local.sh"
