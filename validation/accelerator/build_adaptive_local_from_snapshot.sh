#!/usr/bin/env bash
# Wait/serialization belongs to wait_gpu_idle_then_exec.py. No install update.
set -euo pipefail
test "$#" -eq 3
c8_stage=$1
c8_baseline=$2
c8_results=$3
test -d "$c8_stage/overlay"
test -d "$c8_baseline/source"
test ! -e "$c8_stage/source"
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python - "$c8_results" <<'PY'
import json
import sys
from pathlib import Path
r=Path(sys.argv[1])
s=json.loads((r/'STATUS.json').read_text())
assert s['complete'] and len(s['records'])==10, 'five-seed baseline did not finish'
assert all(v['complete'] for v in s['records']), 'incomplete baseline event'
assert json.loads((r/'CORRECTNESS_GATES.json').read_text())['passed']
PY
cp -a "$c8_baseline/source" "$c8_stage/source"
cp -a "$c8_stage/overlay/." "$c8_stage/source/"
export C8_ADAPTIVE_STAGE="$c8_stage"
exec /bin/bash "$c8_stage/source/validation/accelerator/build_adaptive_v5_local.sh"
