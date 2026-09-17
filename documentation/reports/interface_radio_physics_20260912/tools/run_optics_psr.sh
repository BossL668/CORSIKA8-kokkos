#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_report=$(cd "$(dirname "$0")/.." && pwd)
c8_stage=$(dirname "$c8_report")
c8_rp="$c8_report/external/RadioPropa-544a2d6c4e284e3d724cb741dc481245a0f633d7/radiopropa"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$c8_report/external/build-radiopropa:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
taskset -c 0-255 "$CONDA_PREFIX/bin/g++" -O2 -std=c++17 -I "$c8_stage/source" "$c8_report/tools/beta5_optics_probe.cpp" -o "$c8_report/tools/beta5_optics_probe" > "$c8_report/data/build-beta5-probe.log" 2>&1
for c8_kind in plane gradient; do
  taskset -c 0-255 "$CONDA_PREFIX/bin/g++" -O2 -std=c++17 -fopenmp -I "$c8_rp/include" "$c8_report/tools/radiopropa_${c8_kind}_probe.cpp" -L "$c8_report/external/build-radiopropa" -lradiopropa -o "$c8_report/tools/radiopropa_${c8_kind}_probe" > "$c8_report/data/build-radiopropa-${c8_kind}.log" 2>&1
done
taskset -c 0-255 "$c8_report/tools/beta5_optics_probe" "$c8_report/data"
taskset -c 0-255 "$c8_report/tools/radiopropa_plane_probe" "$c8_report/data"
taskset -c 0-255 "$c8_report/tools/radiopropa_gradient_probe" > "$c8_report/data/radiopropa_gradient.csv"
printf 'PASS optics probes completed on PSR\n'
