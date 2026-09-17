#!/usr/bin/env bash
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
c8_ray_root=$(cd "$(dirname "$0")/.." && pwd)
c8_ray_stage=$(dirname "$c8_ray_root")
c8_ray_old="$c8_ray_stage/interface-radio-report-20260912"
c8_ray_rp="$c8_ray_old/external/RadioPropa-544a2d6c4e284e3d724cb741dc481245a0f633d7/radiopropa"
export OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$c8_ray_old/external/build-radiopropa:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$c8_ray_root"
mkdir -p data figures
taskset -c 0-255 python tools/prepare_inputs.py
taskset -c 0-255 "$CONDA_PREFIX/bin/g++" -O2 -std=c++17 -I "$c8_ray_stage/source" tools/beta5_real_ray.cpp -o tools/beta5_real_ray > data/build-beta5.log 2>&1
taskset -c 0-255 tools/beta5_real_ray data "$c8_ray_stage/psr_inputs/terrain_enu.ply" > data/beta5-replay.log
taskset -c 0-255 python tools/select_rays.py > data/selection.log
taskset -c 0-255 "$CONDA_PREFIX/bin/g++" -O2 -std=c++17 -fopenmp -I "$c8_ray_rp/include" tools/radiopropa_real_ray.cpp -L "$c8_ray_old/external/build-radiopropa" -lradiopropa -o tools/radiopropa_real_ray > data/build-rp.log 2>&1
taskset -c 0-255 tools/radiopropa_real_ray data > data/rp-run.log 2>&1
taskset -c 0-255 python tools/make_figures.py > data/figures.log 2>&1
taskset -c 0-255 tools/beta5_real_ray data "$c8_ray_stage/psr_inputs/terrain_enu.ply" check_external
taskset -c 0-255 python tools/prepare_sp.py
taskset -c 0-255 "$CONDA_PREFIX/bin/g++" -O2 -std=c++17 -I "$c8_ray_stage/source" tools/sp_beta5_probe.cpp -o tools/sp_beta5_probe > data/build-sp.log 2>&1
taskset -c 0-255 tools/sp_beta5_probe data
taskset -c 0-255 python tools/make_sp_figures.py > data/sp-figures.log 2>&1
