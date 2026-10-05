# beta5 documentation map / 文档导航

Start from the `kokkos-beta5` branch, not the repository's older default branch.
Beginner instructions are maintained in both languages; detailed research notes
below are mainly Chinese. They are dated evidence, not interchangeable recipes.

请在 `kokkos-beta5` 分支阅读。入门教程提供中英文；下面多数研究记录为中文，
其结论对应记录中的代码、物理设置和硬件，不应混作当前安装命令。

| Task / 任务 | Entry / 入口 |
|---|---|
| Empty environment → first air shower / 从零安装空气程序 | [English](../README.md), [中文](../README_CN.md) |
| DEM, antennas, terrain build and first shower / 地形全流程 | [English](terrain_getting_started.md), [中文](terrain_getting_started_CN.md) |
| PROPOSAL patched dependencies / 原生样条依赖 | [PROPOSAL recipe](../third_party/conan/proposal), [CubicInterpolation recipe](../third_party/conan/cubicinterpolation) |
| Backend profiles / 后端构建配置 | [Kokkos recipe and profiles](../dependencies/kokkos/README.md) |
| Geographic preparation and height conventions / 地理与高程 | [Preparation workflow](terrain_preparation_workflow_CN.md) |
| Simple convex mountain example / 有限凸体示例 | [Application guide](mountain_neutrino_user_guide_CN.md) |
| Shared material-interface transport / 通用界面输运 | [Interface API](generic_material_interface_transport_CN.md) |
| DEM radio / 山体界面射电 | [Published interface radio](interface_kokkos_radio_CN.md) |
| Neutrino scope / 中微子模型边界 | [Original-module alignment](terrain_original_neutrino_alignment_CN.md) |
| EGS4 electromagnetic backend / EGS4 电磁模块 | [Module layout and usage](egs4_CN.md) |
| Combined air executable / 空气组合程序 | [CUDA/OpenMP experiment](cuda_em_refactor/beta5_dual_cuda_openmp_experiment_CN.md) |
| Cooperative queues / 空气协同队列 | [Independent subshowers](cuda_em_refactor/beta5_independent_subshower_queues_CN.md) |
| Output memory / 输出内存行为 | [Event-boundary release](cuda_em_refactor/beta5_event_memory_release_20260907_CN.md) |
| Historical ZHS edge artifacts / 历史射电端点问题 | [ZHS correction](cuda_em_refactor/beta5_zhs_window_edge_fix_20260907_CN.md) |
| New photon fallback physics correction / 光子回退物理修复 | [Vertex correction](cuda_em_refactor/beta5_photon_fallback_vertex_fix_20260917_CN.md) |

## What a validation claim means / 如何理解验收

- Compiling a backend does not validate its physics or speed. OpenMP/CUDA have
  recorded tests; HIP/SYCL still need target-hardware acceptance.
- Equal seeds do not guarantee identical CPU/GPU shower trees. Use the recorded
  process oracles, replay conditions and ensemble settings when comparing.
- A physics fix does not update archived binaries or repair old datasets. In
  particular, old accelerator ensembles precede the photon fallback vertex fix.
- Empty or truncated radio traces need observer/time-window checks. Terrain's
  interface/moment propagation is not the air observer path or a full-wave solver.
- Raw data, installed libraries, FLUKA and local `build/` directories are not
  shipped on GitHub. A historical local path is provenance, not a download link.

对应中文：编译、物理、性能是三类验收；相同 seed 不保证相同 shower tree；修复不会自动
更新旧二进制/样本。空波形需检查窗口与天线，山体界面矩方法不等于空气 observer 或全波解。
原始数据、FLUKA 和本机构建目录不随 GitHub 分发，历史本地路径不作为下载链接。

## Maintenance check (2026-09-17) / 本次教程检查

The README routes, recipe names, CLI flags and optional-target install behavior
were checked against source. The dual helper now builds the complete configured
target set before its global install; previously a warm tree could hide missing
interface/terrain artifacts. No physics code was changed for this documentation
update. Tests of the helper use mocked external build commands, not a fresh
network dependency build. A clean-machine end-to-end install and HIP/SYCL hardware
certification are **not** claimed by this documentation audit.

本次按源码核对入口、配方、参数与安装目标；双实例脚本改为完整构建后再全局安装，避免
旧缓存掩盖缺少界面库/山体程序的问题。无物理代码改动。脚本回归使用隔离的外部命令模拟，
不把它当作空机器联网全量编译或 HIP/SYCL 硬件验收。

Checks completed / 已完成检查：

- Build-helper contract tests: **8 passed**; full configured build before install,
  optional flags, failures, CPU isolation and CUDA profile discovery.
- Terrain Python regression in the existing environment: **41 passed**, with
  32 existing pyproj/NumPy scalar-conversion deprecation warnings.
- README/tutorial shell syntax and relative-file/heading links checked.
- The documented terrain `--estimate-only` command ran without network access
  and created no output directory. No shower or production build was started.

Reproduce the lightweight checks from the source directory / 源码目录复跑：

```bash
bash -n tools/build_kokkos.sh tools/build_kokkos_dual.sh
python -m unittest discover -s validation -p test_build_kokkos_helpers.py -v
# Optional test dependency, not required to run a shower:
python -m pip install pytest
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  validation/terrain/test_workflow.py \
  validation/terrain/test_geographic_terrain.py \
  validation/terrain/test_terrain_array.py
```

The terrain regression requires the Python package dependencies from the terrain
tutorial. This audit used an existing environment, not a newly provisioned one.
地形回归需要先按地形教程安装 Python 依赖；本次使用既有环境，未声称重新搭建了空环境。
