# 山体应用：配置、地形准备与首次运行

[English](terrain_getting_started.md) · [主安装教程](../README_CN.md)

本文针对已发布 beta5 的应用接口，不依赖旧 mountain 项目，也不把未提交的射电原型当作已发布功能。
先完成主 README 第 2–4 节的空环境、分支/子模块、Conan 配方与合法 FLUKA 配置。
只安装 Python 准备工具不会生成 C++ 模拟程序。

## 1. 编译可选应用

完整山体 shower 需要授权 FLUKA 的库与 Fortran 头文件。几何验证器本身不需要 shower
物理，但下面的完整流程需要 FLUKA：

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
conda activate corsika_venv
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export FLUPRO="$HOME/fluka"
test -r "$FLUPRO/libflukahp.a"
test -r "$FLUPRO/flukapro/(FLKMAT)"
test -r "$FLUPRO/flukapro/(DIMPAR)"
test -r "$FLUPRO/flukapro/(FLKCMP)"
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON \
  -DC8_INTERFACE_EXECUTION_SPACE=OPENMP
```

任何文件检查失败应先停止并处理安装问题。授权包头文件布局不同，可传
`-DC8_TERRAIN_FLUKA_INCLUDE_DIR=/absolute/header/directory`，并检查该目录内的三个文件。
不要更换物理模型绕过要求。脚本建立自己的 Conan 工具链并构建固定版本 Pythia/TAUOLA，
不要求已有 `build/external`。在已有 OpenMP 构建中开启此选项会增加目标；不要覆盖生产任务正在使用的程序。

回到项目容器目录验证安装：

```bash
cd ..
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_terrain_environment --help
install/openmp/bin/c8_terrain_cascade --help
install/openmp/bin/c8_mountain_neutrino --help
```

`c8_mountain_neutrino` 是较早的有限凸体示例；真实 DEM、大气嵌入与介质穿越用下面的
`c8_terrain_cascade`。这些是直接可执行文件，不经过空气统一启动器。

### NVIDIA 路线

完成主教程的 CUDA Toolkit 安装后，将构建命令替换为：

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda \
  -DWITH_FLUKA=ON -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON \
  -DC8_INTERFACE_EXECUTION_SPACE=CUDA
```

之后使用 `install/cuda/bin/...`、对应的 `CORSIKA_DATA` 和 `--threads 1 --device 0`。
架构/profile 选择规则同空气程序。山体界面执行空间在**构建时**选择，空气程序的
`--kokkos-execution cuda-openmp` 不是山体参数。若用组合构建脚本，也要明确选择
`C8_INTERFACE_EXECUTION_SPACE=OPENMP` 或 `CUDA`，路径改为 `install/cuda-openmp`；
这仍不表示山体双端协同。

## 2. 准备小范围地理场景

在源码目录安装独立地形 Python 包，`pyproject.toml` 会声明所需地理依赖：

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
python -m pip install ./python
c8-terrain --help
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" --estimate-only
```

四个数依次为西经度、南纬度、东经度、北纬度，单位为度。估算模式不下载。检查规模后再执行：

```bash
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" \
  --check-with ../install/openmp/bin/c8_terrain_environment
```

CUDA 路线相应替换验证器路径。首次下载 DEM/大地水准面数据，之后相同请求复用已校验缓存。
`--cache PATH` 可放其他磁盘，`--offline` 只适用于已有完整缓存。改变场景时换输出目录，不覆盖科研产品。

不提供站位时只有**一个演示天线**，不是实际 21CMA 阵列。
地理站位 CSV 用 `--antennas-csv stations.csv`（列名 `name,longitude_deg,latitude_deg`），
原测量文件格式用 `--station-directory DIR`。默认高度是 DEM + 1 m；实测椭球高/EGM96
正高需要显式配置。见[地形与高度基准说明](terrain_preparation_workflow_CN.md)及
[可编辑配置](../configs/mountain/terrain_region_21cma.yaml)。

检查 `terrain/scene.yaml`、`terrain_manifest.yaml`、`observers.yaml`、
`native_environment_check.yaml` 和三张准备诊断图。完整 `terrain/` 包含网格和相对路径引用，
迁移时一起保留。地形 ENU 不是空气程序天线的 NWU，也不是分析用的 Ex′/Ey′/Ez′。

## 3. 先测试传播，再开启射电

这个小场景将原点设在区域中心地表；必须从 manifest 和原生几何检查确认，不要将这组注入坐标
直接套在其他场景。在项目容器目录执行：

```bash
cd "$HOME/corsika-21cma-kokkos-beta5"
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_terrain_cascade \
  --scene "$HOME/CorsikaData/terrain/demo/terrain/scene.yaml" \
  --primary photon --energy-GeV 0.002 --seed 67101 \
  --position-m 0 0 1 --direction 0 0 -1 \
  --em-backend kokkos --threads 2 --batch 8 \
  --resident-capacity 256 --device-memory-MiB 512 \
  --output "$HOME/CorsikaData/terrain/photon_entry"
```

这是低成本 2 MeV 光子入山/界面示例，不是 shower 或射电性能测试。
标量对照改为 `--em-backend proposal` 并换输出名称；CUDA 使用对应安装程序、
`--threads 1 --device 0`，不能仅给 CPU 二进制换个后端标签。
首次运行可能初始化介质缓存；预算不足时不得关闭内存门禁强行运行。

检查 `terrain_run.yaml` 的完成状态、实际执行空间、介质转换、待处理粒子、缺失物理计数
及诊断截断。产生 CSV 或进程返回不等于完整传播。
`--track-row-limit` 限制记录行数，`--transport-step-limit` 是失败预算，
`--transport-window-ns` 是可选物理时间截断，窗口末端存活粒子不能当作沉积能量。
额外能量诊断用 `--energy-ledger`，它会增加开销。

## 4. 可选界面射电与中微子

DEM 应用现在可在 `--em-backend kokkos` 下加 `--radio`，根据场景 `radio:` 配置同时计算
CoREAS/ZHS；这不是空气程序的 `--radio-backend kokkos`。先准备有效观测点，可重复第 3 节、
增加 `--radio` 并换输出名称作接线测试；2 MeV 信号可能很弱。科学模拟需要合适的初级、
天线与窗口/收敛检查，不只是增大能量。

已发布路径在所选设备累积矩，在主机重建波形。
`samples`、`sample_rate_GHz`、`start_ns`、`moment_order`、细分与 `memory_MiB` 配置见
[射电使用和限制](interface_kokkos_radio_CN.md)。总输运预算包含射电分配，射电单独预算
不是额外的免费内存。传播模型采用直线段、至多一次透射界面，不包含任意反射、多重穿界、
衍射或全波传播；有限阶矩/时间窗与光学近似必须单独验收，本文不宣称波形严格等价。

中微子使用 `--primary nu_tau`（或其他受支持味）、合适的 `--energy-GeV` 和经过几何检查的
注入点/方向。不加 force 参数就是自然抽样，短岩石弦长通常无反应。
`--force-vertex-cc` / `--force-vertex-nc` 要求顶点在岩体内，是条件事例而非事件率样本。
默认 CC+NC，τ 默认 TAUOLA 衰变；参见[原版模块对齐与模型能区](terrain_original_neutrino_alignment_CN.md)。
coverage、complete-weak、事件极化与介质退极化要求门禁在物理不可用时拒绝运行，不会自动安装
缺失模型；不能将输运通过表述成完整中微子物理验收。

## 5. 验收与可迁移边界

先看几何测试清单与小 CPU/OpenMP 事例，显卡测试需有资源许可。编译与生产验收是两件事：

```bash
ctest --test-dir build/openmp -N -R 'Terrain|Interface|Mountain'
```

核对测试列表后再加 `--output-on-failure` 执行。
当前是有限闭合介质体嵌入原生大气，支持相应材料卡，不是通用任意多体积导航器或全球固体地球。
准备工具的岩体海拔/网格门禁仍适用。新场景要检查人工底面/侧壁、物理能区、记录完整性、
数值收敛以及 CPU/加速结果。HIP/SYCL 仍需目标硬件验收，存在 profile 不是验收保证。
