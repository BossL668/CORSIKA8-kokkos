# 真实山体：空气磁场、曲线边界与模块化应用验收

日期：2026-09-08。接续[零场多介质输运阶段](terrain_multimaterial_kokkos_transport_validation_CN.md)。

## 结论与范围

后续[严格浮点首次分叉诊断](terrain_floating_point_diagnosis_CN.md)已用独立 GPU 探针
确认首个出山差异来自 FMA 舍入，并发现 DEM 共同顶点的面归属需要完善；未修改生产算法。

`c8_terrain_cascade` 现在默认使用 **IGRF14 / 2027 年的均匀局地空气磁场**，
岩石节点始终 **B=0**。CPU 与 Kokkos 的 γ/e± 输运均接通真实非凸 DEM 岩气边界。
应用主体只保留参数、场景/物理模型装配、后端选择和输出生命周期，底层求交与输运不在 `main()` 实现。

本轮 **18 个受控事例完整运行、材料检查通过**，不是“山体生产级验收全部完成”：
真实 DEM 的岩内/岩外跨界 CoREAS/ZHS 尚未接入这个应用，`radio=disabled`；
CPU/Kokkos 系综、完整能量账本及射电时间窗验收尚待完成。
OpenMP/CUDA 逐行严格浮点比较也没有全部通过，下面保留差异。
不要用另一个凸体示例的内部射电测试替代此处的跨界射电验收。

## 1. 应用与可复用模块

```text
c8_terrain_cascade.cpp：参数 → 模型/过程顺序 → 后端 → 输出关闭
  ├─ TerrainScene                         地理坐标、高程、DEM/天线门禁
  ├─ TerrainAtmosphere + TerrainMagneticField
  │                                        五层空气有 B；岩石无 B
  ├─ TerrainMagneticTracking               原空气 leapfrog + DEM 曲线求交
  │    └─ TerrainCurvedBoundary            CPU/Kokkos 共用的 POD/BVH 算法
  ├─ CPU Cascade                          原 PROPOSAL/强子/中微子模块
  └─ runAcceleratedTerrain                应用适配器，一次调用
       ├─ TerrainEmPreparation            实际空气、岩石 calculators 各自导出
       ├─ TerrainEmRouter                 有界前沿、CPU 指定过程回退
       └─ TerrainEmSession                Kokkos EM 步进，复用 beta5 物理函数
              └─ TerrainShowerOutput      流式轨迹、跨界及完整性审计
```

| 文件 | 主要职责 |
|---|---|
| `applications/c8_terrain_cascade.cpp` | 七段带注释的高层流程；保留可见的物理序列，不含 CUDA/Kokkos kernel 或折射计算 |
| `applications/detail/mountain/TerrainAcceleratedRun.hpp` | registry、表准备、session、fallback、HybridCascade 装配；不是另一套物理算法 |
| `corsika/modules/terrain/TerrainAtmosphere.hpp` | 原生 USStdBK 大气、SiO₂ 岩石节点及各自的场属性 |
| `corsika/modules/terrain/TerrainMagneticField.hpp` | 复用 `GeomagneticModel`，校验 epoch，并将 NWU 磁场转到地理 ENU |
| `corsika/modules/terrain/TerrainMagneticTracking.hpp` | CPU 跟踪；非 DEM 几何仍委托原空气跟踪器；缓存 DEM 扁平数组 |
| `corsika/geometry/terrain/TerrainCurvedBoundary.hpp` | 与 leapfrog 相同的二次轨迹/BVH 求交，CPU 与 Kokkos 共用 |
| `corsika/modules/terrain/TerrainParticleAdmission.hpp` | CPU → 设备方向舍入误差的有界修正，不消费随机数 |
| `src/terrain/TerrainEmSession.cpp` | 将 DEM 曲线候选与反应、磁偏转、能损和大气层候选竞争 |
| `validation/terrain/run_multimaterial_acceptance.py` | 独立事例、资源守护及二进制哈希记录 |
| `validation/terrain/summarize_magnetic_acceptance.py` | 配置/表哈希/材料完整性与严格逐行比较，保留失败结论 |
| `validation/terrain/check_magnetic_gates.py` | 磁场文件、年份、顶点、方向与输出保护门禁 |

空气生产程序没有在本轮修改或替换。仅共享 `LeptonTransportStep.hpp` 的磁偏转分支补齐了
`external.disable_observation` 检查：地形材料界面不是大气应用的吸收观测面。
默认 external 参数不变，原空气调用仍保留观测面行为。

## 2. 与原 mountain、原空气程序的对应

### 地理场景

沿用原 mountain 的真实 DEM、地理 ENU、WGS84/EGM96 高程与天线准备方法。
与归档原 mountain 场景逐项比较：**80 个站的位置、海拔、空气密度、折射率，以及地形/原点高度差均为零**。
证据：`environment_reference_comparison_v2.yaml`。网格为 64694 顶点、129384 面、73905 BVH 节点。
几何/地理准备的 125 项 Python 测试及 11 项场景拒绝门禁也已通过。

这些站的高度仍是 **DEM + 1 m 的建模高度**，不是已经校准的实测天线海拔。
DEM 仍限定嵌入原生最低空气层；不支持的场景应拒绝，不能把任意山体塞入同一介质节点。

### 磁场与高度基准

与空气应用使用同一个 `GeomagneticModel::getField()`，但其结果是 North/West/Up：
在地理 ENU 中使用 `(E,N,U)=(-W,N,U)`。
IGRF 输入使用 **椭球高**；大气密度使用 **ASL 正高**，二者不混用。
本场景正高 2802 m，椭球高 2750.2445913867664 m，EGM96 起伏 −51.75540861320508 m。

2027 年空气场 ENU，单位 Tesla：

```text
(1.1256674219440438e-6, 2.4892494879019225e-5, -5.1356450219810849e-5)
```

岩石三个分量严格为零；五个空气层共用上述局地均匀场。
这与原空气模拟的“以站点值近似整场均匀磁场”范围一致，不是三维空间变化 IGRF。
系数文件 SHA-256：`02fc7f4573ef453158eec55d2e5b7f4a25df98ca2475a20463f40b8f2bbfebcf`。

保留 `--magnetic-field none` 的原直线 CPU 路径。
两个零场光子事例与修改前 `tracks.csv` **逐字节一致**：

```text
photon_up    79b08301d9c784bc9e8f0c2809ba7746a7fd8420e53eaf8e8894b950cb0280a3
photon_down  c463bca2f61e0e92c60ef6c17ca0c7ca47ee432f2a0794a23907e8aa18aac5a8
```

### 曲线边界

求交使用原 leapfrog 的 `r(L)=r0+L*d+L²*q`，其中 `L=|v|Δt`，不是弦长或螺旋线弧长。
BVH 裁剪包围整个二次曲线，包括坐标极值；每个有限三角形检查两个平面根、
面内位置与逻辑材料侧。相切不制造跨界；材料切换不吸收粒子、不重置 history、权重或时间。
非 DEM 的空气层仍调用原球形层跟踪。

与原 CPU 空气跟踪器的同状态轨迹对照及 2000 个带电状态的 SI/几何 oracle 通过。
另发现原有 CPU 单位常数给出的磁偏转换算为 `0.29979251468343893`，
共享设备公式为 `0.299792458`，相对差约 `1.89e-7`。
测试将“相同系数下的求交误差”和“既有单位常数差”分开，以解析界约束后者；
**本次未修改空气程序常数，也不声称跨后端逐位一致**。

## 3. 捕获并修复的真实失败

使用最终椭球高配置时，强制 CC 同种子事例在两个加速后端都遇到一个 CPU 生成的电子：
方向平方范数误差约 `−1.9691e-12`。原地形 host 入口允许 `1e-10`，
共享设备几何要求 `1e-12`，导致通过 host 后在设备侧被拒绝。
它不是物理材料错误，也不是应该重抽的反应。

修复仅在地形 CPU → device 交接点进行：

- 范数误差 `≤1e-12` 原样保留；在 `(1e-12,1e-10]` 归一化，不改变方向角、能量、权重、时间和 history。
- 非有限值或更大误差仍拒绝；session 入口也统一检查 `1e-12`。
- 不新增随机数；计数 `cpu_direction_roundoff_canonicalizations` 写入结果。
- 使用原失败种子重跑，OpenMP/CUDA 均发生 2 次修正并完成；不换种子掩盖失败。

非法状态、未知回退和没有有效反应顶点的回退仍终止事例。
开发中失败目录保留 `complete=false`，不纳入通过样本。

## 4. 本轮验收记录

独立 OpenMP/CUDA Release 构建；FLUKA + QGSJet-II.04（80 GeV/n），原物理序列不变。
地形/磁场模块 **10 个用例、18246 项断言通过**。
共享输运、沉积、中微子、地形及既有均匀射电回归 **8/8 CTest 通过**。
既有均匀射电回归不是新 DEM 跨界射电的验收。

各执行空间再运行 **1,000,000 个解析二次曲线查询**，长度最大绝对误差：
OpenMP `0 m`、CUDA `4.44089e-16 m`，检查逻辑入射/出射的解析根。
双根、切向与曲线内部极值的保守 BVH 候选另由针对性的单元测试覆盖。

真实 DEM，seed=67101；四个 1 GeV EM 事例 emthin=1e-6，10 TeV νe 的小成本测试 emthin=0.1。
OpenMP 2 线程，CUDA 单设备；batch=64，流式轨迹上限 500000。

| 事例 | CPU 步数 | OpenMP 步数 | CUDA 步数 | 完整性/材料检查 |
|---|---:|---:|---:|---|
| 光子出山 | 1548 | 1489 | 1489 | 通过 |
| 光子入山 | 1615 | 2109 | 2109 | 通过 |
| 电子出山 | 1383 | 1639 | 1639 | 通过 |
| 正电子出山 | 1433 | 1492 | 1492 | 通过 |
| νe 自然传播 | 6 | 6 | 6 | 通过；没有抽中 CC 是合理结果 |
| νe 强制 CC | 222622 | 251819 | 251819 | 通过；记录未截断 |

所有事例材料错配为零、无 NaN、设备队列排空。
强制 CC 加速事例有 227310 个设备步骤、14 次指定 CPU 回退。

应用入口排版整理后，两种独立 Release 再次编译通过。最终 CUDA 二进制重复相同强制 CC
种子，251819 行轨迹逐字节一致、所有诊断计数一致：
`3c13db7e68f22023553f18a6922df1c03182b286529ac2b45db425c3a10a3cb5`。
对应 `magnetic_cuda_readable_repeat` 与 `magnetic_same_backend_repeat.json`。
这是同设备/同后端可重复性，不代表跨后端逐位相等。

### 没有通过的严格要求

OpenMP/CUDA 的 step/PID/material 离散序列一致，但沿用 `rtol=1e-10, atol=1e-12`
的逐字段紧比较只在自然 νe 用例完全通过。
四个低能 EM 事例最大位置差 `9.66e-9 m`，能量差 `6.78e-15 GeV`；
强制 CC 事例最大位置差 `4.565e-4 m`、时间差 `3.336e-13 s`、
能量差 `5.106e-9 GeV`、方向分量差 `1.704e-6`。
没有为了通过而放宽门限；第一次差异、全部最大值及微小权重差写入 JSON。
尚未用逐过程 replay 证明这些长链差异的所有来源。

CPU 与加速路径 RNG 调度不同，单事例 shower 不应要求相同。
强制 CC 加权沉积 CPU 为 6605.636 GeV，CUDA 为 7187.083 GeV；
不能凭一个强 thinning 条件事例就判定是统计涨落，也不能据此宣称等价。
还需要多种子与完整能量账本验收。

### 资源

Kokkos 固定数据/工作区约 92.6 MiB，应用预算 128 MiB。
CUDA 测试观测设备总显存增量峰值 **468 MiB，约 8188 MiB 的 5.7%**；
子进程 RSS 峰值 740036 KiB。
另有 512 MiB 显存增量、2 GiB RSS 和 240 s 超时守护，仅停止自己的测试进程组。
WSL 的设备总显存增量不是精确的逐进程显存测量，0.25 秒采样也不是硬件级上限保证。

小事例墙钟约 28–37 s，含模型初始化、表导出/缓存、CSV 等，并有同时运行的验证任务。
CPU 与 Kokkos 的 `shower_seconds` 起点包含的准备工作不同；**不据此计算加速比**。

## 5. 使用与复现

完整依赖配置见[独立场景构建说明](terrain_native_scene_build_CN.md)。
在项目容器目录、`corsika_venv` 环境中使用独立 build；不覆盖现有 air build/install：

```bash
cmake --build build/mountain-openmp --target c8_terrain_cascade --parallel 1
cmake --build build/mountain-cuda --target c8_terrain_cascade --parallel 1

# 默认：1 GeV 光子、IGRF14/2027、空气有场/岩石无场。
build/mountain-cuda/applications/c8_terrain_cascade \
  --scene build/mountain-validation-20260908/21cma_scene_v2.yaml \
  --output build/terrain-magnetic-example \
  --position-m 0 0 -0.01 --direction 0 0 1 \
  --em-backend kokkos --device-memory-MiB 128
```

位置是此场景的 ENU，不可直接照搬到不同地形。
CPU 参考去掉 `--em-backend kokkos`；OpenMP 换对应二进制并加 `--threads 2`。
`--magnetic-field none` 仅用于明确需要的零场对照。
`--force-vertex-cc --primary nu_e --energy-GeV 10000` 是岩内给定顶点的 CC 条件级联，**不是事件率**。

从源码目录重现受控测试（输出目录必须不存在）：

```bash
python validation/terrain/run_multimaterial_acceptance.py \
  --binary ../build/mountain-cuda/applications/c8_terrain_cascade \
  --scene ../build/mountain-validation-20260908/21cma_scene_v2.yaml \
  --output-root ../build/terrain-new-cuda-magnetic \
  --aux-cache ../build/mountain-validation-20260908/aux-cache \
  --mode cuda --magnetic-field igrf14 --track-row-limit 500000 \
  --cases photon_up photon_down electron_rock positron_rock nue_natural nue_forced
```

完整证据在容器目录 `build/mountain-validation-20260908/`：

- `magnetic_cpu_final`、`magnetic_openmp_admission`、`magnetic_cuda_admission`：同配置完整输入、轨迹、资源与二进制哈希。
- `magnetic_acceptance_admission/{summary.json,REPORT_CN.md}`：三路径配置校验和严格差异。
- `magnetic_cpu_zero_final`：零场逐字节回归。
- `magnetic_admission_unit.log`、`magnetic_admission_regression_ctest.log`：单位/回归结果。
- `magnetic_application_gates.json`：7 项应用失败门禁全部通过，包含不覆盖既有输出。
- `magnetic_{openmp,cuda}_geometry_v1.log`：百万解析曲线查询。
- 早期 `magnetic_*_v1` 使用 ASL 作为 IGRF 高度；只作为开发记录，**不与最终椭球高样本混用**。

## 6. 生产门禁仍需完成

1. 原 mountain 地形光路的独立迁移：真实有限面、遮挡、Snell/Fresnel、空气/岩石光程和偏振；
   不能把均匀介质 `nR/c` 当作岩气透射。原 mountain 的近地空气/单界面适用域必须显式保留或重新验证。
2. 对同一轨迹的 CPU/Kokkos CoREAS/ZHS，比较岩内、岩外贡献及相干和、完整时间窗与端点行为。
3. 多种子 photon/e±/CC 系综、强子 rest-mass 能量账本、低 thinning 和长期资源测试。
4. 非零场下接近切线的真实 DEM shower：当前流式材料审计仍检查弦中点，
   应补充轨迹内部几何证明，避免将曲线与弦的不同误报为材料错误。
5. 目前是有界同步验证调度器，不是生产常驻波前的性能认证；HIP/SYCL 山体硬件尚未验收。

因此目前可以进行**受控的带磁场多介质输运测试**，尚不能将此应用标记为完整的 21CMA 山体射电生产程序。
