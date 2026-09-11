# Mountain 跨后端严格浮点轨迹：单步 replay 与原 CPU oracle

日期：2026-09-09。接续[曲面边界验收](terrain_curved_boundary_acceptance_CN.md)。

补充：用户后续明确以物理一致性、而不是相同 shower tree 为目标。下文保留严格数值诊断结果，
但严格超限不自动等于物理分布错误；分类判断及新增真实 CPU 分布测试见
[物理迁移复核](cpu_kokkos_physics_migration_audit_20260909_CN.md)。

## 结论

**不能宣称 mountain 的完整跨后端轨迹已经严格一致。** 本轮把两个问题分开了：

1. **Kokkos-OpenMP 对 Kokkos-CUDA：**相同输入、history/step 和物理表的局部步骤，
   在本轮四个 1 GeV 事例中均满足原严格门限；连续自由演化的完整轨迹仍有字段超限。
   FMA 对照确认了融合运算的贡献，但关闭隐式 FMA 不能使整棵 shower 逐位相同。
2. **原标量 CPU 对 Kokkos：**额外确认磁偏转换算系数和 Molière 数值求解策略不同。
   后者在相同随机数的首步 oracle 中造成了方向超限，不能把全部差异解释成硬件舍入。

本轮没有修改磁场、散射、表格、cut、随机流或生产编译选项来强行通过。
测试用完整 DEM、多介质 native-PROPOSAL、空气 IGRF14/2027、岩石 B=0；没有开启跨界射电。

## 1. 比较对象与严格门限

| 比较 | 输入如何对齐 | 能回答什么 |
|---|---|---|
| 完整 OpenMP/CUDA 轨迹 | 同 seed、scene、表、batch，独立演化 | 浮点差异是否沿后续步骤积累 |
| `same-input` shadow | 每次真实设备调用的同一粒子状态、history、step、bank | 一整个局部步骤的跨执行端误差 |
| `aligned-stages` shadow | 每个阶段检查后，给下一阶段注入已记录的设备中间状态 | 当前阶段的误差，而不是上游传播来的误差 |
| 原 CPU oracle | 相同首步轨迹/grammage、能量，以及显式固定的三个散射 uniform | 原 PROPOSAL 与 portable 算法是否数值等价 |

保持 `math.isclose(rtol=1e-10, atol=1e-12)` 等价判据，未改动
`compare_multimaterial_runs.py`；整数过程/组分/状态字段要求精确相等。
另外记录 binary64 位差、ULP 和按字段的最大误差。
这不是“每个字段必须小于 16 ULP”的测试，也不是每个过程一百万点的完整矩阵。
接近零及相减抵消时 ULP 数可能很大，需与该字段的绝对/相对误差一起读。

旧轨迹 CSV 没有 history/process 字段，因此只按其行号作漂移图，**不把行号当成 shower tree 身份证明**。
原标量 CPU 使用不同随机流与调度，不能把其独立 shower 和 Kokkos 按行强制配对。

## 2. 独立诊断实现

- [TerrainEmSession.cpp](../src/terrain/TerrainEmSession.cpp)：同一个实际输运 functor 按存储空间实例化；
  仅 `C8_TERRAIN_STEP_AUDIT` 构建增加有界的 host shadow 与阶段记录。
  host/device 使用相同 native table 数组、辅助数据、输入粒子和 history 分配。
- [TerrainStepAudit.hpp](../validation/terrain/TerrainStepAudit.hpp)：逐 batch/index/history/step，
  记录 selection、DEM boundary、transport、Molière、vertex、末态与输出的字段差异。
  比较数值成员，不比较结构体 padding；只流式写出差异，不在内存保留整个 shower。
- [run_step_replay.py](../validation/terrain/run_step_replay.py)：运行独立诊断程序，保留二进制哈希、
  命令、资源记录及完整性结果；复用现有表和辅助缓存。
- [analyze_step_replay.py](../validation/terrain/analyze_step_replay.py)：分析局部差异、校验 CSV/计数一致性、
  比较旧轨迹字节哈希，绘制完整自由演化的端点分离曲线。
- [probe_scalar_first_lepton.cpp](../validation/terrain/probe_scalar_first_lepton.cpp)：真正调用场景中
  `ContinuousProcess` 构建的 PROPOSAL displacement 和相同介质的 `MoliereInterpol`，不是 portable 函数的自我比较。
- [probe_scalar_magnetic_units.cpp](../validation/terrain/probe_scalar_magnetic_units.cpp)：直接调用原
  `LeapFrogTrajectory`，用固定系数对照隔离单位换算差异。

`c8_terrain_cascade_audit` 和 `c8_terrain_cascade_audit_nocontract` 为 **EXCLUDE_FROM_ALL、未安装**的诊断目标。
后者只给诊断输运目标增加 `--fmad=false`，没有更改生产或空气应用的浮点选项。
显式补偿几何所需的 `std::fma` 仍保留，因此它不是“删除所有 FMA”的实验。
普通应用不分配 shadow 数据，不执行诊断输出。

## 3. 实际运行与局部 replay 结果

四个事例均为 seed=67101、batch=64、emthin=1e-6：光子出山、光子入山、岩内电子、岩内正电子。
同一套原始 21CMA DEM，64694 顶点、129384 三角面；每组 6729 个实际加速步骤。
每个步骤有两种 shadow 比较，共 13458 个步骤对、1928822 次字段比较/组。

| 执行组 | 字段非逐位相等次数 | 原局部严格门限超限 | 阶段对齐后的离散决策差异 |
|---|---:|---:|---:|
| OpenMP + host shadow | 0 | 0 | 0 |
| CUDA 默认 + host shadow | 170695 | 0 | 0 |
| CUDA 诊断关闭隐式 FMA + host shadow | 40601 | 0 | 0 |

所有 12 次 shower 完成；表哈希一致。默认 OpenMP/CUDA 的 8 个 track CSV 与修改前归档**逐字节一致**，
说明本轮诊断没有改变这批实际 shower。关闭隐式 FMA 的组是另一项数值对照，不能冒充原生产输出。

默认 CUDA 首个岩内电子步骤的首次阶段差异是连续步长准备：

```text
continuous_step_grammage_g_per_cm2:
host   62.619193531525184
device 62.619193531525639    (64 ULP)
```

该步真实传播受另一限制约束，不会实际传播上述整个连续步长。
首步轨迹终能分别为 `0.99877706830043844` 和 `0.99877706830043766` GeV，相差 7 ULP。
默认组局部终能最大差 `2.44e-15 GeV`；散射角局部最大差 `3.16e-11 rad`。
关闭隐式 FMA 后相应最大差降为 `2.22e-16 GeV` 与 `1.82e-13 rad`。
这些数值是 portable host/device 的比较，**不是**下节原 CPU PROPOSAL 的结果。

已记录的原始 uniform/key 未发现不同。
默认 `same-input` 中有 28 条 `photon_final.split_uniform` 的差异：这个字段在 Compton/光电分支保存的是
`interaction.loss_quantile`，由率区间运算导出，不是新抽取的 Philox uniform。
阶段输入对齐后这 28 条差异消失；没有伴随 process/component/draw-ID 的整数分叉。
拒绝抽样内部每一次临时 draw 未全部单独写出，本报告不宣称完成了逐 draw 的全流程 CPU tape。

## 4. 为什么完整轨迹仍不通过

下表仍使用原门限，给出自由演化时首次超限的轨迹行号；四组 PID/material 顺序和总行数相同。

| 事例 | 总行数 | 默认 CUDA 首次超限 | 关闭隐式 FMA 首次超限 |
|---|---:|---|---|
| photon_up | 1489 | 165，x1 | 1395，y1 |
| photon_down | 2109 | 125，nx0 | 270，x1 |
| electron_rock | 1639 | 25，ny1 | 197，nz1 |
| positron_rock | 1492 | 63，nx1 | 95，ny0 |

局部差异会被后续方向更新、距离传播、下一次率/散射求值和子粒子继承。
几何求交本身局部没有位差，不等于使用已经不同的轨迹查询时，交点还能相同。
关闭隐式 FMA 推迟了首次超限，却没有让所有点的偏差都变小，也没有使完整轨迹通过。
剩余差异还涉及 CPU/device 数学库和相同函数在不同编译目标上的运算展开；
尚未把每个剩余位差都定位到具体的 `log/sin/erf` 库调用，不能武断归因。

本地诊断图：
[trajectory_drift.png](/mnt/d/CorsikaData/corsika_validation_results/mountain_strict_step_replay_20260909_v1/trajectory_drift.png)。
图中纵轴为端点欧氏距离差，横轴是已排序轨迹行号；不同粒子交错造成锯齿，不能解释为单粒子轨迹振荡。

## 5. 原 CPU oracle 揭示的两项非严格等价

### 5.1 磁偏转单位换算

原 CPU 单位库得 `0.2997925146834389`，portable GeV/T 实现采用 `0.299792458`，相对差约 `1.89e-7`。
用原 `LeapFrogTrajectory` 测试 1 GeV 电子质量、方向 (0.6,0,0.8)、正/负电荷、0/50 μT、
0.001/1/1000 m 的 12 个组合。50 μT、1000 m 示例的位置分量最大差约 `1.13e-6 m`。
**只在诊断中**把原轨迹系数换成同一设备换算系数后，位置差降为约 `1.14e-13 m`；
12 个同系数控制测试均满足 64 ε 的尺度门限。

这说明该系统差是真实的，不是 DEM 的边界容差或 GPU RNG 问题；
它也不能解释 OpenMP/CUDA 之间的差异，因为两个 Kokkos 后端原本就使用同一系数。

### 5.2 Molière 数值求解

对上述电子/正电子的首个 **B=0、直线岩内步骤**，使用实际 CPU 岩石 grammage 积分和
PROPOSAL `UpperLimitTrackIntegral`、`CalculateScatteringAngle2D`，固定：

```text
u1   = 0.94671186350751668
u2   = 0.17442977114114910
uphi = 0.52222779823932797
```

| 首步 | 原 CPU 相对记录的终能差 / GeV | 原 CPU 相对记录的最大方向分量差 |
|---|---:|---:|
| electron_rock | −1.11e−16 | 7.1916e−11 |
| positron_rock | −2.66e−15 | 7.1923e−11 |

例如电子的 CPU 角度为 `0.0040124693584382264` rad，记录方向对应
`0.0040124692858147393` rad。能量结果相近，但该方向差**超过本轮严格门限**。
把 CPU oracle 的终能改成记录中的相同终能，角度不变；原 PROPOSAL 此实现本来也不使用 ef。
因此不能把该方向差归因于约 1e-15 GeV 的能量差。

代码层面确定的不同包括：

- 原 PROPOSAL 的 Molière B 从 15 出发做 6 次 Newton；portable 使用渐近初值加 Halley 更新。
- 原角度 Newton 用 Gaussian 初值、相邻迭代相对差 `1e-4` 停止；portable 使用缓存改进初值、`5e-5` 停止。
- portable 还预计算/重排了 PDF/CDF 中的不变量与多项式求值。

两条路径在实现同一分布的数值求解，但不是逐位相同的算法。
本轮已证实固定随机数下的实际差异；尚未通过逐项消融确定各优化对这两个角度差的独立贡献，
也未证明哪一条更接近数学上的精确逆 CDF。不能仅因为 portable 门限更小就断言它更正确。

## 6. 建议与验收状态

1. 先建立 **terrain 限定的 CPU-reference 数值策略诊断**：统一磁刚度换算，
   对 Molière 初值、B 求解、停止条件逐项对齐/消融，再重复同状态 oracle。
   不通过改变真实 B 值、放宽边界容差或换随机数来绕过问题。
2. 如目标是整棵树逐位一致，还需统一特殊函数和运算顺序，并采用共同决策 tape；
   代价必须另测，不能直接把诊断的 `--fmad=false` 推成生产默认。
3. 物理生产验收需独立的多能量、多种子、全部组分与射电误差预算。
   本次 4 个 1 GeV 事例和 2 个首步 oracle 不替代这些测试。

当前状态：局部 portable host/device 门限通过；完整严格轨迹未通过；原 CPU Molière 严格方向门限未通过。
跨界 CoREAS/ZHS、完整能量闭合和高能大样本均不属于本次通过项。

## 7. 资源保护和复现

测试程序限制 128 MiB 算法设备分配、batch=64；外部 guard 另限制设备总显存增量 ≤512 MiB 且 ≤10%、
子进程 RSS ≤2 GiB、全局可用内存 ≥3 GiB，并设 120 s 单次时限。
实际最大子进程 RSS 约 743 MiB、设备总显存增量最大 457 MiB，无 guard 中止。
后者包含 CUDA context 且是设备级增量，不是精确的每进程显存；和正常空气生产并行时不能用此次时间测性能。

数据、CSV、命令和图保存在 D 盘：
`CorsikaData/corsika_validation_results/mountain_strict_step_replay_20260909_v1/`。
四事例默认诊断与旧归档 8/8 轨迹字节一致；普通（无 hook）CPU/OpenMP/CUDA
电子事例另做了 3/3 字节一致回归。OpenMP/CUDA 的环境、session、曲面 CTest 各 3/3 通过；
分析器 5 个测试通过。源码哈希和普通回归位于结果目录的 `regression_and_sources.json`。
未改 `c8_air_shower.cpp`、未更新 install、未重新制表、未重启/暂停已有高能生产任务。

在已配置好的独立 mountain build、`corsika_venv` 中：

```bash
cmake --build ../build/mountain-openmp --target c8_terrain_cascade_audit -j2
cmake --build ../build/mountain-cuda --target c8_terrain_cascade_audit c8_terrain_cascade_audit_nocontract -j2

# 使用新的输出目录，不覆盖现有诊断；不需要重新生成物理表。
FLUPRO=/path/to/fluka python validation/terrain/run_step_replay.py \
  --binary ../build/mountain-cuda/applications/c8_terrain_cascade_audit \
  --scene /path/to/21cma_scene_v2.yaml --aux-cache /path/to/existing/aux-cache \
  --output /path/to/new/strict-replay/cuda --gpu

python -m pytest validation/terrain/test_step_replay_analysis.py -q
```
