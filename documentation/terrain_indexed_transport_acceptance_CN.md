# beta5 山体：共享顶点求交接入与曲线介质检查

日期：2026-09-09。对象仅为真实 DEM 应用 `c8_terrain_cascade`，不是空气生产程序。

后续更新：本文保留本阶段历史结果；非零曲率的共顶点与近切线问题已继续处理，
见[曲线边界修复与验收](terrain_curved_boundary_acceptance_CN.md)。

## 本轮结论

完成两项修复，并完成 **24 个小型测试配置 + 2 次 CUDA 重复**的完整运行与轨迹完整性检查。
这些不是 26 个独立统计样本，不能据此宣布物理系综或山体射电已经验收。

- **共享顶点直线求交已接入实际 CPU/Kokkos 输运**，不再只是独立 nvcc 探针。
- **介质审计改用曲线路径中点**，避免用弦中点误判带磁场轨迹。
- **严格跨后端浮点轨迹门禁仍未全过**；真实 DEM 跨界 CoREAS/ZHS 仍未接通。

## 1. 修复的来源与范围

### 共享顶点处选中不同三角面

旧数据只存 `origin + edge1 + edge2`。相邻面以不同算术路径恢复同一个顶点，
遇到 FMA 或边缘舍入时可能得到不同面编号。面法向随之不同，不能简单当作无关的报告误差。

现在 `FlatTerrainData` 同时持有原始顶点数组和面顶点编号；
`exportFlatTerrain()` 保留原始编号，`KokkosTerrainSession` 一次上传并复用。
上传前拒绝缺失/越界/重复编号、非有限顶点，以及索引表示与旧曲线表示不一致的数据。
曲线代码需要的旧面数据暂时保留，不以重建顶点替代原始顶点。

直线射线使用已有经过补偿运算验证的 `IndexedTerrainIntersection.hpp`：
按原始编号固定运算顺序、显式 FMA、共享边缘的补偿判定；
共享特征的合格相交面取最小原始面编号。没有平均法向、移动物理顶点、修改 RNG，
也没有扩大几何接受容差。尖角处这个法向选择是确定性约定，不是边缘衍射模型。

`StraightTrackingView` 和 `MagneticTracking` 的零曲率分支都调用该只读查询。
旧 Möller–Trumbore 留作诊断 oracle；没有索引的旧测试视图可显式使用它，
但正式 Kokkos 会话不允许缺少索引数据。

### 曲线中点与弦中点不是同一个点

CPU 的 `Step` 不保留原轨迹对象，只保留端点差；末端方向还可能已被多重散射修改。
因此不能从末端方向重建磁偏转，也不能直接用两个端点的平均值检查材料。

对于本应用的二次 leapfrog 路径，有

\[
\mathbf r(1/2)=\mathbf r_0+
\frac{(\mathbf r_1-\mathbf r_0)+\mathbf v_0\Delta t}{4}.
\]

新增 `TerrainTrajectoryAudit.hpp` 提供纯诊断函数：CPU 使用原始速度与飞行时间，
设备使用起始方向乘输运参数长度，并在末态改变前记录路径中点。
原有直线分支不变。**这不修改 PROPOSAL 既有的沿弦连续能损积分**，
也不声称单个中点检查能证明整条曲线没有跨界；最近边界仍由原求交器限定。

## 2. 实测验收

使用 `corsika_venv`、Release、原有独立 `build/mountain-openmp` 与 `build/mountain-cuda`，
单进程编译；没有覆盖生产 build/install。空气应用源码 SHA-256 前后相同：
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。

| 检查 | 结果 |
|---|---|
| OpenMP/CUDA 构建 | 山体应用、设备探针、几何单测均通过 |
| CPU 环境/磁场单测 | 通过；含 stock leapfrog 中点与新诊断函数比较、弦误判的独立几何反例 |
| Kokkos 会话单测 | 两后端通过；含非法拓扑拒绝、共享边及相邻 1 ULP 输入、固定容量跨批次复用 |
| 二次曲线解析 oracle | 每后端 1,000,000 条；最大距离误差 `4.44e-16 m` |
| 真实 DEM 直线查询 | 每后端 1,000,006 条，重复两轮；与同一索引算法的 host 求值距离/面编号**精确不符 0** |
| 原 mountain 风格的独立旧 mesh oracle | 百万随机射线的有无交点、面编号、既定距离门限不符均 0；最大距离差 `3.40e-10 m` |
| 历史故障点：顶点 31842 | 六条竖直射线全选面 62909；距离为 0.01、1、100 m，正反方向一致；旧算法六条均存在精确结果差异 |
| 场景/磁场失败门禁 | 7/7 通过，未覆盖已有输出 |
| 地形/地理天线/Python 准备工具回归 | 40 通过；有现有 pyproj/NumPy 弃用警告，非测试失败 |

真实场景沿用原 mountain 的 64694 顶点、129384 面、80 个台站：
ENU 原点约 `42.9425° N, 86.7125° E`，正高 2802 m；
空气 IGRF14/2027 局地均匀场，岩石零场。新增只读索引使几何量从
17,668,320 增至 **24,396,336 bytes**，已计入分配前预算。

完整测试配置：

- IGRF14：CPU/OpenMP/CUDA 各运行光子出山、光子入山、岩石内电子、正电子、
  自然 νe 和强制 CC νe 六例。EM 初级 1 GeV，νe 初级 10⁴ GeV，seed=67101。
- B=0：三后端各运行光子出山与岩石电子两例，覆盖零场标量直线入口。
- CUDA：重复 IGRF14 光子出山及强制 CC 两例，分别 **1489 行、251819 行轨迹 CSV 逐字节相同**。

26 次运行全部完成，无截断 CSV、非有限轨迹、倒退时间、非法能量/权重、介质错分、
未回收前沿粒子或驱动退出异常。这里验证的是记录/材料/运行完整性，
**不是强子静质量完整能量闭合验收**。两 Kokkos 后端物理 bank 哈希、cut、几何、场与种子也逐项比对。

显存仍配置 **128 MiB** 应用预算；本轮设备分配记账 103,823,152 bytes。
监控保留 512 MiB 或显存总量 10% 中较小者的显存增量上限、2 GiB 子进程 RSS 上限、
3 GiB 全局可用内存余量。采样最大显存增量 **441 MiB**，最大 RSS **731.4 MiB**，
最小可用内存约 **4.94 GiB**，未触发保护。WSL 数值是设备总用量相对基线增量，
不是精确 NVML 进程归属；也不是无限时长或多 shower 的内存无泄漏证明。

## 3. 没有隐藏的未通过项

六对 IGRF14 OpenMP/CUDA 的**行数、逐行 PID 与介质序列**相同；
但这些字段不包含 history/过程/random draws，不能据此称完整 shower tree 一致。
旧严格门禁（每列统一 `rtol=1e-10, atol=1e-12`，诊断字典精确比较）仍为 **未通过**，没有放宽门限。

| 事例 | 第一次数值不一致 | 第一次超旧门限 |
|---|---|---|
| photon_up | 第 3 行 `x1_m` | 第 165 行 `x1_m` |
| photon_down | 第 3 行 `y1_m` | 第 125 行 `nx0` |
| electron_rock | 第 1 行 `E1_GeV`（相差 7 ULP） | 第 25 行 `ny1` |
| positron_rock | 第 1 行 `nx1` | 第 63 行 `nx1` |
| nue_natural | 无 | 无 |
| nue_forced | 第 89 行 `x1_m` | 第 373 行 `nx1` |

光子第一次出山的共享顶点分叉已经消除。电子/正电子在远离该故障点的能损和散射中仍有舍入差异。
四个 EM 事例的对应端点最大位置差约 `2.5e-11–7.4e-9 m`；强制 CC 的最大差约
`0.4565 mm`，是长轨迹累计差，不是本次共享顶点的距离残差。
原有 CPU SI 磁偏转常量与设备 GeV/T 常量差异也未在本轮修改。

下一阶段仍需：

1. 为**非零曲率**求交补齐共享边/顶点所有权与近切线/双根测试；当前仍保留旧曲线面内判定。
2. 对能损、散射和长曲线增加中间量 replay，区分局部误差与累计误差。
3. 接通真实 DEM 岩气界面的光路与 CoREAS/ZHS，再做同轨迹与波形验收；本应用仍显式 `radio: disabled`。
4. 完整能量台账、多种子系综和生产性能验收。不能用本次小例运行时间推断生产加速比。

## 4. 证据与复现

全部证据位于项目容器目录的 `build/mountain-validation-20260909-indexed/`：

- `summary.json`：26 次完整性、资源、严格比较及第一次数值/门限差异；严格门禁失败显式保留。
- `dem_{openmp,cuda}_exact.yaml`：百万随机与共享顶点的正式会话验证。
- `cuda_unit/`、`ctest_openmp_v2.log`、`magnetic_gates.json`、`terrain_python.log`。
- `{cpu,openmp,cuda}_{igrf,zero}/`、`cuda_repeat/`：命令、二进制哈希、日志、CSV 与总结。
- `c8_terrain_cascade_before_{openmp,cuda}`：本轮修改前的独立诊断二进制，未删除。

在源码目录重新检查（不覆盖已有 `summary.json`；去掉 `--check-only` 才创建新报告）：

```bash
conda activate corsika_venv
python validation/terrain/summarize_indexed_transport.py \
  --root ../build/mountain-validation-20260909-indexed --check-only
```

新增/修改职责：`FlatTerrainPrimitives.hpp` 分离 POD 与旧诊断基础函数；
`IndexedTerrainIntersection.hpp` 负责稳健直线算法；`FlatTerrain{Data,Export}.hpp` 与
`KokkosTerrainSession.hpp` 管理原始拓扑和常驻数据；
`TerrainBoundaryTracking.hpp` / `TerrainMagneticTracking.hpp` 接入标量入口；
`TerrainTrajectoryAudit.hpp` / `TerrainEmSession` / `TerrainShowerOutput` 修正中点审计。
未更改共享 EM 公式、空气应用、射电内核、物理表或种子。
