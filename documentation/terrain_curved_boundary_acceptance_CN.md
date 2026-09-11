# beta5 山体：非零曲率 DEM 边界修复与验收

日期：2026-09-09。范围仅为 `c8_terrain_cascade` 的地形边界，不改空气生产应用、
EM 反应公式、随机流、PROPOSAL 物理表或射电内核。

本轮接续[共享顶点直线求交验收](terrain_indexed_transport_acceptance_CN.md)，
解决其中尚未处理的**非零曲率**求交问题。以下数值测试不等同于生产级物理系综或跨界射电验收。

**本轮边界与运行完整性验收通过：24 个配置 + 2 次 CUDA 重复全部完成。
整个山体程序的生产验收尚未完成，严格跨后端轨迹和跨界射电仍有未通过/未接通项。**

后续逐阶段对照见[严格单步 replay 与原 CPU oracle](terrain_strict_step_replay_CN.md)：
已区分局部舍入的传播、原 CPU 磁偏转单位系数和 Molière 数值策略差异，未放宽门限。

## 1. 找到并修复的问题

先建立 9 种边界用例，每种测试两种 BVH 面遍历顺序：旧实现出现 **8/18 项失败**。
修复基础用例后再用真实 DEM 检查，另外暴露出单位法向量表示的平面偏差，未将其掩盖在距离容差内。

| 问题 | 后果 | 修复 |
|---|---|---|
| 仅比较 `flight < best` | 相同交点的面编号依赖遍历顺序 | 先比较距离；距离精确相同时取最小原始面编号 |
| 面内坐标允许 `-1e-12` | 共边外侧相邻 1 ULP 的输入也可能被接纳 | 使用原始顶点和补偿面积符号判定，不再扩大三角面接受范围 |
| 直接计算 `b*b - 4*a*c` | 近切线两根抵消成零，漏掉合法交点 | 补偿判别式、稳定二次公式及残差修正 |
| 单位法向量 + 单个顶点定义平面 | 舍入后的平面不再严格经过另外两个原始顶点 | 用原始三顶点的补偿叉积直接构造平面多项式 |
| 最小飞行长度钳制后未复查上限 | 可能返回超过本次物理步长上限的交点 | 检查钳制后的长度仍不超过 `maximum_length_m` |

真实 DEM 中，第一版仅修补根和边缘判定后，共顶点 31842 的六条曲线仍全部选错面，
有的距离偏差仅约 `1.05e-16 m`，但面编号会变成 63448/63449。因此不能只凭距离误差小就忽略。
从原始顶点构造平面后，六条曲线都得到原始面 **62909**，距离精确等于预设值。
中间失败报告 `curved_openmp.yaml` 保留，最终报告为 `curved_openmp_v2.yaml`，二者不混用。

## 2. 算法及代码职责

轨迹仍是原来的二次 leapfrog 近似，不改为另一种磁场积分器：

\[
\mathbf r(L)=\mathbf r_0+L\mathbf d+L^2\mathbf q.
\]

从原始顶点构造
\(\mathbf N=(\mathbf v_1-\mathbf v_0)\times(\mathbf v_2-\mathbf v_0)\)，再解

\[
aL^2+bL+c=0,\quad
a=\mathbf N\cdot\mathbf q,\quad b=\mathbf N\cdot\mathbf d,\quad
c=\mathbf N\cdot(\mathbf r_0-\mathbf v_0).
\]

这些几何系数和关键差值使用 `Twofold` 补偿运算，根仍是 double。
系数按二的幂缩放，避免直接判别式的溢出；稳定公式给出两根，再用补偿多项式残差修正。
原有法向量保留用于导数的方向和尺度判定。面内检查直接计算 `vertex - r(L)` 的补偿形式，
不先把全局交点舍入成一个 Vec3 再相减。

- [`CurvedIntersectionNumerics.hpp`](../corsika/geometry/terrain/CurvedIntersectionNumerics.hpp)：
  原始顶点、补偿平面系数、二次求根和有限三角面判定。
- [`TerrainCurvedBoundary.hpp`](../corsika/geometry/terrain/TerrainCurvedBoundary.hpp)：
  BVH 遍历、逻辑材料侧、物理步长限制、最近根和面编号选择；CPU/Kokkos 共用。
- [`testTerrainCurvedBoundary.cpp`](../tests/accelerator/testTerrainCurvedBoundary.cpp)：
  host/OpenMP/CUDA 单测，独立预期值及百万随机曲线。
- [`c8_terrain_curved_probe.cpp`](../applications/c8_terrain_curved_probe.cpp)：
  真实 DEM 的独立 long-double 几何 oracle 和实际 Kokkos 查询比较。
- [`TerrainDeviceQuery.cpp`](../src/terrain/TerrainDeviceQuery.cpp)：
  探针的具名 Kokkos functor；几何只上传一次，查询工作区固定 4096 条，重复两轮。
- [`summarize_curved_transport.py`](../validation/terrain/summarize_curved_transport.py)：
  完整性、输入/表哈希、零场回归、重复运行与严格轨迹门禁的汇总。

没有移动粒子来掩盖边界问题，也没有新增射线细分或随机抽样。
既有 `1e-12` 穿界方向门限、`1e-8 m` 查询回溯容许量和 `1e-9 m` 最小飞行约定保留。
**曲线可能两次碰到同一特征，不能不看距离就统一取最小面编号**；本轮仅对距离精确相同的合格候选破平局。

## 3. 几何和构建验收

使用 `corsika_venv` 与独立 Release `build/mountain-openmp`、`build/mountain-cuda`。
本轮没有更新生产安装目录。空气应用的源码哈希保持
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。

| 检查 | 结果 |
|---|---|
| Release 构建/CTest | OpenMP、CUDA 山体应用与探针构建通过；两后端各 3/3 CTest 通过 |
| host-only 曲线单测 | 2,000,038 次检查通过 |
| OpenMP 曲线单测 | 同样的检查通过，与 host 精确不符 0 |
| CUDA 曲线单测 | 同样的检查通过，与 host 精确不符 0 |
| 独立预期值 | 普通随机曲线使用 long double；故意严重抵消的近切线用例用 100 位十进制独立计算的 binary64 根 |
| 共边、非共面共享顶点 | 正反穿界、两种遍历顺序、1 ULP 边侧和精确面编号均通过 |
| 双根/切线/短步长 | 进入根、离开根、重根、负判别式、步长不足及零区间均通过 |
| 系数缩放 | `2^-400`、1、`2^400` 的两根保持 0.5 和 1.5 |
| host ASan/UBSan/LSan | 同一 2,000,038 次几何检查通过，无内存/未定义行为/泄漏报告；只覆盖该独立单测，不代表整个 shower 的长期内存证明 |
| 真实 DEM 随机曲线 | 两后端各 100,000 条，重复两轮；host/device 距离/面编号精确不符 0 |
| DEM long-double oracle | 有无交点、面编号、距离门限不符均 0；最大距离差 `4.90e-15 m` |
| DEM 原始顶点 31842 | 每后端六条；长度 1/128、1/64、1/32 m，正反方向都精确匹配 |
| 地形、经纬度/天线准备工具 | 40 项通过；26 条既有弃用警告 |
| 场景/磁场失败门禁 | 7 项通过，拒绝错误参数且不覆盖已有输出 |

2,000,038 次包含同一百万随机输入的两种遍历顺序，**不是两百万独立 shower**。
随机曲线和专项用例的最终最大距离误差为 `1.12e-16 m`；
这只是当前独立预期值测试矩阵的结果，不是任意网格/任意坐标动态范围的误差定理。
真实 DEM oracle 独立实现 long-double 平面求根和面内判定，但复用 BVH 包围盒阶段，
所以不能将其表述为完全独立的几何软件互证。

## 4. 完整输运验收与剩余门禁

完整矩阵为 24 个配置加 2 次 CUDA 重复：

- IGRF14/2027：CPU、OpenMP、CUDA 各六种初级/方向配置。
  光子出山/入山、岩石内电子、正电子为 1 GeV；自然与强制 CC νe 为 10⁴ GeV。
- 零场：三条路径各运行光子出山和岩石电子两例，与上轮逐字节回归。
- CUDA 重复：带磁场光子出山与强制 CC 两例。
- seed=67101，21CMA 原 DEM/天线场景不变；空气有局地均匀场，岩石场为零。
  原始 air/rock PROPOSAL 表与辅助缓存复用，没有重新制表。

本节最终运行完整性、严格比较与资源结果以 `build/mountain-validation-20260909-curved/summary.json` 为准。
本轮全部 26 次运行完成，共 **1,008,349 条轨迹记录**，没有非有限数、材料错分、
CSV 截断、未回收粒子或异常退出。六份零场轨迹与上轮逐字节相同。
CUDA 重复的光子出山 **1489 行**和强制 CC **251819 行**也逐字节相同。
两 Kokkos 后端及修复前后的输入/物理 bank 哈希逐项检查通过。
标量参考以本轮 CUDA-capable 二进制的 `--em-backend proposal` 分支运行，不初始化 Kokkos 输运；
另有独立 OpenMP 构建的环境单测，以及完全不依赖 Kokkos 的 host-only 几何单测。

运行器设置 128 MiB 应用分配预算；实测完整输运的最大设备用量增量 **441 MiB**、
最大子进程 RSS **730.6 MiB**，最小系统可用内存 **5.47 GiB**。
512 MiB/显存总量 10% 的较小增量上限、2 GiB RSS 上限、3 GiB 可用内存余量均未触发。
WSL 设备用量增量是相对运行前全设备基线的采样值，不是精确的进程显存计量，
也不构成长期或 `N>1` 的内存无泄漏证明。

汇总脚本要求所有配置完成、CSV 不截断、轨迹有限、步数/材料计数一致、无待回收粒子，
并检查零场轨迹未变、CUDA 同种子重复一致。任何缺失或门禁失败均拒绝生成通过汇总。
**跨后端严格浮点门禁独立报告，不以“完整运行通过”替代它。**

六对 OpenMP/CUDA 的逐行 PID、介质、行数一致，但并没有因此证明 history/过程/random draws 全部一致。
现有严格门限仍为 `rtol=1e-10, atol=1e-12`，未放宽。

| 配置 | 首次数值差异 | 首次超严格门限 |
|---|---|---|
| photon_up | 第 3 行 `x1_m` | 第 165 行 `x1_m` |
| photon_down | 第 3 行 `y1_m` | 第 125 行 `nx0` |
| electron_rock | 第 1 行 `E1_GeV` | 第 25 行 `ny1` |
| positron_rock | 第 1 行 `nx1` | 第 63 行 `nx1` |
| nue_natural | 无 | 无 |
| nue_forced | 第 89 行 `x1_m` | 第 373 行 `nx1` |

这些首次差异行号/字段与上轮相同；本轮几何修复没有新增第一次分叉位置，
也没有消除既有能损/散射和长轨迹差异。四个 EM 小例的最大单坐标差约
`2.50e-11–7.43e-9 m`，强制 CC 长轨迹最大单坐标差约 **0.4565 mm**。
不能据此宣称差异已在统计意义上无害，仍需要后续过程级 replay 和系综验收。

仍未完成的生产验收：

1. 能损/散射与长轨迹的完整中间量 replay；既有严格 OpenMP/CUDA 逐行浮点门禁仍非全部通过，未放宽容差。
2. 真实非凸 DEM 的岩气跨界 CoREAS/ZHS。本应用仍明确写入 `radio: disabled`，没有射电波形验收结论。
3. 强子静质量在内的完整能量台账、多种子物理系综与生产性能。
4. 更广泛的曲线共边、尖角、极端尺度/近切线对抗性测试；补偿 binary64 不是任意精度精确谓词，
   面编号约定也不构成边缘衍射模型或任意尖角处材料转换的证明。

## 5. 复现与证据

所有本轮证据在项目容器目录的 `build/mountain-validation-20260909-curved/`。
修改前山体二进制、初始失败日志和中间 DEM 失败报告均保留，没有覆盖历史数据。
`host_sanitizer_guard/` 保存 sanitizer 日志和资源采样；`source_sha256.txt` 记录本轮关键源码哈希。
`acceptance.json` 保存实际命令、二进制哈希、结束状态和资源采样。
探针的时间包含 staging/传输，不能标为 kernel 时间；小样本运行也不是生产加速比测量。

在源码目录：

```bash
conda activate corsika_venv
cmake --build ../build/mountain-openmp --target testTerrainCurvedBoundary c8_terrain_curved_probe --parallel 1
cmake --build ../build/mountain-cuda --target testTerrainCurvedBoundary c8_terrain_curved_probe --parallel 1
OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  ctest --test-dir ../build/mountain-openmp \
  -R '^(testTerrainEnvironment|testKokkosTerrainSession|testTerrainCurvedBoundary)$' --output-on-failure
python validation/terrain/summarize_curved_transport.py \
  --root ../build/mountain-validation-20260909-curved \
  --previous ../build/mountain-validation-20260909-indexed --check-only
```

真实 DEM 探针使用原场景中的 mesh 文件，不下载新地形：

```bash
../build/mountain-cuda/applications/c8_terrain_curved_probe \
  --mesh /path/to/terrain_enu.ply --output /path/to/new_report.yaml \
  --samples 100000 --threads 1 --shared-vertex-id 31842
```

最后一个参数只适用于此次原点共顶点回归夹具；其他地形不应照抄顶点编号。
局部 GPU 测试继续使用 `run_guarded_diagnostic.py` / `run_multimaterial_acceptance.py` 的资源保护，
而非在生产 GPU 任务旁无保护地运行。
