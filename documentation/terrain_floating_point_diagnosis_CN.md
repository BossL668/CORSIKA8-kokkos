# 山体 OpenMP/CUDA 严格浮点差异：首次分叉诊断

2026-09-08。仅诊断，不修改输运/物理公式、生产编译参数或验收门限。
输入为 `build/mountain-validation-20260908/magnetic_{openmp,cuda}_admission`。

2026-09-09 后续：已实现共享顶点索引与补偿求交的独立候选，见
[索引几何候选验收](terrain_indexed_boundary_validation_CN.md)。
已解决独立探针中的顶点归属分歧，但尚未接入磁场中的完整山体输运；
本页记录的生产/完整轨迹限制仍有效。

后续接线与回归见[2026-09-09 索引输运验收](terrain_indexed_transport_acceptance_CN.md)：
原共享顶点的第一次直线分叉已消除，后续能损/散射和长曲线的严格浮点差异仍显式保留。

## 1. 先澄清比较对象和判据

之前的 `strict_trace_passed=false` 比较的是 **Kokkos-OpenMP 与 Kokkos-CUDA**，
不是原标量 CPU PROPOSAL 与 CUDA 的同随机流比较。
两个 Kokkos 后端共享物理源码和表；同设备 CUDA 重复的 251819 行轨迹逐字节一致。

`compare_multimaterial_runs.py` 有两种判据：

- 各轨迹浮点字段使用 `rtol=1e-10, atol=1e-12`；m、s、GeV、方向分量共用这一数值门限。
- 整个 diagnostics 字典还要求 Python **精确相等**，其中包括浮点沉积总和。

因此总判定失败并不自动说明物理过程不同；但本轮确实也有轨迹字段超限，
不能只把字典比较改成近似相等就宣称问题解决。
step/PID/material 顺序相同也不证明每一次 process、component、v 全部相同，
现有 CSV 没有这些逐过程字段。

## 2. 第一处分叉有可复现的因果证据

`photon_up` 的第 1 步，1 GeV 光子从岩内 `(0,0,-0.01 m)` 沿 +z 到岩气面：

| 数值 | OpenMP | CUDA |
|---|---:|---:|
| 末端 z / m | −6.591949208711867e−17 | −1.1622647289044608e−16 |
| 末端时间 / s | 3.3356409519814989e−11 | 3.3356409519814821e−11 |

此步能量、方向不变；光子无磁偏转，岩内磁场为零。因此最早差异不是磁场、
连续能损、散射或末态算法引起，而是几何边界距离运算。

独立探针 `validation/terrain/probe_boundary_roundoff.cu` 使用相同 PLY 和共享
`TerrainBoundary.hpp`/`FlatTerrain.hpp`，只以一个包围叶节点排除 BVH 遍历干扰。
它不调用 PROPOSAL、不抽随机数、不链接山体应用。

默认 nvcc：

```text
inside=1 host_distance=0.0099999999999999343
         device_distance=0.009999999999999884
         delta=-5.0306980803327406e-17
         host_face=62909 device_face=63448
```

只在探针加入 `--fmad=false`：

```text
inside=1 host_distance=0.0099999999999999343
         device_distance=0.0099999999999999343
         delta=0 host_face=62909 device_face=62909
inside=0 host_distance=0.0099999999999998857
         device_distance=0.0099999999999998857
         delta=0 host_face=62910 device_face=62910
```

第一组默认 GPU 距离完整复现了 shower 的首次出山差异。
入山默认探针与完整 BVH 的末位结果不完全相同，也不能声称它复现了所有遍历选择；
关闭 FMA 后两个探针方向均与 host 一致。

实际 OpenMP 编译为 `-O3 -std=c++17 -fopenmp`，CUDA 为 `-O3 -std=c++17`、sm_89，
均未开启 fast-math。**不启用 fast-math 不等于关闭 FMA**。
点积/叉积包含乘加、乘减，融合与分步舍入可以给出不同末位。
本实验确认了这一具体分叉的来源，不证明关闭 FMA 就能消除整 shower 的全部差异。

## 3. 同时发现了 DEM 顶点归属问题

射线正好穿过顶点 31842，坐标 `(0,0,0)`。面 62909、62910、63448 共享该顶点。
当前 `intersect()` 用严格 `d < best_distance` 选择最近面，面内判断也直接使用浮点重心坐标。
在数学上的相同交点，末位距离/重心差可以选择不同面。

这些面的单位法向并不一样，例如：

```text
62909: (-0.0876249263, -0.0644447126, 0.9940667741)
63448: (-0.0870846585, -0.1280538639, 0.9879364707)
```

本例仍切换到同一个空气材料，距离差约 5e-17 m，没有观察到材料错配。
但在未来 Snell/Fresnel 射电处理中，选中的面会决定法向；因此这里需要建立
**共同边/顶点的稳健归属和确定性 tie-breaking 规则**，并定义非光滑边界的法向语义。
不能仅依赖精确浮点距离排序，也不能靠任意放大几何容差掩盖问题。
这属于生产级几何鲁棒性待办，不应一概称为无害舍入。

## 4. 后续误差与尚未证明的部分

电子事例第 1 步的终能差仅 `7.77e-16 GeV`（约 7 ULP），方向分量差约 1–2 ULP，
说明即使不跨界，连续能损/散射计算也会出现最后几位的差异。
源码确实包含 native range 求值/反演、Molière 求根和三角函数，但尚未对这一电子步骤
做完整中间变量 replay，所以不能将所有后续偏差直接归因于某一个函数。

强制 CC 事例：首次非逐位差异在第 89 行的空气光子位置/飞行时间；
首次超过原门限在第 373 行方向字段。
最大坐标分量差 `4.565e-4 m` 出现在第 229414 行、约 110 km 高度的空气光子末端。
该步开始前位置与方向已经不同，不能解释为这一处 DEM 求交发生了 0.46 mm 错误。
长距离传播会将角度差转换为位置差，级联会继承上游误差；
但目前还没有逐 history/逐过程记录来完整量化每条误差传播链。

另外，旧 CPU 单位常数与设备磁偏转换算的 `1.89e-7` 差异属于
**标量 CPU 对设备**的问题；OpenMP 和 CUDA 这里使用同一设备表达式，
所以该常数差不能用来解释本轮 Kokkos 后端之间的分叉。

## 5. 建议的处理顺序

1. 单独修复/验收 DEM 边与顶点的稳定归属：面内、共同边、共同顶点、切向、
   不同 BVH 遍历与 FMA 模式；显式测试返回面法向，不只比较距离。
2. 为测试构建增加逐 history/step 的过程、组分、随机 key、v、连续损失和散射前后状态。
   对同一输入做单步 CPU/Kokkos oracle，分开局部误差与上游继承误差。
3. 保留严格 ULP/replay 诊断；生产物理判据再按量纲与观测量定义，
   不用“所有字段都给 1e-12 绝对误差”代替能量、几何、时间与射电相位的误差预算。
4. 不能据此直接宣称系综或跨界射电已通过；也不建议未经性能/物理回归就在生产中
   全局关闭 FMA。此次只编译独立小探针，生产二进制完全未改。

复现（源码目录、`corsika_venv`）：

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -I. \
  validation/terrain/probe_boundary_roundoff.cu -o /tmp/terrain-roundoff
nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false -I. \
  validation/terrain/probe_boundary_roundoff.cu -o /tmp/terrain-roundoff-nofma
# 参数为此前验证的 64694 顶点 / 129384 面二进制 PLY；探针限定此布局。
/tmp/terrain-roundoff /path/to/terrain_enu.ply
/tmp/terrain-roundoff-nofma /path/to/terrain_enu.ply
```
