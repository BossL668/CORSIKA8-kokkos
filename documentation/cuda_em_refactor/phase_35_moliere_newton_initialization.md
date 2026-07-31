# 阶段 35：Molière 多重散射数值重构与 Newton 初值加速

## 1. 本阶段结论

阶段 34 消除 Epair 末态中的逐粒子 Romberg 积分后，lepton pipeline 的最大
单项热点变成 Molière 多重散射。代表性的
\(10^{17}\,\mathrm{eV}\)、seed 32017 基线为：

```text
lepton transport                    1135.350 ms
其中 Molière                         840.862 ms
CUDA EM kernel total                2266.247 ms
HybridCascade::run()                2512.379 ms
```

本阶段没有更换散射物理模型，也没有使用 `fast-math` 或混合精度。最终仍由
完整多介质组分 Molière PDF/CDF 和 Newton 迭代给出散射角；新增的二维表只
产生 Newton 初值，不直接产生最终随机变量。

保留的优化包括：

1. 预计算每个介质组分在一次 step 内不变的比例因子；
2. 在一次组分遍历中同时计算 PDF 和 CDF；
3. 在环境快照中预计算静态 screening 因子；
4. 用 Lambert \(W_{-1}\) 的渐近式和一次 Halley 更新求 Molière \(B\)；
5. 删除只用于初值的 Acklam quantile 二次 Halley 修正；
6. 建立 \(64\times128\) 的单组分逆分布初值修正表；
7. 显式处理 \(u=0.5\) 的精确中位数；
8. 增加每个轨迹的 Newton 次数、总次数和最大次数诊断。

四组 UHE shower 中：

- Molière 时间下降 62.4%–63.3%；
- lepton transport 时间下降 47.0%–49.9%；
- CUDA EM kernel 总时间下降 22.6%–25.8%；
- `HybridCascade::run()` 下降 19.9%–24.3%；
- GPU/CPU production Molière 逐点最大相对误差为
  \(2.45918\times10^{-6}\)；
- 24 个 GPU 测试全部通过；
- 相同 GPU、配置和 seed 的 9 个 Parquet/NPZ 科研输出逐字节相同；
- 未出现 Epair fallback、Epair envelope violation、GPU queue overflow 或
  radio fixed-point overflow。

本阶段只是性能与单元物理验收，不代替计划中的多 shower 统计物理验收。

## 2. Molière 抽样在输运中的位置

对每个 \(e^-/e^+\) step，GPU lepton transport 先确定最近限制：

```text
离散相互作用 / 连续步长 / 层边界 / 磁偏转步长 / 观测面 / cut
```

得到实际 grammage 后，多重散射计算：

```text
grammage + step 起始能量
  -> 各空气组分的 chi_a²、chi_c² 和 B
  -> 一维投影角 theta_x 的逆 CDF
  -> 一维投影角 theta_y 的逆 CDF
  -> sqrt(theta_x² + theta_y²)
  -> 独立方位角
  -> 更新粒子方向
```

相关入口位于：

```text
corsika/gpu/em/MoliereScattering.hpp
src/gpu/em/MoliereInterpolation.cpp
src/gpu/em/CudaLeptonTransport.cu
src/gpu/em/CudaLeptonSelectionTransport.cu
src/gpu/em/CudaProfileProjection.cu
src/gpu/em/CudaEmBackend.cu
tests/gpu/testGpuMoliere.cu
applications/c8_air_shower.cpp
```

主要设备函数为：

```cpp
sampleMoliereScatteringAngle2DForCapacity(...)
sampleOneDimensional(...)
evaluateDistribution(...)
applyMoliereDirection(...)
```

## 3. 原算法为什么昂贵

Molière 一维投影分布对每个介质组分 \(i\) 使用参数
\(\chi_{c}^{2}\)、\(B_i\) 和权重 \(w_i\)。定义：

\[
x_i=\frac{\theta^2}{\chi_c^2 B_i}.
\]

一维 PDF 可写为：

\[
f(\theta)=
\frac{1}{\sum_i w_i}
\sum_i
\frac{w_i}{\sqrt{\pi\chi_c^2B_i}}
\left[
e^{-x_i}+\frac{f_1(x_i)}{B_i}
+\frac{f_2(x_i)}{B_i^2}
\right].
\]

对应的有符号 CDF 为：

\[
F(\theta)=
\frac{\operatorname{sgn}\theta}{\sum_i w_i}
\sum_i w_i
\left[
\frac{1}{2}\operatorname{erf}\sqrt{x_i}
+\frac{1}{\sqrt{\pi}}
\left(
\frac{F_1(x_i)}{B_i}
+\frac{F_2(x_i)}{B_i^2}
\right)
\right].
\]

每次 Newton 更新需要：

\[
\theta_{n+1}=
\theta_n-\frac{F(\theta_n)-(u-1/2)}{f(\theta_n)}.
\]

原设备实现分别调用 PDF 和 CDF 函数，因而每次 Newton 对每个组分重复计算：

- \(\theta^2/(\chi_c^2B_i)\)；
- \(1/B_i\) 和 \(1/B_i^2\)；
- \(w_i/\sqrt{\pi\chi_c^2B_i}\)；
- 组分状态读取和循环控制。

每个 step 又要独立求两个投影角，所以这些开销会乘以：

```text
GPU lepton steps × 两个投影 × Newton 次数 × 空气组分数
```

在 UHE shower 中，Molière trial 数达到约
\(9.0\times10^5\)–\(2.1\times10^6\)，因此一个标量上不显眼的重复除法、
平方根或特殊函数会成为 GPU 主热点。

## 4. 一次 step 内不变量的预计算

`DistributionStateForCapacity` 现在为每个组分保存：

```cpp
double inverse_scale[];   // 1 / (chi_c² B_i)
double inverse_B[];       // 1 / B_i
double density_weight[];  // w_i / sqrt(pi chi_c² B_i)
double prefactor;         // sqrt(chi_c² B_ref)
```

`buildDistribution()` 每个 step 只构造一次这些值。随后
`evaluateDistribution()` 在同一个组分循环中同时累计 PDF 和 CDF：

```cpp
auto const x = theta_squared * state.inverse_scale[index];
auto const inverse_B = state.inverse_B[index];

density_result += ...;
cumulative_result += ...;
```

PDF 和 CDF 的求和顺序仍与原实现一致，没有对介质组分做近似合并。

## 5. 静态 screening 因子前移

下面两个量只依赖粒子种类、介质组分和已选 PROPOSAL 参数化：

```cpp
chi_0_squared_MeV2
coulomb_correction
```

它们现在由 `makeMoliereSnapshot()` 在 host 初始化阶段计算，放入
`MoliereComponentSnapshot`。设备端每个 step 只保留与能量和 grammage
有关的组合。

同样，原来 \(B\) 循环内部重复形成的对数项被整理为一次
`log(chi_c_squared / chi_a_squared)`。这些变换不改变公式，只改变计算位置。

## 6. Molière \(B\) 的快速求解

每个介质组分需要求解：

\[
B-\ln B=r,\qquad B>1.
\]

旧实现从 \(B=15\) 开始固定执行六次 Newton 更新。该方程的物理解为：

\[
B=-W_{-1}(-e^{-r}).
\]

`solveMoliereB()` 使用 \(W_{-1}\) 分支的四阶渐近展开构造初值，并执行一次
三阶 Halley 更新。适用域严格限制为：

\[
B\geq4.5,\qquad
r\geq4.5-\ln4.5.
\]

任何非有限结果或落在允许域之外的解都返回失败状态，不会静默夹断。
参数扫描中该方法与收敛解的相对差约为 \(10^{-12}\)。

## 7. Newton 初值表

### 7.1 初值坐标

对参考组分定义无量纲坐标：

\[
t=\frac{\theta}{\sqrt{\chi_c^2B_{\rm ref}}}.
\]

纯 Gaussian 初值来自：

\[
t_0=\frac{\Phi^{-1}(u)}{\sqrt{2}}.
\]

此前 Acklam 正态分位数近似后还执行一次包含 `erfc` 和 `exp` 的 Halley
修正。但这个分位数只是完整 Molière Newton 的初值，并不是最终随机变量。
Acklam 有理近似已经位于收敛域内，因此删除这次一次性修正。

### 7.2 表中存储什么

新表的坐标为：

```text
B_ref:  64 nodes,  [4.5, 64]
|t_0|: 128 nodes,  [0, 5]
```

表项不是散射角，而是：

\[
\Delta t(B,t_0)=
t_{\text{one-component Moliere}}-t_0.
\]

host 初始化时，对单组分 production 插值 CDF 使用最多 64 次二分构造该表。
设备端进行双线性插值：

\[
t_{\rm initial}=t_0+\Delta t.
\]

若 \(B\) 或 \(t_0\) 越界、表指针为空或插值结果非法，直接退回 Gaussian
初值。

### 7.3 为什么这不是物理查表近似

初值表之后仍执行：

```text
完整介质组分 PDF
+ 完整介质组分 CDF
+ Newton convergence test
```

因此表只减少迭代次数；它不决定最终分布，也不替换空气中氮、氧等组分的
混合。该表占：

\[
64\times128\times8=65536\ \text{bytes}.
\]

它由已经加载并校验的 versioned PROPOSAL Molière 系数在运行初始化阶段
派生，不改变磁盘 rate-table 格式。显存统计已计入这 65536 B。

## 8. 数值边界和诊断

### 8.1 精确中位数

分布关于零对称，因此 \(u=0.5\) 的精确结果是：

\[
\theta=0.
\]

显式返回零可以避免相对收敛判据
\((\theta_n-\theta_{n+1})/\theta_{n+1}\) 在零点除零。该路径不消耗 Newton
迭代。

### 8.2 每条轨迹的迭代计数

`LeptonTransportRecord` 记录：

```text
multiple_scattering_applied
multiple_scattering_status
multiple_scattering_iterations
```

summary 输出：

```yaml
moliere:
  trials: ...
  deflections: ...
  zero_deflections: ...
  newton_iterations: ...
  maximum_newton_iterations: ...
  mean_newton_iterations: ...
```

`CudaProfileProjection` 使用 warp reduction 后再原子累加总迭代次数。验证时
发现，若在具有数据依赖循环次数的 profile-bin 累积之后使用
`__activemask()`，独立线程调度会让这个纯诊断量出现低计数。最终实现要求
所有 lane 在分支前执行 `__ballot_sync()`，显式把 mask 传入归约。

修复后两次独立运行：

```text
Molière total iterations = 3,858,618 / 3,858,618
maximum iterations       = 8 / 8
9 Parquet/NPZ files      = byte-identical
```

这个问题从未改变粒子状态或波形，只影响 summary 中的迭代诊断。

## 9. 单元与逐点验证

`testGpuMoliere` 覆盖：

- \(u=0.5\) 精确中位数；
- \(10^{-10}\) 到 \(1-10^{-10}\) 的 15 组 uniform；
- 6 个能量；
- 5 个 grammage；
- 270 个实际发生散射的二维样本；
- host production `MoliereInterpol` 与 CUDA 结果比较；
- analytical Molière 与 production interpolation 的独立参数化差异监测；
- 非法输入、零 grammage、方向旋转和状态码。

当前结果：

```text
checks                                  455
deflected samples                       270
mean two-axis Newton iterations         4.11111
maximum two-axis Newton iterations      6
max GPU/CPU production relative error   2.45918e-6
```

analytical Molière 与 PROPOSAL production interpolation 在极端尾部可相差
约 0.0348。它们是两个不同的数值参考，因此该差异单独报告，不能混入
GPU/CPU production 一致性阈值。

## 10. 四组 UHE 性能验收

环境：

```text
GPU             NVIDIA GeForce RTX 4060 Laptop GPU
compute         capability 8.9
CUDA            driver/runtime 12.6
precision       double
fast-math       disabled
radio backend   CUDA CoREAS + ZHS
EM thinning     1e-3
max weight      1e6
GPU min batch   64
```

阶段 34 是已经使用 production Epair rejection、尚未进行本阶段 Molière
优化的直接基线。

| 能量 / seed | GPU 粒子数 | Molière ms（前→后） | transport ms（前→后） | kernel ms（前→后） | run ms（前→后） |
|---|---:|---:|---:|---:|---:|
| \(10^{17}\) / 32017 | 962,167 | 840.862 → 315.055 | 1135.350 → 601.519 | 2266.247 → 1731.072 | 2512.379 → 1900.814 |
| \(10^{17}\) / 32018 | 979,916 | 810.824 → 303.586 | 1083.810 → 566.729 | 2130.980 → 1649.338 | 2269.002 → 1816.942 |
| \(10^{18}\) / 33018 | 2,107,997 | 1210.051 → 454.885 | 1597.937 → 815.033 | 3216.691 → 2444.962 | 3318.377 → 2578.126 |
| \(10^{18}\) / 33019 | 2,294,524 | 1298.557 → 476.242 | 1710.549 → 856.591 | 3428.829 → 2545.800 | 3661.619 → 2816.011 |

四组的平均降幅：

```text
Molière                 62.71%
lepton transport        48.41%
CUDA EM kernel total    23.99%
HybridCascade run       22.42%
```

最终 UHE 运行中的 mean two-axis Newton iterations 为 4.29–4.41，最大均
为 8。

## 11. 科研输出比较与正确解释

阶段 34 与阶段 35 四个相同 seed 的稳定量比较：

- GPU 粒子数四组完全相同；
- radio track 数三组相同，一组相差 1；
- weighted deposit 相对变化为
  \(5.5\times10^{-11}\)–\(2.0\times10^{-10}\)；
- 三组纵向粒子 profile 逐 bin 完全相同；
- 这三组 CoREAS 波形相对 \(L_2\) 差异为 0.00125–0.00723；
- 这三组 ZHS 波形相对 \(L_2\) 差异为
  \(3.63\times10^{-5}\)–\(3.57\times10^{-4}\)。

seed 32017 的微小散射角变化触发了不同的后续 Monte Carlo 分支：

```text
纵向 profile relative L1       0.00064–0.00480
dEdX relative L1               0.00206
CoREAS waveform relative L2    0.1366
ZHS waveform relative L2       0.1255
```

这类同 seed 分支放大不等价于分布偏差。一个 shower 的射电波形对个别轨迹
和相干相位高度敏感；优化前后使用不同但都满足误差限的 Newton 终止路径时，
固定 seed 轨迹不必逐条对应。正确的生产验收是：

1. 单过程逆 CDF/PDF 逐点误差；
2. 大样本散射角分布检验；
3. 多 shower 的 \(X_{\max}\)、profile、ground 和波形统计量；
4. 关键均值偏差不超过既定 1% 阈值。

因此本阶段如实记录单 shower 差异，不把“同 seed 波形接近”作为物理正确性
的必要条件。当前四组只构成工程回归，不构成最终多 shower 统计验收。

## 12. CoREAS/ZHS 与粒子传播的时序

CPU CORSIKA 8 中，`RadioProcess::doContinuous(step)` 在每个
\(e^-/e^+\) step 形成后立即调用：

```cpp
implementation().simulate(step);
```

所以射电贡献与粒子传播在线累计。`endOfShower()` 只负责：

- 对 ZHS vector potential 做时间差分；
- 读取已经累计的 observer waveform；
- 写出 Parquet；
- reset observer。

CUDA 路径保持相同语义：

```text
resident lepton wavefront
  -> LeptonTransportRecord[] 仍在 GPU
  -> CudaRadioAccumulator::accumulateLeptonTracksOnDevice()
  -> CoREAS/ZHS device waveform 累积
  -> 下一 wavefront
```

只有 shower 结束时才把波形统一回传并写盘。它不是等待所有粒子完成后，再从
最终粒子列表重建射电信号。更完整的射电后端设计见
`phase_25_resident_gpu_radio.md`。

## 13. 当前剩余热点和下一阶段

Molière 已从主导项降到 0.30–0.48 s，新的主要候选热点为：

```text
lepton selection                0.20–0.37 s
vertex selection                0.24–0.32 s
final-state classification      0.17–0.25 s
endpoint compaction             0.17–0.22 s
```

下一阶段应优先分析：

1. endpoint compaction 的多次 scan/selection 是否可以融合；
2. vertex selection 中 rate-table 访问、PID/介质分支和 fallback 分类；
3. profile/radio stream 与主 pipeline 的剩余同步点；
4. 完成更大规模多 shower CPU/GPU 统计物理验收。

任何进一步优化都必须保留：

- double precision；
- Philox history-keyed random stream；
- 显式 CPU fallback；
- deterministic radio/profile accumulation；
- 无 NaN、非法 PID、负能量和静默过程丢失。
