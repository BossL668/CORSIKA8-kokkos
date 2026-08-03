# 阶段 38：Molière 混合介质初值表与热缓存

## 1. 本阶段结论

阶段 37 后，lepton transport 中最大的单项热点是 Molière 多重散射：

```text
1e17 eV: 约 320--338 ms
1e18 eV: 约 473--504 ms
```

每个带电粒子步需要对两个投影角分别反演 Molière 累积分布。阶段 37
使用只依赖参考组分 \(B\) 的二维初值表，最终角度仍由完整混合介质
Newton 方程求解。生产 shower 中两个投影合计平均需要：

```text
4.29--4.41 次 Newton 分布求值
```

本阶段实现：

1. 将初值表扩展为快照专用的
   \(B_\mathrm{ref}\times\beta^2\times q\) 三维表；
2. 初值表显式包含空气中全部介质组分，而不是只近似最大权重组分；
3. 表只提供 Newton 初值，最终 PDF/CDF、组分顺序和随机数完全不变；
4. 将设备方向单位向量检查从 `sqrt(norm2)` 改成严格等价的平方区间检查；
5. 将三维辅助表写入带版本、快照、长度和校验和的磁盘热缓存；
6. 将 Newton 相对停止阈值由 \(10^{-4}\) 收紧为 \(5\times10^{-5}\)，使扩展
   随机 oracle 的最大 GPU/PROPOSAL 相对差保持在 \(10^{-5}\) 内；
7. 增加冷缓存、热缓存和损坏缓存自动重建测试。

最终四个 UHE 样本的平均两轴 Newton 次数下降 18.6%--19.6%。按 Molière
trial 数归一化的 kernel 耗时下降约 1.2%--7.3%。24 个 GPU 测试全部
通过，同一 GPU、配置和 seed 的重复运行 9 个科学文件逐字节相同。

阶段 38 会改变相对于阶段 37 的浮点求根路径，因此不要求阶段 37 与
阶段 38 的同 seed shower 逐字节相同。一个 \(10^{18}\,\mathrm{eV}\)
样本出现了典型的级联混沌分叉；这意味着最终的 \(X_{\max}\)、地面和
射电“均值小于 1%”验收必须使用计划中的大样本分布，不能由四个配对
seed 代替。本阶段已经完成点级物理 oracle 和确定性回归，但没有声称
完成最终大样本 shower 物理验收。

## 2. 原热点的数据流

每个 lepton transport record 完成连续能损和几何推进后，独立 kernel
执行：

```text
buildDistribution
  -> 计算 momentum^2、beta^2、chi_c^2
  -> 对每个空气组分求 B_i
  -> 构造 inverse_scale、inverse_B、density_weight

sampleOneDimensional(u1)
  -> normalPpf
  -> 二维旧初值表
  -> Newton: 同时求混合介质 density 和 cumulative

sampleOneDimensional(u2)
  -> 同上

applyMoliereDirection
  -> 两次 AngleAxis 旋转
  -> 输出方向归一化
```

一次 Newton 分布求值对每个介质组分执行：

```text
exp(-x)
erf(sqrt(x))
f1 插值
f2 插值
F1 插值
F2 插值
```

所以减少一次 Newton 迭代比优化只执行一次的方向旋转更有价值。

## 3. 混合介质无量纲关系

Molière 参数满足：

\[
B_i-\log B_i
=
\log\frac{\chi_c^2}{\chi_{a,i}^2}
+1-2\gamma_E.
\]

屏蔽角可以写成：

\[
\chi_{a,i}^2
=
\frac{\chi_{0,i}^2}{p^2}
\left(
1.13+\frac{c_i}{\beta^2}
\right).
\]

选取最大权重组分 \(r\) 为参考组分。给定
\(B_\mathrm{ref}=B_r\) 和 \(\beta^2\)，其他组分的方程右端不需要
grammage、绝对动量或能量：

\[
\mathrm{rhs}_i
=
B_\mathrm{ref}-\log B_\mathrm{ref}
+
\log
\frac{
\chi_{0,r}^2
\left(1.13+c_r/\beta^2\right)
}{
\chi_{0,i}^2
\left(1.13+c_i/\beta^2\right)
}.
\]

再用设备和 host 共用的 `solveMoliereB()` 求 \(B_i\)。

定义无量纲投影坐标：

\[
\theta
=
\sqrt{\chi_c^2 B_\mathrm{ref}}\,q.
\]

则每个组分的 Molière 自变量为：

\[
x_i
=
\frac{\theta^2}{\chi_c^2B_i}
=
q^2\frac{B_\mathrm{ref}}{B_i}.
\]

因此，固定介质快照的完整混合介质 CDF 只依赖：

```text
B_ref
beta^2
q
```

这正是三维初值表的坐标。表格生成时对每个节点反演完整组分 CDF，保存：

\[
\Delta q
=
q_\mathrm{Moliere}-q_\mathrm{Gaussian}.
\]

运行时做三线性插值并得到：

\[
\theta_0
=
\sqrt{\chi_c^2 B_\mathrm{ref}}
\left(q_\mathrm{Gaussian}+\Delta q\right).
\]

`theta_0` 只是起点。最终返回值仍由 `evaluateDistribution()` 的完整
混合介质 PDF/CDF Newton 解决定。

## 4. 表结构和网格

最终网格为：

```text
B_ref nodes       64, range [4.5, 64]
beta^2 nodes       8, range [0.25, 1]
coordinate nodes 128, range [0, 5]
```

初值数组包含：

```text
64 * 8 * 128 = 65536 doubles
```

即 512 KiB。加上四组 Molière cubic polynomial 后，生产缓存文件为：

```text
539032 bytes
```

三维数组按下列顺序连续存放：

```cpp
((b_index * beta_node_count + beta_index)
 * coordinate_node_count + coordinate_index)
```

设备端一次查询读取相邻的：

```text
2 个 B 节点
2 个 beta^2 节点
2 个 q 节点
```

共 8 个 double，再按 `q -> beta^2 -> B` 顺序插值。

超出表范围时回退到 Gaussian 初值，不跳过散射，也不改变最终 Newton
方程。

## 5. 表生成

`MoliereInterpolation.cpp` 首先为每个
\((B_\mathrm{ref},\beta^2)\) 节点构造全部 \(B_i\)。对每个正
`q` 节点：

1. 计算目标 Gaussian CDF；
2. 扩大上界直到混合 Molière CDF 覆盖目标；
3. 执行 24 次单调二分；
4. 保存反演坐标与 Gaussian 坐标的差。

二分只在初始化或缓存重建时执行，不在 shower kernel 中执行。

开发中测试过用解析 density 做带括区 Newton 的 host 表生成。虽然生成
更快，但部分已经到根的节点在更新括区后被误退到中点，测试平均迭代从
3.60 恶化到 6.88。该实现已完全撤回。最终使用更简单、单调且容易验证
的二分生成器。

还测试过更小的：

```text
32 * 8 * 96
```

网格。生成时间降到 0.40 s，但极端测试平均 Newton 次数从旧版 4.11
恶化到 4.73，因此否决。

最终 `64 * 8 * 128` 网格与最初的 `64 * 16 * 128` 网格得到相同的
3.60 次小样本迭代均值，而生成时间从约 1.98 s 降到约 0.68 s。

## 6. 磁盘热缓存

生产缓存路径由 rate table 路径派生：

```text
<rate-table>.moliere-initial-v1.c8cache
```

当前构建中的实际文件为：

```text
gpu_em_tables/
  production_v9_1e-3_1EeV.c8emrt.moliere-initial-v1.c8cache
```

缓存 header 保存：

```text
8-byte magic
format version
sizeof(MoliereSnapshot)
sizeof(MoliereInterpolationTable)
FNV-1a table checksum
完整 pointer-free MoliereSnapshot
```

加载时依次检查：

```text
文件长度
magic
format version
snapshot/table 字节数
快照逐字节一致
table checksum
所有 polynomial 和 delta 都为 finite
```

任一检查失败时，缓存不会参与模拟；代码重新生成完整表并通过临时文件
加 `rename` 替换旧缓存。

缓存只影响 Newton 初值，所以缺失或损坏时可以安全重建。PROPOSAL rate
table 的版本、content hash 和误差检查仍由原有严格路径负责。

自动测试覆盖：

```text
冷缓存创建
热缓存加载后表逐字节相同
篡改最后一个字节
checksum 拒绝损坏缓存
重建后表逐字节恢复
```

代表性缓存测试：

```text
cold process wall  5.60 s
hot process wall   5.14 s
```

冷热两次同 seed 的 9 个科学输出逐字节相同。

## 7. Newton 停止准则

扩展到 4096 个确定性随机分位点后，原 \(10^{-4}\) 停止准则的最大
GPU/PROPOSAL 相对差为：

```text
1.00124e-5
```

它只超过内部 \(10^{-5}\) guard 约 0.124%，但本阶段没有直接放宽
测试。最终将相对步长停止准则收紧为：

\[
\left|
\frac{\theta_{n+1}-\theta_n}{\theta_{n+1}}
\right|
\le 5\times10^{-5}.
\]

最终扩展 oracle：

```text
checks                             4557
deflected samples                  3342
max GPU/CPU interpolated error     9.83364e-6
max analytical/interpolated error  3.48269e-2
mean two-axis Newton iterations    3.47786
max two-axis Newton iterations     5
```

`3.48269e-2` 是 PROPOSAL 自身 analytical Moliere 与生产
MoliereInterpol 在尾部的参数化差异；真正的 GPU/CPU production
对照是 `9.83364e-6`。

测试现在额外断言：

```text
mean two-axis iterations <= 3.6
max two-axis iterations  <= 6
GPU/CPU relative error   <= 1e-5
```

## 8. 方向旋转的小优化

旧方向输入检查为：

```cpp
norm = sqrt(x*x + y*y + z*z);
abs(norm - 1) <= 1e-12;
```

最终改成数学等价的平方区间：

```cpp
norm2 = x*x + y*y + z*z;
(1 - 1e-12)^2 <= norm2 &&
norm2 <= (1 + 1e-12)^2;
```

这样每个实际偏转步少一次 double `sqrt`，输出方向计算和最终归一化
不变。单独固定 seed 的 9 个科学文件与阶段 37 逐字节相同。

## 9. 被否决的设备优化

### 9.1 CUDA `sincos`

将两组独立 `sin/cos` 改成两次 CUDA `sincos` 后，科学输出逐字节相同，
但 kernel registers 从 74 增加到 77，Molière 时间没有下降。已撤回。

### 9.2 32 threads/block

将 Molière block 从 64 改成 32 threads 后：

```text
64 threads: 约 338--352 ms
32 threads: 388 ms
```

小 block 增加调度开销，已撤回。

### 9.3 预存 inverse scale root

曾把每次 Newton 的 `sqrt(x)` 改成预存
\(1/\sqrt{\chi_c^2B_i}\)。它把 registers 从 74 增到 75，实测
Molière 没有改善，而且改变射电和地面文件的浮点末位。已撤回。

### 9.4 强制最少两次 Newton

为减少初值依赖，曾要求每个投影至少两次 Newton。seed 33019 的宏观
shower 分叉仍然存在，说明分叉不是求根不够精确，而是级联对任意微小
方向扰动敏感。额外迭代不能解决统计验收问题，已撤回。

## 10. Kernel 资源

RTX 4060 Laptop、sm_89 的最终资源：

```text
applyMoliereScatteringKernel<4>
  registers/thread   73
  stack bytes         0
  local bytes         0
  shared bytes        0
  constant[0]      2616
  constant[2]       720
```

阶段 37 为 74 registers/thread。本阶段虽然增加三线性初值查询和
`beta_squared` 状态，但编译器最终少使用 1 个 register，没有 local
spill。

三维表使设备 table memory 增加 458752 bytes：

```text
1e17 peak device bytes  689492958
1e18 peak device bytes  823710686
```

增量小于 0.5 MiB，队列容量不变。

## 11. 四组 UHE 性能

配置：

```text
GPU                 RTX 4060 Laptop, sm_89
CUDA                12.6
primary             electron
energy              1e17 / 1e18 eV
EM thinning         1e-3
maximum weight      1e6
gpu-min-batch       64
radio               CUDA CoREAS + ZHS
stage timing        enabled
Moliere cache       hot
```

| 能量/seed | 阶段 | mean iterations | Molière ms | transport ms | all CUDA EM kernels ms |
|---|---:|---:|---:|---:|---:|
| \(10^{17}\), 32017 | 37 | 4.287930 | 337.671 | 630.806 | 1466.252 |
| \(10^{17}\), 32017 | 38 | 3.490011 | 333.477 | 629.998 | 1497.167 |
| \(10^{17}\), 32018 | 37 | 4.289799 | 320.448 | 586.798 | 1397.984 |
| \(10^{17}\), 32018 | 38 | 3.490798 | 300.711 | 583.430 | 1410.389 |
| \(10^{18}\), 33018 | 37 | 4.406035 | 473.494 | 855.789 | 2133.706 |
| \(10^{18}\), 33018 | 38 | 3.542867 | 455.605 | 843.531 | 2157.653 |
| \(10^{18}\), 33019 | 37 | 4.407936 | 504.423 | 905.804 | 2277.207 |
| \(10^{18}\), 33019 | 38 | 3.543357 | 464.228 | 857.900 | 2198.048 |

相对变化：

| 能量/seed | iterations | Molière | transport | all CUDA EM kernels |
|---|---:|---:|---:|---:|
| \(10^{17}\), 32017 | -18.61% | -1.24% | -0.13% | +2.11% |
| \(10^{17}\), 32018 | -18.63% | -6.16% | -0.57% | +0.89% |
| \(10^{18}\), 33018 | -19.59% | -3.78% | -1.43% | +1.12% |
| \(10^{18}\), 33019 | -19.61% | -7.97% | -5.29% | -3.48% |

前三个事件的 Molière trial 数与阶段 37 完全相同。归一化耗时为：

```text
1e17 seed 32017  372.280 -> 367.656 ns/trial
1e17 seed 32018  347.144 -> 325.762 ns/trial
1e18 seed 33018  242.832 -> 233.657 ns/trial
```

第四个事件在后续级联分叉后 trial 数略有不同；按各自 trial 归一化约为：

```text
238.213 -> 220.783 ns/trial
```

四个样本稳定证明 Newton 次数下降。Molière 局部计时也全部同向改善。
总 kernel/total-run 包含不同级联路径和 WSL 调度噪声，四个样本不足以
证明稳定端到端加速，因此不作该声明。

## 12. 物理、错误和确定性检查

最终：

```text
ctest --output-on-failure -I 3,26 -j1

24/24 passed
0 failed
7.56 s
```

四个 UHE 样本均：

```text
queue_overflows                 0
cross_species.host_spills       0
profile.fixed_point_overflows   0
profile.invalid_records         0
radio.fixed_point_overflows     0
epair cpu_fallbacks             0
epair envelope_violations       0
```

最终 seed 32017 独立重复运行的下列文件 9/9 逐字节相同：

```text
CoREAS/observers.parquet
ZHS/observers.parquet
energyloss/dEdX.parquet
interaction_hist/inthist_cms_1.npz
interaction_hist/inthist_lab_1.npz
interactions/interactions.parquet
particles/particles.parquet
production_profile/profile.parquet
profile/profile.parquet
```

阶段 37 与阶段 38 之间不逐字节相同，这是预期的：新的初值改变 Newton
达到停止阈值的迭代路径。单个 seed 的 shower 可能完全相关、轻微变化，
也可能因 crossing thinning/branching threshold 而去相关。

在四个配对样本中，三个事件的主要科学量近乎不变；seed 33019 出现宏观
分叉，例如射电峰值有几十个百分点的单事件差异。该观察既不能证明偏差，
也不能证明无偏。正确验收必须比较计划中的独立事件分布：

```text
Xmax
N_e(X), N_gamma(X)
dE/dX
ground spectra and lateral distribution
arrival time
CoREAS/ZHS waveform statistics
```

并要求关键均值偏差小于 1%。该大样本矩阵仍是总目标的未完成项。

## 13. 主要代码位置

```text
corsika/gpu/em/MoliereScattering.hpp
  三维表类型
  三线性查询
  beta^2 distribution state
  5e-5 Newton stop
  direction norm-squared check
  cache API

src/gpu/em/MoliereInterpolation.cpp
  混合介质 B_i 推导
  无量纲混合 CDF
  24 次二分表生成
  cache read/write/checksum/rebuild

src/gpu/em/CudaEmBackend.cu
  从 rate-table 路径派生 cache
  cache load/rebuild
  device upload 和统计字节数

tests/gpu/testGpuMoliere.cu
  4096 个确定性随机分位点
  边界和尾部 grid
  iteration performance guard
  cold/hot/corrupt cache test
```

## 14. 下一步

阶段 38 后，下一阶段不应继续修改 Molière 求根语义。优先事项为：

1. 建立 CPU PROPOSAL 与 CUDA 的大样本 shower distribution harness；
2. 完成计划中的 1 TeV、1 PeV、\(10^{17}\) 和 \(10^{18}\) eV 统计矩阵；
3. 在物理统计通过后，用 Nsight Compute 或等价硬件计数器检查
   Molière interpolation cache hit、special-function latency 和 occupancy；
4. 转向 transport record 读写、radio/profile 重复读取及 endpoint
   compaction，而不是继续压缩已低于 3.6 次的 Newton；
5. 最终按原始实施计划逐条审计过程覆盖、CPU fallback、输出元数据、
   错误策略和端到端 5 倍加速。

本阶段是局部高性能优化，不代表整个 CORSIKA 8 CUDA 重构已经完成。
