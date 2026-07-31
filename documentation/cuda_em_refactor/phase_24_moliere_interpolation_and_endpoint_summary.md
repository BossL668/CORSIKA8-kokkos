# CUDA 电磁后端重构记录：阶段 24，Molière 生产插值与同步回读压缩

## 1. 本阶段结论

本阶段完成了三个互相关联的工作：

1. 修复 GPU inverse-CDF 尾部回退事件缺少能损 \(v\) 时被误当成完整
   PROPOSAL 末态的问题；
2. 把带电轻子输运中的 Molière 多重散射拆成独立 kernel，并把设备算法从
   70 阶解析多项式切换到与 CPU `PROPOSAL::MoliereInterpol` 相同的
   100 节点 cardinal cubic B-spline；
3. 把 photon/lepton endpoint 阶段的多个同步标量回读压缩成每个
   wavefront 一次连续 summary 回读。

当前结果不是最终的科研生产验收，但已经达到以下阶段性状态：

- 全部 23 个 GPU 测试通过；
- GPU 与 CPU `MoliereInterpol` 的 108 个有效散射样本最大相对差为
  \(3.63\times10^{-15}\)；
- 相同 GPU、相同种子下，1 TeV 和 1 PeV 的七类 Parquet 输出均逐字节重复；
- 1 PeV kernel 时间由正确性修复后的 22.66 s 降至约 9.44 s；
- 完整 1 PeV `c8_air_shower` 相对现有单核 CPU 基准约为 3.3 倍，而不是
  目标中的 5 倍。

最后一点必须准确解释：当前应用仍启用 photonuclear 及其强子末态。强子、
中子、质子和 μ 分支依法留在 CPU，因此这不是“只含电磁过程”的性能验收。

## 2. inverse-CDF 尾部回退的物理正确性修复

### 2.1 原问题

电子/正电子 pair-production 的 GPU 逆 CDF 表只覆盖有限分位区间。以当前
生产表为例，表内分位点约为：

```text
q in [1e-4, 0.98]
```

当设备端已经抽取了：

```text
process + target component + q
```

但 \(q\) 落在表格尾部时，旧事件记录中的 \(v\) 保持默认值 0。旧判断只检查
“过程和组分是否已指定”，随后把 \(v=0\) 直接送入
`SecondariesCalculator::CalculateSecondaries()`。这不是一个合法的
pair-production 末态，曾导致大量积分 NaN 日志。

### 2.2 新状态机

现在明确区分三种 CPU fallback：

```text
generic fallback
  -> CPU 重新执行完整标量输运

selected process/component/q, loss unresolved
  -> CPU ProposalRateProvider::sampleSelectedLoss()
  -> 得到精确 v
  -> 对指定过程和组分生成末态

selected process/component/v, fully resolved
  -> CPU 只生成指定末态
```

新增的关键接口包括：

- `ProposalRateProvider::sampleSelectedLoss(...)`
- `proposalFallbackRequiresSelectedLoss(...)`
- `hasSpecifiedProposalInteractionIdentity(...)`
- `hasResolvableProposalSelectedLoss(...)`
- `makeProposalInteractionRecordForSelectedLoss(...)`

输出统计新增：

```yaml
cpu_completed_selected_losses: 75
```

当前 1 PeV 样本中的 75 个 reason 8 尾部事件都得到精确补全，shower 完整
结束，未再出现由 \(v=0\) 触发的积分 NaN。

## 3. 为什么旧 GPU Molière 很慢

PROPOSAL 7.6.2 同时提供：

- `Moliere`：解析函数；
- `MoliereInterpol`：生产 CPU 路径使用的插值函数。

旧 GPU 实现逐式翻译了 `Moliere`。每次 Newton 迭代都要计算：

```text
density    -> f1M(x), f2M(x)
cumulative -> F1M(x), F2M(x)
```

在低 \(x\) 区域，每个函数都包含最高 70 项的 Horner 多项式。空气有多个
介质组分，一次二维散射还要分别反演两个一维角度，因此同一粒子会重复执行
大量 double 精度乘加、`pow`、`log` 和 `sqrt`。

把大气输运和 Molière 放在同一个 kernel 时，线程还必须同时保存几何、
grammage、磁场、连续能损和 Newton 状态。拆分前资源压力约为：

```text
transport kernel: about 198 registers, 360-byte local stack
```

拆出 `applyMoliereScatteringKernel`，并直接在输出槽上更新 record 后，
输运 kernel 降至约：

```text
transport kernel: 116 registers, 112-byte local stack
```

拆分保持了原 Philox 键：

```text
(seed, shower_id, history_id, step_id,
 ContinuousScatteringRandomProcessId, draw_id)
```

所以 kernel 边界变化不会改变某条 history 使用的随机数。

## 4. 精确复现 PROPOSAL `MoliereInterpol`

### 4.1 CPU 参考算法

PROPOSAL 7.6.2 对四个函数分别建立：

```cpp
LinAxis<double>(0., 20., size_t(100))
```

底层是 `CubicInterpolation::CubicSplines<double>`，它又使用 Boost
`cardinal_cubic_b_spline`。端点导数不是简单设为零，而是通过 Boost
六阶有限差分估计。

因此，下列替代都不能称为严格复现：

- 100 节点线性插值；
- 自然三次样条；
- 自行指定零端点导数；
- 直接对最终散射角做逆 CDF 表。

### 4.2 Host 端建表

新增：

```text
src/gpu/em/MoliereInterpolation.cpp
```

初始化过程使用项目链接的同一套 CubicInterpolation 构造四张 spline：

```text
f1M, f2M, F1M, F2M
```

每个区间的 spline 被精确重写为局部坐标 \(q\in[0,1]\) 上的 Hermite
三次多项式：

\[
p(q)=c_0+q(c_1+q(c_2+q c_3)).
\]

这里使用 spline 在区间两端的函数值和一阶导数，并没有对 spline 做第二次
拟合。设备端一次表内函数求值只需读取四个 double 系数并执行三层 Horner。

### 4.3 设备布局和边界行为

四个函数各有 99 个区间，每个区间四个 double：

```text
4 functions * 99 intervals * 4 coefficients * 8 bytes
= 12,672 bytes
```

这块连续显存通过 `MoliereInterpolationView` 传入散射 kernel。运行时规则为：

```text
0 <= x <= 20 -> cubic interpolation
x > 20       -> original analytical formula
```

这与 CPU `MoliereInterpol` 的区间外回退一致。表格完整性检查在采样入口执行
一次；经过 `CudaEmBackend::initialize()` 验证的生产路径不会在 Newton 热
循环中重复检查四个指针和静态边界。

显存统计由：

```text
15,598,552 bytes
```

增加到：

```text
15,611,224 bytes
```

差值正好为 12,672 bytes。该内存在初始化失败和正常析构时都会释放，并被
计入后端显存预算。

## 5. 数值验证

`testGpuMoliere` 使用相同 Air medium、electron definition、grammage、
能量和随机分位点，比较：

1. CPU `Moliere`；
2. CPU `MoliereInterpol`；
3. GPU cubic interpolation。

结果为：

```text
183 checks
108 deflected samples
max GPU/CPU interpolated relative error = 3.62791e-15
max interpolated/analytical difference  = 0.020001
```

第二个差异不是 GPU 误差，而是 PROPOSAL 自己的解析版与生产插值版差异。
GPU 现在跟随实际 CPU 生产路径。

完整 GPU 回归：

```text
23/23 tests passed
Cascade regression: 39 assertions in 4 test cases
PROPOSAL regression: 69 assertions in 1 test case
```

覆盖反应率、逆 CDF、interaction selection、pair/brems/epair 末态、LPM、
thinning、球形大气、磁场、Molière、resident photon/lepton 级联、CPU
fallback 和 Hybrid route。

## 6. endpoint summary：减少同步标量回读

### 6.1 原流程

Lepton endpoint 在三次 exclusive scan 后分别读取：

```text
next lepton count
generated photon count
observation count
endpoint error
```

即每个 resident lepton wavefront 有四次同步 D2H API。Photon endpoint
有三次。

### 6.2 新流程

新增一个单 block、单 thread 的设备汇总 kernel：

```text
last offset + last flag -> compact count
error flag              -> summary error
```

Lepton 写入连续 16-byte summary，Photon 写入连续 12-byte summary，host
各只回读一次。原 exclusive scan 和按 offset 写出的稳定顺序完全不变。

Nsight Systems 在当前 WSL 中仍不能记录 GPU kernel timeline，但 CUDA API
统计可用。相同 1 TeV 配置的调用计数为：

| API | 修改前 | 修改后 |
|---|---:|---:|
| `cudaMemcpy2D` | 4,123 | 2,061 |
| `cudaMemcpy` | 4,975 | 4,975 |
| `cudaLaunchKernel` | 35,916 | 36,619 |

多出的 703 次 launch 对应 703 个 resident endpoint wavefront；减少的
2,062 次 `cudaMemcpy2D` 对应被合并的同步计数回读。

在 1 TeV 小 shower 中，kernel 时间由插值版修改前的约 1.56–1.69 s
进一步降到 1.44–1.49 s。1 PeV 中该改动落在运行波动内，不能宣称显著
收益；其主要价值是为后续完全 device-resident 的状态汇总建立接口。

## 7. 性能结果

### 7.1 基准条件

共同参数：

```text
primary: photon
GPU: RTX 4060 Laptop
backend: cuda
gpu-min-batch: 64
table tolerance: 1e-3
EM cut: 0.5 MeV
EM thinning: 1e-4
maximum weight: 100
fast-math: disabled
precision: double
```

### 7.2 1 PeV 固定种子

| 版本 | runtime | lepton backend | kernel |
|---|---:|---:|---:|
| 正确补全 loss、拆分前 | 42.49 s | 22.45 s | 22.66 s |
| 独立解析 Molière kernel | 36.89 s | 16.85 s | 16.68 s |
| CPU 等价 cubic Molière | 29.53–34.05 s | 10.22–10.39 s | 9.42–9.49 s |
| cubic + endpoint summary | 31.30 s | 10.38 s | 9.44 s |

当前完整应用相对约 106.3 s 的单核 CPU 基准约为 3.3 倍。不能把这个数字
写成 5 倍，因为 CPU photonuclear/hadronic 分支和应用初始化仍占显著时间。

### 7.3 物理路径变化的解释

从解析 Molière 改为 CPU 等价插值后，散射角会出现 PROPOSAL 允许的点对点
差异。虽然 Philox 随机数不变，后续几何和相互作用历史会分叉，所以两版
shower 不能要求逐粒子相同。

当前 cubic 版本自身的重复运行满足：

- 1 TeV：七个 Parquet 输出全部逐字节相同；
- 1 PeV：七个 Parquet 输出全部逐字节相同；
- 所有非计时统计相同；
- 1 PeV 的 75 个尾部 loss fallback 均相同；
- weighted deposited energy 相同。

## 8. 当前限制

1. 当前 WSL 的 Nsight Systems 可以记录 CUDA Runtime API，但报告中的 GPU
   kernel 表为空；Nsight Compute 连接后返回设备 `Unknown Error`。因此尚
   无可靠的 occupancy、warp divergence 和 memory-throughput 硬件指标。
2. `c8_air_shower` 是全物理应用，不是隔离的纯 EM 性能驱动。它会把
   photonuclear 末态送回 CPU，这是正确物理，不应为了计时而静默删除。
3. 目前仍有大量 host 驱动的阶段计数和错误码回读。endpoint summary 只是
   第一处压缩，并未把整个 resident wavefront 变成单次 device graph。
4. 单固定种子只验证确定性和工程性能，不能替代 CPU/GPU shower ensemble
   对 \(X_{\max}\)、纵向 profile、地面谱和时间分布的统计检验。

## 9. 下一阶段优先级

### 9.1 建立纯 EM 验收驱动

新增一个不初始化 PYTHIA、TAUOLA、UrQMD 和强子 interaction chain 的
production-like 驱动，只启用：

```text
gamma, e-, e+
pair / brems / Compton / photoelectric
ionization / annihilation / epair
LPM / Moliere / cut / thinning
spherical atmosphere / uniform magnetic field
```

该驱动用于回答原验收标准中的“纯 EM 端到端是否达到 5 倍”，不能用全物理
应用的 3.3 倍代替。

### 9.2 合并 device pipeline 状态

优先把以下多次回读合并成每个大 wavefront 的单个 `DevicePipelineSummary`：

```text
selection fallback count
transport fallback count
interaction candidate count
vertex classification counts
final-state classification counts and error
endpoint counts and error
```

更进一步可用 CUDA Graph 固定小 kernel 拓扑，降低当前每个 1 TeV shower
约 3.6 万次 launch 的 host 开销。

### 9.3 异步输出

为 step、observation、fallback 和次级粒子使用 pinned host staging，并把
结果回传与下一 wavefront 的 GPU 工作重叠。现阶段不要在没有测量的情况下
引入多 stream 顺序变化。

### 9.4 恢复硬件 profiler

需要同时检查：

- Windows NVIDIA driver 的 WSL/CUPTI 支持；
- Nsight Systems 与 CUDA 12.6 runtime 的兼容性；
- GPU performance counter 权限；
- WSL 内核和 `/dev/dxg` 状态。

在 kernel timeline 恢复前，不应根据 register 数单独决定下一项低层优化。

### 9.5 统计物理验收

完成至少：

```text
1000 x 1 TeV
200  x 1 PeV
electron and photon primaries
zenith 0, 45, 80 degrees
multiple cuts and thinning settings
```

比较 CPU `MoliereInterpol` 与 GPU cubic 后端的 \(X_{\max}\)、\(N_e(X)\)、
\(N_\gamma(X)\)、\(dE/dX\)、地面横向分布和到达时间。只有 ensemble 检验
通过后，才能把本阶段实现标为科研生产可用。
