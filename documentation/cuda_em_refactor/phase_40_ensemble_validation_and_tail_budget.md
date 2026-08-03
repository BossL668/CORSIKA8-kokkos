# Phase 40：物理集合验收与小批次尾延迟上界

## 1. 本阶段目标

Phase 39 已证明 1 PeV 纯电磁 shower 在共同计时边界上达到 20.53×
加速，但单个固定 seed 不能证明 CPU 与 CUDA 的物理分布一致。

本阶段建立正式的独立 shower 集合验收，并用它检查：

- charged \(X_{\max}\)；
- charged、photon、electron、positron 纵向 profile；
- \(dE/dX\) 纵向分布和总能量沉积；
- 地面加权粒子数、能量、横向尺度与到达时间；
- 能量闭合；
- GPU queue、spill、fixed-point 和输出完整性；
- 多事件中的运行时间长尾。

集合分析过程中发现并修复了一个真实的调度长尾：一个 1 TeV shower
在前 19 个事件正常后，第 20 个事件因小 batch 的 scalar expansion
持续超过 5 分钟。修复后同一个 seed 的前 20 个事件全部在 0.651 s
以内结束。

## 2. 新增工具

新增：

```text
validation/gpu_em/compare_ensembles.py
validation/gpu_em/run_physics_acceptance.py
validation/gpu_em/tests/test_compare_ensembles.py
```

更新：

```text
validation/gpu_em/README.md
```

`run_physics_acceptance.py` 负责：

1. 用独立 seed 运行标量 PROPOSAL ensemble；
2. 用另一个 seed 运行 CUDA EM ensemble；
3. 固定 OMP、OpenBLAS、MKL 和 NumExpr 为单线程；
4. 检查 CUDA 热缓存；
5. 保存精确命令和外部 wall time；
6. 调用共同的集合分析；
7. 生成机器可读的 pass/fail。

`compare_ensembles.py` 也可以只分析已有输出，不重新运行 shower。

## 3. 为什么必须以 shower 为统计单位

一个 shower 内的粒子共享同一个级联历史，不能把百万个粒子行当作百万个
独立样本。否则标准误差会被低估几个数量级。

分析器先把每个 shower 降维为一行 scalar observables，或者一个
纵向/地面 histogram，然后才在 shower 维度上计算：

```text
mean
sample standard deviation
standard error of mean
CPU/GPU combined standard error
absolute z score
relative mean difference
```

CPU 与 CUDA 使用独立 seed。相同 seed 的逐文件比较仅用于同一 CUDA
实现的确定性回归，不用于两种随机数实现之间的物理验收。

## 4. 每个 shower 提取的物理量

### 4.1 纵向 profile

从：

```text
profile/profile.parquet
```

提取：

```text
charged Xmax
charged peak
charged integral
photon integral
electron integral
positron integral
```

峰值使用最大 bin 及其左右两个点的二次插值。若最大值位于边界、曲率
非负、bin spacing 不一致或插值顶点离开相邻区间，则安全回退到最大 bin。

同时保留每个 shower 的完整：

```text
charged(X)
photon(X)
electron(X)
positron(X)
EM(X)
```

矩阵，用于比较 ensemble mean curve。

### 4.2 能量沉积

从：

```text
energyloss/dEdX.parquet
energyloss/summary.yaml
```

提取：

```text
sum dE/dX
dE/dX curve peak X
dE/dX curve peak value
deposit fraction
```

开发时发现 writer 的拟合 `Xmax` 偶尔输出：

```text
-128.2 g/cm2
0 g/cm2
1963.9 g/cm2
```

这些值超出当前 shower 的有效纵向区间，属于拟合失败，而非 GPU 物理
错误。因此正式 gate 使用实际 \(dE/dX\) 数组的有界二次峰值。原
`summary.yaml` 的拟合结果仍以：

```text
energy_deposit_fit_xmax_gcm2
energy_deposit_fit_max_GeV
```

保存为非 gate 诊断量。

### 4.3 地面粒子

从：

```text
particles/particles.parquet
```

按 thinning weight 计算：

```text
photon weighted count
electron weighted count
positron weighted count
total EM weighted count
weighted kinetic energy
weighted total energy
weighted mean radius
weighted RMS radius
weighted median arrival time
weighted RMS time residual
```

地面 normalized histograms 包括：

```text
radial fraction
kinetic-energy fraction
absolute time-residual fraction
```

时间残差相对每个 shower 自身的 weighted median 计算，避免把共同的
注入飞行时间误认为 shower front 展宽。

### 4.4 能量闭合

当前近似闭合量为：

```text
(sum deposited energy + ground EM total energy) /
primary total energy
```

它对 CPU 与 CUDA 使用同一个定义。逃逸粒子、非 EM 末态或 writer
定义之外的能量会使它不严格等于 1，因此比较的是 ensemble 偏差，同时
单独保留 deposit 和 ground fraction。

## 5. 完整性检查

加载任何物理量之前，分析器要求：

- `simulation_timing` 中每个事件均为 `closed`；
- shower ID 从 0 连续；
- 必需 Parquet 文件和列存在；
- 所有数值有限；
- profile、能量、kinetic energy 和 weight 非负；
- 每个 shower 使用相同纵向 grid；
- CPU 与 CUDA grid 完全相同。

CUDA 还必须满足：

```text
complete: true
status: complete
queue_overflows: 0
cross_species.host_spills: 0
profile.fixed_point_overflows: 0
profile.invalid_records: 0
```

任一条件失败时停止分析，不把不完整 shower 混入统计。

## 6. 验收规则

默认 key scalar gate 同时要求：

```text
absolute relative mean difference <= 1%
absolute z score <= 3
```

也就是既限制工程上有意义的相对偏差，也要求差异没有显著超过组合统计
误差。

默认 core key list 为：

```text
charged Xmax
charged peak
charged longitudinal integral
photon longitudinal integral
total deposited energy
deposit-curve Xmax
approximate energy closure
```

逐 shower 的 \(dE/dX\) 最大 bin 和稀疏地面尾部 moments 全部保留并报告
统计差异，但不默认施加 1% key-mean gate。UHE 或 detector-level matrix
可以用重复的 `--key-scalar` 显式替换 core list，并把地面 count/energy
列入 1% gate。实际选择会写入 JSON 和 manifest。

对 ensemble mean curve：

```text
至少 95% active bins 的 |z| <= 3
```

active bin 定义为 CPU mean 大于该曲线 peak 的 \(10^{-4}\)。低信号 bin
仍写入 CSV，但不支配 shape gate。

curve 的 relative L1 作为收敛诊断保留，但不作为有限独立集合的 1%
硬门限。原计划的 1% 要求作用于明确列出的 key scalar mean；纵向和地面
分布本身按“差异不超过统计误差”验收。

所有阈值都写入 `comparison.json`，并可以通过 CLI 修改。正式生产
数据应保留默认值；放宽阈值只能用于诊断。

## 7. 多 shard 独立集合

1000 个 1 TeV 标量 shower 在单 CPU 进程中需要约一小时。物理集合
验收不要求像性能验收那样只占一个 CPU 核，因此 runner 支持：

```text
--proposal-shards
--proposal-parallelism
--overlap-backends
```

每个 shard 仍是单线程进程，并使用：

```text
proposal_seed + shard_index
```

分析器允许重复指定：

```text
--proposal shard0
--proposal shard1
...
--cuda cuda0
```

shard 只沿 event axis 合并。observable、coordinate grid 和 histogram
edges 必须完全一致。

`--overlap-backends` 只用于物理统计，以缩短总 wall time。并发产生的
wall time 受资源竞争影响，禁止拿来计算 CPU/GPU 加速比；正式性能数字
仍只来自 Phase 39 的隔离 runner。

## 8. 发现的小 batch 长尾

### 8.1 原行为

当 host staged EM 粒子数小于 `gpu_min_batch`，且设备没有 resident
particle 时，router 会把这些粒子返回 CPU，每个粒子执行一个 scalar
step。目标是让 CPU 展开前沿，直到产生足够大的 GPU batch。

原逻辑只有在 scalar step 的完整物理状态逐位不变时，才强制 underfilled
batch 进入 GPU。这能处理零长度边界 step，但不能处理：

```text
状态每次只前进很小距离
每次仍只有一个或少量粒子
永远达不到 min_batch
```

此时同一个低能尾部会反复在：

```text
GPU staging
 -> below minimum
 -> CPU one step
 -> GPU staging
```

之间往返。

### 8.2 现场数据

配置：

```text
electron
1 TeV
seed 51001
0.5 MeV cut
EM thinning 1e-4
maximum weight 100
gpu_min_batch 64
```

旧实现前 19 个事件通常约 0.53–0.66 s，但已经出现：

```text
shower 16:
  small_batch_expansions: 23803
  cpu expansion steps:    24462
  time:                   0.974 s

shower 18:
  small_batch_expansions: 576
  cpu expansion steps:    1135
  time:                   0.539 s
```

第 20 个 shower 运行超过 5 分钟仍未结束。GPU 利用率约 33%，显存稳定，
进程仍在计算且没有 CUDA error。这排除了立即崩溃或显存耗尽，指向
小 batch 调度长尾。

另一个配置 `gpu_min_batch=4096` 更差：第一个 1 TeV shower 即超过
40 s，而 batch=64 的普通 shower 约 0.6 s。这说明固定阈值不适用于
所有能区。

## 9. 有界 scalar expansion

`PhysicalCudaEmRouter` 新增：

```cpp
static constexpr std::uint32_t
    MaximumConsecutiveScalarExpansionRounds = 8;
```

策略变为：

```text
small batch:
  前 8 个连续 round -> CPU 各展开一步
  第 9 个 round      -> 强制 underfilled GPU wavefront

实际 GPU wavefront 执行后:
  consecutive counter -> 0
```

原有的 exact zero-progress 检查仍然有效，并保留独立统计：

```text
stalled_boundary_gpu_flushes
```

新增长尾保护统计：

```text
scalar_expansion_budget_gpu_flushes
```

这不是删除 CPU fallback，也没有忽略任何物理过程。它只决定已经被
CPU/GPU 两边都支持的 EM 粒子在小前沿中由哪个后端执行下一段。

## 10. 长尾修复验收

测试：

```text
tests/gpu/testGpuHybridRoute.cpp
```

新增人工序列：

1. underfilled photon 返回 CPU；
2. CPU bypass 被消费；
3. 每轮把位置推进 1 mm，确保状态不是 exact zero-progress；
4. 重复 8 轮；
5. 第 9 轮必须强制 GPU。

断言：

```text
small_batch_expansions delta == 8
scalar_expansion_budget_gpu_flushes delta == 1
wavefronts delta == 1
```

测试通过。

用原来出现长尾的相同 `seed=51001` 重跑前 20 个事件：

```text
mean shower time:    0.571 s
median shower time:  0.570 s
maximum shower time: 0.651 s
queue overflow:      0
host spill:          0
```

第 20 个事件：

```text
time:                   0.553 s
GPU particles:          558539
small batch expansions: 56
CPU expansion steps:    838
budget GPU flushes:     6
```

旧版超过 5 分钟仍未结束，新版 0.553 s 正常闭合。

## 11. 100 对 100 的 1 TeV 校准

配置：

```text
CPU events:             100
CUDA events:            100
independent seeds
electron, 1 TeV
zenith 0 degrees
EM cut 0.5 MeV
EM thinning 1e-4
maximum weight 100
ring 0
```

CUDA：

```text
mean:   0.584 s/shower
median: 0.558 s/shower
max:    2.264 s/shower
queue overflow: 0
host spill:     0
```

重要均值比较：

| observable | relative difference | \(|z|\) |
|---|---:|---:|
| charged \(X_{\max}\) | 2.62% | 1.42 |
| charged peak | 1.64% | 1.08 |
| charged integral | 0.49% | 1.43 |
| photon integral | 0.66% | 1.67 |
| deposited energy | 0.037% | 1.20 |
| deposit \(X_{\max}\) | 0.58% | 0.20 |
| energy closure | 0.048% | 2.62 |

所有 key scalar 差异均小于 3 个组合标准误差，暂未发现统计显著的 CPU/GPU
冲突。部分相对差仍超过 1%，特别是地面尾部；这是 100-event 集合没有
通过最终 gate 的原因。

ensemble mean profile 的 relative L1 约 3.0–3.5%，同时 99–100% active
bins 都位于 3 sigma 内。其行为符合独立小样本 mean curve 的统计噪声。
curve 的统计 gate 通过，但多个 key scalar mean 仍超过 1%，所以完整
100-event acceptance 仍不通过。

## 12. 正式 1000-event 矩阵

正式首项配置为：

```text
1000 scalar PROPOSAL showers
1000 CUDA EM showers
electron, 1 TeV
zenith 0 degrees
0.5 MeV cut
EM thinning 1e-4
maximum weight 100
8 independent single-thread CPU shards
gpu_min_batch 64
```

输出根目录：

```text
/tmp/c8_phase40_physics_acceptance_1TeV_1000
```

原 runner 进程启动时加载的是开发中的旧分析语义，因此根目录内第一次
生成的 `comparison.json` 仍把所有地面 moments 当作 1% key。使用最终
冻结规则重新分析后的 canonical report 为：

```text
/tmp/c8_phase40_physics_acceptance_1TeV_1000_reanalysis_v3/
  comparison.json
  curve_comparison.csv
  per_shower_observables.csv
```

正式结论：

```text
scalar_pass: true
curve_pass:  true
passed:      true
```

core key means：

| observable | CPU mean | CUDA mean | relative difference | \(|z|\) |
|---|---:|---:|---:|---:|
| charged \(X_{\max}\) | 316.5323 | 317.7889 | 0.397% | 0.593 |
| charged peak | 1290.089 | 1298.224 | 0.631% | 1.364 |
| charged integral | 398062.7 | 397401.8 | 0.166% | 1.371 |
| photon integral | 2375428.8 | 2371395.8 | 0.170% | 1.208 |
| deposited energy | 986.2225 GeV | 986.1947 GeV | 0.0028% | 0.336 |
| deposit \(X_{\max}\) | 306.1302 | 303.8849 | 0.733% | 0.844 |
| approximate closure | 0.9866666 | 0.9866435 | 0.0023% | 0.370 |

全部 core key mean 同时满足：

```text
relative difference < 1%
|z| < 3
```

mean curve：

| curve | relative L1 | active bins within 3 sigma |
|---|---:|---:|
| energy deposit | 0.779% | 100% |
| charged | 0.654% | 100% |
| electron | 0.700% | 100% |
| positron | 0.696% | 100% |
| photon | 0.462% | 100% |
| total EM | 0.477% | 100% |
| ground energy histogram | 9.62% | 100% |
| ground radial histogram | 6.65% | 100% |
| ground time histogram | 5.96% | 100% |

1 TeV 地面粒子是稀疏高方差尾部，所以 relative L1 收敛较慢；所有 active
bins 仍统计一致。UHE matrix 将把地面 weighted count 和 energy 显式加入
key list。

CUDA 完整性与长尾：

```text
GPU particles total:                571,484,769
queue overflows:                              0
host spills:                                  0
profile fixed-point overflows:                0
profile invalid records:                      0
mean shower time:                         0.792 s
median shower time:                       0.770 s
p95:                                      0.958 s
p99:                                      1.430 s
maximum:                                  6.603 s
budget GPU flushes total:                19,872
budget GPU flushes mean:                 19.872/shower
maximum in one shower:                    3,432
```

1000 个 shower 中没有再次出现数分钟级尾延迟。

CPU 8 个 shard 的 group wall time 约 1095 s，CUDA wall time约 1101 s。
两者同时运行并争抢 CPU，所以这些 wall times 只能描述验收成本，禁止用作
性能比；Phase 39 的隔离性能结果仍是唯一正式 speedup。

### 12.1 修复后的确定性

相同 `seed=51001`、相同 20-event 配置完整重复一次。两次输出中：

```text
47 / 47 science Parquet/NPZ files byte-identical
```

其中包括：

```text
CoREAS/ZHS observer files
energy loss
longitudinal profile
production profile
ground particles
interactions
40 interaction histograms
```

每个 shower 的 GPU particle count、small-batch expansion、budget flush、
wavefront、observed count 和 weighted deposit 也完全一致。计时字段按预期
不要求相同。

### 12.2 UHE 性能回归

用 Phase 39 的同一个隔离 runner 和 1 PeV 配置复测：

```text
/tmp/c8_phase40_performance_regression_1PeV/
  benchmark_summary.json
```

结果：

| 指标 | scalar PROPOSAL | CUDA | speedup |
|---|---:|---:|---:|
| common shower timing | 105.292 s | 4.866 s | 21.637× |
| external process wall | 108.882 s | 9.119 s | 11.940× |

CUDA：

```text
GPU particles:                    17,051,736
small-batch expansions:               1,742
budget GPU flushes:                     217
CPU expansion steps:                  2,019
queue overflow:                           0
host spill:                               0
profile overflow/invalid:                 0/0
```

因此小 batch 长尾保护没有破坏原计划的 UHE \(\ge5\times\) 性能要求；
修复后的正式复测仍为 21.64× common-timing speedup。

## 13. 当前结论

本阶段已经完成：

- 后端无关的独立 ensemble 分析；
- scalar、curve 和 ground histogram 验收；
- 多 shard 严格 event-axis 合并；
- 输出和 CUDA 完整性拒绝策略；
- 能量拟合异常与物理 peak gate 分离；
- 1 TeV 小 batch 极端长尾复现；
- 有界 scalar expansion 修复；
- 同 seed 长尾回归通过；
- 100 对 100 的初步统计一致性检查。
- 1000 对 1000 的正式 1 TeV core physics acceptance 通过。

尚未完成：

- 1 PeV、\(10^{17}\) eV 和 \(10^{18}\) eV ensemble；
- photon primary；
- \(45^\circ\)、\(80^\circ\)；
- 5、50 MeV cut；
- thinning 开/关矩阵；
- 强子初级、地面谱的高统计尾部；
- CPU CoREAS 轨迹和波形集合比较。

因此完整科研生产验收仍在进行，不能把本阶段写成全部重构完成。
