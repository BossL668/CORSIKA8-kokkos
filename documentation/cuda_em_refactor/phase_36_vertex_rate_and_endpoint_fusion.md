# 阶段 36：反应率共享插值与稳定多类别压缩融合

## 1. 本阶段结论

阶段 35 完成 Molière Newton 初值优化后，lepton pipeline 中三个可观测的
调度热点为：

```text
interaction selection
vertex selection
endpoint compaction
```

这些阶段并不生成新的物理模型，主要负责：

- 对 PROPOSAL rate table 做能量插值；
- 从总反应率中选定 process/component；
- 把 interaction、continuation、fallback 分类；
- 稳定压缩下一 wavefront 的 lepton、photon 和 observation。

旧实现的主要问题是，同一粒子、同一能量的 rate-table bracket 被每个
process/component column 重复查找；互斥类别又分别执行多次 CUB scan 和
多个收尾 kernel。

本阶段完成：

1. 一个粒子的所有 rate columns 共享一次能量 bracket；
2. vertex 总反应率求和和 process/component 选择合并为一次列遍历；
3. interaction、continuation、fallback 使用一个三分量整数 scan；
4. next-lepton、generated-photon、observation 使用一个 transform-input
   三分量整数 scan；
5. endpoint 压缩、observation 压缩和 pipeline summary 写回合并到一个
   kernel；
6. 保留每个类别内部的输入顺序和原有 Philox 随机数身份。

四组 UHE shower 的结果为：

- `vertex_selection_ms` 下降 53.1%–55.9%；
- `selection_ms` 下降 27.8%–40.2%；
- `endpoint_compaction_ms` 下降 17.4%–22.9%；
- CUDA EM kernel 总时间下降 11.9%–14.9%；
- `HybridCascade::run()` 端到端下降 12.1%–16.6%；
- 四组样本的 9 个 Parquet/NPZ 科学输出均与阶段 35 逐字节相同；
- 同一 GPU、配置和 seed 的重复运行逐字节相同；
- 24 个 GPU 测试全部通过；
- 没有 queue overflow、radio fixed-point overflow 或新增 CPU fallback。

本阶段没有启用 `fast-math`、混合精度或非稳定原子队列，也没有改变
PROPOSAL 表格、物理过程概率、thinning 或次级粒子 history 编号。

## 2. 涉及的源码

主要修改位于：

```text
corsika/gpu/em/tables/FlatRateTable.hpp
src/gpu/em/CudaLeptonVertexSelector.cu
src/gpu/em/CudaLeptonSelectionTransport.cu
tests/gpu/testGpuEmFlatRateTable.cpp
```

相关 pipeline 顺序为：

```text
lepton selection
  -> lepton transport
  -> interaction extraction
  -> vertex selection
  -> final-state generation
  -> endpoint compaction
  -> next resident wavefront
```

## 3. 原 rate-table 查询的重复工作

每种粒子在一个介质中有多个离散反应率 column。一个 column 对应一个
`process_id × component_hash`。同一粒子的所有 columns 使用相同的能量
网格：

```text
particle_energy_offsets[particle]
particle_energy_counts[particle]
rate_energies_MeV[]
```

旧 `queryTotalRate()` 对每个 column 调用一次
`interpolateRateColumn()`。每次调用都会重复：

```text
检查能量范围
  -> 在相同能量轴上二分查找
  -> log(E)
  -> log(E_lower)
  -> log(E_upper)
  -> 计算同一个 interpolation fraction
  -> 读取当前 column 的两个 rate
```

vertex selection 随后为了找出随机阈值跨过的 column，又对所有 columns
执行第二次完整遍历。

因此每个发生离散反应的 lepton 原来近似执行：

```text
2 × process-column count × binary search/log interpolation
```

对于高能 shower 中几十万次离散顶点，这部分重复标量工作成为明显热点。

## 4. 共享 `RateInterpolationBracket`

`FlatRateTable.hpp` 新增：

```cpp
struct RateInterpolationBracket {
  TableLookupStatus status;
  std::uint32_t lower;
  std::uint32_t upper;
  double fraction;
};
```

以及两个设备/host 共用函数：

```cpp
rateInterpolationBracket(
    view, particle_index, energy_MeV);

interpolateRateColumnAtBracket(
    view, particle_index, column_index, bracket);
```

现在查询流程为：

```text
粒子 PID + 能量
  -> 一次二分查找
  -> 一次对数插值坐标
  -> RateInterpolationBracket
  -> 对所有 columns 只读取 rate 值并插值
```

`queryTotalRate()` 和 vertex selector 都复用这个 bracket。已有
`interpolateRateColumn()` 外部语义不变，它内部也委托给新的 bracket
函数。

边界行为保持显式：

- 非有限能量返回 `NonFiniteInput`；
- 能区之外返回 `RateEnergyOutOfRange`；
- 非法 offset/count 返回 `InvalidTableView`；
- 不允许夹断到表格边界后继续模拟。

新增的 host 单元测试逐 column 比较：

```text
interpolateRateColumn(...)
    ==
interpolateRateColumnAtBracket(...)
```

覆盖能量网格端点、内部网格点、对数中点、越界能量和 NaN，共使
`testGpuEmFlatRateTable` 达到 279 个检查。

## 5. vertex rate 求和与反应选择合并

反应类型抽样使用：

\[
r = u\sum_j\lambda_j(E),
\]

然后选择第一个满足

\[
\sum_{j=0}^{k}\lambda_j(E) \ge r
\]

的 column \(k\)。

旧流程为：

```text
第一次 columns 循环：求 total rate
随机数阈值 = u × total rate
第二次 columns 循环：寻找 threshold crossing
```

新 vertex kernel 在一次 columns 循环中同时：

1. 按原 column 顺序累计精确 total；
2. 用已经抽取的随机数所对应的等价阈值选择第一个 crossing；
3. 即使已经选中也继续完成相同顺序的 total 累加；
4. 保存原 `process_id`、`component_hash` 和 loss-query 随机数键。

列顺序、浮点加法顺序和第一个 crossing 规则没有改变。四个固定种子
shower 的逐字节输出相同，证明这次合并没有改变已验收路径的随机抽样
结果。

## 6. vertex 三类别稳定 scan

vertex 输出互斥地属于：

```text
interaction
continuation
fallback
```

旧实现为每个类别分别创建 flag/offset，并执行三次
`cub::DeviceScan::ExclusiveSum`。现在每条记录写一个：

```cpp
struct VertexCategoryCounts {
  std::uint32_t interactions;
  std::uint32_t continuations;
  std::uint32_t fallbacks;
};
```

然后执行一次：

```cpp
cub::DeviceScan::ExclusiveScan(
    categories,
    offsets,
    AddVertexCategoryCounts{},
    VertexCategoryCounts{},
    count);
```

三分量加法只包含非负整数，结合律精确成立。对某一类别，当前元素的
exclusive prefix 就是它在该类别紧凑数组中的位置，因此类别内部仍严格
保持输入顺序。

压缩 kernel 同时：

- 写 interaction 数组；
- 写 continuation 数组；
- 写 fallback 数组；
- 用最后一个 prefix 和 category 写 vertex summary。

原独立单线程 summary kernel 被删除。

## 7. endpoint 单次 scan 和压缩融合

每个 lepton final state 最多预留三个 child slots。child 可能成为：

```text
next lepton
generated photon
empty
```

source record 还可能产生 observation。旧实现分别执行：

```text
lepton child scan
photon child scan
observation scan
```

并由独立 kernel 完成：

```text
child compaction
observation compaction
summary finalization
```

新实现定义：

```cpp
struct LeptonEndpointCategoryCounts {
  std::uint32_t leptons;
  std::uint32_t photons;
  std::uint32_t observations;
};
```

`LoadLeptonEndpointCategoryCounts` 通过 CUB
`CountingInputIterator + TransformInputIterator` 构造虚拟输入：

```text
index < source_count:
    child lepton flag
    child photon flag
    observation flag

index >= source_count:
    child lepton flag
    child photon flag
    observation = 0
```

这样无需额外 materialize 一个三分量输入数组，只执行一次三分量
exclusive scan，就得到三个稳定输出位置。

新的 combined compaction kernel 一次完成：

- next-lepton compaction；
- generated-photon compaction；
- observation compaction；
- endpoint summary；
- vertex/final-state summary 的最终计数写回。

child slot 的原子操作只用于检测同一个预分配 slot 被重复声明的程序错误；
原子返回顺序不用于分配输出位置。实际紧凑位置完全由稳定 prefix scan
决定。

## 8. workspace 与显存

vertex 的两个三分量数组与旧六个 `uint32_t` flag/offset 数组字节数相同。

endpoint 输出 prefix 从两个 child offsets 加一个独立 observation offset，
变成一个三分量 child-slot prefix。由于 child-slot 数为
`3 × source_count`，最坏情况下比紧接本阶段之前的版本多约：

\[
8N\ \mathrm{bytes},
\]

其中 \(N\) 是 source lepton 数。seed 32017 的最大 resident lepton batch
为 333690，对应约 2.67 MB。

实际四个样本中：

- 配置的 workspace 仍为 128 MiB；
- `workspace_limit_checkpoints = 0`；
- seed 32017 的 peak device bytes 仍为 689034206；
- 两个 \(10^{18}\,\mathrm{eV}\) 样本的 peak device bytes 仍为 823251934；
- 没有 host spill。

即本阶段用少量已有 workspace 余量换取了更少的 scan 和 kernel launch，
没有提高实测峰值显存。

## 9. 四组 UHE 性能结果

运行配置：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA: 12.6 runtime/driver path
primary: electron
energy: 1e17 或 1e18 eV
EM thinning: 1e-3
max weight: 1e6
gpu-min-batch: 64
radio backend: CUDA CoREAS + ZHS
table: production_v9_1e-3_1EeV.c8emrt
detailed stage timing: enabled
```

单位均为 ms：

| 能量/seed | 阶段 | selection | vertex | endpoint | CUDA EM kernel | total run |
|---|---:|---:|---:|---:|---:|---:|
| \(10^{17}\), 32017 | 35 | 210.454 | 255.925 | 176.234 | 1731.072 | 1900.814 |
| \(10^{17}\), 32017 | 36 | 125.845 | 120.151 | 142.752 | 1525.162 | 1657.453 |
| \(10^{17}\), 32018 | 35 | 203.831 | 235.607 | 171.650 | 1649.338 | 1816.942 |
| \(10^{17}\), 32018 | 36 | 123.692 | 105.575 | 132.311 | 1403.050 | 1515.238 |
| \(10^{18}\), 33018 | 35 | 351.015 | 313.060 | 215.461 | 2444.962 | 2578.126 |
| \(10^{18}\), 33018 | 36 | 241.962 | 138.180 | 168.870 | 2135.774 | 2265.011 |
| \(10^{18}\), 33019 | 35 | 372.269 | 320.663 | 220.540 | 2545.800 | 2816.011 |
| \(10^{18}\), 33019 | 36 | 268.715 | 141.452 | 182.138 | 2240.461 | 2467.975 |

对应的 Phase 36 相对变化：

| 能量/seed | selection | vertex | endpoint | CUDA EM kernel | total run |
|---|---:|---:|---:|---:|---:|
| \(10^{17}\), 32017 | -40.20% | -53.05% | -19.00% | -11.89% | -12.80% |
| \(10^{17}\), 32018 | -39.32% | -55.19% | -22.92% | -14.93% | -16.61% |
| \(10^{18}\), 33018 | -31.07% | -55.86% | -21.62% | -12.65% | -12.15% |
| \(10^{18}\), 33019 | -27.82% | -55.89% | -17.41% | -11.99% | -12.36% |

`transport_ms` 和 radio/profile kernel 没有在本阶段修改。不同进程运行时
GPU boost clock 和 WSL 调度会造成少量波动，因此验收重点是被修改阶段、
四个配对样本的一致趋势和端到端结果，而不是单次未修改阶段的差值。

## 10. 物理、确定性和错误验收

### 10.1 阶段 35 与阶段 36 对照

下列四对输出分别比较：

```text
1e17 seed 32017
1e17 seed 32018
1e18 seed 33018
1e18 seed 33019
```

每个 shower 的 9 个 `*.parquet`/`*.npz` 文件全部 `cmp` 相同：

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

四组样本的以下计数也保持相同：

```text
gpu_particles
gpu_final_states
physical_secondaries
weighted_deposit_GeV
radio_tracks
cpu_specified_final_states
observed / escaped / cut
```

### 10.2 重复性

Phase 36 的 \(10^{17}\,\mathrm{eV}\)、seed 32017 独立重复运行：

- 9 个科学文件逐字节相同；
- GPU 粒子数、末态数、次级数、能量沉积、radio tracks、fallback、
  observation 和峰值队列计数相同；
- 时间字段不参与确定性比较。

### 10.3 测试

```text
ctest --output-on-failure -R '^testGpu'
```

结果：

```text
24/24 passed
0 failed
```

另外：

```text
testGpuEmFlatRateTable: 279 checks passed
git diff --check: passed
```

四组 UHE shower 均满足：

```text
complete = true
queue_overflows = 0
workspace_limit_checkpoints = 0
cross_species.host_spills = 0
radio.fixed_point_overflows = 0
epair_sampler.cpu_fallbacks = 0
epair_sampler.envelope_violations = 0
```

## 11. 下一阶段热点

Phase 36 后，seed 32017 的主要 lepton pipeline 时间为：

```text
transport                         643.158 ms
  transport physics              190.214 ms
  Molière                        347.831 ms
  transport control              98.058 ms
final state                      238.511 ms
  classification                195.804 ms
  scan                             8.284 ms
  summary                          4.788 ms
  write                           27.793 ms
endpoint compaction              142.752 ms
selection                        125.845 ms
vertex selection                 120.151 ms
```

因此下一阶段不应继续只优化已经很小的 final-state scan。优先审计：

1. final-state classification 中是否重复查询 loss table、重复分支和重复
   写临时记录；
2. Molière 中剩余的特殊函数与每 step 两次投影 Newton；
3. transport physics/control 是否能共享环境、grammage 和连续损失中间量；
4. endpoint 是否还能减少临时 child-state 全量写回。

任何后续融合都必须继续满足：

- 类别内部稳定顺序；
- `(history_id, step_id, process_id, draw_id)` 随机键不变；
- 非法 table/physics 状态显式 fallback 或终止；
- 固定种子科学输出逐字节比较；
- 四组 UHE 配对性能回归。
