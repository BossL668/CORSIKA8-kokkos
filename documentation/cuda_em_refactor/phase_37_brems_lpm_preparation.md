# 阶段 37：Brems LPM 静态项设备预计算与 block 级缓存

## 1. 本阶段结论

阶段 36 后，lepton final-state classification 是剩余主要热点之一。代表性
\(10^{17}\,\mathrm{eV}\)、seed 32017 数据为：

```text
final-state classification       195.804 ms
final-state total                238.511 ms
```

classification 对每个 brems 顶点调用 PROPOSAL 7.6.2 的 BremsLPM 公式。
其中多项只依赖版本化介质和组分，却在每个粒子上重复计算：

```text
pow(Z, -1/3)
pow(A, 0.27)
s1
log(s1)
完整 snapshot/component 有效性检查
```

本阶段实现：

1. 初始化 CUDA 后端时，在目标 GPU 上一次性计算每个介质组分的
   \(s_1\) 和 \(\log s_1\)；
2. 使用设备端双精度 `pow/log`，避免 host 数学库和 device 数学库末位
   不同；
3. 每个 classification block 只把 prepared components 复制到 392 B
   shared memory；
4. 每个 brems 线程只执行与 \(E,v,\rho\) 有关的动态 LPM 部分；
5. 常见过程计数使用 warp 聚合，一组活跃 lanes 只发出一次全局
   `atomicAdd`；
6. 非法或未准备的 LPM snapshot 在 shower 开始前终止，不允许静默使用
   未准备路径。

四组 UHE shower 中：

- `final_state_classification_ms` 稳定下降 5.6%–9.4%；
- `final_state_ms` 下降 3.5%–7.5%；
- 四组样本的 9 个 Parquet/NPZ 科学输出与阶段 36 逐字节相同；
- 同一 GPU、配置和 seed 的重复运行逐字节相同；
- 24 个 GPU 测试全部通过；
- queue、workspace、radio fixed-point 和 Epair sampler 均无新增错误。

本阶段在未修改阶段的进程间计时噪声之上没有证明稳定的端到端加速，因此
不把单次 `total_run` 或总 kernel 波动表述为整体收益。已经证明的收益是
classification/final-state 局部阶段。

## 2. 原 BremsLPM 计算

对组分 \(i\)，PROPOSAL BremsLPM 使用：

\[
Z_i^{-1/3},
\qquad
d_{n,i}=1.54 A_i^{0.27},
\]

\[
a_i=
\frac{m_e}
{m_\ell Z_i^{-1/3} L_i},
\]

\[
s_{1,i}=
a_i^2
\left[
1+
\left(
\frac{m_\ell}{m_\mu}d_{n,i}
\right)^2
\right]\sqrt{2}.
\]

随后每个粒子根据局部质量密度、总能量 \(E\) 和能损比例 \(v\) 计算：

\[
s_p=
\frac18
\sqrt{
\frac{E_{\rm LPM}v}
{\rho_{\rm corr}E(1-v)}
},
\]

\[
h=\frac{\log s_p}{\log s_{1,i}},
\]

以及 \(\xi,\gamma,s,\phi(s),G(s)\) 和最终存活概率。

旧设备函数对每个顶点重新：

```text
遍历 component_hash
检查 snapshot 的所有静态字段
检查 component 的 Z、A、log constant
执行两个 pow
构造 s1
执行 log(s1)
继续动态公式
```

seed 32017 中接受的 GPU final states 为 624905，其中：

```text
annihilation       715
ionization      184772
electron pair     2432
brems           436986
```

还存在被 LPM suppression 返回 continuation 的 brems 顶点。因此静态
特殊函数会执行约几十万次，而介质实际只有少数组分。

## 3. `BremsLpmPreparedSnapshot`

`corsika/gpu/em/BremsLpm.hpp` 新增：

```cpp
struct BremsLpmPreparedComponent {
  std::uint64_t component_hash;
  double s1;
  double logarithm_s1;
};

struct BremsLpmPreparedSnapshot {
  std::uint32_t valid;
  std::uint32_t component_count;
  BremsLpmPreparedComponent components[16];
};
```

prepared snapshot 不替换原始 `BremsLpmSnapshot`。原始 snapshot 继续保存：

```text
baseline density
molecular density
E_LPM
particle masses
fine-structure constant
component Z/A/log constants
```

这样：

- Epair LPM 和其他使用者仍可访问完整原始参数；
- prepared 数据只是严格派生缓存；
- 不改变 rate-table 磁盘格式；
- 不重复保存与每粒子动态变量有关的值。

## 4. 为什么在 GPU 上准备

如果在 host 使用 `std::pow/std::log` 生成 \(s_1\)，结果可能与 CUDA
设备端双精度数学函数有若干 ulp 差异。即使 LPM 接受/拒绝没有改变，
`BremsFinalStateRecord` 中保存的存活概率也可能不再逐位相同。

初始化阶段现在调用：

```cpp
prepareBremsLpmSnapshotForCuda(snapshot, device);
```

该函数：

```text
cudaSetDevice
  -> 分配一个临时 device result
  -> 启动 1 block × 32 threads 的 preparation kernel
  -> 每个组分由一个线程计算 pow/pow/s1/log(s1)
  -> 下载 392 B prepared snapshot
  -> 立即释放临时 device allocation
  -> 校验 valid 和 component_count
```

后端将结果保存在：

```cpp
BremsLpmPreparedSnapshot brems_lpm_prepared_;
```

如果 snapshot 非法，preparation 直接抛出异常，CUDA shower 不会开始。

validation API 同样使用目标 device 生成 prepared snapshot，保证 oracle、
production 和固定种子回归走同一数值来源。

## 5. block 级 shared-memory 缓存

直接把 prepared snapshot 作为 kernel 参数后，每个线程仍需要从 constant
参数区遍历 component。实测该方案比 per-block shared 缓存慢。

最终 classification kernel 在 block 开始时执行：

```cpp
__shared__ BremsLpmPreparedSnapshot prepared_lpm;

if (threadIdx.x == 0) {
  prepared_lpm.valid = prepared_lpm_input.valid;
  prepared_lpm.component_count =
      prepared_lpm_input.component_count;
}
if (threadIdx.x < prepared_lpm_input.component_count) {
  prepared_lpm.components[threadIdx.x] =
      prepared_lpm_input.components[threadIdx.x];
}
__syncthreads();
```

随后所有 brems threads 从 shared memory 查找 `component_hash` 并读取：

```text
s1
logarithm_s1
```

资源检查：

```text
classification registers        120
classification shared bytes     392
preparation registers            38
preparation shared bytes        392
```

阶段 36 classification kernel 原来也是 120 registers，因此本阶段没有
增加 register pressure。392 B shared memory 远低于 block 配额，不改变
occupancy。

与早期“每 block 重新执行 pow/log”原型相比，最终版本：

- 从两个 barrier 减少为一个；
- 删除每 block 的两个 `pow` 和一个 `log`；
- 只在 shower 初始化时执行一次特殊函数；
- 保留 shared-memory component lookup。

## 6. warp 聚合诊断计数

classification 会统计：

```text
BremsCount
IonizationCount
AnnihilationCount
ElectronPairCount
BremsSuppressionCount
ElectronPairSuppressionCount
Epair diagnostic counts
```

原来每个成功线程直接：

```cpp
atomicAdd(global_counter, 1U);
```

现在同一分支中的活跃 warp lanes 使用：

```cpp
active = __activemask();
leader = __ffs(active) - 1;
if (lane == leader) {
  atomicAdd(counter, __popc(active));
}
```

整数总数完全相同。该改动单独实测只有约 0.6% classification 变化，不是
本阶段主要收益，但它减少了热点地址的全局原子请求，并没有增加 workspace
或改变物理记录。

Epair rejection trial count 的每线程增量不一定为 1，且总量很小，因此
继续使用原有精确 `atomicAdd(trial_count)`，避免为稀有诊断引入复杂的
warp value reduction。

## 7. 被否决的方案

### 7.1 常见/稀有过程双 kernel

曾将 brems/ionization 与 annihilation/Epair 编译为两个 specialization：

```text
common kernel registers: 67
rare kernel registers:  128
```

虽然 common kernel 的寄存器明显下降，但每个 wavefront 增加一次 full
capacity launch。实测 classification 从约 194.7 ms 上升到 202.6 ms。
此外编译形态改变使少数 thinning 临界顶点出现浮点末位变化。

该方案已经完全撤回。

### 7.2 128 classification threads/block

120 registers/thread 下，64 和 128 threads/block 的理论 resident warp
数量接近。128 threads 可减少 block 数，但会增加小 wavefront 的空 lanes。

实测：

```text
64 threads    classification 178.910 ms
128 threads   classification 186.600 ms
```

最终保留 64 threads/block。

### 7.3 gamma 静态乘积预计算

还测试了把 \(\gamma\) 公式中的三个简单常量乘积加入 prepared snapshot。
结果保持逐字节一致，但 classification 没有改善：

```text
before  178.910 ms
after   179.951 ms
```

额外字段和代码已经撤回。只保留能消除 `pow/log` 的 prepared terms。

## 8. 四组 UHE 性能结果

配置与阶段 36 相同：

```text
GPU: RTX 4060 Laptop, sm_89
CUDA: 12.6
primary: electron
energy: 1e17 / 1e18 eV
EM thinning: 1e-3
max weight: 1e6
gpu-min-batch: 64
radio: CUDA CoREAS + ZHS
detailed stage timing: enabled
```

单位为 ms：

| 能量/seed | 阶段 | classification | final state | CUDA EM kernel | total run |
|---|---:|---:|---:|---:|---:|
| \(10^{17}\), 32017 | 36 | 195.804 | 238.511 | 1525.162 | 1657.453 |
| \(10^{17}\), 32017 | 37 | 177.426 | 220.728 | 1466.252 | 1651.182 |
| \(10^{17}\), 32018 | 36 | 164.027 | 202.552 | 1403.050 | 1515.238 |
| \(10^{17}\), 32018 | 37 | 153.311 | 193.625 | 1397.984 | 1557.112 |
| \(10^{18}\), 33018 | 36 | 245.676 | 296.572 | 2135.774 | 2265.011 |
| \(10^{18}\), 33018 | 37 | 227.658 | 279.566 | 2133.706 | 2254.125 |
| \(10^{18}\), 33019 | 36 | 238.104 | 290.348 | 2240.461 | 2467.975 |
| \(10^{18}\), 33019 | 37 | 224.774 | 280.162 | 2277.207 | 2481.847 |

相对变化：

| 能量/seed | classification | final state | CUDA EM kernel | total run |
|---|---:|---:|---:|---:|
| \(10^{17}\), 32017 | -9.39% | -7.46% | -3.86% | -0.38% |
| \(10^{17}\), 32018 | -6.53% | -4.41% | -0.36% | +2.76% |
| \(10^{18}\), 33018 | -7.33% | -5.73% | -0.10% | -0.48% |
| \(10^{18}\), 33019 | -5.60% | -3.51% | +1.64% | +0.56% |

classification/final-state 在四个配对样本中同向改善。总 kernel 和 total
run 包含未修改的 transport、Molière、radio、profile、endpoint、CPU
fallback 及 WSL 调度；本轮改动只有约 10–20 ms，因此总计时被约几十 ms
的进程间波动覆盖。

## 9. 物理与确定性验收

阶段 36 与阶段 37 的四组固定种子样本分别比较。每个样本的以下 9 个
文件全部逐字节相同：

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

Phase 37 seed 32017 的独立重复运行同样 9/9 文件逐字节一致，以下诊断
也相同：

```text
gpu_particles
gpu_final_states
physical_secondaries
weighted_deposit_GeV
radio_tracks
fallback counts
observed / escaped / cut
peak resident batches
peak device bytes
```

完整 GPU 测试：

```text
ctest --output-on-failure -R '^testGpu'

24/24 passed
0 failed
```

关键 oracle：

```text
testGpuBremsFinalState:
  81391 checks
  3010 accepted
  1086 LPM-suppressed
  3 explicit fallbacks

testGpuEpairFinalState:
  122528 checks
  1024 PROPOSAL interactions

testGpuLeptonTransport:
  2697 checks
```

四个 UHE 样本均：

```text
complete = true
queue_overflows = 0
workspace_limit_checkpoints = 0
cross_species.host_spills = 0
radio.fixed_point_overflows = 0
epair_sampler.cpu_fallbacks = 0
epair_sampler.envelope_violations = 0
```

峰值显存仍为：

```text
1e17: 689034206 bytes
1e18: 823251934 bytes
```

## 10. 下一阶段

Phase 37 后主要时间仍为：

```text
Molière                 约 300–400 ms
transport physics       约 180–250 ms
final classification    约 150–230 ms
selection               约 120–270 ms
vertex                  约 100–145 ms
endpoint                约 125–185 ms
```

下一阶段应转向 transport/Molière 数据流，而不是继续给 BremsLPM 添加
未证明收益的简单常量缓存。重点审计：

1. Molière 每个投影的 Newton 特殊函数和相同 step 状态读取；
2. transport physics 与 Molière 是否重复计算能量、grammage、密度和介质
   组分状态；
3. transport control 是否能减少记录的全结构读写；
4. radio/profile 对同一 `LeptonTransportRecord` 的多次全量读取是否能合并。

所有后续改动继续要求：

- 双精度、无 `fast-math`；
- Philox 键不变；
- 固定种子科学输出逐字节回归；
- 24 项 GPU 测试；
- 四组 UHE 配对性能验证；
- 非法 snapshot/table/CUDA 状态立即终止。
