# 阶段 88：确定性强子调度、profile/radio 原子优化与 `emthin=1e-6` 正式运行

## 1. 本阶段结论

本阶段解决了一个会直接破坏固定种子可重复性的强子调度问题，并在不改变物理过程
集合的前提下继续优化了高质量 shower 的 GPU profile 和射电投影。

当前主要结论是：

1. 强子任务仍按模型、粒子种类、能量和核质量分类存储，并由四个独立 FLUKA
   进程执行；
2. 调度器不再把本次运行测得的 wall-clock 时间反馈给后续批次，因此同一非零
   seed 的 shower 主体重新达到逐字节可重复；
3. 1 PeV、`emthin=1e-6`、81 个 21CMA 天线、CUDA CoREAS/ZHS 的完整事例已经
   成功运行；
4. 该高质量事例中四个 FLUKA worker 的累计末态计算时间只相差 0.71%，说明
   “按类别存储并让各 worker 获得接近计算量”的目标已经达到；
5. FLUKA 进程池 wall time 只占完整 shower 的 0.175%，当前性能门槛已经不是
   强子末态，而是 GPU 射电投影和 profile 累计；
6. 新冻结二进制的 500+500 原版 CPU/CUDA 独立样本中，九条平均曲线全部通过，
   九个关键分布的 KS 检验全部通过，六个关键均值已经进入 1% 门限；其余三个
   高方差尾量仍是统计不足，不能写成已通过，也没有达到拒绝同分布的证据。

500+500 ensemble 和 `emthin=1e-6` 正式事例使用的冻结二进制：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower
SHA256:
b8bf232591a68a567938376907564b3343a04ed0fdca63f262683a4e8029610d
```

随后只修改了一处 fallback 日志字符串并重建。当前二进制为：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower
SHA256:
0356ee5b2579cb9e009292ddce03e100f47a574e679bbacfb3ba9e46583e23a2
```

第 6.6 节给出新旧二进制固定 seed 物理文件逐字节相同的证据。

之后增加了不参与物理决策的 radio 轨迹诊断，并修正了 GPU radio metadata
把 μ 轨迹误计入 `segment_count` 的问题。最新诊断二进制为：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower
SHA256:
4c809f11876a4d0f281fd1acd6a74ea5a2c754abcc07a5c1fc77e28af189ad98
```

波形 kernel 从始至终都只接受 \(e^\pm\)；计数修复不代表旧波形包含 μ，只代表
旧 metadata 的分母偏大。第 6.8 节给出新诊断的定义和控制结果。

原版比较二进制：

```text
/home/yuhanglu/21CMA/corsika-21cma/corsika-build/applications/c8_air_shower
SHA256:
133daa7ee0a4de4ba7355df5facb44e9a09e23bb5fff8b9549e7b6da546daeb9
```

GPU 物理表：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/
  production_v10_muons_1e-3_1EeV.c8emrt
SHA256:
14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521
```

## 2. 修复强子调度的固定种子不确定性

### 2.1 问题

原来的分类队列会用已经完成的 FLUKA 请求实测耗时更新某一 work class 的预计
cost，再用这个 cost 决定：

- 一个 homogeneous batch 中放多少请求；
- 该 batch 分给哪个 worker；
- 何时达到目标 batch cost。

物理随机数本身并没有使用 wall-clock 时间，但操作系统调度抖动会使实测耗时略有
不同。预计 cost 随之改变后，后续 batch 边界和 worker 分配会改变。FLUKA worker
是有内部状态的独立进程，因此相同 seed 最终可能产生不同 shower。

修复前，同一个 100 TeV proton、seed `24680` 连续运行两次：

```text
determinism_probe_100TeV_seed24680_atomic_a
determinism_probe_100TeV_seed24680_atomic_b
```

已经出现：

- GPU 粒子数：2,345,508 对 2,394,396；
- FLUKA prepared request：2,431 对 2,324；
- batch 数：803 对 729；
- particles、profile 和 dEdX Parquet 哈希不同。

这不是合理的 Monte Carlo fluctuation，因为输入 seed 完全相同；它是调度实现缺陷。

### 2.2 修复

相关文件：

```text
corsika/framework/core/HadronicWorkQueue.hpp
corsika/detail/framework/core/HybridCascade.inl
tests/framework/testHadronicWorkQueue.cpp
```

新增：

```cpp
double deterministicHadronicInteractionCost(
    HadronicWorkKey const& key,
    double reference_cost);
```

cost 现在只依赖：

```text
(model, species, energy_bin, mass_number_bin, reference_cost)
```

其中 species 至少区分：

- nucleon；
- pion；
- kaon；
- strange baryon；
- nucleus；
- other。

系数使用 FLUKA 2025 profiling 中平滑的物种/能段趋势，只作为调度提示，不参与
截面、过程选择、随机数或末态生成。实测时间仍写入 metadata，但只用于事后
profiling，不再进入本次 shower 的调度决策。

`HybridCascade::estimateHadronicInteractionCost()` 现在是：

```cpp
return deterministicHadronicInteractionCost(
    key,
    hadronic_worker_config_.initial_interaction_cost_ms);
```

### 2.3 修复后的逐字节证据

同一 100 TeV proton、seed `24680` 连续运行：

```text
determinism_probe_100TeV_seed24680_deterministic_cost_c
determinism_probe_100TeV_seed24680_deterministic_cost_d
```

两次均得到：

- GPU 粒子数：2,317,802；
- FLUKA prepared request：2,948；
- batch 数：945；
- worker assignment 数：217、235、244、249；
- particles、profile、dEdX、interactions 和 production profile SHA256
  全部一致。

两次 wall time 可以不同，分别约 1,172.2 ms 和 1,166.1 ms；计时不是物理输出，
也不再改变调度。

需要保留的语义边界是：

- 同一 CUDA 配置、同一 seed：要求逐次可重复；
- 原版 scalar 与 CUDA hybrid：随机流组织不同，不要求单 shower 相同，要求
  ensemble 统计一致；
- worker 数改变时仍使用 history-keyed 随机流和稳定 sequence ID，不以完成先后
  写回主栈。

## 3. profile/radio 固定点原子累加优化

### 3.1 原实现的瓶颈

相关文件：

```text
src/gpu/em/CudaProfileProjection.cu
src/gpu/em/CudaRadioAccumulator.cu
```

确定性 profile 和 radio 使用 64-bit signed fixed point。旧代码为了检查 signed
overflow，对每个累加执行：

1. load 当前值；
2. 在 host 语义下检查加法溢出；
3. `atomicCAS`；
4. CAS 失败后重新读取并重试。

在 shower maximum 附近，大量线程竞争同一个 longitudinal bin；在一个射电脉冲
附近，大量轨迹也竞争相同时间 bin。CAS retry 会把竞争进一步放大。

### 3.2 新实现

现在使用一次原生 64-bit：

```cpp
auto previous_bits = atomicAdd(storage, increment_bits);
```

CUDA `atomicAdd` 返回在线性化点之前的旧值，因此可以用返回值检查：

```cpp
(increment > 0 && previous > LLONG_MAX - increment) ||
(increment < 0 && previous < LLONG_MIN - increment)
```

若出现溢出：

- 设备计数器增加；
- shower 下载计数器时立即失败；
- 已 wrap 的输出不会被当成有效科研结果。

在没有溢出的正常运行中，fixed-point 整数和与累加顺序无关，物理值没有改变；
代码仍未启用 fast-math 或混合精度。

### 3.3 1 PeV 归一化性能证据

早期 CAS 与新 atomic-add 的 1 PeV 事例因当时尚未发现的强子调度问题不是同一个
shower，不能把总 wall time 当严格 A/B。但用处理量归一化可以定位原子瓶颈：

| 指标 | CAS | atomic add | 改善 |
|---|---:|---:|---:|
| profile time / GPU step | 0.642 μs | 0.309 μs | 2.08× |
| radio time / logical pair | 6.01 ns | 4.25 ns | 1.41× |

同时：

- fixed-point overflow 均为 0；
- queue overflow 均为 0；
- NaN/负能量保护没有触发。

## 4. CoREAS/ZHS 共享轨迹运动学

### 4.1 实现

相关文件：

```text
corsika/gpu/radio/Types.hpp
src/gpu/em/CudaRadioAccumulator.cu
applications/c8_air_shower.cpp
tests/gpu/testGpuRadioProjection.cpp
```

当 CoREAS 与 ZHS 的 observer 集合具有完全相同的：

- observer 数；
- 位置；
- start time；
- duration；
- sampling rate；
- bin 数；

后端使用一个 `coreasZhsKernel`。每个线程仍只负责一个 track-observer pair，但：

1. 只读取一次 `LeptonTransportRecord`；
2. 只计算一次起止位置、位移、时长、\(\boldsymbol\beta\)、\(|\beta|\)、轨迹长度、
   电荷和权重常量；
3. 分别调用原 CoREAS endpoint 和 ZHS segment 公式；
4. 分别写入两块波形数组。

若 observer snapshot 不同，就自动回到两个独立 kernel，不能错误地共享几何。

新增 metadata：

```yaml
radio:
  track_observer_pairs: <CoREAS+ZHS 的逻辑总数>
  fused_track_observer_pairs: <一次共享预处理的物理线程对数>
```

### 4.2 固定种子严格 A/B

参数：

- 100 TeV proton；
- zenith 27°、azimuth 180°；
- seed `13579`；
- `emthin=1e-5`；
- 81 个 observer；
- CUDA CoREAS 和 ZHS；
- 四个 FLUKA worker。

输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  radio_fusion_ab_100TeV_seed13579_atomic_only_v1
  radio_fusion_ab_100TeV_seed13579_fused_v2
```

结果：

| 指标 | 独立 kernel | 融合 kernel | 改善 |
|---|---:|---:|---:|
| 外部 wall time | 70.71 s | 68.35 s | 3.34% |
| shower internal | 65.515 s | 63.085 s | 3.71% |
| radio device time | 53.670 s | 50.587 s | 5.74% |
| logical radio pairs | 13,219,753,554 | 13,219,753,554 | 相同 |
| fused physical pairs | 0 | 6,609,876,777 | 新路径 |

物理文件 SHA256：

- particles：相同；
- longitudinal profile：相同；
- production profile：相同；
- dEdX：相同；
- interactions：相同；
- ZHS waveform Parquet：相同。

CoREAS 的浮点下载值不是逐字节相同。合并 kernel 后寄存器分配和指令组合使少数
单轨迹贡献在 fixed-point rounding 边界的最后几位不同：

- 最大绝对差：\(1.97\times10^{-17}\,\mathrm{V/m}\)；
- 最大差相对该分量峰值：\(1.90\times10^{-10}\)；
- RMS 差相对 RMS signal：\(8.20\times10^{-11}\)；
- 所有波形有限；
- fixed-point overflow 为 0。

这比 1% 物理验收门限小约八个数量级，并通过既有 scalar-radio 单元比较；因此保留
融合路径，但不能把 CoREAS 文件描述成 bitwise identical。

## 5. 冻结二进制的 500+500 当前统计验收

输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_frozen_deterministic_atomic_fused_vs_original_proton_1TeV_500_v4
```

参数：

- 1 TeV proton；
- zenith 27°、azimuth 180°；
- 500 个原版 CPU 独立 seed；
- 500 个 CUDA 独立 seed；
- `emthin=1e-4`、`max-weight=100`；
- 无 radio；
- 原版使用 scalar FLUKA；
- CUDA 使用四个分类 FLUKA worker；
- 所有进程设置单线程 BLAS/OMP。

结果：

| observable | CPU mean | CUDA mean | CUDA−CPU | \(|z|\) | KS / 95% 临界值 |
|---|---:|---:|---:|---:|---:|
| profile \(X_{\max}\) (g/cm²) | 353.620 | 353.964 | +0.097% | 0.044 | 0.040 / 0.086 |
| charged \(N_{\max}\) | 649.676 | 648.079 | −0.246% | 0.136 | 0.040 / 0.086 |
| charged profile integral | 203,997 | 202,588 | −0.690% | 0.421 | 0.066 / 0.086 |
| photon profile integral | 1,070,001 | 1,059,754 | −0.958% | 0.499 | 0.044 / 0.086 |
| total energy deposit (GeV) | 692.015 | 692.298 | +0.041% | 0.041 | 0.042 / 0.086 |
| dE/dX peak depth | 176.832 | 184.146 | +4.136% | 0.693 | 0.062 / 0.086 |
| ground EM weighted count | 421.576 | 387.850 | −8.000% | 0.881 | 0.038 / 0.086 |
| ground EM kinetic energy (GeV) | 17.234 | 15.832 | −8.136% | 0.541 | 0.054 / 0.086 |
| energy closure fraction | 0.709276 | 0.708154 | −0.158% | 0.178 | 0.048 / 0.086 |

判读：

- 九条预定义平均曲线全部通过；
- 九个关键 scalar 的经验 KS 距离全部低于 95% 临界值；
- 没有一个关键均值达到 3σ；
- 六个关键量通过严格 1% 均值门限；
- dE/dX peak depth 和两个 ground tail 的相对方差很大，当前 500+500 对 1%
  bias 根本没有足够精度；
- 这三个量只能标记为
  `relative_threshold_failed_but_statistically_inconclusive`。

正式状态仍是：

```text
failed: strict 1% scalar gate not complete
physics distributions: no rejection evidence
```

这比旧二进制的五项通过改善为六项通过，特别是 \(X_{\max}\) 已从临界的 1.020%
变为 0.097%，但不能把三个高方差量省略后宣称全部验收完成。

同一批运行中，CUDA 500 shower 用时 312.6 s。原版每个 50-shower 单核 shard
用时 135.7–154.9 s，对应约 2.71–3.10 s/shower；CUDA 连续运行约
0.625 s/shower。由于 CPU shard 与 CUDA 重叠运行，这不是严格独占硬件 benchmark，
只能作为约 4.5× 的批量吞吐量指示。

## 6. `config.yaml` 等价的 `emthin=1e-6` 正式事例

输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  config_yaml_equiv_proton_1PeV_emthin1e-6_seed2718281828_cuda_frozen_v1
```

参数：

```text
primary        proton (PDG 2212)
energy         1e6 GeV = 1 PeV
zenith         27 deg
azimuth        180 deg
core           (0, 0) m
emthin         1e-6
seed           2718281828
antennas       /home/yuhanglu/21CMA/data/antennas.txt
radio          CUDA CoREAS + CUDA ZHS
hadronic       SIBYLL 2.3d + FLUKA 2025.0.0
FLUKA workers  4
```

注意用户 YAML 顶层 `energy_range` 写为 `1e9` GeV，但 task 的实际参数是
`-E 1e6`，本次严格按 task 模拟 1 PeV。

### 6.1 运行和 GPU 负载

| 指标 | 数值 |
|---|---:|
| 外部 wall time | 676.61 s |
| shower internal | 671.43 s |
| GPU particles | 933,230,048 |
| GPU final states | 137,572,096 |
| physical secondaries | 275,160,361 |
| CPU particle steps | 181,776 |
| CPU small-wavefront expansion | 43,728 |
| peak device bytes | 5,210,688,716 |
| queue overflow | 0 |
| memory spill | 0 |
| radio tracks | 809,292,501 |
| logical track-observer pairs | 135,247,483,458 |
| fused physical pairs | 67,623,741,729 |
| radio device time | 528.68 s |
| profile steps | 933,230,048 |
| profile kernel time | 515.93 s |
| fixed-point overflow | 0 |

radio 与 profile 在不同 CUDA stream 中有重叠，不能把两项时间相加。radio device
time 占 internal wall time 约 78.7%，是当前最明确的门槛。

与先前 1 PeV、`emthin=1e-5` atomic-add 事例相比：

- GPU 粒子数增加 4.26×；
- internal time 增加 4.21×；
- 每个 GPU step 成本基本保持线性，没有在 9.33 亿 step 时突然恶化。

本事例 metadata 中：

```yaml
thinning:
  em_fraction: 1e-6
  maximum_weight: 0.5
  hillas_vertices: 0
  statistical_vertices: 0
```

这里需要区分“参数被接受”与“实际发生薄化”。只指定 `--emthin` 确实是原版
支持的有效用法，CUDA 也没有要求额外指定 `--max-weight`。但是原版和 CUDA
都在 `parentWeight >= maxWeight` 时直接返回。本事例自动
`maximum_weight=0.5`，初始权重为 1，因此实际没有 thinning；`hillas_vertices`
和 `statistical_vertices` 均为 0 正是直接证据。它比 `1e-5` 处理更多粒子、
运行更慢、数据质量更高，但在 1 PeV 这个精确边界上应称为未薄化极限，而不是
“已经发生的更弱薄化”。

### 6.2 强子分类与负载平衡

| worker | requests | batches | FLUKA final-state time |
|---:|---:|---:|---:|
| 0 | 6,568 | 1,849 | 608.005 ms |
| 1 | 6,186 | 2,010 | 608.686 ms |
| 2 | 6,091 | 2,064 | 604.403 ms |
| 3 | 6,018 | 2,074 | 607.722 ms |

请求数并不相同，因为不同 species/energy class 的预计成本不同；累计实际计算时间
却被平衡到：

```text
max/min = 1.0071
```

即最大和最小只差 0.71%。这正是分类成本调度应该达到的效果。

完整强子统计：

- SIBYLL interactions：1,928；
- FLUKA interactions：24,863；
- process pool execute wall：1.173 s；
- FLUKA 累计末态 CPU 时间：2.429 s；
- process pool execute / shower internal：0.175%；
- 四 worker 全部 clean exit。

因此在这个 1 PeV 高质量事例上，继续把 SIBYLL/FLUKA 整体移植 GPU 预计不会带来
可见端到端收益。当前更合理的是保留 CPU 分类进程池，把优化资源投入 radio。

### 6.3 回退与完整性

记录到：

- `unsupported_geometry`：2 个，显式返回 CPU scalar transport；
- CPU-only process：7,593 个；
- inverse-CDF unavailable：794 个；
- loss quantile out of range：773 个；
- loss energy out of range：121 个；
- epair rejection envelope exceeded：4 个；
- specified CPU final state：9,285 个；
- forced CPU decay：14,177 个。

未 GPU 化过程没有被忽略。允许的几何回退会保持 history/step identity 返回主栈，
其余指定末态由 CPU PROPOSAL/光核模型处理。两个几何回退原日志中的
`unhandled` 只是措辞错误，源代码随后改成
`CUDA EM fallback returned to scalar transport`。

### 6.4 射电输出检查

CoREAS 和 ZHS 均输出：

- 81 个 observer；
- 每个 observer 400 个时间 bin；
- 总计 32,400 行；
- `Time, Ex, Ey, Ez` 全部有限；
- 没有 fixed-point overflow。

全局波形范围：

| algorithm | Ex (V/m) | Ey (V/m) | Ez (V/m) |
|---|---:|---:|---:|
| CoREAS | \([-7.34\times10^{-7},1.08\times10^{-6}]\) | \([-9.68\times10^{-7},1.04\times10^{-5}]\) | \([-5.31\times10^{-7},3.62\times10^{-7}]\) |
| ZHS | \([-8.25\times10^{-7},8.46\times10^{-7}]\) | \([-1.12\times10^{-6},9.18\times10^{-6}]\) | \([-9.68\times10^{-7},7.53\times10^{-7}]\) |

这只证明单事例输出完整和数值有限，不等于已经证明 CoREAS/ZHS 或 CPU/CUDA
脉冲分布统计一致。后者仍需要多个独立 seed，并用用户的
`pulse_analysis_modular` 定义比较地磁分量振幅和方波拟合宽度。

### 6.5 能量 ledger 的限制

本完整 hadronic shower 的设备 ledger 报告：

```yaml
complete_coverage: false
accepted: false
relative_closure_error: 0.129
```

该字段当前只完整覆盖 GPU EM terminal、deposit、cut rest mass、observation 和
escape；它没有完整纳入所有 CPU 强子/μ/衰变路径，所以不能把 12.9% 当成物理能量
不守恒，也不能把该 ledger 用于纯 EM 的 \(10^{-4}\) closure 验收。

正确的现有统计检查是与原版比较相同定义的 `energy_closure_fraction`；500+500 中
原版均值 0.709276，CUDA 均值 0.708154，差 −0.158%、0.178σ，KS 通过。

生产化前仍应补齐统一的 hadron/μ/decay energy ledger，使
`complete_coverage=true` 后再对完整 shower 使用绝对 closure gate。

### 6.6 日志措辞重建后的非物理回归

日志字符串修改后，用当前二进制重新运行第 4.2 节完全相同的 100 TeV、
seed `13579` 事例：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  radio_fusion_ab_100TeV_seed13579_post_log_wording_v3
```

wall time 为 68.28 s。它与修改前 `fused_v2` 的下列文件 SHA256 全部相同：

- `particles/particles.parquet`；
- `profile/profile.parquet`；
- `production_profile/profile.parquet`；
- `energyloss/dEdX.parquet`；
- `interactions/interactions.parquet`；
- `CoREAS/observers.parquet`；
- `ZHS/observers.parquet`。

因此从冻结 ensemble 二进制 `b8bf...9610d` 到当前二进制 `0356...23a2` 的唯一
源码差异没有改变 shower、profile 或射电波形。

### 6.7 相同 EM 轨迹上的 CPU/CUDA 射电严格控制实验

输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_radio_backend_equivalence_electron_1TeV_10_atomic_fused_v2
```

该实验不是比较两个独立 shower ensemble。它用同一个当前二进制、同一个 seed
`424242` 和完全相同的 CUDA EM 输运运行两次，唯一变化是：

```text
--radio-backend cpu
--radio-backend cuda
```

参数：

- 10 个 1 TeV electron shower；
- zenith 27°、azimuth 180°；
- `emthin=1e-4`、`max-weight=100`；
- 81 个 observer；
- CoREAS 和 ZHS 同时输出。

五类非射电输出：

- particles；
- longitudinal profile；
- production profile；
- dEdX；
- interactions；

不但行数相同，而且 Parquet SHA256 全部相同。

逐 shower、逐 observer、逐 \(E_x/E_y/E_z\) 比较共：

```text
10 showers × 81 observers × 3 components × 2 algorithms
= 4,860 component comparisons
```

结果：

| 指标 | CoREAS | ZHS | 门限 |
|---|---:|---:|---:|
| 最大 normalized difference | \(5.81\times10^{-8}\) | \(1.20\times10^{-6}\) | \(10^{-4}\) |
| 最大 relative L2 | \(4.31\times10^{-8}\) | \(4.69\times10^{-7}\) | \(10^{-4}\) |
| 最大 relative fluence difference | \(1.51\times10^{-8}\) | \(2.59\times10^{-7}\) | \(5\times10^{-4}\) |
| 最大 absolute difference (V/m) | \(2.21\times10^{-16}\) | \(3.38\times10^{-15}\) | 仅作诊断 |
| accepted | true | true | — |

端到端时间：

| radio backend | 10 shower wall time | 每 shower |
|---|---:|---:|
| CPU | 1,068.53 s | 106.85 s |
| CUDA | 15.73 s | 1.57 s |

同轨迹射电后端端到端加速：

```text
1068.53 / 15.73 = 67.9×
```

内部十个 shower 累计时间分别为 1,063.67 s 和 10.72 s，对应 99.2×；外部
67.9× 还包含两次相同的模型初始化和输出。

第一次控制运行沿用了验证脚本旧默认 `mucut=1e13 GeV`，而生产 EM+μ 表 metadata
要求 0.3 GeV。应用能力检查立即中止并把输出标成 incomplete，没有静默切换后端。
验证脚本默认值随后改为 0.3 GeV，并增加说明“必须匹配 μ 表 metadata”。

### 6.8 轨迹转角、时间残差与 radio 偏差归因

为避免把独立 shower 的涨落误判成 GPU radio 公式错误，CPU `RadioProcess` 和
CUDA resident-track 路径现在用相同定义记录：

- \(e^\pm\) segment 数、权重和总轨迹长度；
- electron/positron 与 signed-charge 加权轨迹长度；
- \(\sum w\Delta\theta\) 和 \(\sum w\Delta\theta^2\)；
- \(\sum wL(1-\beta)\)；
- \(\sum w(\Delta t-L/c)\)；
- 最大 segment 长度和最大转角；
- 分能段加权轨迹长度；
- signed-charge 加权的三维方向变化。

CUDA 路径只在显式指定：

```text
--gpu-radio-track-diagnostics
```

时执行这些额外 reduction，生产运行默认关闭。相关实现和分析代码：

```text
corsika/modules/radio/RadioProcess.hpp
corsika/detail/modules/radio/RadioProcess.inl
corsika/gpu/radio/Types.hpp
src/gpu/em/CudaRadioAccumulator.cu
validation/gpu_em/analyze_cpu_cuda_radio.py
validation/gpu_em/pool_radio_ensembles.py
```

同 seed `2800001` 的 5+5 个 1 TeV proton 诊断输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_scalar_proton_1TeV_seed2800001_angular_trackdiag5_v2
  current_cuda_proton_1TeV_seed2800001_angular_trackdiag5_v1
```

需要强调：CPU 与 CUDA 在首次 EM 抽样后随机流组织不同，所以相同 seed 不是成对
相同 shower。5+5 只能定位数量级，不能作为 ensemble 验收。

| 每 shower 的 intensive 轨迹量 | CPU | CUDA | CUDA/CPU−1 |
|---|---:|---:|---:|
| 权重平均 segment 长度 | 7.551 m | 7.312 m | −3.16% |
| 权重平均单段转角 | 0.36723 rad | 0.36977 rad | +0.69% |
| 权重 RMS 单段转角 | 0.49491 rad | 0.49721 rad | +0.47% |
| 转角/加权轨迹长度 | 0.04915 rad/m | 0.05480 rad/m | +11.50% |
| 加权 \(1-\beta\) | 0.008394 | 0.008630 | +2.80% |
| 权重平均时间残差 | \(2.253\times10^{-10}\) s | \(2.230\times10^{-10}\) s | −1.05% |
| 轨迹加权平均动能 | 0.2645 GeV | 0.2346 GeV | −11.28% |

单段转角和 RMS 转角分别只差 0.69% 和 0.47%，bootstrap 区间均包含 0；
这不支持“CUDA Molière 多重散射发生数量级过强”这一解释。转角/长度较高主要同时
伴随 CUDA 平均 segment 较短和轨迹加权能量较低。

这个 5+5 样本的 shower 主体本身并不平衡：

- CPU/CUDA 平均能量沉积为 751.4/602.3 GeV，差 −19.8%；
- charged \(X_{\max}\) 为 364.9/483.2 g/cm²；
- weighted track length 差 −28.1%；
- raw radio band energy 的 CUDA/CPU 差约 +76% 到 +100%，但只有 5 个样本，
  bootstrap 区间很宽。

因此该样本不能把 radio 差异归到 GPU 投影，也不能作为统计一致或不一致的正式
结论。结合第 6.7 节完全相同轨迹上的严格投影比较，当前证据链是：

1. CPU/CUDA radio 数学投影在相同轨迹上已经通过；
2. CUDA 单位轨迹角散射和时间残差未显示大偏差；
3. 独立 proton ensemble 仍需更多样本，且必须同时检查 shower 主体是否平衡；
4. 只有主体分布通过后，才能解释 radio 振幅、宽度和 radiation-energy 分布。

诊断汇总：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_cuda_proton_1TeV_seed2800001_angular_trackdiag5_v1/
    paired_radio_and_angular_track_diagnostic.json
    physics_comparison/comparison.json
```

## 7. 已执行测试

```text
testGpuPhotonWavefront     passed
testGpuRadioProjection     passed
testOutput                 passed
testFramework              passed
Python radio runner tests  3 passed
```

`testGpuRadioProjection` 额外检查：

- compatible CoREAS/ZHS observer 确实使用 fused kernel；
- logical pair 和 fused pair 计数正确；
- reset 后统计清零；
- 同一 GPU 上 deterministic waveform 重复运行逐位一致；
- scalar/CUDA CoREAS 和 ZHS 单轨迹比较通过；
- μ transport record 不再计入 \(e^\pm\) radio segment 数；
- 轨迹转角、\(\beta\) deficit 和时间残差诊断有限且定义一致；
- fixed-point overflow 为 0。

## 8. 下一步

1. 为完整 hadronic shower 补齐 CPU strong、μ 和 decay 的统一 energy ledger；
2. 使用多个独立 seed 做 1 PeV radio ensemble：
   - `emthin=1e-5` 只用于较快的开发和 scaling 趋势；
   - `emthin=1e-6` 保留为 1 PeV 高质量正式验收档；该写法会被原样接受，
     但自动 `maximum_weight=0.5` 使其实际未薄化，运行更慢且数据质量高于
     `1e-5`；
   - 地磁振幅和宽度严格调用用户的 `pulse_analysis_modular`；
3. 继续优化当前真正门槛：
   - radio 传播表查找；
   - CoREAS endpoint 和 ZHS subdivision 的共同传播量；
   - observer/track 二维分块与缓存；
   - profile 与 radio 的 stream overlap；
4. 除非新的 profile 显示 strong final state 重新超过 10% wall time，否则不启动
   SIBYLL/FLUKA GPU 移植。当前 0.175% 的占比不支持该投入。
