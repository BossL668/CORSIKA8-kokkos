# CUDA 电磁后端重构记录：阶段 10，photon pair 末态

下一阶段记录见
[`phase_11_photon_pair_lpm.md`](phase_11_photon_pair_lpm.md)。

## 1. 本阶段结果

本阶段在阶段 9 的“已选离散相互作用记录”之后，第一次生成了真实的 GPU
物理次级粒子：

\[
\gamma \rightarrow e^- + e^+.
\]

当前数据流为：

```text
EmInteractionRecord batch
          │
          │ process capability map
          ├───────────────┬────────────────┐
          ▼               ▼                ▼
 GPU Photopair       CPU fallback     NoDiscreteInteraction
          │               │                │
          │ rho table     │                │
          │ Sauter angle  │                │
          ▼               ▼                ▼
   e-/e+ secondaries  specified event   continuation
```

实现同时完成：

- GPU/CPU 过程能力表；
- CPU-only 稀有末态的显式路由；
- 独立 photon-pair 末态能量分配表；
- Sauter 角分布；
- CUB 稳定次级粒子编号；
- 能量闭合、随机数确定性和 history ID 验证。

这仍是开发验证路径。真实末态还没有替换
`CudaEmBackend::advanceWavefront()` 中的 toy branching。

## 2. 为什么不能直接把阶段 9 的 \(v\) 当成能量分配

实现过程中发现了一个必须明确区分的 PROPOSAL 语义。

阶段 9 保存：

```text
EmInteractionRecord::energy_fraction
```

它来自：

```cpp
CrossSectionBase::CalculateStochasticLoss(...)
```

对于 PROPOSAL 的 photon pair production，该过程是
`only-stochastic`，所以光子被完全吸收：

\[
v_{\mathrm{interaction}}=1.
\]

这个 \(v\) 仍需要保留，因为现有 CORSIKA CPU 路径把它用于已选相互作用记录和
LPM 判断。但它不是电子与正电子之间的能量分配。

CPU `PhotoPairProductionKochMotzSauter` 还会使用另一个随机数调用：

```cpp
PhotoPairProductionKochMotzSauter::CalculateRho(E, u, component)
```

得到：

\[
\rho = \frac{E_{e^-}}{E_\gamma},
\qquad
E_{e^-}=\rho E_\gamma,
\qquad
E_{e^+}=E_\gamma-E_{e^-}.
\]

因此 GPU 末态使用新的 Philox draw 独立抽取 \(\rho\)，没有重用阶段 9 的
\(v\)。这也保持了 CPU 路径中“相互作用选择变量”和“末态变量”相互独立的
统计关系。

## 3. photon-pair 末态辅助表

### 3.1 零 rate 辅助 column

当前 v3 文件格式已经允许“零 rate 但带 inverse-CDF”的 column。本阶段使用：

```text
physical Photopair process ID       1000000013
final-state auxiliary process ID    2000000013
```

每个 dry-air target component 增加一个辅助 column：

```text
process_name       PhotopairFinalState
parameterization   PhotoPairProductionKochMotzSauter
rates              全部为 0
inverse-CDF        rho(E,u)
```

它的 rate 恒为零，所以：

- 不改变总反应率；
- 不改变 process/component 抽样概率；
- 永远不会被阶段 9 选择成物理过程；
- 只通过明确的 final-state 查询访问。

这样避免把两种不同含义的 inverse-CDF 混在一起，也不需要在设备视图中复制
第二套表格布局。

### 3.2 生成与独立验证

`gpu_em_tablegen` 版本更新为：

```text
c8-gpu-em-tablegen-0.4
```

生成器直接调用与 CPU 默认末态相同的：

```cpp
PROPOSAL::secondaries::
    PhotoPairProductionKochMotzSauter::CalculateRho()
```

对 N、O、Ar 三个 PROPOSAL component 分别建立二维自适应表：

\[
\rho(E,u).
\]

行内使用 `logit(u)` 和 `log(rho)` 插值，行间使用 `log(E)` 和
`log(rho)` 插值。自适应 probe 之后，再使用没有参与细化的低差异样本独立
验证。

### 3.3 首版能力边界

PROPOSAL 的 \(\rho(E,u)\) 在低能阈值附近随能量快速变化，高 \(Z\) 的 Ar
组分在数 GeV 仍有明显结构。强行用当前坐标表示整个能区，会需要超过一万个
energy rows，且不能在配置的 4096 点上限内达到 \(10^{-3}\)。

首版因此明确限定：

```text
E_gamma >= 10 GeV
1e-4 <= u <= 1 - 1e-4
```

域外事件产生有原因、有计数的 CPU fallback。没有夹紧能量或分位点，也没有
使用均匀能量分配替代 PROPOSAL。

这项限制适合当前优先加速 UHE shower 高能前沿的目标。低能域后续可改用
归一化 pair asymmetry 变量，降低阈值附近的能量曲率后再扩展。

## 4. GPU process capability map

`ProcessCapabilities.hpp` 不包含 PROPOSAL host-only 头文件，只保存
PROPOSAL 7.6.2 稳定过程号和设备可调用的分类函数。

当前分类为：

| 类别 | 过程 |
|---|---|
| GPU 已实现 | photon `Photopair` |
| CPU-only 物理末态 | Photonuclear、MuonPair、Hadrons、WeakInt、Decay、Photoproduction、PhotoMuPair |
| GPU 尚未实现 | Brems、Ioniz、Epair、Compton、Annihilation、Photoeffect 等 |

后两类都会生成 `ProposalFallbackEvent`，但原因不同：

```text
CpuOnlyProcess
GpuProcessNotImplemented
```

这使输出统计能够区分“设计上必须由 CPU 处理”和“后续应继续 GPU 化”。

fallback 保留：

- 原粒子及 history；
- 原 batch `input_index`；
- 已选物理 process ID；
- 64 位 component hash；
- 相互作用 \(v\)；
- 相互作用分位点和 draw ID；
- 已抽取的 final-state uniform 和 draw ID。

因此未来 CPU `ProposalFinalStateGenerator` 可以消费已经选定的指定末态，不
需要重新抽取过程。

## 5. 末态随机数

photon pair 使用与父粒子物理身份绑定的 Philox key：

```text
(seed, shower_id, parent history_id, parent step_id,
 Photopair process_id, draw_id)
```

draw 分配固定为：

| draw ID | 用途 |
|---:|---|
| 0 | 阶段 9 的相互作用 \(v\) |
| 1 | 末态能量分配 \(\rho\) |
| 2 | 共同方位角 |
| 3 | 电子 Sauter 极角 |
| 4 | 正电子 Sauter 极角 |

输入 batch 顺序改变不会改变同一个 parent history 的 \(\rho\) 或三个角随机数。

## 6. Sauter 角分布

对每个次级粒子，定义：

\[
p=\sqrt{E^2-m_e^2},
\qquad
q=2u-1.
\]

PROPOSAL 使用的 Sauter 极角为：

\[
\cos\theta=\frac{Eq+p}{pq+E}.
\]

在 \(E\gg m_e\) 时，分子分母几乎相等，直接计算会发生灾难性消减。GPU
实现使用等价的稳定形式：

\[
1-\cos\theta
  = \frac{(E-p)(1-q)}{E+pq},
\qquad
E-p=\frac{m_e^2}{E+p}.
\]

电子和正电子共享一个随机方位角，二者方位相差 \(\pi\)，极角分别使用独立
随机数。局部方向再旋转到父光子的全局方向。

这次稳定改写把完整 UHE 表上最初出现的约 \(10^{-9}\) CPU/GPU 方向差消除到
测试的 \(10^{-12}\) double 精度尺度内。构建仍未使用 `--use_fast_math`。

## 7. CUB 分类与次级编号

第一个 kernel 为每个输入生成三个计数：

```text
child_count       2 或 0
fallback_flag     1 或 0
continuation_flag 1 或 0
```

三次 CUB exclusive scan 得到三个稳定 offset。三类计数满足：

\[
\frac{N_{\mathrm{children}}}{2}
  + N_{\mathrm{fallback}}
  + N_{\mathrm{continuation}}
  = N_{\mathrm{input}}.
\]

成功 photon pair 的 history ID 为：

```text
electron history = first_secondary_history_id + child_offset
positron history = electron history + 1
```

输出严格保持输入顺序。实现检查：

- 32 位 scan offset 上限；
- 64 位 history ID 溢出；
- `generation` 溢出；
- 父方向有限且归一；
- \(E_\gamma\ge2m_e\)；
- 两个次级粒子均不低于静止质量；
- \(\rho\) 有限且位于 \((0,1)\)。

history ID 恰好使用到 `UINT64_MAX` 是合法的；只有真正越界才失败。

## 8. 输出记录

新增：

```cpp
struct PhotonPairFinalStateRecord {
  std::uint64_t input_index;
  std::uint64_t parent_history_id;
  std::uint64_t secondary_offset;
  std::uint32_t secondary_count;
  double energy_split_fraction;
  double split_uniform;
  double azimuth_uniform;
  double electron_polar_uniform;
  double positron_polar_uniform;
  // fixed draw IDs
};
```

以及：

```cpp
struct EmFinalStateBatchResult {
  std::size_t input_interactions;
  std::size_t gpu_interactions;
  std::vector<PhotonPairFinalStateRecord> final_state_records;
  std::vector<EmParticleState> secondaries;
  std::vector<EmInteractionRecord> continuations;
  std::vector<ProposalFallbackEvent> fallback_events;
};
```

`GpuEmStatistics` 增加：

```text
final_state_batches
gpu_final_states
physical_secondaries_generated
```

## 9. 代码位置

| 功能 | 文件 |
|---|---|
| 过程能力表 | `corsika/gpu/em/ProcessCapabilities.hpp` |
| 末态 API 和 draw 常量 | `corsika/gpu/em/CudaPhotonPairFinalState.hpp` |
| 分类、\(\rho\) 查询、Sauter 角和 CUB scan | `src/gpu/em/CudaPhotonPairFinalState.cu` |
| backend 验证入口和统计 | `src/gpu/em/CudaEmBackend.cu` |
| 记录与 fallback schema | `corsika/gpu/em/Types.hpp` |
| \(\rho\) 表生成与独立验证 | `applications/gpu_em_tablegen.cpp` |
| CPU/GPU 逐事件参考测试 | `tests/gpu/testGpuPhotonPairFinalState.cpp` |

开发验证接口为：

```cpp
EmFinalStateBatchResult
CudaEmBackend::generateFinalStatesForValidation(
    std::vector<EmInteractionRecord> const&,
    std::uint64_t first_secondary_history_id);
```

它不修改现有 toy resident queue。

## 10. 完整物理表

生成文件：

```text
/tmp/c8_gpu_em_full_strict_v3_phase10.c8emrt
```

配置：

```text
PROPOSAL 7.6.2
dry air
0.5 MeV cut
0.6 -- 1e12 MeV
1e-3 rate/loss tolerance
gamma, electron, positron
```

gamma 从 15 个物理 column 增加为：

```text
15 physical columns + 3 zero-rate rho columns = 18 columns
```

三个 \(\rho\) 表结果：

| target | energy rows | ragged quantile values | 最大自适应误差 |
|---|---:|---:|---:|
| N | 69 | 5,800 | \(7.496189\times10^{-4}\) |
| O | 65 | 5,472 | \(7.498600\times10^{-4}\) |
| Ar | 72 | 6,119 | \(7.496694\times10^{-4}\) |

全表结果：

```text
文件大小                         15,460,690 bytes
payload SHA-256                  b6b1448a00593212b2fa0e1f9af6261f
                                 22dfb0ecb60fc7bc49c557c5c1aa4e3b
完整文件 SHA-256                 2737f9f888d660b3c5f89e37b40e9b7b
                                 b088c6b09737590c1f8bcb9bf1c494e3
最大 rate 误差                   9.982391e-4
最大 inverse-CDF 误差            7.499968e-4
首次热 cache 生成                30.70 s
重复生成                         29.63 s
峰值 RSS                         约 100 MiB
```

重复生成文件通过 `cmp`，完整文件 SHA-256 完全相同。

## 11. 测试结果

### 11.1 合成 ragged 表

```text
4,091 interaction records
1,626 GPU photon pairs
2,464 explicit fallback
1 continuation
120,543 checks passed
744 device table bytes
```

### 11.2 完整 UHE 表

表上传与查询：

```text
2,720 device queries
13,609 checks passed
15,426,740 device bytes
```

相互作用选择：

```text
8,192 particles
8,116 selected/no-interaction records
76 table fallback
155,207 checks passed
```

photon-pair 末态：

```text
10,172 interaction records
3,148 GPU photon pairs
7,023 explicit fallback
1 continuation
251,380 checks passed
elapsed 0.49 s
peak host RSS 172,164 KiB
```

末态测试覆盖：

- 与独立 host 参考逐字段比较；
- 每个 photon pair 恰好产生 \(e^-\) 和 \(e^+\)；
- 逐对总能量闭合；
- 次级能量不低于 \(m_e\)；
- 次级方向归一；
- 位置、时间、介质和权重继承；
- generation、parent history 和 step ID；
- 相同输入重复执行确定；
- 完全反转输入顺序后每个 parent 的物理结果不变；
- success、fallback、continuation 数量闭合；
- history ID 精确上限和真正溢出；
- 非归一父方向拒绝；
- CPU-only 与 GPU-not-implemented 原因区分。

### 11.3 回归

| 测试 | 结果 |
|---|---|
| CUDA 相关 CTest | 9/9 通过 |
| 纯 CPU 相关 CTest | 3/3 通过 |
| `testGpuEmHost` | 163 checks 通过 |
| `testGpuEmRateTable` | 32 checks 通过 |
| `testGpuEmFlatRateTable` | 152 checks 通过 |
| CUDA fixture table query | 7,730 checks 通过 |
| `[ScalarCascadeStepper]` + `[HybridCascade]` | 15 assertions 通过 |

## 12. 构建与复现

生成正式表：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/\
gpu_em_tablegen /tmp/c8_gpu_em_full_strict_v3_phase10.c8emrt \
  --proposal-cache /tmp/c8_gpu_em_proposal_cache \
  --energy-min-MeV 0.6 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --initial-intervals 16 \
  --max-points 20000 \
  --loss-initial-energy-intervals 8 \
  --loss-initial-quantile-intervals 8 \
  --loss-max-energy-points 4096 \
  --loss-max-quantile-points 2048 \
  --loss-validation-samples 64
```

运行完整末态测试：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/gpu/\
testGpuPhotonPairFinalState \
  /tmp/c8_gpu_em_full_strict_v3_phase10.c8emrt
```

## 13. 当前限制

本阶段尚未完成：

1. LPM suppression 仍未在 GPU 上判断；
2. 已选相互作用 grammage 尚未与 tracking 边界竞争；
3. 粒子尚未移动到真实 interaction vertex；
4. validation bridge 每个 batch 仍会临时 `cudaMalloc/cudaFree`；
5. 末态还没有直接写入 resident 双缓冲 SoA；
6. CPU fallback 尚未批量调用 `ProposalFinalStateGenerator`；
7. bremsstrahlung、Compton、photoelectric 等 GPU 末态尚未实现；
8. `advanceWavefront()` 仍是 toy branching。

因此 0.49 s 是完整表读取、上传和多轮验证的测试时间，不是 production shower
性能结果。

## 14. 下一阶段

下一阶段建议先实现 LPM 和真实输运竞争，再把当前末态接入 resident queue：

1. 为 photon pair 加入与 CPU `PhotoPairLPM` 对照的设备参数和 acceptance；
2. LPM 抑制时保留父光子并重新抽取下一候选步，而不是错误地产生 pair；
3. 把 \(X_{\mathrm{int}}\) 与介质边界、连续步长和观测面竞争；
4. 将父粒子推进到真实 interaction vertex；
5. 把 classification、scan 和输出 buffer 变成 backend 常驻内存；
6. 将成功的 \(e^-/e^+\) 直接写入下一 wavefront SoA；
7. 批量消费 CPU fallback，并把返回的 EM 粒子重新入队。
