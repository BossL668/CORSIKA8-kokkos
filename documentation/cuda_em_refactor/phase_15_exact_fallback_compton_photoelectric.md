# Phase 15：精确 CPU 末态回退、Compton 与光电效应

## 1. 本阶段结论

本阶段解决了三类会直接影响物理正确性的问题：

1. GPU 已经选定 `process/component/v` 后，稀有过程能够在 CPU 端只生成
   该指定 PROPOSAL 末态，不再重新抽取反应；
2. 光子 Compton 散射的末态已在 GPU 上生成，散射光子会进入下一
   wavefront，而不是提前终止 resident photon cascade；
3. 光电效应的单电子末态已在 GPU 上生成，并记录 K 壳层束缚能沉积。

当前光子设备链为：

```text
γ
  -> rate / process / component / v selection
  -> spherical-atmosphere transport
  -> pair production --------> e- + e+
  -> Compton -----------------> γ(next wavefront) + e-
  -> photoelectric -----------> e- + binding-energy deposit
  -> unsupported rare process -> exact specified CPU PROPOSAL final state
```

完整 GPU 测试矩阵为 `18/18 passed`。

## 2. v8 物理表身份校验

物理表格式升级为：

```text
format version = 8
magic          = C8EMRT08
```

新增两个不能缺省的身份字段：

- `RateTableMetadata::proposal_medium_hash`
- `ParticleRateTable::interaction_hash`

前者对应 `PROPOSAL::Medium::GetHash()`，后者对应该粒子
`PROPOSAL::Interaction::GetHash()`。CUDA 后端初始化时同时校验：

- PROPOSAL 版本；
- 介质名称、组成和 medium hash；
- 粒子 interaction hash；
- cut、能区和误差界；
- 表格内容 SHA-256。

因此，一个来自相同名称但不同参数化、cut 或内部 calculator 集合的缓存不能
被误用。

## 3. 指定末态 CPU fallback

核心文件：

- `corsika/gpu/em/ProposalFallbackAdapter.hpp`
- `corsika/gpu/em/ProposalCpuFallbackHandler.hpp`
- `corsika/modules/proposal/ProposalInteractionRecord.hpp`
- `corsika/modules/proposal/InteractionModel.hpp`

`ProposalFallbackEvent` 现在携带：

```text
medium_hash
interaction_hash
process_id
component_hash
energy_fraction (v)
selection_uniform
particle state
Philox identity
```

`ProposalFallbackAdapter` 将设备单位转换回 PROPOSAL：

```text
GeV -> MeV
m   -> cm
```

并构造完整 `proposal::ProposalInteractionRecord`。
`InteractionModel::doSpecifiedInteraction()` 会验证 medium 和 interaction
hash，然后直接调用：

```cpp
SecondariesCalculator::CalculateSecondaries(...)
```

它不会再次调用 rate sampling 或 process selection。

CPU 末态随机数仍使用按 history 寻址的 Philox：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

CPU 专用 draw 从 \(2^{32}\) 开始，避免与 GPU 内核使用的小 draw ID
重叠。测试还会故意破坏 `interaction_hash`，确认处理器立即拒绝该事件。

## 4. 通用光子末态 scan

最初的 pair-production kernel 假设每个成功反应固定产生两个次级，并用：

```cpp
record_offset = child_offset / 2;
```

这对单电子的光电效应不成立。本阶段将两个概念拆开：

```text
child_counts  -> exclusive scan -> secondary_offset
record_flags  -> exclusive scan -> record_offset
```

过程计数也独立 scan：

```text
pair_flags
compton_flags
photoelectric_flags
```

批次闭合条件为：

\[
N_\text{record}
  =N_\text{pair}+N_\text{Compton}+N_\text{photoelectric},
\]

\[
N_\text{secondary}
  =2(N_\text{pair}+N_\text{Compton})
   +N_\text{photoelectric},
\]

\[
N_\text{input}
  =N_\text{record}+N_\text{fallback}
   +N_\text{continuation}+N_\text{LPM-suppressed}.
\]

任何一项不闭合都会终止当前 shower。

## 5. GPU Compton 末态

实现逐式翻译自 PROPOSAL 7.6.2
`secondaries::NaivCompton`。

设入射光子能量为 \(E\)，转移比例为 \(v\)，则：

\[
E_{\gamma'}=E(1-v),
\qquad
E_{e^-}=m_e+Ev.
\]

方向由能量动量守恒决定：

\[
\cos\theta_{\gamma'}
 =1-\frac{v m_e}{E(1-v)},
\]

\[
\cos\theta_{e^-}
 =\frac{v(E+m_e)}
 {\sqrt{2vEm_e+v^2E^2}}.
\]

方位角使用：

```text
process_id = Compton
draw_id    = 1
```

电子方位角比散射光子相差 \(\pi\)。无 `fast-math`，所有状态字段使用
double。

### 5.1 resident cascade 的续传修复

pair production 不产生下一代光子，而 Compton 会产生一个散射光子。
旧 endpoint compactor 只合并：

- layer-boundary photon；
- LPM-suppressed photon。

若仅把 Compton 末态加入原 kernel，普通 Hybrid 路由虽然能在主机侧重新入队，
但 `runResidentPhotonCascadeForValidation()` 会把散射光子误当作终端次级。

当前实现新增设备端 `scatterGeneratedPhotonsKernel`：

1. 检查记录过程必须为 Compton；
2. 检查第一个 child 必须为 photon；
3. 按原 `input_index` 写入 `raw_next[source]`；
4. 与 boundary/LPM continuation 共用 exclusive scan；
5. `atomicCAS` 只检测互斥路径重复，不分配输出顺序。

因此 Compton 光子保持 device-resident。`PhysicalCudaEmRouter` 只从
`final_states.secondaries` 追加非光子 child，避免散射光子重复入队。

## 6. GPU 光电效应

实现逐式翻译自 PROPOSAL 7.6.2
`secondaries::PhotoeffectNoDeflection`。

光电效应的 stochastic loss 必须满足：

\[
v=1.
\]

由目标组分的核电荷 \(Z\) 计算 K 壳层束缚能：

\[
I=\frac{(Z\alpha)^2m_e}{2}.
\]

输出电子总能量为：

\[
E_{e^-}=m_e+E_\gamma-I.
\]

方向继承入射光子方向，这与 PROPOSAL 和标准 EGS4 的该近似一致。GPU 从
版本化 LPM component snapshot 读取同一 `component_hash -> Z` 映射；出现
下列情况时显式回退：

- component hash 不存在；
- \(Z\)、\(\alpha\)、能量非有限；
- \(E_\gamma<I\)；
- 表中抽到的 \(v\) 不等于 1；
- generation 溢出。

`PhotonFinalStateRecord::energy_split_fraction` 对光电效应保存：

\[
\frac{E_\gamma-I}{E_\gamma}.
\]

因此 router 可以重建 \(I\)，写入 `EmStepRecord::deposited_energy_GeV`，
并按粒子权重累加总沉积能量。

## 7. 与真实 PROPOSAL 的验证

`testGpuPhotonPairFinalState` 不只使用手写 host oracle，还链接真实
`PROPOSAL::PROPOSAL`：

- Compton 能量与方向逐事件对照
  `PROPOSAL::secondaries::NaivCompton`；
- 光电效应的 multiplicity、电子能量和方向逐事件对照
  `PROPOSAL::secondaries::PhotoeffectNoDeflection`；
- synthetic nitrogen/oxygen/argon 的 target \(Z\) 分别验证；
- mixed pair/Compton/photoelectric 批次验证可变 multiplicity scan；
- 反转输入队列后，每个 history 的物理随机结果不变；
- 次级 history ID 上界和溢出仍被检查。

`testGpuPhotonWavefront` 进一步验证：

- 五层球形大气中的 process selection 与输运；
- Compton photon 按 source index 稳定进入下一 wavefront；
- resident cascade 与 host-loop oracle 完全一致；
- photoelectric 是终端 photon 路径；
- 所有批次计数和次级 multiplicity 闭合。

`testGpuHybridRoute` 使用只开启光电截面的高率合成表，确认：

- 光电反应一定在观测面前发生；
- 电子回到 EM 路由；
- K 壳层束缚能在 step 和带权总沉积中均为正值。

## 8. 当前仍未完成

下一阶段优先级为：

1. GPU positron annihilation；
2. GPU electron/positron pair production；
3. 离散 ionization 末态；
4. 把 photon/lepton 双队列完全留在同一 device scheduler，消除当前
   `PhysicalCudaEmRouter` 每个物理 wavefront 的 host round-trip；
5. 正式接入 `c8_air_shower` CLI、writer、profile、ground output 和
   CoREAS；
6. 接入生产 EM thinning；
7. 完整 shower 的 CPU/GPU 物理分布和性能验收。

本阶段通过测试不代表已经达到最终的 \(5\times\) 端到端加速验收。
