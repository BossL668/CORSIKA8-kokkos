# Phase 20：GPU 电子/正电子直接电子对产生

## 1. 本阶段目标

本阶段将高能电子或正电子的直接电子对产生

\[
e^\pm + Z \rightarrow e^\pm + e^- + e^+ + Z
\]

从显式 CPU fallback 移入 resident CUDA charged-lepton cascade。实现必须同时
满足四个约束：

1. 相互作用率和能量损失比例 \(v\) 仍来自版本化 PROPOSAL 表；
2. 条件末态分布与 CORSIKA 当前使用的 PROPOSAL 7.6.2 默认末态生成器一致；
3. Epair LPM 接受/拒绝逻辑与
   `corsika/detail/modules/proposal/InteractionModel.inl` 一致；
4. 一个接受的顶点产生三个而不是两个粒子，因此所有稳定压缩、history-ID
   预留和 resident checkpoint 都必须扩大到最大 multiplicity 3。

## 2. 需要区分的两个 PROPOSAL 模型

PROPOSAL 在这条路径中使用了两个不同的参数化：

- `EpairForElectronPositron` 提供 CORSIKA 相互作用率和 \(v\) 分布；
- 默认 `SecondariesCalculator` 使用
  `KelnerKokoulinPetrukhinEpairProduction` 生成给定 \(v\) 的三体末态。

因此不能简单地把率表的二维逆 CDF 当作最终 \(e^-e^+\) 能量分配。GPU 首先
从率表得到 \(v\)，随后按 KKP 条件分布采样非对称变量 \(\rho\)：

\[
\begin{aligned}
E_{\rm survive} &= E(1-v),\\
E_{e^-} &= \frac{Ev}{2}(1+\rho),\\
E_{e^+} &= \frac{Ev}{2}(1-\rho).
\end{aligned}
\]

这三个能量之和逐事件等于父粒子总能量。

## 3. KKP 条件末态的设备实现

实现位于：

```text
corsika/gpu/em/EpairFinalState.hpp
```

`kkpRhoWeight()` 是 PROPOSAL
`EpairKelnerKokoulinPetrukhin::FunctionToIntegral()` 的 host/device
翻译。只删除了与 \(\rho\) 无关的整体因子；这些因子会在归一化条件 CDF 中
严格抵消。核电荷、辐射对数常数和粒子质量均来自版本化
`BremsLpmSnapshot`，不存在设备端隐含介质常数。

PROPOSAL 的 seeded 结果不仅由物理公式决定，也由
`PROPOSAL::Integral` 的数值次序决定。为避免同一随机数产生约
\(10^{-3}\) 量级的分位点差异，本阶段移植了它的数值流程：

1. 开放式三分点梯形积分；
2. 五点 Romberg 外推；
3. 在积分过程中构造初始分位点；
4. Newton 与二分混合的上限修正；
5. 修正阶段使用二点 Romberg 外推；
6. 12 次积分细化、20 次根修正和 \(10^{-6}\) 精度参数。

GPU 数值积分不能收敛时返回 `IntegrationFailed`，调用方生成带原因的
`ProposalFallbackEvent`，不会夹紧 \(\rho\) 或静默改变末态。

PROPOSAL 为 Epair 消耗三个末态随机数。本实现固定 draw ID：

```text
1 -> |rho|
2 -> rho sign
3 -> direction
4 -> Epair LPM accept/reject
```

当前 PROPOSAL 7.6.2 的方向函数忽略第三个数并让两个新粒子与父粒子共线，
但 GPU 仍生成并记录该 draw，以维持随机数协议和未来升级的审计能力。

## 4. Epair LPM

实现位于：

```text
corsika/gpu/em/EpairLpm.hpp
```

该函数逐项复现 PROPOSAL `EpairLPM::suppression_factor()`。特征
\(E_{\rm LPM}\) 可由现有 brems LPM 介质快照中的组分数据重新构造，因此本
阶段不需要修改 v8 表格文件格式。

PROPOSAL 的闭式表达式在生存概率非常接近 1 时存在浮点相消，有时返回略大于
1 的数。GPU 将大于 `1 - 1e-6` 的结果规范为 1。由于接受随机数位于开区间
\((0,1)\)，这一分支与 PROPOSAL 的接受结果等价，同时避免把舍入误差误判为
非法概率。

`gpu_em_tablegen` 现在也会以一组能量、\(v\)、\(\rho^2\) 和密度直接对照
PROPOSAL Epair LPM，并将最大相对误差打印到生成日志。生成器版本更新为
`c8-gpu-em-tablegen-0.9`。

## 5. 三次级稳定压缩

实现的主要改动位于：

```text
src/gpu/em/CudaBremsFinalState.cu
src/gpu/em/CudaLeptonSelectionTransport.cu
corsika/gpu/em/PhysicalCudaEmRouter.hpp
src/gpu/em/CudaEmBackend.cu
```

原 charged final-state kernel 的最大 multiplicity 是 2，且曾隐含使用
`child_offset / 2` 作为记录索引。现在拆分为两套独立 scan：

- `child_counts`/`child_offsets` 分配 0、2 或 3 个次级槽；
- `gpu_flags`/`record_offsets` 为每个接受的顶点分配一个稳定记录槽。

因此 Epair 的三个连续 child slot 不会破坏 annihilation、ionization 和
brems 的记录顺序。任何批次都检查：

\[
N_{\rm child} =
2(N_{\rm brems}+N_{\rm ann}+N_{\rm ion})
+3N_{\rm epair}.
\]

设备 workspace、host 重建测试和边界检查的最坏槽数均改为
`3 * input_count`。

## 6. History-ID 安全

一个接受的 Epair 顶点依次为：

1. surviving \(e^\pm\)；
2. 新 \(e^-\)；
3. 新 \(e^+\)

分配稳定、连续且互不重复的 history ID。resident cascade 在启动下一
wavefront 前检查剩余区间至少能容纳 `3 * current_count`。若不能容纳，整个
当前队列作为无损 checkpoint 返回 host，由 router 预留新的不相交区间后继续。

这保证新增三体过程不会让 GPU 越过 CPU stack 的共享 history allocator。

## 7. 统计与诊断

新增统计量包括：

- `electron_pair_final_states`；
- `electron_pair_lpm_trials`；
- `electron_pair_lpm_suppressions`；
- batch 级 `electron_pair_interactions`；
- 独立 brems/Epair LPM trial 和 suppression 计数。

每个 `BremsFinalStateRecord` 还保存第三个末态随机数及其 draw ID，确保
CPU/GPU 结果可以逐顶点审计。

## 8. 验证结果

### 8.1 Epair LPM oracle

`testGpuEpairLpm` 对 750 组能量、\(v\)、\(\rho^2\) 和密度，在 host 与
RTX 4060 CUDA kernel 中对照真实 PROPOSAL：

```text
2252 checks passed
maximum relative error = 5.72321e-11
```

### 8.2 三体末态 oracle

`testGpuEpairFinalState` 使用 1024 个合法 PROPOSAL 相互作用，覆盖
20 MeV 到 \(10^{14}\) MeV，并检查：

- capability 分类；
- Philox draw 和重复运行确定性；
- PROPOSAL `CalculateRho()`；
- PROPOSAL `CalculateSecondaries()`；
- 三个 child 的 PID、能量、generation、parent/history ID；
- 逐事件能量闭合；
- LPM 接受和拒绝；
- 非法 PID、损失和 component 的显式 fallback。

结果：

```text
24255 checks passed
accepted vertices = 1009
LPM-suppressed vertices = 15
maximum normalized rho error = 5.26914e-4
```

### 8.3 全套 CUDA 回归

所有 CUDA 目标重新链接后运行：

```text
22/22 tests passed
total test time = 5.75 s
```

回归范围包括率表、设备上传、interaction selection、全部已实现末态、
球形大气、均匀磁场、Molière、resident photon/lepton cascade、
HybridCascade router 和指定 CPU fallback。

## 9. 当前边界

本阶段完成的是可被 `PhysicalCudaEmRouter` 使用的物理内核和 resident
cascade 闭环。正式 `c8_air_shower` 仍需完成输出适配：把 GPU
`EmStepRecord`、`RadioTrackRecord` 和 `ObservationRecord` 合并进现有
energy-loss、longitudinal、ground-particle 和 CPU CoREAS/ZHS writer。
在该适配完成前，不应把测试接口误称为完整生产后端。
