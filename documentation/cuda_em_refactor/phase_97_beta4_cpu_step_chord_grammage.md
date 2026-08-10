# Phase 97：beta4 连续输运与 CPU `Step` 弦轨迹对齐

## 1. 修复目标

beta4 从 beta2 完整复制而来，只修复一个已经通过代码审计和真实轨迹审计确认的语义差异：带电粒子在均匀磁场中完成 leapfrog 步进后，GPU 曾继续使用“步起点方向 × leapfrog 长度参数”计算实际连续能损和 Molière 多重散射所需的 grammage；CPU CORSIKA 8 则在确定步端点后，通过 `Step::getStraightTrack()` 使用两个实际端点之间的直线弦。两种路径在零磁场中相同，在存在磁偏转时一般不相同。

这个问题不是 PROPOSAL 表格的插值误差，也不是随机数差异。它位于 tracking 已经给出端点之后、连续过程读取该步轨迹的接口语义上。修复不改变 beta2 的物理表、过程选择、末态采样、Philox 键、CPU 强子模型、FLUKA 或射电算法。

## 2. CPU 原路径的准确语义

CPU 端分成两个阶段：

1. `ScalarCascadeStepper` 在原始 tracking trajectory 上比较 interaction、decay、几何边界和 continuous step limit，并调用 `track.setLength(min_distance)`；
2. `Step` 只保存这个 trajectory 的起点状态、端点位移、时间和方向变化。PROPOSAL 的 `ContinuousProcess::doContinuous()` 随后调用 `step.getStraightTrack()`，沿端点弦计算介质积分。

关键代码链为：

- `corsika/detail/framework/core/ScalarCascadeStepper.inl`：确定 `min_distance`，创建 `Step`，再调用 `sequence_.doContinuous(step, ...)`；
- `corsika/framework/core/Step.hpp`：`getStraightTrack()` 用 `getDisplacement() / getDiffT()` 构建直线轨迹；
- `corsika/detail/modules/proposal/ContinuousProcess.inl`：把 `step.getStraightTrack()` 交给介质的 `getIntegratedGrammage()`，再用 PROPOSAL range integral 求末态能量，并以同一 grammage 计算多重散射。

因此，CPU 同时存在两种有明确用途的长度/轨迹：

| 量 | 用途 |
|---|---|
| 原 leapfrog 轨迹的长度参数与起始切向 | 决定最近的 step limit，并把抽样 grammage 反演为 tracking 长度 |
| 实际保存的两个端点之间的弦 | `doContinuous()` 中的实际 grammage、连续能损和多重散射 |

把第二行误写成第一行就是 beta2 漏洞的来源。

## 3. beta4 修改

### 3.1 端点弦 grammage

修改文件为 `src/gpu/em/CudaLeptonTransport.cu`。

beta4 保留原有的切向 grammage，用于 interaction/continuous/geometry 竞争和 grammage-to-distance 反演；完成 `advanceUniformMagneticField()` 后，再执行以下操作：

1. 从最终保存的 `end.position_m - start.position_m` 重建弦位移；
2. 由该位移重新计算弦长并归一化弦方向，而不直接复用解析 leapfrog 弦长，避免地心坐标加减造成的末位差异；
3. 用弦方向和弦长重新调用 `atmosphereGrammage()`；
4. 用这个 grammage 查询 `queryEnergyAfterContinuousLoss()`；
5. 把同一个值写入 `LeptonTransportRecord::traversed_grammage_g_per_cm2`，供后续 Molière kernel、纵向 profile、能量沉积和轨迹输出共同使用；
6. 即使 continuous limit 获胜，也不再直接把末态能量固定为预选的 10% target，而是像 CPU 一样按实际弦 grammage 重新求末态能量；
7. 若弦 grammage 已把粒子推进至 transport cut，则按 CPU 的 `continuous -> ParticleCut` 顺序终止该粒子，并把剩余动能计入 cut deposit，禁止把该状态误报为表格 fallback。

修复没有把弦 grammage 用于最初的 step-limit 竞争，因为那样会改变 CPU tracking 本身的定义，并形成需要迭代求解的另一种算法。

### 3.2 每个粒子自己的 transport cut

500 例长跑还暴露了一个独立的 beta2 边界漏洞。生产表同时包含电子和 μ 子连续输运；电子默认 kinetic cut 是 0.5 MeV，μ 子默认 cut 是 300 MeV。表的最低总能量按 CPU `getMaxStepLength()` 的定义保存为

\[
E_{\min}=m+0.9999\,E_{\mathrm{cut}}.
\]

beta2 的 interaction selector 和步后终止判断却复用了 `table.em_transport_cut_MeV`，这个字段编码的是电子 cut，不能用于 μ 子。旧实现又在 continuous step 获胜时直接把末态能量固定到表的最低锚点，因此多数时候会偶然越过 μ 子 cut，没有暴露循环。改成真实弦 grammage 后，一个总能量为 405.6283745 MeV 的 \(\mu^-\)（动能已经略低于 300 MeV）在表下边界产生了约 \(10^{-13}\,\mathrm{g\,cm^{-2}}\) 的步；range 插值因浮点舍入返回与初态完全相同的能量，resident scheduler 因而不断重复同一状态。

beta4 在 `CudaInteractionSelector.cu` 和 `CudaLeptonTransport.cu` 中都由对应 PID 的连续表反推出实际 cut：

```cpp
auto const transport_cut_MeV =
    (minimum_energy.value - mass.value) /
    tables::ContinuousCutSafetyFactor;
```

步首和步后都使用这个 per-PID 值；连续步获胜却没有产生严格正能损时则显式返回 `InvalidTableQuery`，禁止 resident wavefront 静默循环。用原失败 seed 重跑时，第 24 个事例从超过 9 分钟无进展恢复为正常完成。

## 4. 新增回归测试

`tests/gpu/testGpuLeptonTransport.cpp` 的均匀磁场 pipeline 测试现在对每一条带电轨迹独立重建端点弦，并要求：

- GPU 记录的 `traversed_grammage` 与端点弦 oracle 一致；
- GPU 末态能量与用弦 grammage 查询 continuous-range 表的结果一致；
- 测试样本中至少存在一条切向 grammage 与弦 grammage 可分辨的轨迹，避免零磁场或退化输入让测试虚假通过；
- 相同输入重复运行仍保持完全确定性。
- production μ 子表测试显式构造位于 `0.9999*cut` 锚点与真实 300 MeV cut 之间的 \(\mu^\pm\)，要求 selector 直接标记 `ParticleCut`，且 GPU 生成零长度 cut record；
- 连续步若在表查询后没有严格降低能量，必须显式失败而不能重新排队。

该测试在 RTX 4060 Laptop GPU 上完成 71,756 项检查并通过。

## 5. 真实轨迹审计

修复前，使用 beta2 scalar backend 记录了一个 10 GeV、electron primary、zenith 47 度、azimuth 180 度、seed 470180、默认最大偏转角 0.2 rad 的完整 replay tape。样本包含 13,787 条带电粒子轨迹。对每一条轨迹分别计算旧 GPU 切向 grammage 和 CPU `Step` 弦 grammage，得到：

| 指标 | \(|X_\mathrm{tangent}-X_\mathrm{chord}|/X_\mathrm{chord}\) |
|---|---:|
| 中位数 | 0.0272% |
| 90% 分位 | 0.137% |
| 99% 分位 | 0.527% |
| 99.9% 分位 | 1.487% |
| 最大值 | 1.855% |
| 有符号均值 | -0.0604% |

使用同一张 5e-4 production table 查询末态能量时，弦方法相对 scalar CPU 实际末态能量的最大偏差为 0.0209%，处在表格精度范围内；旧切向方法的最大偏差为 0.1811%。这说明修补把剩余差异恢复为预期的物理表插值误差，而不是 tracking 轨迹定义错误。

## 6. 构建与功能验收

beta4 使用独立的 source/build/install 目录，从空 build tree 进行 Release 构建，配置包括 CUDA、compute capability 8.9 和 FLUKA 2025。验收结果如下：

| 验收项 | 结果 |
|---|---|
| 27 个 `testGpu*` 测试 | 全部通过，24.16 s |
| 完整 `ctest` | 34/34 通过，220.33 s；包含 framework/media/stack/modules/output、PROPOSAL、强子模型和 FLUKA |
| validation Python tests | 207/207 通过，1.734 s |
| production μ 子表专项测试 | 102,121 项检查通过；8,192 个 μ 子完成输运 |
| 版权检查 | 通过；同时补齐复制自 beta2 的诊断工具头部 |
| beta4 scalar 与 beta2 scalar 固定 seed tape | 逐字节相同 |
| scalar tape SHA-256 | `6565df2d481bb433f3fa3658098169038238931147e78c9f93d30994172db7cf` |
| beta4 CUDA decision replay | 14,849 条记录全部接受，0 byte-hash mismatch |
| replay host/device ordered hash | 均为 `e7029f15e3bd64b6` |
| 相同 seed 的 beta4 production CUDA 重复运行 | profile、dEdX、particles、production profile、CoREAS、ZHS 六类输出哈希全部相同 |
| 能量账本（10 GeV 单例） | relative closure error `1.63e-16`，通过 `1e-4` 门限 |
| 未预期 CPU fallback、queue overflow、fixed-point overflow | 均为 0 |

## 7. 统计与性能验收

最终回归样本使用 10 GeV electron primary、zenith 47 度、azimuth 180 度、IGRF14/2027、`emthin=1e-4` 和相同 81 天线输入。CPU 为 4 个独立的 125-event scalar PROPOSAL 分片；CUDA 为一个 500-event GPU EM + GPU CoREAS/ZHS 批次。两侧物理配置经比较脚本规范化后完全相同。

未薄化的 10 GeV 诊断样本曾出现一个动能约 0.597 MeV 的近水平电子，在高空磁场中合法累计超过 4,096 个 0.2 rad 磁偏转小步。CPU 也必须演化这类长寿命低能尾部，因此 beta4 没有加入任意 step-count cut。正式高统计回归采用既有 EM thinning，让低于约 1 MeV 的尾部走原有无偏薄化逻辑。

### 7.1 shower observables

500 对 500 的纵向曲线统计门全部通过：electron、positron、总带电、photon、总 EM 和能量沉积活跃 bin 的通过率为 98.8%--100%，RMS active-bin z-score 为 1.02--1.50。关键积分量为：

| observable | CUDA 相对 CPU 均值差 | 结果 |
|---|---:|---|
| charged profile integral | -0.254% | 1% 与统计门均通过 |
| photon profile integral | +0.569% | 1% 与统计门均通过 |
| total deposited energy | -0.0480% | 1% 与统计门均通过 |
| energy-closure fraction | -0.0499% | 1% 与统计门均通过 |
| charged \(X_{\max}\) | +2.334% | 未过 1% 门；z=1.04、KS=0.050<0.086，统计上不显著 |
| charged peak | -1.302% | 未过 1% 门；z=1.15、KS=0.066<0.086，统计上不显著 |
| deposited-energy \(X_{\max}\) | +1.592% | 未过 1% 门；z=0.657，统计上不显著 |

因此 `curve_pass=true`；严格 scalar 总门因为三个宽分布均值没有收敛到 1% 而为 false，但比较器把它们分类为 `relative_threshold_failed_but_statistically_inconclusive`，没有任何 key scalar 被判为统计不一致。10 GeV 下地面粒子几乎全为零，相对百分比地面门不具备解释力。

### 7.2 射电脉冲

使用 `pulse_analysis_modular` 的相同定义，在 \(r_\perp=100\) m 对每个 shower 的八个方位天线聚合：

| radio observable | CUDA/CPU | bootstrap 95% interval | shape result |
|---|---:|---:|---|
| CoREAS amplitude | 1.00087 | [0.99935, 1.00331] | 严格 KS 失败 |
| CoREAS width | 0.99072 | [0.97543, 1.00801] | 通过 |
| ZHS amplitude | 1.00019 | [0.99980, 1.00059] | 通过 |
| ZHS width | 1.00000 | [1.00000, 1.00000] | 通过 |

CoREAS amplitude 的 KS 失败不能隐藏：该低能样本中几乎全部振幅落在同一个数值底噪/离散峰值上，CPU 与 CUDA 中位数只有约 \(2\times10^{-7}\) 的相对位移，却令经验 KS 达到 0.996。均值和区间高度一致，但这个退化样本不能作为可观测射电脉冲形状等价的充分证据。相同轨迹的 CUDA radio replay 仍保持 host/device hash 相同和 0 fixed-point overflow；独立 shower 的严格 CoREAS 形状验收应使用已有的更高能、可观测信号样本。

### 7.3 计时

四个 CPU 分片的程序内运行时间总和为 745.81 CPU-core-s，即 1.492 s/event；CUDA 程序内时间为 209.77 s，即 0.420 s/event，对应 3.56 倍。按外部端到端时间求和，CPU 为 752.41 CPU-core-s，CUDA 为 213.34 s，对应 3.53 倍。这个低能、小 wavefront 回归点不用于替代 100 TeV 和 \(10^{17}\) eV 的生产性能结论。

正式原始数据、shower 比较 CSV/JSON、射电脉冲图和 replay 审计文件统一归档在：

`/mnt/d/CorsikaData/corsika_validation_results/beta4_chord_grammage_electron_10GeV_theta47_phi180_emthin1e-4_cpu500_cuda500_v1`

## 8. 结论和适用边界

beta4 已在代码语义、单元测试、真实 scalar 轨迹、固定 seed CPU 回归、GPU replay、生产 CUDA 确定性和能量闭合几个互相独立的层面覆盖本次漏洞。这里所说的“同一轨迹一致”是指：给定同一组 tracking 端点，CPU 与 beta4 使用同一条端点弦计算 grammage，最终差异只剩受声明精度约束的表格插值。它不表示 scalar PROPOSAL 与 production CUDA 使用相同随机流或会生成逐事件相同的完整 shower；完整 shower 仍应通过独立样本的统计分布进行验收。
