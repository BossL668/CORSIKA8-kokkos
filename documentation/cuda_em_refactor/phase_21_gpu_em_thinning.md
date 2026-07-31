# Phase 21：GPU EMThinning 与稳定可变 multiplicity

## 1. 为什么必须在生产接线前实现 thinning

UHE 电磁级联的粒子数会随 generation 快速增长。若 GPU resident cascade
只移植相互作用而不移植 CORSIKA 当前的 `EMThinning`，即使单个 kernel 很快，
显存和总计算量仍会很快失控；同时 CPU 与 GPU shower 的权重统计也不再代表
同一个算法。

本阶段把现有

```text
corsika/detail/modules/thinning/EMThinning.inl
```

的全部一变二分支移植到 host/device 公共原语：

```text
corsika/gpu/em/EmThinning.hpp
```

## 2. 精确保留的 CPU 语义

`applyEmThinning()` 接收父粒子总能量、父权重、两个子粒子总能量和两个显式
随机数。它保留以下判断次序：

1. 只在 thinning 已启用时工作；
2. 父权重达到 `maximum_weight` 时不 thinning；
3. 父总能量高于 threshold 时不 thinning；
4. 若两个 \(1/p_i\) 均不超过允许权重因子，使用 Hillas 二选一；
5. 否则两个孩子分别以限制后的概率独立接受；
6. `erase_zero_weight=false` 时保留权重为零的孩子，即现有 multithin
   语义。

Hillas 分支：

\[
p_i=\frac{E_i}{E_1+E_2},\qquad
w_i'=\frac{w_{\rm parent}}{p_i}.
\]

最大权重分支：

\[
f_i'=\min\left(\frac{1}{p_i},
\frac{w_{\max}}{w_{\rm parent}}\right),
\qquad P_i=\frac{1}{f_i'}.
\]

两个统计接受独立进行，因此一个顶点可以保留 0、1 或 2 个孩子。

## 3. 随机数协议

thinning 使用与末态抽样相同的 keyed Philox，但 draw ID 放在独立区间：

```text
0x100 -> first thinning decision
0x101 -> second statistical decision
```

key 中仍包含：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

因此打开 thinning 不会改变相互作用率、\(v\)、末态角度或 LPM 随机流；GPU
wavefront 排序变化也不会改变同一 history 的 thinning 结果。

## 4. 在 stable scan 之前决定 multiplicity

photon 和 charged final-state classification kernel 在 CUB exclusive scan
之前调用 thinning。其 `child_counts` 可为：

- photon pair、Compton、brems、annihilation、ionization：0、1 或 2；
- photoelectric：固定 1；
- Epair：固定 3，因为现有 CPU `EMThinning` 只处理一变二 EM 末态。

`record_flags` 与 `child_counts` 继续使用独立 scan。即使一个统计 thinning
顶点保留 0 个孩子，仍会生成完整 `FinalStateRecord`，但后续顶点的
`secondary_offset` 不产生空洞。

每个记录新增：

- thinning status；
- 原始 keep mask；
- 两个均匀随机数；
- 两个 draw ID。

保留的孩子携带修正后的权重。其 history ID 根据压缩后稳定 offset 分配，
在同一配置和输入下完全可重复。

## 5. Endpoint compaction 的变化

旧代码曾隐含假设：

- Compton 总有 `[photon, electron]` 两个孩子；
- charged 一变二过程总有两个孩子。

启用 thinning 后这些假设不成立。现在：

- Compton endpoint kernel 在 0–2 个压缩孩子中按 PID 搜索 photon；
- 只保留 recoil electron 时不创建 resident photon；
- charged endpoint kernel 接受两体过程的 0–2 个孩子，并逐 PID 分配到
  lepton/photon 队列；
- resident photon 的 layer-boundary 计数直接由 transport terminal
  accounting 得到，不再错误地假设每个 Compton 都继续一个 photon。

## 6. multithin

`erase_zero_weight=false` 会保留两个槽并把未接受孩子的权重设为零。后端粒子
验证从“严格正权重”改为“有限非负权重”，因此零权重粒子可以继续经过 GPU
selection、transport 和 final state，与现有 CPU multithin 行为一致。

## 7. 统计量

`GpuEmStatistics` 新增：

- `thinning_hillas_vertices`；
- `thinning_statistical_vertices`；
- `thinning_particles_discarded`。

multithin 中的零权重粒子仍被保留，所以不会计入
`thinning_particles_discarded`。

## 8. 验证

### 8.1 公共原语

`testGpuEmThinning`：

```text
8206 checks passed
8192 host/device exact comparisons
200000 deterministic statistical samples
```

检查了 threshold、最大权重早退、Hillas 两分支、独立统计接受、multithin
和非法输入。统计样本的平均权重与无 thinning 期望一致。

### 8.2 真实末态

`testGpuBremsFinalState` 和 `testGpuPhotonPairFinalState` 额外启用 thinning，
逐顶点重建 Philox key，并检查：

- keep mask；
- 0/1/2 multiplicity；
- child PID、权重、history ID 和连续 offset；
- Hillas 与 statistical 统计量；
- photoelectric 一变一不受影响；
- Epair 一变三不受影响；
- multithin 保留零权重孩子；
- 零权重父粒子可继续生成零权重末态。

### 8.3 完整 device pipeline

photon 与 lepton device-chain 测试验证了 thinning 后 endpoint queue 可由
host oracle 完整重建。resident photon cascade 在启用 Hillas thinning 后可
正常排空，不会因 Compton photon 被丢弃而触发错误。

最终重新链接并运行：

```text
23/23 CUDA tests passed
total test time = 6.37 s
```

## 9. 下一阶段

物理内核现在覆盖生产计划要求的现有 EM thinning。下一阶段是
`c8_air_shower` 接线：

1. 增加 CUDA CLI 和严格能力检查；
2. 每个 shower 将 `emthin`、`max-weight` 和 `multithin` 转换为
   `GpuEmConfig::thinning`；
3. 初始化版本化表和五层环境快照；
4. 把 GPU step/deposit/radio/observation 记录合并进现有 writer；
5. 只有输出适配完成后才让 CUDA 路径取代标量 `Cascade`。
