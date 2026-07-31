# Phase 77：近阈光致强子闭合与最终热/冷性能验收

## 1. 冷启动验收暴露的物理空隙

Phase 76 后首次执行严格 `cold-each` 性能验收时，固定 seed `176101` 的
1 PeV electron 事件中止。真正原因不是 Molière 缓存，而是：

```text
selected process       PROPOSAL Photoproduction
selected target        atmospheric oxygen component
sampled nucleon        proton
sqrt(s)                1.0784298032941142 GeV
SOPHIA lower limit     1.079166345 GeV
```

PROPOSAL 7.6.2 的 Rhode 实光子截面在 pion-production cutoff 后已经非零，
但 SOPHIA 的硬下限略高，因此存在一个窄的近阈区间。旧上游路径会警告后返回
成功但不产生次级；Phase 75 将其改为 hard failure，避免静默破坏能量账本，
但该区域仍没有可用末态。

## 2. 显式两体阈值末态

新增：

```text
corsika/modules/proposal/ThresholdPhotoproductionModel.hpp
```

它只在以下能力区间有效：

```text
projectile = photon
target     = proton or neutron
m_N + m_pi0 <= sqrt(s) < 1.079166345 GeV
```

末态固定为物理开放的最低两体通道：

\[
\gamma + N \rightarrow \pi^0 + N.
\]

实现使用 Källén 两体相空间：

\[
p^* =
\frac{\sqrt{[s-(m_N+m_\pi)^2][s-(m_N-m_\pi)^2]}}
     {2\sqrt{s}},
\]

在质心系各向同性抽取方向，再通过 `COMBoost` 精确变回 lab。若低于
\(m_N+m_{\pi^0}\) 或高于 SOPHIA 下限，模型明确拒绝，不能把这个窄区近似
外推到其他能区。

生产低能模型链现在是：

```text
preferred: SOPHIA
fallback:  ThresholdPhotoproductionModel
```

summary 新增：

```yaml
photo_hadronic_generator:
  sophia_interactions: ...
  threshold_fallback_interactions: ...
```

原有的 SIBYLL → lazy QGSJet-II 高能回退计数继续保留。

单元测试直接验证：

- 能力边界上下端；
- 只接受 photon + proton/neutron；
- 正好产生 \(\pi^0+N\)；
- lab 总能量与三动量闭合到 double 舍入精度；
- 不支持的能区继续 hard failure。

## 3. 原失败 seed 的确定性复现

修复后的同一事件：

```text
/tmp/c8_phase77_threshold_repro_cuda_v1
```

结果：

```text
complete                         true
SOPHIA interactions             411
threshold two-body fallback       1
SIBYLL preferred                 15
QGSJet-II fallback                1
discarded final states            0
```

因此原来会中止或被旧代码丢弃的两个不同光致强子边界，都在同一个真实 shower
中生成了明确末态。

## 4. 当前生产构建来源

本阶段最终证据使用：

```text
c8_air_shower SHA-256
1d1a8dd59d4ae182d2d39496b0bc92e9eb4d9d92046c3e6302e75e8c5be5c7f5

production table SHA-256
140347373b9b5014c8b5bbd6cbcffca5f928eb5190e58c54c725d5843c2b295e

performance runner SHA-256
e610b0dc654586633171a438e124e9cf8bb8341f412d005d374848d7ddb3eac7
```

runner 在计时前后重新哈希 executable 和 table。

## 5. 五次热缓存性能

输出：

```text
/tmp/c8_phase77_current_threshold_hot_performance_1PeV_5rep_v1
```

1 PeV、0.5 MeV cut、EM thinning \(10^{-4}\)、maximum weight 100、
Release sm_89、单 CPU 核：

```text
status                                  passed
external-wall median ratio              10.04695
summed-shower-timing median ratio       15.73474
paired external median                   9.61225
paired external minimum                  9.39020
paired shower-timing median             15.16763
paired shower-timing minimum            14.91918
```

## 6. 五次最坏路径 cold-each 性能

输出：

```text
/tmp/c8_phase77_current_threshold_cold_each_performance_1PeV_5rep_v2
```

配置固定使用原失败 seed `176101`。每次 CUDA repetition 都：

1. 在运行前确认 Molière 派生缓存不存在；
2. 在计时内重新生成 539032-byte 缓存；
3. 初始化并使用 QGSJet-II；
4. 执行一次近阈 \(\gamma N\to\pi^0N\) 回退；
5. 完整结束，`discarded_final_states=0`。

结果：

```text
status                                  passed
external-wall median ratio               5.34560
summed-shower-timing median ratio        6.61987
paired external median                   5.33481
paired external minimum                  5.00552
paired shower-timing median              6.61216
paired shower-timing minimum             6.20029
```

五次缓存 provenance 都显示 repetition 1–4 删除前一次生成物，运行前不存在，
运行后重新生成。生产 rate table 本身从未删除或修改。

## 7. 当前源码回归

两体回退集成后重新执行：

```text
CUDA all target        PASS
CUDA CTest             32 / 32 PASS
CPU-only all target    PASS
CPU-only CTest         10 / 10 PASS
Python validation      53 / 53 PASS
```

## 8. 结论

当前二进制同时关闭了：

- SIBYLL 不支持氩靶时的高能光核末态丢弃；
- PROPOSAL 与 SOPHIA 阈值不一致时的低能末态空隙；
- QGSJet-II 无条件初始化的固定开销；
- 当前来源哈希的热缓存和最坏 cold-each 5× 性能门禁。

下一项生产门禁是用同一 executable/table SHA-256 建立代表性 shower 统计矩阵。
