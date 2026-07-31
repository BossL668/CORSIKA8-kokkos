# Phase 48：1 PeV、200+200 事件正式物理验收结果

## 1. 配置与完整性

正式运行：

```text
/tmp/c8_phase42_physics_acceptance_1PeV_200
```

配置为 1 PeV electron、天顶角 0°、0.5 MeV cut、\(10^{-4}\) EM thinning、
最大权重 100。PROPOSAL 和 CUDA 使用独立 seed；CPU 分成 8 个各 25 事件的
单线程 shard，CUDA 为 200 事件。两后端同时启动只用于物理统计，wall time
不用于性能结论。

实际外部时间：

```text
PROPOSAL shard group: 4725.19 s
CUDA:                  1539.00 s
```

CUDA 200 个 shower 全部通过 output/integrity 检查：完整状态、queue overflow、
host spill、profile fixed-point overflow 和非法记录均符合验收要求。

## 2. 结果

9 个预先选择的 key scalar 中，8 个通过“相对均值差不超过 1% 且不超过
3 combined standard errors”的双重门禁：

| Observable | 相对均值差 | \(|z|\) | 结果 |
|---|---:|---:|---|
| charged \(X_{\max}\) | 0.6101% | 0.815 | PASS |
| charged peak | 0.4201% | 0.544 | PASS |
| charged integral | 0.1644% | 0.273 | PASS |
| photon integral | 0.2300% | 0.375 | PASS |
| deposited energy | 0.0759% | 0.454 | PASS |
| deposit \(X_{\max}\) | 0.8346% | 0.340 | PASS |
| energy closure diagnostic | 0.0281% | 0.472 | PASS |
| ground EM weighted count | 0.0353% | 0.008 | PASS |
| ground EM kinetic energy | 1.9916% | 0.293 | **FAIL** |

所有 longitudinal 和 ground histogram curve gate 通过。每条 active-bin
统计门禁的通过率均为 100%；charged/profile 曲线相对 L1 约为 0.66%，ground
energy histogram 相对 L1 为 0.256%。

因此整个正式 gate 的结果必须记录为：

```text
scalar_pass = false
curve_pass  = true
passed      = false
```

不能把 8/9 key scalar 通过改写成总体通过。

## 3. 唯一失败项的解释

ground EM kinetic energy 是强长尾 observable：

```text
PROPOSAL mean ± SE: 22807.16 ± 956.39 GeV
CUDA mean ± SE:     23261.38 ± 1220.02 GeV
combined SE:        1550.20 GeV
mean difference:     454.22 GeV
```

点估计相差 1.99%，超过 1% 门槛；但只相当于 0.293 combined standard
errors，没有统计显著性。combined SE 相对 CPU mean 为 6.80%，3σ 精度宽度为
20.39%，远宽于要判定的 1% 效应。

长尾敏感性检查进一步显示：

```text
median relative difference:      0.99%
upper-1%-trimmed mean difference: 0.18%
raw mean difference:             1.99%
```

CUDA 样本中一个约 197 TeV 的 ground-energy shower（CPU 样本最大约 109 TeV）
明显影响了有限样本 raw mean。因为两组 seed 独立，这不能被逐事件配对消除。
当前结果是“严格门禁失败，但对该指标统计上尚无结论”，不是已检测到 2% 物理
偏差，也不是通过 1% 等价性检验。

## 4. 验收工具改进

`compare_ensembles.py` 保持原 pass/fail 逻辑不变，同时新增：

- `relative_combined_standard_error`；
- `sigma_scaled_relative_precision`；
- 每个 key scalar 的 `outcome`；
- 顶层 `key_scalar_outcomes` 分类。

重新分析输出：

```text
/tmp/c8_phase48_physics_acceptance_1PeV_200_reanalysis
```

会把唯一失败项明确标为：

```text
relative_threshold_failed_but_statistically_inconclusive
```

这项标注只防止误读，不降低原 1% 验收标准。下一步应保留这 200 个昂贵 CPU
reference shower，在 Release 当前 CUDA 版本上补跑独立 CUDA ensemble；若 raw
mean 仍稳定偏离，再扩大 CPU 样本或针对高能地面尾部做专门采样研究。
