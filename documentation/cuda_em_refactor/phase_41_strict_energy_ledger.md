# CUDA 电磁后端重构记录：阶段 41，严格逐 shower 能量账本

## 1. 为什么原有能量比值不是守恒检验

此前 application 打印：

```text
dEdX + ground total energy
```

并除以初级粒子总能量。这个量适合作为快速诊断，但不能证明能量守恒。
原因有两个：

1. GPU `ParticleCut` 对电子和正电子沉积剩余动能，不把静质量混进
   `dEdX`；
2. 以下原子电子过程的末态总能量包含一个来自介质的电子静质量：

```text
gamma + atomic e- -> gamma + e-       Compton
gamma + atomic e- -> e- + binding     photoelectric
e± + atomic e- -> e± + e-             ionization
e+ + atomic e- -> gamma + gamma       annihilation
```

因此严格账本定义为：

\[
E_{\rm initial}
+E_{\rm medium\ rest}
=E_{\rm deposit}
+E_{\rm cut\ rest}
+E_{\rm observed}
+E_{\rm escaped}.
\]

所有粒子端点项都使用包含静质量的总能量，并乘以 thinning weight。严格
逐事件门槛只在 thinning 关闭时启用；随机 thinning 的加权能量本来就是
无偏估计量，而非逐事件恒等式。

## 2. 设备端确定性实现

在：

```text
corsika/gpu/em/detail/ProfileProjection.hpp
src/gpu/em/CudaProfileProjection.cu
```

为现有 resident profile 定点累加器增加四个 64-bit signed counter：

```text
weighted_medium_rest_mass_input
weighted_cut_rest_mass_energy
weighted_observed_total_energy
weighted_escaped_total_energy
```

它们复用 profile 的：

```text
energy_scale = 2^62 / configured_energy_limit
```

及 checked CAS 加法。因此：

- 不使用非确定顺序的 double `atomicAdd`；
- 同一 GPU、相同配置和 seed 的账本逐位可重复；
- 溢出增加原有 `fixed_point_overflows`，并使 shower 硬失败；
- 不增加逐轨迹 device-to-host 数据；
- shower 结束只随 profile counters 下载常数大小的数据。

### 2.1 transport endpoint

`accumulatePhotonStepsKernel()` 与
`accumulateLeptonStepsKernel()` 直接从 `record.end` 累计：

```text
ObservationSurface -> observed total energy
EscapedEnvironment -> escaped total energy
```

轻子 `ParticleCut` 另外累计：

```text
electron_mass * parent_weight
```

其动能仍由原 `continuous + cut deposit` 路径进入能量沉积。

### 2.2 interaction vertex

`accumulatePhotonFinalStatesKernel()` 对 Compton/photoelectric，
`accumulateLeptonFinalStatesKernel()` 对 ionization/annihilation，分别
根据 parent history/input identity 找回 transport parent weight，并累计：

```text
ElectronMassGeV * parent_weight
```

轻子在连续损失后会重新选择顶点过程，所以 transport candidate 中的旧
`process_id` 不能用于验证最终过程。阶段 41 的第一次端到端运行正是由
新的 invalid-record hard gate 捕获了这个语义差异；修复后按
`input_index + parent_history_id` 关联，过程身份由 final-state record
本身决定。

## 3. host/full-record 兼容路径

`PhysicalCudaEmRouter` 对不启用 resident profile 的调试和 CPU-radio
轨迹路径也维护同样四类项：

- observation records 直接累计 observed/escaped；
- lepton cut 记录静质量；
- photon atomic-electron final state 从匹配的 step 获取 weight；
- 聚合的 resident lepton result 中 `input_index` 会在每个 wavefront
  重用，因此 host 路径按唯一 `history_id` 建立 weight map，不能对整个
  result 做一次全局 `input_index` 二分查找。

生产 resident 路径仍以设备端定点账本为准，不发生双重累计。

## 4. application 硬门槛

`c8_air_shower` 在每个 CUDA shower 的
`gpu_em/summary.yaml` 中写入：

```text
statistics.energy_ledger.definition
statistics.energy_ledger.complete_coverage
statistics.energy_ledger.scalar_em_steps
statistics.energy_ledger.initial_total_GeV
statistics.energy_ledger.medium_rest_mass_input_GeV
statistics.energy_ledger.deposited_GeV
statistics.energy_ledger.cut_rest_mass_energy_GeV
statistics.energy_ledger.observed_total_energy_GeV
statistics.energy_ledger.escaped_total_energy_GeV
statistics.energy_ledger.source_GeV
statistics.energy_ledger.terminal_GeV
statistics.energy_ledger.residual_GeV
statistics.energy_ledger.relative_closure_error
statistics.energy_ledger.acceptance_tolerance
statistics.energy_ledger.accepted
```

完整覆盖要求同时满足：

```text
EM primary
thinning disabled
zero scalar e-/e+/gamma steps
zero generic CPU fallback
zero specified CPU final states
```

在完整覆盖时：

```text
relative_closure_error > 1e-4
```

立即抛出异常，当前 shower 被 `GpuEmRunOutput` 标为 incomplete。部分覆盖
事件仍输出原始账本项，但不伪装成严格通过。

## 5. 独立验证器

新增：

```text
validation/gpu_em/validate_energy_ledger.py
validation/gpu_em/tests/test_validate_energy_ledger.py
```

验证器重新从原始项计算 source、terminal、residual 和 relative error，
而不是信任 summary 中已经写好的 residual。它还检查：

- 所有项有限且非负（residual 除外）；
- 存储值在 double rounding 范围内自洽；
- 完整覆盖标记；
- \(10^{-4}\) 门槛；
- application `accepted` 标记。

`compare_ensembles.py` 的 CUDA 完整性入口也会拒绝任何声明完整覆盖却没有
通过严格账本的 shower。

## 6. 端到端证据

### 6.1 10 GeV electron

输出：

```text
/tmp/c8_phase41_energy_ledger_10GeV_v2
```

配置：

```text
primary: electron
energy: 10 GeV
zenith: 0 deg
emcut: 0.5 MeV
thinning: off
gpu-min-batch: 1
scalar EM steps: 0
CPU fallback: 0
```

结果：

| 项 | GeV |
|---|---:|
| initial | 10 |
| medium rest input | 0.720508514001 |
| deposited | 9.888091230804 |
| cut rest | 0.832417283197 |
| observed | 0 |
| escaped | 0 |
| residual | \(-1.7764\times10^{-15}\) |
| relative error | \(1.6570\times10^{-16}\) |

### 6.2 100 GeV photon, 45 degrees

输出：

```text
/tmp/c8_phase41_energy_ledger_100GeV_photon_v1
```

结果：

| 项 | GeV |
|---|---:|
| initial | 100 |
| medium rest input | 7.784046945942 |
| deposited | 98.912675966527 |
| cut rest | 8.869408707458 |
| observed | 0 |
| escaped | 0.001962271956 |
| residual | \(2.8422\times10^{-14}\) |
| relative error | \(2.6369\times10^{-16}\) |

这组事件同时验证 photon primary、斜 shower 与 escape 端点项。

独立汇总：

```text
/tmp/c8_phase41_energy_ledger_acceptance.json

events:                         2
complete events:                2
maximum relative closure error: 2.6369124407305587e-16
status:                         passed
```

相对于要求的 \(10^{-4}\)，当前残差由 double 舍入主导，余量约十二个
数量级。

## 7. 回归

```text
testGpuPhotonWavefront:
  52,927 checks passed

Python validation:
  10 tests passed

CUDA test suite:
  24 / 24 passed, 9.63 s
```

重点覆盖：

- photon atomic-electron input oracle；
- electron/positron weighted cut rest mass；
- observed/escaped endpoint oracle；
- resident profile 无逐轨迹回传；
- host aggregated lepton wavefront index reuse；
- invalid association 与 arithmetic corruption 均被拒绝。

## 8. 当前边界

严格账本关闭的是原计划中的“纯 EM、无 thinning、完整 GPU 覆盖”
\(10^{-4}\) 验收项。以下仍是独立的生产验收任务：

- thinning 开启时的 ensemble 能量分布；
- 稀有非 EM CPU fallback 的完整介质/核静质量账本；
- 强子 shower 的全粒子种类能量账本；
- 1 PeV、\(10^{17}\)、\(10^{18}\) eV 统计矩阵。

因此 Phase 41 证明了纯 EM 能量闭合，但不等于整个 GPU 重构已经完成。
