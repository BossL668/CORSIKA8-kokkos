# Phase 61–64：\(10^{17}\)–\(10^{18}\,\mathrm{eV}\) 正式 CPU/CUDA 系综

## 1. 范围

此前 Phase 33 只完成少量 UHE CUDA 完整性、确定性和射电 smoke。本阶段按
原测试矩阵的事件数，第一次运行独立 CPU PROPOSAL/CUDA 系综，并同时覆盖
electron、photon 和 0°/45°/80°：

| phase | primary | energy | zenith | CPU + CUDA |
|---|---|---:|---:|---:|
| 61 | electron | \(10^{17}\) eV | 0° | 50 + 50 |
| 62 | electron | \(10^{18}\) eV | 0° | 20 + 20 |
| 63 | photon | \(10^{17}\) eV | 45° | 50 + 50 |
| 64 | photon | \(10^{18}\) eV | 80° | 20 + 20 |

所有运行使用：

```text
EM cut                  0.5 MeV
EM thinning             1e-3
maximum weight          1e6
GPU minimum batch       64
table                   production_v9_1e-3_1EeV.c8emrt
non-EM cuts             1e13 GeV
```

高 non-EM cut 不删除稀有过程：GPU 仍选择过程，CPU PROPOSAL 仍生成指定
hadron/μ/τ 末态并计数，随后由 CORSIKA cut 处理。这样隔离了已知 UrQMD
进程级退出，同时保留 EM 后端的回退完整性。

## 2. 输出

```text
/tmp/c8_phase61_electron_1e17eV_zenith0_50_v1
/tmp/c8_phase62_electron_1e18eV_zenith0_20_v1
/tmp/c8_phase63_photon_1e17eV_zenith45_50_v1
/tmp/c8_phase64_photon_1e18eV_zenith80_20_v1
```

## 3. 完整性与规模

| sample | complete | GPU particles | specified CPU final states | peak device | generic / overflow / spill |
|---|---:|---:|---:|---:|---:|
| e, \(10^{17}\) | 50/50 | 49,923,758 | 9,208 | 687.5 MB | 0 / 0 / 0 |
| e, \(10^{18}\) | 20/20 | 43,250,022 | 4,952 | 956.0 MB | 0 / 0 / 0 |
| γ, \(10^{17}\), 45° | 50/50 | 51,182,551 | 9,197 | 687.5 MB | 2 / 0 / 0 |
| γ, \(10^{18}\), 80° | 20/20 | 53,252,888 | 5,398 | 956.1 MB | 0 / 0 / 0 |

Phase 63 的两个 generic 项均为计划允许、显式计数的
`unsupported_geometry` 标量回退；没有数值 transport failure。四组样本
均在 RTX 4060 Laptop 8 GiB 的 70% 显存预算内完成。

命名 CPU fallback 包括：

- `cpu_only_process`；
- `loss_quantile_out_of_range`；
- 少量 `loss_energy_out_of_range`；
- Phase 61/62/63 中分别 3/1/3 次
  `epair_rejection_envelope_exceeded`。

没有 `invalid_final_state`、表哈希错误、CUDA error 或残留设备粒子。

## 4. 曲线验收

四组样本的全部 longitudinal profile、energy-deposit curve 和 active
ground histogram bin 均通过统一曲线门禁：

```text
curve_pass = true
```

这覆盖表格 \(10^{12}\,\mathrm{MeV}\) 闭区间端点、LPM 长级联、UHE
thinning、球形大气 0°/45°/80° 和 electron/photon 初级。

## 5. 关键均值

| sample | 1% 内关键均值 | 超过 1% 但统计不显著的最大 \(|z|\) |
|---|---:|---:|
| e, \(10^{17}\) | 5 / 9 | 0.784 |
| e, \(10^{18}\) | 3 / 9 | 0.640 |
| γ, \(10^{17}\), 45° | 3 / 9 | 1.571 |
| γ, \(10^{18}\), 80° | 6 / 9 | 0.699 |

UHE thinning 的 event-to-event 方差远大于表格误差。所有越过 1% 的均值都
低于 \(1.6\sigma\)，因此没有统计证据支持 CPU/GPU 系统偏差；但原验收
要求同时满足 1% 和 \(3\sigma\)，所以四组总体 `scalar_pass` 均严格保留为
false，而不是用“统计一致”替代 1% 条件。

能量相关量最稳定：

- Phase 61 deposit sum 差 0.085%，closure 差 0.302%；
- Phase 62 closure 差 0.090%；
- Phase 63 deposit sum 差 0.102%，closure 差 0.222%；
- Phase 64 deposit sum/closure 均差 0.299%。

## 6. 结论

原计划明确列出的 UHE 事件数已经对 electron 和 photon 各执行一遍，所有
shower shape 与生产完整性门禁通过。尚未完成的是用足够大的 UHE thinning
系综把每一个高方差 raw mean 的置信精度压到 1%；这属于仍需保留的统计
验收缺口，不能被当前 50/20 个事件的低 \(z\) 值冒充为已证明的 1% PASS。
