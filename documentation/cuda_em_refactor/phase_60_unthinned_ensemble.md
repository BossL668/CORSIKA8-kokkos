# Phase 60：80°、1 TeV 无 thinning 扩展系综

## 1. 配置

为扩展 Phase 53 的 20+20 无 thinning 诊断，本阶段运行 CPU PROPOSAL 与
CUDA 各 50 个独立 shower：

```text
/tmp/c8_phase60_electron_1TeV_zenith80_unthinned50_v1
```

```text
primary                 electron
energy                  1 TeV
zenith                  80 deg
EM cut                  0.5 MeV
EM thinning             off
CPU shards              10
GPU minimum batch       64
```

## 2. 完整性

```text
complete CUDA showers       50 / 50
GPU particle advances       91,975,629
generic fallback            0
queue overflow              0
host spill                  0
```

显式 PROPOSAL 回退全部为已定义的表能力边界或 CPU-only 过程：

```text
loss_energy_out_of_range       14
loss_quantile_out_of_range     70
cpu_only_process              606
```

## 3. 物理结果

全部 longitudinal、energy-deposit 和 ground curve family 通过逐 bin
门禁。关键均值中：

| observable | 相对均值差 | \(|z|\) | 状态 |
|---|---:|---:|---|
| photon integral | 0.683% | 0.231 | PASS |
| deposit sum | 0.00779% | 0.474 | PASS |
| deposit \(X_{\max}\) | 0.433% | 0.113 | PASS |
| closure fraction | 0.00779% | 0.474 | PASS |
| charged maximum | 2.06% | 1.08 | inconclusive |
| charged integral | 1.27% | 1.52 | inconclusive |

80° 的 ground tail 在 50 个事件中只有 CPU 侧 1 个到达粒子，CUDA 为 0；
count 与 kinetic energy 的差都只有 \(1\sigma\)。profile \(X_{\max}\) 的
事件级拟合在极斜、低能、无 thinning 样本中具有很长的尾部，均值差
11.97% 但只有 \(0.22\sigma\)；相应 ensemble mean curve 本身通过。

## 4. 结论

无 thinning 的应用级覆盖已从 20+20 扩大到 50+50，所有 shape gate 和运行
完整性通过。统一九均值 strict gate 仍因极斜 shower 的稀疏地面尾部与
不稳定单事件 peak fit 保持 FAIL/统计不充分，不能通过事后删指标或放宽
1% 阈值得到 PASS。
