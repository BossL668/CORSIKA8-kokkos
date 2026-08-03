# Phase 55：5 MeV cut 表格与正式 shower 验收

## 1. 目标

此前生产级应用验收主要使用 0.5 MeV EM cut。本阶段生成独立的 5 MeV
PROPOSAL/CUDA 表，并验证：

- 表 metadata 与应用 cut 的 hard compatibility；
- rate、逆 CDF、连续能损和设备上传误差；
- 稀有过程与无法表格化数值区间的显式 CPU 回退；
- 1 TeV electron CPU/CUDA shower 的纵向、沉积和地面分布。

## 2. 原 PROPOSAL Epair 逆 CDF 的数值边界

初始自适应表在约 145--146 MeV、极低分位出现约 5% 的不可收敛误差。新增
`validation/gpu_em/probe_proposal_epair.cpp` 直接调用 PROPOSAL
`CalculateStochasticLoss()` 后确认：

- 该跳变存在于 `proposal_interpolated` 输出；
- 对 nitrogen/oxygen target 都可复现；
- 主要位于 \(u\lesssim1.2\times10^{-4}\)；
- 它不是 CUDA 插值或设备浮点产生的。

连续二维表不能同时复现跳变两侧。生产能力边界设为：

```cpp
EpairLossQuantileMinimum = 2.e-4;
```

最低 0.02% 的已选中 Epair 顶点显式返回 PROPOSAL CPU 末态生成器，并记录
process/reason。该处理不删除反应，也不把数值跳变平滑成新的物理分布。

详细的两层插值关系见 `proposal_and_cuda_interpolation.md`。

## 3. 5 MeV 生产表

文件：

```text
gpu_em_tables/production_v10_cut5MeV_1e-3_1EeV.c8emrt
```

关键 metadata：

```text
format                         10
generator                      c8-gpu-em-tablegen-0.12
PROPOSAL                       7.6.2
cut                            5 MeV
energy maximum                 1 EeV
SHA-256                        6d9b4dc69f896f758194c2db21b23c86e853edc4e11c6a6a79134af00f5f8b83
measured max rate error        9.967184e-4
measured max inverse-CDF error 7.499997e-4
```

实验性的规则三维 Epair \(\rho\) 表无法满足误差门禁，因此没有进入生产文件。
生产路径继续使用已经验证过的 KKP 变量变换 rejection sampler；这与 Phase 34
的设计一致，不是过程覆盖缺失。

## 4. 应用 smoke

3-event thinning smoke：

```text
/tmp/c8_phase55_cut5_gpu_smoke_v1
```

结果为 3/3 complete，后两个 shower 均复用后端，无 queue overflow 或 memory
spill。一次冷启动 CPU specified fallback 建表耗时约 110 s；热缓存事件不再
支付该成本。

另运行一个 unthinned smoke：

```text
/tmp/c8_phase55_cut5_gpu_unthinned_smoke_v1
```

该事件完整处理 418,359 个 GPU 粒子且无 overflow。它实际触发了 scalar EM
前沿和 photoproduction fallback，因此不满足“纯 GPU 完整覆盖”能量账本的
前置条件；账本被正确标为 partial，而不是错误宣称严格闭合失败。

## 5. 200 + 200 正式验收

运行目录：

```text
/tmp/c8_phase55_electron_1TeV_cut5_200_v1
```

配置为 1 TeV electron、45°、5 MeV cut、EM thinning \(10^{-4}\)、最大权重
100。CPU PROPOSAL 和 CUDA 各 200 个独立事件。

全部 curve family 通过：

- energy deposit；
- charged/electron/positron/photon/EM longitudinal profile；
- ground energy/radius/time fraction。

关键均值：

| 观测量 | 相对差 | \(z\) | 严格结果 |
|---|---:|---:|---|
| charged \(X_{\max}\) | -0.211% | 0.094 | PASS |
| charged maximum | -1.794% | 1.606 | 统计不充分 |
| charged integral | -0.238% | 0.553 | PASS |
| photon integral | +0.0488% | 0.214 | PASS |
| deposit sum | +0.00199% | 0.187 | PASS |
| deposit \(X_{\max}\) | +2.425% | 1.368 | 统计不充分 |
| ground weighted count | +51.3% | 0.518 | 稀疏尾部，统计不充分 |
| ground kinetic energy | +8.42% | 0.085 | 稀疏尾部，统计不充分 |

所以曲线验收为 PASS，但统一的 1% key-mean 总门禁仍为 FAIL。报告没有把低
\(z\) 自动改写为通过。

CUDA 200-event 汇总：

```text
GPU particles                  48,021,937
CPU scalar steps                  65,376
specified CPU fallbacks             2,515
generic CPU fallbacks                   4
queue overflow / memory spill            0
peak device bytes              687,624,974
```

specified fallback 主要为 photoproduction 2,348 次、Epair 低分位 146 次和
photonuclear 19 次。4 次 generic fallback 来自大气边缘的
`invalid_mass_density`，保留为待排查项，不能算作预注册的物理回退。

## 6. bootstrap 稀疏零参考修复

地面稀疏观测量的某些 bootstrap 重采样可能恰好得到 CPU mean = 0 而 CUDA
mean 非零。旧脚本会构造无穷相对差并让整个 driver 异常退出。

现在 `analyze_scalar_stability.py`：

- 只对有限的 CPU-relative 样本计算区间；
- 记录 `undefined_zero_reference_repetitions`；
- 有效重复数不足时标为 `inconclusive_zero_reference`；
- 不改变原始 1% 物理门禁。

新增两个零参考单元测试后，validation Python suite 为 38/38 PASS。

## 7. 结论

5 MeV cut 的设备表、混合输运、所有 shower shape 和失败可见性已经通过正式
覆盖。该矩阵项仍需更大地面尾部样本才能满足“所有关键原始均值小于 1%”的
最终要求。
