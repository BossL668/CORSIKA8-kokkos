# Phase 56：50 MeV cut 表格与严格物理验收

## 1. 50 MeV 表格生成中的停止条件

50 MeV 表在 positron ionization 逆 CDF 的高分位处遇到最大相对误差
\(7.522687\times10^{-4}\)。反复增加能量点后误差不再下降，表明限制来自
PROPOSAL 插值逆求根的局部数值分支，而不是能量网格过粗。

原生成器同时使用：

- 用户验收容差：\(10^{-3}\)；
- 内部细化目标：用户容差的 0.75 倍，为独立 validation 留 headroom。

局部误差略高于内部目标 \(7.5\times10^{-4}\)，但低于用户明确要求的
\(10^{-3}\)。生成器 0.13 的停止规则改为：

1. 未达到点数上限时继续按内部目标细化；
2. 达到点数上限且完整 adaptive probe 仍超过用户容差时，hard fail；
3. probe 已低于用户容差时，允许保留有界网格；
4. 最终仍必须通过未参与细化的独立 validation。

这不是放宽 \(10^{-3}\) 验收标准，只避免因无法改善的 PROPOSAL 数值细节
无限增加节点。

## 2. 生产表

文件：

```text
gpu_em_tables/production_v10_cut50MeV_1e-3_1EeV.c8emrt
```

metadata：

```text
format                         10
generator                      c8-gpu-em-tablegen-0.13
PROPOSAL                       7.6.2
cut                            50 MeV
energy range                   50 MeV -- 1 EeV
SHA-256                        1b3249acae9aa44731f5d01d1c9b1354902f4f140021da2502129da467a5cc9b
measured max rate error        9.955661e-4
measured max inverse-CDF error 7.522687e-4
```

positron inverse CDF 的独立 validation 最大误差为
\(7.015517\times10^{-4}\)，低于用户容差。

真实 CUDA table test 在 RTX 4060 Laptop GPU 上通过：

```text
CUDA rate-table validation passed 14889 checks
for 2976 device queries, 3 particles
```

## 3. 混合级联 smoke

运行目录：

```text
/tmp/c8_phase56_cut50_gpu_smoke_v1
```

3/3 shower 完整，后两个事件 `backend_lifecycle.reused=true`。没有：

- generic fallback；
- queue overflow；
- device memory spill；
- table/cut/hash mismatch。

第一个事件的 photoproduction CPU fallback 触发 50 MeV cut 对应的冷 PROPOSAL
缓存生成，约占 116 s；总进程耗时 232.84 s。这个冷启动成本不会出现在后续
热缓存事件中。

## 4. 200 + 200 正式验收

运行目录：

```text
/tmp/c8_phase56_electron_1TeV_cut50_200_v1
```

配置：

```text
primary             electron
energy              1 TeV
zenith              45 deg
EM cut              50 MeV
EM thinning         1e-4
maximum weight      100
events              200 PROPOSAL + 200 CUDA
```

严格总体结果：

```text
scalar_pass = true
curve_pass  = true
passed      = true
```

9 个预注册关键均值全部通过：

| 观测量 | 相对差 | \(z\) |
|---|---:|---:|
| charged \(X_{\max}\) | +0.412% | 0.285 |
| charged maximum | +0.803% | 0.814 |
| charged integral | +0.0110% | 0.070 |
| photon integral | +0.269% | 1.274 |
| deposit sum | +0.00600% | 0.588 |
| deposit \(X_{\max}\) | +0.472% | 0.253 |
| ground weighted count | 0 | 0 |
| ground kinetic energy | 0 | 0 |
| energy closure fraction | +0.00600% | 0.588 |

energy deposit、charged/electron/positron/photon/EM 六类有效 longitudinal
curve 均通过。该高 cut 配置没有到达地面的 EM 粒子，因此三个 ground
fraction curve 没有 active bin；零对零的关键标量按精确定义通过，没有把
空曲线当成非零分布证据。

## 5. CUDA 完整性统计

200 个 CUDA shower：

```text
complete showers                 200 / 200
backend reused                   199 / 200
GPU particles                    12,494,987
CPU scalar particle steps             8,857
specified CPU final states            2,523
completed selected losses               116
generic CPU fallbacks                     1
queue overflows / memory spills           0
peak device bytes                605,039,274
total HybridCascade run time          38.308 s
summed CUDA kernel time               34.190 s
```

specified fallback 包括 photoproduction 2,389、photonuclear 18、Epair
低分位 115 和 photon pair 1。唯一 generic fallback 是大气数值边缘的
`invalid_mass_density`，已保留为后续修正目标。

## 6. 结论

50 MeV cut 项已经满足表格、设备查询、应用完整性、全部曲线和 9/9 关键均值
门禁，是 cut 矩阵中第一个在 200 + 200 样本下获得严格总体 PASS 的非默认
cut。

> 后续 Phase 57 统一了层查询与 leapfrog 的 0.1 mm 边界保护。该修复会改变
> 许多边界后轨迹，因此本文件的严格 PASS 是修复前基线，不能直接代表当前
> 代码。修复后 200 + 200 的全部 curve 仍通过、generic fallback 降为零，
> 但 3 个 key mean 在统计不充分状态下超过 1%；当前结论以后续扩大样本为准。
