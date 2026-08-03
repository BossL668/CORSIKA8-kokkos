# Phase 51：1 TeV photon-primary 正式系综

## 1. 配置

本阶段补齐正式 photon-primary 入口，不再只依赖单过程或严格小 shower：

```text
primary                     photon
energy                      1 TeV
events                      1000 CPU + 1000 CUDA
zenith / azimuth            0 / 0 degrees
EM cut                      0.5 MeV
EM thinning                 1e-4
maximum weight              100
CPU                         16 shards, one thread each
CUDA min batch              64
table tolerance             1e-3
```

输出位于：

```text
/tmp/c8_phase51_photon_physics_1TeV_1000_v1
```

CUDA 1000/1000 个事件完整结束，所有事件的 process registry 均通过，未出现
queue overflow、显存 spill、profile overflow 或 invalid record。

## 2. 曲线验收

所有 active bin 都通过三标准误差统计门禁：

```text
energy deposit                        104/104
charged/electron/positron profile     103/103
EM/photon profile                     104/104
ground energy fraction                 16/16
ground radial fraction                 12/12
ground time-residual fraction          34/34
```

关键纵向量的 CUDA/CPU 相对均值差为：

| observable | relative difference | z |
|---|---:|---:|
| charged \(X_{\max}\) | -0.135% | -0.156 |
| charged maximum | -0.044% | -0.087 |
| charged integral | -0.131% | -0.504 |
| photon integral | +0.039% | +0.145 |
| deposited energy | +0.015% | +1.202 |
| deposit \(X_{\max}\) | +0.641% | +0.615 |
| approximate energy closure | +0.024% | +3.596 |

能量闭合的均值差虽然因样本量大而被统计分辨，但绝对相对差只有
\(2.35\times10^{-4}\)，远小于 1% 物理容差，因此分类为
`statistically_significant_difference_within_relative_tolerance`。

## 3. ground tail 的解释

两个低占空比 ground-tail 均值超过原始 1% 样本均值门禁：

| observable | CUDA-CPU | z | 95% bootstrap interval |
|---|---:|---:|---:|
| weighted ground EM count | +10.21% | 0.861 | [-11.74%, +37.03%] |
| ground EM kinetic energy | +9.68% | 0.704 | [-15.15%, +41.31%] |

它们的事件间方差极大，当前差异不到 \(1\sigma\)。按当前方差，使三标准误差
精度达到 1% 分别需要约 1.27 和 1.70 million 个事件/后端。因此本阶段不能
把这两个量宣告为已分辨偏差，也不能把原始 1% 门禁改写成通过；严格总状态
保持 `FAIL / statistically inconclusive`。

相反，ground radial mean/RMS 的均值差只有 +0.644%/+0.788%，全部径向
直方图 active bin 通过，未出现与 Phase 50 修正前相同的 lateral-shape
系统信号。

## 4. 结论

正式 photon-primary 路径在纵向发展、能量沉积和三类地面 shape 上均通过。
其物理一致性证据与 electron-primary 相互独立。尚未关闭的是极低统计量的
ground particle/energy raw-mean 1% 门禁，而不是 photon GPU 级联本身的
分布失败。
