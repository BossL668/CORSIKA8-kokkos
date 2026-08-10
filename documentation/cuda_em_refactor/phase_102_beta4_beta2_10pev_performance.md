# Phase 102：beta4 与 beta2 的 10 PeV CUDA 性能对比

## 配置

在同一台 NVIDIA GeForce RTX 4060 Laptop GPU 上交替运行三对固定 seed 事例。
初级为 `1e7 GeV` 垂直质子，`emcut=5e-4 GeV`、`emthin=1e-6`、
`max-weight=100`，使用 IGRF14/2027、81 个天线、CUDA EM 与 CUDA radio。
两版都使用同一张 contract 0.18、误差 `5e-4` 的物理表，显存比例为 70%，
强子路径为固定到同一逻辑核的标量后端。

## 结果

| seed | beta2 | beta4 | beta4/beta2 |
|---:|---:|---:|---:|
| 2026089101 | 315.10 s | 318.81 s | 1.0118 |
| 2026089102 | 258.21 s | 264.92 s | 1.0260 |
| 2026089103 | 312.10 s | 313.83 s | 1.0055 |

beta2 与 beta4 的平均 wall time 分别为 295.14 s 和 299.19 s；配对比值的
均值为 1.0144，中位数为 1.0118。三个事例总时间的 beta4/beta2 比值为
1.0137。按总 GPU 粒子数归一化后的 wall time、kernel time 分别增加 2.24%
和 1.71%；按 radio track-observer pair 归一化后的设备时间增加 1.48%。

六个事例全部 `complete/closed`，两版均无显存 spill，三例合计各出现一个近
观测面的显式几何返回。人工运行期采样显示两版在主要级联阶段均能长期保持约
93--100% GPU 利用率，最高外部显存采样约 5.06 GiB。

## 解释

beta4 修复了连续轨迹、观测平面、首相互作用输出和时间 cut 等物理语义；即使
seed 相同，修复后的后续 shower 也不再与 beta2 逐粒子一致。本组 GPU 粒子数
的配对比值为 0.915--1.169，因此单个 wall-time 差不能全部解释为 kernel
回退。三对数据只支持“beta4 与 beta2 基本处于同一性能水平，当前差异约
1--3%”的结论，不提供严格的性能置信区间。

原始输出、CSV、JSON、图和完整配置保存在 D 盘结果目录：

```text
CorsikaData/corsika_validation_results/
  beta2_beta4_performance_proton_10PeV_vertical_emthin1e-6_3paired_20260809_v1/
```
