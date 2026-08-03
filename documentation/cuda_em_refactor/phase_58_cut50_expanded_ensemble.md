# Phase 58：50 MeV cut 的 1000+1000 扩展系综

## 1. 目的

Phase 57 修正球形大气层边界的 \(0.1\,\mathrm{mm}\) ownership guard 后，
50 MeV cut 的 200+200 样本虽然所有 curve gate 通过，但三个关键均值仍因
统计精度不足越过 1% 门槛。本阶段把同一配置扩展到 CPU PROPOSAL 和 CUDA
各 1000 个独立 shower，不沿用修复前的 CUDA 结果。

输出位于：

```text
/tmp/c8_phase58_electron_1TeV_cut50_strict_1000_v1
```

配置为 1 TeV electron、45°、50 MeV cut、EM thinning \(10^{-4}\)、最大
权重 100。CPU 分成 32 个单核 shard；CUDA 复用同一个已初始化后端处理
1000 个 shower。

## 2. 物理验收

全部九组 longitudinal/energy-deposit/ground curve family 通过逐 bin
门禁。九个关键标量中七个满足 1% 和 \(3\sigma\) 双门禁：

| observable | 相对均值差 | \(|z|\) |
|---|---:|---:|
| charged \(X_{\max}\) | 0.546% | 0.757 |
| charged maximum | 0.492% | 0.989 |
| charged integral | 0.0938% | 0.611 |
| photon integral | 0.0903% | 0.546 |
| deposit sum | 0.00646% | 1.473 |
| deposit \(X_{\max}\) | 0.656% | 0.733 |
| energy closure fraction | 0.00647% | 1.476 |

剩余两项来自极稀疏的地面尾部：

```text
CPU ground arrivals:  1 / 1000
CUDA ground arrivals: 2 / 1000
weighted-count z:      0.578
kinetic-energy z:      0.861
```

因此统一 strict result 仍为 `scalar_pass=false`，但不能把 1 对 2 个到达粒子
解释成已分辨的系统偏差。曲线验收为 `curve_pass=true`。

## 3. 完整性与资源

```text
complete showers                 1000 / 1000
GPU particle advances            62,347,236
queue overflows                  0
cross-species host spills        0
generic scalar fallbacks         0
peak device bytes                605,039,274
reused backend showers           999
```

本次运行还暴露三个 Epair 拒绝采样包络越界事件。修复前它们被笼统记录为
`invalid_final_state`，虽然 CPU PROPOSAL 能完成其已经指定的末态，但该名称
违反了“非法末态必须硬失败”的生产策略。Phase 59 对其进行单独分类和授权。

## 4. 结论

修复后的 50 MeV cut 已通过完整 shower shape 验收，七个主体 shower 均值
进入 1%。总体 strict gate 只剩每千次约一个的地面到达尾部没有足够统计量，
原始 FAIL 状态保留，不通过放宽门槛获得 PASS。
