# Phase 54：最小 batch 阈值诊断

## 1. 问题

80° shower 中 GPU 队列尾部可能长期小于默认 batch。需要验证把
`--gpu-min-batch` 从 64 降为 1 是否能消除 CPU 前沿展开并提高性能。

## 2. 同配置实测

使用同一 1 TeV electron/80° 配置，各运行 5 个事件：

```text
gpu-min-batch = 64   total shower time = 32.724 s
gpu-min-batch = 1    total shower time = 54.241 s
```

`min-batch=1` 确实把 CPU wavefront expansion 降到零，但 resident
lepton wavefront 从约 12,469 增加到约 85,453。大量只有几个粒子的 kernel
启动和同步使总时间变为原来的 1.66 倍。

## 3. 决策

生产默认值继续使用 64。这个实验说明：

- “更多粒子在 GPU 上计算”不等于“更快”；
- GPU 需要足够的并行度摊薄 kernel launch、scan 和同步成本；
- 小前沿的有限 CPU 展开是调度优化，不是物理过程 CPU 回退；
- 正式统计分别记录 scalar expansion 与 specified physics fallback，二者
  不能混为一谈。

默认 4096 仍是后端通用配置接口的保守值；当前 21CMA 应用级性能验收使用
实测更合适的 64。
