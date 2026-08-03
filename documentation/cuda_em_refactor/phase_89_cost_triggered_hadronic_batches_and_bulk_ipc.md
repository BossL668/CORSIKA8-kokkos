# 阶段 89：按预计耗时触发的强子批次与批量 IPC

## 1. 先明确 `emthin` 的语义

只指定 `--emthin` 是原版支持的完整配置方式，不要求同时显式指定
`--max-weight`。当 `--max-weight` 未给出时，`c8_air_shower` 使用自动 Kobal
权重上限。但是“配置了 thinning”不等于给定能量下必然实际丢弃粒子：原版和
CUDA 的 `EMThinning` 都先检查

```cpp
if (parentWeight >= maxWeight) return;
```

自动上限为

\[
w_{\max}=0.5\,\epsilon_{\rm thin}E_0/\mathrm{GeV}.
\]

因此在本项目的精确组合 \(E_0=10^6\) GeV、`emthin=1e-6` 下，
\(w_{\max}=0.5\)，小于初始粒子权重 1，thinning 无法从 unit-weight 粒子启动。
这个组合实际退化为未薄化的高质量极限。它比 `1e-5` 更慢、保留更多粒子仍然
成立，但原因不是“更弱的 thinning 已经发生”，而是没有发生 thinning。

- 在更高初能或显式 `--max-weight > 1` 时，`emthin=1e-6` 会正常实际薄化；
- 对当前 1 PeV 配置，`1e-6` 是未薄化的最高质量验收档；
- `1e-5` 或 `1e-4` 只能作为明确标记的开发/性能诊断档，不能替代最终
  `1e-6` 数据。

本阶段为快速隔离强子调度开销使用 1 PeV、`emthin=1e-4`、关闭射电的开发档。
这不会改变上述正式验收要求。

## 2. 为什么不能只按粒子数形成强子批次

原来的在线触发条件是：

```text
pending_interactions >= hadronic_min_batch
```

默认 `hadronic_min_batch=64` 时，四个 FLUKA worker 平均每个只得到约 16 个
interaction。分类队列还会把它们拆成许多
`model × species × energy_bin × mass_bin` 微批次，因此 IPC 相对过碎。

受控扫描结果如下。三组运行使用相同主参数和 seed，但改变批次边界会改变请求被
分配到哪个持久 FLUKA 进程；FLUKA 具有进程内部状态，因此不同阈值产生的是不同但
合法的 shower，不能用单 shower 总 wall time 横向判断优劣。可以比较的是每个
interaction 的进程池开销和 worker 负载。

| `min-batch` | 两个 shower 的 interaction | IPC batch | request/IPC batch | process-pool µs/request | worker 累计耗时 max/min |
|---:|---:|---:|---:|---:|---:|
| 64 | 38,466 | 2,429 | 15.84 | 46.56 | 1.020–1.028 |
| 128 | 40,584 | 1,301 | 31.19 | 42.86 | 1.014–1.030 |
| 256 | 41,211 | 677 | 60.87 | 35.27 | 1.043–1.084 |

把 raw count 提到 256 能降低约 24% 的进程池 wall/request，但不同粒子类别的
实际代价不同，因此固定粒子数不是最终接口。

## 3. 新的 cost-triggered flush

### 3.1 队列维护总预计耗时

`ClassifiedHadronicWorkQueue` 现在同时维护：

```cpp
std::size_t size_;
double pending_cost_;
```

每次 enqueue/pop 都同步更新 `pending_cost_`。队列清空后将其精确重置为 0，
避免大量浮点减法留下微小负残差。

实现位置：

- `corsika/framework/core/HadronicWorkQueue.hpp`

### 3.2 正常触发条件

正常 flush 现在要求：

\[
N_\text{pending}\ge N_\text{minimum}
\]

并且

\[
\sum_i \widehat t_i \ge
N_\text{workers}\,t_\text{target}.
\]

默认四个 worker、`target=5 ms` 时，队列要先积累约 20 ms 的预计总工作量。
`minimum_pending_interactions=64` 现在是延迟下限，而不是正常情况下唯一的触发
阈值。

还保留两个活性条件：

1. 如果预测模型系统性低估，pending 数达到
   `workers × hadronic-max-batch` 时用 `capacity` 触发；
2. 如果 scheduler 和 GPU router 都没有可独立推进的工作，使用 `drain`
   触发尾批，避免级联死锁。

每个 shower 输出以下审计字段：

```text
cost_triggered_flushes
capacity_triggered_flushes
drain_triggered_flushes
target_total_cost_ms
flush_loads[].trigger
flush_loads[].predicted_cost_by_worker
flush_loads[].actual_final_state_time_by_worker_ms
```

### 3.3 1 PeV 实测负载质量

固定 seed 51001 的一个 1 PeV、`emthin=1e-4`、无射电 shower：

- 20,204 个 FLUKA interaction；
- 95 个正常 cost-triggered flush；
- 7 个不可继续等待的 drain flush；
- 0 个 capacity flush；
- 正常 flush 平均 207.68 个 interaction，范围 152–226；
- 预测 worker max/min：均值 1.0165，95 分位 1.0255，最差 1.0342；
- 实际 FLUKA worker max/min：均值 1.2765，95 分位 1.4446，最差 1.8955；
- 实际 worker 耗时的平均变异系数为 9.22%。

预测负载比实际负载更整齐是正常的：即使分类、能量相同，单次强子末态的多重数和
生成时间仍会随机波动。重要的是调度决策本身不读取 wall clock，因此不会把操作
系统噪声反馈进物理顺序。

## 4. 分类微批次与 IPC super-batch

分类仍然决定：

- cost estimate；
- largest-processing-time-first worker 分配；
- 同类 FIFO 顺序；
- 调度审计。

但一个 flush 内已经分配给同一 worker 的多个分类微批次，会合并为一个
`HadronicWorkerSuperBatch` IPC envelope。这不会混淆分类统计，也不会改变
worker 内的 request 顺序。

在旧的 `min-batch=64` 双 shower 控制中：

- 分类微批次仍为 12,686；
- IPC batch 从 12,686 降到 2,429，下降 80.9%；
- FLUKA process-pool execute time 从 2,359.50 ms 降到 1,790.85 ms，
  下降 24.1%；
- 所有物理 Parquet 文件 SHA256 逐字节相同。

## 5. 响应协议 v3

协议 v2 的 worker 输出顺序是：

```text
batch header
result header 0
secondaries 0
result header 1
secondaries 1
...
```

即使 request 已合并，父进程仍要执行约两次逻辑读取/interaction。协议 v3 改成：

```text
batch header
contiguous result_headers[N]
contiguous flattened_secondaries[sum(secondary_count)]
```

父进程每个 worker super-batch 只需两个 payload read。worker 先完成本批所有
末态，再一次发送响应。因此 profiler 中生成器等待时间现在主要落在
`poll_wait_time_ms`，而不再被错误地包含在 `receive_time_ms` 中。

同一个固定 seed shower 的 v2/v3 对照：

| 指标 | streamed v2 | bulk v3 |
|---|---:|---:|
| interactions | 20,204 | 20,204 |
| IPC batches | 404 | 404 |
| process-pool execute | 843.51 ms | 695.20 ms |
| poll wait | 3.73 ms | 666.21 ms |
| receive | 824.05 ms | 14.12 ms |
| Hybrid total | 7,493.89 ms | 7,405.63 ms |

`poll + receive` 的归属变化主要是计时定义变得正确；process-pool execute 的
单次下降约 17.6%，Hybrid total 的单次下降约 1.2%，后者仍需多次重复后才能
当作正式性能结论。

以下物理表在 v2/v3 间由 PyArrow 逐项比较完全相同：

- `particles/particles.parquet`
- `profile/profile.parquet`
- `production_profile/profile.parquet`
- `energyloss/dEdX.parquet`
- `interactions/interactions.parquet`

## 6. 可重复性证据

cost-triggered 版本使用相同 seed 和相同配置完整重复两次：

- 每个 flush 的 request hash 相同；
- 每个 flush 去掉 wall-time 字段后的 response hash 相同；
- particles、profile、production profile、dEdX、interactions 的文件
  SHA256 相同。

加入 `flush_loads` 观测后，第一场 shower 的各 PyArrow table 仍与加入观测前
逐项相同。负载记录不参与调度或随机数。

## 7. 测试

已通过：

```text
testFramework
102 个 validation/gpu_em Python unittest
fluka_batch_worker --pool-self-test 1000 --pool-workers 4
```

持久进程池自测对同一批 1000 个 request 执行两遍，response hash 相同。
FLUKA 版本仍为 2025.0.0，四个独立进程均正常初始化和退出。

本阶段冻结二进制：

```text
c8_air_shower SHA256:
7f31b49c0d80510dcdd59a6df30db2e8a47824086b2eb33ce856995e44611fcc

fluka_batch_worker SHA256:
89402364853bfd5a5c7b637c880b6677de0736bd2b955b0e8fd06318991892ec
```

## 8. 当前结论和下一步

强子 GPU 化目前不应直接替代 FLUKA：

- 完整 1 PeV、`emthin=1e-6`、81 天线射电事例中，FLUKA process-pool wall
  只占 Hybrid total 的 0.175%，GPU 强子输运不会改善当时的主瓶颈；
- 在关闭射电的 1 PeV、`emthin=1e-4` 开发档中，旧进程池一度占 Hybrid
  total 的约 17.8%，所以 CPU 多进程批处理值得优化；
- 经过 super-batch、cost-trigger 和 bulk response 后，应重新测量
  `emthin=1e-5` 与正式 `1e-6` 的分项占比，再使用固定规则决定是否启动强子
  GPU 原型。

### 8.1 `emthin=1e-5` 预验收结果

已经完成一个 1 PeV、seed 52001、`emthin=1e-5`、自动 maximum weight、无射电
事例：

| 指标 | `emthin=1e-4` 控制 | `emthin=1e-5` 预验收 |
|---|---:|---:|
| 自动 maximum weight | 50 | 5 |
| GPU particles | 22,375,972 | 172,832,805 |
| Hybrid total | 7.406 s | 25.904 s |
| GPU router | 3.839 s | 23.219 s |
| FLUKA interactions | 20,204 | 15,425 |
| FLUKA process-pool wall | 0.695 s | 0.554 s |
| process-pool / Hybrid | 9.39% | 2.14% |
| IPC wall / interaction | 34.41 µs | 35.93 µs |
| 四 worker 累计耗时 max/min | 1.034 | 1.036 |

两场 shower 的物理发展本来就不同，不能把这一行当作严格 thinning scaling
拟合；但 GPU 粒子数增加约 7.7 倍且运行明显变慢，方向与“`1e-5` 比 `1e-4`
保留更多粒子、数据质量更高”完全一致。

高能 SIBYLL 在 `1e-5` 事例中只有 1,232 次末态、累计 55.1 ms；低能 FLUKA
末态的四 worker CPU 时间总和为 1.910 s，经过并行后的 wall 为 0.554 s。
即使把 SIBYLL 末态完全搬到 GPU，单事例理论上也只能节省约 0.21%；即使连
FLUKA process-pool wall 一并假设为零，也只节省约 2.35%。因此当前不启动
强子 GPU 物理内核，先保留可审计的四进程 CPU 后端。

下一步验收顺序：

1. 使用 YAML adapter 在 `emthin=1e-4` 或 `1e-5` 做无射电多 seed 的原版
   CPU/CUDA 物理预验收；
2. 使用原始 `emthin=1e-6` 做至少一个完整配置重复和阶段计时，并明确标记
   该 1 PeV/自动权重组合实际未发生 thinning；
3. 扩大原版 CPU/CUDA 独立 seed ensemble，验证 \(X_\max\)、纵向曲线、地面粒子
   和射电统计；
4. 仅当优化后的强子进程池仍稳定占 shower wall 的 10% 以上，才推进强子 GPU
   原型；否则优先优化射电或剩余 scalar transport。
