# Phase 46：受控最低能粒子 CPU spill

## 1. 问题

常驻 photon/lepton 跨物种队列有固定显存容量。旧路径在目标队列尾部空间不足时
会把新生成的一批次级粒子整体返回 host。这虽然不会静默丢粒子，但不满足原始
计划的两个要求：

1. 必须优先保留高能、GPU 收益最大的粒子；
2. spill 粒子必须显式进入 CPU 标量输运，而不是被当成普通 GPU 输出。

## 2. 全局低能选择

`CudaEmBackend::Impl::rebalanceCrossSpeciesQueue()` 在队列容量不足或头部消费造成
尾部碎片时执行：

1. 下载目标队列中仍 pending 的全部粒子；
2. 与本 wavefront 新生成的目标物种粒子合并；
3. 按 `energy_GeV` 降序做 `std::stable_sort`；
4. 前 `queue_capacity` 个最高能粒子重新上传并继续常驻；
5. 其余最低能尾部作为 `cpu_spill_particles` 返回。

每次真实 spill 都检查：

```text
minimum(retained energy) >= maximum(spilled energy)
```

不满足时立即抛出异常并终止当前 shower。相同能量保持原相对顺序，因此该选择在
相同设备、seed 和配置下可重复。

正常有连续尾部空间时仍使用 device-to-device append，不发生 host 往返。

## 3. HybridCascade 所有权

resident photon 与 lepton 返回值都新增独立的 `cpu_spill_particles`。物理 router
对每个 spill 粒子执行：

1. 用原 PID、能量、位置、方向、时间、权重和完整 history identity 导入 CORSIKA
   主栈；
2. 把该 transport identity 登记到 `cpu_memory_spill_steps_`；
3. 下一次 `canRoute()` 对它返回 `false`；
4. `ScalarCascadeStepper` 恰好推进一个完整 CPU step；
5. 若该 step 后仍是电磁粒子，后续调度可以再次送入 GPU。

因此 spill 是明确的一步 CPU fallback，不会丢失、重复输运或永久把整条 history
锁在 CPU。

## 4. 统计和能量账本

新增统计：

- `cross_species.spill_rebalances`；
- `cross_species.host_spills`；
- `cross_species.particles_spilled_to_cpu`；
- `cross_species.low_energy_ordering_checks`；
- `cross_species.cpu_spill_steps_executed`；
- 顶层 `cpu_memory_spill_particles`。

CPU spill 属于合法、显式的内存 fallback，但当前严格 GPU 能量账本不观察 CPU
step 的连续沉积。因此只要发生过 spill，`energy_ledger.complete_coverage` 就为
`false`；程序不会错误地把局部账本标记为完整闭合。

## 5. 验证

`testGpuPhotonWavefront` 使用 1% 显存预算，反复产生 charged secondary，直到
常驻 lepton 队列达到容量。测试要求：

- 队列最终正好保留 `queue_capacity` 个粒子；
- `electromagnetic_secondaries` 为空，spill 只出现在专用返回字段；
- `cpu_spill_particles` 非空；
- 保留区是全局最高能部分，CPU 返回区是最低能尾部；
- `particles_spilled_to_cpu` 与返回粒子数完全一致；
- 每次 `host_spill` 都有一次低能边界检查。

该受控压力测试以及 photon/lepton/hybrid 回归均通过：

```text
testGpuPhotonWavefront: 52900 checks
testGpuLeptonTransport: 2697 checks
testGpuHybridRoute:     passed
```

此外，当前 CUDA build 的 `ctest -I 3,27 -j1` 共 25 个 GPU/host-contract
测试全部通过，用时 14.59 s。

真实 10 GeV 正常显存运行
`/tmp/c8_phase46_spill_normal_smoke_v2` 的 spill、rebalance 和 CPU spill step 均为
零，严格能量闭合仍为
\(6.5848822917282037\times10^{-16}\)。这同时验证了快速 D2D 主路径没有因新增
保护逻辑发生物理回归。

正式 UHE science matrix 仍要求生产显存配置下 spill 为零，以避免 CPU fallback
污染性能比较；受限显存时则允许以上受控机制保持物理完整性。
