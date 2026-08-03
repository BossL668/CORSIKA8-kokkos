# CUDA 电磁后端重构记录：阶段 13A，可复用设备工作区

## 1. 本阶段结论

阶段 12 的完整 photon wavefront 已经物理闭合，但每一轮会依次调用 selection、
transport 和 final-state 三个 bridge。此前每个 bridge 都执行十几次
`cudaMalloc/cudaFree`，即使 batch 尺寸不变也会重复建立和销毁临时缓冲。

本阶段最初新增一个后端持有的 `DeviceWorkspace`：

- 初始化时按 `min_batch_size` 预分配；
- selection、transport 和 final-state 共享同一块 device allocation；
- 每个阶段只重置 bump offset，并按类型/对齐切片；
- batch 变大时只按 2 的幂增长；
- 每次增长受 `GpuEmConfig::memory_fraction` 的剩余预算约束；
- 统计同时记录 table、resident toy queue、physical workspace 和总峰值。

三个物理 `.cu` 文件中已经没有 `cudaMalloc` 或 `cudaFree`。4096 photon 的三轮
完整 wavefront 验证确认预分配容量没有增长，说明稳定 batch 的 shower
过程中不再发生物理临时缓冲分配。

这一步消除了 allocator 抖动，但还没有消除三个阶段之间的 bulk
device-to-host/host-to-device copy。因而它是设备驻留 wavefront 的内存基础，
不是 Phase 13 的最终状态。

> 后续 Phase 13B 已在相同抽象上扩展为两个预分配 arena。当前 photon queue
> 在两个 arena 间 ping-pong，使上一轮输出可以直接成为下一轮输入；两个
> arena 的容量总和计入 `physical_workspace_bytes`。本文件其余内容保留
> Phase 13A 完成时的历史设计说明。

## 2. Arena 设计

`DeviceWorkspace` 只有一块基础 device pointer，并维护：

```text
device
byte_limit
capacity_bytes
offset_bytes
high_water_bytes
```

调用阶段先使用 `WorkspaceSize` 以完全相同的顺序计算需求：

```cpp
WorkspaceSize required;
required.add<EmInteractionRecord>(count);
required.add<PhotonTransportRecord>(count);
required.add<std::uint32_t>(count);
required.addBytes(cub_scan_bytes);

workspace.prepare(required.bytes());
auto* interactions =
    workspace.acquire<EmInteractionRecord>(count);
```

`prepare()` 是唯一允许增长的位置；任何 `acquire()` 超出预先计算的容量都被
视为内部逻辑错误。这避免某个后续切片触发 reallocation 后使先前 device
pointer 失效。

每次切片按 `alignof(T)` 对齐，CUB temporary storage 按 256 bytes 对齐。
加法、乘法、2 倍扩容和 secondary 的 `2 * count` 都执行溢出检查。

## 3. 初始化和显存预算

backend 先上传 rate table 并建立现有双缓冲粒子队列，再把剩余预算交给
physical workspace：

```text
memory budget
  ├─ versioned PROPOSAL table
  ├─ resident particle SoA / toy scan buffers
  └─ physical DeviceWorkspace
```

初始化预分配量为

```text
4096 + 4096 * min_batch_size bytes
```

4096 bytes/particle 是当前三个阶段中最大临时 schema 的保守上界。实际调用仍
计算精确需求；若将来 record 扩大或 batch 超过初始值，arena 可以在预算内
增长。超过预算直接终止，不会静默 spill 或让 CUDA allocator 决定行为。

新增统计：

- `physical_workspace_bytes`：当前 arena capacity；
- `peak_device_bytes`：table、resident queues 和 workspace 的合计峰值。

后续 resident queue 扩容也把 workspace capacity 纳入预算判断。

## 4. 改动文件

| 文件 | 作用 |
|---|---|
| `corsika/gpu/em/detail/DeviceWorkspace.hpp` | checked sizing、对齐 bump arena 和预算增长 |
| `corsika/gpu/em/CudaInteractionSelector.hpp` | selector 接收共享 workspace |
| `corsika/gpu/em/CudaPhotonTransport.hpp` | transport 接收共享 workspace |
| `corsika/gpu/em/CudaPhotonPairFinalState.hpp` | final-state 接收共享 workspace |
| `src/gpu/em/CudaInteractionSelector.cu` | 用 arena slices 替换 7 组临时 allocation |
| `src/gpu/em/CudaPhotonTransport.cu` | 用 arena slices 替换 7 组临时 allocation |
| `src/gpu/em/CudaPhotonPairFinalState.cu` | 用 arena slices 替换 16 组临时 allocation |
| `src/gpu/em/CudaEmBackend.cu` | workspace 生命周期、初始化预分配和总显存核算 |
| `corsika/gpu/em/Types.hpp` | workspace 显存统计 |
| `tests/gpu/testGpuPhotonWavefront.cpp` | 检查多轮 wavefront 不发生 arena growth |

## 5. 验证

内存重构后，完整 photon wavefront 的确定性结果保持不变：

```text
4096 input photons
2141 interactions
1948 layer boundaries
630 accepted pair final states
402 explicit final-state fallbacks
18416 checks
```

当前完整 GPU suite：

```text
12/12 passed
0 failed
```

源码检查结果：

```text
CudaInteractionSelector.cu:       no cudaMalloc/cudaFree
CudaPhotonTransport.cu:           no cudaMalloc/cudaFree
CudaPhotonPairFinalState.cu:      no cudaMalloc/cudaFree
```

现有 `CudaEmBackend.cu` 中仍有 allocation，是初始化和增长 toy/resident SoA
所需；它们下一步会被物理 resident queue 取代。

## 6. 下一步

下一小阶段必须消除 bulk host round trip：

1. 把三个 kernel 的 device input/output view 从 host validation wrapper 中抽出；
2. selection 的 compact interaction pointer 直接作为 transport 输入；
3. transport 在设备端再分成 interaction、boundary、observation 和 fallback；
4. interaction pointer 直接进入 photon-pair/LPM kernel；
5. boundary 和 LPM suppressed photon 直接 compact 到下一轮 device queue；
6. 只下载 CPU fallback、observation、调试记录和最终需要交给 CPU 的
   secondaries。

完成这些以后，才开始用 CUDA events 对比 workspace bridge 和真正 resident
wavefront 的 kernel/transfer 时间。
