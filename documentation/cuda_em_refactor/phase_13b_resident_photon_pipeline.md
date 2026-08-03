# CUDA 电磁后端重构记录：阶段 13B，驻留式光子级联

## 1. 本阶段结论

阶段 13B 已把以下光子物理链连接为同一条 device-resident pipeline：

```text
photon device queue
  → PROPOSAL 表格率与逆 CDF 采样
  → 五层球形大气最近限制输运
  → interaction vertex 稳定压缩
  → photon-pair 末态与 LPM acceptance
  → boundary / LPM-suppressed photon 稳定压缩
  → 下一轮 photon device queue
```

selection 的紧凑 interaction 数组不再下载后上传给 transport，transport
产生的 interaction vertex 也不再经过主机进入 pair/LPM kernel。跨
wavefront 的 photon continuation 使用两个预分配 `DeviceWorkspace` 交替，
初始 photon batch 只上传一次。

当前只有下列结果离开 GPU：

- CPU-only 或尚未实现过程的显式 `ProposalFallbackEvent`；
- photon pair 产生的 \(e^-\) 与 \(e^+\)；
- 到达观测面或逃逸环境的 `ObservationRecord`；
-科研验证所需的 pair final-state record。

因此，本阶段实现的是“完整 photon 分支驻留”，还不是完整的电磁 shower
驻留。下一阶段必须实现 \(e^\pm\) 输运，才能把 photon pair 的次级粒子继续
留在设备上。

## 2. 为什么不能继续使用 host bridge

原验证路径每个 wavefront 依次执行：

```text
host photons
  H2D → selector → D2H interactions
  H2D → transport → D2H transport records
  host compact interaction vertices
  H2D → pair/LPM → D2H final states
  host compact next photons
```

即使 Phase 13A 已消除 kernel 间的 `cudaMalloc/cudaFree`，中间结构仍会反复
穿过 PCIe。尤其在高能 shower 前沿很宽时，传输的是完整
`EmInteractionRecord` 和 `PhotonTransportRecord`，而不是少数真正需要 CPU
处理的事件。

新路径把 host wrapper 降为验证 oracle，生产型内部接口传递 device pointer、
count 和 workspace slice。主机不再承担正常物理分支的调度。

## 3. Device batch stage 接口

新增内部头文件：

```text
corsika/gpu/em/detail/DeviceBatchStages.hpp
```

其中定义设备批次 view：

- `DeviceInteractionSelectionBatch`
- `DevicePhotonTransportBatch`
- `DeviceTransportInteractionBatch`
- `DevicePhotonPairFinalStateBatch`
- `DevicePhotonEndpointBatch`
- `DevicePhotonPipelineBatch`

每个 stage 都有两类接口：

1. `append...Workspace()`：按固定顺序计算本 stage 需要的 arena bytes；
2. `launch...OnDevice()`：从已经准备好的 workspace 获得切片并启动 kernel。

组合器首先完整计算整条 pipeline 的空间，再调用一次 `prepare()`。这保证后续
stage 不会扩容 arena，从而不会使上游 stage 返回的 device pointer 失效。

## 4. 稳定压缩和错误检测

所有物理输出仍按输入 source index 排序：

1. 每个 source 只有一个 flag；
2. CUB exclusive scan 计算紧凑数组 offset；
3. scatter kernel 写入唯一确定的位置；
4. 下一轮 queue 顺序不依赖 block 调度顺序。

boundary photon 与 LPM-suppressed photon 都写入同一套 `raw_next[source]`。
`atomicCAS` 只用于检测同一个 source 被错误地产生两次；它不负责分配输出
位置，因此不会引入非确定顺序。观测记录使用独立 flag/offset 数组。

若 source index 越界或一个输入产生两个互斥终点，device error flag 会使当前
shower 失败，不能静默丢弃或覆盖粒子。

## 5. 双 arena 驻留循环

后端现在持有：

```cpp
detail::DeviceWorkspace physical_workspace_;
detail::DeviceWorkspace physical_workspace_next_;
```

两者在初始化时各获得一半剩余显存预算。每一轮执行：

```text
arena A: current photon queue + pipeline temporaries
arena B: compact next-photon queue
swap(A, B)
```

下一轮 selector 直接读取上一轮 compact queue。只有 cascade 完成、显式
fallback、观测输出，或达到调用者设置的 `max_wavefronts` 时才下载数据。

达到 wavefront 上限不是粒子损失：`ResidentPhotonCascadeResult` 返回
`completed=false` 和完整的 `remaining_photons` checkpoint，调用者可以继续。

结果结构还记录：

- wavefront 数；
- peak resident photon 数；
- transport record 总数；
- interaction vertex 数；
- layer-boundary 数；
- LPM suppression 数；
- e± 次级、fallback、observation 和 final-state record。

## 6. 公共验证入口

新增或重构的后端入口为：

```cpp
selectAndTransportPhotonsForValidation(...)
runPhotonDevicePipelineForValidation(...)
advancePhotonWavefrontForValidation(...)
runResidentPhotonCascadeForValidation(...)
```

前三个接口保留足够的主机可见中间记录，用于与旧 bridge 做逐项 oracle
比较。`runResidentPhotonCascadeForValidation()` 才是跨 wavefront 驻留路径；
它不会下载正常 transport record。

这些接口仍带 `ForValidation` 后缀，因为完整 `HybridCascade` 尚未路由
\(e^\pm\) device queue。当前不能把 photon-only 局部闭合误称为完整 shower
生产后端。

## 7. 数据传输统计

`GpuEmStatistics` 新增：

```text
physical_host_to_device_bytes
physical_device_to_host_bytes
```

统计只覆盖物理 pipeline 的显式 payload copy，便于区分 kernel 性能与 PCIe
调度开销。

4096 photon 固定种子验证得到：

```text
第一轮：
  2141 interactions
  1948 layer boundaries
   630 accepted photon-pair final states
   402 explicit CPU final-state fallbacks

单轮 device pipeline 相对三个旧 bridge：
  H2D 减少 1,395,520 bytes
  D2H 减少   573,552 bytes

完整 resident photon cascade 相对 host wavefront loop：
  H2D 减少   602,000 bytes
  D2H 减少 5,991,792 bytes
```

单轮节省值由测试按结构体实际 `sizeof` 和 compact count 精确推导，不是手工
估算。resident 对比使用完全相同的输入、Philox key、表格和 wavefront 上限。

## 8. 物理与确定性验证

`tests/gpu/testGpuPhotonWavefront.cpp` 同时运行两组 oracle：

### 8.1 旧 bridge 对比单轮 device pipeline

逐项比较：

- selection fallback；
- transport limit、距离、grammage、密度和顶点状态；
- process、component、\(v\) 与随机 draw ID；
- pair 能量分配、方向、LPM probability/uniform；
- e± PID、能量、方向、history/parent ID；
- boundary/LPM next queue 和 observation。

### 8.2 host wavefront loop 对比 resident cascade

host oracle 每轮下载结果并重新上传 `next_photons`。resident path 只在 GPU
交换 arena。两者逐项比较：

- wavefront/peak/counter；
- 所有 fallback；
- 所有 e± 次级；
- final-state record；
- observation；
- 完成状态和剩余 checkpoint。

固定输入得到：

```text
36,241 checks passed
```

完整 CUDA test suite：

```text
12/12 passed
0 failed
```

## 9. 改动文件

| 文件 | 作用 |
|---|---|
| `corsika/gpu/em/detail/DeviceBatchStages.hpp` | device stage view、workspace 和 launcher 声明 |
| `src/gpu/em/CudaInteractionSelector.cu` | selector device stage 与旧 bridge oracle |
| `src/gpu/em/CudaPhotonTransport.cu` | transport device stage 与旧 bridge oracle |
| `src/gpu/em/CudaPhotonPairFinalState.cu` | pair/LPM device stage 与旧 bridge oracle |
| `corsika/gpu/em/CudaPhotonSelectionTransport.hpp` | chained validation API |
| `src/gpu/em/CudaPhotonSelectionTransport.cu` | pipeline 组合、endpoint 稳定压缩 |
| `corsika/gpu/em/CudaEmBackend.hpp` | resident cascade 公共入口 |
| `src/gpu/em/CudaEmBackend.cu` | 双 arena 生命周期、驻留循环和传输统计 |
| `corsika/gpu/em/Types.hpp` | resident result 和统计 schema |
| `tests/gpu/testGpuPhotonWavefront.cpp` | bridge/pipeline/resident 三路确定性 oracle |

## 10. 当前边界与下一阶段

尚未完成的关键路径：

1. photon pair 的 \(e^\pm\) 目前仍下载到 host；
2. electron/positron bremsstrahlung 尚未实现 GPU final state；
3. continuous ionization、multiple scattering、磁偏转尚未进入 e± transport；
4. Epair、annihilation 和稀有 CPU fallback 尚未形成混合闭环；
5. 还没有完整 \(10^{17}\)–\(10^{18}\,\mathrm{eV}\) shower 的物理/性能验收。

下一阶段先实现 `BremsElectronScreening` 的表格采样、LPM suppression 和
\(e^\pm\to e^\pm+\gamma\) 末态，并把生成的 photon 与 surviving lepton
分别压入统一的下一轮 EM device queue。只有这条主增殖链留在 GPU 上以后，
才有资格测量完整 EM shower 的端到端加速。
