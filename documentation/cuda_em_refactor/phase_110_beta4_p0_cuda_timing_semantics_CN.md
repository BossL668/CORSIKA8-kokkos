# 阶段 110：Beta4 P0 CUDA 计时语义与等待分解

## 1. 目标

本阶段不修改任何粒子输运、反应截面、随机数、thinning、profile 或射电物理。
目标是解决旧 `transfer_time_ms` 把真实 copy、等待先前 kernel 和 CUDA API 开销
混在一起的问题，为后续 radio/profile 优化建立可信的性能基线。

开始修改前在提交 `f206b277` 上建立了本地 annotated tag：

```text
beta4-pre-p0-20260819
```

`modules/data/PROPOSAL` 中运行时自动产生的 cache 没有进入版本控制。

## 2. Timing schema 2

旧字段继续输出：

```yaml
transfer_time_ms: ...
kernel_time_ms: ...
```

它们用于兼容已有分析脚本，但语义明确为：

- `transfer_time_ms` 是选定同步 copy 外层的 host API wall time，不是纯 PCIe 时间；
- `kernel_time_ms` 是多个 CUDA stream 的 event duration 之和；物理 pipeline、profile
  和 radio 可以重叠，因此该值可以超过 shower wall time。

当指定 `--gpu-detailed-stage-timing` 时，新增：

```yaml
timing_schema_version: 2
transfer_timing:
  device_event_timing_enabled: true
  operations: ...
  host_to_device_operations: ...
  device_to_host_operations: ...
  device_to_device_operations: ...
  host_api_time_ms: ...
  device_copy_time_ms: ...
  host_wait_upper_bound_ms: ...
synchronization_timing:
  physical_pipeline_waits: ...
  physical_pipeline_wait_time_ms: ...
  profile_input_waits: ...
  profile_input_wait_time_ms: ...
radio:
  input_slot_waits: ...
  input_slot_host_wait_time_ms: ...
```

`device_copy_time_ms` 由紧贴 copy 前后的 CUDA events 测量。event 起点排在先前
default-stream 工作之后，因此不会把先前 kernel 时间算进 copy。`host_api_time_ms`
由主机 steady clock 测量。二者的非负差写入 `host_wait_upper_bound_ms`；它包含隐式
同步等待，也包含少量 CUDA API 和主机计时开销，所以是上界而不是精确的纯等待。

## 3. 实现位置

- `corsika/gpu/em/Types.hpp`
  - 新增 `GpuTransferTimingStatistics`；
  - 新增 `GpuSynchronizationTimingStatistics`；
  - 保留旧字段，避免破坏历史 YAML parser。
- `src/gpu/em/CudaEmBackend.cu`
  - `copyWithTiming()` 统一统计生产路径中的 H2D、D2H 和 D2D copy；
  - detailed timing 打开时复用两个 persistent CUDA events；
  - 记录 physical pipeline 和 profile input-slot 的显式 host wait；
  - 修正 `appendStaging()` 旧计时把 scatter kernel launch 混入 transfer 的问题。
- `src/gpu/em/CudaRadioAccumulator.cu`
  - 记录 radio 双输入槽复用前的 host wait。
- `applications/c8_air_shower.cpp` 与 `c8_ice_cascade.cpp`
  - 输出 timing schema 2 和字段语义。
- `tests/gpu/testGpuEmCuda.cu`
  - 检查 detailed transfer timing 已启用；
  - 检查 H2D/D2H/D2D 操作计数闭合；
  - 检查所有计时为有限非负数；
  - 原有固定输入确定性比较继续保留。

正常生产模式不会创建 transfer timing events，也不会增加新的 device
synchronization；只保留原本已有的 host wall 计时和操作计数。详细模式会为每次
被测 copy 增加 event completion wait，因此只用于 profiling，不能直接作为正式
production wall-time 基准。

## 4. 本地构建和测试

Release/CUDA 全树构建通过，包括：

```text
CORSIKA8GpuEm
testGpuEmCuda
c8_air_shower
c8_ice_cascade
all configured targets
```

完整 CTest 结果：

- 34 个已注册测试；
- 2026-08-19 在 `corsika_venv` 和 RTX 4060 Laptop GPU 上重新执行后，34/34
  全部通过，无 skip、无失败，总用时 218.76 s；
- 其中 27 个 `testGpu*` 测试全部通过，用时 18.54 s，覆盖 rate table、过程选择、
  wavefront bucketing、pair/brems/Epair LPM、EM thinning、Molière、球形大气、
  磁场、轻子输运、光子 wavefront 和 GPU radio projection；
- `testModules`（含 FLUKA/PROPOSAL）通过，用时 199.05 s。

第一次完整门禁发现已有的
`validation/gpu_em/compare_same_random_processes.cpp` 缺少标准版权头；补充版权头
后 `copyright_notices` 单独重跑通过。该改动不影响程序行为。

首次提交前的受限执行进程没有 `/dev/dxg`，所以设备测试曾按约定返回 skip code
77；后续补测环境已经能访问 `/dev/dxg`，`nvidia-smi` 确认设备为 NVIDIA GeForce
RTX 4060 Laptop GPU，driver 560.94，CUDA runtime capability 12.6，Conda 环境中的
`nvcc` 为 12.6.85。上述 34/34 结果取代首次的 skip 结果。

## 5. RTX 4060 生产路径补测

使用 10 GeV electron、zenith 80 degree、azimuth 180 degree、IGRF14/2027、
`emthin=1e-4`、GPU EM 和 GPU CoREAS/ZHS 跑一个固定 seed 8119001 的完整事例。
输出状态为 `complete: true`，得到：

- GPU particles：19555；
- photon/lepton resident wavefronts：22/213；
- radio tracks：17912；
- CPU generic fallback、CPU specified final state、memory spill、queue overflow：均为 0；
- peak device memory：553750590 bytes；
- shower run wall time：6878.55 ms（程序冷启动端到端 14.34 s）。

详细计时结果：

- transfer operations：377，等于 H2D 95 + D2H 111 + D2D 171；
- device copy：14.2063 ms；
- host API：55.5715 ms；
- host wait upper bound：41.3652 ms；
- physical pipeline wait：235 次、7.0240 ms；
- profile input wait：235 次、0.1308 ms；
- radio input-slot wait：213 次、489.3130 ms。

因此在该小型事例中，真实 copy 时间不是首要成本；radio input-slot wait 明显更大。
这只是用于定位下一步优化对象的单事件 profiling 结果，不能当作生产系综性能结论。

随后以完全相同的 seed 和物理参数关闭 `--gpu-detailed-stage-timing` 重跑。以下七类
物理输出的 SHA-256 均逐字节相同：

1. `profile/profile.parquet`；
2. `production_profile/profile.parquet`；
3. `energyloss/dEdX.parquet`；
4. `particles/particles.parquet`；
5. `interactions/interactions.parquet`；
6. `CoREAS/observers.parquet`；
7. `ZHS/observers.parquet`。

这证明详细计时 instrumentation 在该固定轨迹测试中不改变 shower、地面输出或
GPU 射电波形。关闭详细计时的 shower wall time 为 6931.60 ms；单次冷启动的时间
波动大于 instrumentation 差异，正式性能比较仍应关闭详细计时并使用热缓存多次
重复。

## 6. 下一步判断方法

- 若 `device_copy_time_ms` 很小而 physical pipeline/profile/radio wait 很大，说明旧
  `transfer_time_ms` 主要是同步等待，不应优先优化 PCIe 带宽；
- 若 radio input-slot wait 长，优先做 track-kinematics 预计算和 observer tiling；
- 若 profile input-slot wait 长，优先做差分 histogram 或 block-local reduction；
- 若实际 copy 时间和字节数都显著，再引入 pinned staging、`cudaMemcpyAsync` 和
  double-buffered control summaries。
