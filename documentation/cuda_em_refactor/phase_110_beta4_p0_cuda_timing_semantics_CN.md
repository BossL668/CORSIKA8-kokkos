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
- 14 个在当前环境中可执行，全部通过；
- `testModules`（含 FLUKA/PROPOSAL）通过，用时 200.87 s；
- 20 个 CUDA 设备测试返回约定的 skip code 77。

第一次完整门禁发现已有的
`validation/gpu_em/compare_same_random_processes.cpp` 缺少标准版权头；补充版权头
后 `copyright_notices` 单独重跑通过。该改动不影响程序行为。

当前 Codex 执行环境没有 `/dev/dxg`，`nvidia-smi` 报告 GPU access blocked by the
operating system。因此本阶段完成了 CUDA 编译、链接、host/FLUKA/PROPOSAL 回归和
设备测试入口检查，但不能在该受限进程内生成 RTX 4060 的实际 timing schema 2
数值。不能把 skip 解释为 GPU device test pass。

## 5. 在正常 WSL 终端完成设备复测

先验证 GPU 可见：

```bash
nvidia-smi
```

然后运行新增的确定性 timing test：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate corsika_venv
ctest --test-dir "$C8_BUILD" --output-on-failure -R '^testGpuEmCuda$'
```

生产事例使用原命令，只额外加入：

```text
--gpu-detailed-stage-timing
```

应验收：

1. `timing_schema_version == 2`；
2. transfer operation 分类之和等于 `operations`；
3. `device_copy_time_ms`、所有 wait time 有限且非负；
4. fixed seed 的物理输出 SHA-256 与关闭详细计时时一致；
5. 关闭详细计时后的 wall time 才能与 P0 前生产性能比较。

## 6. 下一步判断方法

- 若 `device_copy_time_ms` 很小而 physical pipeline/profile/radio wait 很大，说明旧
  `transfer_time_ms` 主要是同步等待，不应优先优化 PCIe 带宽；
- 若 radio input-slot wait 长，优先做 track-kinematics 预计算和 observer tiling；
- 若 profile input-slot wait 长，优先做差分 histogram 或 block-local reduction；
- 若实际 copy 时间和字节数都显著，再引入 pinned staging、`cudaMemcpyAsync` 和
  double-buffered control summaries。
