# CUDA 电磁后端重构记录：阶段 3，wavefront 基础设施

## 1. 本阶段的定位

本阶段建立 CUDA 电磁输运后端的最小可编译骨架，验证以下工程问题：

1. 原有 CPU 构建是否可以完全不依赖 CUDA。
2. CORSIKA 8 的 CMake 工程能否条件式启用 CUDA 12 和 CUDA C++17。
3. 主机粒子状态能否无损转换为设备端 SoA 队列。
4. 每个粒子的随机数能否只依赖物理身份，而不依赖线程号或队列位置。
5. 次级粒子能否通过 count、exclusive scan、write 三阶段稳定分配。
6. current/next 双缓冲能否在 wavefront 结束后交换。
7. 没有可访问 GPU 时，测试能否明确报告跳过，而不是假装 kernel 已通过。

本阶段的 kernel 是专门用于测试队列的 toy branching process，不是电磁物理模型。
它产生的任何粒子数、能谱或轨迹都不能用于科研分析。

## 2. 构建系统

### 2.1 默认 CPU 构建

顶层新增选项：

```cmake
CORSIKA_ENABLE_CUDA=OFF
```

默认值保持 `OFF`。此时：

- 不调用 `enable_language(CUDA)`；
- 不寻找 CUDA Toolkit；
- 不生成 `CORSIKA8GpuEm`；
- 原有 `CORSIKA8` header-only target 不增加 CUDA 链接依赖；
- 纯主机 POD 和 Philox 验证仍然可以运行。

因此普通 CPU 用户不需要安装 `nvcc` 或 CUDA runtime。

### 2.2 CUDA 构建

显式设置 `CORSIKA_ENABLE_CUDA=ON` 后：

- 要求 CMake 3.24 或更新版本；
- 要求 CUDA Toolkit 12.x；
- 要求 C++17 和 CUDA C++17；
- 生成静态库 `CORSIKA8GpuEm`；
- 不启用 `--use_fast_math`；
- 链接 `CUDA::cudart_static`，避免运行时依赖 Conda 临时设置的
  `LD_LIBRARY_PATH`。

本机使用的编译组合为：

```text
CMake       3.31.0
CUDA        12.6.85
nvcc host   GCC 13.3.0
CUDA arch   75，同时保留 PTX
build type  Release
```

CUDA 构建目录与 CPU 构建目录相互独立：

```text
corsika8_gpu_refactor_build_clean   CUDA=OFF
corsika8_gpu_refactor_build_cuda    CUDA=ON
```

阶段 3 的基础设施最初在 `Release` 下验收。进入阶段 4 后，持续开发用的 CUDA
构建目录改为 `RelWithDebInfo`，因为当前 Conan 依赖只为该配置导出了完整的
include/link 属性；这不改变阶段 3 的物理或队列实现。

## 3. 公共数据结构

公共类型定义在：

```text
corsika/gpu/em/Types.hpp
```

### 3.1 `EmParticleState`

`EmParticleState` 是 `alignas(16)` 的 standard-layout、trivially-copyable POD。
所有数值使用固定单位：

| 字段 | 单位或语义 |
|---|---|
| `pid` | PDG 编码，当前接受 22、11、-11 |
| `energy_GeV` | GeV |
| `position_m[3]` | m |
| `direction[3]` | 无量纲单位方向 |
| `time_s` | s |
| `weight` | Monte Carlo 权重 |
| `medium_id` | 环境快照中的介质编号 |
| `history_id` | 稳定的粒子历史编号 |
| `parent_history_id` | 直接母粒子的 history ID |
| `generation` | 级联代数 |
| `step_id` | 该 history 已完成的输运步数 |

CORSIKA 的物理单位类型不能直接复制到设备，因此以后
`HybridCascade -> CudaEmBackend` 的边界必须显式执行单位转换。

### 3.2 记录类型

本阶段同时固定了后续接口需要的记录结构：

- `EmStepRecord`
- `ProposalFallbackEvent`
- `RadioTrackRecord`
- `ObservationRecord`
- `GpuEmStatistics`
- `EmBatchResult`

toy kernel 目前只填充输入/输出粒子计数。物理 step、射电轨迹和 fallback
记录将在对应过程实现后产生。

### 3.3 环境和表格 metadata

`EnvironmentSnapshot` 已固定五层大气的设备侧 schema，包括：

- 地球中心；
- 最多五个大气层；
- 层半径和密度参数；
- medium ID；
- 均匀磁场；
- 观测面半径。

`ProposalTableSet` 当前只有格式版本、过程数和内容哈希。
本阶段只验证 metadata，不做任何 PROPOSAL 表格查询。

## 4. Philox 随机数

实现位于：

```text
corsika/gpu/em/Philox.hpp
```

底层使用项目中已有 Random123 的 `Philox4x32-10`。
零 counter、零 key 的输出通过官方参考向量验证：

```text
6627e8d5 e169c58d bc57ac4c 9b00dbd8
```

随机数输入固定为：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

映射方式为：

```text
Philox 128-bit counter:
  history_id low 32
  history_id high 32
  step_id low 32
  step_id high 32

Philox 64-bit key:
  SplitMix64(seed, shower_id, process_id, draw_id)
```

因此随机数不使用：

- CUDA thread ID；
- block ID；
- 粒子在队列中的下标；
- wavefront 的 batch 大小；
- host 向 GPU 提交的时刻。

当同一个 history 被放到不同队列位置时，只要六元 key 不变，随机数就不变。

## 5. 双缓冲 SoA 队列

公开接口接收 AoS 形式的 `EmParticleState`，便于与 CORSIKA stack 交互。
设备上的正式队列则是 SoA：

```text
current:
  pid[N]
  energy[N]
  position_x[N], position_y[N], position_z[N]
  direction_x[N], direction_y[N], direction_z[N]
  ...

next:
  同样的独立 SoA
```

两组队列不共享数组。一次 wavefront 完成后只交换指针集合：

```cpp
std::swap(device_current_, device_next_);
```

### 5.1 staging

主机侧使用 `std::vector<EmParticleState>` 暂存新粒子。
上传时经过以下路径：

```text
host AoS
  -> device AoS staging buffer
  -> scatter kernel
  -> current SoA
```

调试下载执行相反路径：

```text
current SoA
  -> gather kernel
  -> device AoS staging buffer
  -> host AoS
```

staging buffer 与 current/next 队列分别分配，不会被当作第三个输运队列。

### 5.2 显存管理

初始化时按 `min_batch_size` 预留容量。容量不足时按 2 的幂扩展。
一次扩展同时重新分配：

- current SoA；
- next SoA；
- AoS staging；
- child count；
- child offset；
- CUB temporary storage。

扩展前估算总字节数，并与初始化时记录的
`memory_fraction * free_device_memory` 比较。
超过预算会抛出异常、增加 `queue_overflows`，而不是越界写入。

当前版本尚未实现计划中的低能粒子 CPU spill；因此显存预算不足仍会终止
当前 shower。实现 spill 之后才能把这一异常改成受控 fallback。

## 6. wavefront 算法

实现位于：

```text
src/gpu/em/CudaEmBackend.cu
```

一次 `advanceWavefront()` 的执行顺序是：

1. 将 host staging 粒子 scatter 到 current SoA。
2. `countToyChildren` 为每个输入写入 0、1 或 2。
3. CUB `DeviceScan::ExclusiveSum` 计算每个输入的输出起点。
4. 只回传最后一个 count 和 offset，得到总输出数。
5. `writeToyChildren` 将结果写入 next SoA。
6. 同步检查 kernel 状态。
7. 交换 current 与 next。

没有使用全局原子 append。输入 `i` 的输出区间严格为：

```text
[offset[i], offset[i] + child_count[i])
```

只要输入顺序相同，次级粒子的输出位置和 history ID 分配就相同。

### 6.1 toy branching 规则

toy kernel 使用 0.5 MeV 作为终止能量：

```text
E <= 0.0005 GeV  -> 0 个输出
u < 0.35         -> 0 个输出
0.35 <= u < 0.85 -> 1 个连续粒子
u >= 0.85        -> 2 个等能子粒子
```

单输出保持原 history ID 并增加 step ID。
双输出获得新的 history ID、记录 parent history、generation 加一，并从
step 0 开始。

这只是用来覆盖吸收、连续推进和分支三种队列行为。

## 7. 错误策略

以下情况立即抛出异常：

- backend 重复初始化；
- device 编号非法；
- `memory_fraction` 不在 `(0, 1]`；
- table tolerance 非正；
- atmosphere layer 超过五层；
- table format version 为零；
- 非 gamma/electron/positron 入队；
- 非有限能量或非正权重；
- history ID 为零或无法继续递增；
- size、capacity 或 history ID 溢出；
- 任意 CUDA runtime、kernel launch 或 CUB 错误；
- 预估显存超过配置预算。

CUDA 模式不会在这些错误后静默切换成 CPU。

## 8. 测试

### 8.1 CUDA=OFF

CPU-only 构建成功生成：

- `testGpuEmHost`
- `testFramework`
- `c8_air_shower`

`testGpuEmHost` 完成 150 项检查，覆盖：

- POD/布局性质；
- 固定单位字段；
- Philox 官方参考向量；
- 六元随机键中每个字段确实影响输出；
- `uniformOpen01` 严格位于 `(0, 1)`；
- 默认配置。

原有 `testFramework` 继续通过。

### 8.2 CUDA=ON

以下目标均已成功编译和链接：

```text
CORSIKA8GpuEm
testGpuEmHost
testGpuEmCppLink
testGpuEmCuda
```

`testGpuEmCppLink` 是普通 `.cpp` 可执行文件。它验证调用方不需要改成 CUDA
translation unit，也能链接已经完成 device-symbol resolution 的静态后端。

设备测试使用 `SKIP_RETURN_CODE=77`，所以没有 GPU 的构建机可以明确跳过。
受限执行沙箱中无法访问 GPU，因此沙箱内结果为：

```text
nvidia-smi:
  Failed to initialize NVML: GPU access blocked by the operating system

CTest:
  testGpuEmHost     Passed
  testGpuEmCppLink  Passed
  testGpuEmCuda     Skipped
```

在获得设备执行权限后，WSL 可以访问：

```text
GPU                 NVIDIA GeForce RTX 4060 Laptop GPU
driver              560.94
memory              8188 MiB
compute capability  8.9
```

真实 GPU 上的两次独立确定性运行都由 16 个输入产生相同的 11 个输出。
这实际覆盖了 scatter、SoA、CUB scan、branch write、双缓冲交换和 gather。

压力模式以 262144 个初级粒子为一个完整 drain，重复创建和销毁 backend。
累计结果为：

```text
drained showers     8
particles advanced  10,475,643
particles produced   8,378,491
queue overflows      0
final queues          empty
```

NVIDIA Compute Sanitizer 在当前 WSL/WDDM 配置下无法启动 debugger interface，
要求先在 Windows 管理员环境运行 `EnableDebuggerInterface.bat`。这是工具权限限制；
被包装的 toy 测试本身仍完成并产生确定性的 11 个输出，但本阶段没有把这次
sanitizer 调用记为通过。

## 9. 可复现构建命令

CPU：

```bash
/usr/bin/cmake \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  -DCORSIKA_ENABLE_CUDA=OFF

/usr/bin/cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  --target testGpuEmHost testFramework c8_air_shower -j2
```

CUDA：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -DCMAKE_TOOLCHAIN_FILE=/home/yuhanglu/21CMA/corsika-21cma/corsika/conan_cmake/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_COMPILER=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCORSIKA_ENABLE_CUDA=ON \
  -DPYTHON_EXECUTABLE=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python3

/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target CORSIKA8GpuEm testGpuEmHost testGpuEmCppLink testGpuEmCuda -j2
```

## 10. 下一步

端到端 toy 路由已经在阶段 4 完成，见
[phase_04_toy_hybrid_route.md](phase_04_toy_hybrid_route.md)。

阶段 3 尚可独立补充的设备验收包括：

1. 在 Windows 管理员环境启用 WDDM debugger interface，再运行 Compute
   Sanitizer。
2. 增加低显存预算、非法输入和 history overflow 的设备集成测试。
3. 加入 pinned host staging 和预分配策略，测量 transfer 与 kernel 时间。
4. 实现显存不足时的显式 CPU spill。

后续主线进入 PROPOSAL rate/final-state 拆分与表格生成阶段。
