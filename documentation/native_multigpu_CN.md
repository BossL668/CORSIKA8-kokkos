# 原生多 GPU 入口

CUDA、CUDA/OpenMP 和 HIP 构建直接在 **同一个 c8_air_shower 二进制**中提供
统一的 `--device` 参数：一个编号走单卡，多个编号走多卡。
不需要填写 JSON，也不调用 Python 协调器。
显式开启 `--prefix-workers N` 后，一个 GPU 也可接在静态 CPU 并行前缀之后；
未开启时保留原来的单卡直接执行/多卡串行前缀行为。

## 构建

按主 [README](../README_CN.md) 准备编译器、Conda/venv 和授权 FLUKA，
在源码目录选择所需后端：

~~~bash
# NVIDIA：自动选择本机 GPU 对应的 profile
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda

# Hygon Z100L：先加载 DTK，再显式选择匹配的 profile
source /opt/dtk-22.10.1/env.sh
C8_KOKKOS_PROFILE=hip-vega906-dtk221001 C8_BUILD_JOBS=1 \
  bash tools/build_kokkos.sh hip
~~~

这两种 GPU 构建包含共用协调器，不需要先构建 CUDA/OpenMP 组合版。
需要组合版时仍可使用 `tools/build_kokkos.sh cuda-openmp`。
HIP 的 `c8_hip_devices` 必须与 `c8_air_shower` 放在同一个安装目录。
运行阶段的前沿分配、进程管理、Parquet/YAML 读取和结果合并都使用 C++。
Conan 和构建时的代码生成工具仍按原项目要求安装。

## 使用

~~~bash
cd ../install/cuda/bin
export CORSIKA_DATA="$(cd ../share/corsika/data && pwd)"
./c8_air_shower -p 2212 -E 1e5 -N 10 -s 20260924 \
  --em-backend kokkos-proposal --radio-backend kokkos --kokkos-execution cuda \
  --device 0,1,2,3 --gpu-memory-fraction 0.90 \
  --ring 1 --antenna-file /dev/null -f demo_multigpu
~~~

- `--device 0`：直接运行单卡；`--device 0,1,2,3` 或 `--device 0 1 2 3`：同一例 shower 使用四卡。
  CUDA 编号与 `nvidia-smi` 一致，也接受完整 GPU UUID。HIP 使用 `c8_hip_devices`
  列出的非负编号，不接受 NVIDIA UUID；八卡可写成 `--device 0,1,2,3,4,5,6,7`。
- `-N`：依次运行的 shower 数量；每例使用全部选中的卡。前缀事件种子为 `-s`
  加从 0 开始的事件序号；GPU worker 种子由协调器派生，复用时 CPU 随机流连续使用。
- --gpu-memory-fraction：每个 worker 的预算，默认 0.5，可显式设置到 0.9。
- --multigpu-timeout：每个前缀/worker 的超时秒数，默认 7200；很慢的高能模拟可调大。
- --multigpu-frontier-energy：可选的调度根粒子能量上限（GeV），默认自动选择；它不是物理 cut。
- --prefix-workers：静态 CPU prefix 进程数，默认 0 保留串行；正数与 GPU 数量独立。
- --prefix-split-energy：CPU 分支拆分的能量上限（GeV），默认 0 使用
  `primary_energy / (96 * prefix_workers)`，与 GPU 数量独立；不是物理 cut。
- `--persistent-workers`：协调模式下默认开启；PROPOSAL 且 `-N > 1` 时复用
  GPU worker 进程。用 `--persistent-workers=false` 逐例新建，详见下文。

HIP 使用 `../install/hip/bin`，加载构建时的 DTK 环境，并将示例的
`--kokkos-execution cuda` 改为 `--kokkos-execution hip`。
`--device` 在未显式设置后端时选择该构建的 Kokkos GPU 输运和射电。
若构建时传入 `-DCORSIKA_ENABLE_EGS4=ON`，也可选择 `--em-backend kokkos-egs4`。
不传 `--device` 时保持原有行为；CPU-only 用户仍用原 OpenMP 构建，不传 GPU 编号。
旧的 `--devices` 保留为兼容别名，但只传一个编号时现在同样走直接单卡路径，
不再创建单 worker 的协调目录。`--kokkos-device` 仅保留给旧脚本及内部 worker，
其含义仍是可见设备内的序号；不要与 `--device`/`--devices` 混用。

固定初级能量、物理 cut、磁场、大气、薄化和天线等参数继续传给原空气程序。
不支持在此路径同时选择能谱抽样、强制初级反应或 --compress。
输出目录需为新目录。

## 静态 CPU 并行前缀

仅在需要时开启，当前支持 `kokkos-proposal`，不支持与 EGS4 并行前缀组合。
串行 EGS4 前缀不变。以下命令在已安装的 CUDA/组合版 `bin` 目录执行；
HIP 可在对应 `bin` 目录使用相同设备列表和前缀参数：

~~~bash
# 16 个 CPU prefix 进程，随后四张卡共同推进一例；90% 是每卡预算。
./c8_air_shower -p 2212 -E 1e8 -N 1 -s 20261019 \
  --device 0,1,2,3 --prefix-workers 16 --gpu-memory-fraction 0.90 \
  --multigpu-timeout 86400 --ring 1 --antenna-file /dev/null -f parallel_prefix

# 两个 CPU prefix 进程，随后一张卡；同样使用协调器和 merged/ 输出。
./c8_air_shower -p 2212 -E 1e4 -N 1 -s 20261019 \
  --device 0 --prefix-workers 2 --gpu-memory-fraction 0.70 \
  --ring 1 --antenna-file /dev/null -f parallel_prefix_single_gpu
~~~

物理参数需按自己的实验统一；上面的能量/进程数仅是使用示例，不是普适最优参数。
保持普通串行/单卡行为时不传 `--prefix-workers`；多卡命令也可显式传 0。

1. 串行 bootstrap 正常推进初级粒子，捕获已产生且低于 CPU 拆分上限的次级。
   它不复制初级、不强制首次反应；捕获包括强子、核和轻子，不只收集 GPU 粒子。
2. 按既有静态分配方法把这些根分到 CPU 子进程。每个子进程使用原标量
   PROPOSAL/强子模型继续推进，生成可交给 GPU 的 EM/μ 前沿。
3. 等所有 CPU 子进程完成后，校验并汇总前沿，再静态分到所选 GPU。
   GPU 数与 CPU 进程数不必相同；每个 GPU worker 保留自己的 CPU 回退路径。
4. 结束后合并 bootstrap、所有 CPU prefix 和 GPU worker 的输出。
   profile/沉积与带符号射电场相加，首次相互作用来自 bootstrap，不重复计数。

没有运行中偷取任务、动态迁移或频繁重新分配。CPU 模型在独立进程中运行；
每个 worker 的 OpenMP/BLAS/Arrow 线程数限制为 1。CPU worker 隐藏 GPU 设备，
不预占 GPU 显存；请为同时运行的多个 shower 显式分配不冲突的 CPU/GPU 资源。
CPU worker 与 GPU worker 总数最多 255，以确保各自的 history 编号范围独立。
协调模式的显存预算默认 0.50，包括并行 prefix 后接单卡的情形；可显式设置
`--gpu-memory-fraction`，如上例的 0.70/0.90。普通直接单卡的 0.70 默认值不变。
预算依据后端初始化时的可用显存，并不保证进程恰好占整卡总容量的该比例。

### 两个能量上限与静态导入块

- `--multigpu-frontier-energy`：决定哪些 EM/μ 根粒子可导出到 GPU 前沿；
  自动值为 `primary_energy / (8 * number_of_selected_GPUs)`。
- `--prefix-split-energy`：bootstrap 捕获 CPU 分支的总能量上限；默认 0 使用
  `primary_energy / (96 * prefix_workers)`，显式正值覆盖。与 GPU 数量及
  `--multigpu-frontier-energy` 独立；`--prefix-workers 0` 时不使用此阈值。
  次级粒子降到该阈值后可被保存，随后按原有静态方法分配给 CPU 进程。
  **两者只影响任务交接，不是 cut 或薄化参数。**
- `C8_STATIC_FRONTIER_IMPORT_BATCH`：可选环境变量，控制 GPU worker 一次导入
  多少静态根，取值 1--65536；未设置时保留原后端最小批次对应的导入规则。
  例如在命令前加 `C8_STATIC_FRONTIER_IMPORT_BATCH=32768`，不需要重新编译。
  它不改变 `--gpu-min-batch`，也不会在本地驻留工作/CPU 回退未排空时插入新根。
- `--gpu-min-batch`：默认 **16384**，控制 GPU 派发及小批次检查，与静态导入块
  分开设置，不指定 CUDA/HIP 线程数或固定的 kernel 粒子数。

CPU 拆分的默认系数基于高能大气质子前缀的测时；其他能量、初级粒子和机器
可通过显式 `--prefix-split-energy` 调优。

大块减少交接次数，但不保证所有事件都更快。已经测试的高能质子配置中使用过
32768；低能事件的进程初始化开销可能超过收益，不应自动照搬高能进程数。
并行进程数、分支上限和导入块可能改变后续随机历史；比较物理时固定完整配置，
使用统计验收而不是要求逐事例相同。

### 前沿格式与可追溯记录

CPU 根使用 `C8_CPU_FRONTIER_V1`，保存原粒子 PID（核含 A/Z）、总能量及独立的
动能、权重、位置/方向/时间、history/parent/generation/step 和逻辑环境节点。
动能独立保存可避免从大总能量减静质量时损失精度。CPU 后代输出使用
`C8_STATIC_FRONTIER_V3`，额外记录输入 history 的编号上限；GPU 后代从更高且
互不重叠的范围开始。原串行 prefix 仍写 V2，不应把并行 V3 前沿交给旧 worker。
应用与 worker 要由同一版源码构建；环境指纹不匹配会报错，不重新猜测节点。

启用并行后保留 `bootstrap/`、`cpu_prefix_*/`、`worker_*/`、CPU/EM 前沿、
`CPU_PARTITION.json`、`CPU_FRONTIER_FEED.json`、`FRONTIER_JOIN.json` 和 GPU 分片审计。
`CONFIG.json` 的 `cpu_prefix_split_energy_effective_GeV` 和
`em_frontier_energy_effective_GeV` 分别记录实际 CPU 拆分上限和 EM 交接上限；
原 `cpu_prefix_split_energy_GeV` 保留用户请求值（0 表示自动）。
`TIMING.json` 增加 `cpu_prefix_workers` 和 `prefix_phase_wall_s`；后者包含
bootstrap、CPU 分片、CPU worker 推进与前沿汇总，不只是单个 worker 的物理耗时。
哈希、完整消费、编号冲突、输出网格和有限值检查失败时，不产生成功完成标记。

前缀、分片和合并实现在
[`AirShowerCoordinator.cpp`](../src/kokkos/scheduling/AirShowerCoordinator.cpp) 与
[`StaticFrontierIO.cpp`](../src/kokkos/scheduling/StaticFrontierIO.cpp)，
回归测试位于 [`tests/application_support/coordinator/`](../tests/application_support/coordinator/)。

## 多事例的 worker 复用

CUDA/HIP 的 `kokkos-proposal` 协调模式在 `-N > 1` 时默认保留 GPU worker
进程、已加载模型、表格和设备会话。首个 CPU 前缀结束后才启动 worker，
首个真实请求才初始化 GPU；当前不提前启动 GPU 与首个 CPU 前缀重叠。
CPU 前缀仍按事件运行，不在这个 GPU worker 池内。

每例重置粒子队列、计数、射电和输出状态，读取新的前沿；每卡仍静态接收自己
的分片。worker 内的原生 CPU 随机流连续使用，设备随机状态按事件种子重置，
因此后续事例不保证与每例冷启动逐值相同，应做统计比较。
`--persistent-workers=false` 恢复逐例新建 GPU worker；EGS4 仍走逐例新建路径。
普通单进程、串行前缀加单卡的 `-N > 1` 原本就保留模型和后端，不使用这个选项。

常驻进程的日志与请求记录在顶层 `worker_pool/worker_*/`，每例物理结果仍在
`seed_<seed>/worker_*/shower/` 和 `seed_<seed>/merged/`。每例 `TIMING.json`
记录该例墙钟及前缀耗时；其中 worker 请求时间不包含进程启动。
`worker_pool/TIMING.json` 记录常驻进程寿命，顶层 `TIMING.json` 记录整个批次。
这些范围有重叠，不能直接相加；首例初始化与后续稳定耗时应分别报告。

## 计算与输出

### CPU 轨迹的射电计算

在 `kokkos-proposal` 配合 `--radio-backend kokkos` 的 CUDA/HIP 路径中，
电子、正电子即使由 CPU 推进，其 CoREAS/ZHS 计算也交给 GPU。
CPU 只记录该步的起止位置、时间、方向、能量和权重；每条轨迹只记录一次，
由共用的 GPU 投影器计算两种射电模型，不再同时执行原生 CPU 射电计算。
这不改变粒子输运、反应采样、cut 或随机数流。

GPU worker 使用独立的 4096 条射电轨迹缓冲，满批提交，在输出波形前提交
最后不足一批的轨迹。此缓冲不由 `--gpu-min-batch` 控制；小粒子批次仍可
进行原来的 CPU 输运。新增缓冲计入显存预算，不逐轨迹分配显存。
CPU 轨迹缓冲与粒子输运队列分离；上传及射电投影使用该后端原有的执行流，
与设备轨迹的投影按序执行，共同累计而不重置波形。上传完成后才复用 CPU
缓冲，设备缓冲的复用同样服从执行流顺序。它们会共享 GPU 计算资源，
但不从两个独立流同时读写同一块轨迹暂存区。

串行及并行 CPU 前缀不创建 GPU 上下文，而是写入版本化的
`radio_tracks.bin`。协调器在前缀结束后把这些文件静态分配给 GPU worker，
并记录 `RADIO_TRACK_PARTITION.json`，校验条数、完整消费及 SHA256。
前缀输出本身不再累计射电场，其物理贡献包含在接收轨迹的 GPU worker
波形中，最终仍按原顺序合并，不能再次把这些轨迹投影到 CPU 波形。

`gpu_em/summary.yaml` 的 `host_tracks_enqueued`、`host_track_batches`
分别记录转交的轨迹数和提交批次数；原生射电模块的
`offloaded_segment_count` 记录其转交数，而不是 CPU 实际投影数。
普通 CPU 射电路径及 EGS4 路径保持不变。实验性 cooperative 模式的
OpenMP endpoint 仍保留其原有 Kokkos/OpenMP 射电计算。

以下说明针对默认串行前缀的多 GPU 路径；未开启并行前缀时，单卡仍使用普通应用的输出结构。
一个 CPU 前缀导出互不重叠的 EM/μ 根粒子，按未加权能量代理分配一次；
每张卡在独立进程内推进自己的子簇、CPU 回退和随机流。子进程重新执行同一个
c8_air_shower，协调进程不创建 Kokkos GPU 上下文。每个 worker 有独立工作目录，
结束时间独立记录。全部成功后按固定 worker 顺序合并。

每例保留 prefix/、worker_*/、前沿及分片哈希、PARTITION.json、
TIMING.json，最终数据在 merged/。-N 1 直接写入指定目录；
-N > 1 为每例创建 seed_<seed>/，顶层保留总计时和批次清单。

profile/沉积以及**带符号**的 CoREAS、ZHS 场在完全相同的网格上以补偿
float64 加法合并；地面粒子拼接，首次相互作用只取全局前缀记录。
GPU 分片和合并顺序固定；worker 复用的随机流行为见上文。
COMPLETE 只在正常退出、输出有限值/网格/配置、前沿消费和溢出检查通过后生成。
中断/超时只结束本次创建的子进程，保留已输出的数据。

COMPLETE.json 表示流程完成，不自动声明新的统计物理验收通过。
当前安装只需使用 `c8_air_shower`，无需旧的独立多卡应用或 Python 队列。

## 验证

协调器测试与主工程一起构建，不再使用旧的 `validation/accelerator/` 独立工程。
完成对应后端的安装构建后，从源码目录运行：

~~~bash
# 使用实际的构建目录；HIP 则改为 ../build/hip
ctest --test-dir ../build/cuda -R '(native_coordinator_|native_worker_session)' \
  --output-on-failure -j1
~~~

测试定义在 [`cmake/CoordinatorContracts.cmake`](../cmake/CoordinatorContracts.cmake)，
覆盖参数、worker 种子、前缀分配、输出合并及常驻请求生命周期。
可选的 `CORSIKA_TEST_COORDINATOR_PROCESS=ON` 会额外运行真实进程的失败／超时
检查，需要可查询的 NVIDIA 设备 0、1；它不是 HIP 测试的必选项。

这些接口和数值检查不能代替完整 shower 的统计验收。比较运行方式时应记录
源码版本、物理条件、设备列表、显存预算、前缀参数、派发门槛、静态导入块和
worker 复用设置，并区分冷启动、稳定事件墙钟、GPU 活动时间及占用卡时。
