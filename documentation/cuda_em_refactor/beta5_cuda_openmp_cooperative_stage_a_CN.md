# beta5 空气 CUDA＋OpenMP 协同：阶段 A 实施记录

日期：2026-09-10。状态：**底层共享生命周期与重叠探针已实现；完整协同 shower 尚未接通，不是生产验收通过。**

后续进度：[阶段 B1：真实光子波前提交、驻留续推和重叠检查](beta5_cuda_openmp_cooperative_photon_front_CN.md)。本页保留阶段 A 当时的测试范围。

## 1. 当前能做什么、不能做什么

已经可以在同一进程、同一协调线程上，先异步提交 CUDA kernel，再执行 OpenMP kernel。两端借用同一个 Kokkos 生命周期。当前验证的是执行原语，不是 EM cascade。

正式空气入口仍保持 `--kokkos-execution cuda|openmp` 二选一；**尚不接受 `cuda-openmp` 作为协同运行选项**。没有将尚未接通的调度器伪装为可用后端。山体程序、物理公式、PROPOSAL 表、射电算法和生产默认值均未修改。

官方例子要求先提交设备工作，并指出主机结果归约造成的同步可能阻止重叠。本轮依照这个顺序验证，不通过两个普通主机线程同时调用 Kokkos。[Kokkos：Overlapping Host and Device work](https://kokkos.org/kokkos-core-wiki/usecases/OverlappingHostAndDeviceWork.html)

## 2. 本轮新增结构

| 文件 | 已实现的职责 | 尚不代表 |
|---|---|---|
| `KokkosRuntime.hpp/.cpp` | 显式 cooperative owner、不可伪造的只读 lifetime lease、同线程/设备/线程数检查、最后持有者释放 runtime | 应用已经能同时执行两个完整 shower 子树 |
| `CooperativeScheduling.hpp` | 一在途＋一暂存的状态检查、恰好一次提交、独占 history 范围分配、EWMA 批量建议 | 已接上真实粒子分配和迁移 |
| `CudaCompletionTicket.hpp` | 对单 CUDA stream 记录/查询完成事件 | 所有 resident loop 已经异步化 |
| `CudaWavefrontControl.hpp` | 预分配 pinned POD、异步 D2H、显式 poll/take、异常析构先等待本次传输 | 已下载粒子队列或改动输运 kernel |
| `CheckedFixedAccumulatorMerge.hpp` | 同布局/尺度整数向量的合并；先检查所有元素溢出，再写结果 | 已连接 profile/radio 原始整数下载接口 |
| `CooperativeScheduleJournal.hpp` | 流式调度日志、配置身份检查、顺序回放、首次字段分叉、截断和长度门禁 | 逐过程物理 decision-tape replay 已通过 |

上述新头文件位于 `corsika/accelerator/em/detail/` 或 `corsika/accelerator/em/kokkos/`。新增 runtime 配置仅为内部接口，没有改变现有 CLI。

普通 CUDA 仍使用一个 host 线程；只有显式 cooperative owner 可初始化多线程 OpenMP。默认合作线程数考虑进程 CPU affinity，取可用逻辑 CPU 数减二、最多八个、最少一个。独立 CUDA/OpenMP 构建拒绝 cooperative owner，组合构建也拒绝跨协调线程借用。

## 3. 实际重叠证据

不能仅凭 GPU utilization 非零，或 CUDA 完成事件尚未就绪，就宣布实际 kernel 重叠。因此加入仅测试目标使用的 CUPTI 12.x activity 记录：CUDA 实际 kernel 起止时间与主机 OpenMP 区间使用同一个 CUPTI 时间域。只启用 `CONCURRENT_KERNEL`，不使用会序列化 kernel 的 tracing 类型。[NVIDIA CUPTI Activity API](https://docs.nvidia.com/cupti/12.8/api/group__CUPTI__ACTIVITY__API.html)

一次实测：

- CUDA 测试 kernel：约 **5.470 ms**。
- OpenMP 4 线程工作区间：约 **4.291 ms**。
- 实际区间交集：**4.291 ms**；丢失 activity records：**0**。
- 两端整数算术结果均与独立主机计算完全一致。

![实际 kernel 时间线；不是 shower benchmark](../../../build/overlap-probe-20260910/figures/actual_kernel_overlap.png)

原始绝对时间戳在 `build/overlap-probe-20260910/figures/timeline.json`。图只是去掉共同时间原点，没有平移两端使其人为对齐。

这里仅证明机制可行。不能由 4.291 ms 推算 shower 加速比，也不能代替 EM/radio 的物理验收。测试时生产 GPU 仍有任务，未进行独占性能比较。

## 4. 已完成的检查

| 检查 | 本轮结果 |
|---|---|
| CUDA_OPENMP runtime/control 独立 Release 探针 | 6 项 CTest 通过，含 CUPTI 实际重叠 |
| 独立 CUDA runtime 探针 | 3 项 CTest 通过 |
| 独立 OpenMP runtime 探针 | 3 项 CTest 通过；动态依赖中没有 CUDA runtime |
| 状态/定点合并/调度日志 | 主机单元测试及 UBSan 通过 |
| 同一 pinned 缓冲复用 | 连续 32 次控制传输地址不变、数据精确；过早领取及重复领取被拒绝 |
| 出错/结束状态 | 重复提交、错误提交 ID、history 溢出、日志截断/配置不符被拒绝；定点溢出时不部分写入 |
| 源码保护 | 与启动快照相比，已有源码只修改 runtime 两个文件和测试 CMake；山体及物理 kernel 未修改 |
| 安装与生产保护 | 既有安装二进制 SHA-256 不变；生产 service 未停止、未替换 |

这些是基础单元测试，不是完整主项目构建矩阵或山体回归。32 次控制传输也不等于 `N=32` shower 的内存验收。

独立 OpenMP 探针需要当前构建生成的粒子属性头以及 spdlog/Eigen/Boost 的公开依赖。初次隔离构建发现这些依赖遗漏后，已补入探针 CMake；没有改动 Conan 缓存或物理头来规避检查。

### 冻结的物理基线

`build/overlap-baseline-20260910/` 保存：

- 启动时源码 diff、源文件/安装二进制 SHA-256、三种安装程序的 `--help`。
- 三份原安装二进制副本（`binaries/`）。
- 1 GeV 光子、种子 `26091021`、垂直、`emthin=1e-6`、三个固定 NWU 天线的三条路径，各 `N=2`：标量 PROPOSAL、组合程序选择 CUDA、组合程序选择 OpenMP。
- 原 CPU decision tape，两种 Kokkos 执行端的 process trace、profile/粒子/射电 Parquet 和 metadata。
- `boundary_and_output_audit.json`：保护边界及输出文件哈希。

CUDA/OpenMP 第二个事件均报告 backend reuse。上述运行用的是**冻结的旧安装程序**，用于后续逐字段对照，不能称为本轮重构后的完整物理回归。当前工作树包含此前尚未统一发布的修改；后续比较必须明确源文件基线，不能把它们引起的差别算到协同重构头上。

测试由 `validation/accelerator/run_overlap_guarded.py` 监控：启动前检查系统至少可用 4 GiB；运行时每 0.1 s 检查可用内存、测试进程树 RSS 和超时。触发门限只终止测试新建的进程组，不触碰生产/IDE。三个小事件基线采样到的峰值进程树 RSS 分别约 609 MiB、608 MiB、232 MiB；这些数字不是严格的逐瞬时内存上界。

## 5. 复现小探针

从 beta5 源码目录执行，复用已有组合构建依赖，不重装依赖、不改生产 build/install：

```bash
conda activate corsika_venv
cmake -S validation/accelerator/overlap_probe -B ../build/overlap-probe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/../build/cuda-openmp/deps/conan_toolchain.cmake" \
  -DKokkos_DIR="$PWD/../build/cuda-openmp/deps" \
  -DC8_PROBE_BACKEND=CUDA_OPENMP -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DC8_PROBE_CUPTI=ON
cmake --build ../build/overlap-probe -j1
python validation/accelerator/run_overlap_guarded.py \
  --output ../build/overlap-probe/check-1 -- \
  ctest --test-dir ../build/overlap-probe -V
```

架构 89 对应本机，其他设备应与其 Kokkos 包一致。CUPTI 为可选的诊断依赖，未加入生产程序；当前诊断结构锁定 CUDA 12.x。无 CUPTI 时可关闭该选项，但就没有实际 activity 时间线门禁，不能当作同等验收。

OpenMP-only 使用 `-DC8_PROBE_BACKEND=OPENMP`、OpenMP 的 Conan toolchain/Kokkos_DIR，以及 `-DC8_GENERATED_INCLUDE="$PWD/../build/openmp"`，不启用 CUDA 语言。

## 6. 下一阶段必做项（未完成）

1. 将 `KokkosResidentPhotonCascade.hpp`、`KokkosResidentLeptonCascade.hpp` 的同步 while 循环拆成持久 frame。已有 front control POD 可供异步控制对象使用；需保留原 kernels、scan 顺序、history/随机计数和全部异常路径。
2. 在 `KokkosBackendInstance.inl` / backend instance 接口中公开 submit/advance/poll/take，保留原同步包装。先跑真实无射电 EM 双端重叠与单端输出回归。
3. 协调器独占 Stack/fallback，接入动态分配、有限迁移、背压及日志。对 mixed PID、空队列、半完成批次、身份冲突做测试；终止条件包含所有在途/驻留/fallback 输出。
4. 补齐两端 profile/radio 的原始整数导出及完整 counters 合并。尤其 ZHS 的存储量是势的时间积分表示，转换尺度还依赖各天线采样率，不能把两个已转成浮点的波形直接相加。
5. 接入应用 CLI、双端 EM/radio 和 metadata；建立显存/主存独立预算与事件后清理。只有实际完整模式接通后才接受 `cuda-openmp`。
6. 同源重构前后完整构建/回归、山体回归、`N=1/2/32`、decision tape、500 例统计和 GPU 空闲后的交替五种子性能测试。

目前**没有**满足协同全射电快 5%、单端退化不超过 3%、完整 shower 内存稳定等发布门禁。生产推荐及默认后端保持原状。
