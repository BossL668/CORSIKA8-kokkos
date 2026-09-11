# beta5：独立 CUDA / OpenMP 驱动（实验验收中）

本页保留批次级独立驱动的历史结果。后续移除成对批次屏障的重构见
[独立子级联队列](beta5_independent_subshower_queues_CN.md)，不要把本页
100 PeV 耗时或 tail-wait 字段当作该新路径的测试结果。

## 修改边界

仅显式 `--kokkos-execution cuda-openmp` 的空气协同后端使用新驱动。
单 CUDA、单 OpenMP、标量 PROPOSAL 的选择分支、物理 kernel、表、
cut、thinning、磁偏转常数，以及山体 session 均不修改。
隔离构建为 `build/cooperative-independent-20260911`；不覆盖生产安装。

## 执行逻辑

```text
协调器（也是 OpenMP 初始化线程）
  ├─ 预留不重叠的 history 区间、分配批次
  ├─ 独立 CUDA driver：提交 → 驻留推进 → 本端射电 → 返回结果
  └─ OpenMP：驻留推进 → 本端射电 → 返回结果
                  ↓ join 安全边界
        检查身份/计数 → 合并结果 → CPU fallback → 下一批
                  ↓ 全部队列清空
        检查并合并原始定点 profile / CoREAS / ZHS
```

OpenMP 不再只在 GPU 等待回调中执行；CUDA 的后续提交也不再等
OpenMP 批次返回。两个执行端分别推进原来的光子/轻子驻留实现。
每端至多一个在途调用；不使用 detached thread，不增长无限任务队列。
任何异常先等待另一端安全退出，再销毁输入、View 和 runtime；
错误不静默降级为单端继续。

这是**批次级独立推进**，不是整个 shower 完全无同步：
现有同步 router 仍在批次结果汇总处 join，fallback 仍由唯一协调器执行。
因此不能保证双方一直满载，也不能据此宣称已经获得净加速。

## 负载分配和资源

按光子/轻子分别维护两端吞吐移动平均，考虑已驻留的积压，
使预测完成时间接近，不强制各分一半。
OpenMP 保留独立有界 arena（当前 2048 输入容量），GPU 沿用原显存预算；
该容量是内存保护，并非 CPU 只能在 GPU 等待时工作。
每个事件清空负载估计和结果缓存，静态物理表继续复用。

当前调度修订还包含：

- CPU 每次调用最多推进 8 个 wavefront，随后保留未完成粒子供下一轮；
  这是工作量检查点，不是物理时间、能量或距离 cut，不丢弃粒子。
- 尚无吞吐估计时，不把不足一个有效 GPU 波前的初始输入全部分给 CPU。
- 支持两个方向的安全边界迁移，但不迁移在途粒子、不强行平均分配。
  若当前可处理前缀被容量截断，不额外抽取未预留 history 的粒子。
- CPU 容量取实际分配成功的 arena 容量，而不是假定始终有 2048 个槽位。
  队列预算不包括只读物理表、射电和返回结果，不能当作进程 RSS 上限。

CPU 先结束仍会等待当前 GPU 批次，反之亦然；本轮没有将上层同步 router
改造成无限连续的双端任务流。后续如需进一步隐藏这段等待，需要独立设计
跨批次 history 预留及 fallback 提交，不能通过工作线程直接改共享栈实现。

## 源码入口

| 文件 | 本轮职责 |
|---|---|
| `corsika/accelerator/em/detail/IndependentEndpointDriver.hpp` | 单一持久驱动线程、有界提交、异常传播、析构排空 |
| `src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp` | 两端批次分配、独立调用、安全汇总、容量及身份检查 |
| `corsika/accelerator/em/common/Types.hpp` | 仅增加协同诊断字段 |
| `applications/detail/air_shower_kokkos/KokkosShowerReport.hpp` | 仅在协同输出中写入新增字段 |

单端 `KokkosBackendInstance.inl`、物理 kernel 和 runtime 初始化策略未在
本轮修改。驱动实现只编入 `CUDA_OPENMP` 构建。编译通过不等同于性能验收。

Kokkos 4.7.03 运行时在主线程唯一初始化/释放；借用实例也在该线程创建。
CUDA 驱动线程只调用自己独占的 CUDA 实例，并显式选择设备；
OpenMP 在原初始化线程执行。共享 CPU 栈、writer、PROPOSAL fallback 不进入工作线程。
这不是对任意 Kokkos 后端/版本的线程安全承诺：
[官方线程安全说明](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Machine-Model.html#thread-safety)
指出其安全性取决于实现，故本修改仅用于 CUDA_OPENMP 组合构建，必须实际验收。

## 诊断字段

仅协同报告增加 `independent_drivers`、`independent_joint_calls`、
`cuda_driver_wall_ms`、`joint_wall_ms`、
`endpoint_window_overlap_ms`、
`cuda_finished_before_host_ms`、`host_finished_before_cuda_ms`。
窗口交集包括主机提交、拷贝、等待，**不是实际 kernel 重叠时间**；
真实重叠必须用 CUPTI 时间线确认。
旧 2 ms 回调统计保留，但新驱动不再采用该时间片策略，相关旧计数为零。

## 初版及第二版的验收记录

- 主机驱动单测：已通过有界提交、线程身份、异常传播、32 次复用和析构排空。
- 真实 PROPOSAL、双端 EM＋射电：20 线程，两次事件通过，终止身份无重复，
  profile 无非法记录/定点溢出；工作区两次相同（321988542 bytes）。
- 热缓存 CUPTI 时间线：实测 kernel / OpenMP 区间重叠 1.290183 ms，
  丢失记录为零。这只是一个批次的真实重叠证据，不是 shower 加速比。
- 单端回归：标量 PROPOSAL、单 CUDA、单 OpenMP，分别 N=2；
  每个模式的 11 个物理数组文件全部一致，metadata 无非允许差异；
  两个加速单端的 decision trace 字节完全相同。
- 同一 argv[0] 下 `--help` 完全一致；空气和山体应用均编译通过。
- 新双端 N=32：32/32 正常关闭，queue/profile/radio 溢出均为零，
  事件间工作区恒为 56815550 bytes，峰值 RSS 795.43 MiB，
  系统可用内存始终高于 4 GiB。不是无限时长无泄漏的证明。
- 高能耗时和大样本物理统计：待完成，不作为生产推荐。

结果位于 D 盘 `CorsikaData/corsika_validation_results/beta5_independent_drivers_20260911`：
`BOUNDED_ACCEPTANCE.json`、`air-regression-v2/`、`backend20-warm/`、
`overlap-figure/actual_kernel_overlap.png`。

首次冷启动时间线未通过（CPU 批次先完成，GPU 尚在首次工作区准备）；
该失败记录保留在 `backend20/`。已在并行启动前预分配两端工作区，
热缓存验收单独记录，不把冷启动失败删除或解释成通过。

旧双端 100 PeV 任务使用归档二进制，已自行结束，本修改未替换它。

## 100 TeV 调度诊断（非最终性能验收）

固定质子 `1e5 GeV`、`theta=47`、`phi=180`、seed `2026110001`、
`emthin=1e-6`、原版默认 max-weight、同一天线、全 CoREAS/ZHS，
CUDA 显存预算 70%，协同 OpenMP 为 20 线程，强子 worker 为 1。
本参数下自动 max-weight 为 0.05，记录显示 thinning 没有从单位权重激活；
所有比较端均保持该默认值，不通过调薄化取得加速。

| 实现 | shower 时间 [s] | 进程总时间 [s] |
|---|---:|---:|
| 修改前 GPU 等待回调式协同 | 85.614 | 95.275 |
| 第一版独立驱动（未限制 CPU 长批次） | 175.694 | 185.647 |
| 第二版独立驱动（CPU 最多 64 波前） | 95.568 | 106.034 |
| 第三版独立驱动（CPU 最多 8 波前） | 87.417 | 97.340 |
| 单 CUDA 对照 | 92.158 | 101.956 |

初版变慢的证据：小输入先分给 CPU，加上长批次，GPU 完成后等待 CPU
累计约 39.3 s。第二版改进分配后该等待降到 9.59 s，但仍没有净加速。
第三版进一步把 CPU 检查点缩到 8 个波前，GPU 等 CPU 累计降到 0.136 s。
单例相对单 CUDA 的 shower 时间缩短约 5.1%，但仍比旧双端慢约 2.1%。
这些都是单个种子、动态分配后不同 shower 树的诊断，不是五种子热缓存
中位数验收；不能用它们宣布稳定加速。失败版本和输出均保留。

第三版开启 20 个 OpenMP 线程，但进程总 CPU 时间折算平均只有约 **1.61 核**，
不能声称把 20 核吃满。CPU 调用窗口合计 2.130 s，与 GPU 调用窗口交集
约 1.980 s；CPU 先结束后等待 GPU 累计约 **81.09 s**。
这说明独立提交已工作，但大多数工作仍在 GPU，批次边界仍限制 CPU 连续工作。
GPU 设备级显存采样峰值约 5363 MiB，进程树 RSS 峰值约 1202 MiB。

第三版重新通过单端 N=2 物理数组/decision trace/CLI 帮助完全一致、
双端 N=32 正常关闭及固定工作区测试（峰值 RSS 796.44 MiB）；
真实 CUPTI 重叠为 1.512456 ms，未丢失记录。
详见 `BOUNDED_ACCEPTANCE_V3.json`、`PILOT_TIMING_COMPARISON.json`、
`air-regression-independent-v3/`、`overlap-independent-v3/`。

所有高能 pilot 的加速器能量账本都标记 `complete_coverage=false`，因为
还有标量/强子侧能量未纳入该局部账本；`accepted=false` 也保留原值。
本轮不把正常关闭或局部账本残差当作完整能量守恒验收，未宣称已通过 500 例统计。

## 使用与发布边界

### 最终小样本回归及 100 PeV 测试

异常安全加固后再次完成单端 N=2 的数组/decision trace 对照、双端 N=32
生命周期检查，以及空气/山体构建。`BOUNDED_ACCEPTANCE_FINAL.json` 为本轮
最终小样本记录：32 次工作区恒为 56815550 bytes，峰值 RSS 795.94 MiB。
真实 PROPOSAL 双端探针的热缓存 CUPTI 重叠为 2.194647 ms，丢失记录为零。
这不替代完整 shower 的性能和统计验收。

最终 100 TeV 单例记录为 shower 87.316 s、进程 96.901 s；
与单 CUDA 101.956 s 的差异仍只是单种子诊断。
详见 `FINAL_PILOT_TIMING_COMPARISON.json` 和 `overlap-final/`。

按照用户要求，2026-09-11 启动更高能量的隔离测试：

- 质子 100 PeV（`1e8 GeV`），theta 47°、phi 180°，seed 2026110001。
- emthin `1e-6`，max-weight 保持默认，同一天线，全 CoREAS/ZHS。
- 独立双端，OpenMP 20 线程、强子 worker 1、CUDA 显存预算 70%。
- 不覆盖生产程序；使用冻结二进制，SHA-256 为
  `78625546090037e5af3310adcc55b19a296b2e765d26c1c0864cb684f7a037b8`。
- systemd 用户服务独立于终端；单例限时 2 h，进程树 RSS 上限 4 GiB，
  系统可用内存低于 4 GiB 时停止测试，不自动更换种子重试。
- 每秒记录进程及逐线程 CPU 时间、RSS、可用内存、显存和设备利用率。
  CUDA 提交线程标记为 `c8-cuda-driver`；线程数不等于实际核利用率，
  CPU 时间也包含驱动/忙等，不能直接当作纯物理计算时间。

结果目录：
`CorsikaData/corsika_validation_results/pilot_beta5_independent20_proton100PeV_theta47_phi180_20260911_v1`。
完成后独立报告服务自动生成 `PERFORMANCE_REPORT_CN.md`、
`PERFORMANCE_RESULT.json`、`resource_timeline.png`；失败则保留失败报告，
不输出不完整事件的加速比。本文记录的是启动状态，是否完成以该目录结果为准。

旧同参数回调式双端为 4095.827 s，历史单 CUDA 为 4369.329 s。
历史二进制不同、且当前动态调度可改变 shower 树，因此最终比较也只是
单事件描述性比值，不能替代同版本交替多种子测试。
高能是否缓解 CPU 提前完成后的批次等待，须由完整记录判断。

### 隔离构建调用

仅隔离构建中的空气应用启用此次实验：

```bash
# 在源码目录运行；用小事例检查连接，不替代高能性能测试。
../build/cooperative-independent-20260911/applications/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos \
  --kokkos-execution cuda-openmp --kokkos-num-threads 20 \
  -p 22 -E 1 -N 2 -s 26091021 -f "$HOME/CorsikaData/independent_probe" \
  --antenna-file examples/beta5/antennas_minimal_nwu.txt
```

使用原有 `corsika_venv`、`FLUPRO` 和 `CORSIKA_DATA` 设置。
保留旧双端及 v2/v3 的源码快照、旧双端和 v1/v2/v3 的二进制；
v1 仅保留二进制快照。本轮未推送 GitHub、未安装到生产前缀，
除上述高能单例外未启动新的大样本 campaign。
当前仍是实验实现：保留单端推荐，不以一个种子的差异修改生产默认值。
