# 独立跨介质电磁输运与常驻队列

日期：2026-09-11。模块名 `CORSIKA8::InterfaceEm`，实现位于
`src/transport/` 和 `corsika/modules/transport/`。
本模块复用 beta5 的光子/电子单步物理与通用队列基础设施；空气应用、空气常驻
后端和共享电磁物理源码保持原样。

执行模型现已增加大气后端同类的多 wavefront 常驻调用。
大气标准、具体差异及本轮验收见 [多轮常驻对齐记录](interface_air_standard_alignment_CN.md)。
原逐轮版本的 CUDA 活动记录保留在 [PSR GPU 常驻复测](interface_cuda_residency_profile_CN.md)。

## 范围与接口

`InterfaceEmSession` 是独立实现，原 `TerrainEmSession` 名称保留为兼容别名。
输入是一张经验证的定向闭合三角形网格、两侧逻辑区域/材料表映射及不可变材料 bank。
网格可以表示山体或其他闭合形状，材料由实际 PROPOSAL calculator 导出。
当前几何接口是**一个闭合界面的两侧**，没有任意多个相交介质体的导航器。

每个 bank 分别提供密度模型和磁场向量，支持均匀密度或已有的分层密度快照。
磁场在各区域内为独立的均匀三维向量；两侧可取不同方向、不同强度或零场。
同一材料的两个 bank 也可以只在磁场上不同。
空间连续变化或随时间变化的磁场不属于当前接口。

设备输运范围为 γ/e−/e+。中微子、强子、μ/τ 和指定稀有过程通过调用方的
CPU 模块处理。界面射电尚未实现。

```cpp
#include <corsika/modules/transport/InterfaceEmSession.hpp>
namespace api = corsika::interfaces;

api::EmConfig config;
config.threads = 256;                    // 独立 OpenMP 构建
config.batch_size = 64;
config.resident_capacity = 65536;
config.resident_record_capacity = 4096;  // 设备诊断账本，至少能容纳一个 batch
config.maximum_device_bytes = 128u * 1024u * 1024u;
config.interface = {17, 93, 0, 1};       // 区域号独立于 bank 下标
api::InterfaceEmSession session(mesh, banks, config);

session.submit(cpu_particles);          // 只提交 CPU 新生成/回退后再路由的粒子
while (session.pendingParticles()) {
  auto result = session.runResidentCascade([&](std::size_t slots) {
    return stack.reserveTransportHistoryIds(slots);
  });
  // 批量输出 result.records；处理其中指定的 CPU fallback。
  // 存活粒子与电磁次级已经保存在执行空间内，不要再次 submit。
}
```

会话独占一个 Kokkos runtime，销毁顺序保证所有 View 先于 runtime 释放。
一个会话可以排空后再次接收粒子，复用已有几何、材料表和队列分配。
`advance(input, first)` 保留为无状态单步参考；它不修改常驻队列。
`resident_capacity=0` 只启用这一参考接口。
`advanceResident(first)` 保留逐轮常驻参考；山体默认走 `runResidentCascade()`。
多轮调用的 history 回调只允许预留编号，不允许重入会话或向其注入粒子。

## 常驻推进的职责

```text
CPU 新粒子 → 有界 staging → 常驻 FIFO
                              ↓
                 跨介质/磁场电磁单步 kernel
                              ↓
               在设备端恢复逻辑区域、计数、稳定 scan
                              ↓
             存活粒子/次级 scatter → 常驻 FIFO 尾部

             16 字节控制记录 → 后端常驻循环
             设备诊断账本 → 检查点批量输出、指定 fallback → 主机
```

后端在一次调用内连续推进多个 wavefront，直到队列排空、出现指定 CPU 回退、
达到标量交错条件，或命中输出容量/轮数检查点。默认最多 1024 个 wavefront。
每轮在设备上验证结果、计数及扫描，只读取 16 字节控制记录并同步。
完整步记录先写入设备账本，调用返回时一次性下载有效范围。
主机仍负责 kernel 调度、精确的 history 预留、输出和 CPU 回退；粒子前沿无需反复上传。
这与大气常驻后端同属主机协调的多 wavefront 模型，大气后端本身也有逐轮控制同步。

FIFO 复用 `KokkosWavefrontQueue.hpp` 中的 `KokkosPendingParticleQueue`。
单步参考复用 `CountScanFunctor`；多轮路径把扫描总数写入设备控制 POD。
队列可混合 γ/e±，存活项和次级按原主机队列的稳定顺序追加。
历史编号按实际输入数预留，每项最多三个次级槽，保持原随机键语义。
到达材料界面只切换逻辑区域；大气内部切层不会覆盖该区域。

容量不足在消费输入前失败，不丢弃粒子，也不静默回退到无界主机队列。
这一保证针对当前 wavefront；多轮调用出错不会回滚已经完成的前序 wavefront，
调用方应将该 shower 标为失败，不能忽略异常并宣称完整输出。
分配预算包括队列压紧时旧/新分配短暂共存的峰值。
`deviceBytes()` 是当前分配账本，`projectedPeakDeviceBytes()` 是含压紧的峰值估算；
二者不包含 CUDA context，OpenMP 对应的是主机内存。

## 山体入口

`c8_terrain_cascade --em-backend kokkos` 默认使用常驻队列。
通过 `--em-scheduler batched` 选择保留的主机队列参考；
`--resident-capacity` 设置常驻容量，默认 65536。
独立 OpenMP 与独立 CUDA 使用各自构建的二进制。
若现有 Kokkos 包同时含多个后端，可用构建选项
`-DC8_INTERFACE_EXECUTION_SPACE=OPENMP` 或 `CUDA` 固定本模块的执行空间；
默认 `DEFAULT` 沿用包的默认执行空间。每份模块只实例化一个空间，不进行协同调度。
含 CUDA 的 Kokkos 包仍会初始化其 CUDA runtime，即使本模块固定在 OpenMP 执行。
山体应用的现有磁场场景设置保持原样，通用模块的独立材料/磁场由 bank 接口提供。

输出新增 `em_scheduler`、`cpu_uploaded_particles`、`device_enqueued_particles`、
`peak_resident_particles`、`peak_host_staging_particles` 与峰值内存估算。
多轮路径另记录 `resident_cascade_calls`、`maximum_call_wavefronts`、
`control_downloads`、`record_downloads`、`downloaded_records`、`peak_buffered_records`。
这些计数用于验证粒子常驻和容量；加速比需要单独测量。

## 验证

- `testKokkosInterfaceQueue`：混合次级、零权重、指定回退、200 次队列循环、
  入队/次级容量拒绝及拒绝后的输入保持。
- `testInterfaceTransport`：实际 PROPOSAL 水/岩石 bank，任意区域编号、相同材料、
  四面体、相反磁场、交换 bank，以及相同材料不同磁场。
  同一后端的常驻队列与主机 FIFO 按完全相同的 history 预留逐记录比较，
  也检查中途 CPU 注入、内存不增长、出入界面和次级继续推进。
  多轮回放使用 96 条记录的设备账本，强制容量检查点；与独立主机 FIFO 逐记录、逐位比较。
  可用 `C8_INTERFACE_TEST_THREADS=256` 指定 OpenMP 线程数。
- `validation/terrain/run_resident_interface_acceptance.py`：单步参考与常驻入口的
  真实 DEM 完整 CSV 哈希、材料表哈希、诊断与指定回退计数比较。
  可额外通过 `--baseline` 加入冻结旧二进制对照。

## 首次逐轮版本 PSR 验收：2026-09-11（历史记录）

服务器 `ssh psrpku2025_PKU`，NVIDIA T400 4GB。构建和测试均固定在 CPU 0–255，
拓扑核对为 256 个物理核；OpenMP 实际执行并发数为 256。
收到迁移测试的要求后，本地未再编译或运行测试，本地中止项不计为通过。

复用了 PSR 已有的 Kokkos 4.7.3 CUDA+OpenMP 包和 CUDA 12.6 工具链，
分别构建 `C8_INTERFACE_EXECUTION_SPACE=OPENMP` 与 `CUDA` 的独立模块。
两份山体应用、接口测试、队列测试和原阶段审计库均编译通过。
原 PSR 源码目录的 646 个受核对文件保持原样，所有测试使用独立源码/构建目录。

| 验收 | 单 OpenMP | 单 CUDA |
|---|---:|---:|
| 队列及六种材料/几何/磁场配置 | 7/7 通过 | 7/7 通过 |
| 真实 DEM，4 个事例 × 两种队列 | 8/8 通过 | 8/8 通过 |
| 同后端 batched/resident 完整 CSV、表、诊断、回退计数 | 全部一致 | 全部一致 |
| 完成、零待处理粒子、无材料错配、无输出截断 | 全部通过 | 全部通过 |

以下常驻计数在两种后端上相同。`CPU 上传` 包括初级及 CPU 过程新生成/重新路由的
电磁粒子，设备产生的存活粒子和次级不会因此重复上传。

| 事例 | 电磁推进步数 | CPU 上传粒子数 | 指定 CPU 回退 | 常驻粒子峰值 |
|---|---:|---:|---:|---:|
| 1 GeV 光子向上，零场 | 1,489 | 1 | 0 | 52 |
| 1 GeV 光子向下，场景 IGRF | 2,109 | 1 | 0 | 53 |
| 1 GeV 岩内电子，场景 IGRF | 1,639 | 1 | 0 | 50 |
| 10 TeV νe，强制 CC，thinning=0.1 | 194,841 | 22,535 | 27 | 194 |

常驻容量为 65,536，batch 为 64，主机 staging 峰值不超过 64。
模块当前分配为 111,186,480 字节，含压紧的预计峰值为 118,526,512 字节，
均通过 128 MiB 预算。进程采样 RSS 全部低于 857 MiB；CUDA 测试的最大采样显存
增量为 233 MiB，包含模块账本以外的 runtime 分配。
计时包含表加载、强子库初始化和完整诊断输出，本次结果用于正确性与常驻验证，
没有据此声称加速比。

上述逐位/哈希一致性针对同一个后端内的两种队列。
另对原冻结源码做了单步函数文本核对：类型/命名空间重命名后函数体保持一致；
原主机上的逻辑区域恢复搬入设备 wrapper，以便设备直接保留后继。
本地 859 个受核对源码文件中，空气应用、空气后端与共享电磁源码均未改变。

PSR 根目录：

```text
/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/
  source/                       独立源码
  build-openmp/applications/c8_terrain_cascade
  build-cuda/applications/c8_terrain_cascade
  build_psr.sh                  本次构建命令（--parallel 256）
  test_psr.sh                   本次测试命令及运行环境
  acceptance-openmp/            完整输出、哈希和资源记录
  acceptance-cuda/              完整输出、哈希和资源记录
  acceptance_summary.json       汇总
  psr-results.tar.gz            日志、配置和小型结果归档
```

日志与小型结果已取回本地项目的
`build/interface-resident-validation-20260911/psr-results/`。
完整轨迹 CSV 保存在 PSR，归档记录其 SHA-256。
再次运行验收脚本时需指定新的 `--root`，避免覆盖已有记录。
首次 PSR OpenMP 启动因运行库搜索路径缺少 `libcudart.so.12` 失败；修正
`LD_LIBRARY_PATH` 后上述测试全部通过，初次失败日志亦保留。
