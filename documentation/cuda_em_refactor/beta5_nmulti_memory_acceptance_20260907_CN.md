# beta5 同进程 N>1 内存补测（2026-09-07）

## 结论

本轮已完成单进程 `N=16/64/256` 测试。CUDA 和 OpenMP 的每个测试都由一个 PID 连续执行多个 shower，不是把多个 `N=1` 进程拼接成结果。

**没有发现每个 shower 重复分配 PROPOSAL 表、或输运工作区持续无界扩容的证据；但主机 RSS 确实随 N 增长，尚不能宣称“长时间运行内存恒定”或“整个程序绝无泄漏”。** 目前有代码及独立实验支持的累积来源是输出缓冲和逐事件报告。它们仍可能在大 N、高能量或更大射电窗口下造成内存压力，不能因为不是典型泄漏就忽略。

本轮没有修改 EM、射电、随机流或生产二进制，没有覆盖已有 install。仅增加诊断工具、内存保护测试和说明。GitHub 初始推送已完成：私有仓库 `BossL668/corsika8-gpu-hybrid`，分支 `kokkos-beta5`，提交 `d5eea700a2a998827a647a39ce0216d1b54e2a70`；本轮新增补测文件暂留本地。

## 条件与结果

使用 `corsika_venv`，1 GeV 光子、vertical、seed `26090711`、emthin `1e-6`、自动 max-weight、IGRF14/2027、81 个测试天线、完整 CoREAS/ZHS。1 GeV 用于快速检查生命周期，不代表高能生产性能；该能量下自动权重限制不会形成有效 thinning。

组合程序为 `install/cuda-openmp/bin/c8_air_shower`，SHA-256：

```text
382d5b9c4b6e18b7108e5d45dbd2bef5a4f32a3b7b04798f161501e3dbdd0053
```

它是推送前已经编译和验证的程序；提交没有重新编译，二进制内嵌的 revision 仍是当时的 dirty revision，不能把新 Git 提交号冒充其内嵌版本。

固定容量组使用 min-batch=16、resident-batch-limit=4096、memory-fraction=0.15。自动容量组不传这三项，使用默认 min-batch=4096、自动容量和 0.70 显存预算。OpenMP 使用4线程；CUDA 调度使用1线程。每组顺序运行，不在两个模拟之间争用显卡。

| 程序/模式 | 同进程 N | 第8例结束 RSS / MiB | 最后一例结束 RSS / MiB | 进程峰值 RSS / MiB |
|---|---:|---:|---:|---:|
| 组合 CUDA，固定容量 | 64 | 610.9 | 738.1 | 770.7 |
| 同修订独立 CUDA，固定容量 | 64 | 612.1 | 736.7 | 769.3 |
| 组合 OpenMP，固定容量 | 64 | 656.0 | 792.1 | 823.5 |
| 同修订独立 OpenMP，固定容量 | 64 | 556.2 | 692.4 | 723.8 |
| 组合 CUDA，默认自动容量 | 16 | 808.1 | 816.1 | 820.2 |
| 组合 OpenMP，默认自动容量 | 16 | 655.2 | 664.9 | 669.0 |
| 组合 CUDA，固定容量 | 256 | 609.5 | 1177.5 | 1178.1 |
| 组合 OpenMP，固定容量 | 256 | 655.9 | 1213.7 | 1245.7 |

以上是8个有效运行/800个 shower；另有两个各64例的旧安装版内存参考，不计入同修订物理回归。

- 256例后半段 RSS 增长斜率约为 CUDA 2.13、OpenMP 2.15 MiB/shower；**此处尚未出现主机内存平台**。不可将该斜率直接外推到所有能量或所有 N。
- 各有效模拟进程 swap 均为0；模拟期间观测到的系统可用内存最低约7.77 GiB。
- 两种后端的固定容量输运 workspace 均从约0.42 MiB 升到约0.84 MiB，随后稳定至第256例。只读表分配保持 `78,665,960` bytes；native/aux 哈希未变。
- 自动容量 CUDA 组的后端跟踪总分配在16例间保持4694.50 MiB；设备全局显存采样峰值5616 MiB，进程退出后回到约717 MiB。**NVIDIA 数据含桌面与驱动，不是进程独占显存。**
- 组合 OpenMP 仍会初始化 CUDA context，约100 MiB 的额外加载开销在前面的 N=2 与本次 N=64 均存在；这不意味着 EM/radio 在两个执行空间同时计算。源码统计的 OpenMP `device_bytes` 实际属于 HostSpace，不应称为显存。

## 物理输出与复用检查

同修订的组合 CUDA 对独立 CUDA、组合 OpenMP 对独立 OpenMP，各自64例的所有135个 Parquet/NPZ 文件内容逐项相同，两组共270个文件比较通过。包括 longitudinal、dEdX、地面粒子、interaction、production profile、交互直方图以及 CoREAS/ZHS 波形。

这不是要求 CUDA 与 OpenMP 跨后端逐位相同，也不是500/2000例统计验收。所有有效事件都有 complete 标记、有效波形数据；从第二例起 `reused=true`，ordinal 顺序正确，表哈希保持不变。

一个重要的对照陷阱：独立 `install/cuda` 和 `install/openmp` 中仍留有旧随机流/物理修订程序。初次误用它们做逐项比较，结果不一致；此后改用已有的 `build/physics_alignment/cuda` 与 `build/physics_alignment/openmp`，两者 SHA-256 分别为：

```text
671bd3f9a52ab954ff61609040552e931ebc8f053c8e53fbd5a3d26ee994b159
14167c3d35ae225935f8fed2c079b19d19823fc4db4c134402618ea8ed73e8f9
```

正确对照恢复逐项相同。旧安装版数据和最初失败的比较记录保留，不覆盖、不混入正式回归；它们不能用来说明组合二进制出现新的物理错误。使用程序时应核对实际二进制 SHA、`rng_domain_version` 和 `physics_alignment_revision`，不能只看文件夹名称。

## 为什么主机内存还会上升

### 1. 表与工作区：预期保留，未发现逐事件复制

`src/accelerator/em/kokkos/KokkosEmRunSession.cpp:beginProposalNative()` 仅在没有 backend 时导出表、建立后端；后续调用 `beginShower()`。

`src/accelerator/em/kokkos/KokkosBackendInstance.inl:beginShower()` 重置统计、profile、radio 和 pending 队列，同时复用只读表及已经扩容的 workspace。`KokkosWavefrontQueue.hpp` 中的 clear 重置逻辑长度而不是强制释放容量，属于有上限的复用策略。本次分配计数和曲线支持这种行为。

### 2. 输出缓冲：不是每个 shower 都把整个 writer 关闭

`corsika/detail/modules/radio/RadioProcess.inl:endOfShower()` 写出当前波形并调用 observer.reset()；Kokkos radio accumulator 也在 beginShower 时清零。因此这里不是前一例的波形继续累计到后一例。

但 `ParquetStreamer` 的 writer 直到 endOfLibrary 才关闭。当前 Arrow 的 StreamWriter 默认 row-group 阈值为512 MiB，应用未显式缩小或逐事件结束 row group。**该阈值不是整个进程的 RSS 上限。**

实测256例输出中，CoREAS、ZHS 各有一个 row group，编码后未压缩大小分别约198.47、199.26 MiB。结合 writer 生命周期，这为射电输出缓冲占用提供了明确依据；未通过 heap profiler 把每一 MiB RSS 都精确归因到它。

### 3. 报告：随 N 保留，能释放，但不应无限积累

`corsika/accelerator/em/common/GpuEmRunOutput.hpp:recordComplete()` 把每个 shower 的完整 YAML 插入 `summary_`，到 endOfLibrary 才统一写文件。`SimulationTiming`、radio diagnostic summary 等也保存逐事件摘要。

独立 YAML 所有权实验用本次真实报告作为模板，重复上述保留方式：原型加载后 RSS 12.81 MiB，保留256份约109.15 MiB；销毁后 glibc 暂不向系统归还空闲页，诊断性调用 `malloc_trim(0)` 后为13.04 MiB。**这说明 RSS 不立即下降不等于对象未释放；但生产进程仍有理由限制单进程 N。** 该探针不是 sanitizer，也没有将 malloc_trim 插入生产热路径。

另一个需要后续控制的容器是 `InteractionCounter::timing_samples_`：逐强子反应 append、当前没有逐 shower 清理接口。这轮主要是 EM 光子，不能用本轮结果界定其在高能强子长批次中的上限。

## 防护与实际处理

测试服务使用 systemd user service，不依赖临时交互终端：

- 模拟进程 RSS >3 GiB 或主机 available <3 GiB：停止本测试。
- 监控器同时检查测试 runner RSS、子进程及进程树总占用。
- cgroup 设 `MemoryMax=5G`、`MemorySwapMax=0`、`KillMode=control-group`、`Restart=no`，防止采样间隔内快速增长拖垮 WSL。
- 停止逻辑只作用于自己启动的进程树，不杀 IDE、远端任务或其他用户进程。
- 4项现有内存保护单元测试、4项新增生命周期/表哈希检查测试均通过。

本轮保护曾正确停止**Python 分析 runner**：连续读取多个结果后，其 Arrow/YAML 分配器缓存使 RSS 超过1 GiB。未通过提高阈值绕过，而是将每次输出验证和比较放到独立短进程，进程退出后释放地址空间；然后复用已完成的模拟数据继续测试。最终模拟均无内存门禁触发。

另有首次启动时服务未继承 FLUPRO 的环境错误，发生在 shower 开始前；已显式传入激活 corsika_venv 后的 FLUPRO/PATH。没有因此切换强子模型。

## 目前建议与下一步

当前短批次可正常执行，但**不建议在本机直接把单进程 N 设为2000并视为内存风险已解除**。资源紧张时优先一个 shower 一个独立进程、为每例保存明确 seed 和状态；需要复用时先用小批次及同样的内存保护。不要在已定义随机流的 campaign 中随意改变 N 后继续冒充同一组事件。

真正限制长批次主机内存的下一步应是输出层，而不是删物理轨迹或每例重新加载 GPU 表：

1. 给 Parquet row group 设置较小的显式上限，或在事件边界 flush；验证读取后的数组不变、评估压缩率/I/O成本。
2. 将逐事件详细报告流式写盘或分片，再生成兼容摘要；必须保留 complete/incomplete 和已有 schema 语义。
3. 在报告消费后释放不再使用的强子 timing samples，同时保留累计计数；验证全部调用方。
4. 再做高能 N>1、较大天线/时间窗和 heap/allocator 分配栈检查。本轮不将有限的低能测试冒充这些验收。

上述生产层调整本轮没有实施；没有未经验证修改输出 schema、物理算法或安装二进制。

## 文件与复现入口

诊断源码：

- `validation/accelerator/run_nmulti_memory_acceptance.py`：同进程矩阵、RSS/PSS/私有页/显存采样、事件日志检查点及保护。
- `validation/accelerator/analyze_nmulti_memory.py`：同修订比较、CSV/JSON 和内存诊断图。
- `validation/accelerator/probe_yaml_report_retention.cpp`：孤立 YAML 保留/释放实验。
- `tests/accelerator/test_nmulti_memory_acceptance.py`：拒绝缺失复用、变化表哈希、不完整事件；避免把逐事件求解迭代数误当只读表哈希。

D盘结果位于 `CorsikaData/corsika_validation_results` 下：

```text
diagnostic_beta5_dual_nmulti_memory_20260907_v2/
  # N64、N16原始日志/监控；含明确排除的旧安装版对照
diagnostic_beta5_dual_nmulti_memory_20260907_N256/
  # 两个同进程N256运行
diagnostic_beta5_dual_nmulti_memory_20260907_aligned_N64/
  # 同修订独立对照、最终memory_review.json、memory_metrics.csv、图和报告
```

每次命令、二进制 SHA、唯一 PID、运行时间与采样记录均保存。事件检查点是读到日志时的近似采样，不是插入 C++ 的同步屏障；device-wide GPU 采样频率约1 Hz，RSS/PSS约4 Hz，不能排除更短的峰值。源码计数用于交叉检查，内核/cgroup保护负责峰值兜底。

GitHub 推送不包含模拟数据。补测数据留D盘，README中的记录说明未同步替换远端代码或正在执行的远端任务。

