# 双端优先级接口：交付范围与逐项证据

## 本轮需求

保留原有 `cuda-openmp` GPU 优先独立子级联模式，新增 `openmp-cuda`
CPU 优先、GPU 辅助模式。按最新任务安排，本地验收 GPU 优先 20 线程，
PSR 验收 CPU 优先 130 线程；此前完成的其他模式数据全部保留。

这是一项接口、调度与短测交付，不等于承诺所有硬件上双端都比单端快。
大样本物理等价、100PeV性能、完整强子能量闭合不由5个Fe种子证明。

## 实现和保护边界

| 要求 | 源码及已检查证据 | 结论范围 |
|---|---|---|
| 两种模式明确选择，不自动改变默认 | `applications/c8_air_shower.cpp` 的 execution 枚举；`KokkosBackendSelection.cpp`；两机 `HELP.json` | 新旧帮助信息与既有参数保持一致；显式接受 CPU 优先 |
| 原 GPU 优先保留 | `KokkosCooperativeBackend.cpp` 的 `!adaptive_ && !cpu_primary_` 分支；`IndependentSubshowerPump.hpp` 的原分配/工作段逻辑 | 本次 v1→v2 简化变更限制在 CPU 优先；不把历史跨轮性能波动解释为旧 GPU 策略被重写 |
| CPU 优先且 GPU 辅助 | `CpuPrimarySubshowerPolicy.hpp`；pump 的 `enqueueCpuPrimary`、`rebalanceCpuPrimary` | 保留 CPU 完整 arena 和一个有效 staging batch；只学习输入份额，空闲安全边界调配 |
| 两端独立推进、主线程唯一提交 | `testIndependentSubshowerPump` 将 GPU 阻塞，断言 CPU 多工作段持续前进；混合 photon/lepton 身份与提交数量检查 | 不读写在途设备队列，不丢失/重复提交；不由此宣称生产全程满载 |
| 不改物理公式和山体代码 | `SOURCE_BOUNDARY_AUDIT.json`：v1→v2 共 4 个允许的调度/报告文件变化、662 个核心文件不变；主工作树3个调度文件哈希与冻结v2一致 | 只限定本功能冻结快照，不重置工作树里其他山体开发 |
| CPU/单CUDA/单OpenMP 回归 | 两机 `proposal-regression.json`、`cuda-regression.json`、`openmp-regression.json` | N=2物理数组及非计时metadata一致；两条单Kokkos decision trace逐字节一致 |
| 大一点的单 CUDA 输出保护 | 本地5份 `Fe100TeV-*-single-cuda-regression.json` | 同种子45份物理数组和非计时metadata一致 |
| 双端真实 EM 和射电 | 两机 `fixture-cuda-openmp-guard`、`fixture-openmp-cuda-guard`；`testKokkosCooperativeBackend.cpp` 使用显式 `require` | 8192个e±及生成的EM次级，两事件、两端实际工作、CoREAS/ZHS非零、结果领取不重复、无溢出 |
| 能量账本 | 真实EM fixture显式计入fallback流出与介质输入；本地相对残差约5.5e-9，门限1e-4 | 仅加速器边界账本；Fe完整强子账本覆盖不足，不能认证全shower能量闭合 |
| 多事件内存和生命周期 | 两机 `N32_LIFECYCLE_AUDIT.json` | 两模式N=32缓存、工作区、hash、ordinal、提交计数及内存门禁通过；不声称所有能区已证明无泄漏 |
| 非法组合门禁 | 两机 `bad-policy.json` 和 `bad-workers.json` | CPU优先拒绝另一个adaptive策略；强子worker保持1 |
| 核绑定和终端独立 | PSR资源日志逐线程亲和性；独立systemd执行和报告服务 | 130线程限定CPU382–511；没有随SSH断开而终止；不影响其他生产任务 |

本地隔离 Release 的 `testIndependentSubshowerPump`、
`testAdaptiveSubshowerControl`、`testCooperativeDriverAffinity` 又复跑3/3通过。
队列修订、历史模式保留、报告样本数和输出检查另有主机Python单元测试。
上述检查程序的覆盖范围已对照源码，而不是只看顶层 `passed=true`。

## 结果位置与当前完成条件

本地：
`CorsikaData/corsika_validation_results/beta5_priority_endpoints_v2_simple_20260912/run`。
PSR原始数据：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_priority_endpoints_v2_simple_130_20260912/run`。
服务器小报告镜像：
`CorsikaData/corsika_validation_results/psr_priority_endpoints_v2_simple_130_20260912_report/run`。

本地15/15完整（包括所需GPU优先5例），已再次重读输出；PSR所需CPU优先
5例也已完整结束，连同3例保留参照全部重读通过。报告服务退出0，父临时
单元已回收，无法事后认证其退出状态，但独立子事件完整性已检查。
小报告、图像已同步并通过内容校验，两机指定主端总结已生成并目视检查。
原始粒子、轨迹和波形数据不搬回WSL。

**短测交付完成不意味着性能问题解决。** 新的诊断表明PSR CPU优先中的
有效OpenMP吞吐明显低于单端，仅保持容量和核占用不足以证明保护了性能。
详见测试记录最后一节；该问题的隔离归因仍未完成，不将此版本升级为生产推荐。

最终结果参见[简化策略测试记录](beta5_priority_endpoints_v2_simple_20260912_CN.md)
及两机报告。此文件列出的源码路径相对于源码根目录。
