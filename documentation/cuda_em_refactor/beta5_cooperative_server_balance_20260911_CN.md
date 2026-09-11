# 双端独立队列：PSR 大核数负载均衡修正

## 适用范围

仅空气应用显式 `cuda-openmp` 独立子级联模式；不更改单 CUDA、单 OpenMP、
标量 PROPOSAL、山体 session、物理公式、cut、thinning、射电公式或默认 max-weight。
双端动态调度会改变 history 分配时序，不承诺修改前后同 seed 的 shower 树相同。

## 已确认的旧版限制

PSR 与本地的六个调度/执行关键文件哈希一致，服务器实际编译 `sm_75` Release。
不是丢失了独立队列版本。T400 + 130 线程的短事件在单 GPU 任务期间连续执行了
32 个 OpenMP 工作段。物理核心 126–255 对应逻辑 CPU 382–511，每核一个 SMT 线程。

旧 100 PeV 试跑的一个采样窗口：最近 60 s CPU 约 491%，最近 300 s 约 607%，
历史峰值约 12935%；GPU 当时 99–100%。许多 OpenMP 线程在 futex 等待，
但仅凭该观测不能量化各类等待的贡献。旧完整事件并未完成，不能作为最终耗时基线。

源码存在以下可修正限制：

1. CPU batch 上限固定 2048，与 130 线程不匹配。
2. CPU 初始份额固定 1/16，只有整个 CPU 队列为空才补给。
3. GPU 在途时整个 `rebalance()` 直接返回，连协调器中尚未提交的粒子也不能迁移。
4. GPU 轻子 epoch 固定最多 1024 波前，慢 GPU 的安全交接边界可能很久才到达。
5. CUDA driver 继承 OpenMP master 的单 CPU 亲和性；PSR 两者均固定在 CPU 382。

## 修改

- 共享 CPU arena 按 `next_pow2(max(2048, 64*threads))` 调整，上限 16384。
  20 线程仍为 2048，130 线程为 16384。不是每线程复制一个 arena；既有字节预算仍生效。
- 初始 CPU 份额为 `clamp(threads/320, 1/16, 1/2)`，随后使用原来的吞吐移动平均。
- 每类 CPU 待处理量低于一批时提前补给，常规待发队列限定两批；原 spill 安全上限仍为四批。
- GPU 在途时仅允许迁移 coordinator-owned waiting vectors：
  提交给 driver 的 job 已独立持有输入；不访问在途 endpoint 的队列、计数或设备 View。
- CPU 无工作等待超过 5 ms 且 GPU 轻子 epoch 超过 100 ms 时，逐步降低 GPU epoch 波前上限，
  最低 16。CPU 不缺工作且 epoch 低于 50 ms 时，可恢复到最高 1024。
  时间阈值只是调度反馈，不是粒子传播时间或物理 cut，不中断执行中的 kernel。
- GPU driver 使用 OpenMP 当前 place partition 的 CPU 集合；无绑定的运行库则保持
  `sched_getaffinity()` 返回的集合，不更改主线程或 OpenMP workers 的绑定。
  实测 libgomp 在 `main()` 前已经将 master 绑到单核，仅在 Kokkos 初始化前读取
  `sched_getaffinity()` 仍然太晚。PSR 的 partition 正确保留逻辑 CPU 382–511。

新增双端报告计数：提前补给、在途期间待发粒子迁移次数/粒子数、GPU epoch 缩短次数、
最终轻子波前上限、初始 CPU 份额及最长 GPU epoch。原有定点 profile/radio 合并不变。

## 安全验收

- 受控 GPU 阻塞时，CPU 可反复消费未提交尾部；测试禁止主线程读取 live endpoint。
- 混合 PID、完整回收、唯一 history、提前补给、队列溢出、零进展拒绝、异常清理仍检查。
- CPU 容量覆盖 20、130 与极大线程数，验证上限。
- PSR 系统 GCC 的 ASan/UBSan 调度测试通过；Conda sanitizer 在该主机异常，
  未以其运行失败作为通过证据。
- 完整二进制、真实 PROPOSAL/CoREAS/ZHS、单端 N=2 输出与 decision trace、
  双端 N=32 生命周期及 130 线程真实 EM 检查由独立服务器流水线执行。

## 部署及结果

源代码与二进制隔离在服务器 `build/psr-cooperative-balanced-20260911`，
旧服务器二进制及本地生产二进制不被覆盖。本地不进行编译或 GPU 测试。

结果目录（PSR）：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_balanced130_20260911_v1`

v1 先执行正确性门禁，并完成两个 seed 的 100 TeV 旧/新双端诊断。
用户随后明确：主要判据是增加 T400 能否超过纯 OpenMP130，而不是比旧双端快。
因此停止后续旧版计时，v2 使用同一个新二进制交替运行两 seed 的 100 TeV
纯 OpenMP130 / T400+OpenMP130，随后进行同 seed 的 100 PeV 两模式对照。
物理参数不变，全射电，70% 显存预算，两种模式均限定逻辑 CPU 382–511。
v2 结果目录：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_balanced130_20260911_v2`
短事件通过不等于高能性能通过；提高 CPU 占用不能单独证明加速，最终应比较总耗时。

## 2026-09-11 已完成的复核

- PSR Release 构建通过；三项调度/driver CTest 通过，包括慢 GPU 后续 epoch 缩短测试。
- 系统 GCC 的 ASan/UBSan 复测通过。
- 标量 PROPOSAL、单 CUDA、单 OpenMP 的固定 seed N=2：各 11 类物理数组逐项一致，
  非计时 metadata 无差异；CUDA/OpenMP 的 decision trace 分别逐字节相同；help 相同。
- 新双端 100 GeV photon N=32、20 线程：32 例均完整、两端均有输入，
  queue/profile/radio 溢出为零，OpenMP 工作区均为 18,892,272 bytes。
- 130 线程真实 EM + CoREAS/ZHS fixture N=2：能量通量相对残差
  `5.51859e-9`、`5.43419e-9`，工作区总量均为 454,229,950 bytes。
  测试固定 arena 在 T400 的 10% 预算下被正确拒绝，改用显式 20% 诊断预算后通过；
  不修改空气生产任务的 70% 上限。
- 1 TeV 新双端启动检查：11.279 s，CPU 采样峰值 10349.6%，
  CPU batch 上限 16384、初始份额 0.40625；GPU 在途时搬运安全待发粒子 2602 个，
  无需领取正在执行的 GPU 队列。此短事件时间不是高能加速比。

独立构建使用 RUNPATH，比较时两组显式加载同一个冻结的物理共享库目录。
初次检查发现新构建启用了 FLUKA `flkavr` 查询，旧构建使用字符串提取，版本显示不同；
已统一实际加载的 FLUKA 库，再通过严格 metadata 比较，没有忽略该字段。
初次诊断记录保留在结果目录的 `metadata_linker_attempt/` 与
`fixture_10percent_budget_attempt/`，不是成功事件。

第一组 100 TeV 完整对照（seed 2026110001，47°/180°，emthin=1e-6，默认 max-weight，
全 CoREAS/ZHS，T400 + 130 线程）：

| 指标 | 旧独立双端 | 队列/批量优化版 |
|---|---:|---:|
| 端到端秒数 | 366.574 | 208.735 |
| shower 秒数 | 356.337 | 198.668 |
| 平均 CPU 占用 | 2152.8% | 6431.9% |
| 峰值 CPU 占用 | 12906.1% | 12999.1% |
| coordinator 无 CPU 工作的轮询等待 | 268.405 s | 63.972 s |

单对观测耗时下降 43.1%，但动态调度会改变 shower 树，不能用单对宣称平均加速。
该组尚未包含最终 place-partition 绑定修正；两种版本 CUDA driver 仍绑在 CPU 382。
后续修正需新二进制、新结果目录并重过回归，不混用二进制。
等待计时和 driver 窗口交集不是 CUDA kernel 活动的硬件时间线。

完整强子 shower 的 accelerator-only 能量账本标记 `complete_coverage=false`；
不能将其 residual 当成完整能量守恒证书。本轮守恒门禁使用真实纯 EM fixture，
强子大样本物理统计仍需另外执行。

v1 第二组完成：旧双端 377.010 s，队列优化版 184.967 s。
这些结果只证明比旧调度改善，不能证明比纯 OpenMP 更快。
v2 会重新验证最终亲和性修正、单端回归和生命周期，再运行纯 OpenMP 对照。
尚不宣称加入 GPU 的净加速或高能性能门禁通过。

## 同版本纯 OpenMP 对照：当前主要结论

最终亲和性修正已完成 Release 编译、4 项 CTest、单端固定 seed N=2 数组/trace/help 回归、
双端 N=32 和 130 线程真实 EM+radio 复测。
实际工作的 CUDA driver 已允许逻辑 CPU 382–511；另一个兼容路径的空闲 driver
仍绑定 382、CPU 时间为零，不参与独立队列工作。

v2 第一对（seed 2026110001，100 TeV，全射电，相同二进制和物理参数）：

| 模式 | 端到端时间 | CPU 平均占用 |
|---|---:|---:|
| 纯 OpenMP130 | 52.064 s | 10620.5% |
| T400 + OpenMP130 | 189.001 s | 4922.2% |

单对结果中协同比纯 OpenMP 慢约 **3.63 倍**，并没有因加入显卡而提速。
因此前述旧/新双端耗时改善不能作为协同成功的判据。
纯 OpenMP 样本实际 GPU 利用率为 0，metadata `gpu=false`，不是混合运行。

已确认仍不同的调度条件：纯 OpenMP 最大驻留批量为 65536，
其轻子每次最多 1024 波前；当前协同 CPU 端为 16384 和 8 波前。
这会增加协同 CPU 端交接频次，但尚未通过消融测试量化其占比；
不能将全部差距归咎于 T400 硬件，也不能保证仅增加批量即可得到净加速。
CPU 端仍自行选择 OpenMP 的 chunk/tiling，并未直接使用 CUDA 的 tiling。
该协同样本的 coordinator 无 CPU 工作的轮询等待累计 74.370 s，
最长 GPU 工作段 20.864 s；提前补给 275 次，在途期间转移安全待发粒子 299307 个。
因此补给修正已实际执行，但仍未消除等待。CUDA提交/提交结果计数均为 136，未漏收结果。

继续交替跑完第二对 100 TeV，随后运行同 seed 的 100 PeV 纯 OpenMP130 与协同对照。
目前只完成小样本运行和正确性回归，不是大样本物理验收，不推荐将协同设为生产默认。
