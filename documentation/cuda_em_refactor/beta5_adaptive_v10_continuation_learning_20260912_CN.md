# adaptive-v10：自主 GPU 调用更新私有成本估计

## 结论与边界

已复现并修复一处调度估计未更新的问题；**尚未证明 v10 有真实设备净加速**。
只改变显式 adaptive 双端的自主续跑控制，不改物理 kernel、随机数、表、
cut、thinning、射电公式、单端、legacy 或山体路径。生产安装和正在运行的
v9 100 PeV 对照不替换。这里不改随机数生成/身份规则；动态调度顺序仍可改变
独立 shower tree，不承诺与 v9 逐事件相同。

## 证据

PSR v9 的五对 100 TeV 质子全射电测试完整结束，同一二进制、130线程绑定
382–511，GPU 为 T400：纯 OpenMP 中位 51.378 s，双端 67.026 s，后者多用
30.458% 时间。设备竞争检查通过。这不是已达到广泛适配的结果。

v8/v9 的 GPU 自主 packet 在启动时复制 `AdaptiveSubshowerControl`，但在
packet 内多次调用时没有更新这份副本。`waves()` 同时依赖实测每 input-wave
成本和上一次请求大小；因此旧估计可能反复发出过短或过长的租约，直到
协调器领取整个 packet 才学习。PSR 较多的小波前与这一机制相容，但不能仅凭
总计数认定它解释了全部回退。GPU 结果领取等待也不能直接从 wall time 扣掉。

## 修改与并发安全

```text
协调器：复制估计 + 预留有界 history bank → 提交
GPU driver：完整调用 → 私有 observe → 决定下一次粒子/波数租约
                         ↓ 达到时间、内存、次数或交还条件
协调器：逐结果提交一次 → 更新全局估计
```

在 `IndependentSubshowerPump.hpp` 的自主循环中，下一次计划前将刚完成调用的
输入数、实际波数、请求波数、输运步数和实测时间传给私有 controller 的
`observe()`。driver 不写协调器的 controller，也不访问 CPU 栈或 writer。
协调器仍对每条完成结果仅观察/提交一次；私有预测更新不是增加一次物理执行。

原有最多16次调用、软时间范围、64 MiB额外结果预算加一个常规受限结果、
history 范围和异常失败门禁保持不变。metadata 标识为
`adaptive-v10-continuation-learning`，并说明私有估计的语义。

## 可复现的调度测试

`testIndependentSubshowerPump.cpp` 使用 mock endpoint，不是物理 oracle。
主协调器被 OpenMP mock 阻塞直到 GPU 第四次调用。第一个 GPU mock 调用
额外保留25 ms的确定性最小成本，使64波的冷启动租约在后续同种调用中应降低；
不再依赖未插桩机器的瞬时运行速度来断言租约必须增加。

- 同一新测试配冻结 v9 头文件：退出1，明确报告自主 GPU 复用旧估计。
- 配 v10 头文件，ASan＋UBSan＋leak：退出0，实际租约64 → 7；其他所有权、
  波前进展、结果容量、history、异常清理测试同时通过。
- TSan：退出0，实际租约64 → 6，完整测试通过；不把“未报告 race 但断言
  失败”的早期测试算作通过。

早期仅断言租约增加的测试在 TSan 下失败：插桩增加成本后，正确动作可能是
缩短租约。现采用上述有明确成本的 fixture，检验是否学习，不把断言放宽为
允许不学习。隔离的 `setarch x86_64 -R` 只作用于 TSan 子进程，不改系统 ASLR。

## 待完成验收

隔离 Release 构建、单端固定种子输出/trace、N=2/N=32、真实双端 EM/radio、
PSR 五对纯 OpenMP/双端计时。随后在本地 GPU 空闲时再构建并做同二进制对照。
这些通过前不替换安装、不推送、不宣称 v10 更快；最终还需高能及大样本验证。

## 冻结与排队记录

PSR source 从冻结v9复制，仅覆盖pump、应用report、相应mock测试和两个验收器
的版本门禁，另加入诊断文档；`diff -qr`未发现物理kernel或山体文件变化。

- stage：`build/psr-adaptive-v10-continuation-learning-20260912`。
- pump SHA-256：`025334b910b7906a12fbfcd7e422a3e7a49f592ca7d5262c55bcd038f55a4e0f`。
- test SHA-256：`88a29a3ea1ade380f8ff27db6f2e3b5f0c8d12533713a320747c575e5c442516`。
- 构建service：`c8-psr-adaptive-v10-build-20260912`，等已有山体v13 workflow
  PID293415（creation1789154280.48）结束，GPU空闲30s后在0–125上128并行编译。
- 测试service：`c8-psr-adaptive-v10-tests-20260912`，等构建成功后绑定382–511，
  五对100TeV同二进制纯OpenMP130／双端130，全射电。GPU竞争检查启用。
- 输出：PSR `psr_t400_adaptive_v10_continuation_learning130_20260912`。

本地100PeV v9单CUDA／双端仍使用原归档二进制，不在中途替换；本地暂不编译。
排队记录不是构建成功或实测性能证据。新同二进制分析工具
`accept_adaptive_same_binary_pair.py`按实际backend区分输运步数，单CUDA不存在的
cooperative job字段记为未提供，不再硬套旧双端的批次计数。五项host contract
测试通过，涵盖物理参数、缺失GPU独占证据、重复CLI项及端点/提交计数不闭合。

## 同二进制分析工具实样本复核

新工具已在一对完成的v9 Fe100TeV数据上端到端执行：逐批Parquet、表/常数/参数/
二进制核对、profile、实际端点步数及原pulse_analysis坐标投影均通过，生成耗时、
单事件profile和CoREAS/ZHS Ex′/Ey′/Ez′图。只是验证分析工具，不是新的v10性能样本。
输出在D盘`beta5_adaptive_v9_foreground_20260912/pair_audit_tool_fixture`。

实样本同时暴露旧分析器的计数假设不能直接套到单CUDA：
`PhysicalAcceleratedEmRouter::returnFallbacks()` 中，独立双端立即执行的指定
过程会计入`particles_returned_for_cpu_fallback`；单CUDA先延后，计入
`deferred_cpu_fallbacks_queued/flushed`。因此历史名称`cpu_generic_fallbacks`
不能直接横比，也不能一律要求等于所有reason计数之和。

工具采用代码对应的完整关系：

```text
全部fallback事件 = returned_to_scalar + deferred_queued
deferred_queued = deferred_flushed（结束时）
立即指定完成 = specified_final_states - deferred_flushed
未指定标量回退 = returned_to_scalar - 立即指定完成 ≥ 0
```

另外检查按process/reason的总数与native selected-loss完成数。新增mock分别
覆盖立即/延后模式及损坏计数，现为6项测试全部通过。该修复仅在验收工具中；
没有改变router、fallback实际执行或正在计时的二进制。

## PSR CPU 调用变慢的进一步诊断（未定因）

v9 seed2026110001 的完整摘要显示：纯 OpenMP 有109,727,231个实际输运步、
9,572个驻留波前，photon/lepton后端调用累计41.108 s；双端的CPU端仅有
100,712,644步、9,302波前，但CPU调用累计51.676 s。GPU另处理4,122,488步，
17,890波前。指定fallback耗时201.8 ms →277.9 ms，不能解释10.57 s的CPU
调用差。因此“CPU批次更多”或“fallback占主要时间”都不足以解释该对样本。
两事件物理工作不同，以上不是固定轨迹的kernel性能定论。

热路径检查没有发现显式全局`Kokkos::fence()`或无execution-space的双参数
`deep_copy()`；OpenMP staging仍为HostSpace，非CUDA pinned space。CPU arena
仍为单端大小65,536，没有继承旧legacy的16,384限制。不能在没有证据时将
原因归给这些已检查的机制。

130线程运行记录中，仅比较线程池存在的区间：纯OpenMP样本10.68–48.04 s
平均约127.98核；双端10.69–59.65 s约126.25核，其中独立CUDA driver约0.352核。
双端130个OpenMP线程和driver都被限制在同一130核集合；本地20线程则仍有
额外逻辑CPU可供driver运行。这支持检查抢占/屏障放大的必要性，但CPU使用率
包括自旋，不能凭此认定发生了10秒调度损失。尚未修改线程数、绑核、OpenMP
等待策略或系统调度策略。

### 无额外全局等待的独立调用探针

`validation/accelerator/KokkosHostCallProfile.cpp`是可选Kokkos Tools插件，
不链接到生产目标，不依赖CUDA/OpenMP runtime，不改kernel。使用Kokkos 4.7.03
安装头中的ABI，显式返回`requires_global_fencing=false`。若不返回该设置，
Kokkos工具默认全局等待会破坏双端重叠，导致错误诊断。

探针聚合parallel_for/scan/reduce、fence、deep-copy和分配调用。最多2,048个
标签、1,024个未结束调用、每host线程64层copy；超限、版本错误、未闭合调用或
重复结束均将报告标记为无效。已有报告不覆盖。所有耗时是**主机API包围区间**：
CUDA项不是设备kernel时间；嵌套/跨线程区间不能直接相加当总时间。分配字节是
累计分配，不是内存泄漏量。插桩可能改变动态调度，不并入生产加速统计。

PSR上独立`g++ -O2 -Wall -Wextra -Werror`编译通过；8类ABI回调测试通过：
双调用线程的计数、重复结束、未闭合、标签上限、在途上限、copy嵌套上限、版本
不符和已有文件保护。只用了0–3上的主机编译/测试，没有访问GPU。

`run_kokkos_host_call_diagnosis.py`在v10正式五对计时成功之后才启动两个种子的
OpenMP/双端诊断，顺序交替，绑定仍为382–511。读取已验收命令，只改输出目录
和加载插件；记录二进制/插件hash，启用RSS/GPU独占门禁。单OpenMP额外逐项
比较插桩前后物理输出；双端允许调度变化，但检查完成、提交和累积溢出。

- 独立探针stage：`build/psr-host-call-profile-20260912`。
- 排队service：`c8-psr-adaptive-v10-host-profile-20260912`。
- 结果目录：PSR `psr_t400_adaptive_v10_host_calls130_20260912`。

本节截至排队时仅完成插件自身测试，**尚无真实shower调用分解结果**。本地正在
运行的v9同二进制100PeV对照和PSR正式计时均不加载这个插件。

### 构建后进展（2026-09-12 03:55 CST）

PSR已有山体v13 workflow正常结束后，v10隔离Release构建自动启动并退出0。
`testIndependentSubshowerPump`、`testAdaptiveSubshowerControl`、
`testCooperativeDriverAffinity`三项CTest通过，总计0.52 s。
随后真实回归启动：标量PROPOSAL的11份物理数组及metadata比较通过；此时
CUDA/OpenMP与生命周期回归仍在运行，尚无v10正式计时结论。插件SHA-256为
`c629d9e4ee4c32e270fe4cd4dd08218b3677c7661c9c5867d2a6e0b61b32dbf3`。

随后真实回归全部通过：三个单端各11份物理数组和metadata一致，CUDA/OpenMP
trace相同，真实双端EM/radio N=2通过，adaptive N=32用时49.588 s并正常关闭。
`CORRECTNESS_GATES.json`记录`passed=true`、`physics_statistics_accepted=false`；
五对100TeV正式计时已经启动。该回归结果不等于大样本通过或v10净加速成立。

## PSR 五对正式计时结束：仍未通过净加速

10/10事件正常结束，CPU亲和性及GPU独占检查通过；同二进制、130线程、
100TeV质子、47°/180°、默认max-weight、全射电。

| seed尾号 | OpenMP130 / s | v10双端130 / s |
|---|---:|---:|
| 0001 | 50.810 | 62.081 |
| 0002 | 57.425 | 69.987 |
| 0003 | 46.861 | 58.146 |
| 0004 | 53.847 | 64.713 |
| 0005 | 47.230 | 60.818 |
| 中位 | 50.810 | 62.081 |

双端中位仍多用22.18%时间，五对均慢于纯OpenMP。相对v9的双端中位67.026s
降低约7.38%，但动态shower工作量发生改变，不能将全部差值归为调度效率改善。
未更改生产推荐或默认模式。

第一seed的GPU驻留波前17,890→12,296，实际GPU步数4,122,488→4,013,625；
但GPU调用数1,106→1,259，其中输入小于4096的调用1,184次（94.04%）。
CPU端8,541波前、100,860,798步、49.125s后端调用，相比纯OpenMP9,572波前、
109,727,231步、40.581s，仍需解释单位工作的CPU耗时。迁移总耗时0.671ms，
不支持“搬运本身占了十几秒”的说法。粒子组成/轨迹仍不同，需调用探针确认。

已生成逐seed耗时图、资源时间线、batch-input直方图并同步到D盘：
`psr_t400_adaptive_v10_continuation_learning130_20260912_summary`。
历史命名的`report_psr_adaptive_v4.py`现从实际CONFIG读取policy、线程和绑核，
不会把v10误标为v4；新增四项metadata测试通过。

独立host-call诊断第一次被旧冻结source中的guard拒绝（不支持
`--require-exclusive-gpu`），模拟未启动，不影响上表。启动器现在显式指定
已审查的新guard、启动前检查其`--help`能力并记录guard哈希；三项门禁测试通过。
旧失败目录保留。重新诊断使用`psr_t400_adaptive_v10_host_calls130_20260912_r2`
和service`c8-psr-adaptive-v10-host-profile-r2-20260912`；不修改冻结物理source。

本地v9同二进制100PeV单CUDA也已结束：3930.715s，3,250,768,551实际输运步、
650,122驻留波前，结束/队列/溢出门禁通过，RSS峰值1.316GiB。20线程双端随后
自动开始，尚未结束，因此此时不预报100PeV加速比。

## 04:33 更新：探针扰动、profile 分片实验与监控修复

全量/1:64抽样host-call诊断均完成，两种子、四次运行各自正常关闭；两种
探针下单OpenMP的9份数组和metadata逐项一致。抽样插件SHA-256为
`6e53d75e76c8c2b0e06198aab6a4d64397e081e3f7c9d9b9c2e5cb4da7465538`。
新增抽样计数/方差分析器和5项host测试；10类插件ABI测试均通过。

不能把插桩当无扰动：第一seed全量双端62.081→89.259s，CPU波前
8,541→18,360；1:64抽样为69.504s、9,107波前。单OpenMP即使抽样仍比
未插桩的50.810s慢约14.7%。第二seed抽样双端72.152s（原69.987s），
CPU波前18,112→9,550。动态调度和系统噪声均不能排除，不能从此直接算
生产损失百分比。两种探针都将射电投影和输运统计/profile累积列为主要热点。

新报告 `report_kokkos_host_call_diagnosis.py` 区分样本数量与估计总量，输出
抽样SE，稀疏行显式标注；不将CUDA host区间当kernel时间、不将重叠区间相加。
结果已同步到D盘 `psr_t400_adaptive_v10_host_calls_comparison_20260912`。

### 相同轨迹账本的分片实验

新增独立 `validation/accelerator/ProfileShardingProbe.cpp`，直接调用已有
`accumulateLeptonProfileStep`，没有复制物理公式或改变shower后端。
冻结v10的头文件、生成常数和依赖作为编译输入；130线程绑定382–511。
每次262,144条合成轻子步，覆盖e±/mu±、不同终止条件、沉积和散射计数，
不代表真实shower的轨迹组成。预热后每配置5次，顺序交替，含checked最终合并：

| 账本分片数 | 中位耗时 / ms |
|---|---:|
| 1 | 35.663 |
| 4 | 22.027 |
| 16 | 9.339 |
| 32 | 5.034 |

全部整数直方图和DeviceProfileCounters与串行同公式oracle逐项一致，未溢出。
32分片在这个隔离热点上约7.08倍；**不是全shower加速**，也不证明仅加入GPU
有收益。原型未接入生产，下一步需要有界内存、N>1重置、溢出、真实轨迹和
同二进制端到端验收。若仅优化双端CPU的profile，必须将这项kernel改进与
GPU协同贡献分开报告，不能把全部收益归因于两端重叠。

扩大候选范围后的独立复测也通过（同一130线程、相同合成账本、每配置5次）：
1/4/16/32/64/128/256个分片的中位总耗时分别为
35.653/21.866/9.238/4.936/2.787/1.876/0.309ms，所有整数结果一致。
20线程在PSR同平台上的初步测量为1/4/16/32分片
47.211/22.801/12.981/0.621ms；不是本地笔记本的测试。
这里256分片的直方图仅2MiB，但真实profile网格可能更大，生产实现仍须按
显式总字节预算选择并有禁用路径，不能按线程数无限复制。即使只有少数线程
共享计数器也会成为每波屏障的长尾，因此不能不经测试把32写死为通用最优值。
当前只证明同公式累积存在可消除的缓存争用；没有将上述局部比值外推至shower。
原型输出与诊断报告一起同步到D盘，生产kernel仍未更改。

### 本地100PeV没有完成：被旧监控误杀

04:23，v9双端运行1133.132s时，独占检查中的`nvidia-smi` PID查询超过3s，
异常进入旧guard的finally并杀掉模拟（returncode=-9）。当时RSS峰值1.570GiB，
最低可用内存10.690GiB，没有foreign GPU PID证据；不是内存门限或已记录的
物理失败。原 `adaptive/` 和失败记录保留，1133s不得当作完成用时。

`run_overlap_guarded.py`现将运行中的外部查询超时/错误记为“未知”：继续模拟
和内存/RSS/总时限保护，下一轮重新观测；实际发现foreign GPU仍只停止自己的
子进程。任何独占观测缺口永久使本次`performance_valid=false`，不伪装为空闲。
启动前查询失败仍不启动；错误样本至多保存16条，另外计总数。无法用这些
监控缺口判断驱动或kernel已失败。UHE wrapper区分模拟完成和计时有效性。

10项monitor测试通过，包括真实子进程在模拟查询超时后正常退出、恢复观测、
有界错误记录、foreign PID和启动前门禁；其余配对分析与launcher测试通过。
只更换诊断guard，不改归档模拟二进制、物理参数或种子。新service
`c8-adaptive-v9-uhe-adaptive-r2-20260912` 已启动在新目录 `adaptive-r2/`，
04:33仍运行；不存在自动失败重试循环。本轮仍待高能完整结果，不能宣称通过。
