# adaptive-v8：按实测时长、有界内存自主续跑

## 当前结论

实验性双端调度，不更改单CUDA、单OpenMP、标量PROPOSAL、legacy双端或山体流程。
没有替换生产install，物理kernel、表、cut、thinning、磁偏转常数不变。
**主机所有权/边界测试已通过，跨机器真实性能验收尚未完成。不能宣称已得到最优加速。**

本地五种子同二进制 Fe 100 TeV 对照现已完成，10/10输出完整性通过：单CUDA
中位74.039s，双端20线程60.280s，描述性1.228×、耗时降低18.58%。五对均更快。
v7双端为62.776s，但v8工作量不同；每百万输运步的中位耗时仅改善约1.14%，
不能把中位时间3.98%的全部改善解释为算法收益。GPU平均76.8%–78.6%，CPU平均
16.4–17.5逻辑核，RSS峰值1.16GiB，系统最低可用11.16GiB。完整报告与图：
D盘`beta5_adaptive_v8_bounded_continuation_20260912/run/performance_5seeds/ACCEPTANCE_REPORT_CN.md`。
N=2/N=32、单端固定种子数组/trace、真实双端EM/radio等门禁通过；这不是500例
统计验收，也未完成v8的100PeV或PSR净收益测试。下文排队记录保留为实施历史。

本地v5的100PeV最终为3540.618s（59.01min），比v4慢10.97%。
输运步数反而减少2.83%，驻留物理波前增加34.64%；GPU服务等待从365.417s
减少到270.372s。因此不能把回退只解释为shower较大或GPU等主机。
完整输出检查和计数对账通过，但局部能量账本不完整，不是全shower能量闭合验收。
报告保存在D盘`beta5_adaptive_v5_work_quantum_20260912/acceptance_uhe_20260912`。

PSR的v7首例：纯OpenMP51.796s、双端60.442s，仍没有净收益。
第二例受另一GPU工作流影响，明确剔除；不使用它计算两例均值。
首例CPU最长调用约750ms；GPU一个packet最多执行两次，仍可能早早结束后
等待CPU返回。v8只针对这个可观测的剩余限制，不假设它能消除所有分端开销。

## 实现与边界

`corsika/accelerator/em/detail/IndependentSubshowerPump.hpp`中，一个GPU driver
packet可以包含最多16次原有驻留调用。不是16个并行kernel，也没有把逻辑调用
合并成一个统计数。粒子仍在本端常驻队列演化，完成记录等待主协调器消费。

```text
协调器：预留history bank → 提交GPU packet → 执行OpenMP调用
                          │                     │
GPU driver：调用1 → 调用2 → … → 调用≤16          │
             每次完整返回后检查边界              │
                          └──── 主协调器逐份提交 ─┘
```

在安全驻留边界检查任一条件即交还：

1. 本端光子/轻子都已排空，或上一调用没有进展；
2. 已保留结果达到64MiB，或已完成16次调用；
3. 主协调器原子通知需要交还，例如CPU无工作或要处理scalar fallback；
4. packet已达到按CPU实测吞吐估计的工作时长。

时间是软边界，不抢占正在运行的调用，不改变物理步长。检查时刻之前的单次
驻留调用可以超出估计时长，必须记录实测，不应宣传成硬实时上限。
64MiB是**先前结果的额外保留预算**；最高保留量允许再加一份正常调用结果，
而该结果继续受原有backend容量限制。不是“总RSS≤64MiB”。计算向量capacity，
不仅计算size；不为凑满利用率无限扩大结果邮箱或CPU队列。

### history与输出

- 所有history范围仍由主协调器预留，GPU driver不调用全局分配器。
- 预留单调的统一bank，每个潜在调用一个slot；光子和轻子交替不能返回较早的
  per-species bank。按原过程的history上界取slot stride，未用区间允许留空。
- 检查乘法、加法溢出、范围重叠和每次调用的租约使用量，非法范围在GPU提交前终止。
- driver只读本端队列和提交时控制器快照，不访问CPU栈、waiting向量、writer或
  PROPOSAL fallback。迁移仍只操作已归还的安全状态。
- 每份完成结果仍由主协调器按sequence提交一次；后续调用异常使整个shower失败，
  不能把第一份成功伪装成整个事件完成。
- CoREAS/ZHS仍在各端积累，在原结束位置汇总，本轮未改射电公式或定点算法。

### 可审计计数

```text
CUDA逻辑调用 = driver packets + autonomous continuations
autonomous continuations ≤ 15 × packets
maximum_cuda_packet_calls ≤ 16
sum(cuda_continuation_stops) = packets
```

继续记录逐端实际输运步、驻留物理波前、输入批量直方图、完成原因和迁移字节。
新增保留预算/峰值、最大packet调用数、停止原因。`cuda_result_service_delay_ms`
从最后一次GPU调用结束计算；`cuda_completion_buffer_delay_ms`可包含后续GPU
有效计算，不能全部称为GPU空闲。窗口重叠仍不是设备kernel级重叠证明。

## 安全验收证据

PSR隔离Release构建SHA-256：
`2f3b52f50e4095b2bc26a19c1b7a3bf75c5f1c72f8ac63d16f10718ef61e217a`。
三个CTest通过：`testIndependentSubshowerPump`、`testAdaptiveSubshowerControl`、
`testCooperativeDriverAffinity`。

新增/扩充`tests/accelerator/testIndependentSubshowerPump.cpp`：

- 主协调器阻塞在模拟CPU调用时，GPU确实能执行至少四次驻留调用；
- 16调用的真实packet边界被执行，不能用事后截断计数伪装；
- 250ms的首个调用越过初始200ms软预算后，不继续提交下一调用；
- 1byte测试保留预算令有诊断向量的完成记录立即交还；
- 统一history bank单调，bank溢出在GPU启动前拒绝；
- 混合光子/轻子、迁移、空队列、两端溢出、异常销毁、exact-once终止身份和计数对账。

这些测试在PSR上分别通过ASan＋UBSan＋leak检测和ThreadSanitizer。
TSan使用`setarch x86_64 -R`仅关闭该测试进程的ASLR，避免已知地址布局启动冲突；
没有改变系统设置。测试编译/运行绑定0–3；测试二进制为隔离stage下的
`pump-boundaries-sanitized`与`pump-boundaries-tsan`。它们验证host调度与所有权，
不替代CUDA设备race检查、真实物理replay或大样本验收。

## 已提交的真实设备检查

### PSR：T400＋后130核心

隔离stage：`build/psr-adaptive-v8-bounded-continuation-20260912`。
持久服务：`c8-psr-adaptive-v8-tests-20260912`。
数据：`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v8_bounded_continuation130_20260912`。

启动时等待山体v12工作流PID238618（创建时间1789149851.39），然后核验GPU连续
空闲30秒；只等、不停止该工作流。绑定382–511，130线程；24/32GiB内存软/硬
保护，至少32GiB可用；GPU预算70%。真实测试还逐次检查外部GPU进程，竞争样本
不参与性能结论。等待文件不是进程存活证据，需同时查systemd/PID。

顺序为单端N=2固定种子数组/trace对照v7、CLI失败门禁、真实EM＋radio N=2、
N=32生命周期，再进行两种子100TeV质子、47°/180°的纯OpenMP/双端交替短测。
短测通过后还需至少五种子和高能测试，不能仅凭两例推广服务器净收益。

### 本地：RTX4060＋20线程

v7先完成五种子Fe100TeV的同二进制单CUDA/双端对照，不被本轮替换。
v8文件冻结到`build/adaptive-v8-bounded-continuation-20260912/overlay`。
`c8-adaptive-v8-build-20260912`等待v7工作流PID1744497及创建时间1789148714.72、
10个完整事件和正确性门禁、GPU空闲及至少12GiB可用内存后，复制v7的独立源码
并覆盖本轮文件，2并行构建；不在v7计时期间编译。

`c8-adaptive-v8-pilots-20260912`在构建成功后跑相同的单端回归、生命周期与
真实EM/radio门禁，再做五种子Fe100TeV、单CUDA/双端交替测试。
输出在D盘`beta5_adaptive_v8_bounded_continuation_20260912/run`。
GPU预算70%，双端20线程，单CUDA调度1线程，每例独立进程，RSS限制5GiB，
系统至少保留4GiB；暂不自动追加100PeV，先确认短测净收益和安全。

本轮还修复了小型性能汇总脚本对单端空`cooperative`字段的处理，避免对照
运行正确但汇总崩溃；只有已完成的种子对进入中位时间和探索性bootstrap。
少种子的bootstrap不覆盖硬件温度/频率漂移、竞争或跨机器不确定性，不能作为
正式性能认证。没有更换默认推荐，没有推送远端仓库或改动山体WIP。
