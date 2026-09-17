# adaptive-v9：标量前台期间的独立 GPU 续跑

## 状态与范围

这是实验性双端调度修复。只改变显式 `adaptive` 策略，不改物理 kernel、
PROPOSAL 表、cut、thinning、射电投影、单 CUDA、单 OpenMP、legacy 双端或山体路径。
没有替换生产安装或推送仓库。真实设备的性能和物理验收尚未完成。

v7 本地五种子 Fe 100 TeV 全射电同二进制对照，中位时间为单 CUDA 74.060 s、
双端 20 线程 62.776 s；所有五对都较快，但动态调度允许工作量不同，样本仍少。
这不能外推 100 PeV；v5 的该高能例实际仍为 3540.618 s，比 v4 慢 10.97%。
v8 的五种子测试及 PSR 测试是独立冻结版本，不被本次源码修改替换。

## 已复现的限制

`PhysicalAcceleratedEmRouter::advanceIndependentSubshowers()` 的
`yield_to_scalar` 表示消费完成结果时，已有粒子返回标量栈。`HybridCascade`
随后让唯一的主协调器处理这些粒子。它**不表示** GPU 中另一组粒子的计算
依赖标量处理完成。

v8 把这个标志同时用于请求 GPU 交还，并且不允许此时启动自主续跑。
即使 GPU 还有独立的驻留粒子，也只能执行已经提交的一次调用，然后等待。
测试固定 256 个模拟 GPU 粒子、空 OpenMP 队列，让主线程退出 `advance()`
模拟标量工作，再等待 GPU 的第四次调用：冻结 v8 头文件下测试退出 1，报
`GPU stalled during independent scalar foreground work`。

这是可控 host 调度复现，不是根据利用率猜测，也不是电磁物理 oracle。
v5 100 PeV 的标量 stepper 时间约 359 s 说明此路径值得检查，但不能把这段
时间直接当作可回收加速：其中可能已经重叠，GPU 也可能没有可用工作。

## 修改

`IndependentSubshowerPump.hpp` 由协调器发布两项原子状态：

- 主机正在让出控制权处理标量工作；
- 主机确实缺少工作，需要 GPU 到安全边界后交还。

```text
GPU 驻留队列 ── 调用 → 调用 → 有界完成 packet
                  与主机独立推进
主协调器       ── OpenMP 工作 或 标量栈/fallback ── 领取结果
```

adaptive 模式下，只有 OpenMP 队列为空且没有标量前台工作才请求交还。
完成回调可能新产生标量工作，因此在消费 GPU 结果之后重新发布状态，再决定
下一次提交。driver 只读取原子标志；不捕获或调用栈、writer、fallback 或
history 分配回调。legacy 保留原逻辑。

v8 的 16 调用、按实测 CPU 工作时长、64 MiB 额外结果保留预算继续有效；
预算仍允许加一份受 backend 容量限制的正常结果。history bank 仍在提交前
统一预留并检查溢出，逐结果仅提交一次。不抢占 kernel，不丢弃返回粒子，
不在另一个主机线程执行 PROPOSAL/强子模型。

新增统计 `subshower_cuda_foreground_packets` 和
`subshower_cuda_foreground_continuations`，表示在标量让出状态下提交的调用。
它们不等于实测 CUDA kernel/标量重叠时间；metadata 明确写出该限制。

## 测试与待验收

首个标量前台复现修复后，PSR 隔离测试通过 ASan＋UBSan＋leak 检测及 TSan。
TSan 用 `taskset -c 0-3 setarch x86_64 -R`，不改变系统 ASLR 或其他任务。
另外增加实际结果 consumer 才产生 scalar-yield 的状态转换测试，避免仅覆盖
启动前就设置标志的理想情况。更新后的完整测试再次通过 ASan＋UBSan＋leak
检测及 TSan；两次实测退出码均为0，使用相同的隔离stage，不替换应用二进制。

后续门禁：冻结新源码，独立 Release 构建；单端固定种子数组与 decision trace
保持一致；N=2/N=32 生命周期、真实 EM＋radio 双端计数对账；五种子同二进制
单 CUDA/双端交替短测；再进行 100 PeV 与 PSR 130 核测试。没有通过这些门禁
之前，不宣称 v9 更快或适配完成。计时期间不同时编译，不停止无关生产任务。

## 本地隔离排队记录

v8五对已完整结束并通过文件/参数/计数对账后，冻结v9小型overlay到
`build/adaptive-v9-foreground-20260912/overlay`，基于v8独立source复制，未夹带
主工作树中的山体修改。GPU连续空闲10秒且可用内存12.37GiB时启动构建。

- `c8-adaptive-v9-build-20260912`：2并行，6/8GiB内存软/硬门限，无install。
- `c8-adaptive-v9-pilots-20260912`：等待该构建及CTest成功，随后执行单端回归、
  真实双端EM/radio、生命周期及五种子单CUDA/20线程双端交替短测。
- 输出：D盘`beta5_adaptive_v9_foreground_20260912/run`；测试4/5GiB软/硬门限，
  保留至少4GiB系统可用内存；不自动追加100PeV。
- 本地冻结pump SHA-256：`751215dccbf37ebaf5e6e79e4b4b9ee7eb63d32013f0eef96c7e72f245b1b280`。
- 本地冻结测试SHA-256：`08d04bc60c89cf29c84d65988c2110c9847b167c6e8f32d4eeca1b39fe2480a9`。

这是启动记录，不是构建/真实性能完成证据。PSR v8仍等待已有山体v12工作流和
GPU空闲，未停止该工作流；v9在PSR仅完成host sanitizer测试，尚未启动完整应用对照。

## 后续验证链

本地v9 Release现已构建完成，三个CTest、单端固定种子数组/decision trace、
N=2/N=32及真实双端EM/radio门禁通过，五种子Fe计时正在执行。高能验证不再
只引用旧版单CUDA：已排队等待Fe输出验收后，用同一v9归档binary依次跑单CUDA
和20线程双端的100PeV质子（47°/180°、seed2026110001、emthin1e-6、自动
max-weight50、全射电）。计划与保护记录位于D盘
`beta5_adaptive_v9_uhe_paired_20260912/TEST_PLAN_CN.md`。此处不预言结果。

监控新增可选GPU独占门禁，查询失败不当作空闲；仅终止新诊断自身进程组。
五项host mock测试通过，默认guard行为不变。`run_validated_adaptive_uhe.py`
支持CUDA/adaptive/legacy选择，命令构造测试确认原物理参数不变、错误能量
直接拒绝，并原子记录成功/失败的终端状态，验收状态另行标记。

PSR的v9隔离源码已冻结。`c8-psr-adaptive-v9-build-20260912`等待v8测试成功
且GPU空闲后，在0–125上128并行编译；不夹在v8计时过程中。
`c8-psr-adaptive-v9-tests-20260912`等待该构建和GPU空闲，使用后130个物理核
对应的382–511编号，执行五种子纯OpenMP/双端交替对照，GPU竞争样本拒收。
服务内存24/32GiB软/硬限制。输出目录为服务器
`psr_t400_adaptive_v9_foreground130_20260912`。排队不等于构建或性能通过。

## PSR 回退的进一步排除检查

直接读取v7首个未受竞争影响的配对：

| 时间口径 | 纯OpenMP/s | 双端/s |
|---|---:|---:|
| 进程端到端 | 51.796 | 60.442 |
| shower内部 | 42.934 | 51.024 |
| 两者之差（启动/关闭等） | 8.862 | 9.418 |

增加的8.647s中约8.090s发生在shower内部，不能解释成只是CUDA首次初始化。
双方OpenMP工作区上限均为65536；双端实测input_limit已增长至65536，因此
也不是仍被锁在早期16384的小arena。CPU端实际处理约1.019亿步（单端总计
1.097亿步），CPU调用累计49.051s，而单OpenMP光子＋轻子调用约41.604s。
双端CPU驻留波前增加、过程组合/每步成本改变，以及协调成本仍需结合v8/v9
数据进一步分解；这些总量不能证明唯一根因。**没有据此盲目扩大内存或修改物理。**

山体v12工作流已自然结束，PSR v8现已进入真实回归与warmup，v9仍等待其后。

## 本地短测完成更新

v9五对Fe100TeV现已完成，输出逐批检查通过。单CUDA中位74.063s、双端61.245s，
描述性1.209×（耗时缩短17.31%），五对均更快；但v8双端中位60.280s，不能
宣称v9已进一步加速。双端工作量与v8不同，仍需要目标高能测试。
RSS峰值约1.22GiB、最低系统可用11.05GiB，所有监控及计数门禁通过。
本地binary SHA-256为`eaaacf2a6ec4b0aaf9b85b75be10432e34a821e953fa8339090215efa84fdbc5`。
完整报告在D盘`beta5_adaptive_v9_foreground_20260912/run/performance_5seeds/ACCEPTANCE_REPORT_CN.md`。

100PeV同二进制单CUDA基线已在GPU空闲和完整短测验收后启动；双端等待其成功
结束。两者都还不是已完成高能性能数据。PSR v8首个有效100TeV配对为纯OpenMP
51.403s、双端64.137s，仍无收益，第二对及v9五对需要继续检查。

## PSR 五对完成更新

v9服务已正常退出，10/10计时完整、GPU竞争检查通过。纯OpenMP中位51.378s、
双端67.026s，双端耗时增加30.46%；服务器仍不推荐adaptive作为性能方案。
输出为`psr_t400_adaptive_v9_foreground130_20260912/PERFORMANCE_RESULT.json`。
这一结果不以本地Fe的收益覆盖。下一步只在隔离v10中检验自主调用私有成本
估计未刷新的问题，详见[v10诊断](beta5_adaptive_v10_continuation_learning_20260912_CN.md)。
