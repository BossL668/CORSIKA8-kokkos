# 自适应双端退化诊断与 v3 协调器响应预算

## 结论边界

adaptive-v2 的高能表现不合格，不能作为优化成功发布。修改仅影响显式
`--kokkos-cooperative-policy adaptive` 的空气双端模式；保留单 CUDA、单 OpenMP、
标量 PROPOSAL、legacy 独立双端及山体路径。没有改变物理公式、表、thinning、
max-weight、磁偏转常数或射电计算。v3 仍须真实性能和统计验收。

## 正确的历史基线

同参数 100 PeV proton（theta47、phi180、emthin1e-6、seed2026110001、
20 OpenMP 线程、70% GPU 预算、完整射电、默认 max-weight）的已完成记录：

| 模式 | 端到端耗时 | 状态 |
|---|---:|---|
| 历史单 CUDA | 4369.329 s / 72.82 min | 完成 |
| 较早的成对批次双端 | 3407.670 s / 56.79 min | 完成，但不是最佳独立子队列基线 |
| 独立子队列双端 | **2176.256697 s / 36.27 min** | 完成；GPU 平均利用率 85.59% |
| adaptive-v2 | **至少 4915.57 s / 81.93 min** | 主动中止，不能当作完整事件耗时 |

最佳基线二进制 SHA-256：
`da71a04c4cf52e3c0c0b9bf749dd2b0da9cfd47401c16cc0c348b60e3744e926`。
数据位于 `beta5_subshower_queues_20260911/pilot-100pev`。
adaptive-v2 中止前按采样间隔加权 GPU 利用率约 25.68%，CPU 平均约 18.12 个逻辑核。
数据位于 `beta5_adaptive20_Fe100TeV_proton100PeV_20260911_v1`，明确标记 incomplete。
这些是历史生产/试跑证据，不是同机交替多种子受控性能结论。

## 从代码与计数器确认的问题

1. **相同的 200 ms 批次目标忽略主机职责不对称。** CUDA 独立 driver 只执行
   一个已经准备好的 job；CPU OpenMP 在协调器线程上执行同步 job。GPU 提前完成后，
   回收结果、处理返回物和续发仍需协调器。CPU 大 job 不返回就无法续发，
   “两个独立队列”并不意味着 GPU driver 可以无限自行分配 history 和写结果。
2. **把 arena 容量误当成连续执行量。** v2 恢复普通 OpenMP 的 65536 容量，
   `prepare()` 随即取满该容量；旧本机 legacy 容量是 2048。
   即使将波前数降到 1，也可能是很长的 65536 粒子批次。
3. **迁移模型缺少稳定性约束。** 旧估计按瞬时积压和平均 steps/ms 均分预计负载，
   只设 1 ms 加搬运成本门槛，也允许搬走 GPU 下一批 staging 工作。
   在旧 `count <= load_gap/(cost0+cost1)` 的理想线性条件下，donor 减少量确实
   等于预测关键路径收益，不能把这一代数式本身误报为错误。
   真正不足是 steps/ms 不是未知子 shower 的总剩余成本，波动下缺少相对迟滞和
   保持有效批量的约束。v3 显式使用 max 成本公式，但实质改变来自这两个约束。

在已完成的 v2 五例 Fe100TeV 中，CPU 平均批次持续 59–63 ms，最长 285–472 ms；
GPU 完成后等待回收累计 **22.17–25.76 s/event**，约占协调器循环时间的 37–41%。
这是实际 `cuda_result_service_delay_ms`，不是仅看 nvidia-smi 的推断。
这段等待与 CPU 有用计算重叠，**不能简单从事件时间扣除后宣称加速**。
100 PeV 的 v2 未正常结束，没有它的最终累计计数，不伪造完整分解。

## v3 修改

- 保留 CPU 的大 arena、待处理队列和 20 个执行线程；单次输入 prefix 按实测
  input/wave 耗时调整。冷启动最多 2048，随后可增长至原 arena 容量，并非永久
  把服务器限制在 2048。
- GPU 在途时，CPU 响应目标为近期 GPU job 耗时的 5%，当前限制在 2–10 ms；
  GPU 不在途时放宽至 200 ms。目标只决定批次，不抢占 kernel、不截断物理 step。
  最小输入与单波前不可抢占，故它不是硬实时保证。
- 同时限制外部输入和 backend 从 resident queue 自动补入的数量；只限制外部
  vector 会被 backend 自动补满，达不到预期。调用结束及异常时恢复原设置。
- 跨端迁移比较 `max(两端预计剩余时间)` 在迁移前后的变化，计入搬运成本；
  两端均有工作时增加 10% 预测负载迟滞。CPU 已有工作时不为了微小差值搬空
  GPU 的 4096 staging 量；CPU 空闲时仍可接手有收益的尾部。
- metadata 的 adaptive 策略标识改为 `adaptive-v3-service-budget`；legacy 和
  单端字段不增加此标识。沿用实际 service delay、CPU epoch、迁移、队列峰值计数。

这不是对 CPU 利用率设置百分比上限。双方仍可连续计算，只是协调器必须有足够
频繁的安全服务机会。未来若改成 GPU driver 自主多 job，则必须另做有界结果邮箱、
history 预租约与异常排空设计；本轮没有偷偷加入这些并发所有权变更。

## 验证与产物

- 调度单元测试：混合 PID、在途禁止访问、唯一完成身份、尾部迁移、输入 prefix、
  CPU 大 arena 不自动补满、正常及异常恢复限制、旧 legacy 不调用新输入限制。
- `run_adaptive_service_pilot.py` 使用隔离构建和冻结二进制：固定 seed 单端 N=2
  输出及 decision trace、adaptive N=32、真实 EM＋射电 N=2，再交替比较两例
  Fe100TeV 的 legacy/adaptive，随后使用正确历史参数复测 100 PeV。
- 任何失败停止后续验收，不替换生产程序。全过程可用内存底线 4 GiB、测试进程
  RSS 上限 5 GiB、GPU 预算 70%；通过 systemd 用户服务运行。
- 本次目录：`beta5_adaptive_v3_service_20260911`，含修改前调度源码、编译日志和
  `run/STATUS.json`、逐事件命令/guard telemetry。最新真实结果以该目录为准。

已通过 Release 构建、三项调度/亲和性 CTest、ASan/UBSan（含泄漏检测）的调度
测试。真实固定种子单端 N=2 的物理输出、CUDA/OpenMP decision trace 均与冻结
基线一致，帮助文本只允许已有 adaptive 选项和 citation 源码行号差异。
adaptive N=32 完成；真实 EM＋CoREAS/ZHS 的 N=2 能量通量残差分别为
`5.56165e-9`、`5.45529e-9`，不是强子完整 shower 账本证明。

初步短事件结果（v2 是前一轮保存值，v3/legacy 是本轮同二进制交替测试）：

| Fe100TeV seed | v2 时间 s | v3 时间 s | v2 GPU 结果等待 s | v3 GPU 结果等待 s | v3 GPU 平均利用率 |
|---|---:|---:|---:|---:|---:|
| 85000001 | 69.085 | 69.500 | 22.169 | 1.261 | 80.56% |
| 85000002 | 68.375 | 69.970 | 22.200 | 1.250 | 83.19% |

CPU 平均 epoch 从约 62 ms 降至 1.98/2.08 ms，单批最大值仍可达 102.7/44.2 ms，
因此不能把响应目标称为硬截止。队列/定点累计溢出为零，CUDA 提交全部回收；
试跑峰值 RSS 约 1.05–1.07 GiB。该结果确认协调器等待显著下降，**没有证明这两例
比 v2 更快**，总时间基本持平；不能把更高利用率直接当作净加速。

本轮同二进制 legacy 两例为 93.012/98.412 s，v3 为 69.500/69.970 s；这两例
v3 比本轮 legacy 短约 25–29%。它与“比之前 adaptive-v2 更快”不是同一个结论，
也不是 100 PeV 高能加速结果；仅两例且动态调度不保证同一棵树。

100 PeV 已在持久服务 `c8-adaptive-v3-pilots-20260911.service` 正常完成，退出码0。
最终端到端时间 **4012.206470 s / 66.87 min**，未恢复36.27 min参考，
**v3高能性能验收不通过**。报告服务只刷新小型诊断文件，不控制任务。
冻结的新二进制 SHA-256 为
`22d64274c6c2fe6de3d815db2c0ba6a7889f33da47890e1fe2e82eb20cccddc1`。
最新进度和同机
legacy 对照见 `run/SERVICE_PERFORMANCE_CN.md`。不据单例、GPU 峰值或
“CPU 核数更高”宣布性能成功。

## 完成后的工作量核验

完整报告和图：
[acceptance_100PeV_v3/ACCEPTANCE_REPORT_CN.md](/mnt/d/CorsikaData/corsika_validation_results/beta5_adaptive_v3_service_20260911/acceptance_100PeV_v3/ACCEPTANCE_REPORT_CN.md)。
分析脚本：`validation/accelerator/accept_adaptive_service_uhe.py`。

| 指标 | 36.27 min 独立双端 | v3 | 新/旧 |
|---|---:|---:|---:|
| 实际加速输运步数，两端合计 | 2,891,986,825 | 3,287,619,200 | 1.137 |
| 加速末态薄化后子状态数 | 1,015,348,997 | 1,145,721,319 | 1.128 |
| 射电轨迹段数 | 2,463,084,218 | 2,792,521,691 | 1.134 |
| CUDA job 次数 | 11,190 | 50,834 | 4.543 |
| OpenMP job 次数 | 148,848 | 1,977,848 | 13.288 |
| CUDA 每 job 平均入口状态数 | 128,080.6 | 12,127.2 | 0.095 |
| OpenMP 每 job 平均入口状态数 | 737.9 | 146.9 | 0.199 |
| GPU 采样间隔加权平均利用率 | 85.59% | 87.05% | — |
| 进程 CPU 平均逻辑核当量 | 15.80 | 19.28 | — |

`gpu_particles`名称容易误导：report中它是router的photons_advanced与
leptons_advanced之和，二者累加的是`result.transport_records`；双端下包括
OpenMP。它不是去重粒子数，也不是仅GPU计数。`physical_secondaries`包含
薄化后保留的反应末态子状态及出射母粒子状态，不覆盖标量生成器的全部粒子。
`cuda/openmp_input_particles`则在每次job入口累加，可重复统计同一粒子。
现有输出没有逐端总transport_records或全history集合，不能恢复精确的逐端
真实输运步数或全shower去重粒子数。

总耗时增加84.4%，输运步数增加13.7%；按总步数描述性归一化后，每步墙钟成本
增加62.2%。不同树的过程/能量组合仍不同，因此这不是精确的因果时间分解，
但不能把退化仅归因于shower更大。OpenMP平均job约1.72 ms，job数量激增，
是上一次强调协调器响应而缩小输入prefix的实际代价。高利用率没有转化为吞吐。

### 代码对应的切分机制

- `AdaptiveSubshowerControl::hostTargetMs`在GPU在途时返回2–10 ms目标；
  `inputLimit`用`target/(8*ms_per_input_wave)`反推输入上限，最低上限256。
  这是可处理量上限，不保证队列有256个粒子，所以实际平均入口可以低于256。
- `IndependentSubshowerPump::prepare`将该上限同时应用到外部输入及resident
  队列输入，`execute`确实将其传入backend，因此并非只缩小一个无效CLI参数。
- CUDA仍逐job回收；协调器在CUDA空闲且队列非空时提交，未建立最小有效批量
  的持续合批条件。`job.minimum=1`是驻留波前退出阈值，并非证明启动批量达到
  4096的门禁。两者不能混淆。
- 成本估计仅按PID类别区分，`elapsed/(初始输入×波前数)`同时包含固定调度成本
  和增长/衰减中的粒子群；它不是可直接外推任意子shower剩余成本的量。
  固定开销在小job中占比增大可能进一步收紧输入估计，尚需批次时间线验证反馈。

下一步方向应是把完成通知服务与物理计算批量解耦、保持两端有效批量、在必要的
history/fallback/队列安全边界收取结果，而不是继续缩短时间片。需要同时增加
每端实际transport_records和批量直方图，防止再次把入口状态重复计数当作负载。
本轮仅验收和诊断，未修改调度/物理代码或替换生产二进制。

### 验收边界

全部7类Parquet逐批读取有效，地面输出3,949,164行；81天线的CoREAS/ZHS各
400点、无NaN/Inf、峰值不在首尾5点。提交/回收匹配、pending清零、队列和定点
溢出及native反解失败为0。进程树RSS峰值1.65 GiB，可用内存最低8.09 GiB。
这支持单事件运行完整性通过，不足以证明N>1无泄漏或物理统计等价。

新旧局部能量账本均明确`complete_coverage=false`、`accepted=false`；新残差
8.151%（旧8.715%），不能将其包装成完整能量闭合验收，也不能由这个未覆盖
标量/强子/薄化全部通道的账本直接断言能量损失。单事件profile/波形仅诊断展示，
不代替系综检验。本轮不自动恢复Fe生产或展开新高能样本。
