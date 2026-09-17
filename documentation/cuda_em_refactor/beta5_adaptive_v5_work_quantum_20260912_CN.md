# adaptive-v5：按有效工作校准双端量子

## 状态与边界

这是显式 `--kokkos-cooperative-policy adaptive` 的实验优化，不改变默认
legacy 策略、单 CUDA、单 OpenMP、标量 PROPOSAL、物理 kernel 或山体 session。
没有替换生产 install。动态分配/history 租约会变化，不要求新旧 adaptive
同 seed 产生相同 shower tree。最终目标仍是相对最有效的单端得到净收益，并
适配 GPU 较强和 CPU 较强的机器；目前**未证明达到该目标**。

## 已完成 v4 数据揭示的问题

- 本地 100 PeV：v4 3190.54 s，v3 4012.21 s，旧独立双端 2176.26 s。
  v4 两端处理了 3,316,785,422 个输运步，GPU 完成到领取累计等待 365.42 s。
  累计等待与 CPU 运算重叠，不能直接从总时间扣除。
- PSR 同机同二进制 100 PeV：v4 双端 3579.77 s，纯 OpenMP130 为
  1899.88 s；CPU/GPU 都有负载不等于有效加速。
- v4 将 OpenMP 驻留调用限制为最多 8 个波前；一些 GPU 通道的
  `full_batch_observations=0`。它们没有填满最大 arena，并不代表没有可用的
  吞吐样本。此前会让稀疏尾部反复覆盖速度估计。
- 每次输入会在多个 job 中重复计数。比较粒子工作量使用
  `transport_records`，不将 `input_particles` 当作独立粒子总数。

## 修改

1. **有效样本不等于显存满容量。** 吞吐校准采用
   `inputs >= 7/8 * min(input_limit, 4096)`。得到有效样本后，稀疏尾部不再
   改写吞吐估计。首次有效样本丢弃此前的未校准历史；之后按实际耗时衰减并
   累计工作量/时间，而非每次调用等权。4096 不是 arena 或输入上限。
2. **两端使用相同驻留波前上限。** 光子 16、轻子 1024；冷探针仍为 CPU8、
   GPU64（光子服从16）。实测速度份额决定软量子：
   `clamp(1000 * share^2, 50, 800) ms`，设备对换时策略对称。若 GPU 更快，
   主线程的 OpenMP 量子另受近期 GPU 时长的响应预算约束。该规则是待验证的
   调度启发式，不是最优性保证或硬实时截止。
3. **OpenMP 宽度可增长。** 完整前缀即使因种类结束只推进了2–7个波前，也
   可用于每波前成本判断；不再要求恰好推进8次才能扩大宽度。仍使用迟滞，
   单波前持续过长才缩小宽度，用户容量和内存门禁保持不变。
4. **GPU 大波前保留原最小批量 checkpoint。** 大输入衰减到4096以下可
   返回，交由调度器与新产生的另一种粒子合批；一开始就小于4096的尾批次
   仍以 minimum=1 排空，不强迫每推进一步就返回。
5. **避免反复搬运小尾部。** 迁移至少256个状态；两端都有工作时需要预测
   关键路径改善超过25%及测得复制成本。仍然禁止读取在途 GPU 队列，所有
   粒子身份保留，CPU 栈、history 预留和结果回调只由协调器操作。
6. **增加有界退出原因计数。** 每端/每种类的 `completion_reasons` 按
   `completed, minimum_batch, workspace, history_lease, wave_lease, other`
   顺序记录，每个 job 恰好一个原因。与提交/回收和输入直方图对账。没有
   增长的逐 job 内存日志；旧 `full_batch_observations` 字段保留，但 v5
   `adaptive_calibration_rule` 明确其有效样本口径已不同于 v4。

## 验证记录（持续更新）

- 主机调度/所有权单元测试通过；ASan、UBSan 和泄漏检测通过。
- 本地独立 Release 构建成功，三项调度/亲和性 CTest 通过。
- 与冻结 v4 比较：help 完全一致；标量 CPU、单 CUDA、单 OpenMP 的 N=2
  物理输出通过逐项比较；CUDA/OpenMP decision trace 完全一致。
- 双端 N=32 已结束；真实 EM＋CoREAS/ZHS N=2 fixture 通过。
- Fe100TeV 两个种子的同二进制 legacy/adaptive 交替测试在运行。PSR
  同步独立构建、回归与两种子100TeV纯OpenMP/双端测试。**尚未性能验收**。
- 本地内存保护：至少4 GiB系统可用，测试进程树RSS上限5 GiB，70%显存。
  编译进程组单独限制8 GiB。不修改其他项目的 IDE 或进程。

本地源码快照：`build/adaptive-v5-work-quantum-20260912/source`；从 HEAD
归档后只叠加本次双端文件与测试，CONEX 复用既有源码，PROPOSAL 数据、
TAUOLA 和 Pythia8 复用既有同版本依赖，不生成新物理表。
数据在 `D:/CorsikaData/corsika_validation_results/beta5_adaptive_v5_work_quantum_20260912`。
本地服务为 `c8-adaptive-v5-build-20260912` / `c8-adaptive-v5-pilots-20260912`。
PSR 数据在 `/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v5_work_quantum130_20260912`；
服务为 `c8-psr-adaptive-v5-build-20260912` / `c8-psr-adaptive-v5-tests-20260912`。
PSR 运行绑定逻辑CPU382–511（后130个物理核每核一个硬件线程）；编译使用128并行。

先看短测的实际步数、退出原因、批量和总耗时，再决定是否进行100PeV长测。
少量种子不能替代500例物理统计；单端小样本输出回归不证明大样本性能不退化。

## 完成的短测及后续候选

本地两个 Fe100TeV 的 v5 为 **66.272 / 65.004 s**，v4历史值为
67.687 / 66.247 s，差别约2%，不足以声称稳定加速。v5实际输运步数为
75,041,221 / 69,666,741；GPU jobs为331/391，结果服务等待2.955/3.098 s。
同一次新二进制的 legacy 对照为94.383/107.961 s（72,875,779/71,672,136步），
但 legacy本身也是动态调度，不能将两例1.54倍描述性比值推广为稳定中位加速。
两例退出原因均无history耗尽/未知原因；每批退出原因与job数对账通过。

PSR v5的100TeV两种子：纯OpenMP **51.248/57.309 s**，双端
**78.708/74.515 s**，仍无净收益。seed1已将CPU调度job降到621，但CPU
轻子驻留波前仍为39,068；单OpenMP是8,269。增加驻留租约后，CPU minimum=1
会用大量130线程同步来排空小尾部。不能只统计host job数就认为碎化已消失。

因此主工作树继续发展为 **adaptive-v6-tail-grain**：只修改双端OpenMP的
无损checkpoint粒度，设 `minimum=min(gpu_min_batch,max(1,input_count/4))`；
CPU跨种类合批也使用有效batch阈值，而不要求先填满整个64k arena。
这不是能量cut，所有未结束状态仍保留。v6在PSR独立目录编译/测试：
`build/psr-adaptive-v6-tail-grain-20260912`；对应测试数据为
`psr_t400_adaptive_v6_tail_grain130_20260912`。目前不声称v6通过性能验收。

本地100PeV已使用**冻结v5**二进制启动，20线程、70%显存，原种子和物理参数；
`c8-adaptive-v5-uhe-20260912`，输出在本地v5目录的`high-energy/`。
它不会因为主工作树继续迭代v6而切换版本。启动后的短主机单元测试编译有数秒
CPU重叠，此次长测视为探索性性能样本；正式门限仍需无干扰多种子对照。
无生产install更新，无山体修改，无GitHub推送。
