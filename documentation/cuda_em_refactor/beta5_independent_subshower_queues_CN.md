# beta5 空气双端：独立子级联队列

## 范围与状态

仅显式 `--kokkos-execution cuda-openmp` 的空气应用使用本次调度。
不改物理公式、随机数算法、cut、薄化、默认参数、山体 session，
也不更换 PROPOSAL 表。单端保持原来的同步波前 API。

目前为隔离构建/验收中的实验修改，不覆盖生产 install，不推送远端。
先前批次级独立版本的 100 PeV 单例已经完整结束（3407.670 s），
不是本次实现的成绩。该事件平均约 6.28 核，CPU 先结束后等 GPU 的
调用窗口累计约 1368.696 s；这正是本轮要去除的批次屏障。

## 控制流程

```text
唯一协调器：分配 history 区间、处理输出/CPU fallback
    │
    ├─ CUDA 驱动线程：光子或轻子工作段 → 本端次级保留
    │        └─ 单一 future 完成槽（有界，不直接访问共享栈）
    │
    └─ OpenMP 初始化线程：本端光子/轻子交替推进
             └─ GPU 未返回，也可连续处理多个工作段

协调器仅领取已就绪的 GPU 结果，随后再次提交本端工作
两端皆无粒子、在途工作、未提交结果或 fallback → 最终输出合并
```

不再为一个光子/轻子批次调用两个后端后强制 `get()` 汇合。
GPU future 只在 `ready` 后领取；CPU 每次前进一个工作段便可再次前进。
两端自己选择光子/轻子队列，不要求两边处于相同的粒子种类或代数。
CUDA 每段仍使用原来的驻留内核；OpenMP 8 个轻子波前是交接检查点，
不是只允许在 GPU 空隙执行，也不是物理截断。

CPU 没有本端任务时使用最多 1 ms 的完成查询等待，避免忙轮询。
这不是 CUDA/CPU 批次屏障。主机长工作段、输出和 fallback 仍可能延迟
下一次 CUDA 提交，因此不承诺两端永久满载或无任何调度开销。

## 所有权、内存和失败

- history 由协调器从原栈分配器预留，跨端区间递增且不重叠，检查溢出。
- 工作线程不保存/执行输出回调，不修改 Stack、PROPOSAL 或强子模型。
- GPU 每次最多一个在途任务/完成结果，主机待发队列有界。
- 同种类继续粒子和异种次级留在原端；沿用既有驻留/回传策略，
  本轮没有额外宣称“所有同种类 continuation 永不回传主机”。
- 每端每 PID 类别的主机暂存粒子上限为该端 batch 容量的四倍。
  执行端 cross-species arena 溢出的粒子优先进入本端主机暂存队列；
  本端满时可进入另一端有界待发队列，确实都满才返回原标量 spill 通道。
  没有完成任何波前的分配拒绝不重新入队，避免对同一个无法分配的批次无限重试。
  不丢粒子，不扩大为无界缓存，不修改执行端原有内存门禁。
- 仅安全边界迁移未执行粒子，完整搬运 history/parent/step/random state。
  默认保留有效 GPU 波前，不强制各分一半，也不来回搬运极小尾部。
- specified fallback 在协调器上执行，可在另一端仍运行时处理，
  不再强制等两端 EM 全部排空；生成的粒子经原 scalar/router 接口继续演化。
- 完成序号保证每段结果只提交一次；原始整数 profile/射电只在终态合并。
- 任一端或输出回调异常时，停止正常提交、排空当前驱动再释放资源，
  禁止把失败 shower 标记完整。
- `beginShower` 要求无待处理工作；事件结束后销毁子级联队列，复用原物理表。

## 源码位置

| 文件 | 职责 |
|---|---|
| `corsika/accelerator/em/detail/IndependentSubshowerPump.hpp` | 有界队列、独立推进、history 租约、结果领取及尾部迁移 |
| `corsika/accelerator/em/detail/SubshowerCallbacks.hpp` | 仅在主机进度调用期间借用的分配/输出回调 |
| `src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp` | 空气双端实例启用 pump，终态输出及生命周期门禁 |
| `corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp` | 空气调度入口及主线程输出/fallback 交接 |
| `IAcceleratedEmBackend.hpp` / `KokkosEmBackend.*` | 可选子级联协议；单端默认不启用 |

router 的旧 photon/lepton 结果消费代码只抽成两个共享函数，没有重写公式。
同步波前兼容入口仍供局部验证使用；空气双端生产入口走新子级联协议。

## 新诊断字段

`accelerator.cooperative` 增加：

- `independent_subshowers`
- `subshower_cuda_submissions` / `subshower_cuda_commits`
- `subshower_openmp_epochs` / `maximum_host_epochs_per_cuda_job`
- `subshower_tail_migrations` / `subshower_host_queue_peak_bytes`
- `coordinator_idle_wait_ms` / `cuda_result_service_delay_ms`
- `subshower_requeued_spill_particles` / `subshower_cross_endpoint_spill_particles`

执行端原 `cross_species.particles_spilled_to_cpu` 计数含义仍是 arena 导出的粒子数。
新路径中其中一部分可能被上述主机暂存队列收回，实际执行标量 step 的数量
应看 `cpu_spill_steps_executed`，不能把“离开 arena”和“已交标量输运”等同。

`oversized_slices_2ms` 统计实际超过 2 ms 的 OpenMP 工作段，并非硬实时限制。
首个 100 TeV 调度试验漏加这一诊断计数（最大工作段已正常记录），后已补齐；
该旧试验中该字段的零值无效，不代表所有工作段短于 2 ms。

旧成对批次的 tail-wait 字段不再适合解释新路径，不能把其零值当作“零等待”。
`endpoint_window_overlap_ms` 仍只是调用窗口；实际内核重叠用 CUPTI 检查，
多段 CPU 时间线按每个真实区间求交，不把中间间隙算作 CPU 计算。

## 验收记录（持续更新）

- 主机调度控制测试已通过：人为阻塞 GPU，CPU 仍连续推进至少 4 段；
  精确一次输出、主线程分配/回调、双向接口支持的安全尾部迁移、
  history 冲突失败关闭；AddressSanitizer/UBSan 初次检查通过。
- CUDA+OpenMP Release 的空气及山体目标编译通过；未修改山体物理/session，
  本轮尚未另做完整山体事件的数组回归，不能把编译等同物理验收。
- 真实 PROPOSAL、20 线程 EM＋CoREAS/ZHS：`backend20/` 通过。
  连续两事件使用固定大小 workspace，两端均有工作、GPU 提交全部领取，
  profile、CoREAS、ZHS 非零；重复下载不重复累计。
  CUPTI 记录实际内核与四个主机工作区间交叠 2.843 ms，丢记录数为零。
- 单端 N=2：标量 PROPOSAL、Kokkos-CUDA、Kokkos-OpenMP 各 11 个输出
  数组逐项一致，无非计时 metadata 差异；CUDA/OpenMP decision trace 完全一致；
  规范化程序名后的 `--help` 字节一致。
- 原 1 GeV N=32 正常完成，但仅一例实际用到了 OpenMP，不能作为充分的双端
  生命周期证明。补做 100 GeV 光子 N=32：每例两端均工作，全部关闭/完成，
  无 queue/profile/radio 溢出，workspace 恒定为 56,815,550 bytes；
  采样峰值 RSS 796.7 MiB，主机队列峰值 220,752 bytes。
  这是有限事件数/测试规模下的内存稳定性证据，不是全能区无泄漏证明。
- 高能性能/500 例统计：尚未验收；不以调度单测代替物理统计。

结果集中于 D 盘 `CorsikaData/corsika_validation_results/beta5_subshower_queues_20260911`。

## 首个 100 TeV 全射电单例

质子，theta=47°，phi=180°，emthin=1e-6，seed=2026110001，默认 max-weight，
20 OpenMP 线程、4096 min-batch、70% 显存预算，与之前同类试验的输入一致。

- `timing-100tev/`：进程 122.085 s，shower 112.000 s，正常完整退出。
- GPU 提交/领取均为 4919；OpenMP 3446 段；单个 GPU 在途任务期间最多 218 段。
- CPU 工作窗口 48.184 s，与 GPU 调用窗口重叠约 46.441 s；后者不是 CUPTI 内核时间。
- 主机无工作轮询等待 5.736 s；GPU 结果就绪至领取累计延迟 6.223 s。
- 8 次安全尾部迁移，约 0.90 MB；主机队列峰值约 91.6 MiB。
- 峰值 RSS 1205.1 MiB；全过程系统可用内存保持 9.07 GiB 以上。
- 进程平均约 6.22 核当量，GPU 设备采样平均利用率约 67.9%；
  请求 20 线程不等于任意时段都能占满 20 核。

此前单 CUDA 约 101.956 s、上一轮成对独立驱动约 96.901 s。
本次单例没有显示净加速，不能作为生产性能通过；动态调度的 shower 树不同，
这些历史单次时间也不是同一二进制、交替多种子的受控比较。
去掉批次屏障已由调度测试证明，但收益仍需要 100 PeV 及多种子测试判断。

同日用重构二进制重新跑的单 CUDA 为 102.372 s（shower 93.024 s），
与该历史单 CUDA 时间接近；仍是一个种子，不是中位数回归门禁。
首个双端试验有 106,206 个 memory spill 回到标量，引起约 21.343 s
`set_nodes` 时间；全部标量步骤时间约 34.944 s（也包含其他标量物理）。
因此新增上述队列优先 spill 交接，尚需新完整事件验证，不能把这次 122 s
当作该补丁之后的速度。主机受控测试已覆盖两端均满、跨端接收、零进展拒绝，
ASan/UBSan 通过。

### 队列优先 spill 版本的复核

二进制 SHA-256：`da71a04c4cf52e3c0c0b9bf749dd2b0da9cfd47401c16cc0c348b60e3744e926`。
`backend20-queue-first`、`air-regression-queue-first` 与
`BOUNDED_ACCEPTANCE_QUEUE_FIRST.json` 对应该版本：
三种单端各 11 个物理数组及两个 decision trace 仍完全一致；
100 GeV N=32 每例两端有工作，全部完成/关闭，无队列或定点溢出，
workspace 仍为 56,815,550 bytes；峰值 RSS 797.5 MiB。
小型真实 EM＋射电测试 CUPTI 重叠 3.350 ms、零丢记录。

主机调度单测另故意产生超容量包，验证本端暂存、跨端暂存、两端满后
交回标量这三条路径；全部唯一身份被回收。零进展分配拒绝、GPU 异常、
输出回调异常和 history 租约冲突均失败关闭，没有无限重试或悬挂 worker。
这些是本次重构的有界正确性测试，不代替大样本统计。

该版本 100 TeV 复测位于 `timing-100tev-queue-first/`：进程 **85.054 s**，
shower **75.028 s**，GPU 提交/领取 **1788/1788**，OpenMP **3257** 段。
694,433 次 arena 溢出粒子被重新排队（不是唯一粒子数），其中 112,315 次
移到另一端待发队列；实际标量 memory-spill step 为 **0**。
`set_nodes` 降为 1.874 s；标量 step 合计 0.376 s。
CPU 平均约 10.71 核当量；GPU 完成至领取累计延迟 0.908 s，
无主机任务时轮询等待累计 24.864 s；queue/profile/radio 溢出计数为零。
同日单 CUDA 为 102.372 s，新双端此单例用时少约 16.9%。
动态 shower 树不同，且只有一个种子，尚不能宣称稳定净加速或统计等价。

## 后续 100 PeV 测试与保护

已通过用户级持久服务 `c8-beta5-subshower20-100pev-20260911.service`
启动同种子 100 PeV 质子（theta=47°、phi=180°、emthin=1e-6、默认 max-weight、
20 线程、70% 显存、完整射电）。使用 `pilot-100pev/c8_air_shower` 的冻结副本；
该运行不依赖临时终端，不覆盖历史数据或生产安装。
进程树 RSS 超过 4 GiB、系统可用内存低于 4 GiB 或运行超过 7200 s，
监控器仅终止这次测试进程组。报告服务在结束后生成资源曲线及耗时比较；
失败事件不用于计算加速比。

## “真实 EM”测试覆盖什么？

两端均有光子及带电轻子队列：光子产生的电子/正电子、电子辐射或正电子
湮灭产生的光子，默认在同一端交替演化。光子不是只在 CUDA 上、电子也不是
只在 OpenMP 上；同一个粒子不会在两端各算一次。
两端的带电轨迹各自在本端计算 CoREAS/ZHS，事件结束才合并整数累积。

真实小测试导出实际 PROPOSAL 插值，初态为 8192 个电子/正电子，含生成光子的
级联，并检查 profile/CoREAS/ZHS；不是用虚拟粒子替代物理。
支持的 μ± 输运沿用共享轻子实现，但该电子初态的小测试**不能**作为所有
μ 子过程都已专项验收的证据。中微子、τ 和强子不因此被划入这套 EM 加速队列。
