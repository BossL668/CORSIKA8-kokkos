# 单个 CORSIKA 8 shower 的多 CPU 强子并行方案

## 1. 结论先行

单个 shower 想把约 20 个 CPU 执行单元长期占满，不能只把
`--hadronic-workers` 从 4 改成 20。当前 CUDA 分支已经能把**低能 FLUKA 末态生成**
批量交给多个隔离进程，但以下工作仍然串行：

- 强子 tracking、介质边界和相互作用/衰变距离竞争；
- 高能 SIBYLL、QGSJet、EPOS 或 Pythia 的末态生成；
- μ 子和其他非 EM 粒子的标量输运；
- CORSIKA 主栈写回、history 分配和部分输出；
- 每次 FLUKA flush 期间的主线程等待。

因此推荐分三层推进：

1. **立即可用**：扫描并调优已有 FLUKA 进程池，加入 CPU affinity；
2. **下一项核心重构**：把同步 `execute()` 改为异步 `submit/poll/commit`，让
   FLUKA worker、GPU EM 和主线程标量任务真正重叠；
3. **完整强子并行**：先把选定的高能模型末态放进同一进程协议，再把一整个
   “强子推进到下一物理边界”的 transport step 变为进程 worker 请求。

只有第三层才能在 shower 中后期提供足够多的独立 CPU 工作。shower 最开始只有一个
初级粒子，因果关系决定了这一小段不可能占满 20 核；这是物理任务图的限制，不是线程
设置错误。

## 2. 当前代码已经实现了什么

### 2.1 主线程仍拥有唯一 Stack

`HybridCascade` 逐个从 scheduler 取粒子，并调用标量 stepper：

```cpp
auto scheduled = scheduler_.acquireNext();
advance_result = stepper_.advance(scheduled.particle);
```

位置：

- `corsika/detail/framework/core/HybridCascade.inl`，约 164–318 行。

这个设计保证 Stack、history 和 writer 只有一个修改者，物理语义清楚；代价是普通
强子推进仍由一个主线程执行。

### 2.2 FLUKA final state 已可延期到隔离进程

FLUKA model 暴露：

```cpp
static constexpr HadronicWorkerModel hadronic_worker_model =
    HadronicWorkerModel::Fluka;
```

`InteractionCounter` 在已经选定 projectile、target 和 interaction vertex 后，不直接
调用 FLUKA，而是把请求交给当前 deferral context。主栈中的 projectile 被挂起，worker
返回后再由主线程提交次级粒子。

相关文件：

- `corsika/modules/fluka/InteractionModel.hpp`；
- `corsika/detail/framework/process/InteractionCounter.inl`；
- `corsika/framework/core/HadronicInteractionDeferral.hpp`；
- `corsika/framework/core/HadronicBatchProtocol.hpp`。

请求是无指针 POD，包含 model、PDG、四动量、history/step 和随机数 key。FLUKA 的
COMMON block 因此由不同进程隔离，不能用普通 C++ thread 代替。

### 2.3 已有分类、预计代价和 LPT 负载均衡

待处理 interaction 按以下键分类：

```text
model × species class × kinetic-energy bin × nuclear-A bin
```

同类请求保持稳定顺序。batch 按预计耗时形成，再使用 longest-processing-time-first
分配给当前预计负载最小的 worker。相关实现：

- `corsika/framework/core/HadronicWorkQueue.hpp`；
- `planHadronicWorkerAssignments()`；
- `coalesceHadronicWorkerAssignments()`。

flush 的正常条件为：

\[
N_\mathrm{pending}\ge N_\mathrm{min},\qquad
\sum_i \widehat t_i \ge N_\mathrm{worker}t_\mathrm{target}.
\]

所以 worker 数变大时，需要积累的总工作量也变大；若强子前沿不够宽，很多 worker
仍然不会得到任务。

### 2.4 当前最大结构性限制：flush 是同步的

`HybridCascade::flushHadronicInteractions()` 当前执行：

```cpp
auto completed = hadronic_process_pool_->execute(request_batches);
```

位置：`corsika/detail/framework/core/HybridCascade.inl`，约 493–575 行。

`execute()` 虽然把不同 batch 同时发给多个 FLUKA 进程，但父进程必须等本轮所有
worker 返回后才能继续 Cascade。这个等待阶段无法同时：

- 从 Stack 取下一批 CPU 粒子；
- 形成下一批强子请求；
- 主动推进下一 GPU wavefront；
- 提交已经先完成的无依赖结果。

这正是“增加 worker 后 CPU 仍不能长期吃满”的首要调度原因。

### 2.5 高能模型目前没有接入 worker

协议 enum 已预留 SIBYLL、QGSJet-II、EPOS-LHC 和 Pythia8，但当前
`fluka_batch_worker` 对非 FLUKA 请求直接返回 `InvalidRequest`，且只有 FLUKA model
声明了 `hadronic_worker_model`。所以现有 CLI：

```text
--hadronic-backend fluka-process
```

准确含义是“低能 FLUKA final state 多进程”，不是“全部强子输运多进程”。

## 3. 本机 CPU 拓扑与建议分配

当前 WSL 的 `lscpu` 观测为：

```text
CPU model: Intel Core i9-14900HX
logical CPUs: 32
reported core IDs: 16
threads per reported core: 2
NUMA nodes: 1
```

这是 WSL 提供的虚拟拓扑；不能把“32 logical CPU”当成 32 个等速物理核，也不能把
用户口中的 20 核自动理解为 20 个物理核。

若 WSL 实际只允许使用 20 个 logical CPU，建议第一轮分配：

```text
1 个主 Cascade/coordinator
1 个预留给 CUDA driver、I/O 和 WSL
8、10、12、16 个 worker 分别扫描
```

不要第一步就使用 19 worker。原因包括：

- logical siblings 竞争同一 core 的执行资源；
- 过多 worker 提高 FLUKA 内存占用；
- worker 越多，cost-trigger 的总 flush 门槛越高；
- 单次 shower 不一定同时有 19 个已选定 FLUKA vertex；
- GPU EM 或射电可能才是 wall-time 主瓶颈。

当前 32 logical CPU 环境可以先用“一条 sibling/core”的集合做物理核优先扫描，例如
`0,2,4,...,30`。为主线程和 CUDA runtime 保留两个 core ID 后，约 12–14 个 FLUKA
worker 是比 19 更合理的初始上限。最终必须以实测决定，不能只依据 `top` 的瞬时占用。

## 4. 现在无需重编译即可做的实验

现有 CUDA+FLUKA build 已包含 `fluka_batch_worker`，worker 数是运行时参数，不需要
为每个数量重新编译。需要注意，当前实现只在 `--em-backend cuda` 的
`HybridCascade` 路径中接受 `fluka-process`，不能把该参数直接加到原版
`--em-backend proposal` 路径。当前生产参数为 4 worker：

```text
--hadronic-backend fluka-process
--hadronic-workers 4
--hadronic-min-batch 64
--hadronic-target-batch-ms 5
--hadronic-max-batch 256
```

在当前 500+500 验收结束前不要改变正在运行的任务。之后选择一个固定 seed、关闭或
固定所有无关变化，依次测试：

```text
workers = 1, 2, 4, 8, 12, 16
```

第一轮建议保持 batch 参数不变，以隔离 worker 数的影响。第二轮再扫描：

```text
hadronic-min-batch       = 64, 128, 256
hadronic-target-batch-ms = 5, 10, 20
hadronic-max-batch       = 256, 512
```

一个偏向 12 worker 的开发起点可写为：

```bash
--hadronic-backend fluka-process \
--hadronic-workers 12 \
--hadronic-min-batch 128 \
--hadronic-target-batch-ms 10 \
--hadronic-max-batch 256
```

这只是扫描起点，不应直接替换正式默认值。

同时设置：

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

否则每个进程内部库再次开线程，会造成严重 oversubscription。

### 4.1 必须记录的指标

启用：

```text
--cpu-detailed-step-timing
```

并从每个 shower metadata 读取：

- `hybrid_total`；
- `scalar_stepper_time_by_pdg`；
- 高能/低能 final-state time；
- `hadronic_worker_execute_time`；
- prepare、IPC dispatch/poll/receive、commit 时间；
- 每个 flush 的 predicted/actual worker load；
- cost/capacity/drain flush 数；
- 每个 worker 的 request 数和 final-state CPU time；
- GPU kernel、transfer、host postprocess 和 radio 时间；
- 进程 CPU time 总和 / shower wall time。

“CPU 是否吃满”应定义为积分量：

\[
U_\mathrm{effective}=
\frac{\sum_p T_{\mathrm{CPU},p}}
     {N_\mathrm{allocated}\,T_\mathrm{wall}},
\]

而不是一次 `top` 或 `nvidia-smi` 快照。

## 5. 阶段 A：加入固定 CPU affinity

当前 worker 由 `HadronicProcessPool::startWorker()` fork/exec，但没有为每个 child 固定
CPU。短期可以用：

```bash
taskset -c <allowed-list> c8_air_shower ...
```

限制整个进程树，但这只是允许集合，不保证每个 worker 使用不同 core。

建议增加：

```text
--main-cpu CPU
--hadronic-cpu-list CPU0,CPU1,...
```

并在 child `exec()` 前调用 `sched_setaffinity()`。要求：

- worker CPU 列表不可重复；
- 主线程 CPU 默认不在 worker 列表；
- 配置和实际 affinity 写入 summary；
- affinity 设置失败必须终止，不得静默忽略；
- heterogeneous P/E core 的 worker capacity 使用离线固定权重，不能用本 shower 的
  wall clock 动态改变调度。

## 6. 阶段 B：把进程池改成异步状态机

### 6.1 新接口

把同步：

```cpp
std::vector<HadronicCompletedBatch> execute(batches);
```

拆为：

```cpp
HadronicSubmission submit(std::vector<HadronicRequestBatch> const&);
std::vector<HadronicCompletedBatch> pollCompleted();
bool hasInFlight() const;
std::size_t idleWorkerCount() const;
void drain();
```

`submit()` 只能向 idle worker 发请求并立即返回。`pollCompleted()` 使用非阻塞
`poll()`/`waitpid(WNOHANG)`，不得等待最慢 worker。

### 6.2 HybridCascade 事件循环

事件循环的活跃条件改为：

```text
CPU scheduler 非空
或 GPU router pending
或 hadronic queue 非空
或 hadronic batches in flight
或 completed response reorder buffer 非空
```

每轮按预算执行：

1. 提交可形成的强子 batch；
2. 推进一个或若干 GPU wavefront；
3. 推进一组仍可标量处理的 hadron/μ/decay；
4. 非阻塞收取 worker 响应；
5. 按稳定 sequence fence 提交已就绪响应；
6. 只有没有任何独立工作时才进行阻塞 drain。

这样 FLUKA 计算能与 GPU EM 和其他 CPU 工作重叠，而不是父进程站在 socket 前等待。

### 6.3 确定性 reorder buffer

worker 完成顺序受 OS 调度影响，不能直接决定 Stack 写回顺序。使用：

```text
map<sequence_id, CompletedInteraction>
next_sequence_to_commit
```

只有连续的 `next_sequence_to_commit` 已到达时才提交。更晚的响应先缓存。这样 worker
数量和完成顺序不会改变 history 分配、主栈顺序和输出。

需要设置：

- 最大 in-flight interaction 数；
- 最大缓存次级数/字节；
- head-of-line 等待时间统计；
- 超限 backpressure；
- worker crash、协议错误或丢失 sequence 时 fail closed。

## 7. 阶段 C：把高能强子末态接入同一协议

### 7.1 通用 worker，而不是线程调用 Fortran

SIBYLL、QGSJet、EPOS 等模型含 Fortran 全局状态或模型内部可变状态，同样优先使用
进程隔离。把 `fluka_batch_worker` 泛化为：

```text
hadronic_batch_worker --low-model FLUKA --high-model SIBYLL-2.3d
```

每个 worker 在 CUDA context 创建前启动，并只初始化本次运行选中的模型，避免每个
worker 同时加载所有大型模型。

### 7.2 需要修改的模块

- `HadronicBatchProtocol.hpp`
  - 增加 QGSJet-III enum；
  - 增加 model/version/config fingerprint；
  - 如模型需要，加入明确的 model-specific request flags。
- SIBYLL/QGSJet/EPOS/Pythia model adapter
  - 提供不接触 CORSIKA Stack 的 `generateFinalState()`；
  - 声明对应 `hadronic_worker_model`；
  - 把末态统一编码为 PDG、动能、方向。
- worker executable
  - 按 request.model dispatch；
  - 每个模型独立初始化、独立随机数 callback；
  - 输出模型版本、初始化哈希和每模型计时。
- `InteractionCounter`
  - 继续只延期已选定的 final state；cross section、目标选择和顶点仍留在主进程。

### 7.3 模型状态必须显式处理

不能只给高能 model 加一行 `hadronic_worker_model`。例如某些实现含交替 projectile、
调用计数或 COMMON block 状态；若状态依赖 worker 先前处理了哪些请求，改变 worker 数
就会改变 shower。必须满足以下二选一：

1. 每个 request 前把模型状态完全重置为 history-keyed 状态；或
2. 把所有影响物理的状态编码进 request，并用稳定 sequence 计算。

随机数 key 应至少包含：

```text
(seed, shower_id, history_id, step_id, model_id, draw_domain)
```

不能包含 worker ID、batch ID、完成时间或 dispatch 顺序。

## 8. 阶段 D：并行完整的强子 transport step

即使全部 final-state generator 并行，tracking 和 step competition 仍在一个主线程。
若 profiling 证明这部分占比高，再增加 `HadronTransportWorker`。

### 8.1 请求 POD

```cpp
struct HadronTransportState {
  int32_t pid;
  double total_energy_GeV;
  double position_m[3];
  double direction[3];
  double time_s;
  double weight;
  uint32_t medium_id;
  uint64_t history_id;
  uint64_t parent_history_id;
  uint32_t generation;
  uint64_t step_id;
  RandomKey random_key;
};
```

worker 只推进到一个明确的 commit boundary：

- 离散 interaction/decay；
- 几何层边界；
- 磁偏转/连续损失上限；
- observation/escape；
- 产生需要中央 GPU 处理的 EM 次级。

### 8.2 返回 record，而不是修改共享对象

worker 返回：

```text
trajectory segment
continuous energy deposit
selected process and target
updated parent state
secondary POD array
observation/escape record
next random step identity
```

主线程按 history/step 稳定顺序：

- 提交强子/μ 次级到主 Stack；
- 把 gamma/e± 送入 GPU EM 队列；
- 合并 profile、energy deposit、ground 和 radio input；
- 分配新 history ID。

第一版每次只推进一个 step，物理边界最清楚。确认一致后才考虑让 worker 连续推进到
下一次 branching，以降低 IPC。

### 8.3 为什么不能直接让多个线程共享 Stack

共享 Stack、SecondaryView 和 writer 会带来：

- iterator 失效；
- history ID 冲突；
- 输出顺序不确定；
- thinning 权重重复或漏计；
- Fortran model 状态竞争；
- 相同 seed 随 thread schedule 改变。

所以保持“worker 只计算 POD，主线程唯一 commit”是科研可审计性的关键。

## 9. 建议的 20 logical CPU 运行时角色

异步和高能 worker 完成后，建议不要建立彼此固定且长期闲置的多个 pool，而是使用一个
具有 model capability 的共享进程池：

```text
CPU 0:     Cascade coordinator / deterministic commit
CPU 1:     CUDA driver、transfer、I/O 预留
CPU 2–17:  16 个 hadronic worker（按实际物理 core 调整）
CPU 18–19: writer/compression 或系统余量；也可纳入 worker 扫描
GPU:       EM transport + CoREAS/ZHS
```

如果 20 logical CPU 其实只有约 10 个 physical core pair，初始应改为：

```text
1 coordinator + 1 reserve + 8 physical-core-first workers
```

再比较 SMT sibling 是否带来净收益。目标是最短 wall time，而不是让任务管理器显示
100%。

## 10. 性能上限与是否值得做

假设可并行强子部分占总 wall time 的比例为 \(f\)，20 worker 的理想上限为：

\[
S_{20}\le\frac{1}{(1-f)+f/20}.
\]

例如：

| 可并行强子占比 | 理想 20-worker 上限 |
|---:|---:|
| 2% | 1.02× |
| 10% | 1.10× |
| 30% | 1.41× |
| 50% | 1.90× |

已有阶段 89 的一个 1 PeV 正式配置中，FLUKA pool wall 只占 Hybrid total 的约
0.175%；另一个较轻 thinning、无射电开发配置中占比更高。说明是否值得扩展必须以
目标能量、thinning、radio 和模型组合重新 profile，不能把 worker kernel 加速当成
端到端加速。

当前 100 TeV、垂直质子、`emthin=1e-6`、CUDA EM+radio 正式 500-event
campaign 的首个 25-event 闭合批次进一步给出：

| 第一批 25 例的事件平均计时 | 时间 | 占平均事件 wall 的描述性比例 |
|---|---:|---:|
| `hybrid_total` | 111.461 s | 100% |
| `router_advance` | 80.142 s | 71.9% |
| `scalar_stepper` | 27.314 s | 24.5% |
| 4-worker `hadronic_process_pool.execute` | 0.0640 s | 0.057% |

这些阶段计时存在父子嵌套，不能逐项相加；它们用于定位瓶颈，不是互斥的时间预算。
该批次中仅 FLUKA worker 执行所占比例极小，因此把现有 worker 从 4 增到 20 的
Amdahl 上限接近 1。若要让多 CPU 对这类事件产生可测的端到端收益，必须并行
`scalar_stepper` 中的高能强子/完整 transport，并与 GPU 异步重叠。25 例只用于
性能结构诊断，不用于提前判断完整 500 vs 500 的物理分布。

对于当前 CUDA EM + CUDA radio 的生产模式，更现实的目标是：

- CPU pool 在 GPU 工作期间并行处理强子；
- 消除同步 socket 等待；
- 降低高能 shower 的 CPU serial tail；
- 保持 GPU 不因等待 CPU 末态而空闲。

## 11. 验收矩阵

### 11.1 正确性

- 固定 seed，worker=1/2/4/8/12/16：request/response fingerprint 相同；
- worker 数改变时，profile、particles、dEdX、ground、CoREAS/ZHS 输出逐项一致；
- scalar 原版与 process backend：至少做独立 seed ensemble 统计一致性；
- 强制 worker 延迟乱序，commit 结果仍相同；
- worker crash、非法 PID、NaN、协议/模型哈希不符必须停止 shower；
- FLUKA、高能模型版本和每个 worker 初始化哈希写入 metadata。

### 11.2 性能

- 每个 worker 数至少重复 5 次，使用相同固定 seed；
- 记录冷初始化和热运行；
- CPU affinity、频率/功耗模式和 GPU 配置固定；
- 报告中同时给出：
  - end-to-end wall；
  - 强子 worker wall 和 CPU-sum；
  - worker occupancy/imbalance；
  - queue wait、head-of-line wait、IPC、commit；
  - GPU idle fraction；
  - 峰值 RSS；
  - 有效 CPU 利用率。

### 11.3 阶段完成门槛

1. **调参阶段**：相同物理输出，找到本机 FLUKA pool 的最优 worker/batch；
2. **异步阶段**：固定 seed 输出不变，GPU/CPU 重叠增加，端到端无回退；
3. **高能阶段**：所有支持模型通过 1-vs-N worker 决定性测试；
4. **完整 transport 阶段**：全组分 profile、ground、radio 和能量闭合通过正式
   ensemble 验收；
5. 只有端到端中位时间有稳定改善才调整生产默认值。

## 12. 推荐实施顺序

```text
P0  完成本轮 500+500，提取真实 100 TeV phase timing
P1  1/2/4/8/12/16 FLUKA worker 固定 seed 扫描
P2  CPU affinity + worker RSS/occupancy 记录
P3  HadronicProcessPool async submit/poll/drain
P4  HybridCascade in-flight 状态机和稳定 reorder commit
P5  SIBYLL worker（当前默认高能模型）
P6  QGSJet-II/III、EPOS、Pythia adapters
P7  完整 HadronTransportState/Record 一步输运
P8  20-logical-CPU 自动拓扑探测与离线标定配置
P9  固定 seed 决定性、独立 ensemble 物理与端到端性能验收
```

最值得先做的是 P3/P4，而不是盲目把现有 worker 数设为 20。它既能提升已有 FLUKA
并行的实际重叠，也为后续高能模型和完整强子 transport 共用同一调度框架。
