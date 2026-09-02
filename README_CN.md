# CORSIKA 8 GPU 混合级联分支：中文说明

> 状态：本仓库是面向 21CMA 科研模拟的本地研究分支，不是 CORSIKA 8
> 官方发布版，也尚未经过 CORSIKA 8 上游评审。默认 CPU 路径仍是原来的
> `Cascade + PROPOSAL`；CUDA、CUDA 射电和 FLUKA 进程池都必须显式启用。

> **当前分支建议（2026-09-01）：**beta4 是从 beta2 派生的维护与验收分支。
> 它保留 beta2 的调度和性能模型，同时修复代码审计确认的 CPU/GPU 语义差异：
> 端点弦 grammage 与逐粒子 transport cut、局部水平观测平面、GPU 首相互作用
> 输出、原版 10 ms 物理时间 cut，以及强制初级相互作用/衰变。CUDA 启动时还会
> 对六类过程执行 fail-closed 兼容性门禁。beta4 现在还提供实验性的
> `proposal-native` 物理源，把实际 PROPOSAL calculator 的原生插值状态只读导出
> 为常驻 GPU 数据；在直接原始数据系综和性能门禁全部完成前，默认生产物理源仍是
> `.c8emrt`。beta3 的实验性强子多核调度不属于本次修补。

文档导航：

- [新增命令行参数完整参考](documentation/cuda_em_refactor/cli_reference.md)
- [CUDA 后端生产使用指南](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md)
- [验证脚本和复现实验](validation/gpu_em/README.md)
- [架构文档与阶段记录索引](documentation/cuda_em_refactor/README.md)
- [英文主 README 与新服务器部署](README.md#build-from-source)

本文回答四个问题：

1. 相对原版 CORSIKA 8，这个分支加入了什么？
2. CPU 粒子蒙特卡洛为什么可以改写为 GPU wavefront？
3. 电磁、μ 子、强子和射电分别在哪一端计算？
4. 与当前官方公开版本相比，哪些方向是本分支的特色，哪些方面官方版本更强？

从零迁移、CUDA 架构选择和多核服务器调优的完整命令见
[英文主 README](README.md#build-from-source)；物理覆盖、表格、
fallback 和输出字段见
[CUDA 后端生产指南](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md)。

## beta4 修复与验收状态

beta4 从 beta2 复制出来，是为了保留已经验证过的 beta2 wavefront 调度器，同时
逐项修补标量 CPU 与 CUDA 之间的语义合同；默认 `proposal` 标量路径没有被这些
修复改写。

| 检查项 | beta4 处理 | 证据 |
|---|---|---|
| 磁场步的连续 grammage 和不同粒子的 cut | 连续能损/散射使用保存端点之间的弦；电子和 μ 子分别解析自己的 cut | [Phase 97](documentation/cuda_em_refactor/phase_97_beta4_cpu_step_chord_grammage.md) |
| 观测几何 | GPU 与 CPU 使用通过 shower core 的同一局部水平面；球面只用于大气边界 | [Phase 98](documentation/cuda_em_refactor/phase_98_beta4_cpu_observation_plane_alignment.md) |
| 首物理相互作用输出 | 在薄化之前保存 generation-zero GPU 快照，并交给原 `InteractionWriter` | [Phase 99](documentation/cuda_em_refactor/phase_99_beta4_gpu_first_interaction_writer_alignment.md) |
| 迟发粒子 | 严格实现原版 `timePost > 10 ms` 的事件内物理时间条件 | [Phase 100](documentation/cuda_em_refactor/phase_100_beta4_particle_cut_time_alignment.md) |
| 强制初级与自定义过程 | 指定的初级顶点在 GPU 路由前执行一次；六类过程必须全部登记 | [Phase 101](documentation/cuda_em_refactor/phase_101_beta4_forced_primary_and_process_compatibility_gate.md) |
| 性能回归 | 同机三对 10 PeV 质子表明 beta4 与 beta2 的时间差约为 1--3% | [Phase 102](documentation/cuda_em_refactor/phase_102_beta4_beta2_10pev_performance.md) |
| 最终代码路径复核与 100 TeV 门禁 | 未发现新的生产阻断型遗漏/重排；已确认新版 beta2 CPU500 参考，并启动匹配的 beta4 CUDA500 | [Phase 103](documentation/cuda_em_refactor/phase_103_beta4_gpu_cpu_path_reaudit_and_100tev_campaign.md) |
| PROPOSAL 原生样条导出 | 已实现版本锁定只读依赖接口、canonical 哈希、GPU Hermite 求值、能区预检、metadata 和端点显式回放 | [Phase 113](documentation/cuda_em_refactor/phase_113_proposal_native_gpu_tables_CN.md) |
| 原生路径 decision 语义 | 完整 CPU tape 可重放输运和射电；固定过程/组分/条件分位点后可复刻 live PROPOSAL，但只给相同初始 seed 不保证同一事例 | [Phase 114](documentation/cuda_em_refactor/phase_114_beta4_proposal_native_decision_tape_replay_CN.md) |

原 beta4 语义修补阶段记录了 C++/CUDA `34/34`、Python validation `241/241`
通过；原生表扩展另有 table/fallback/selection/final-state/transport/radio/cache
以及 decision oracle 门禁。代码级正确性、固定 decision 回放和独立 shower
系综是三个不同证据层级，不能用其中一层替代另外两层。

## 1. 项目定位

原版 CORSIKA 8 的空气簇射主循环是标量调度：

```text
从 Stack 取一个粒子
  -> 决定下一相互作用/衰变/边界/连续步长
  -> 推进粒子
  -> 生成次级粒子并压回 Stack
  -> 继续处理同一深度优先粒子树
```

这种方式物理逻辑清楚，而且同一随机数种子可以复现同一标量 shower；但同一时刻
只处理一条粒子历史，难以把一个大电磁前沿变成 GPU 上成千上万个相互独立的线程。

本分支保留原 Stack 和 CPU 物理模块，在旁边增加一条混合路径：

```text
                 CORSIKA CPU Stack
                         |
              +----------+-----------+
              |                      |
        hadron / tau           gamma / e± / μ±
              |                      |
    scalar step + FLUKA       HybridCascade router
              |                      |
              |              resident GPU queues
              |                      |
              |       select -> transport -> final state
              |              -> thinning -> reroute
              |                      |
              +----------+-----------+
                         |
          CPU rare final state / decay / output
                         |
              CoREAS/ZHS: CPU 或 CUDA
```

这里的“GPU μ 子”是有边界的：GPU 处理其连续损失、range、Molière 散射、磁场/
大气输运、边界、cut 和衰变距离竞争；实际衰变末态以及 μ bremsstrahlung、
pair production、photonuclear 等稀有末态仍交给 CPU 精确生成。

## 2. 本分支新增内容

### 2.1 标量输运与调度解耦

新增
[`ScalarCascadeStepper`](corsika/framework/core/ScalarCascadeStepper.hpp)，把原
`Cascade::step()` 中的单粒子物理推进抽成可复用组件。原 CPU Cascade 仍可调用
同一 stepper，因此默认路径不需要变成并发 Stack。

新增
[`HybridCascade`](corsika/framework/core/HybridCascade.hpp)，负责：

- CPU 主栈和 GPU staging 队列之间的物种路由；
- 小前沿的有界 CPU 展开；
- GPU wavefront 的提交与回收；
- 指定 PROPOSAL fallback、μ 衰变和 FLUKA worker 的返回；
- history、parent history、generation 和 step 标识；
- 输出、能量沉积、纵向 profile 和 shower 结束条件。

### 2.2 编译型 CUDA 后端

新增静态库目标 `CORSIKA8GpuEm`，核心接口位于
[`CudaEmBackend.hpp`](corsika/gpu/em/CudaEmBackend.hpp)，实现位于
[`src/gpu/em`](src/gpu/em/)。

设备粒子使用可直接复制的 POD 状态：

```text
PID, medium ID, generation
energy [GeV]
position [m], direction
time [s], weight
history ID, parent history ID, step ID
```

GPU 不持有 CORSIKA 的模板 Stack，也不在 device code 中使用带单位的复杂 C++
对象。主机/设备边界显式完成单位转换，device 内固定使用 GeV、m、s、
g/cm² 和 Tesla。

### 2.3 PROPOSAL 物理表

新增 `gpu_em_table_prepare`、`gpu_em_tablegen` 和版本化 `.c8emrt` 文件。
前者负责“介质 YAML → 规范化哈希 → 查找/生成/校验”，后者负责底层
PROPOSAL 数值制表。表格保存：

- 每个粒子、介质组分和过程的相互作用率；
- 过程与目标组分概率；
- 能损比例 $v$ 的逆累计分布；
- 连续 $dE/dX$、range $R(E)$ 和逆 range；
- photon-pair、brems 和 electron-pair 的 LPM 参数；
- Molière 散射参数；
- PROPOSAL 版本、参数化、cut、能区、误差、schema 和内容哈希。

当前格式和数据结构见
[`RateTable.hpp`](corsika/gpu/em/tables/RateTable.hpp)，设备平坦视图见
[`FlatRateTable.hpp`](corsika/gpu/em/tables/FlatRateTable.hpp)。

beta4 另提供实验性的 `--gpu-physics-source proposal-native`：它从当前
`InteractionModel` 和 `ContinuousProcess` 已经构造的 calculator 中只读导出
PROPOSAL 7.6.2 / CubicInterpolation 0.1.5 原生样条，转换为 GPU POD 并常驻
显存，不再制作完整 `.c8emrt`。小型 LPM/Molière 辅助数据自动存入
`.c8emaux`；旧 `.c8emrt` 仍是默认生产路径。实现和当前验收边界见
[Phase 113](documentation/cuda_em_refactor/phase_113_proposal_native_gpu_tables_CN.md)。
当前 58 个 photon/electron/positron/muon 过程列均已通过每列一百万点的
依赖、缓存和 host/device 求值检查；五种粒子共五百万次完整选择的过程和目标
组分 mismatch 为 0。严格百万点 loss oracle 保留一个高能 Compton 点：其
`v` 相对差为 `4.57e-10`，超过预设 `1e-10` 门槛，因此不能写成无条件全通过。
另一个固定语义 decision 的 20,480 顶点测试中，过程、组分和 live
`SampleLoss` mismatch 均为 0，已支持过程的最大 `v` 相对差低于 `8e-13`。
CPU-only 过程和极小端点带使用同一 selection random number 显式回放，不增加
新的随机数。1 TeV native 2000 例已完成本地阶段检查；与 CPU 精确同 seed 集合
的 100 TeV native 2000 例正在运行。直接原始数据与性能门禁尚未全部完成，
因此默认值不变。

表查询不在能区外静默 clamp。缺列、超范围、无逆 CDF 或哈希/误差不匹配都会
成为明确 fallback 或 hard failure。

这里有两个容易混淆的“表”：

- PROPOSAL 自身的插值 cache：准备/生成工具会在介质哈希目录中按需创建；
- CUDA 运行时使用的 `.c8emrt`：可以先由独立 `gpu_em_table_prepare` 自动查找
  或生成；默认 `c8emrt` 模式只读显式路径；
- 实验性原生 GPU 表：运行时从上述 PROPOSAL cache 对应 calculator 导出，
  不要求介质 YAML 或完整 `.c8emrt`，同一进程的后续 shower 复用设备上传。

`proposal-native` 并不是把 PROPOSAL C++ 直接放进 CUDA kernel。PROPOSAL 仍在
主机端建立或加载自己的插值器；程序只读导出轴、系数和物理标识，转换为扁平
POD 后在 GPU 上按相同坐标变换和 Hermite 语义求值。不支持的插值轴、参数化、
化学组成或配置能区会在 shower 输运前明确失败。

因此，“更换介质不需要重新制表”应准确理解为：用户不再需要手工运行
`gpu_em_table_prepare` 生成完整 `.c8emrt`，而不是所有介质共用同一张表。

- 只改变同一化学组成的密度 profile、磁场或观测几何时，可以复用相同的
  PROPOSAL 物理系数；
- 改变元素组成、组分比例、材料常数、cut、PROPOSAL 参数化或依赖版本时，
  PROPOSAL 会自动建立或读取另一套 host cache，并得到新的 canonical native
  hash；第一次运行仍可能较慢，但不需要用户手工制完整表；
- 当前一个 native backend 实例只接受一种化学组成，空气/岩石/冰同时存在的
  mixed-media geometry 会在输运前拒绝；
- 当前 `c8_air_shower` 的 CUDA 环境仍只完成五层 `AirDry1Atm` 干空气验收。
  真正运行岩石、冰或月壤还要实现相应 device environment snapshot、geometry、
  grammage、medium-ID 映射和物理验收；原生表导出本身不会自动补齐这些功能。

所以，对已经由运行时环境支持的单一新介质，不再需要手工完整制表；首次运行会
自动产生 PROPOSAL cache、新的 native 表哈希和必要的 `.c8emaux` 辅助缓存。

`gpu_em_tablegen --medium-yaml` 已支持 schema 1 的自定义材料，准备工具会对
组成和全部 PROPOSAL 材料参数规范化并计算 SHA-256。当前 `c8_air_shower`
的五层 GPU 环境仍固定使用 `AirDry1Atm` 标准干空气。仅改变密度随高度分布、
观测高度或磁场时，可以复用同一表；改变元素组成、组分比例或材料参数时必须用
新 YAML 制表并重新验收。岩石、土壤、月壤和冰还需要同步扩展运行时环境快照、
几何和 medium ID，不能只换表。

### 2.4 GPU 粒子过程

当前 GPU 末态覆盖：

- photon pair production；
- Compton scattering；
- photoelectric effect；
- $e^\pm$ bremsstrahlung；
- $e^\pm$ electron-pair production；
- 离散和连续 ionization；
- positron annihilation；
- Molière multiple scattering；
- pair/brems/epair 的 LPM 抑制；
- ParticleCut（包括原版严格的 `timePost > 10 ms` 迟发粒子条件）和
  EMThinning。

当前 GPU μ 子覆盖：

- 连续电离与 range；
- Molière 散射；
- 五层球形大气和均匀磁场 tracking；
- 离散相互作用、连续步长、边界、观测面、cut 和衰变距离竞争；
- GPU 端 ionization 选择。

CPU 指定末态包括：

- photoproduction/photonuclear；
- photon-induced muon pair；
- μ 子 bremsstrahlung、pair production 和其他稀有离散末态；
- μ 子实际 decay final state；
- 产生 hadron/μ/τ 的非 GPU 末态；
- 预先声明的介质或几何能力边界。

过程能力表见
[`ProcessCapabilities.hpp`](corsika/gpu/em/ProcessCapabilities.hpp)。

### 2.5 GPU CoREAS/ZHS

新增
[`CudaRadioAccumulator`](corsika/gpu/radio/CudaRadioAccumulator.hpp)，可用：

```text
--radio-backend cuda
```

设备端对每条 $e^\pm$ 轨迹、每个天线和时间 bin 计算 CoREAS endpoint 与 ZHS
贡献。为使并行累加顺序不改变结果，波形使用带范围检查的定点累加器；超过
`--gpu-radio-field-limit` 会终止 shower，而不是产生未标记的环绕误差。

P1 射电优化先为每条轨迹预计算一次与天线无关的运动学量，再以
`8 tracks × 32 observers` 的二维 tile 在 shared memory 中复用轨迹和天线元数据。
它自动用于 CUDA radio，不增加 CLI 参数；双缓冲 track workspace 计入统一显存
预算。RTX 4060 上 10 个 paired 生产事件的 projection device time 缩短 5.52%，
端到端 wall time 缩短 3.80%，所比较的 shower 与射电文件逐字节一致。P1 预计算和
P2 tile 优化分别见
[`phase_111_beta4_p1_radio_track_precompute_observer_tiling_CN.md`](documentation/cuda_em_refactor/phase_111_beta4_p1_radio_track_precompute_observer_tiling_CN.md)
与
[`phase_112_beta4_p2_radio_8x32_tiling_CN.md`](documentation/cuda_em_refactor/phase_112_beta4_p2_radio_8x32_tiling_CN.md)。

原始 CPU 射电路径仍可用：

```text
--radio-backend cpu
```

无论 CPU 还是 GPU，射电贡献都随粒子轨迹段产生而在线累计；并不是等所有粒子
存盘以后再重新读取整棵 shower。最终波形在 shower 结束时下载或写出。μ 轨迹
不直接加入射电源项，因为原标量 `RadioProcess` 同样只观察 $e^\pm$；μ 衰变
产生的电子会正常进入射电计算。

### 2.6 FLUKA 多进程末态

新增 [`fluka_batch_worker`](applications/fluka_batch_worker.cpp) 和：

```text
--hadronic-backend fluka-process
```

这不是“把强子模型写成 CUDA kernel”。HybridCascade 把低能强子相互作用顶点
按工作类别和预估代价稳定分组，再交给多个隔离的持久 FLUKA CPU 进程批量计算，
减少重复初始化和逐顶点 IPC。默认起点为：

```text
--hadronic-workers 4
--hadronic-min-batch 64
--hadronic-target-batch-ms 5
--hadronic-max-batch 256
```

新服务器应单独扫描 worker 数量；GPU 性能计时时不能并行运行另一套 CPU shower。

FLUKA 初始化进程内相互作用对象时会安装周期性的 quota 定时器。在
`fluka-process` 模式中，隔离 worker 启动完成后，CUDA 主进程会关闭自身冗余的
定时器；worker 仍保留原生 FLUKA 初始化和物理。这样可以避免多线程 CUDA 主进程
在 `SIGALRM` handler 中调用 C stdio，并因 libc 流锁重入而永久死锁。默认标量
后端不受影响。

### 2.7 确定性、回放和审计

设备随机数使用 Random123 Philox4x32-10。随机地址为：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

实现见 [`Philox.hpp`](corsika/gpu/em/Philox.hpp)。队列位置、block 大小和
wavefront 重排不进入随机地址，因此同一 GPU、同一表和配置应可重复。

新增 `cuda_decision_replay`，可读取原版标量 decision tape，在 GPU 上消费指定
的输运记录并重新计算同轨迹射电。这能回答“GPU 能否精确消费原版某一事例”，
但它不是 production CUDA 独立抽样。

生产 CUDA 与原版 CPU 使用不同的随机流地址和调度顺序，所以相同 seed 一般不会
生成逐粒子完全相同的独立 shower。正确的两层验收是：

1. decision replay：同一指定粒子树/轨迹的逐过程调试；
2. independent ensemble：不同独立 seed 下 $X_{\max}$、纵向粒子数、能量
   沉积、地面分布和射电脉冲统计一致。

## 3. 粒子蒙特卡洛算法

### 3.1 相互作用距离

设粒子在能量 $E$ 和介质 $m$ 中，各离散过程的质量厚度反应率为
$\lambda_p(E,m)$，总率为：

$$
\lambda_{\mathrm{tot}}(E,m)=\sum_p\lambda_p(E,m).
$$

抽取 $u_X\in(0,1)$，下次离散相互作用的质量厚度为：

$$
\Delta X_{\mathrm{int}}
=-\frac{\ln u_X}{\lambda_{\mathrm{tot}}}.
$$

在非均匀大气中，$\Delta X$ 不是几何长度。tracking 必须沿轨迹积分
$\rho(\mathbf{x})\,ds$，并求逆得到实际顶点位置。

### 3.2 竞争限制

一次 step 不是无条件走到相互作用点，而是比较：

```text
离散相互作用距离
衰变距离
大气层/介质边界
磁场最大偏转步长
连续能损允许步长
观测面或环境逃逸面
```

取最近限制推进。到达边界时粒子保留并进入新介质；到达相互作用点才生成离散
末态；到达 cut 时沉积剩余允许能量；到达观测面时写 observation record。

### 3.3 过程、目标和能损抽样

用第二个随机数 $u_p$ 在累计率中选择过程和目标组分：

$$
\sum_{q<p}\lambda_q
\le u_p\lambda_{\mathrm{tot}}
<\sum_{q\le p}\lambda_q.
$$

对有随机能损的过程，用 $u_v$ 查询逆累计分布：

$$
v=F^{-1}_{p,m}(E,u_v).
$$

CPU PROPOSAL 原本就会使用自己的插值/逆累计计算。本分支为了 GPU 设备访问，
额外生成平坦、自适应细化且带实测误差的表；它不是声称原 PROPOSAL 完全没有
插值。二者区别见
[PROPOSAL 与 CUDA 插值说明](documentation/cuda_em_refactor/proposal_and_cuda_interpolation.md)。

### 3.4 连续能损

对带电粒子，表中 range 定义为从 transport cut 到当前能量的累计质量厚度。
走过 $\Delta X$ 后：

$$
E_1=R^{-1}\!\left(R(E_0)-\Delta X\right).
$$

差值 $E_0-E_1$ 进入连续能量沉积。若 $\Delta X$ 跨过 cut，则 step 在 cut
附近终止，避免查询表外能量。

### 3.5 次级粒子与薄化

每个输入粒子先计算会产生多少 continuation、次级、observation、fallback 和
记录。设备端用 CUB exclusive scan 得到稳定写入偏移：

```text
child_count:  [2, 0, 1, 2]
exclusive:    [0, 2, 2, 3]
```

这样每个线程写入自己的确定区间，不使用“哪个线程先抢到全局原子计数器”的不
确定追加顺序。history ID 按 scan 偏移稳定分配。

EM thinning 在 GPU 端使用同样的能量阈值、权重和保留概率语义。被保留粒子的
权重补偿保证 ensemble 期望值不变；单事例会有由 thinning 引入的统计涨落。

一个容易误判的细节是 `c8_air_shower` 的自动最大权重。未显式指定
`--max-weight` 时，CPU 原版和本分支都使用

```text
maxWeight = 0.5 * emthin * E_primary[GeV]
```

而 `EMThinning` 在 `parentWeight >= maxWeight` 时直接返回。初级和第一代未加权
粒子的权重为 1，所以自动 `maxWeight` 必须大于 1，薄化才可能从这些 history
启动。例如 $E_\mathrm{primary}=10^5$ GeV、`emthin=1e-6` 时，能量阈值是
0.1 GeV，但自动 `maxWeight=0.05`，因此该配置实际上不会启动 EM thinning。
$10^6$ GeV、`emthin=1e-6` 对应的 `maxWeight=0.5`，同样如此。若确实需要薄化，
应显式给出物理上合适且大于 1 的 `--max-weight`，并把该值纳入 CPU/CUDA
共同配置和验证；不能只根据运行时间判断薄化是否生效。程序会把
`can_activate_from_unit_weight` 写入 summary，并在不可能启动时给出 warning。

## 4. GPU wavefront 为什么成立

同一 shower 内不同粒子 history 在给定当前状态后可以独立推进一个物理限制。
因此可以把深度优先树改写为“同一轮推进许多粒子一次”：

1. host staging 收集可路由粒子；
2. 按 `PID × medium_id × energy_bin` 生成 bucket key；
3. stable radix sort 形成相似工作负载；
4. 每个 CUDA thread 推进一个粒子到最近限制；
5. exclusive scan 分配输出；
6. $\gamma/e^\pm/\mu^\pm$ 在设备常驻队列交叉路由；
7. observation、profile、radio、fallback 批量回收；
8. 队列为空后 shower 才结束。

稳定分桶实现见
[`CudaWavefrontBucketing.cu`](src/gpu/em/CudaWavefrontBucketing.cu)，设备工作区
和双缓冲见
[`DeviceWorkspace.hpp`](corsika/gpu/em/detail/DeviceWorkspace.hpp)。

这种调度的主要收益来自：

- 一次 kernel 启动处理大量相似粒子；
- photon/lepton 次级不必每一步往返主机；
- 物理表连续读取更适合 GPU cache；
- profile 和 radio 可在设备上归约；
- Philox 随机数不依赖执行顺序。

主要代价是：

- 小前沿无法占满 GPU；
- 稀有 CPU 末态会形成同步边界；
- 不同过程分支造成 warp divergence；
- 高质量低 thinning shower 会产生很大的队列和射电工作量；
- 强子、输出或 CPU fallback 可能成为 Amdahl 瓶颈。

`--gpu-min-batch` 控制小前沿何时先做有界 CPU 展开。它不是越小越好；不同 GPU
应对 64、256、1024、4096、8192 等值做固定配置的中位数扫描。

## 5. 环境与 21CMA 配置

当前 CUDA snapshot 包含：

- 五层球形大气；
- 四层指数密度和一层线性密度；
- 干空气介质及 medium ID；
- 球面边界求交；
- grammage 积分和逆积分；
- 均匀磁场 leapfrog；
- 与球形大气层边界独立、并与 `c8_air_shower` CPU 路径一致的局部水平观测面。

当前 `c8_air_shower` 默认从 `GeoMag/IGRF14.COF` 读取模型，默认配置为：

```text
year       2027
latitude   42.5527 deg
longitude  86.4153816422 deg
altitude   2680.444195 m
```

可用 `--geomagnetic-model IGRF13|IGRF14` 选择系数文件，并用
`--geomagnetic-year` 选择 1900--2030 年间的计算年份。CPU 与 CUDA 使用同一个
计算后磁场矢量，模型、年份和值都会写入 `gpu_em/config.yaml`。默认场矢量
（NWU/CORSIKA 坐标）为约 `[25.180518, -1.141517, -50.996641]` μT。项目已在
`resources/GeoMag` 中附带 IGRF14，并会在安装时自动复制；若
`CORSIKA_DATA/GeoMag/IGRF14.COF` 已存在，则优先使用该文件。

山体、月壤、冰和一般三维介质尚未进入当前 GPU snapshot。官方 CORSIKA 8 的
通用环境框架能够表达跨介质问题，不等于本地 CUDA kernel 已支持这些几何。

## 6. 输出和失败策略

CUDA 模式写出：

- `gpu_em/config.yaml`：GPU、driver/runtime、表哈希、环境和配置；
- `gpu_em/summary.yaml`：粒子数、wavefront、fallback、显存、溢出和分阶段时间；
- profile、能量沉积和 observation 输出；
- CPU 或 CUDA CoREAS/ZHS 波形；
- 验证 provenance 和独立 ensemble 报告。

以下情况必须停止当前 shower：

- CUDA error、NaN、负能量或非法 PID；
- 表格 schema/hash/cut/能区/误差不匹配；
- 非法未登记过程；
- 大气积分、LPM、Molière 或磁场计算失败；
- 队列/定点射电累加溢出；
- 能量或记录计数不一致。

只有预先声明的稀有指定末态、能力边界和受控低能显存 spill 可以回 CPU。禁止
自动切换成 CPU 后把输出伪装成完整 CUDA shower。

CUDA 模式下的 `--force-interaction` 和 `--force-decay` 保持标量
`Cascade` 的接口语义：调度器下一次取出的初级粒子先执行且只执行一次强制
标量顶点，随后它的次级粒子再走正常混合路由。两个请求互斥；实际执行次数写入
`gpu_em/summary.yaml` 的 `forced_primary`。

CUDA 启动门禁会递归检查过程序列中的 `ContinuousProcess`、
`SecondariesProcess`、`InteractionProcess`、`DecayProcess`、
`BoundaryCrossingProcess` 和 `StackProcess`。每个类型必须声明为设备替代、记录
回放、CPU 延迟、对 routed EM 不适用或仅诊断；未登记类型会使启动失败，不会被
静默跳过。五类策略计数和六类未登记计数写入 `process_registry`。

## 7. 新服务器构建和运行

CUDA 构建要求 CMake 3.24+、CUDA toolkit 12.x、C++/CUDA 17。下面是一套可
迁移流程；更详细的故障排查和多 GPU 调优见
[英文主 README](README.md#build-from-source)。本节假设 Conda 环境是全新的，
不依赖另一份 CORSIKA 构建目录或旧 Conan profile。7.1--7.7 应在同一个已激活
`corsika_venv` 的 shell 中依次执行；若中途重新登录，必须重新执行相应的
`export`，不能继承另一份旧构建的变量。

### 7.1 检查 driver、GPU 和编译器

Conda 负责安装项目级工具、CUDA toolkit 和 Python 包，但下面经过实测的流程
刻意使用宿主机同一套 GCC/G++/GFortran。全新的 Ubuntu/WSL 系统应先安装：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  gfortran \
  git \
  ca-certificates
```

没有 `sudo` 的集群应加载管理员提供且互相匹配的 GCC、G++、GFortran 和 CUDA
module。首次构建还需要能够通过 HTTPS 访问 Conda channel、Conan Center、私有
GitHub fork、KIT 的公开子模块，以及 Pythia/Tauola 源码下载地址。

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total \
  --format=csv
gcc --version
g++ --version
gfortran --version
make --version
git --version
ldd --version
free -h
df -h .
```

NVIDIA driver 属于宿主系统。Conda 中只有 toolkit 而 `nvidia-smi` 失败时，
需要先修复宿主 driver；在 WSL2 中应安装 Windows NVIDIA driver，不应在 WSL
内再安装第二套显示驱动。

`nvidia-smi` 显示的 `CUDA Version` 是当前 driver 能支持的最高 CUDA driver API，
不是本项目实际使用的编译工具包版本；应以 `nvcc --version` 为准。因此 driver
显示 CUDA 13.x 时，仍可正常运行用受支持的 CUDA 12.6 toolkit 编译的程序。

### 7.2 建立参考 Conda 环境

下面从一个完全不存在的新环境开始，同时安装验收脚本需要的 Parquet 和
`pytest` 支持；这些 Python 包不会由后面 Conan 编译的 C++ Arrow 自动提供。

若新的 SSH 或非交互 shell 中还不能使用 `conda activate`，可以先执行下面一条，
不必修改 shell 启动文件：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

```bash
conda create -n corsika_venv \
  -y \
  -c conda-forge \
  python=3.9 \
  cmake=3.31 \
  conan=2.11 \
  pip \
  git \
  make \
  pkg-config \
  particle=0.25.1 \
  numpy pandas scipy matplotlib pyyaml pyarrow pytest

conda activate corsika_venv
conda config --env --set channel_priority strict

conda install \
  -y \
  -c nvidia/label/cuda-12.6.3 \
  -c conda-forge \
  cuda-nvcc=12.6.85 \
  cuda-cudart-dev=12.6.77 \
  cuda-cccl=12.6.77
```

上面三个 CUDA 包是本项目在 T400 上经过实测的最小集合。若服务器必须安装完整
toolkit，而且历史 NVIDIA label 仍能解析元包，可以用下面这一条替换上面的三个
CUDA 包；不要因为一次求解失败而把两套方案重复叠加安装：

```bash
conda install \
  -y \
  -c nvidia/label/cuda-12.6.3 \
  -c conda-forge \
  cuda-toolkit=12.6.3
```

这组版本复现当前开发环境。HPC 上也可以使用管理员提供的 CUDA/GCC module；
关键是 Conan 建 profile、CMake 和 NVCC host compiler 必须选择同一套 C++
工具链。

部分 Conda CUDA 包会把 C/C++ wrapper 加入 `PATH`，却没有同时提供相同版本的
Fortran 编译器。下面显式选择 Ubuntu/WSL 原生工具链和 Conda 中的 NVCC；集群
使用 module 时，应把前三个路径一起换成该 module 提供的路径：

```bash
test "$CONDA_DEFAULT_ENV" = corsika_venv

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="$CXX"

for program in "$CC" "$CXX" "$FC" "$CUDACXX" cmake conan python make git; do
  test -x "$(command -v "$program")" || {
    printf '缺少必需程序：%s\n' "$program" >&2
    exit 1
  }
done

"$CC" --version | head -n 1
"$CXX" --version | head -n 1
"$FC" --version | head -n 1
"$CUDACXX" --version
cmake --version | head -n 1
conan --version
python -c 'import numpy, pandas, pyarrow, pytest, scipy, yaml; print("Python 依赖：OK")'
```

GCC、G++ 和 GFortran 必须来自同一个 major toolchain。后面的
`conan-install.sh` 会在这些变量已经生效后创建项目 profile；不能先用一套编译器
建 Conan profile，再用另一套编译器运行 CMake。

### 7.3 配置 FLUKA

```bash
export C8_FLUPRO=/path/to/fluka
export FLUPRO="$C8_FLUPRO"
export FLUFOR=gfortran
test -f "$C8_FLUPRO/libflukahp.a"
```

FLUKA binary 必须匹配服务器的 glibc 和 Fortran runtime。
`-DWITH_FLUKA=ON` 在本分支是 fail-closed：找不到库时 CMake 直接失败，不会
静默换成 UrQMD。`FLUPRO` 和 `FLUFOR` 同时也是运行期环境变量；每次重新登录
服务器后、启动 `c8_air_shower` 前都要再次导出。仅仅链接成功并不代表运行期
能够找到 FLUKA 数据目录。

### 7.4 自动取得 GPU architecture

CMake 把 compute capability `8.9` 写成 `89`：

```bash
export C8_CUDA_ARCHS="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
  tr -d ' ' |
  sed 's/\.//g' |
  sort -u |
  paste -sd';' -
)"

printf 'CUDA architectures: %s\n' "$C8_CUDA_ARCHS"
case "$C8_CUDA_ARCHS" in
  ''|*[!0-9\;]*)
    printf '无法自动取得有效的 CUDA architecture。\n' >&2
    false
    ;;
esac
```

若 NVCC 不认识新 GPU 的 architecture，应升级 toolkit，不能把一个无关旧架构
当成正式性能构建。不要添加 `--use_fast_math`，否则现有数值验收失效。

### 7.5 创建统一工作目录、取得源码并安装 Release 依赖

推荐先创建一个独立的总目录，再把源码、构建目录和安装目录并列放在其中，例如：

```text
~/corsika-21cma-cuda/
├── corsika8_gpu_refactor/              # Git 源码
├── corsika8_gpu_refactor_build_cuda/   # CMake 构建文件
└── corsika8_gpu_refactor_install_cuda/ # 安装后的程序和资源
```

先创建总目录，并明确 clone `cuda-em-icrc2025-beta4` 分支。私有仓库推荐使用
已经加入 GitHub 账户的 SSH key：

> **部署状态：**`cuda-em-icrc2025-beta4` 当前仍是本地工作分支，尚未发布到
> 私有 GitHub remote。下面的 clone 命令需等代码审查并 push 后才会生效；在此
> 之前应使用这份完整 checkout，或完整传输 beta4 源码树。仅 clone beta2 不会
> 包含上文列出的修复。

```bash
export C8_WORKSPACE=~/corsika-21cma-cuda
mkdir -p "$C8_WORKSPACE"
cd "$C8_WORKSPACE"

ssh -T git@github.com

git clone \
  --branch cuda-em-icrc2025-beta4 \
  --single-branch \
  git@github.com:BossL668/corsika8-gpu-hybrid.git \
  corsika8_gpu_refactor
```

GitHub 在 SSH 验证成功后会提示不提供 shell access，这是正常现象。不能使用 SSH
时，可以在新 Conda 环境安装并认证 GitHub CLI：

```bash
conda install -y -c conda-forge gh
gh auth login
gh auth status

cd "$C8_WORKSPACE"
gh repo clone BossL668/corsika8-gpu-hybrid \
  corsika8_gpu_refactor \
  -- --branch cuda-em-icrc2025-beta4 --single-branch
```

这里不要给 GitHub clone 增加 `--recursive`。上游 `.gitmodules` 使用针对 KIT
GitLab origin 的相对 URL；从 GitHub fork clone 时会被错误解析成不存在的 GitHub
仓库。应先把两个子模块指向公开的上游地址，再初始化：

```bash
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"

git -C "$C8_SOURCE" config \
  submodule.modules/data.url \
  https://gitlab.iap.kit.edu/AirShowerPhysics/corsika-data.git
git -C "$C8_SOURCE" config \
  submodule.modules/conex.url \
  https://gitlab.iap.kit.edu/AirShowerPhysics/cxroot.git

git -C "$C8_SOURCE" submodule update --init --recursive --jobs 8
git -C "$C8_SOURCE" submodule status --recursive

test -f "$C8_SOURCE/modules/data/CMakeLists.txt"
test -f "$C8_SOURCE/modules/conex/cxroot/CMakeLists.txt"
test "$(git -C "$C8_SOURCE" branch --show-current)" = \
  cuda-em-icrc2025-beta4
```

随后统一定义三个目录：

```bash
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"
export C8_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cuda"
export C8_INSTALL="$C8_WORKSPACE/corsika8_gpu_refactor_install_cuda"

# 同时根据可用内存和核心数设置；内存较小的笔记本应从更小值开始。
export C8_BUILD_JOBS=16
export C8_CONAN_JOBS="$C8_BUILD_JOBS"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="$CXX"

test -f "$C8_SOURCE/conanfile.py"
test -f "$C8_FLUPRO/libflukahp.a"

"$C8_SOURCE/conan-install.sh" \
  --source-directory "$C8_SOURCE" \
  --release

test -f "$C8_SOURCE/conan_cmake/conan_toolchain.cmake"
conan profile show -pr corsika8
```

当前 `conan-install.sh` 会自动从 `third_party/conan/` 导出并解析两个版本锁定包：

```text
cubicinterpolation/0.1.5@c8gpu/stable
proposal/7.6.2@c8gpu/stable
```

补丁只增加只读导出 API，不改变普通 CPU 求值、缓存生成或随机数状态；但它们会
扩展 C++ 类布局，因此与未打补丁的 vanilla 二进制不具备 ABI 兼容性。迁移时
必须保留 `third_party/conan/`，修改任一 recipe 后要使用干净的 Conan/CMake
构建目录，不能混用 vanilla header、library 或旧 object。标准构建流程不需要
手工执行 `conan create`；独立依赖审计命令见
[`third_party/conan/README.md`](third_party/conan/README.md)。

迁移到其他服务器时，只需把 `C8_WORKSPACE` 改成服务器上的绝对父目录，建议保持
三个子目录的名称不变。不要把构建文件写入源码目录，也不要复制复用另一台机器、
另一 CUDA toolkit、编译器或 GPU 架构产生的构建目录。

### 7.6 配置、编译和安装

```bash
# testModules 会初始化 FLUKA；若新 shell 丢失运行期变量，应在这里明确失败。
test -f "$FLUPRO/libflukahp.a"

cmake \
  -S "$C8_SOURCE" \
  -B "$C8_BUILD" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=$C8_CUDA_ARCHS" \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUPRO/libflukahp.a" \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"

cmake --build "$C8_BUILD" --parallel "$C8_BUILD_JOBS"
cmake --install "$C8_BUILD"
```

CUDA 编译会消耗较多主机内存，`--parallel` 应同时考虑核心数和 RAM。CMake 会在
第一次配置时缓存编译器和 CUDA 选择；如果这些选择需要改变，应创建新的空构建
目录，不要尝试修补旧 cache。安装后检查：

```bash
for program in \
  c8_air_shower \
  gpu_em_table_prepare \
  gpu_em_tablegen \
  cuda_decision_replay \
  fluka_batch_worker; do
  test -x "$C8_INSTALL/bin/$program" || {
    printf '缺少安装程序：%s\n' "$program" >&2
    exit 1
  }
done

test -d "$C8_INSTALL/share/corsika/data/PROPOSAL"
test -f "$C8_INSTALL/share/corsika/GeoMag/IGRF14.COF"
test -f "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml"

if ldd "$C8_INSTALL/bin/c8_air_shower" | grep -q 'not found'; then
  ldd "$C8_INSTALL/bin/c8_air_shower"
  false
fi

"$C8_INSTALL/bin/c8_air_shower" --help >/dev/null
```

### 7.7 构建后测试

```bash
# 先运行直接覆盖 GPU/CUDA 的快速子集。
ctest \
  --test-dir "$C8_BUILD" \
  -R '^testGpu' \
  --output-on-failure \
  --parallel "$C8_BUILD_JOBS"

# 正式科研生产前再运行完整 C++/CUDA 测试集。
ctest \
  --test-dir "$C8_BUILD" \
  --output-on-failure \
  --parallel "$C8_BUILD_JOBS"

cd "$C8_SOURCE"
python -m unittest discover \
  -s validation/gpu_em/tests \
  -p 'test_*.py'
```

### 7.8 在新 shell 恢复运行环境

注销后 Conda activation 和 export 变量都会消失。重新制表或运行 shower 前，应
恢复安装路径以及 FLUKA 运行期数据目录：

```bash
conda activate corsika_venv

export C8_WORKSPACE=~/corsika-21cma-cuda
export C8_SOURCE="$C8_WORKSPACE/corsika8_gpu_refactor"
export C8_BUILD="$C8_WORKSPACE/corsika8_gpu_refactor_build_cuda"
export C8_INSTALL="$C8_WORKSPACE/corsika8_gpu_refactor_install_cuda"
export C8_FLUPRO=/path/to/fluka

export FLUPRO="$C8_FLUPRO"
export FLUFOR=gfortran
export CORSIKA_DATA="$C8_INSTALL/share/corsika/data"
export PATH="$C8_INSTALL/bin:$PATH"

test -x "$C8_INSTALL/bin/c8_air_shower"
test -f "$FLUPRO/libflukahp.a"
nvidia-smi
```

安装程序已经包含指向 `$C8_INSTALL/lib/corsika` 的 runpath，通常不需要手动修改
`LD_LIBRARY_PATH`。随意加入其他系统或 Conda library 目录反而可能破坏
Fortran/FLUKA runtime 的一致性。

### 7.9 运行安装树的端到端冒烟测试

单元测试能证明 CUDA kernel 可以执行，但不会初始化完整 shower。正式生成高能
生产表之前，先制作一张 100 GeV 小表并运行一个安装后的质子事例。首次制表会
调用 PROPOSAL，可能需要数分钟；相同请求以后直接复用缓存。

```bash
export C8_TABLE_CACHE="$C8_WORKSPACE/gpu_em_table_cache"
export C8_SMOKE_TABLE="$(
  "$C8_INSTALL/bin/gpu_em_table_prepare" \
    --medium-yaml "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e11 \
    --energy-margin 1.05 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 5e-4 \
    --loss-tolerance 5e-4 \
    --nonmonotonic-loss-policy proposal-monotone \
    --print-path-only
)"

test -f "$C8_SMOKE_TABLE"

export C8_SMOKE_ANTENNAS="$C8_WORKSPACE/antennas_smoke.txt"
printf '100 0 0\n' > "$C8_SMOKE_ANTENNAS"

# c8_air_shower 启动前输出路径不能已经存在。
export C8_SMOKE_OUTPUT="$C8_WORKSPACE/smoke_$(date +%Y%m%d_%H%M%S)"

"$C8_INSTALL/bin/c8_air_shower" \
  --pdg 2212 \
  --energy 100 \
  --seed 40077 \
  --filename "$C8_SMOKE_OUTPUT" \
  --emthin 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  --antenna-file "$C8_SMOKE_ANTENNAS" \
  --ring 0 \
  --em-backend cuda \
  --radio-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 128 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_SMOKE_TABLE" \
  --gpu-table-tolerance 5e-4 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true

test -f "$C8_SMOKE_OUTPUT/summary.yaml"
test -f "$C8_SMOKE_OUTPUT/gpu_em/summary.yaml"
test -f "$C8_SMOKE_OUTPUT/CoREAS/observers.parquet"
test -f "$C8_SMOKE_OUTPUT/ZHS/observers.parquet"
grep -q 'complete: true' "$C8_SMOKE_OUTPUT/gpu_em/summary.yaml"
```

该命令必须以状态 0 结束，而且 `gpu_em/summary.yaml` 中应存在非零的光子/轻子
GPU 步和射电轨迹。它只验证迁移和执行链，不能替代多随机种子的 CPU/CUDA 物理
验收。

#### NVIDIA T400 4 GB 迁移实测

2026-08-06 已在 NVIDIA T400 4 GB（compute capability 7.5、driver
595.71.05）上完整执行工作目录、配置、构建、制表和运行流程。由于服务器没有
认证私有 Git remote，本次把同一份 beta2 源码快照复制到服务器，因此源码 clone
的身份认证不属于本次测试范围。隔离的 Conda 环境使用 Python 3.9.23、CMake
3.31.8、Conan 2.11、系统 GCC/G++/GFortran 13.3，以及本节列出的最小 CUDA
12.6.3 包集合；以 `CMAKE_CUDA_ARCHITECTURES=75` 完成了 Release CUDA+FLUKA
配置、编译和安装。

27 项针对性 GPU 测试全部通过，覆盖物理表读取、Hybrid 路由、CPU fallback、
真实 CUDA kernel、wavefront 队列、过程与末态抽样、LPM、薄化、多重散射、磁场
和球形大气输运、μ 子、光子以及 CUDA 射电投影。随后现场生成并成功加载了一张
制表合同 0.18、上限 105 GeV 的干空气表；在目标容差 `1e-3` 下，实测最大 rate
误差为 `9.996133e-4`，inverse-CDF 误差为 `8.541396e-4`。

最后运行了一个 100 GeV 质子端到端冒烟事例，实际启用 SIBYLL、FLUKA、
PROPOSAL、IGRF14/2027、CUDA EM、CUDA 射电和 81 个外部天线。程序以状态 0
正常结束，耗时 6.94 s，并写出了完整的顶层、GPU、CoREAS 和 ZHS 输出组。
GPU summary 记录 3,643 个光子步、31,534 个带电轻子步、5,497 个 GPU 末态、
29,774 条射电轨迹、0 次 CPU fallback，设备峰值分配为 344.4 MiB。该结果证明
beta2 能够在 T400 上完成构建与执行迁移，但它只是冒烟测试，不是 CPU/CUDA
物理一致性或性能验收。该低能事例的 CUDA 专用能量账本注明 coverage 不完整，
普通总能量预算差为 -3.51%，因此不能把它引用为高精度能量闭合结果。

这次迁移暴露出的两个构建问题已由当前源码处理：普通 C++ consumer 能够获得
公开 GPU header 所需的 CUDA include 路径；Conda compatibility sysroot 中的
`libm`/`librt` 会替换成匹配的系统 multiarch 库。Pythia 8.315 也已改用官方
GitLab release archive，因为原 `pythia.org/download` 压缩包地址已经返回 404。

### 7.10 准备正式生产用干空气物理表

物理表不是 GPU 架构文件。换显卡或 CUDA 架构不需要重新生成。最简单且可靠的
方式是使用下文的 `gpu_em_table_prepare` 自动查找或生成内容寻址表。旧的
`production_v10_muons_1e-3_1EeV.c8emrt` 表产生于切分 cut 合同之前，不能用于
当前版本。一个兼容的默认表应满足：

- CORSIKA `AirDry1Atm` 标准干空气；
- `--emcut 0.0005` GeV；
- `--mucut 0.3` GeV；
- 初级总能量不高于 \(10^{18}\) eV；
- CUDA 电磁和可选 CUDA \(\mu^\pm\) 输运。

如果不需要 CUDA μ 子输运，也可以生成只含 \(\gamma/e^\pm\) 的表；
表中没有成对的 PDG `13/-13` 时，μ 子保留在 CPU 路径。

默认 `c8emrt` 模式不会在缺表时自动执行生成器。应在运行 shower 前显式调用
`gpu_em_table_prepare`；它会自动查找兼容表，并在未命中时通过跨进程锁生成、
回读校验和写 manifest。这样避免生产事件隐式制表，同时允许多任务共享缓存。
查找还会核对制表器合同版本；旧生成器产生的表不会被静默复用。
若只改变 GPU 型号、GPU 数量、磁场、天线、观测高度或同一干空气的密度
profile，直接复用表。

如果明确选择 `--gpu-physics-source proposal-native`，则不传
`--gpu-table-cache`；首次新介质/cut 仍可能由 PROPOSAL 建立自己的 cache，
但用户无需单独运行完整制表工具。不支持的参数化或轴会直接终止，当前尚未通过
完整系综/性能门禁，因此不能替代默认生产路径。
当前 canonical-v6 完整空气表包含 58 个随机过程列，占用 78,129,000 bytes
显存；同一进程只上传一次，后续 shower 按相同哈希复用。第一阶段每个 backend
只支持一种化学组成：相同干空气组成的分层密度 profile 可以使用，不同空气、
岩石或冰组成混合的 geometry 会在启动时明确拒绝。

当前正式干空气原生 artifact 为：

```text
PROPOSAL / CubicInterpolation: 7.6.2 / 0.1.5
canonical format:              v6
native table SHA-256:          7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f
auxiliary SHA-256:             5d389cde09fb75bf4d53475ef7f8cdebaa2993923c8e9a720df4fd0af7c67c9f
rate columns / nodes:          58 / 550000
device bytes:                  78129000
```

这些哈希只标识当前依赖、干空气组成、cut 和辅助算法合同，并不是其他介质或
构建也应具有的固定常数。输出 metadata 会记录实际链接版本、原生/辅助表哈希、
节点和显存字节数、PROPOSAL cache 命中、Newton/bisection、inverse failure
以及 selection replay 计数。
以下列表和后面的 `10^19` eV 示例专指 `.c8emrt` 模式；这些变化必须生成并
重新验收完整表。`proposal-native` 不要求手工制表，而会让 PROPOSAL 自动建立
对应 cache、重新导出原生样条并产生新的 canonical hash，但仍需要对新物理配置
重新验收：

- PROPOSAL 版本或物理参数化；
- 介质组成、组分比例或材料常数；
- `--emcut`、`--mucut`；
- 所需最大能量或表格式版本。

例如缓存中的旧表只到 \(10^{18}\) eV。为 \(10^{19}\) eV 质子准备新表：

```bash
export C8_TABLE_CACHE=/path/to/c8_gpu_table_cache
export C8_TABLE="$(
  "$C8_INSTALL/bin/gpu_em_table_prepare" \
    --medium-yaml "$C8_INSTALL/share/corsika/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e19 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 5e-4 \
    --loss-tolerance 5e-4 \
    --print-path-only
)"
```

默认 1.05 安全系数使上限为 \(1.05\times10^{13}\) MeV。首次严格制表可能耗时
很长；同一请求随后直接命中缓存。生成成功不替代新能区的 shower 和射电验收。

这里的 `--em-cut-MeV` 是 CORSIKA 面向用户的产生/输运 cut。准备工具会自动
复用标量 PROPOSAL 的标准缓存选择规则：例如用户设置 0.5 MeV 时，电子和光子
输运 cut 仍是 0.5 MeV，但随机 PROPOSAL 过程表使用不高于该阈值的最近标准值
0.4 MeV；manifest 会同时记录这两个数值。这个区别会直接影响离散 μ 子电离率，
不能把命令中的 0.5 手工改成 0.4。旧制表流程若把两者都设为 0.5 MeV，新版
`c8_air_shower` 会明确拒绝该表，必须用当前 `gpu_em_table_prepare` 重新准备。

#### PROPOSAL 插值与氩组分轫致辐射列

PROPOSAL 7.6.2 的缓存插值与逆求解器，在干空气中电子/正电子对氩靶组分的
轫致辐射随机能损逆 CDF 上可能给出很窄的局部非单调区间。这里的“氩”是靶
组分，不是入射氩核；关闭 PROPOSAL 插值并对相同点进行直接数值积分和求根后，
该区间恢复单调。因此这是原 PROPOSAL 插值/数值反演误差，不是氩的物理异常。
原标量路径不会跨相邻分位点检查单调性，通常直接接受插值求解器返回的结果。

一个干空气诊断点为 \(E=623.7318908\) MeV：分位点从 0.9859885644 增加到
0.9859897698 时，插值能损比例由 0.8847353442 反向降至 0.8836791531，局部
反转约 0.119%；相同两点的非插值直接计算则从 0.8592897295 单调增加到
0.8593004245。这些数值用于记录数值问题，不应解释成对氩物理过程的新修正。

beta2、beta3 和 beta4 通过 `--nonmonotonic-loss-policy` 提供两种离线策略：

- `proposal-monotone`（默认）仍从原标量程序使用的 PROPOSAL 缓存插值出发，只把
  局部下降投影到此前的累计最大值；发现随能量移动的反转后，该高分位分支不再
  驱动能量网格细化，从而保持跨能量表面平滑。该列标记为
  `proposal_interpolated_monotone`。它最接近原标量且表较小；这里明确把 0.119%
  的局部反转视为插值噪声，因此表格容差描述的是对“修复后单调参考”的逼近，
  而不是逐点复刻原始反转。
- `proposal-direct` 会关闭 PROPOSAL 插值，用直接积分/求根重建受影响的完整列，
  写入 `proposal_direct` 并对直接参考验收。该策略适合物理诊断，但代价很大：
  在 0.4 MeV--105 GeV、\(10^{-3}\) 的干空气测试中，仅这一列就提出超过五万个
  能量节点，因此不作为紧凑默认值。它的独立节点预算由
  `--direct-loss-max-energy-points` 控制，默认 65536。

已完成的 0.4 MeV--105 GeV 干空气 `proposal-monotone` 测试表大小为 3.69 MB；
rate 与“修复后参考”的 inverse-CDF 实测误差分别为
\(9.996\times10^{-4}\) 和 \(8.541\times10^{-4}\)。审计日志同时记录最大局部
单调投影 0.924%、最大原始移动分支偏差 9.90%。后两个数描述被拒绝的 PROPOSAL
数值分支，不属于插值误差声明；正式使用仍需继续做 CPU/CUDA shower 统计验收。

同一制表合同已在 beta2、beta3 和 beta4 中完成 100 GeV 电子初级的 CUDA EM +
CUDA 射电 smoke test；这些运行均未出现轫致辐射/氩 selected-loss 回退。已记录
的少量 CPU 返回属于既定物理路径（例如光致强子和分位边界返回），不是本节的
PROPOSAL 氩插值问题。

修复后的性能复查使用 12 个配对随机种子：100 GeV 垂直电子初级、
`emthin=1e-3`、0.5 MeV 输运 cut、IGRF14/2027，并在同一块 RTX 4060 Laptop GPU
上同时启用 CUDA EM 与 CUDA 射电。干净 Release 重建后，beta1、beta2、beta3
的 shower 内部时间均值/中位数分别为 0.731/0.715 s、0.684/0.594 s 和
0.737/0.670 s；beta3 均值与 beta1 相差 0.8%，beta2 均值低 6.5%。三者有活动时
的采样 GPU 利用率均值分别为 19.2%、25.5% 和 24.2%。beta2/3 均未再出现氩轫致
辐射回退；12 个 shower 中
只剩 1 次和 4 次非氩过程的低频分位边界返回。因此此前约两倍的低能运行时间
回退已经消失；这组数据用于性能核查，不替代 shower observable 的统计验收。

若要逐位复刻原标量中的原始反转，则需要把 PROPOSAL 插值器和逆求解器本身移植
到 CUDA，不能依赖平滑二维表。简单地按粒子或过程多开进程也不能消除 direct
策略的主要成本，因为密集细化集中在单个氩轫致辐射列内部；对行采样做并行需要
先完成 PROPOSAL 线程安全审计，留作后续优化。

当前 `gpu_em_tablegen` 本身是串行实现，设置 `OMP_NUM_THREADS` 不会令单张表加速。
制表属于离线成本，内容寻址缓存会避免相同介质、cut 和能区请求被重复生成；只有
在服务器策略和内存允许时，才应把互不相同的制表请求作为独立任务并行运行。

两种已实现策略都只在制表阶段处理该问题，shower 运行期仍在 GPU 上采样，氩
组分不会触发 selected-loss CPU 回退，也不会切碎 EM wavefront。反应率以及过程/
靶组分选择保持不变。策略名和网格预算都会进入内容寻址请求，制表器合同 `0.18`
会阻止旧的含回退表被静默复用。

介质 YAML、自动准备、完整参数以及 \(10^{19}\) eV 质子新表命令见
[`gpu_em_tables/README.md`](gpu_em_tables/README.md)，全部开关见
[`cli_reference.md`](documentation/cuda_em_refactor/cli_reference.md)。

### 7.11 典型全加速运行

典型全加速运行参数：

```bash
c8_air_shower \
  -p 2212 -E 100000 -N 50 \
  -f /path/to/output \
  --seed 10200001 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --max-weight 100 \
  --em-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_TABLE" \
  --gpu-table-tolerance 5e-4 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true \
  --radio-backend cuda \
  --gpu-radio-field-limit 1 \
  --hadronic-backend fluka-process \
  --hadronic-workers 4 \
  --hadronic-min-batch 64 \
  --hadronic-target-batch-ms 5 \
  --hadronic-max-batch 256
```

性能模式不要启用：

```text
--gpu-detailed-stage-timing
--gpu-full-step-records
--gpu-radio-track-diagnostics
```

它们用于诊断，会增加同步、归约或输出开销。

### 7.12 使用原生 PROPOSAL 物理源

保留应用全部默认参数、同时开启 CUDA 粒子输运、CUDA CoREAS/ZHS 和
`proposal-native` 的最简单实用命令为：

```bash
"$C8_INSTALL/bin/c8_air_shower" \
  -p 2212 -E 1e5 \
  -f /path/to/new_output \
  --antenna-file /path/to/antennas.txt \
  --em-backend cuda \
  --radio-backend cuda \
  --gpu-physics-source proposal-native
```

程序本身真正必填的是初级粒子、正的初级能量和一个尚不存在的输出目录。这里
仍显式给出天线文件，因为没有有效 observer 的 CUDA 射电计算没有科研意义。
若当前工作目录已经存在有效的默认 `antennas.txt`，可以进一步缩短为：

```bash
"$C8_INSTALL/bin/c8_air_shower" \
  -p 2212 -E 1e5 -f /path/to/new_output \
  --em-backend cuda --radio-backend cuda \
  --gpu-physics-source proposal-native
```

这条命令保留当前应用默认值：单事例、垂直入射、方位角 0、seed 0、默认
`emcut/hadcut/mucut/taucut`、`emthin=1e-6`、自动 `max-weight`、IGRF14/2027、
GPU 0、`gpu-min-batch=4096`、当前空闲显存的 70%、确定性 CUDA 随机数以及
标量强子后端。这里的“CUDA 全加速”专指已经实现的 photon/lepton 输运和
CoREAS/ZHS 射电投影；高能强子过程及未支持末态仍由 CPU 处理。

如果要显式固定生产参数和缓存位置，可以使用下面的完整形式；它与上面的最短
命令使用同一原生物理源：

```bash
c8_air_shower \
  -p 2212 -E 100000 -N 1 -z 0 -a 0 \
  --seed 2026085001 \
  -f /path/to/native_output \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --ring 0 \
  --antenna-file /path/to/antennas.txt \
  --em-backend cuda \
  --gpu-physics-source proposal-native \
  --gpu-aux-cache-dir ~/.cache/corsika8/gpu-em-aux \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-tolerance 5e-4 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true \
  --radio-backend cuda \
  --gpu-radio-field-limit 1
```

此模式不能再传 `--gpu-table-cache`。首次遇到新的介质/cut 时，PROPOSAL 可能
先建立自己的 cache；`.c8emaux` 也会加锁自动生成。后续进程命中 cache，同一
进程内相同 canonical hash 的 shower 还会复用已上传的设备表。`--max-weight`
是否出现会改变 thinning 语义；CPU、`c8emrt` 和 native 系综比较时必须三者
一致，省略与显式设置不能混用。

当前验收结论是：

- 完整 CPU transport tape 的 57,313 条记录在 GPU 上 ordered hash 完全一致，
  CoREAS/ZHS 最坏 relative (L_2) 约为 `1.1e-7/1.9e-7`；
- 固定语义 decision 的 20,480 顶点与 live PROPOSAL 过程、组分和
  `SampleLoss` mismatch 为 0；
- 新的 1 TeV native 2000 例相对已保存 CPU 摘要，列出的 11 个均值偏移全部
  比上一批 native 更小，例如 total-EM profile integral 从 `+0.377%` 变为
  `+0.097%`，charged (X_{\max}) 从 `-2.060%` 变为 `-0.114%`；
- 新旧 native 的 longitudinal/ground curve gate 全部通过，但地面 EM 动能的
  KS 距离 `0.0530` 高于 95% 临界值 `0.0430`；
- PSR 上 CPU raw shard 恢复访问后，仍要补做逐深度、KS、ground timing 和射电
  的直接比较；当前只能称为本地阶段验收；
- 与既有 100 TeV CPU 参考完全相同 2000 个 seed 的 native 任务已于
  2026-09-01 启动，完成前不引用其物理或性能结论。

完整代码门禁和逐随机数诊断分别见
[Phase 113](documentation/cuda_em_refactor/phase_113_proposal_native_gpu_tables_CN.md)
与
[Phase 114](documentation/cuda_em_refactor/phase_114_beta4_proposal_native_decision_tape_replay_CN.md)。

## 8. 与官方最新公开状态的比较

### 8.1 比较基线和方法

查询日期为 2026-07-31。

- 官方标签页把
  [`corsika8-v1.0-beta1`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/tags/corsika8-v1.0-beta1)
  标为 2024-12-24 的首个公开 beta；
- 其后有 2025-06-24 的
  [`icrc2025-v1`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/tags)
  研究基线，但标签页没有把它写成新的正式 release；
- 查询时官方 `main` 为
  [`77bedbaf`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/commit/77bedbafec10782018390e144bef258de919fc24)，
  提交日期 2026-07-08；
- 对该 commit 的公开源码树检索
  `CORSIKA_ENABLE_CUDA`、`CORSIKA8GpuEm`、`--em-backend`、
  `--radio-backend`、`cuda_decision_replay` 和 `fluka-process`，未发现本分支
  对应的 CUDA EM/射电构建目标和 CLI；
- 官方 2025 射电论文确认 CORSIKA 8 原生支持同一 shower 同时计算 CoREAS
  endpoint 与 ZHS：
  [arXiv:2409.15999](https://arxiv.org/abs/2409.15999)；
- 官方 2026 总览论文展示通用框架、空气簇射验证、跨介质和冰中射电：
  [arXiv:2604.01850](https://arxiv.org/abs/2604.01850)；
- 官方生态也有 GPU 光学 Cherenkov/fluorescence 方向，但它与本分支的
  $\gamma/e^\pm/\mu^\pm$ 粒子输运不是同一问题：
  [arXiv:2405.04229](https://arxiv.org/abs/2405.04229)。

“官方公开源码未发现”只描述上述 commit 和检索范围，不能排除未公开分支或其他
合作组原型。因此下表使用“公开主线中未见同类接口”，不使用“世界首个”等无法
证明的表述。

### 8.2 能力矩阵

| 能力 | 官方公开版/main | 本地 GPU 分支 | 判断 |
|---|---|---|---|
| C++17、单位、模块化 ProcessSequence | 官方核心能力 | 完整继承 | 官方基础优势 |
| 标量 Cascade + PROPOSAL | 支持 | 默认路径保留 | 等价基础 |
| CoREAS endpoint + ZHS 同 shower | 官方论文和源码支持 | CPU 路径继承 | 官方基础能力 |
| 通用环境、跨介质、冰中级联/射电 | 2026 官方论文展示 | CPU 框架继承；CUDA snapshot 尚限五层大气 | 官方更完整 |
| GPU 光学 photon 传播研究 | 官方生态已有研究 | 不是本分支目标 | 方向不同 |
| $\gamma/e^\pm$ CUDA 粒子输运 | 公开 main 未见同类构建/CLI | 已集成 HybridCascade | 本分支特色 |
| $\mu^\pm$ GPU 连续输运和顶点竞争 | 公开 main 未见同类接口 | v10 表下已集成，末态回 CPU | 本分支特色 |
| resident CUDA CoREAS/ZHS | 公开 main 为通用 RadioProcess，未见 CUDA backend CLI | 可选定点 CUDA accumulator | 本分支特色 |
| 调度无关 Philox history RNG | 未见对应公开设备接口 | 已实现 | 本分支特色 |
| 原版 decision tape → CUDA 精确回放 | 未见对应公开工具 | 已实现 | 本分支特色 |
| 版本化 PROPOSAL GPU 表和 fail-closed hash/tolerance | 未见对应公开设备表 | 已实现 | 本分支特色 |
| FLUKA 低能强子 | 支持标量 FLUKA | 继承并增加进程池批处理 | 本分支调度扩展 |
| 单 shower 跨多 GPU | 未公开声明 | 尚不支持 | 双方均非当前能力 |
| 上游 CI、用户群、通用维护和官方物理结论 | 官方具备 | 本地工作树，尚未上游评审 | 官方显著更强 |

### 8.3 可以合理称为“领先”的具体维度

在“已公开官方 main 与本地当前代码”的限定比较下，本分支的前沿点是：

1. 把真实 PROPOSAL 电磁和部分 μ 子输运，而不是仅光学 photon，集成进
   CORSIKA 8 shower 主循环的 resident CUDA wavefront；
2. 通过 history-keyed Philox、stable radix sort 和 exclusive scan，使并行
   调度具备可审计的确定性；
3. 用版本化物理表、指定末态 fallback 和 hard-failure 边界避免“GPU 加速但
   静默少算物理”；
4. 让 CoREAS/ZHS 可在同一设备端随轨迹在线累计，并保留 CPU 同轨迹验收路径；
5. 把强子瓶颈拆成独立 FLUKA 进程池，与 GPU 轻子前沿形成混合流水；
6. 提供 exact replay、能量账本、provenance、独立 ensemble 和射电脉冲统计
   组成的多层验证体系。

这些是技术实现维度的领先，不等于总体成熟度领先。官方版本在通用几何、跨介质、
冰中级联、上游维护、协作评审和可引用物理结论方面仍有明显优势。

## 9. 当前限制

- CUDA 环境仅覆盖当前五层球形大气；山体/月壤/冰需要新 snapshot 和新表；
- 强子物理没有变成 GPU kernel，FLUKA 加速仍依赖 CPU 多进程；
- μ 稀有离散末态和实际衰变末态仍在 CPU；
- 单 shower 不能跨多 GPU；多 GPU 服务器应一块卡运行一个独立进程；
- 不同 GPU 架构只承诺统计一致，不承诺浮点逐位一致；
- production CUDA 与原版 CPU 相同 seed 不承诺逐粒子同一 shower；
- `.c8emrt` 只在记录的粒子、介质、cut、能区和误差范围内有效；
- `proposal-native` 第一阶段每个 backend 只接受一种化学组成；混合空气、岩石、
  冰的原生表会在输运前拒绝；
- 原生正式介质百万点 oracle 仍保留一个高能 Compton `v` 严格门禁 warning，
  CPU raw 2000 例直接验收以及 100 TeV/1 PeV 热缓存性能门禁尚未完成；
- CUDA radio 的定点范围必须预先配置并检查 overflow；
- 本地工作树包含尚未上游合并的实现，迁移前必须保存精确 commit/patch、二进制
  和表哈希；仅从官方仓库重新 clone 不会得到这些 GPU 功能。

## 10. 验证路线

迁移或修改 kernel 后至少执行：

```text
1. CUDA/CPU CTest
2. Python validation 单元测试
3. 同 GPU 固定 seed 重复性
4. 单过程 rate/v/角度/末态抽样
5. tracking 边界、grammage、时间和磁场对照
6. energy ledger 与 illegal fallback 检查
7. CPU/CUDA independent shower ensemble
8. Xmax、纵向粒子数、能量沉积和地面分布
9. CoREAS/ZHS 地磁振幅和脉冲宽度分布
10. 冷/热缓存分开的五次中位数性能
```

主要工具位于 [`validation/gpu_em`](validation/gpu_em/)，说明见
[`validation/gpu_em/README.md`](validation/gpu_em/README.md)。

必须区分三种“相同”：

- 同一 CUDA 配置重复：应确定性相同；
- 原版 decision tape 与 CUDA replay：指定轨迹应逐记录一致；
- 原版 CPU 与 production CUDA 独立抽样：比较统计分布，不要求逐事例相同。

## 11. 代码阅读顺序

建议按以下顺序理解实现：

1. [`ScalarCascadeStepper.hpp`](corsika/framework/core/ScalarCascadeStepper.hpp)
   和对应 `.inl`：原标量单步蒙特卡洛；
2. [`HybridCascade.hpp`](corsika/framework/core/HybridCascade.hpp)：CPU/GPU
   调度边界；
3. [`PhysicalCudaEmRouter.hpp`](corsika/gpu/em/PhysicalCudaEmRouter.hpp)：
   粒子转换、fallback、输出和强制衰变；
4. [`CudaEmRunSession.hpp`](corsika/gpu/em/detail/CudaEmRunSession.hpp) 与
   [`CudaHybridCascadeRunner.hpp`](corsika/gpu/em/detail/CudaHybridCascadeRunner.hpp)：
   不依赖空气应用的物理源生命周期和通用 HybridCascade 接线；
5. [`CudaAirShowerSetup.hpp`](applications/detail/air_shower_cuda/CudaAirShowerSetup.hpp)
   与 `CudaAirShowerRunner.hpp`：空气模型专属 snapshot、registry、factory 和报告回调；
6. [`Types.hpp`](corsika/gpu/em/Types.hpp)：设备 POD 和记录格式；
7. [`Philox.hpp`](corsika/gpu/em/Philox.hpp)：调度无关随机数；
8. [`CudaEmBackend.cu`](src/gpu/em/CudaEmBackend.cu)：常驻队列和 wavefront；
9. photon/lepton selection、transport 和 final-state CUDA 文件；
10. [`RateTable.hpp`](corsika/gpu/em/tables/RateTable.hpp) 与
   [`gpu_em_tablegen.cpp`](applications/gpu_em_tablegen.cpp)；
11. [`ProposalNativeTable.hpp`](corsika/gpu/em/tables/ProposalNativeTable.hpp)、
   [`ProposalNativeTableExporter.hpp`](corsika/gpu/em/tables/ProposalNativeTableExporter.hpp)
   和 [`CudaProposalNativeTable.cu`](src/gpu/em/CudaProposalNativeTable.cu)；
12. [`third_party/conan`](third_party/conan/) 的版本锁定依赖补丁；
13. [`CudaRadioAccumulator.cu`](src/gpu/em/CudaRadioAccumulator.cu)；
14. [`fluka_batch_worker.cpp`](applications/fluka_batch_worker.cpp)；
15. [`validation/gpu_em`](validation/gpu_em/) 的物理与性能验收工具。

## 12. 引用与科研表述

使用本分支生成科研结果时，应分别引用：

- CORSIKA 8 框架/官方总览论文；
- CORSIKA 8 射电论文；
- PROPOSAL 和所用强子模型；
- FLUKA（若启用）；
- 对本地 CUDA 分支，应记录源码 commit/patch、编译器、GPU/driver/runtime、
  `.c8emrt` SHA-256、CLI、seed、天线文件和验证报告。

在本分支尚未上游发表前，论文中宜写成“基于 CORSIKA 8 的本地 CUDA 研究后端”，
而不是“CORSIKA 8 官方 CUDA 后端”。
