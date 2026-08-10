# beta4 CPU/CUDA 逐过程计算路径复核与完全一致 shower 可行性分析

## 1. 结论先行

本次复核同时检查了源码控制流、当前两张生产表、27 个 GPU 注册测试、逐过程 PROPOSAL oracle、球形大气与磁场输运、cut/thinning、profile、CoREAS/ZHS，以及同一种子的独立 CPU/CUDA shower 和 CPU 决策带 replay。结论分为三层：

1. **没有发现仍被静默跳过的 CORSIKA 过程。** `c8_air_shower` 的 14 个过程节点均被 CUDA 兼容性注册表明确标为设备替代、设备记录重放、CPU 延迟执行、不适用于已路由 EM 粒子或仅诊断；未知的 Continuous、Secondaries、Interaction、Decay、Boundary 和 Stack 过程会在 shower 启动前失败。
2. **当前 beta4 的 CUDA 后端与 CPU/PROPOSAL 目标是统计物理一致，而不是同 seed 的逐事件一致。** pair、Compton、photoelectric、brems、ionization、annihilation、electron-pair、三种 LPM、Molière、thinning、球形大气、磁场和相同轨迹的 CoREAS/ZHS 均有直接 oracle 测试并通过。当前表的普通 rate/loss 插值误差约为 `5e-4` 或更小。
3. **仍存在会阻止“完全相同 shower”的确定性差异。** 最主要的是 CPU 的全局顺序随机流和深度优先栈，与 GPU 的 history-keyed Philox 和 wavefront 顺序不同；此外还有表格近似、氩 Brems 非单调区的单调投影、Epair 的近似采样、Molière 插值、CPU/GPU 浮点运算及输出累计顺序。即使给相同 seed，当前两条生产路径也不应产生相同的树、顶点和射电脉冲。

本次还确认并修正了一个**测试夹具问题**：生产表从 0.4 MeV 开始，而光子输运 cut 是 0.5 MeV，测试曾把合法的 `ParticleCut` 终止记录错误送进末态生成器。修改仅位于 `tests/gpu/testGpuPhotonPairFinalState.cpp`，未改变任何生产物理代码。修正后，生产表的 photon pair、Compton 和 photoelectric 末态已完整通过。

宽能区生产表还使原测试中低于正常 CUDA/host double 数值噪声的阈值暴露出来：次级能量曾因 `8.44e-14 GeV` 的差值超过 `3e-14` 阈值，Compton 方向分量曾因 `5.20e-12` 超过 `5e-12` 阈值。测试现分别采用 `1e-12` 相对能量门限和 `1e-10` 方向门限，并**同时输出实测最大误差**，而不是只放宽门限后隐藏结果。生产物理代码仍未改变。

## 2. 本次检查的固定环境

- 源码：beta4 当前工作树；
- CUDA 构建：`corsika8_gpu_refactor_build_cuda`；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9；
- PROPOSAL：7.6.2；
- 介质：`air_dry_1_atm`，包含 N、O、Ar；
- 表格式：C8EMRT format 10，generator contract `c8-gpu-em-tablegen-0.18`；
- stochastic cut：0.4 MeV；EM transport cut：0.5 MeV；muon transport cut：300 MeV；
- rate/loss 目标误差：`5e-4`；
- 非单调损失策略：`proposal-monotone`；
- 检查的表能区：
  - 0.4 MeV 至 `1.05e5 GeV`；
  - 0.4 MeV 至 `1.05e8 GeV`。

高能表 manifest 报告的 measured rate error 为 `4.9945e-4`，measured loss error 为 `3.9188e-4`；100 TeV 表对应为 `4.9906e-4` 和 `4.3029e-4`。

## 3. CPU 与 CUDA 的端到端路径

### 3.1 CPU 标量路径

CPU 的过程顺序在 `applications/c8_air_shower.cpp` 中构造：

```cpp
make_sequence(stackInspect, neutrinoPrimaryPythia,
              hadronSequence, decaySequence,
              emCascade, prodprof, emContinuous,
              coreas, zhs, longprof,
              observationLevel, inter_writer,
              thinning, cut);
```

`corsika/detail/framework/core/ScalarCascadeStepper.inl` 对栈顶粒子执行：

1. 计算当前介质中的总截面；
2. 用全局 `cascade` RNG 指数抽取相互作用 grammage 和衰变时间；
3. 由 tracking 求几何边界；
4. 由所有 continuous process 求最大步长；
5. 取相互作用、衰变、几何和连续限制中的最短者；
6. 执行 `doContinuous()`；
7. 执行边界、相互作用或衰变；
8. 按 ProcessSequence 顺序执行 `doSecondaries()`；
9. 继续以 LIFO 深度优先方式处理新次级粒子。

PROPOSAL 在 `corsika/detail/modules/proposal/InteractionModel.inl` 中从全局 `proposal` RNG 取一个 `selection_uniform`，再调用 `Interaction::SampleLoss()`。这个随机数同时决定过程、靶组分和该列中的损失分数。末态、LPM rejection、连续散射和 thinning 又继续消耗各自的顺序随机流。

### 3.2 CUDA 混合路径

`HybridCascade` 仍让强子、tau、CPU-only 末态及不适合设备路径的粒子走 `ScalarCascadeStepper`；可路由的 gamma、electron、positron 和支持的 muon 被转为 `EmParticleState`，交给 `PhysicalCudaEmRouter` 和 `CudaEmBackend`。

GPU 不沿一条 history 走到底，而是反复处理 resident wavefront：

1. `CudaWavefrontBucketing.cu` 按 PID/介质/能段稳定分桶；
2. `CudaInteractionSelector.cu` 查询总 rate，抽取距离；光子同时选过程、靶和损失；
3. `CudaPhotonTransport.cu` 或 `CudaLeptonTransport.cu` 推进到最近限制；
4. 带电粒子在连续能损之后由 `CudaLeptonVertexSelector.cu` 以顶点能量完成过程选择；
5. photon 或 lepton 末态内核生成次级粒子并执行 LPM、cut 和 thinning；
6. 同类与跨类 EM 次级进入 resident 设备队列；
7. 稀有或未设备化末态带着已选定的 process/component/v 延迟返回 CPU；
8. profile、能量沉积和射电信号在设备端累计，必要记录批量回传。

GPU 随机数由 `corsika/gpu/em/Philox.hpp` 按

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

直接寻址，所以调换 batch 或 CUDA block 顺序不会改变某个 history 的随机数，但它不会复制 CPU 全局随机流的取数次序。

## 4. 逐模块源码审计

| 层次 | CPU 基准/契约 | CUDA 实现 | 本次判断 |
|---|---|---|---|
| 应用装配 | `applications/c8_air_shower.cpp` | 同一文件中的 backend 分支、14 项 registry | 配置、cut、环境、输出对象共用；CUDA 分支不是另一套应用 |
| 标量输运 | `ScalarCascadeStepper.inl` | `CudaPhotonTransport.cu`、`CudaLeptonTransport.cu` | 竞争项相同；数值实现与 RNG 不逐位相同 |
| 混合调度 | `Cascade/Stack` 的 LIFO | `HybridCascade.inl`、`PhysicalCudaEmRouter.hpp` | 所有权清晰，无并发修改原 Stack；处理顺序改变 |
| 过程门禁 | ProcessSequence traits | `ProcessSequenceCompatibility.hpp` | 未注册过程启动失败，不会静默跳过 |
| 粒子 ABI | CORSIKA 带单位粒子 | `Types.hpp`、`RouterParticleConversion.hpp` | 固定 GeV/m/s/g cm^-2/T；history/step 显式保存 |
| 随机数 | `cascade`、`proposal`、`thinning` 顺序流 | `Philox.hpp` | 分布可一致，随机轨迹不一致 |
| 过程能力 | PROPOSAL InteractionType | `ProcessCapabilities.hpp` | GPU-native 与 CPU-only 末态均有显式映射 |
| CPU 回退 | 原 PROPOSAL final state | `ProposalFallbackAdapter.hpp`、`ProposalCpuFallbackHandler.hpp` | process/component/v 不重抽；末态模型仍是 PROPOSAL |
| 介质与表 | PROPOSAL interpolate=true calculator | `MediumConfig.*`、`ProposalMedium.*`、`RateTable.*` | 介质 hash、版本、cut、能区和误差有硬门禁 |
| 设备表查询 | CPU PROPOSAL 插值 | `FlatRateTable.*`、`CudaRateTable.cu` | 自适应表的二次插值；普通列满足约 5e-4 |
| wavefront | CPU 单粒子 | `CudaWavefrontBucketing.cu` | CUB scan/稳定 compaction，无不确定原子追加 |
| photon 输运 | 无连续损失，几何/相互作用/cut | `CudaPhotonTransport.cu`、`CudaPhotonSelectionTransport.cu` | 球形层、观测面、逃逸、10 ms cut 均覆盖 |
| lepton 输运 | PROPOSAL displacement + Molière + tracking | `CudaLeptonTransport.cu`、`CudaLeptonSelectionTransport.cu` | 连续损失、散射、磁偏转、衰变竞争均覆盖 |
| 顶点重选 | CPU 先连续损失再 interaction | `CudaLeptonVertexSelector.cu` | 保留 pre-step 总 rate 阈值，在 post-step 能量重算累计率，逻辑与 scalar 对齐 |
| photon 末态 | PROPOSAL photon final states | `CudaPhotonPairFinalState.cu` | pair、Compton、photoelectric 和 LPM 均有 oracle |
| lepton 末态 | PROPOSAL e±/mu final states | `CudaBremsFinalState.cu` | brems、ionization、annihilation、electron-pair；mu 复杂末态回 CPU |
| LPM | `InteractionModel::CheckForLPM()` | `BremsLpm.hpp`、`PhotonPairLpm.hpp`、`EpairLpm.hpp` | 概率公式与 PROPOSAL oracle 一致；随机 draw 不同 |
| 多重散射 | PROPOSAL `MoliereInterpol` | `MoliereScattering.hpp`、`MoliereInterpolation.cpp` | 对 CPU 当前插值器误差小于 1e-5，但不逐位相同 |
| 球形大气 | USStdBK 5 层、介质 grammage | `SphericalAtmosphere.hpp`、`EnvironmentSnapshotBuilder.hpp` | 层边界、切线、grammage 及逆函数均有测试 |
| 磁场 | CORSIKA leapfrog、IGRF14 生成的均匀场 | `UniformMagneticField.hpp` | 同一 field snapshot；device 浮点不逐位相同 |
| 观测面 | scalar `ObservationPlane` | `ObservationPlane.hpp`、transport endpoint | 当前均使用真实平面交点，不再用球面近似 |
| cut/thinning | `ParticleCut`、`EMThinning.inl` | `EmThinning.hpp`、末态内核、transport endpoint | 给相同输入 uniform 时分支规则一致；实际 uniform 不同 |
| profile/沉积 | CPU writer 累加 | `CudaProfileProjection.cu` | 相同物理记录的 bin 语义一致；定点/顺序不同 |
| 射电 | CPU CoREAS/ZHS step observer | `CudaRadioAccumulator.cu` | 相同轨迹波形相对差 <=1e-9；独立 shower 不会逐脉冲相同 |
| 输出接回 | scalar writer | `CorsikaOutputSink.hpp` | observation、first interaction、profile、radio 均显式回放 |
| replay | CPU transport tape | `CudaDecisionReplayVerifier.cu`、`cuda_decision_replay` | 可验证并重放完全相同的 CPU 轨迹，但不是 CUDA 自主抽样 |

## 5. 逐物理过程验证结果

### 5.1 相互作用率、过程选择和距离

- `testGpuInteractionSelection` 在 100 TeV 表和 `1.05e8 GeV` 高能表上各检查 8,192 个粒子；高能表共执行 245,785 项比较，8,192 个输入全部完成选择，表查询 fallback 为 0。
- host reference 与 device 对 total rate、相互作用 grammage、process/component 和 loss fraction 的设备执行误差阈值为约 `3e-13`；这里验证的是“同一张表在 host/device 的执行一致性”。
- 表本身相对 CPU PROPOSAL interpolate=true 的 sampled rate 最大差：N Brems `3.36e-4`、O Brems `3.34e-4`、Ar Brems `3.31e-4`、ionization `2.94e-4`。

结论：同表 host/device 查询通过；但 CPU 原生 PROPOSAL 与 GPU 表仍有表格拟合误差，且随机数映射不同。

### 5.2 photon pair production

- `testGpuPhotonPairFinalState` 的 Koch--Motz 解析采样器最大 shape error 为 `1.30e-14`，variance error 为 `1.70e-3`，平均 rejection trials 为 1.872。
- 100 TeV 生产表：10,151 个末态输入中 8,473 次 GPU pair；
- `1.05e8 GeV` 高能表：10,176 个末态输入中 8,891 次 GPU pair；
- photon-pair LPM 对 PROPOSAL 的 450 个 energy/x/density/component 点最大相对差 `8.38e-16`。

结论：过程公式、末态守恒和 LPM oracle 通过；pair 能量 split 仍由 GPU 表/算法和 Philox draw 决定，不复制 CPU 的逐事件 split。

### 5.3 Compton scattering

- 高能生产表自然抽到 1,258 次 GPU Compton；100 TeV 表抽到 1,662 次；
- GPU 与 PROPOSAL `NaivCompton` 的能量、方向和二体末态逐样本比较通过；
- 宽能区测试测得最大方向分量绝对差 `2.36e-13`（100 TeV 表的最坏值为 `5.20e-12`）。

结论：在 double 数值精度内匹配 PROPOSAL；不是位级相同。

### 5.4 photoelectric effect

- 生产表中的自然发生率很低，两张外部表测试各出现 1 次；
- 合成 fixture 额外强制覆盖多次 N/O/Ar 组分和低能边界；
- 电子总能量、结合能修正和方向与 PROPOSAL `PhotoeffectNoDeflection` oracle 比较通过。

结论：代码路径完整；生产 shower 中统计稀少，系综验收仍需专门低能 photon 样本。

### 5.5 electron/positron bremsstrahlung

- `testGpuBremsFinalState` 完成 81,391 项检查：3,010 次接受、1,086 次 LPM 抑制、3 次显式 fallback；
- 末态与 PROPOSAL `BremsEGS4Approximation` 比较；
- brems LPM 的 540 个 energy/v/density/component 点最大相对差 `1.41e-15`。

结论：GPU 末态和 LPM 公式通过；损失分数仍受表格近似和氩非单调策略影响。

### 5.6 discrete/continuous ionization

- `testGpuIonizationFinalState` 对 e-/e+/mu-/mu+ 共完成 118,796 项 PROPOSAL oracle 检查；
- 高能表的 muon 输运：8,192 个 muon 全部传播，3,899 个到达离散顶点，其中 3,831 次 GPU ionization，68 次已指定过程的 CPU fallback，803 次 forced CPU decay；
- sampled ionization inverse-CDF 表相对 CPU 插值最大差 `3.47e-4`。

连续能损不是逐个小碰撞，而是通过 range/displacement 表求给定 grammage 的末能量。因此它与 CPU 的物理函数相同，但存在 table interpolation 和 host/device math 差异。

### 5.7 positron annihilation

`testGpuAnnihilationFinalState` 对 4,096 个样本与 PROPOSAL Heitler 末态进行能量、方向、子粒子数和确定性比较并通过。实际 draw 仍来自 Philox，而不是 CPU `proposal` stream。

### 5.8 electron/positron pair production

- `testGpuEpairFinalState` 对 1,024 个 PROPOSAL interaction 做逐样本比较，并对 81,117 个抽样做分布检验；
- 最大 normalized-rho error `5.26914e-4`；KS `0.008371`；
- Epair LPM 的 750 个 energy/v/rho/density 点最大相对差 `5.72e-11`。

这项是当前 GPU-native 末态中最明显的“高精度近似而非完全复制”：`5.269e-4` 略大于制表目标 `5e-4`，但低于现有单过程验收阈值 `1e-3`。若以后要求所有内部变量都严格小于 `5e-4`，需要继续细化 Epair rho sampler，而不能只提高主 rate table 的精度。

### 5.9 Molière multiple scattering

- 4,557 项检查，3,342 个实际偏转样本；
- GPU 相对 CPU 当前 `MoliereInterpol` 最大相对差 `9.83364e-6`；
- 相对 PROPOSAL direct calculator 最大差 `3.48269e-2`。

这里正确的 CPU 基准应是 `MoliereInterpol`，因为原 `ContinuousProcess.inl` 明确选择的也是这个模型。direct 差异不能解释为 CUDA 相对当前 CPU 路径的 3.5% 错误，但说明两者都不是每次重新直接积分。

### 5.10 球形大气、磁场、几何与观测面

- `testGpuSphericalAtmosphere`、`testGpuUniformMagneticField`、`testGpuPhotonWavefront` 和 `testGpuLeptonTransport` 全部通过；
- 覆盖五层边界、切向/近水平轨迹、grammage 与逆 grammage、观测面、逃逸、零/非零磁场、10 ms 物理飞行时间 cut、连续限制和衰变竞争；
- beta4 当前的 observation 使用真实平面交点；CPU step grammage 也按 scalar 的 straight chord 语义对齐。

结论：没有再发现球面/平面或弧长/chord 的结构性分支差异；最后若干 ULP 仍可能不同，并会被级联放大。

### 5.11 ParticleCut 与 EMThinning

- `testGpuEmThinning` 完成 8,192 个 host/device 决策精确比较和 200,000 个统计样本；
- 给相同父子能量、权重和 uniform 时，Hillas/weight-limited 分支、保留概率和权重更新与 CPU 规则一致；
- GPU 中 10 ms 是粒子物理飞行时间，不是 CPU wall time；能量 cut、时间 cut 和 rest-mass ledger 均有 endpoint 测试。

结论：算法规则一致；当前 CPU/GPU 不读取相同的 thinning uniform，所以同 seed 会留下不同的次级粒子。

### 5.12 稀有过程、muon、衰变和强子末态

以下过程由 GPU 完成 rate/距离/过程身份选择后，把指定末态返回现有 CPU 模块：muon brems、muon epair、photonuclear、photoproduction、photon-induced muon pair、产生 hadron/muon/tau 的末态、weak interaction 和 decay。

`ProposalCpuFallbackHandler` 会验证 process、component、v 和介质，再调用 `doSpecifiedInteraction()`；它不会重新抽取反应类型。随后仍执行原 ProcessSequence 的 `doSecondaries()`，因此 writer、thinning 和 cut 不会被绕过。强子高低能模型和 FLUKA 本身没有被 CUDA 替换。

分布上这仍是原 CPU 模型；逐事件上会不同，因为 fallback 被延迟批量执行，末态随机数和 LPM/强子内部顺序流的消费次序与标量 DFS 不同。

### 5.13 profile、能量沉积、CoREAS 与 ZHS

- `CudaProfileProjection.cu` 用设备端稳定累计产生 longitudinal/profile 和 energy-deposit 记录；
- `testGpuRadioProjection` 将同一条 e± 轨迹同时送入 CPU 和 GPU CoREAS/ZHS，三分量波形最大相对差要求并通过 `1e-9`；
- GPU 固定点射电累加在重复运行中逐位一致，测试无 overflow；
- 真实独立 CPU/CUDA shower 的轨迹不同，因此脉冲不应逐点相同，只能比较振幅、宽度、到达时间和空间 profile 的统计分布。

## 6. 本次发现的真实差异与风险分级

### 6.1 最高优先级：氩 Brems inverse-CDF 的单调投影改变了 CPU reference

当前 `proposal-monotone` 策略会把 PROPOSAL interpolate=true 返回的局部反向波动投影为单调曲线。`gpu_em_tablegen.cpp` 对标记为 `proposal_interpolated_monotone` 的列会跳过独立 raw-PROPOSAL 点对点验证，因为 raw 曲线正是被主动拒绝的 reference。

在 E = 623.7318908 MeV、Ar-40 Brems、u = 0.985988 处：

```text
GPU table        v = 0.8613340522897827
CPU interpolated v = 0.8844723084419007
PROPOSAL direct  v = 0.8592847213524135
table vs CPU-interpolated relative difference = 2.6161%
```

同一 zoom 区域检测到 CPU 插值曲线 130 个局部反向段，而单调表为 0。普通 N/O Brems 和 ionization sampled loss 仍约为 `3.5e-4` 至 `4.2e-4`。

这不是 CUDA kernel bug，而是**制表阶段主动改变 reference**。它可能比原 PROPOSAL 插值更接近 direct 计算，但它明确阻止与原 scalar CPU 的逐过程严格一致。应提供两个显式模式：

- `scalar-proposal-compat`：保留原 raw interpolate=true 行为或只对异常列执行 CPU selected-loss；
- `corrected-monotone`：保留当前物理修正，并明确声明不追求原 CPU 逐事件等价。

### 6.2 高优先级：随机数与调度契约不同

CPU 是全局状态随机流加 DFS；GPU 是无状态键随机数加 wavefront。GPU 还用一个 uniform 选择列、另一个 uniform 选择该列 loss；CPU `SampleLoss()` 用同一个 uniform 在总累计分布中同时完成两者。两种方式给出相同联合分布，但不会把同一个 seed 映射为同一结果。

此外，CPU fallback 延迟到设备 front 排空后执行。这保持了正确的末态模型，却改变了全局 CPU RNG 的消费顺序。

### 6.3 中优先级：高精度数值近似仍不是同一函数调用

- rate/range/loss 表：约 `5e-4`；
- Epair rho：最大 `5.269e-4`；
- GPU/CPU `MoliereInterpol`：最大 `9.83e-6`；
- 相同宽能区 photon 末态的 host/device 能量最大相对差约 `1.6e-13`；
- Compton 方向最大分量差约 `5.2e-12`；
- CPU double writer 与 GPU fixed-point/parallel reduction 的累计次序不同。

这些差异通常远小于 shower 的统计涨落，但任何早期微小差异都可能通过分支级联放大成不同的单事件。

### 6.4 已排除的问题

- 未注册自定义过程被静默跳过：已由 compile-time/runtime registry 排除；
- 表损坏、介质/cut/版本/hash 不匹配后自动转 CPU：当前会 hard fail；
- GPU 已选 process/component/v 后 CPU 再抽一次反应：当前 fallback handler 不会重抽；
- 观测平面仍用球面近似：beta4 当前已使用同一真实平面；
- 10 ms 被误解为计算 wall time：当前是物理飞行时间；
- 强制初级 interaction/decay 被 CUDA 绕过：当前先执行 scalar forced action；
- CoREAS/ZHS 在相同轨迹上算法不一致：同轨迹 oracle 已通过 `1e-9`。

## 7. 测试汇总

### 7.1 注册测试

执行 `ctest -R '^testGpu' --output-on-failure`：27/27 通过，总时间 31.61 s。覆盖 host/table/config/link、Hybrid route、CPU fallback、CUDA table query、selection、bucketing、所有末态、三种 LPM、thinning、Molière、磁场、球形大气、photon/lepton resident wavefront 和 radio。

### 7.2 高能生产表

```text
Interaction selection:
  245785 checks, 8192 selected, 0 fallback

Muon transport:
  102664 checks, 8192 transported, 3899 vertices
  3831 GPU ionization, 68 specified-process CPU fallback
  803 forced CPU decay

Photon final states:
  1868490 checks, 10176 interactions
  8891 pair, 1258 Compton, 1 photoelectric
  25 fallback, 1 continuation
  max host/device secondary-energy relative error = 1.047e-13
  max Compton/PROPOSAL direction absolute error = 2.363e-13
```

### 7.3 CPU decision tape replay

本次额外生成一个相同 seed `20260810` 的 10 GeV electron CPU shower，并记录完整 transport tape：

- CPU tape：14,590 条 EM transport record；
- 其中 lepton 13,434、photon 1,156；
- CUDA replay：invalid/nonfinite/negative/backward/invalid-direction 均为 0；
- host ordered byte hash 与 device ordered byte hash 均为 `58e61ad53fea7728`；
- byte hash mismatch 为 0；
- GPU 射电投影 13,434 条 lepton track，fixed-point overflow 为 0；
- replay accepted。

这证明 CUDA 能读取并处理**CPU 已经决定的完全相同轨迹**。它不证明 CUDA 自主采样会生成同一 shower。

对相同参数和 seed 独立启动 CUDA 后，初级 electron 的第一次 Brems 顶点已经不同：CPU trace 在约 `z=6.41284e6 m`、`E=9.99510 GeV` 选择 Brems；CUDA history 1 在约 `z=6.47100e6 m`、`E=9.999997 GeV` 到达其 Brems 限制。后续粒子树自然完全分叉。

## 8. 能否让 GPU 和原 CPU 得到完全一致的 shower？

### 8.1 若“完全一致”指原始 CPU、同 seed、同粒子树和同射电波形

**可以通过 replay 得到，但不能由当前 CUDA 独立采样直接得到。** 最稳妥方案就是现有两遍模式：

1. 原 CPU 按原 DFS 和原 RNG 运行一次，记录每个 transport segment、离散决定、末态和 observer snapshot；
2. GPU 不再抽随机数，只消费这张 decision tape；
3. GPU 批量重放 profile/radio 或其他确定性计算；
4. 用 ordered hash、history、顶点、能量和 waveform 做逐项比较。

这个模式能保证粒子树来自原 CPU，因此可用于回归和射电实现验证；但 CPU 已经付出了生成完整 shower 的主要成本，所以它不是一个高加速生产后端。

根本困难在于：原 DFS 的某个兄弟粒子从全局 RNG 的哪个 offset 开始，取决于前一个兄弟的整个未知子树消耗了多少随机数。若提前并行处理 wavefront，就无法在不先完成前一棵子树的情况下知道这个 offset。强行保持原随机流会重新串行化随机控制面。

### 8.2 若允许修改 CPU reference，要求“新 CPU”和 GPU 完全一致

这条路线可行性更高：

1. CPU 和 GPU 都改用同一 history-keyed Philox；
2. CPU 也查询同一 `.c8emrt` 表并使用相同 process/loss uniform 映射；
3. pair、Compton、photoelectric、brems、ionization、annihilation、Epair、LPM、Molière、thinning 全部调用共享的 host/device 数值函数；
4. history ID 由 `(parent_history_id, child_ordinal)` 规范生成，不依赖队列分配顺序；
5. 禁止 fast-math，固定 FMA/舍入策略；必要时关键超越函数使用同一确定性实现；
6. CPU 与 GPU 都使用相同 fixed-point profile/radio 累加和稳定输出排序；
7. fallback 的 LPM 和所有末态随机数也必须显式传入，不允许继续读全局 `proposal` RNG。

这样可以建立一个新的 CPU/CUDA 可重复物理契约，并有机会达到相同 decision tree 甚至相同输出。但它不会复刻“公开原版 CPU 对旧 seed 的 shower”，因为 CPU reference 本身已经换了随机数和表格算法。

### 8.3 若既要原 CPU 同 seed 完全一致，又要显著加速

可以实现一个 `strict-original-control-plane`：CPU 只负责按原顺序产生随机决定，GPU 批量执行确定性数值。但为了知道次级粒子，CPU 通常仍要等每个末态完成；而原 DFS 又要求前一子树完成后才能确定后一子树的 RNG offset。最终同步会非常频繁，GPU wavefront 很难变大，预期加速主要来自射电/profile，而不是完整 EM cascade。

因此建议把两个目标明确分开：

- **验证模式**：original CPU decision tape -> CUDA replay，追求逐记录/逐波形一致；
- **生产模式**：history-keyed CUDA，自主并行采样，追求过程级 oracle 和大样本统计一致。

## 9. 建议的下一步

1. 增加显式 `--gpu-physics-reference scalar-proposal-compat|corrected-monotone`，不要让两种氩修复策略共用模糊名称。
2. 为 `scalar-proposal-compat` 生成 raw interpolate=true inverse-CDF，或只对已知非单调 Ar 列执行延迟 selected-loss CPU 批处理；比较二者的性能和 500-event observable。
3. 将 Epair normalized-rho 的验收从 `1e-3` 收紧至 `5e-4`，继续细化 sampler，避免其成为表精度提升后的主导误差。
4. 扩展 replay tape，使其显式记录 distance、process/component、loss、LPM、thinning 和 final-state 每一个 draw，并增加“第一次分叉”自动报告。
5. 建立两套 CI：
   - 逐过程 CPU/PROPOSAL oracle；
   - 固定 CPU tape 的 CUDA replay hash + CoREAS/ZHS waveform comparison。
6. 生产验收继续使用独立 shower 系综，不把同 seed 逐事件相等作为 CUDA wavefront 后端的物理正确性标准。

## 10. 最终判定

| 验收目标 | 当前 beta4 状态 |
|---|---|
| 所有应用过程都有明确 CUDA/CPU 处理策略 | 通过 |
| GPU-native 单过程相对 CPU/PROPOSAL oracle | 通过，见氩单调策略和 Epair caveat |
| 介质、cut、表版本/hash 错误时 hard fail | 通过 |
| 相同轨迹的 GPU/CPU CoREAS/ZHS | 通过，最大相对差阈值 `1e-9` |
| 同 GPU、同配置重复运行 | 通过，history-keyed RNG 与固定点累计 |
| CPU/CUDA 独立 shower 的统计一致性 | 需结合既有 500/1000 系综继续验收 |
| 原 CPU 与 CUDA 同 seed 得到完全相同 shower | **不通过，当前设计并不承诺** |
| CPU tape 由 CUDA 完全重放 | 通过，14,590 条记录 hash mismatch 为 0 |

因此，当前 beta4 可以继续作为“统计等价、过程可审计”的 CUDA 研究后端，但论文和 README 不应表述为“同 seed 复刻原 CORSIKA 8 shower”。若要得到这个更强结论，应采用 decision-tape replay；若要让 CPU/GPU 自主运行也相同，则需要建立新的共享 RNG、表格和数值契约。
