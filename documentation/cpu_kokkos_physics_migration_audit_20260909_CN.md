# 原标量 CPU → Kokkos OpenMP / CUDA：物理迁移复核

日期：2026-09-09。对象为当前 beta5 工作树；不是对所有历史二进制的追溯认证。

## 结论与验收口径

按用户最新要求，目标是**物理过程、概率分布、输运与输出语义一致**，不要求同一 shower tree，也不以逐位相等代替物理验收。

本轮代码核对及真实 CPU 参考补测，**在已覆盖范围内没有发现新的过程遗漏、粒子身份错误或显著的能量转换错误**。输运/cut、沉积、thinning、逐轨迹射电测试在 OpenMP、CUDA 都通过。新增 Epair 分布检验也通过。

但不能据此宣称“全能区、所有介质、所有过程已保证无误”：Molière 尾部、photon-pair 分布、LPM 与罕见回退的完整独立矩阵没有在本轮全部运行。严格末态数值诊断仍有方向分量超限，下面保留失败记录并解释其量级；**没有调宽原门限把失败改成通过**。

本轮只新增诊断代码、可选测试目标及此报告；没有改生产物理公式、随机流、表、空气应用或安装程序，也没有重启/停止现有生产服务。

## 1. 三种比较必须分开

| 比较 | 用途 | 不能推出什么 |
|---|---|---|
| 同一 portable 函数的 host/device 比较 | 找执行空间、数学库、FMA、索引与数据复制差异 | 不能证明公式正确，双方可能共享错误 |
| 实际 CPU CORSIKA/PROPOSAL 对 portable 实现 | 检查模型、单位、概率、末态、过程调用与输出契约 | 有限输入测试不能证明全域正确 |
| 独立 shower 系综 | 检查局部误差累积后的 profile、地面粒子、射电系统偏差 | 不显著的 p 值不等于已证明 1% 等效 |

条件相同的显式末态可以逐项核对；拒绝采样与逆 CDF 使用不同随机数映射时，应比较其目标分布，不强求相同随机数产生同一子粒子。

## 2. 参考版本与实际执行位置

- 比较了本地原 CPU 源码、beta2 CPU 源码和 beta5 中的 scalar 路径；关键文件 SHA-256 在数据目录的 `openmp/manifest.json`、`cuda/manifest.json`。
- 单位库、`TrackingLeapFrogCurved`、`BaseExponential`、`SlidingPlanarExponential` 在上述三个本地版本中哈希一致。
- beta5 的 `ContinuousProcess` 相比 beta2 增加只读 calculator 导出；该文件的连续输运主体没有变化。`ParticleCut` 增加分类能量统计，不改变原 cut 判定。
- beta5 scalar CoREAS/ZHS 算法文件与 beta2 相同；**`TimeDomainObserver` 不同**，含此前 ZHS 时间窗边缘修复。因此“当前真实 CPU observer 通过”不能写成“与未经修复的旧 CPU 时间窗边缘逐点相同”。
- 实际末态 oracle 直接链接 patched PROPOSAL 7.6.2，调用 `make_secondaries` / `CalculateSecondaries` / `CalculateRho`；没有读取序列化缓存来猜公式。
- CUDA 与 OpenMP 分别构建于 `build/mountain-cuda`、`build/mountain-openmp`；它们是独立诊断构建，不替换生产安装。
- 共享执行实现为 [KokkosBackendInstance.inl](../src/accelerator/em/kokkos/KokkosBackendInstance.inl)，入口分别是同目录 `KokkosCudaBackendInstance.cpp`、`KokkosOpenMPBackendInstance.cpp`；物理公式主要在 `corsika/accelerator/em/common/` 和 `detail/`。

## 3. 按粒子经历核对的代码与覆盖情况

下表中的“代码核对”不是“完整数值矩阵通过”；避免将两者混为一谈。

| 环节 | scalar CPU 参考 | Kokkos 共享实现 | 本轮结论/覆盖边界 |
|---|---|---|---|
| 主栈、初级强制反应/衰变 | `ScalarCascadeStepper`、`Cascade` | `HybridCascade.inl`、`PhysicalAcceleratedEmRouter.hpp`、`AcceleratedHybridCascadeRunner.hpp` | 强制标志在初级路由前保留 CPU stepper 处理；不是先把初级送走再设置标志。此项代码复核，未重跑全部强制过程 |
| 可执行过程列表 | `ProcessSequence` 与 PROPOSAL builders | `ProcessCapabilities.hpp`、`ProcessSequenceCompatibility.hpp` | 注册表检查保留；不能用“GPU 不支持”静默删去过程 |
| 单位、总能/动能、质量 | `PhysicalUnits.hpp`、`InteractionModel.inl`、实际 stack | `TransportMass.hpp`、`RouterParticleConversion.hpp` | 实际 stack/secondary adapter 30 组通过；PROPOSAL 质量与 C8 输运质量不能混为同一个常数 |
| 过程/组分选择与反应率 | PROPOSAL `SampleLoss`、`InteractionModel` | `ProposalNativeQueries.hpp`、`NativePhysicsQueries.hpp`、`LeptonVertexSelection.hpp` | 复核 native 顺序、外层接受/拒绝和分量选择；本轮 portable 4096 查询只属自比较，不能冒充 live 百万点 oracle |
| 连续能损、range、步长竞争 | `ContinuousProcess.inl`、原介质 grammage | `LeptonTransportStep.hpp`、native utility spline | 72 个 scalar-step 用例通过；最大 grammage 相对差约 `6.43e-11`；不是全介质全能区误差上限 |
| 磁偏转、密度、边界 | `TrackingLeapFrogCurved.inl`、`BaseExponential.inl` | `UniformMagneticField.hpp`、`SphericalAtmosphere.hpp`、`ObservationPlane.hpp` | 算法结构对照完成；存在约 `1.89e-7` 磁偏转换算系数差，见下一节 |
| 山体材料边界 | scalar 几何及 tracking、terrain 场景 | `ExternalTransportBoundary.hpp`、`ConvexEnvironmentSnapshot.hpp`、`src/terrain/TerrainEmSession.cpp` | 由独立 terrain 调度器处理材料切换；此前跨界/曲面测试见关联报告。不能把空气 wavefront 当成材料切换调度器 |
| Compton | `NaivCompton` | `PhotonFinalStateStep.hpp` | 新增真实末态 oracle，1024 组/后端；身份与能量通过，有小角方向数值差 |
| 光电效应 | PROPOSAL 光电末态、scalar writer | `PhotonFinalStateStep.hpp`、`ProfileAccumulationStep.hpp` | 沉积测试含实际光电末态和独立 binding ledger；不把静质量或 binding 重复记为沉积 |
| photon pair | PROPOSAL Koch–Motz/Sauter | `PhotonPairFinalState.hpp`、`PhotonFinalStateStep.hpp` | 原生表无专用 split 列时走解析拒绝采样，这是正常 native 路径；本轮未完成其独立分布矩阵 |
| brems | `BremsEGS4Approximation` | `LeptonFinalStateStep.hpp` | e−/e+ 各 1024 组/后端真实末态通过；条件固定，本测试不包含 LPM 接受概率 |
| 离散 ionization | `NaivIonization` | `LeptonFinalStateStep.hpp` | e−/e+ 各 1024、μ−/μ+ 各 640 组；身份与能量通过，极前向角有数值差 |
| e+ annihilation | `HeitlerAnnihilation` | `sampleAnnihilationRho`、末态 materialization | 1024 组/后端，真实 rho 反演后能量一致；方向严格诊断未全通过，见第 5 节 |
| Epair | PROPOSAL KKP `CalculateRho` | `EpairFinalState.hpp::sampleEpairRhoRejection` | 新增 24 格真实 CPU CDF 参考，OpenMP/CUDA 各 1,572,864 次采样通过；不是每格一百万 |
| Molière | PROPOSAL `MoliereInterpol`、CPU `Step::add_dU` | `MoliereScattering.hpp`、`MoliereStep.hpp` | 同一分布的不同数值求解；此前实际 CPU 首步约 `7.2e-11` 方向差。新 transport RNG 检查不等于散射分布验证 |
| LPM | `InteractionModel::CheckForLPM`、PROPOSAL 参数 | `BremsLpm.hpp`、`EpairLpm.hpp`、`PhotonPairLpm.hpp` | 代码核对参数/抑制与存活粒子回路；本轮未重跑完整能量×密度×组分概率矩阵 |
| μ 衰变、光核/强子罕见末态 | Pythia8/PROPOSAL/强子模型 | `ProposalCpuFallbackHandler.hpp`、router、HybridCascade | 保留 CPU 指定过程处理和非 EM 返回主栈；μ寿命与当前 C8 一致。未对全部罕见通道做新分布验证 |
| thinning | 实际 `EMThinning`、实际 secondary stack | `EmThinning.hpp`、photon/lepton classification | 200,000 组/后端，决策差 0、权重超 16 ULP 差 0（最大 4 ULP） |
| cut、运动时间终止、观测输出 | `ParticleCut`、scalar writers | photon/lepton transport、router、profile accumulator | 12 个 terminal、4 个 observation+cut 组合及 420 个沉积/输出用例通过；10 ms 是粒子运动时间，不是运算墙钟时间 |
| CoREAS/ZHS | 实际 scalar 模块及当前 observer | `RadioProjectionStep.hpp`、`KokkosRadioAccumulator.hpp` | 普通轨迹、±电荷、Cherenkov 附近、边缘/越界、分块、reuse、加宽时间窗均通过；含势及差分电场 |

## 4. 磁场差异：真实，但不是百分比量级的新物理错误

后续处理见[常数与山体磁传播对齐](terrain_scalar_constants_alignment_CN.md)。下述数值保留为修复前诊断，不能当作当前源码仍未修复的结论。

原 CPU 经本地单位库推导的系数为 `0.2997925146834389`，portable 常数为 `0.299792458`，相对差约 `1.89e-7`，即 **0.189 ppm**。原因是旧单位系统所用常数及换算组合，与 portable GeV/m/T 表达所用常数不同。

它影响有磁场的空气段，也会影响山体应用的外部空气段；岩内设置 B=0 时不产生此项差异。两种 Kokkos 后端共享同一系数，所以它不能解释 OpenMP 与 CUDA 彼此不同。

此前独立 scalar 几何对照中，1 GeV、50 μT、1000 m 的位置分量差约 1.13 μm；统一诊断系数后约 `1.14e-13 m`。这是局部算例，不是完整 shower 或 Cherenkov 尖峰的全局误差界。

建议后续将系数来源集中定义并记录常数版本，以提高可维护性；**不通过修改输入磁场来补偿，也不为了同树回放强行修改物理**。这不是目前优先级最高的物理风险。

## 5. 新增真实 PROPOSAL 末态数值诊断

代码：

- [testKokkosLiveFinalStateOracle.cpp](../tests/accelerator/testKokkosLiveFinalStateOracle.cpp)：实际 PROPOSAL calculator 参考，固定过程、v、必要 random draws。
- [LiveFinalStateDriver.cpp](../tests/accelerator/LiveFinalStateDriver.cpp)：在真实 Kokkos execution space 调用生产共享末态函数。
- 目标 `testKokkosLiveFinalStateOracle` 是 `EXCLUDE_FROM_ALL`，不安装，不加入默认 CTest，不影响生产。

每后端 **7424 个条件末态**：总能量网格 0.002–1e8 GeV（跳过低于该粒子质量的格点）、4 个方向、32 个分位点。它不是自然发生率加权的 shower，也不是任意 v 的完整矩阵。

结果：粒子种类/数量差 0，能量超限 0，方向非有限值 0。最大次级能量相对差 `7.10e-15`，最大方向模平方偏差约 `4.44e-16`。

| 过程 | 最大方向分量绝对差（两后端量级相同） | 最大值对应初能 |
|---|---:|---:|
| brems | `2.22e-16` | 低能格点 |
| Compton | `2.75e-10` | `1e8 GeV` |
| ionization | `1.04e-10` | `1e8 GeV` |
| annihilation | `1.48e-8` | `100 GeV` |

原严格诊断 `atol=1e-12, rtol=1e-10` 下，每后端有 2288 个方向分量比较超限，程序明确返回 1；**没有报告为严格通过**。方向分量计数不是失败事件数。

CPU 与 portable 的基本运动学表达对应，但 CPU 在 MeV 下计算，portable 部分表达在 GeV 下计算，且有不同的防抵消写法、三角函数/旋转运算展开。当 cosθ 极接近 1 时，sinθ 或 θ 对 cosθ 的微小舍入敏感，方向绝对差可大于能量误差。新测试显示这些差异在 OpenMP 与 CUDA 上都有，不是 CUDA 独有的内存错误。

这些数值本身不足以证明有可观测的分布偏差，也不能据此保证所有观测量不受影响。后续重点应是尾部角分布与横向/射电量的敏感性，不是让每个方向分量追平原 CPU 的舍入结果。

## 6. Epair：直接检验分布，不要求相同随机数映射

当前生产使用 `sampleEpairRhoRejection`，不是旧 rho 逆 CDF 辅助表。CPU 默认末态的 KKP rho 密度使用积分反演；portable 对同一目标密度做 `rho=rho_max*sin²(πu/2)` 变换后拒绝采样。

新增 `--epair-cdf` 模式，矩阵为：

- e−、μ−；氮 Z=7、氧 Z=8、氩 Z=18；
- 总能量 1 GeV、10 TeV；v=0.01、0.5；共 24 格；
- 每格 65,536 次真实 execution-space 采样；每个后端合计 1,572,864 次；
- 在 63 个实际 PROPOSAL `CalculateRho` 分位点检查 |rho| 的经验 CDF，另检查正负号概率、合法范围、回退与采样状态；
- 预先使用 DKW/多格 union bound 的抽样门限 `0.00836408`，family α=0.01。它只计抽样误差，不包含 CPU 积分误差；这是分位点检验，不是连续全域 KS 的精确 p 值。

**OpenMP 和 CUDA 的 24 格全部通过**。无回退、无范围错误；本轮最大分位点 CDF 差约 `0.0052643`，低于预设界限。

限制：这不是所有电荷符号、所有介质、全 v/阈值区域的一百万点矩阵，也不验证外层 LPM。当前 `ProcessCapabilities.hpp` 将 μ 的 brems/Epair 末态明确交给 CPU；本测试中 μ 格点是对通用 rho 数值函数的额外回归，**不表示生产 μ Epair 已在 GPU 生成末态，也没有验证该回退接线**。e− 的 12 格对应实际使用的采样公式。

当前拒绝包络是 8 点粗定位+12 次局部细化+5% 裕度，不能把有限点通过当成全域包络上界的数学证明；代码附近“64-point envelope”注释已落后于实现。若实际发现超过包络，代码显式回退；但**尚未采到的超包络区间不会因为存在回退分支就自动被证明安全**。应增加阈值/极端 v 的 PDF 与包络扫描。

## 7. 随机数、时间窗和能量账本：比同树更重要

### 随机数相关性

当前 `RandomDomains.hpp` 为版本 2。版本 1 曾把 native process selection 与 Molière 第一个 draw 放在同一域，引入确定性相关；这是会改变联合分布的真正问题，不能以“边缘直方图相似”忽略。

本轮 65,536 个 key 对照：旧域重复 65,536 次，当前域重复 0 次，样本相关系数约 0.00320。当前修复仍生效，但历史结果必须按 RNG 版本/二进制 SHA 区分，不能只按 beta 名称合并。

portable `uniformOpen01` 为 32-bit midpoint uniform，原 CPU 标准分布取数方式不同。32-bit 情况最小 u=2⁻³³，对应指数尾部约 22.87 个平均自由程；该项影响极小概率尾部，不应无依据称为百分比偏差。若未来研究极罕见尾部，应单列有限 RNG 分辨率预算。CPU 罕见回退和 thinning 仍有原 CPU 流；不承诺整个混合运行只由一个 history key 唯一决定。

### 能量账本

PROPOSAL 的总能量先减 PROPOSAL 质量，再由 C8 stack 加回 C8 质量；这不是同一个数值常数在两处随意替代。靶电子静质量、光电 binding、cut 动能和不可见粒子必须分账，不能以“只加总所有存活粒子总能量等于初能”作为所有过程的守恒判据。本轮用实际 scalar writer/stack 核对了这些已覆盖契约。

### 射电窗口与量化

真实 scalar CoREAS/ZHS 对照通过的门限是绝对底噪项加 `2e-8 × peak`，不是所有点 16 ULP：CoREAS 底噪 `2e-18`，ZHS potential `2e-27`，ZHS 差分 E `4e-18`，均按测试输出单位。

定点累积在极弱信号上可能显示较大的相对误差，例如本轮边缘 ZHS 用例的打印相对量约 `1.32e-4`，但仍满足绝对误差判据。报告不能只挑非定点 ordinary 轨迹的机器精度数字来描述整个后端。

本轮还验证了加宽窗口后公共区间一致、边缘不凭空产生脉冲、清空/复用不残留旧波形。山体跨介质传播的反射/透射与折射率模型是另一层测试；空气单轨迹通过不能替代它。

## 8. 后续优先级

1. **物理验收缺口优先**：补 Molière 与 photon-pair 的 live CPU CDF/角度尾部矩阵，Epair 扩展阈值/极端 v/岩石和其他介质，LPM 扩展能量×局部密度，随后检验罕见回退的条件分布。不要先为同树关闭 FMA 或改采样方法。
2. **边界契约加固**：空气 resident 分类器对新 `MaterialBoundary` 枚举没有显式分支。本轮追踪确认普通空气重载不会返回此值，terrain session 自己处理它，未证明当前有粒子丢失。但未来若混用应明确拒绝或路由，不能依赖隐含前提。
3. **数值维护**：统一磁常数来源；为极前向角保留高精度对照和敏感性数据；修订 Epair 包络旧注释。不把这些项自动等同于重大物理偏差。
4. **系综验收**：按 shower 重采样（不是按粒子 bootstrap）；先核对相同 cut、thinning、磁场/日期、天线、观测面、时间窗和 RNG 版本。关键均值的差值置信区间应落入预设等效区间，不能只写“p>0.05”。profile/时域曲线要考虑跨 bin 协方差与多重比较。
5. **射电展示**：沿用既有 Ex′/Ey′/Ez′ 投影与共同时间基准；不靠每事件重新对齐峰值掩盖到达时间偏差。平均场、幅度/极性与时宽分布是不同观测量，应分别说明。

HIP/SYCL 未在本轮运行；同源代码不等于其硬件精度和编译路径已验收。本轮亦未重跑完整强子生成模型或所有山体中微子初始顶点。

## 9. 数据、复现和运行保护

本轮数据根目录：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta5_cpu_kokkos_migration_audit_20260909_v1/
    final_openmp/           # 最终诊断二进制、源码清单、六项检查
    final_cuda/             # 同上，真实 CUDA 执行
    openmp/                 # 初次四项 scalar 参考 + 4096 portable 自比较
    cuda/                   # 初次四项 scalar 参考
    live_final_openmp_v2/   # 中间诊断记录，非独立新增物理样本
    live_final_cuda/
    epair_cdf_openmp/
    epair_cdf_cuda/
```

`live_final_openmp/` 是增加最大误差上下文输出之前的初次运行，也保留，不混作独立样本。
每次执行保留 `stdout.log`、`resources.json`；两套最终执行另有源码清单、参考文件和可执行文件哈希的 `manifest.json`。最终六项检查包括四项 scalar 模块、条件末态严格诊断、Epair CDF。runner 如实保留严格方向差导致的非零退出状态，不将整组标成全部通过。重复执行使用同一组确定性输入，不重复计算样本量。文件清单用于定位版本，不声称清单中每个函数都已独立动态验证。

在 `corsika_venv`、源目录下：

```bash
cmake --build ../build/mountain-openmp --target testKokkosLiveFinalStateOracle -j2
cmake --build ../build/mountain-cuda --target testKokkosLiveFinalStateOracle -j1

# output 必须为新目录；守护器只会停止自己的诊断子进程
OMP_PROC_BIND=false python validation/terrain/run_guarded_diagnostic.py \
  --output /path/to/new-audit-openmp \
  -- ../build/mountain-openmp/tests/accelerator/testKokkosLiveFinalStateOracle --epair-cdf

python validation/terrain/run_guarded_diagnostic.py --gpu \
  --output /path/to/new-audit-cuda \
  -- ../build/mountain-cuda/tests/accelerator/testKokkosLiveFinalStateOracle --epair-cdf
```

新 CMake 目标首次加入已有构建时，先对独立 build 重新配置。完整 scalar 模块组的可重复入口是
[run_cpu_migration_audit.py](../validation/accelerator/run_cpu_migration_audit.py)，加 `--live-oracles` 同时执行新增的两项 oracle。

诊断限制：每进程 RSS 2 GiB、系统可用内存保留 3 GiB；GPU 设备总显存增量最多 512 MiB 或总量 10%，取较小值；240 s 超时。显存增量是设备级保守监测，不是精确进程级归因，极短峰值仍可能漏采。生产正在运行，因此测试墙钟时间**不作为性能比较**。

本轮结束核对空气应用 SHA-256 仍为 `d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`；生产 service 保持 active。

关联既有诊断：[严格轨迹定位报告](terrain_strict_step_replay_CN.md)、[曲面边界报告](terrain_curved_boundary_acceptance_CN.md)。这些是已有测试，不计入本轮新样本数。
