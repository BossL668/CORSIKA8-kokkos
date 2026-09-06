# beta5 Kokkos 与原标量 CPU：代码路径复核

> 这是修复前的只读审计记录。后续发现、修复和实际回归结果见
> [2026-09-06 修复报告](BETA5_CPU_PHYSICS_ALIGNMENT_FIXES_20260906_CN.md)，
> 包括本记录尚未发现的 native selection / Molière 随机域冲突。

日期：2026-09-06。对象：当前 beta5 的 `c8_air_shower`、Kokkos-OpenMP / Kokkos-CUDA 共享物理层，以及本地 beta2 标量 CPU 源码。不是只根据 k5 的统计图判断。

## 结论

**不能下“已经不存在物理／可观测量定义差异”的结论。** 本轮确认了四项确定性差异：一项连续步长终点公式差异，三项能量沉积／终止记账差异。前三项在直接调用当前共享算法的 OpenMP 和 CUDA 小型诊断程序中均复现；第四项有 PROPOSAL 末态和两侧 writer 调用链的代码证据。

这些发现不等于已证明现有系综有显著偏差，更不能把历史所有电磁 profile 差别都归因于它们。特别要区分粒子数 `N_e(X)` 与能量沉积 `dE/dX`：后三项首先改变的是沉积记录，不是粒子的反应率或次级数。

本次只新增独立诊断和本文档，**没有修改物理实现、生产二进制、任务参数或已完成的数据，也没有推送远端。**

## 1. 连续输运接近 cut 时的目标总能量不一致

CPU 原实现见 beta2 `corsika/detail/modules/proposal/ContinuousProcess.inl:161`；beta5 同一文件约 199 行仍保留相同公式：

```cpp
max(0.9 * E_total, (kinetic_cut + mass) * 0.9999)
```

Kokkos 使用 [NativePhysicsQueries.hpp](../corsika/accelerator/em/common/tables/NativePhysicsQueries.hpp) 的 `queryContinuousMinimumEnergy()`（349–379 行）：

```cpp
minimum = mass + 0.9999 * kinetic_cut;
```

然后 [LeptonContinuousStep.hpp](../corsika/accelerator/em/detail/LeptonContinuousStep.hpp) 用它限制连续步长。代码注释把 CPU 公式也描述成后一种形式，但与实际 CPU 源码不符。

在使用相同质量常数的控制实验中，两个目标能量相差 `1e-4 * mass`：

| 粒子 | 动能 cut | CPU 目标总能量 / MeV | Kokkos 目标总能量 / MeV | 差值 |
|---|---:|---:|---:|---:|
| e− / e+ | 0.5 MeV | 1.010897850105 | 1.010948950000 | 51.099895 eV |
| μ− / μ+ | 300 MeV | 405.61780966245 | 405.62837550000 | 10.56583755 keV |

影响：接近 cut 的最后一步 grammage、位置、时间和散射输入可能不同；电子轨迹变化也可能传入射电。不是全能区截面错误，也不是每个粒子都会损失上述能量：cut 时余下动能还会被记账。

建议：单独保存“用户动能 cut”与“CPU 步长终点总能量”，后者按 CPU 公式计算。**不能只改这一行**：`LeptonContinuousStep.hpp:111` 当前会用 `(minimum - mass)/0.9999` 反推出用户 cut，需要同时解除这种耦合。测试应覆盖 e±、μ±、cut 上下、10% 限制与 cut 限制交接位置。

来源核对：beta4 `corsika/gpu/em/tables/FlatRateTable.hpp:1216` 已有同样表达式，因此不是 beta5 的 Kokkos 平台切换新产生的问题。

## 2. cut 剩余能量被沿整段轨迹分配，而 CPU 放在终点

CPU 分成两次写入：

- `ContinuousProcess::doContinuous()` 把连续损失写到起点—终点的线段。
- [ParticleCut.inl](../corsika/detail/modules/ParticleCut.inl) 的 `doContinuous()`（163 行起）把剩余动能写到 **终点**。
- [EnergyLossWriter.inl](../corsika/detail/modules/writers/EnergyLossWriter.inl) 的线段重载与点重载分别位于 105、194 行。

Kokkos 的 [ProfileAccumulationStep.hpp](../corsika/accelerator/em/detail/ProfileAccumulationStep.hpp)（337–354 行）则合并两种能量：

```cpp
accumulateEnergyProfile(..., X_start, X_end,
    (continuous_deposit + cut_deposit) * weight);
```

因此粒子最后一步跨越 histogram bin 时，即使轨迹和总沉积完全相同，`dE/dX` 的分箱也不同。`ProfileProjectionStep.hpp` 和 router 的非驻留输出分支也有相同合并，不能只修驻留 kernel。

诊断例：`X_start=9.99`、`X_end=10.01 g/cm²`，bin 宽 `10 g/cm²`，连续损失 `4e-5 GeV`、cut 剩余动能 `4.9995e-4 GeV`、权重 1。

| 记账方式 | 前一个 bin / GeV | 后一个 bin / GeV | 总和 / GeV |
|---|---:|---:|---:|
| CPU 线段连续损失 + 终点 cut 的定义 | 0.000020000 | 0.000519950 | 0.000539950 |
| 实际 Kokkos-OpenMP helper | 0.000269975 | 0.000269975 | 0.000539950 |
| 实际 Kokkos-CUDA helper | 0.000269975 | 0.000269975 | 0.000539950 |

没有定点溢出。这是分配位置差异，不是总能量丢失；单个刻意选取的跨 bin 例子不能当成全 shower 的相对偏差。

建议：连续损失仍写线段，cut 损失用 `(X_end, X_end)` 单独写入。同步检查驻留 profile、回传 projected record、CPU output sink，以及时间 cut 的光子分支。补充“总和守恒且逐 bin 对齐”的测试，而不只比较总和。

## 3. 观测面命中与能量 cut 同时发生时，Kokkos 少执行一项 CPU 记账

CPU 的 `ProcessSequence::doContinuous()` 不会因 `ObservationPlane` 返回 `ParticleAbsorbed` 就短路：会继续调用之后的 `ParticleCut`。所以 CPU 可以同时写地面粒子和该粒子的终点 cut 沉积。

Kokkos 的 [LeptonTransportStep.hpp](../corsika/accelerator/em/detail/LeptonTransportStep.hpp) 目前有：

```cpp
if (reaches_cut && !observation_reached) { ... }
```

时间 cut 分支已经有 `observation_surface_reached_before_cut` 双重标记，但能量 cut 分支没有同样处理。

用线性合成 range 隔离终止逻辑，电子初始动能 `0.50001 MeV`，在观测面结束后动能约 `0.49997 MeV`：

- OpenMP 与 CUDA 均正常命中观测面，无 fallback。
- 两者 `cut_deposited_energy_GeV=0`。
- 按 CPU 的实际终点 cut 条件，此时应再写约 `0.00049997 GeV` 的 cut 沉积。
- 不触发 cut 的对照，以及在抵达观测面前先达到 cut 的对照，也已检查。

建议：能量 cut 也保留“先观测，后 cut”的双重输出状态，并统一 profile、cut 计数、sink 行为。注意这是对齐 **CPU 现有 observable 定义**；总能量闭合账本不能把已观测能量和它的重复 cut 记录再次当成两个独立能量出口。若要改变 CPU 这种约定，应作为另一项显式修改，不能在 GPU 一侧偷偷改变。

beta4 的共享 `LeptonTransportStep.hpp` 已有这一判断，亦非 beta5 新引入。

## 4. 光电效应的结合能：Kokkos 增加了原 CPU 未写入的沉积

PROPOSAL 7.6.2 `PhotoeffectNoDeflection::CalculateSecondaries()` 返回一个电子：

```cpp
I = (Z * ALPHA)^2 * ME / 2;
E_electron_total = ME + E_photon - I;
```

Kokkos [PhotonFinalStateStep.hpp](../corsika/accelerator/em/detail/PhotonFinalStateStep.hpp) 的 `photoelectricEnergy()` 使用同样能量关系。因此这里发现的差别 **不是电子末态能量公式**。

差别在之后的记账：

- CPU `InteractionModel::doSpecifiedInteraction()` 把这个电子加入次级栈，没有把 `I` 交给能量沉积 writer。新增的 atomic-electron rest-mass 统计也不是结合能沉积。
- Kokkos `accumulatePhotonProfileFinalState()`（`ProfileAccumulationStep.hpp:293` 起）额外把 `E_photon * (1 - kinetic_fraction)=I` 写入沉积，而且目前沿入射光子的上一段轨迹分配。

用当前 helper 计算，N、O、Ar 的这项单次能量约为 **0.667、0.871、4.408 keV**。这些数值不是对 shower 总偏差的估计；影响还取决于过程发生次数、权重和介质。

物理上可以把结合能计入局部介质响应，但这与原 CPU 的输出定义不同，而且应在反应顶点而非沿前一段光子路径记账。

建议先明确验收口径：

1. 若要求严格复刻原 CPU 输出，把这项放入独立的结合能／闭合账本，不单独改变 Kokkos 的 legacy `dE/dX`。
2. 若要完善局部沉积物理，应同时更新 CPU 和 Kokkos，在顶点写入，并重新定义基准版本。

不能把这项差异描述成原生插值表的误差。

## 5. 本轮同时核对的主要路径

下表的“未发现新增不一致”是本次代码检查范围内的结论，不表示已完成所有过程、能区、介质的百万点验收。

| 路径 | 检查到的实现／结论 |
|---|---|
| 物理过程装配 | CPU 和 Kokkos 分支之前构造 PROPOSAL、强子模型、CoREAS/ZHS；保留同一个 ProcessSequence。未发现为 Kokkos 故意关闭 FLUKA/SIBYLL 的代码路径。具体部署仍需核对构建选项。 |
| native 表身份 | `KokkosEmRunSession` 从实际 calculator 导出；初始化有 requirements 检查，`KokkosEmBackend::initialize()` 重算内容哈希。不是从文件名猜介质。 |
| 率、过程和目标组分选择 | `InteractionSelection.hpp` / `LeptonVertexSelection.hpp` 使用 native 查询；带电粒子在连续损失后的顶点重新选择，保留起始率的接受／拒绝逻辑。 |
| native 逆解 | 保留 PROPOSAL 样条、负率截零和 Newton/bisection 语义。`SampleLoss` 的剩余 quantile 与独立末态随机数有区分，不能当作一个统一随机数流。 |
| γ 末态 | pair、Compton、photoelectric 显式分类；未支持过程进入规定的 CPU 处理，未发现直接吞掉稀有过程。光电沉积差异见问题 4。 |
| e± / μ± | 连续损失、散射和反应竞争为共享 helper；μ 的部分辐射末态仍由指定过程 CPU fallback 生成。近 cut 步长差异见问题 1。 |
| fallback | `ProposalCpuFallbackHandler` 在指定顶点导入粒子，校验组分／calculator，完成指定 loss 和末态，再执行原 doSecondaries 链，不重新随机选择整个过程。 |
| thinning | Kokkos 两次级 helper 的 Hillas / weight-limited 分支与 CPU 对照；μ parent 显式禁用 EM thinning，三次级 Epair 不套两次级 thinning。顺序和随机流不同不等于抽样概率不同。 |
| 大气、磁场、观测面 | 当前空气应用两侧使用相应五层模型、同一磁场配置与平面；grammage 采用与 CPU SlidingPlanarExponential 对应的起点局部径向近似，并用完成轨迹的 chord 计算连续损失。终止与 writer 仍有问题 2、3。 |
| 粒子数 profile | 投影后 ceil/floor 计数边界与 CPU 对照；不能用 `dE/dX` 的记账问题直接断言 `N_e(X)` 一定偏高／偏低。 |
| CoREAS/ZHS 粒子范围 | CPU `RadioProcess::doContinuous()` 本身只允许 e±，所以 Kokkos 同样过滤 μ 并不是漏掉 CPU 已计算的 μ 射电。 |
| 射电数值实现 | 共享 track 预计算、传播和投影公式，保留 Cherenkov 小分母、ZHS 细分和时间窗处理；定点累加与 CPU double 求和仍有数值差异。问题 1 导致轨迹不同也可能间接影响射电。 |

此外，两项算法差异不能与“同一随机数必然同一 shower”混淆：

- 当前 Epair 生产末态调用 `sampleEpairRhoRejection()`，目标是同一个条件密度，但不是 CPU 的数值逆 CDF 映射。即使给同一个第一个均匀随机数，得到的 rho 也不要求相同；需要检查密度、包络和失败计数的独立 oracle。
- Molière 初值／求根、有限精度样条、定点累加均不能承诺跨平台逐位相同。CPU 顺序 RNG 与 Kokkos history-keyed Philox 也不是同一种随机流。抽样分布一致和同 seed 事件完全相同是两种验收要求。

## 6. 测试覆盖上的缺口

[testKokkosProposalNativeTable.cpp](../tests/accelerator/testKokkosProposalNativeTable.cpp) 中部分命名为“CPU oracle”的检查，实际上把同一个 `accelerator::em::detail` 函数分别放到 host 与 Kokkos 执行空间运行。例如约 671 行的期望轨迹直接调用 `detail::transportLepton()`。

这种测试能查移植、执行空间和数据搬运错误，但 **无法发现共享 helper 与原 `ContinuousProcess` / `ParticleCut` 的共同偏差**。不能把它解释成完整的原 CPU 路径 oracle。本文不是否认以前独立 PROPOSAL 测试的价值，而是指出这个现有测试本身的覆盖边界。

k5 的 10 GeV 三组 2000 例符合良好，也不与本轮发现矛盾：边界情况稀少、分箱会掩盖位置差异，且默认 max-weight 在这个能量下不足以激活 thinning；更没有充分激发高能 LPM 等过程。

## 7. 本次实际运行内容和建议下一步

诊断位于源码的同级构建目录：

```text
build/audit/physics_path_20260906/
  boundary_probe.cpp / boundary_probe.txt
  kokkos_boundary_probe.cpp
  kokkos_boundary_openmp / kokkos_boundary_openmp.txt
  kokkos_boundary_cuda / kokkos_boundary_cuda.txt
```

普通 C++ probe 直接调用当前查询、输运、profile、光电能量 helper。Kokkos probe 分别使用现有 Conan OpenMP 与 CUDA 库编译，运行于本地 CPU / RTX 4060 Laptop GPU；只分配一个小结果 View，不初始化大型 shower workspace。

**测试限定**：range 采用线性合成数据以隔离边界逻辑；CPU 预期值由已检查的原 CPU 公式和 writer 语义独立给出，未在本轮执行完整 CPU shower、真实 PROPOSAL 百万点矩阵或全 shower 第一次分叉验收。测试返回成功表示“差异成功复现”，不是物理一致性通过。

推荐顺序：

1. 修复问题 1，并把用户 cut 与步长终点解耦。
2. 修复问题 2、3；分开连续／点状沉积，保留 CPU 观测与 cut 的顺序，重新核对闭合账本。
3. 明确问题 4 的 legacy observable 与物理能量账本口径。
4. 加入真正调用原 CPU 模块的相同输入测试，覆盖每个 cut 附近、上下行跨 bin、观测面、时间 cut、μ 及光电事件。
5. 再做小样本三后端回归和独立逐过程 oracle；最后复用现有参考样本，量化修复对 `N_e(X)`、`dE/dX`、地面谱与平均时域波形的影响。未测量前不承诺“变化小于 1%”。
