# beta5 输运、末态栈适配与随机域对齐说明

本轮以原标量 CPU 的实际 `ContinuousProcess`、`ParticleCut`、`ObservationPlane` 和 PROPOSAL → CORSIKA 栈适配为基准，不改变这些 CPU 模块。下面说明实现及独立测试范围；完整构建和运行结果以本轮总验收报告为准。

## 1. 最后一个连续步长

CPU `corsika/detail/modules/proposal/ContinuousProcess.inl` 的步长目标为：

```cpp
max(0.9 * E_total, 0.9999 * (kinetic_cut + get_mass(pid)))
```

原共享 helper 错误地使用了 `mass + 0.9999 * kinetic_cut`，并从该终点反推出 kinetic cut。现在分别查询用户动能 cut 和步长目标总能量，`InteractionSelection` 与 `LeptonContinuousStep` 不再通过步长终点推算 cut。

另一个独立细节是两套质量常数：CORSIKA 自动生成的质量表与 PROPOSAL 常数存在很小差异。输运速度、磁偏转、动能 cut 和 μ 衰变距离使用 CORSIKA 质量；截面、末态参数化及 Molière 仍使用 PROPOSAL 的质量。`common/TransportMass.hpp` 是设备可见的 POD 质量副本，在普通 host 编译中以 `static_assert` 对照 CORSIKA 自动生成数据；没有修改 PROPOSAL 常数或磁盘表。

`queryContinuousMass()` 和 `EmInteractionRecord::particle_mass_GeV` 继续代表 PROPOSAL 参数化质量；新 `queryContinuousTransportMass()` 代表标量栈质量，避免把输运修复意外传播为末态公式变化。

## 2. 观测与 cut 可以同时发生

CPU 的 `ProcessSequence::doContinuous()` 不会因 `ObservationPlane` 吸收粒子而跳过后续 `ParticleCut`。现在 Kokkos 能量 cut 与原有时间 cut 一样，保存 `observation_surface_reached_before_cut`，使终点地面记录与终点 cut 记录均可输出。到达吸收观测面时不再要求查找平面另一侧的介质。能量账本须扣除这两个 observable 的重复记账，不能把它们算作两个独立能量出口；对应修复位于 profile/router/report 层。

## 3. PROPOSAL 末态进入 CORSIKA 栈

原 CPU `InteractionModel::doSpecifiedInteraction()` 执行：

```cpp
view.addSecondary(code, E_proposal_total - proposal_particle_mass, direction);
// CORSIKA 栈保存动能；getEnergy() 返回动能 + CORSIKA get_mass(code)。
```

原 Kokkos 直接将 PROPOSAL 总能量保存为设备粒子总能量，遗漏了上述质量约定转换。现在在受支持过程的末态 materialization 边界执行一次：

```cpp
E_corsika_total = (E_proposal_total - m_proposal) + m_corsika;
```

这包括 photon pair、Compton、photoelectric、brems、ionization 和 Epair 的带电次级，以及离散反应后的存活带电母粒子；annihilation 的光子无需转换。连续传播、不发生反应的 continuation、LPM 拒绝，以及已经通过 CPU 生成后导入的状态不重复转换。二次级 thinning 接收转换后的栈总能量，方向及其抽样公式仍在 PROPOSAL 能量约定下计算。

实际保留下来的每个次级按其权重累计 `weight * (m_corsika - m_proposal)`，写入独立的有符号 `weighted_mass_convention_correction_GeV`。它是闭合账本的来源修正，不是介质中的能量沉积。

## 4. 新发现：原生 SampleLoss 与 Molière 的随机域冲突

旧 `RandomDomains.hpp` 同时定义：

```cpp
ProposalSelectionRandomProcessId = 0x454d0003;
ContinuousScatteringRandomProcessId = 0x454d0003;
ProposalSelectionDrawId = MoliereFirstAngleDrawId = 0;
```

对于到达离散反应顶点的带电粒子，输运不会先递增 `step_id`；Molière 和 `selectLeptonVertex()` 因此使用完全相同的 `(seed, shower, history, step, process, draw)`。这不是独立随机流间偶然的浮点相等，而是 **第一个散射随机数与原生过程／组分／loss 选择随机数必然相同**，会引入 CPU 原本没有的抽样相关性。

现在原生过程选择使用独立域 `0x454d0004`，其余域不变，新增编译期域唯一性断言，并将 `RandomDomainVersion` 升为 `2`。这项修复有意改变原生后端旧种子的 shower tree；同一新版本仍保持可重复性。旧版 decision tape 不能作为新随机域下的自然生长逐步一致性基准。

本地 beta4 的相同原生后端代码也能找到该域冲突；旧 `.c8emrt` 使用不同的选择路径，不会在相同位置执行新增的原生 SampleLoss 取数。这个差别提供了需要优先检验的原因，但**尚不能据此把历史约 1% profile 差别定量归因于该错误**。独立边缘分布通过，也不足以排除这种过程与散射的相关性；修复后必须重新做系综比较。

## 5. 新增独立测试

入口为 `tests/accelerator/testKokkosCpuTransportAlignment.cpp`，执行空间驱动为 `KokkosCpuTransportAlignmentDriver.cpp`，可分别编译为 OpenMP 或 CUDA。主要测试包括：

- e± / μ±、三个 cut、cut 上下与 10% 步长限制交接区共 72 个输入，调用真实 CPU `ContinuousProcess::getMaxStepLength()`，比较从实际 calculator 导出的 Kokkos range 步长。
- 12 个终止输入：观测面在 cut 前、与 cut 同步和 cut 后；CPU `doContinuous()` 在同一完成 chord 上计算能损，再通过真实 `make_sequence(observation, cut)` 检查双记录，共覆盖四种带电粒子的同步情况。
- 受支持末态与不同保留 mask：真实 CORSIKA `StackView::addSecondary()` 接收 PROPOSAL 动能，与设备 materialization 的总能量、动能、种类、权重和质量账本比较。该部分验证接口适配，不替代各末态角分布的独立 PROPOSAL oracle。
- 65,536 个 history 的同一步随机数检查：实际 Molière 函数记录随机数后以空 snapshot 停在物理计算前，真实 native selector 记录其选择随机数；同时重现旧域相等数并报告新域相等数和相关系数。这是随机数来源检查，不冒充 Molière 物理准确度测试。

这些确定性测试通过不等于全能区、所有介质或完整 shower 的统计验收通过。既有未修复版本的 2,000 例数据应保留为版本化基线，不应混入新版本系综。

## 6. 本轮实测记录

独立 Release OpenMP 构建的 `testKokkosCpuTransportAlignment` 已通过（CTest 约 2.19 s）：72 个标量步长输入、12 个终止输入、4 个观测与 cut 同时命中的输入，以及 30 个真实栈末态适配输入全部通过。CUDA 构建亦通过相同输入矩阵。连续步长最大相对差异分别为 OpenMP `6.4287226156778288e-11` 和 CUDA `6.2353775746048125e-11`，均低于该测试的 `1e-10` 门限。这些值是本组近 cut 步长比较的结果，不应解读为所有表／所有输入的误差上界。

65,536 个 history 的随机数来源检查结果如下：

| 检查 | 结果 |
|---|---:|
| 重现旧域时，散射与 SampleLoss 随机数完全相等 | 65,536 / 65,536 |
| 独立新域下，两随机数完全相等 | 0 / 65,536 |
| 新域两随机数的样本 Pearson 相关系数 | 0.0032028291324081996 |

上表随机数来源检查在 CUDA 上给出相同结果。完整日志由总验收统一归档到 `beta5_cpu_physics_alignment_20260906_v1/after/`。其他物理模块回归及修复后的 shower 系综结论应另行记录；不能由上述单项通过推断全部验收通过。

另一个独立的 `testKokkosScalarThinningAlignment` 在 200,000 个合法双次级输入上调用真实 CPU `EMThinning`、`test::Stack` 和 `StackView`，包含零权重、权重限制、Hillas／statistical thinning 和 erase 开关。OpenMP Release 的保留 mask 差异为 0，权重超出 16 ULP 的数量为 0，最大权重差异为 4 ULP；标量单位转换与 GeV POD 运算带来 30,258 个非逐位相同权重。该测试最初的 raw-pointer mock 曾产生与优化有关的陈旧权重读回，已彻底删除并替换成真实栈，不通过降低优化级别或调试打印绕过问题；没有据此修改 CPU 物理。日志为 `after/openmp_scalar_thinning_real_stack_oracle.log`。
