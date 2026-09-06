# beta5：CPU 能量沉积位置与闭合账本对齐

本修改针对 `BETA5_KOKKOS_CPU_PHYSICS_AUDIT_20260906_CN.md` 中的问题 2、3 的输出记账，以及问题 4。原标量 CPU 的 `ContinuousProcess`、`ParticleCut` 和 `EnergyLossWriter` 未修改。代码变化不等于大样本验收已经通过。

## 修复内容

1. 连续能损仍写起点—终点线段；`ParticleCut` 的剩余动能单独写在实际终点。光子时间 cut 也采用点状沉积。原先相加后沿线段摊分会把跨 bin 的 cut 能量放错位置。
2. 驻留 profile、紧凑 projected record、完整 `EmStepRecord` 和 CPU output sink 使用相同的拆分规则。两个输出记录保留原 `deposited_energy_GeV` 总量，新增包含于总量中的 `cut_deposited_energy_GeV`，避免把额外字段再次相加。纵向粒子计数和射电轨迹仍只写一次。
3. `observation_surface_reached_before_cut` 同时保留观测输出和 CPU 约定的 cut 沉积。另加 `observation_cut_overlap_energy_GeV`，从能量闭合账本减去重复出口。对轻子，重复部分是 cut 动能加被移除的静质量；对光子则是 cut 能量。
4. 光电效应的电子末态公式不变。原 CPU 不把 K-shell 结合能传入 legacy `dE/dX`，故 Kokkos 也不再把它写入该 profile。能量未被静默删除，而是记录到独立 `unwritten_photoelectric_binding_energy_GeV` 并加入闭合出口。
5. cut 静质量记账使用 CORSIKA 的 `get_mass()` 常数，与栈和原 `ParticleCut` 相同；PROPOSAL 原子电子过程的介质静质量输入仍使用其物理模型常数，不混淆两种用途。
6. 末态从 PROPOSAL 总能量转为 CORSIKA 动能栈时，静质量常数的微小差别会改变总能量约定。每个最终保留的带电次级以实际权重记录有符号 `mass_convention_correction_GeV`，单独进入闭合源项，不冒充物理介质能量输入。LPM 拒绝顶点不产生此项。
7. 补齐非驻留 projected 轻子末态的原子电子静质量账本：此前完整记录与 resident 路径有这项，projected 分支缺少对应归并。这只影响完整性诊断，不改变末态或 legacy profile。

因此新闭合定义为：

```text
initial_total + medium_electron_rest + mass_convention_correction
  = legacy_deposit + cut_rest + observed_total + escaped_total
    + unwritten_photoelectric_binding - observation_cut_overlap
```

这是对齐原 CPU 输出约定，不声称原 CPU 的光电介质响应记账已经完整。以后若要把结合能作为真正的局部沉积公开输出，应同时修改 CPU 和加速后端，并另建参考数据。

## 单元测试及边界

新增 `tests/accelerator/testKokkosCpuDepositionAlignment.cpp`：

CPU oracle 主文件使用普通 C++ 编译；`KokkosCpuDepositionAlignmentDriver.cpp` 使用相应后端编译器执行设备 kernel，经 POD 接口交换结果。这样既实际调用原 CPU 模块，也不会要求 nvcc/hipcc 解析整个 CPU writer 的模板依赖。

- 独立 expected value 来自实际 **CPU `EnergyLossWriter` + `ParticleCut::doContinuous()`**，不是再次调用同一套加速 helper。
- 420 组条件覆盖 e±、μ±、γ，顺／逆向跨 bin、极短线段、恰在 bin 边缘、超出最后 bin、多个 bin、零长度，三种权重，以及观测面与 cut 同时发生的状态。
- 每组比较驻留 Kokkos 累积、完整记录 sink、紧凑 projected sink 三条路径的逐 bin 值。
- 连续分箱阈值同时测试默认 `1e-4 g/cm²` 和零。点状沉积不能调用零长度、零阈值的线段公式，否则会产生 `0/0`；因此使用独立点状累积入口。
- 调用真实 PROPOSAL `PhotoeffectNoDeflection::CalculateSecondaries()`，验证空气 N/O/Ar 组分的电子能量和独立结合能账本，确认 legacy profile 不额外增加结合能。
- 检查 fixed-point overflow、非法记录和观测/cut 重复出口。

这里的轨迹起止点和连续能损是控制输入，**不是完整连续输运 oracle，也不是全 shower 统计验收**。Kokkos-CUDA 与 OpenMP 的编译和执行结果由本轮主验收日志记录；不能仅以新增测试源码代替运行通过的证据。

本次已实际完成 OpenMP 和 CUDA 两个独立构建的测试：420 组条件、三条输出路径，以及 N/O/Ar 光电末态 oracle 均通过。详细输出记录于本轮验收目录的 `after/openmp_deposition_oracle.log`、`after/cuda_deposition_oracle.log` 和最终 CTest 日志。这一结论只覆盖上述受控案例，不代替大样本 shower 验收。

本轮不引入性能优化。紧凑 projected 诊断路径为了匹配 writer 的点状分箱，会读取不可变的 bin 配置；后续可以独立评估缓存这项配置。默认设备驻留 profile 路径不经过这一处理。

## 文件职责

| 文件 | 内容 |
|---|---|
| `ProfileAccumulationStep.hpp` | 线段／点状沉积分离；resident 结合能及重复出口账本。 |
| `ProfileProjectionStep.hpp`、`Types.hpp` | 输出记录携带 cut 能量及双重终止标志。 |
| `ProfileProjectionData.hpp`、`KokkosProfileAccumulator.hpp` | 新增确定性定点账本字段和下载。 |
| `CorsikaOutputSink.hpp` | 完整记录用 CPU writer 的点重载；projected 记录按相同点状分箱写入。 |
| `PhysicalAcceleratedEmRouter.hpp` | 非驻留路径及最终 profile 合并的同口径账本。 |
| `KokkosShowerReport.hpp` | 显式导出新账本项，并修正闭合等式。 |
