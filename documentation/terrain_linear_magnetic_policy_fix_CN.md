# 直线求交与磁场推进策略对齐

日期：2026-09-09。范围：beta5 共用轻子输运；不改原 CPU tracking、磁场模型、随机流或 PROPOSAL 表。

## 原因与修复

CPU `TrackingLeapFrogCurved::getTrack()` 在零磁场、横向动量小于 `1 eV/c` 或回旋半径大于 `1e9 m` 时，返回 **B=0、k=0 的直线轨迹**。设备端的 `maximumUniformMagneticStep()` 已有对应判定，球面/观测面求交与 DEM 查询也遵循该判定；但共用 `transportLepton()` 在末端推进时无条件使用物理磁场。因此，同一步可能先按直线求交，随后却带有微小弯曲。

修复位于 [LeptonTransportStep.hpp](../corsika/accelerator/em/detail/LeptonTransportStep.hpp)：

```cpp
double const zero_field_T[3]{0., 0., 0.};
auto const* propagation_field_T =
    magnetic_limit.status == MagneticStepStatus::Linear
        ? zero_field_T : environment.magnetic_field_T;
auto const magnetic_advance = advanceUniformMagneticField(
    start, mass_MeV / 1000., charge_number, propagation_field_T, distance_m);
```

只复用已经作出的策略决定，不重算阈值、不修改环境中的真实磁场；粒子下一步仍重新判定。正常曲线分支仍调用原多项式，参数及操作顺序不变。直线分支的位置、方向、飞行时间、chord/grammage 以及射电 track 的磁偏标记现在使用同一个直线约定。多重散射属于后续独立物理过程，仍按原流程执行。

没有增加或改变 RNG 调用。但受端点修正影响的事件，后续边界/相互作用竞争可能改变，因此不能承诺与旧的有缺陷版本生成相同 shower tree。

这不是山体专有问题：空气和山体的 Kokkos 输运都调用该共用函数。原 CPU 的直线轨迹构造没有此问题。本轮不修改历史 beta4，也不覆盖正在运行的归档生产二进制。

## 专项回归

[MagneticLinearGateChecks.hpp](../tests/accelerator/MagneticLinearGateChecks.hpp) 使用实际 CPU `getTrack()` 构造参考轨迹，通过独立设备翻译单元调用 **生产 `transportLepton()`**，不是只对比两次相同的 portable 函数。

- 每后端 56 例，覆盖 e−/e+/μ−/μ+；36 个直线分支、20 个曲线分支。
- 回旋半径在阈值两侧（0.999、1.001 倍）；100 TeV 高能点；零磁场、平行磁场、常规曲率；e± 横向动量在 1 eV/c 两侧。
- 两类边界：倾斜吸收观测面、非吸收 external material boundary（山体接口使用的合同）。外部边界的这组局部测试是受控输入，不冒称实际 DEM shower。
- 比较端点、末方向、飞行时间、边界残差、grammage、终止类型和磁弯曲标记；不要求跨后端生成相同 shower tree。

修复前运行同一新增测试：56 例中 28 例失败。其中既有非零横向位移，也有平行磁场下残留的非零 `bend_parameter`。100 TeV、50 μT、设定 1000 m 局部步长的横向位移为约 75 μm；阈值附近约 0.5 mm。这些是局部复现用例，**不是完整 shower 的误差界或真实典型步长**。

## 验收结果

修复后 OpenMP、CUDA 专项均为 **56/56 通过**；两后端相对真实 CPU tracking 的最大位置差均为 `3.25e-19 m`、最大方向分量差 `1.11e-16`、观测面残差 `1.95e-19 m`，飞行时间相对差 `4.44e-16`，grammage 相对差 `2.22e-16`。数值均为本组输入上的实测最大值。

独立 `build/mountain-openmp`、`build/mountain-cuda` 的 Release 构建均通过，山体与空气应用均重新链接成功。

| 回归 | OpenMP | CUDA |
|---|---|---|
| 本次真实 CPU tracking / 生产 transport 策略检查 | 56/56 | 56/56 |
| 常数与原始 LeapFrog 多项式 | 288/288 | 288/288 |
| resident terrain、容量、非法输入与边界竞争 | 通过 | 通过 |
| 曲线几何独立 oracle | 2,000,038 检查通过 | 2,000,038 检查通过 |
| 真实 CPU CoREAS/ZHS、时间窗、复用回归 | 通过现有门限 | 通过现有门限 |
| 连续输运/cut/末态适配/RNG 域 | 72/12/30/65,536 点通过 | 72/12/30/65,536 点通过 |

真实 21CMA DEM 集成共 **24/24 完成**：CPU、OpenMP、CUDA 各 8 例，包括上下行光子、岩内 e±、低能 e± 空气入岩、νe 自然传播与粗 thinning 强制 CC。空气使用 IGRF14/2027、岩内 B=0，队列及记录完整性检查通过。

另外，分别将三个后端的新结果与上轮同后端、同条件的 8 个回归用例比较，三个 `*_before_after.json` 均通过；所记录轨迹数值的最大绝对差为 0，diagnostics 相同。这证明这些未触及故障条件的受控回归没有改变，**并非声称不同后端之间生成相同 shower**。

`terrain_summary/summary.json` 仍如实记录 `integration_passed=true`、`strict_trace_passed=false`：此前已有的 OpenMP/CUDA 严格逐行浮点差异未因本次修复消失，也没有新增。这是局部语义与集成回归，不是新的大样本物理认证；本轮实际 DEM 场景没有开启跨介质射电，不能将独立 CoREAS/ZHS 单轨迹测试解释为完整山体射电验收。

结果根目录：

```text
D:\CorsikaData\corsika_validation_results\beta5_linear_magnetic_gate_fix_20260909_v1
```

`before_openmp/` 保留失败证据；`after_openmp/`、`after_cuda/` 保存修复后专项；`modules_openmp_final/`、`modules_cuda_final/` 保存最终测试二进制和源码哈希、五项模块结果及资源记录；`magnetic_{cpu,openmp,cuda}_final/`、`terrain_summary/` 和三个 `*_before_after.json` 保存集成及前后对照。早期 `modules_openmp/` 是补充显式标准库 include 前的同算法测试记录，最终以 `_final` 为准。

旧报告中的 `probe_scalar_magnetic_units --linear-gate` 直接把非零 B 传给底层多项式，用来复现旧调用方式；它不执行已修复的 `transportLepton()`，不能作为当前生产路径是否修复的验收工具。

复现（独立构建，不覆盖生产）：

```bash
conda activate corsika_venv
cmake --build ../build/mountain-openmp --target testKokkosCpuTransportAlignment --parallel 1
python validation/terrain/run_scalar_constants_acceptance.py \
  --build ../build/mountain-openmp --output /path/to/new/openmp_results
# CUDA 使用 mountain-cuda，并在 Python 命令上加 --gpu。
```

诊断沿用 RSS 2 GiB、可用 RAM 保底 3 GiB、设备总显存相对启动基线增量不超过 512 MiB 且不超过总量 10% 的资源门禁。增量不是精确的每进程显存归属。生产仍在运行，本轮不将诊断墙钟时间作为性能基准。

实际 DEM 诊断最大 RSS 约 730 MiB，设备总显存增量最高 446 MiB；模块测试对应最高约 323 MiB、255 MiB，均未触发守护器。无表重建、无生产服务停启、无 install 覆盖。

空气应用源码 SHA-256 仍为 `d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`；生产归档二进制仍为 `671bd3f9a52ab954ff61609040552e931ebc8f053c8e53fbd5a3d26ee994b159`。正在运行的生产任务 **尚未切换到本次修复二进制**，后续部署应独立记录版本，避免混合样本而不标注。
