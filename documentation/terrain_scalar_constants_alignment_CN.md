# beta5：标量 CPU / Kokkos 常数与山体磁传播对齐

日期：2026-09-09。验收目标是物理、单位与接口语义一致，不要求完整 shower tree 逐位相等。

后续修复：本报告末尾记录的直线求交/微小弯曲推进问题已另行修改，专项与回归结果见 [直线磁传播策略修复](terrain_linear_magnetic_policy_fix_CN.md)。以下保留当时的测试结果与失败证据，不追溯改写。

## 本轮修改

1. 新增 [ScalarPhysicalConstants.hpp](../corsika/accelerator/ScalarPhysicalConstants.hpp)：统一当前 CORSIKA 标量单位约定下的磁偏系数、光速、电荷、真空介电常数。普通主机编译会用 CORSIKA 单位类型做静态核对；CUDA/HIP 只读取 double 常数，不引入主机单位模板。
2. [UniformMagneticField.hpp](../corsika/accelerator/em/common/UniformMagneticField.hpp) 的步长限制、磁传播、球面/观测平面求交使用同一磁偏系数；光子飞行时间、μ 衰变飞行距离、射电常数改为共享别名（后三者的数值不变）。
3. [InterfaceEmStep.hpp](../corsika/modules/transport/detail/InterfaceEmStep.hpp)（原 TerrainEmSession.cpp）的 DEM 二次曲线求交去掉重复硬编码。求交改用 `queryContinuousTransportMass()`，与 `transportLepton()` 的 CORSIKA 输运质量一致。原来求交使用 PROPOSAL 质量，实际推进却使用 CORSIKA 质量，属于需要修正的局部几何约定差异。
4. 核对发现 `Types.hpp::MuonMassGeV` 是一个当前生产代码没有引用的旧定义。将其从 `0.1056583755` 改为锁定 PROPOSAL 7.6.2 的 `0.1056583745 GeV`，避免未来误用。**实际生产用的 native calculator / auxiliary cache 质量本来就直接导出，不因此重新制表。**
5. 山体 summary 增加 `accelerator_constants`；通用 EM 配置输出记录 `scalar_transport_constants_version`、`magnetic_rigidity_GeV_per_T_m`，便于区分历史数据。

没有改输入磁场、RNG、截面、LPM、thinning、初级配置或原 CPU 单位库；没有改归档 beta4。生产服务继续使用原归档二进制，本轮只在独立 `build/mountain-openmp`、`build/mountain-cuda` 编译测试，不覆盖正式 install。

## 为什么采用这个磁偏系数

原 CPU 实际计算：

```cpp
auto pSI = constants::c * convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
double coefficient = constants::e * tesla / pSI * meter;
```

| 来源 | GeV–m–T 磁偏系数 |
|---|---:|
| 旧 native CUDA / Kokkos | 0.299792458 |
| 当前原 CPU 单位库推导 | 0.2997925146834389 |
| 本轮 beta5 共享值 | 0.2997925146834389 |

相对修正约 `1.89e-7`（0.189 ppm）。原因是旧 CPU 的 `hBar/hBarC/c/e` 单位换算组合，不是输入 IGRF 模型的经纬度错误。选择它是为兼容当前 CPU，并非宣称其比现代一致 SI 常数更准确。若未来更新上游单位系统，应同时更新共享值和回归基线。

原 CPU/portable 独立 12 点探针中，1 GeV、50 μT、1000 m 的位置分量差从约 1.13 μm 降至约 `1.14e-13 m`；最大方向分量差约 `1.11e-16`。这是局部算例，不是全 shower 或 Cherenkov 极限的全局误差界。

## 其他常数和单位的核对范围

| 用途 | 正确的来源 / 约定 | 处理 |
|---|---|---|
| 飞行时间、射电传播 | CORSIKA `c=299792458 m/s` | 共用一个定义，数值不变 |
| CoREAS/ZHS 归一化 | CORSIKA `e=1.6021766208e-19 C`、`epsilon0=8.8541878128e-12 F/m` | 共用定义，不改公式与累积顺序 |
| 输运、磁曲线、ParticleCut | CORSIKA generated mass：e± `0.0005109989 GeV`，μ± `0.1056584 GeV` | 保留；修正 DEM 求交的质量来源 |
| 截面、末态、散射、LPM | PROPOSAL calculator 的质量、`ALPHA/RE/NA/HBAR` 等 | 保留 PROPOSAL 约定，不能统一替换成 C8 常数 |
| PROPOSAL 次级返回 C8 | 先减 PROPOSAL 静质量得到动能，再由 C8 stack 加 C8 质量 | 原适配及独立 mass-convention ledger 保留 |
| μ 衰变 | C8 `get_lifetime()`，`2.196981e-6 s` | 与设备已有值一致；不是 PROPOSAL 的 `LMU` |
| 终止时间 | `ParticleCut::timePost > 10_ms` | 设备 `10.e-3 s` 一致，不是墙钟耗时 |
| 密度、grammage | `1 kg/m³=0.001 g/cm³`；积分中 m→cm 乘 100 | 核对实际 exporter、介质 snapshot 与积分器 |
| 能量与接口时间 | GeV↔MeV 乘/除 1000；CPU SI 秒 / GPU s；观测波形按自身采样单位换算 | 不把能量切换当质量约定修复 |
| 地球和地磁参考球 | C8 平均半径 6371000 m；IGRF 参考半径 6371200 m | 各有用途，不是应统一的两个错误常数 |

原生表的常数导出见 `src/accelerator/em/tables/ProposalNativeTable.cpp` 和 `ProposalNativeAux.cpp`；本轮未改这些文件或 Conan 缓存。

## 验收设计与复现

新增 [testKokkosScalarConstants.cpp](../tests/accelerator/testKokkosScalarConstants.cpp) 和独立 [ScalarConstantsDriver.cpp](../tests/accelerator/ScalarConstantsDriver.cpp)。主机端使用真实 CORSIKA `LeapFrogTrajectory` 与量纲换算，设备端调用生产传播函数，不用同一个 portable 函数作为两边的唯一参考。

- 288 个轨迹输入/后端：e−、e+、μ−、μ+，4 个动能（0.5001 MeV 至 100 TeV），零/50 μT，3 个方向，3 个步长。低于生产 cut 的点仅用于运动学函数边界测试。
- 检查位置、方向、飞行时间、最大偏转步长、generated mass、PROPOSAL 质量、寿命和 grammage；保留严格局部门限，不要求全树相等。
- 重跑 resident terrain 几何/容量/非法输入门禁、百万曲线求交、真实 CPU 连续输运/cut，以及真实 CPU CoREAS/ZHS 逐轨迹 oracle。
- 实际 21CMA DEM：CPU/OpenMP/CUDA 各跑光子向上/向下、e−/e+ 岩内、1.1 MeV 总能量 e−/e+ 空气入岩、10 TeV νe 自然传播/强制 CC（粗 thinning）的受控用例。空气 IGRF14/2027，岩内 B=0；使用已有表和辅助缓存。

```bash
conda activate corsika_venv
cmake --build ../build/mountain-openmp --target testKokkosScalarConstants
python validation/terrain/run_scalar_constants_acceptance.py \
  --build ../build/mountain-openmp --output /path/to/new/module_results
# CUDA 同理，使用 mountain-cuda，给上面的 Python 命令加 --gpu。
```

设备用例由资源守护器限制：诊断任务 RSS ≤2 GiB、系统可用 RAM ≥3 GiB、设备总显存相对启动基线增量 ≤512 MiB（且 ≤总显存 10%）。显存增量不是 NVML 每进程测量；既有生产仍运行，不能把本轮墙钟时间当独立加速基准。

结果根目录：`D:\CorsikaData\corsika_validation_results\beta5_scalar_constants_terrain_alignment_20260909_v1`。

## 已完成结果

独立 OpenMP、CUDA Release 构建中的 `c8_terrain_cascade`、`c8_air_shower` 和本轮测试目标均编译通过。空气应用源码 SHA-256 保持 `d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`；归档生产二进制 SHA-256 保持 `671bd3f9a52ab954ff61609040552e931ebc8f053c8e53fbd5a3d26ee994b159`。

| 检查 | OpenMP | CUDA |
|---|---|---|
| 288 点常数/LeapFrog 多项式/步长检查 | 全通过；位置/方向最大绝对差 `1.11e-16` | 全通过；位置 `1.11e-16 m`、方向 `2.22e-16` |
| 同组飞行时间、均匀介质 grammage、偏转步长 | 所抽输入与 CPU 数值相同 | 所抽输入与 CPU 数值相同 |
| resident terrain 几何/容量/非法输入门禁 | 通过 | 通过 |
| 曲线求交：1,000,000 随机输入加边界点，两种面排列，共 2,000,038 检查 | 通过；最大距离差 `1.11e-16 m` | 通过；host/device 位差计数 0 |
| 真实 CPU 连续输运/cut/次级适配/RNG 域检查 | 通过；最大 grammage 相对差 `6.43e-11` | 通过；最大 `6.24e-11` |
| 真实 CPU CoREAS/ZHS 逐轨迹与时间窗/复用回归 | 通过 | 通过 |
| 真实 DEM 8 个受控事例 | 8/8 完整 | 8/8 完整 |

加上标量 CPU 的 8 例，共 **24/24** 完成。所有记录有限、时间不倒退、方向合法、无材料归属错误、CSV 未截断、加速队列清空，空气/岩石表身份与场参数检查通过。7 个应用级失败门禁也通过。

新增低能 e−/e+ 事例在三个后端均为 2 步、1 次空气入岩；总能量 1.1 MeV，沉积动能约 `0.0005890011 GeV`，差异仅在最后浮点位。该场景实际覆盖了修正质量约定后的带电曲线穿界。

OpenMP/CUDA 的 8 例中，`step/pdg/medium` 离散序列相同；原来的严格浮点整轨迹门限只有低能 e± 和自然 νe 三例通过，其余五例未通过。**保留 `strict_trace_passed=false`，不将一致步数当作物理等价证明。** CPU/加速随机调度不同，尤其粗 thinning 的单个强制 CC 事例的步数差异不能代替系综检验。

与旧的两个 CPU 光子参考比较：轨迹通过既定浮点门限、能量和权重相同，最大位置差 `2.22e-16 m`。旧通用比较器的总标志仍为 false，唯一 diagnostics 不同是记录上限 `csv_row_limit=500000` 对 `300000`（两者都未截断）；原始结果保存在 `cpu_before_after.json`，没有修改旧文件或掩盖标志。

诊断峰值：山体 CUDA RSS 约 730 MiB，设备总显存增量峰值 441 MiB；模块检查显存增量峰值 253 MiB。均未触发保护，既有生产服务保持运行。不同批次的峰值不能相加成同一时刻的占用。

主要原始记录：

- `modules_openmp/manifest.json`、`modules_cuda/manifest.json`：二进制/源码哈希、CMake 配置、五项测试、资源采样。
- `terrain_summary/summary.json`、`terrain_summary/REPORT_CN.md`：24 例、首个浮点差异、各字段残差、配置和资源。
- `scalar_probe/stdout.log`：独立 CPU 单位推导和原始 12 点磁多项式对照。
- `unused_muon_constant_before/`：修复旧 μ 常量前的失败记录（每个输入都复查同一个全局常量，因此失败 288 次不代表 288 个过程出错）。改正后两后端均通过。

本轮**不**包含大样本系综验收，也**不**把当前真实 DEM 应用（射电尚未接入）的轨迹测试称为跨介质射电验收。已有局部 CoREAS/ZHS oracle 与完整岩石—空气传播验收必须分开。

## 另发现的待修复项：直线退化决策没有完整传递给推进

这不是换算常数问题，本轮未混入额外输运分支修改：

- 原 CPU `TrackingLeapFrogCurved::getTrack()` 在 `p_perp < 1 eV` 或 `R_g > 1e9 m` 时调用 `getLinearTrajectory()`，返回零场轨迹；山体 CPU `MagneticTracking` 也执行该语义。
- 加速 `maximumUniformMagneticStep()` 同样返回 `Linear`，空气/DEM 求交因此走直线；但 `LeptonTransportStep.hpp` 随后的 `advanceUniformMagneticField()` 仍接收原始非零场。后者本来只是多项式推进原语，不自行决定 tracking 策略。
- 新的 `c8_terrain_scalar_magnetic_probe --linear-gate` 实际复现了这两个 portable 调用不一致：100 TeV 电子、50 μT、人为给定 1000 m，`R_g=6.67128e9 m`，限制器为 Linear，推进相对 CPU 直线仍有 `7.49481e-5 m` 横向偏差和 `1.49896e-7` 方向差。**此长度是受控输入，不是自然 shower 的实际步长或全局误差界。** 原始失败记录为 `linear_gate_unresolved/`，程序返回 1。

下一步应在输运调度层将同一个 Linear 决策传给推进：该分支使用局部零场/直线推进，而非修改环境里的物理磁场；常规有弯曲分支保持不变。补上阈值两侧、近磁场平行、极大半径、直线 DEM/观测面命中的真实 CPU 和设备测试后，再重跑本轮集成用例。

因此，本报告的结论是：**换算常数和 DEM 质量来源已修复，指定的局部回归及小样本记录完整性通过；并非所有高能输运语义均已无差异。**
