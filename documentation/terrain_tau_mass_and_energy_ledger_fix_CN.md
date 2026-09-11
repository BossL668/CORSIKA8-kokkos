# 山体 τ 质量修复与独立能量总账

2026-09-10。仅针对 `c8_terrain_cascade` 和山体中微子适配器；没有修改
`c8_air_shower`、TAUOLA 安装库或已有的三例生产数据。

## 1. 修复了什么

旧 TAUOLA Fortran 默认 `AMNUTA=0.010 GeV`，CORSIKA 的 ντ 质量为零。
旧山体适配器先在低能参考系生成，再把三动量按 CORSIKA 质量重新上壳，
最后 boost 回高能实验室系。质量转换引入的能量差因此也被 boost；
这解释了旧三例约 `1.2–1.4e-5` 的 τ 衰变相对缺口。

新增 `corsika/modules/neutrino/TauolaMassConvention.hpp`：在**生成之前**
将 TAUOLA 的稳定末态质量与 CORSIKA 对齐，尤其令 ντ 质量为零。
`ConditionedTauolaDecay` 在一次衰变的作用域内使用该约定，正常返回和异常退出均恢复
原 common-block 数值；不改变库文件，不影响之后的普通 TAUOLA 调用。
实验室 boost 使用实际 CORSIKA 母 τ 的质量。

没有对生成后的能量做整体缩放，没有为了通过验收重抽衰变，也没有增加随机数调用。
TAUOLA 为串行、全局状态生成器，此适配器仍要求主机串行调用，不能在线程间并行调用。
修复改变物理配置，因此旧、新种子不承诺形成相同完整 shower 树。

生成器内部仍有 Fortran `REAL*4` 精度。当前实际测试的 1800 次 τ± 衰变，
覆盖 `1e4 / 1e8 / 1e12 GeV` 和三个方向，最大能量、动量相对残差均为
`1.291e-6`，通过 `2e-6` 门限；不是 double 精度逐位守恒的声明。
低能路径与**同样质量对齐后的** stock TAUOLA 比较末态及 C++/Fortran 随机流；
非 τ Pythia 和显式 Pythia τ 路径另与未改动的 stock 模块比较。
质量作用域恢复、异常恢复、极化测试也单独运行。

## 2. 为什么“沉积 + 剩余粒子能量”不一定等于初级能量

新增诊断选项 `--energy-ledger`，默认关闭。它不是一个补偿项，也不会修改粒子。
定义（均为粒子权重乘**总能量**，GeV）：

- `G`：每个反应/衰变后、薄化前，次级总能量减母粒子能量；包括介质靶粒子
  引入的能量、生成器未跟踪的末态及生成器数值残差，不能把整项叫作“数值错误”。
- `T`：薄化后次级加权总能量减薄化前的值。限重统计薄化仅在期望上保持能量，
  所以单次 shower 的 `T` 可以正也可以负。
- `R`：连续输运和 cut 从粒子栈移走的总能量，减去实际写出的沉积 `D`。
  普通 `ParticleCut` 只写动能，因此移除的静质量在 `R` 中而不在 `D` 中。
- `S`：时间窗末仍存活粒子的加权总能量；`W`：离开整个环境的粒子能量。

逐项核对的恒等式是：

```text
D + S + W - E_initial = G + T - R + unexplained
```

这里 `unexplained` 才是以上分类尚未解释的总账差额。即便它很小，也只是说明
输运账本对得上，**并不能单独证明 G 内所有核残余/靶能量都已被物理建模**。
不把静质量或窗末中微子能量直接改写为热沉积，也不凭空补上射电能量。
本次山体事件关闭射电计算。

## 3. 实现位置与无扰动检查

- `applications/detail/mountain/TerrainEnergyLedger.hpp`：主机累计器和原样委托
  `EMThinning` 的诊断包装器；不储存逐粒子历史，内存不随步数增长。
- `TerrainShowerOutput.hpp`：连续能损、CPU 末态、设备轨迹、世界出口的计数；
  使用原 `ParticleCutStatistics` 得到 CPU cut 动能/静质量。
- `src/terrain/TerrainEmSession.cpp`：仅在开启诊断时，用已经选好的过程和已经
  生成的随机参数重建一次未薄化末态；不重新抽样、不再次入栈、不再次沉积。
- `TerrainTauDecay.hpp`：质量约定和未归一化说明写入 metadata；局部 τ 能量残差保留。

CPU 与 OpenMP 各做 photon/proton/tau 的账本开关对照，要求 `tracks.csv`、
`deposits.csv`、`window_survivors.csv` 的 SHA-256 完全一致。
纯 EM 无薄化和带强子/薄化的低成本事件先验证，再重跑高能事件。

## 4. 高能复跑与结果位置

PSR 独立队列 `c8-nutau-energy-ledger-v2.service`，种子 158、946、3605，
沿用原 100 PeV 自然 CC/NC、自然 τ 衰变、100 μs 时间窗和原 cut/thinning。
使用 OpenMP 120 线程，无 GPU；每例 RSS 24 GiB、服务 28 GiB、4 小时墙钟保护。
大 CSV 留在 PSR，不覆盖旧样本。

结果目录名 `beta5_tau_mass_energy_ledger_20260910_v2`，位于既有
`CorsikaData/corsika_validation_results` 下。`REPORT_CN.md` 是实际已完成样本的
更新结果，`small_regressions.json`、`tau_energy_matrix.csv`、JUnit XML 保存验收依据。
旧三例没有逐顶点 `T` 和完整 `R`，因此不能仅用旧图唯一反推各项。
新账本复跑的数值应标明“修复后复跑”，不能冒充旧文件的逐项复原。

### 第一例实测（seed 158）

2026-09-10 修复后完整复跑：8940447 条轨迹、15265680 条沉积，未截断、
零介质误配、零 pending 粒子。τ 局部相对残差从 `−1.4122e-5` 降至
`−1.6989e-8`（约 769.54 GeV → 0.926 GeV）。

`D+S−E` 为 `−2.743020 PeV`；薄化跳变量 `T=−1.569436 PeV`，
cut 静质量 `10.813720 PeV`，生成器净变化 `G=+9.640137 PeV`。
后两项相抵为 `−1.173583 PeV`。最终未解释差额仅 `−0.0002005 GeV`，
占初级 `−2.005e-12`。**不能把全部百分比差额归因于薄化；也不能把 cut
静质量直接补成热沉积。** 核残余/靶交换的进一步物理细分仍不同于总账闭合。

其余两例由独立服务继续执行，未完成前不标为三例验收通过。具体完成数以结果
目录 `status.json` 和已更新的 `REPORT_CN.md` 为准。
