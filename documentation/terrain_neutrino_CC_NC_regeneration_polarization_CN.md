# 山体中微子：CC/NC、τ 再生和极化扩展

后续原版模块复核与 TAUOLA 接线、随机流／极高能 boost 修复见
[原版对齐报告](terrain_original_neutrino_alignment_CN.md)。本文件保留上一阶段的 Pythia 控制结果；
重放这些结果须显式指定 `--tau-decay-model pythia`。

日期：2026-09-09。只涉及独立 `c8_terrain_cascade` 和项目内中微子模块；
不修改 `c8_air_shower`、生产归档二进制、共享 EM/磁场内核或 PROPOSAL 依赖。

## 结论与未完成门禁

本轮已实现并通过 **CC/NC 竞争、NC 中微子继续输运、再生链连接、指定纵向极化的衰变响应** 控制测试。
**用户要求的完整中微子物理尚未全部完成，不能以本报告代替生产级完整验收。**

|环节|现在的实现/证据|仍需补齐|
|---|---|---|
|ν/ν̄-N CC、NC|CTW2011 总截面；Pythia 8.315 指定 CC/NC 末态；六味 NC 顶点控制|总截面和微分末态使用同一 PDF/低 Q² 定义的验收|
|NC 再散射|NC 后同味中微子和所有其他末态入栈，继续使用 CC+NC 竞争|更长柱深的自然概率与出山谱参考对照|
|ντ→τ→ντ|不限制再生代数；τ 在 CPU 经原生能损、磁传播与衰变；女儿中微子重新入栈|CTW 拟合以下的低能弱输运|
|τ 极化衰变|显式给定纵向 P；两种电荷；πν 解析角分布及全部开启通道的分支比|从 CC 硬顶点求出的事件级自旋密度矩阵，不是给全部 τ 指定一个常数|
|τ 穿介质自旋演化|**未实现**|连续/离散能损退极化、必要的自旋输运及其误差预算|
|完整中微子物理验收|输出 `full_neutrino_physics_validated: false`|独立传播代码的自然再生谱对照及明确能区/过程覆盖|

当前 CTW 拟合仅在 `1e4–1e12 GeV` 内有定义。低于 10 TeV 的次级中微子目前仍以零弱反应率运输到逃逸，
不是被能量沉积，也**不意味着其真实截面为零**。本轮山体 CC 样本确实产生了这样的低能次级；
因此不能把“再入栈成功”写成“全能区再生物理完整”。这一限制明确写入结果元数据。
若要声称所有味、所有过程完整，还必须先定义 ν-e 反应（包括 Glashow 共振）等过程是否在研究范围内。

## 1. 实现路径

```text
νe/νμ/ντ 及其反粒子（CPU）
  ├─ 距离竞争：σtotal = σCC + σNC
  ├─ 目标核选择：组成 × A × σtotal（当前为 isoscalar 近似）
  └─ 电流选择：P(NC) = σNC / σtotal
       ├─ CC → 对应带电轻子 + 强子末态
       │        └─ τ：PROPOSAL 能损 + 原生寿命调度 → 衰变 → ν 再入栈
       └─ NC → 同味 ν + 强子末态 → ν 再入同一竞争

γ/e± → 已有 Kokkos EM（OpenMP 或 CUDA）
μ/τ/ν/强子 → CPU；本轮不计算射电
```

核心代码：

- [`AuditedPythiaNeutrino.hpp`](../corsika/modules/neutrino/AuditedPythiaNeutrino.hpp)：
  `WeakCurrent` 和 `generate(..., current)`。CC 打开 `ff2ff(t:W)`，
  NC 打开 `ff2ff(t:gmZ)`，另一路明确关闭。
  不能用 CC 末态改名假装 NC；NC 审计期望的出射轻子 PDG 就是入射中微子 PDG。
- [`MountainNeutrinoInteraction.hpp`](../corsika/modules/neutrino/MountainNeutrinoInteraction.hpp)：
  总反应率、单次通道选择、完整次级入栈，新增 `history_id`、`parent_history_id`、
  generation、current 和所有女儿的能量/身份记录。诊断达到显式上限则报错，不静默截断。
- [`PrescribedTauDecay.hpp`](../corsika/modules/neutrino/PrescribedTauDecay.hpp)：
  原生 CORSIKA/Pythia 的**静止系 τ** 衰变极化适配，未改动上游衰变器或矩阵元。
- [`TerrainTauDecay.hpp`](../applications/detail/mountain/TerrainTauDecay.hpp)：
  在原生寿命调度上记录 τ 的真实飞行终点和女儿。指定极化的对象只处理显式输运的 τ，
  不修改其他原生强子衰变中的 τ 约定。
- [`c8_terrain_cascade.cpp`](../applications/c8_terrain_cascade.cpp)：只增加参数和模块装配；
  process sequence 次序保持不变。

默认 DEM 应用现在启用 `cc+nc`。核心类构造器的缺省通道仍为 CC-only，
以保持旧有限几何应用的显式历史配置。原 DEM CC-only 诊断需指定 `--neutrino-channels cc`。

## 2. 测试确实找出的两个接口错误

### 2.1 重用 Pythia 后反中微子初始化失败

同一生成器按 ν→ν̄ 或不同靶核子切换时，曾出现
`all processes have vanishing cross sections`。
Pythia 8.315 `BeamSetup::initPDFs()` 只在 `pdfAPtr == 0` / `pdfBPtr == 0` 时建立 PDF，
连续 `init()` 不清空这些对象；原来属于 ν 的 point-lepton PDF 不应继续用于 ν̄。

现在在**真实 projectile/靶核子身份改变**时重建独立 Pythia 对象，并重新配置过程。
仍连接同一个 CORSIKA `pythia` 随机流，**不重置 seed、不跳过失败事件、不换能量**。
同一束流只改变能量时保留对象。六味 × p/n/O 的 NC 测试和正反 τ 再生链测试均覆盖切换。
修复仅限项目内 `AuditedPythiaNeutrinoFinalState`。

### 2.2 静止系 τ 的自旋轴不能直接套用运动 τ 的 helicity 符号

CORSIKA 的原生衰变器先将 τ 放在 Pythia 静止系，再沿实验室 τ 动量方向 boost。
Pythia 8.315 `HelicityParticle::wave()` 在 `|p|+pz == 0` 时采用与 −z 对齐的基；
`p=0` 恰好进入这个分支。直接把沿 CORSIKA +z boost 轴的 P 原样交给 Pythia，
会把 π 的角分布斜率反转。

本轮第一轮测试真实测得：请求 `P=-1`，却有 `<cosθπ>≈+1/3`，与解析预期 `−1/3` 相反。
修复只在这个**静止系适配器**中翻转传给 Pythia 的参数，之后重新 init（其设置在 init 时缓存）。
没有反转检验标准，也没有改 Pythia 矩阵元。τ+ 的共轭符号仍由 Pythia 自己处理。
原来没有明确传入极化的大气衰变路径没有改动。

对于用户参数 `P=Pτ−`，以 boost 轴定义角度，两种电荷的归一化控制分布均应满足

\[
 f(\cos\theta_\pi)=\tfrac12(1+P\cos\theta_\pi),\quad
 \langle\cos\theta_\pi\rangle=P/3.
\]

τ+ 的真实纵向极化是 `−P`，其 π 分析本领的符号也相反。
这项检验确认的是**给定自旋后的衰变响应**，不是确认 CC 产生的自旋已经算对。

## 3. 运行方式

在已完成 beta5 环境和独立 mountain 构建的基础上，使用已有 `scene.yaml`：

```bash
# 自然 CC+NC 传播；位置必须是 scene 的 ENU 坐标。
./build/mountain-openmp/applications/c8_terrain_cascade \
  --scene scene.yaml --output output/nu_natural \
  --primary nu_tau --energy-GeV 1e5 \
  --position-m X Y Z --direction DX DY DZ

# NC 条件顶点，仅用于调试；不能拿其计数推导天然事件率。
./build/mountain-cuda/applications/c8_terrain_cascade \
  --scene scene.yaml --output output/nu_nc \
  --primary nu_tau --energy-GeV 1e4 \
  --position-m X Y Z --direction DX DY DZ \
  --force-vertex-nc --em-backend kokkos --device-memory-MiB 192
```

命令中的 X/Y/Z、DX/DY/DZ 是待替换的几何数值；不是 shell 自动变量。
默认空气使用 IGRF14/2027、山体内部 B=0，原有地形/大气构造未改变。

|参数|含义|
|---|---|
|`--neutrino-channels cc+nc`|默认；自然距离和通道竞争使用 CC+NC|
|`--neutrino-channels cc` / `nc`|单过程对照，不是完整模型|
|`--force-vertex-cc` / `--force-vertex-nc`|首顶点条件化；互斥；之后继续走所启用的自然通道|
|`--tau-minus-polarization P`|显式**衰变控制**，`−1≤P≤1`；τ+ 为 −P；不自动推导 CC 极化|
|不指定极化参数|保留原生未知产生机制的衰变处理；不可称为已传递 CC 自旋|

## 4. 本轮验收

结果目录：

```text
D:\CorsikaData\corsika_validation_results\beta5_neutrino_cc_nc_spin_20260909_v1
```

脚本：[`run_neutrino_extended_acceptance.py`](../validation/terrain/run_neutrino_extended_acceptance.py)、
[`analyze_neutrino_extended_acceptance.py`](../validation/terrain/analyze_neutrino_extended_acceptance.py)。
数值 oracle：[`testMountainNeutrinoExtended.cpp`](../tests/modules/testMountainNeutrinoExtended.cpp)。

|测试|结果|
|---|---|
|旧 CC/初级粒子测试|163 个断言、5 项测试通过|
|新增 CC/NC/极化/再生测试|420274 个断言、5 项测试通过|
|通道抽样|ντ/ν̄τ × 10 TeV / 1 PeV / 1 EeV，各 100 万抽样；总 600 万，满足 6σ 二项门限|
|真实 NC 顶点|六味 × p/n/O，18 个；同味出射、能量动量、电荷、DIS 运动学通过|
|πν 极化响应|两种电荷 × P=−1/0/+1，各 1 万次；共 6 万，解析一阶/二阶矩通过|
|全部 τ 衰变通道|两种电荷 × 三种 P，各 5000 次；共 3 万，电荷/τ 味、四动量及 e/μ 分支比通过|
|再生连接|真实 SecondaryView，正反 ντ 各两轮 CC→τ→ντ→NC；保持能量、history，不重置初能|
|真实山体集成|ντ/ν̄τ × 强制首 CC/NC × CPU/OpenMP/CUDA，共 12 例通过|
|跨后端首顶点|四种控制均逐项一致；不据此声称完整 shower 树一致|
|资源|CUDA 新增显存峰值约 512 MiB，小于 8188 MiB 的 10%；未停止生产任务|

12 例记录了 617105 个轨迹段，介质不匹配为零。最终重建后另外各跑一例 CPU/OpenMP/CUDA，
三个物理 CSV 的 SHA-256 以及 neutrino/tau/diagnostics 均与同后端重建前相同。
最终单元测试再次通过（420270 断言；之前多出的 4 个断言是可选 CSV 文件打开检查）。
最终独立 mountain 二进制：

```text
OpenMP: 6e285a508d6851a2a902cb60aef41ae5724fe1e688a1bc5157ea9522aa547e77
CUDA:   4c2755b56f7a74ecef6e0d63bcf9103293d8394e298fea0f589f7f98c4d58d0c
```

没有替换 production install。`applications/c8_air_shower.cpp` 的 SHA-256 仍为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。

全部开启通道的 τ 衰变测试最大相对能量残差约 `5.95e-9`；
这是 CORSIKA/Pythia 末态转换检查，不是整个 shower 的闭合精度。
分支比参考来自当前 Pythia 的粒子表；不是独立实测分支比拟合。
600 万次抽样验证的是**通道抽样器**，不是 600 万个完整 Pythia 碰撞。
山体条件事例有诊断 cut/thinning 和 1100 ns 窗口，不能用于天然概率或完整能谱声明。
12 例初能均为 10 TeV；多轮再生栈控制从 1e8 GeV 开始，未人为增加碰撞概率。

`figures/` 内保存 PNG/PDF：

1. `01_CC_NC_competition`：σCC/σNC 与抽样频率误差。
2. `02_polarized_tau_pion_and_neutrino`：π 解析角分布及再生 ν 能量分布，含抽样阴影。
3. `03_regeneration_and_inclusive_tau_decays`：两轮**条件链**及 e/μ 衰变分支比。
4. `04_actual_terrain_neutrino_tau_tracks`：实际 DEM 内外 ν/τ 单事例能量轨迹。

## 5. 达到用户要求的完整验收还需要什么

不能仅改 `complete_tau_regeneration_included=true`，也不能把所有 τ 固定为 P=−1 就宣布完成。
下一步按以下门禁推进：

1. **统一弱反应模型及能区**：引入有可验证低能区和统一微分截面的 ν-N provider；
   对接所用 PDF、Q²/W cut、τ 质量、核修正以及 CC/NC 总截面积分。
   本轮没有把 CTW 任意外推到 10 TeV 以下。
2. **CC→τ 自旋密度矩阵**：由选中的硬过程/运动学产生纵向和横向分量，
   通过 history 传给 τ；非 CC τ 不能直接套用同一极化。
3. **τ 自旋输运**：验证能损过程中的退极化及飞行/磁场中的自旋约定；
   需要与所采用的 PROPOSAL 能损模型和微分末态一致，不能增加一个未校验的经验随机翻转。
4. **自然链验收**：按真实柱深抽样，比较多轮再生概率、τ 出射率、能谱、角度、
   衰变距离和各女儿味；用独立实现进行 reference comparison，不能只比较三后端共用的一份 CPU 代码。
5. **收敛/系统误差**：改变低能 cut、步长、Q² 范围、thinning；给出模型误差和抽样误差，
   独立区分有限山体几何与全地球传播。

## 参考依据

- [CTW2011，CC/NC 总截面及拟合能区](https://arxiv.org/abs/1102.0691)。
- [Pythia 粒子衰变手册](https://pythia.org/latest-manual/ParticleDecays.html)：
  强制极化只作用于衰变，并不自动改变或验证产生角分布；实现还逐行核对了本地锁定的 8.315 源码。
- [Pythia 电弱过程](https://pythia.org/latest-manual/ElectroweakProcesses.html)：CC/NC t-channel 开关。
- [τ 极化与相关衰变，PRD108 093004](https://arxiv.org/abs/2303.08104)：πν 角分布及产生—衰变关联。
- [τ 能损退极化，PRD106 043008](https://arxiv.org/abs/2205.05629)：不能把生产极化当作穿介质后恒定不变。
- [NuTauSim 传播研究](https://arxiv.org/abs/1707.00334)、
  [DANTON 作者代码](https://github.com/niess/danton)：后续独立自然再生对照候选，本轮尚未完成它们的数值对照。
