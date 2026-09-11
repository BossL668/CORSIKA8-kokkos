# 参考原版补齐山体中微子／τ 接线

2026-09-09。仅修改独立山体模块和 `c8_terrain_cascade`；不修改大气应用、
共享 Kokkos 输运内核、原生 TAUOLA/Pythia/PROPOSAL 源码或生产安装目录。

## 1. 原版究竟实现了什么

本次逐行检查了用户保留的原版 `corsika-21cma/corsika`，不是只看之前的比较图。

|原版源码|实际行为|山体中的处理|
|---|---|---|
|`corsika/modules/pythia8/NeutrinoInteraction.hpp`|明确注释截面是为 `forceInteraction()` 提供的任意 `4 nb`|不能用它计算穿山自由程；保留 CTW CC+NC 物理截面|
|`corsika/detail/modules/pythia8/NeutrinoInteraction.inl`|已经支持 W 交换 CC、γ/Z 交换 NC，`Q²min=25 GeV²`，选择 p/n、碎裂和复制全部末态|此前补齐的指定电流生成器复用这些设置；NC 女儿中微子回栈|
|`applications/c8_air_shower.cpp`|τ 使用 `tauola::Decay(LeftHanded)`，其他衰变用 Pythia；默认不跟踪中微子|山体现在也接 TAUOLA/Pythia 分流，但保留中微子继续输运|
|`corsika/detail/modules/tauola/Decay.inl`|寿命 `gamma * tau0`；TAUOLA 全通道衰变；固定手性参数|直接复用；不把 τ 强制立即衰变，不只实现 πν 通道|
|`corsika/modules/tauola/Decay.hpp`|明确说明 CORSIKA 不跟踪极化，接口只能指定固定 helicity|原版也没有事件级 CC 自旋密度矩阵，不能把固定左手性当作完整极化输运|
|`ParticleCut`、PROPOSAL、ScalarCascadeStepper|cut、带电轻子能损、反应/衰变/几何距离竞争|沿用现有模块；τ/ν 保留在 CPU，EM 仍由同一个 Kokkos 后端接收|

原 mountain 程序的 `NaturalNeutrino.hpp` 也没有补齐低能模型：其物理截面同样
只覆盖 CTW 的 `10⁴–10¹² GeV`，并只调用 CC 末态。因此“参考原版”可以补齐已有模块的接线，
**不能凭此宣布全能区、全部弱过程已经实现**。

## 2. 本轮代码

```text
ν / ν̄ ── CTW(CC+NC)自由程 ── Pythia 指定弱电流末态
  ├─ NC → 同味中微子及其余女儿 → 原主栈
  └─ CC → e/μ/τ 及强子 → 原主栈
              │
              τ：原生 PROPOSAL 能损、飞行／边界／衰变距离竞争
              └─ 原生 TAUOLA 全通道衰变 → ντ / ν̄τ 等全部女儿回栈

γ/e± → Kokkos OpenMP 或 CUDA；其余种类 → CPU 原生过程
```

- [`TransportedLeptonDecay.hpp`](../corsika/modules/neutrino/TransportedLeptonDecay.hpp)：
  实现原版 τ/non-τ 分流。TAUOLA 为新 DEM 应用默认值；Pythia 旧路径仍可显式选择。
- [`ConditionedTauolaDecay.hpp`](../corsika/modules/neutrino/ConditionedTauolaDecay.hpp)：
  仅负责随机数接线及高能数值变换，物理衰变仍调用原生 TAUOLA。
- [`TerrainTauDecay.hpp`](../applications/detail/mountain/TerrainTauDecay.hpp)：
  保留寿命、轨迹端点、所有女儿 history 和守恒审计；写明模型与 RNG 来源。
- [`NeutrinoModelDomain.hpp`](../corsika/modules/neutrino/NeutrinoModelDomain.hpp)：
  统一 CTW 能区门禁，按 history 去重记录未覆盖的次级；可严格拒绝继续输运。
- [`c8_terrain_cascade.cpp`](../applications/c8_terrain_cascade.cpp)：
  只增加选项、专用随机流注册和模块装配，没有把底层物理塞进应用。

## 3. 参考原版时实际发现并修复的问题

### 3.1 TAUOLA 有两套随机数，不能只保存 CORSIKA RNG

本地旧原版的 TAUOLA `.inl` 有 CORSIKA RNG 回调，而 beta5 所带的对应接口没有这段连接。
TAUOLA 1.1.8 本身又分别使用：

- C++ 自旋接受／拒绝：默认 `rand()`；
- Fortran 衰变核：独立的 RANMAR，默认初始化种子 `54217137`。

第一轮测试保存并恢复 CORSIKA RNG 后，同一个 τ 一次产生 2 个女儿、另一次产生 5 个，
因此之前“同 seed”等价的检验方法对这个接口并不充分。

现在山体适配器将 C++ 回调连接到专用 `tauola` 随机流，并从此流抽取一个合法 RANMAR seed，
**只初始化一次 Fortran 流，不在每个 τ、每个 CC 顶点或再生循环重新设种子**。
该 stream 追加在已有 stream 之后，不改变已有 stream 编号。
`c8_terrain_cascade` 当前每进程仅支持一个 shower；未来若扩展 `-N`，需显式管理 RANMAR 的事件生命周期。
测试回放另外使用库提供的 RMARUT/RMARIN 状态入口恢复 Fortran 序列，并检查两套流的消耗。
生产元数据记录 RANMAR seed；不能声称所有 Fortran draws 都来自 Philox。

### 3.2 原生 TAUOLA 的实验室系高能 boost 会消去质量

实际复现了 `Eν=10⁹ GeV` 再生控制中出现无限大的女儿能量。
TAUOLA 1.1.8 的 `TauolaParticle::boostAlongZ()` 使用：

```cpp
double m = sqrt(boost_e*boost_e - boost_pz*boost_pz);
setPz((boost_e*p + boost_pz*e)/m);
```

极高能时两个平方在 double 中相等，计算出的质量为零。
这不是 Kokkos/CUDA 内核的问题，也不能通过提高 shower 数量解决。

山体适配器在 `Eτ >= 1000 GeV` 时让**同一个原生 TAUOLA**在良态参考动量 `p=mτ` 下生成，
然后通过 CORSIKA 已有的、显式使用质量的 `COMBoost(momentum,mass)` 转回实验室系。
不重抽末态，不把非有限值截到零，不改寿命或 τ 能量。
低于该门限仍使用原生直接调用，以便逐项检查接线等价性。
质量约定沿用 TAUOLA 原接口（包括其内部 τ 质量），末态使用 CORSIKA 质量上壳。
完整末态守恒验收暂用原生模块级相对 `10⁻³` 门限，不把它写成整个 shower 的 `10⁻⁴` 能量闭合认证。

πν 解析检验覆盖两种电荷、左手／无极化／右手、z 轴及倾斜轴、
10 GeV 原调用和 1000 GeV 良态变换，检查了方向约定和变换切换后分布连续性。

## 4. 最小使用差别

```bash
# 使用已有 scene 和原命令；现在默认：CC+NC、TAUOLA 固定左手约定。
c8_terrain_cascade --scene scene.yaml --output result \
  --primary nu_tau --energy-GeV 1e5 \
  --position-m X Y Z --direction DX DY DZ

# 重现之前的 Pythia 衰变控制：
# 原命令后加 --tau-decay-model pythia --tau-minus-polarization -1

# 严格禁止低于已实现弱反应模型能区的次级继续传播：
# 原命令后加 --require-neutrino-model-coverage
```

`X/Y/Z` 等需要替换为实际 ENU 数值。只有选择 `--em-backend kokkos` 才开启加速。
`--tauola-helicity left|unpolarized|right` 是原生 TAUOLA 参数约定，
与仅用于 Pythia 的 `--tau-minus-polarization` 不得混用。
这些参数只用于独立 DEM 应用，不影响原大气应用。

## 5. 验收与尚未补齐的物理

结果保存在 `CorsikaData/corsika_validation_results/beta5_original_neutrino_alignment_20260909_v1`。
`VALIDATION_REPORT_CN.md` 和 `acceptance.json` 由脚本根据实际运行结果生成。

- 原版截面用途检查、540 组低能 τ 的逐女儿/两套 RNG 对照、非 τ/旧 Pythia 路径对照。
- 1800 个最高 `10¹² GeV` 的 τ，有限值、τ 味、电荷、四动量检查。
- 72000 个两体 πν 控制，解析角分布及两种变换路径。
- 正反中微子各三轮实际 SecondaryView CC→TAUOLA→NC 再生，无能量重置；**仍为条件链，不是自然事件率**。
- 真实 DEM 中 ν/ν̄ × 首 CC/NC × CPU/OpenMP/CUDA；空气 IGRF14/2027、岩石零场，不含射电。
- 原 Pythia 固定种子 CSV 回归、启动参数拒绝、真实低能次级触发严格能区拒绝。

测试实现：[`testMountainOriginalLeptons.cpp`](../tests/modules/testMountainOriginalLeptons.cpp)、
[`analyze_original_neutrino_alignment.py`](../validation/terrain/analyze_original_neutrino_alignment.py)。

实际完成结果：新增常规测试 `39270` 个断言通过，πν 独立控制 `216050` 个断言通过；
旧 CC 测试 `163` 个、旧 CC/NC/极化测试 `420270` 个断言重新通过。
12/12 DEM 事例完成，材料错配为零，四种首弱反应顶点分别跨三后端一致。
旧 Pythia 路径三后端各一例的 tracks/deposits/window-survivors 共 9 个 CSV 与修改前 SHA-256 完全相同。
9 个启动拒绝测试、真实再生次级触发的运行时能区拒绝均通过。
πν 控制所有分组的均值最大偏离约 `1.56 sigma`；DEM τ 衰变的最大相对能量残差约 `1.40e-5`。
CUDA 测试新增显存的全设备差分峰值约 `581 MiB`，低于本机显存的 10%；
该差分同时受后台生产负载波动影响，不是精确的进程级显存归因。

独立构建二进制 SHA-256：

```text
mountain-openmp: 7ad22ab7d97b875884b3c5eb2f5f4739bd590e9f832cc5515f47b285a29b32bb
mountain-cuda:   9c5151e30e2877e2948e207da49242968aa87dc51b54587af9de69a447f6c8f6
```

这些单事例中后续 τ 飞行/衰变时刻不要求跨后端完全一致：EM 调度会改变之后消耗的 CPU 随机流。
本轮的严格逐项等价声明仅适用于明确固定了两套随机状态的原生衰变接口测试和旧 Pythia 回归，
不是对单个完整 shower 的同树声明。TAUOLA 为进程全局状态，仅在主线程调用；没有放入 OpenMP 粒子内核。

**目前仍不是完整中微子物理认证：**

1. 低于 10 TeV 的弱相互作用没有物理 provider。默认诊断模式只记录这些未覆盖历史并透传；
   严格模式中止。计数/携带能量不是遗漏反应概率，也不能当成能量沉积。
2. CTW 总截面与 Pythia DIS 末态的 PDF、低 Q² 定义尚未做统一微分归一化验收。
3. TAUOLA 固定手性虽然对齐原版，但不提供事件级 CC 自旋密度矩阵，也不包含介质退极化。
4. 自然多轮再生概率、出山能谱仍需独立参考代码／模型；本轮不涉及 ν-e/Glashow、完整低能 QE/RES 等新模型。

以上缺失不能用原版的任意 `4 nb`、CTW 外推或统一固定极化“补上”。
输出继续保持 `full_neutrino_physics_validated: false`。

物理接口参考：[CORSIKA TAUOLA 接口说明](https://corsika-8.readthedocs.io/en/documentation-upgrade/doxygen/html/classcorsika_1_1tauola_1_1Decay.html)、
[TAUOLA 官方接口手册](https://tauolapp.web.cern.ch/resources/TAUOLA.1.1.6/Tauola_interface_design.1.1.6.pdf)、
[CTW2011](https://arxiv.org/abs/1102.0691)。代码事实以本地锁定源码为准，不把在线最新版当作本机依赖。
