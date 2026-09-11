# 山体弱反应/τ 极化接口补齐与射电建设判断

日期：2026-09-09。对象：beta5 独立 `c8_terrain_cascade`；本轮不修改空气应用、GPU kernel、CTW 截面或 PROPOSAL 能损公式。

## 结论先行

**已补齐可直接、安全复用的 TAUOLA 纵向极化衰变接口，并完成测试；三个待补物理并没有因此全部解决。**
可以开始构建“给定粒子轨迹/给定条件 shower 的山体射电计算”；尚不能把它当作完整中微子探测率模拟。

| 问题 | 本轮实际完成 | 仍缺少的部分 |
|---|---|---|
| 低于 10 TeV 弱输运 | 保留逐粒子能区审计/严格模式；新增完整弱物理启动门禁 | 与末态一致的低能 CC/NC 率和适用的 DIS/QE/RES、核效应；没有外推 CTW 或使用原接口固定 4 nb |
| 事件级 CC 极化 | 可对单个 τ 在衰变时提供连续纵向极化；TAUOLA 正负 τ、角分布和参考系测试通过 | 从每次 CC 产生过程计算极化密度矩阵，以及产生端→输运→衰变端传递；没有用任意常量冒充这个计算 |
| 介质退极化 | 明确元数据和请求门禁；衰变端可消费未来模型给出的纵向极化 | 与每次相互作用对应的自旋转移、介质退极化及可能的自旋进动；PROPOSAL 能损/方向本身不是该模型 |

“没有现成接口链”不代表物理无法建模；意味着现有 C8/PYTHIA/PROPOSAL 调用不足以自动给出结果，仍需引入/实现独立模型并验证。**本轮没有声称实现完整 τ 再生。**

## 1. 已接通的代码与行为

新增 `corsika/modules/neutrino/LongitudinalTauolaDecay.hpp`，通过原有
`ConditionedTauolaDecay` 和原版 CORSIKA TAUOLA 完成衰变，保持全部衰变道和高 γ 数值处理。
未复制另一套 τ 衰变公式，未修改 TAUOLA/Conan 缓存。

纯纵向自旋的密度矩阵是两种螺旋度态的概率混合，权重为 `(1±P)/2`。
所以给定 `P` 的衰变分布可由原有左右螺旋度衰变实现；这对纯纵向密度矩阵是精确的概率表示，
不是“用极化修正能量”。`P=-1,0,+1` 直接走原有对应路径，不增加极化抽样；其余值只在
CORSIKA `tauola` 随机流增加一次混合选择，不重设种子、不重试衰变。

注意两套约定：

- 应用参数 `--tau-minus-polarization P` 指 τ⁻ 的物理纵向极化；τ⁺ 使用 `-P`。
- 逐粒子接口传入的是**该粒子本身**在衰变时的极化；τ⁺ 调用者不要再擅自翻转一次。
- 参考系是 τ 静止系，纵向轴由其当前实验室动量方向定义；不是全局 ENU 的 z 轴。
- 目前只支持纵向分量；没有构建通用三维自旋输运或 CC 自旋相关性。

实际接口：

```cpp
neutrino::TransportedLeptonDecay decay;  // TAUOLA default
// 示例：由调用方在此刻给出实际 τ 的纵向极化。
// 此函数不负责从 CC 运动学推导 P，也不负责沿轨迹演化 P。
decay.doDecayWithLongitudinalPolarization(view, physicalPolarizationAtDecay);
```

该调用仍是 host 串行衰变接口；不能从 OpenMP/GPU kernel 并发调用 TAUOLA 的全局状态。
Kokkos 电磁并行与此接口的使用方式不变。

`TransportedLeptonDecayConfig::prescribedTauolaPolarization` 提供统一衰变控制，
`TerrainTauDecay.hpp` 记录实际应用的极化、TAUOLA 电荷共轭约定、选中的螺旋度和是否多抽了一次随机数。
`spin_density_matrix_transferred_from_CC`、`energy_loss_depolarization_included` 继续明确为 `false`。

原有 `--tau-minus-polarization` 现在也可用于默认 TAUOLA：

```bash
# 在现有 c8_terrain_cascade 场景命令后追加：
--tau-decay-model tauola --tau-minus-polarization -0.5
```

不能与显式 `--tauola-helicity` 同时使用。未给这个参数时，仍保持原来的固定螺旋度和随机流。
这只是衰变敏感性/对照试验，不是建议把 -0.5 当成 CC 的默认物理预测。

## 2. 不允许静默补洞

`NeutrinoPhysicsCapabilities.hpp` 增加三个启动要求：

| 参数 | 当前行为 |
|---|---|
| `--require-complete-weak-transport` | 启动失败：没有完整低能及核弱输运模型 |
| `--require-event-cc-polarization` | 启动失败：规定的衰变极化不能冒充 CC 产生极化 |
| `--require-tau-depolarization` | 启动失败：没有介质自旋转移模型 |
| 已有 `--require-neutrino-model-coverage` | 运行中一旦需要追踪 CTW 能区外中微子则停止，不把它的能量算成沉积 |

前三项在创建输出和初始化昂贵物理之前拒绝。这些门禁是安全措施，**不是对应物理功能的实现**。
允许有限范围展示的默认模式仍显式统计能区外中微子，不删除粒子或忽略这些限制。

官方原版 `corsika/modules/pythia8/NeutrinoInteraction.hpp` 的注释和 `4_nb` 常量说明其用于强制反应。
TAUOLA 有极化输入接口，但这并不产生 νN CC 的密度矩阵。参考
[TAUOLA 官方接口说明](https://tauolapp.web.cern.ch/tauolapp/resources/TAUOLA.1.0.5/Tauola_interface_design.1.0.5.pdf)。
介质退极化还涉及光核散射等过程的自旋信息，不能直接由能损比例替代；参见
[Argüelles 等，PRD 106, 043008](https://arxiv.org/abs/2205.05629)。

## 3. 本轮验收

所有模拟/设备测试使用 OpenMP-only 构建和最多 4 个 OpenMP 线程；没有启动 CUDA。
生产 GPU 服务未被停止或替换。单个诊断使用 RSS 2 GiB、系统可用内存至少 3 GiB 的监控门禁。
本轮编译目录：`../build/mountain-openmp`；没有将实验版本安装覆盖生产安装目录。

| 测试 | 结果 |
|---|---|
| 极化输入/能力门禁及原 TAUOLA 端点回归 | 6,800 项断言通过；原末态和 C++/Fortran RNG 状态一致 |
| τ±→π±ν 解析角分布 | 60 个配置，每配置 3,000 次，共 180,000 次；1,080,170 项断言通过 |
| 角分布能量和方向矩阵 | 10、1,000、10⁸ GeV；两种方向；P = -1,-0.5,0,0.5,1；最大均值偏差 3.174 SEM，小于预设 6 SEM |
| 原模块/寿命/高 γ/有能区覆盖的 CC→τ→NC 再生回归 | 39,269 项断言通过 |
| CC/NC 与既有极化/能区模块回归 | 420,270 项断言通过 |
| 真实 DEM、默认 10 TeV CC 事例 | 完整结束；tracks/deposits/window_survivors 三个 CSV 与修改前逐字节一致 |
| 真实 DEM、10 TeV τ⁻/τ⁺，P⁻=-0.5 | 两例完整结束；衰变记录分别应用 -0.5/+0.5；材料错误为零 |
| 内部均匀岩石射电基础门禁 | 14 项断言通过 |
| Kokkos OpenMP 均匀传播测试 | nR/c、原空气默认路径回归、ZHS 非零有限输出及初始化门禁通过；不是跨真实 DEM 射电验收 |

角分布用两体衰变能量分数重建静止系余弦，避免在极高 γ 下直接相减大四动量。
πν 单道仅在独立测试进程里开启，实际应用保持全道衰变。此测试验证**规定极化如何进入衰变**，不验证该极化值的产生机制。

![规定纵向极化的 τ± 角分布检验](/mnt/d/CorsikaData/corsika_validation_results/beta5_terrain_weak_spin_interfaces_20260909_v1/figures/tau_longitudinal_angular_oracle.png)

原始数据与汇总：
`D:\CorsikaData\corsika_validation_results\beta5_terrain_weak_spin_interfaces_20260909_v1`。
其中 `summary.json`、`angular_matrix.csv`、`application/acceptance.json` 和 `*_guard/` 保存通过条件、日志和资源监控。
应用三例耗时约 85.3、89.8、75.8 s；这不是性能加速比测试。

验证脚本：`validation/terrain/{run,analyze}_weak_spin_interface*.py`；
新增单测：`tests/modules/testMountainTauSpin.cpp`。重新运行必须使用新的输出目录。

## 4. 现在能开始做山体射电吗？

**可以开始，定位应是“给定 shower 的条件射电信号”，不是最终中微子事件率。**
射电计算可先用已知带电轨迹/电子光子 shower 验证，不必等完整低能中微子模型才建设传播模块。
但 τ 衰变能谱随极化变化，最终 ντ 射电预测仍需保留这项源项不确定性。

建议按以下顺序封装，维持独立应用和空气路径不变：

1. **固定轨迹、均匀介质：**分别在空气/岩石中核对 CoREAS 和 ZHS，包括正负电荷、时间窗、Cherenkov 小分母、单位、权重和参考坐标系。
2. **单平面岩气界面：**返回光程、发射/接收方向及偏振传递；对照 Snell、Fresnel、全反射和吸收的解析情况。
3. **真实 DEM：**复用通用地形/BVH，验证路径遮挡、有限三角面、共享边不重复计数；天线坐标沿用已验证的 ENU/地理转换。
4. **接入标准大气与相干求和：**分开岩石源、空气源、界面项，再按相同时间基准相干叠加；不能把两个波形的绝对值相加，也不能把跨界轨迹的人工端点当成完整过渡辐射。
5. **Kokkos 批处理：**在固定轨迹下对 OpenMP 与 GPU 比较波形，先验证传播，再测完整 shower；最后才比较源项模型系统误差。

可参考原 mountain 的 `cpp/TerrainRadio.hpp` 与 `cpp/TerrainAtmosphereRadio.hpp`。
后者明确只做直线空气段折射率积分和局部界面处理，不含完整弯曲空气射线聚焦、反射或过渡辐射，
不能原样搬入后就宣称所有传播效应齐全。

beta5 现有 `applications/detail/mountain/MountainRadio.hpp` 只允许均匀凸岩体内部路径，
真实 DEM 应用仍关闭射电；本轮没有打开这个尚未验收的功能。

生产使用之前还必须确定频率相关介电参数/衰减长度、模型几何光学适用范围、事件权重及弱反应覆盖。
当前 `rho=2.65 g/cm^3, n=2.0` 是测试参数，不是当地岩石实测校准。
