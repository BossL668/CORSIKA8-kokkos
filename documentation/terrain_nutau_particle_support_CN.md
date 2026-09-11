# 真实 DEM 山体的 ντ / τ 粒子支持与诊断

日期：2026-09-09。本轮对象是 **`c8_terrain_cascade`**，不是有限凸体的
`c8_mountain_neutrino`，也不修改 `c8_air_shower`。射电保持关闭。

本文件保留第一轮 CC-only 历史控制结果；后续 NC、再生链和指定极化扩展见
[新的独立验收记录](terrain_neutrino_CC_NC_regeneration_polarization_CN.md)。
复跑本文件的旧模型需显式选择 `--neutrino-channels cc`；不要将新旧模型混为同一次验收。

## 修复了什么

原 `setup::C7trackedParticles` 不含 τ±。将这个集合直接交给 CC 顶点生成器，
会允许 Pythia 在顶点内部衰变 τ，输出的栈中不再有可飞行的 τ。这是粒子保留策略
缺失，不是 GPU 浮点误差。原 mountain 项目已针对这一点采用局部扩展集合；本轮复用该方法。

```cpp
neutrino::MountainNeutrinoInteraction neutrinos(
    neutrino::withTransportedTaus(setup::C7trackedParticles), samplingMode);
```

[TransportedLeptons.hpp](../corsika/modules/neutrino/TransportedLeptons.hpp) 返回集合的副本，
只增加 `TauMinus` 和 `TauPlus`，**不改变全局 C7 集合**。同一顶点种子的测试确认旧集合
输出 τ 数为 0；新接口在 ντ/反ντ 的顶点中保留对应 τ−/τ+。

当前 CLI 的明确粒子名称集中在
[TerrainPrimaries.hpp](../applications/detail/mountain/TerrainPrimaries.hpp)：

- `photon`、`electron`、`positron`；
- `mu_minus`、`mu_plus`、`tau_minus`、`tau_plus`；
- `nu_e`、`anti_nu_e`、`nu_mu`、`anti_nu_mu`、`nu_tau`、`anti_nu_tau`；
- `proton`、`anti_proton`、`neutron`、`anti_neutron`、`pi_plus`、`pi_minus`。

“CLI 可接收”不等于每种粒子都已完成大样本认证。所有六种中微子初级仍受
CTW2011 CC 拟合的 **`1e4–1e12 GeV`** 门禁限制。

## 执行链与代码职责

```text
ντ（CPU，按实际柱深抽取 CC）
 ├─ 无反应：空气 → 岩石 → 空气，保留能量并记录逃逸
 └─ CC：τ + 强子次级
         ├─ τ：CPU PROPOSAL 能损/散射 + 原生寿命/衰变距离竞争
         │     └─ 衰变后的 ν / μ / 强子继续走 CPU
         └─ γ/e−/e+：同一多介质 Kokkos 输运（OpenMP 或 CUDA）
                       └─ 指定的非 EM 末态返回 CPU，不能丢弃
```

τ 可以穿过岩气边界后再衰变；边界不是吸收面。空气使用局地均匀 IGRF14/2027，岩内 B=0。
原 mountain 的零场对照通过 `--magnetic-field none` 保留。

| 文件 | 本轮职责 |
| --- | --- |
| [c8_terrain_cascade.cpp](../applications/c8_terrain_cascade.cpp) | 高层粒子/模型装配、CLI、输出生命周期；原 process 顺序不变 |
| [TerrainTauDecay.hpp](../applications/detail/mountain/TerrainTauDecay.hpp) | 包装原生 Pythia8 `Decay`；只记录 τ 衰变，不额外抽随机数 |
| [TerrainShowerOutput.hpp](../applications/detail/mountain/TerrainShowerOutput.hpp) | 流式 track、沉积、时间窗存活粒子；记录 history/parent ID |
| [TerrainEmSession.cpp](../src/terrain/TerrainEmSession.cpp) | 多介质 EM 的可选步长/时间窗限制；不修改共用空气 kernel |
| [TerrainEmRouter.hpp](../applications/detail/mountain/TerrainEmRouter.hpp) | 独立窗口终止记录；仍只路由 γ/e± |
| [testMountainNeutrino.cpp](../tests/modules/testMountainNeutrino.cpp) | 六种中微子、τ 保留、顶点电荷/四动量、能区/输入门禁测试 |

`TauDecay::getLifetime()` 直接调用原模型；`doDecay()` 在原模型返回后记录 thinning/cut
之前的女儿粒子、四动量残差、τ 位置、介质与 history ID。测试没有调用 `forceDecay()`。
它不重新实现寿命，也不伪造轨迹或强行选择电子/μ 衰变道。

## 原 mountain 对照设置

验证脚本读取原 `terrain_nutau_v1/manifest.yaml` 和五个 YAML，不重新选 seed。
DEM SHA-256 必须为
`5060f55efc6a3dae8ab94c6ef2a27356521b56797ed059a1c150c3a47c2d7c1a`。
这是扩大区域的原始网格（255,746 顶点、511,488 面），没有为了 GPU 降低物理几何分辨率。

| 项目 | 设置 |
| --- | --- |
| 场景 | 原 mountain 的 21CMA 扩大 DEM、80 个站中心、USStdBK 大气 |
| 岩石 | SiO₂，2.65 g/cm³ |
| 入射方向 | ENU `(1/√2, 1/√2, 0)`；从空气走 50 m 入岩 |
| 岩石弦长 | 135.566697 m，进出自然山坡 |
| EM cut | 10 MeV |
| 强子/μ/τ cut | 300 MeV |
| 选中 CC 的 thinning | `1e-3`，最大权重 50；其余控制关闭 thinning |
| 控制步长 | 非中微子至多 1 m；中微子不施加这项步长上限 |
| 物理时间窗 | 1100 ns；不是 CPU 核运算时间 |
| 射电 | 关闭，天线位置仅用于地形图展示 |

五个控制分别是：岩内 100 GeV τ（79100）、10 TeV ντ 的两个未筛选种子（79101/79102）、
10 TeV ντ 的选中 CC 种子（515061）、离出射面内侧 5 mm 的 1 TeV τ（79104）。
种子 515061 沿用原项目根据柱深预扫描选中的种子，实际运行仍自然抽取相互作用，
**没有强制顶点，但样本已经条件化，不能据此估计无偏反应概率或探测效率。**

新增 `--max-weight`、`--max-step-m`、`--transport-window-ns` 用于显式复用上述控制条件。
不指定时保留原应用的最大权重公式、原步长竞争及 stock time cut。有限窗的剩余总能量
单独记为 escape；低于 cut 的粒子留给 ParticleCut，避免同一能量同时记为 cut 沉积和逃逸。

## 复现命令

使用已经配置好的独立山体构建；不要覆盖正在运行的生产二进制：

```bash
conda activate corsika_venv
cmake --build ../build/mountain-openmp \
  --target c8_terrain_cascade testMountainNeutrino --parallel 1
cmake --build ../build/mountain-cuda --target c8_terrain_cascade --parallel 1

python validation/terrain/run_nutau_acceptance.py \
  --mountain /path/to/corsika8-mountain \
  --output /path/to/new/nutau_results \
  --binary ../build/mountain-openmp/applications/c8_terrain_cascade \
  --backend cpu --timeout 900
# OpenMP：换 --backend openmp；CUDA：换 binary 和 --backend cuda。
# 本次扩大 DEM 的 CUDA 诊断显式使用 --gpu-increment-MiB 700。
```

脚本只为测试复用旧项目的场景/控制清单；已生成场景之后，beta5 应用本身不依赖旧二进制。
每个测试使用独立 Fortran 工作目录，已有输出（包括失败）拒绝覆盖。
native PROPOSAL 缓存缺失时由库首次建立，随后复用；本轮没有生成 `.c8emrt`。

分析命令的 `--groups` 必须与结果目录一致，`--cases` 可明确指定磁场补测子集：

```bash
python validation/terrain/analyze_nutau_acceptance.py \
  --output /path/to/nutau_results \
  --groups cpu_none openmp_none cuda_none_guard700
```

## 结果与图的含义

本次数据根目录：

```text
D:\CorsikaData\corsika_validation_results\beta5_nutau_particle_support_20260909_v2
```

正式状态以 `particle_support_acceptance.json` 为准；`field_checks/` 保存独立磁场子集，
不要把未完成目录或带 `stop_reason` 的 guard 记录算作通过。

本轮实测：

| 验证 | 结果 |
| --- | --- |
| OpenMP / CUDA 独立 Release 构建 | 通过；不覆盖生产 install |
| 5 个模块测试，含六种中微子 × 10/100 TeV 实际 Pythia 顶点 | 163 个断言通过；不是百万点采样 |
| 原 mountain 五个零场控制 × CPU/OpenMP/CUDA | 15/15 通过 |
| IGRF14 空气场下的选中 CC、τ 出山衰变 × 三后端 | 6/6 通过 |
| 同后端旧默认 1 GeV 光子控制回归 | 3/3；旧 track 列与旧 diagnostics 逐项相同 |
| CC 前局部对照 | 三后端首次 CC 顶点及输出 τ 四动量相同 |
| 两个直接 τ 控制的首次飞行 | 同条件三后端长度相同 |

21 个 ντ/τ 控制合计 **274,923 步，0 介质错配**；轨迹及沉积流未截断。
15 次 τ 衰变全部与实际轨迹端点衔接，记录中的最大相对四动量残差约 `6.38e-11`。
峰值子进程 RSS 约 **923 MiB**，CUDA 期间设备总显存相对启动基线的最大监测增量为
**513 MiB**（约总显存的 6.3%）；这不是精确的每进程 NVML 显存归属。

零场选中 CC 的入岩深度为 **43.548841 m**，初生 τ 能量 **8242.070620 GeV**。
CPU 中 τ 飞行 **0.369021 m** 后走电子衰变道；OpenMP/CUDA 中飞行 **1.550648 m**
后走 μ 衰变道，后者实际生成的高能 μ 跨过山坡进入空气。这不是要求相同 shower tree
的 replay，不应将两条单事例 profile 的高低差当作统计偏差。
原 mountain 同 seed 的 τ 飞行约 0.320314 m；本轮复刻的是输入场景和物理链控制，
**没有复刻原项目整棵随机树**。

更直接的独立控制是：100 GeV 初级 τ 在三个后端均飞行 **2.498237 mm** 后在岩中衰变；
距出射面内侧 5 mm 的 1 TeV 初级 τ 均飞行 **38.898010 mm** 后在空气中衰变。
这两例没有 CC 后其他次级调度导致的前置随机流差异。

保留的非通过尝试：`v1` 首次冷缓存诊断超时，不算物理验收；`v2/cuda_none/` 首例虽然
计算完成，却在 513 MiB 触发原 512 MiB 诊断门限，不能算资源验收通过。
`cuda_none_guard700/` 使用显式 700 MiB 门限重新运行（另有总量 10% 硬上限），5 例均通过。
原生产服务没有停启，生产显存配置没有改变；这些含初始化、与生产并行的用时不是性能基准。

交付前重新编译了应用，确保后来补入的初始化提示也包含在 CUDA 二进制中。
OpenMP 二进制 SHA-256 不变；最终 CUDA 为
`172ea9299cbcfc5e6d06050cc6eb6906879c1572c8941f5bbf718109900122d0`。
重建后额外复核 OpenMP 零场 CC、CUDA 零场 CC/τ 出山、CUDA 磁场 CC，**4/4 通过**：
`tracks.csv`、`deposits.csv`、`window_survivors.csv` 和 diagnostics/CC/τ 记录均与重建前逐项一致。
回执保留在 `openmp_none_final/`、`cuda_none_final/`、`cuda_igrf14_final/`；
机读比较文件为 `*_comparison.json`。主体图仍来自前述同一轮 21 个控制样本。

旧默认光子回归脚本为 [run_particle_support_regression.py](../validation/terrain/run_particle_support_regression.py)；
新的同组重建对照使用 `run_nutau_acceptance.py --label final --compare-group 原组名`，
不覆盖旧结果。空气源码 SHA-256 仍为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。

- `01_nutau_terrain_3d`：真实 DEM、80 站、入山轨迹和 CC 周围粒子云；只在显示时抽稀。
- `02_nutau_CC_tau_decay_chain`：实际粒子能量、τ 飞行、女儿 PDG/能量及高能 μ 的存在情况。
- `03_shower_development`：EM 加权轨迹长度与沉积的中点分箱；不是精确 `dE/dX`。
- `04_energy_and_resources`：部分能量账、包含初始化的墙钟时间、RSS；不是性能加速比。
- `05_longitudinal_components`：按记录端点重建的各组分加权平面穿越数；不是原生 `N(X)` writer。

所有图均为真实单事例诊断，**不是平均 shower、不是大样本统计一致性证明**。
CPU/加速调度可能改变 τ 前面其他粒子对随机流的消耗，因此不同后端或旧 mountain 的
同 seed 不必具有相同衰变道/飞行长度。必须分别检查自然寿命调度、τ 保留、端点连续、
女儿四动量和介质，而不能仅要求图片中的曲线完全重合。

## 尚未声称解决的物理范围

这是原 mountain **CC-only** 模型的粒子/输运接口补齐，不是完整中微子传播实现。
仍没有 NC、核 shadowing、经验证的 CC→τ 自旋密度矩阵传递、完整 τ 再生链或全球地球传播；
低于 10 TeV 的次级中微子在当前高能 CC 模型中不再反应，继续运输到逃逸。
Pythia 的低 Q² 末态没有完成验证。

沉积 + 窗口逃逸不是完整静质量/靶核/模型/thinning 能量账，不能用二者之差直接宣称能量
不守恒，也不能据此宣称完整能量闭合通过。当前 DEM 应用仍为有界同步验证管线，
有 2,000,000 步等资源门禁；跨介质射电、大样本 ντ 出山效率以及高能性能需后续单独验收。
