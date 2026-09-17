# beta5 光子 CPU 回退漏传播：影响范围与修复验收

日期：2026-09-17。对象为 beta5 的共享 Kokkos 光子管线；CUDA、OpenMP 及双端模式均调用这一实现。

## 1. 结论

这是实际输运错误，不是画图范围、Xmax 寻峰或 profile 平滑问题。光子抽取到需要 CPU 完成的过程（包括光核反应、光子产生 μ 对）后，旧代码在**传播之前**发出 `NativeSelectionReplay`。CPU 仅补全损失/末态，把出发位置误当作反应顶点。刚跨过大气层界的光子因此可能在层界立刻产生强子或 μ 子。

修复顺序为：**抽样候选反应 → 与层界/观测面/cut 竞争 → 传播 → 仅在真实反应顶点请求 CPU 完成末态**。不修改截面、PROPOSAL 表、强子模型、cut、thinning、物理公式或绘图数据。

本报告区分：控制流程修复、真实设备回归、完整 shower 快测、大样本统计验收。前几项通过不代表新的 100 vs 100 或 500 vs 500 已经通过。

## 2. 已确认的影响范围

在 PSR 保存的同种子 CPU/CUDA/OpenMP 各 100 个 100 PeV 质子事件中，theta=47°、phi=180°、emthin=1e-6，直接检查原始 profile 数组，未平滑或重新分箱。

USStdBK 的 7 km 层界沿此入射轴约为 612.5 g/cm²，下一输出格点为 620 g/cm²；11.4 km 层界约为 317.3 g/cm²，对应 320 格点。这里接近强子曲线最大值只是巧合。

定义局部折点指标 `D(X)=N(X)-[N(X-10)+N(X+10)]/2`。下表为 `mean(D)/mean(N)`，**不是整体 profile 的相对误差**：

| 620 g/cm² 处组分 | CPU | Kokkos-CUDA | Kokkos-OpenMP |
|---|---:|---:|---:|
| hadron | 0.063% | 2.471% | 2.438% |
| muplus | 0.038% | 0.974% | 0.957% |
| muminus | 0.045% | 0.851% | 0.844% |
| photon | 0.100% | 0.088% | 0.091% |
| electron | 0.095% | 0.138% | 0.137% |
| positron | 0.096% | 0.153% | 0.146% |
| total EM | 0.099% | 0.096% | 0.098% |
| charged | 0.095% | 0.158% | 0.155% |

两个加速组的 μ⁺、μ⁻ 在此格点均为 **100/100 个事件 D>0**。相对 CPU 的配对局部折点增量，按 CPU 均值归一化：CUDA μ⁺ 为 0.963%（shower-bootstrap 95% CI：0.931%–0.996%），μ⁻ 为 0.828%（0.799%–0.856%）；OpenMP 分别为 0.942%（0.910%–0.973%）和 0.818%（0.793%–0.843%）。这些是定位问题的事后局部检验，不能替代预先定义的整体等价检验。

影响链包括：

- 光核顶点提前 → 强子生成深度错误 → π/K 衰变得到的 μ⁺、μ⁻ 随之改变。
- 光子直接产生 μ 对也经过 CPU-only 过程，受相同漏传播影响。
- 被消费光子的飞行段可能没有记入 photon profile；后续 e± 的产生位置、时间、能量沉积与射电轨迹可能改变。
- 原本先到观测面或时间 cut 的粒子可能被提前消耗，故地面输出与能量去向也须复验。
- 轻子本身的选择顺序原本为距离抽样、连续传播、顶点过程选择，没有发现同一处“反应前漏传播”；本次给其正常反应/衰变顶点补上显式标记，以满足统一入口检查。

不能说“只有强子和 μ 子受影响”，也不能据此把整条曲线的所有差别都归因于这一项。标量 CPU 不经过这条加速分支，已有 CPU 参考可保留。

## 3. 代码改动与安全条件

| 文件（相对源码目录） | 职责 |
|---|---|
| `corsika/accelerator/em/detail/InteractionSelection.hpp` | 把可在 CPU 完成的 selected-loss 请求记录为 pending，不跳过光子输运 |
| `corsika/accelerator/em/detail/PhotonTransportStep.hpp` | 只有真实反应获胜才设置 `interaction_vertex_reached`；层界、观测面、逃逸和 cut 不设置 |
| `corsika/accelerator/em/detail/PhotonFinalStateStep.hpp` | 在已传播的状态上兑现 pending 请求，保留过程、组分、history 与随机数身份 |
| `corsika/accelerator/em/common/ProposalFallback*.hpp` | 统一识别 selected-loss 类别；CPU 指定末态入口拒绝步前状态 |
| `corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp` | 缺少真实顶点标记的指定过程回退明确报错，不能悄悄退回普通 CPU 重采样 |
| `corsika/accelerator/em/detail/LeptonTransportStep.hpp`、`LeptonVertexSelection.hpp` | 给已有的真实轻子顶点及其回退传播标记，不改连续损失或末态算法 |
| `corsika/modules/transport/detail/InterfaceEmStep.hpp` | 移除山体仅对 NativeSelectionReplay 的局部补丁，改用同一共享规则，覆盖其他 selected-loss 失败 |

覆盖 `NativeSelectionReplay`、`InverseCdfUnavailable`、`LossEnergyOutOfRange`、`LossQuantileOutOfRange`。没有可用过程身份/总率的配置或查询错误仍按原有失败策略处理，不能凭空假设一个反应顶点。

在 resident photon 管线中，成功传播后 `source_counts.step=1`，末态 CPU fallback 只增加 fallback 记录，不额外增加一次 step；其轨迹进入原 profile 路径。到层界时只排队延续，不能执行旧的待完成过程。队列重新抽样延续现有 RNG 语义。

这是物理结果会变化的修复：旧 seed 并不意味着修复前后的 shower tree 相同。新增的是内部 POD 字段，需重新编译相应后端/应用；无需重新生成 PROPOSAL 缓存或 `.c8emaux`。

历史 beta4 `src/gpu/em/CudaInteractionSelector.cu` 中亦存在同形的 native-selection 步前回退分支（静态审计）。本轮修改和设备验收对象为 beta5，**没有宣称历史 beta4 二进制已修复**；使用旧分支需单独回移补丁并验收。

## 4. 验收记录

独立目录：`../build/photon-vertex-fix-20260917`。没有覆盖生产 build/install 或历史 campaign 二进制。

- `testProposalFallbackVertexGate`：20 个 PID/selected-loss 组合及 6 个 CPU-only/Epair 末态组合的步前拒绝、真实顶点接受，通过。
- `testPhotonFallbackVertex`：CUDA 57,344 个合成场景，通过。
- `testPhotonFallbackVertexOpenMP`：OpenMP 57,344 个合成场景，通过。
- 场景包括光核、μ 对、pair/Compton 选择、逆 CDF 失败、层界/观测面/材料面先到、时间和能量 cut，并检查原 process/component/history/u 保持不变。
- 合成表仅隔离控制流程；不是截面精度或真实 PROPOSAL oracle 的替代。
- `testKokkosProposalNativeTable`：默认 16,384 点真实 PROPOSAL 原生表/共享 host evaluator 对照，以及阶段输运、末态、驻留复用、profile/射电回归，通过（232.91 s）。其中共享 host evaluator 不是独立的完整 CPU shower oracle；本次增加的独立解析飞行距离测试补足旧测试漏掉的时序条件。
- `testKokkosCpuTransportAlignment`：真实标量连续输运对照，通过；含 e±、μ±，72 个 scalar-step、12 个 terminal、4 个 observation/cut 重合、30 个 secondary-adapter 及 56 个磁偏转门禁案例。
- `testTerrainCurvedBoundary`：2,000,038 次检查通过，最大距离差 `1.11e-16 m`，host/device 精确不一致计数为 0；`testKokkosInterfaceQueue`：有界 FIFO/溢出/200 次轮转通过。接口输运库重新编译通过；这不替代整套中微子山体 shower 验收。
- 修复后的大样本统计：未运行。旧数据的局部输运门禁失败结论不撤销。

### 完整应用快测

数据：`/mnt/d/CorsikaData/corsika_validation_results/diagnostic_beta5_photon_vertex_fix_20260917_v1/`。

共 **14 次独立进程、16 个 shower**，全部正常结束，profile/地面粒子/能量沉积/CoREAS/ZHS Parquet 可读、数值有限，profile 非负、深度轴单调，shower 编号齐全，待完成 fallback 队列全部提交。命令、二进制 SHA-256、耗时、RSS、完整性检查保存在 `*_runs.json` 与 `OUTPUT_REGRESSION.json`。

| 快测 | 结果 |
|---|---|
| 1 TeV 质子，CUDA `N=2` / OpenMP 4 线程 `N=2` | 完成；同进程缓存/输出生命周期正常 |
| 10 GeV 光子、μ⁺、μ⁻，覆盖 CUDA/OpenMP | 完成 |
| 1 TeV 质子，CUDA+OpenMP 4 线程 | 完成；指定 CPU 末态 41 次，其中光核 4 次 |
| 1 TeV 质子，标量 PROPOSAL | 完成；作为应用 smoke，不作小样本等价声明 |
| 100 TeV 质子，CUDA，3 个独立 seed | 54.699、58.716、50.170 s；光核指定完成分别为 850、843、774 次 |
| 1 GeV 无薄化光子，CUDA/OpenMP，默认 batch | 完成，但小波前包含 scalar EM 步，设备账本不是完整闭合测试 |
| 1 GeV 无薄化光子，CUDA/OpenMP，诊断用 `gpu-min-batch=1` | 两端账本完整覆盖且通过；相对闭合误差分别 `6.21e-16`、`2.07e-16` |

快测方向 theta=47°、phi=180°，使用 3 个示例天线、1 GHz、2000 ns 窗口、显存预算 40%，并保留至少 4 GiB 系统可用内存。实际子进程树峰值 RSS 不超过 0.84 GiB。100 TeV 用时仅说明此诊断配置可正常运行，不能和不同天线配置的历史生产时间直接计算加速比。

第三个 100 TeV seed 有 **3 次普通 `unsupported_geometry` 回退**，并非零回退：低能 e± 起点位于 7 km 球面以上约 60/82 μm，`process=component=0`。这些状态经通用标量步重新传播，没有调用指定反应末态；几何回退本来就在显式许可名单内。其余指定过程回退均通过新的顶点门禁。不要把正常的几何 CPU 输运与本次修复的“在步前执行已选末态”混为一谈。

另外发现已有双端 metadata 的命名局限：`cpu_generic_fallbacks` 实际来自节点刷新的返回计数，会把立即完成的指定末态也算入。本次双端案例该字段为 41，但 `cpu_fallback_steps_executed=0`，41 次全部是 `native_selection_replay`，不存在普通标量回退。验收脚本同时核对 specified/flushed/reason/执行步数，显式分开这两类；本轮没有顺带改双端调度或历史 metadata schema。

**能量账本限制：**含强子、scalar EM 或指定 CPU 末态的应用事件中，现有 `gpu_em.energy_ledger.complete_coverage=false`。它只覆盖设备路由的能量项，不能把其 residual 当作全 shower 丢失能量，也不能宣布全 shower 能量守恒已验收。上表最后一项刻意关闭薄化并把 batch 设为 1，消除小前沿 scalar 处理，才满足现有严格闭合判据。

修复后的空气应用：`../build/photon-vertex-fix-20260917/applications/c8_air_shower`；SHA-256：`e6ace6b0a2bd3e9b7d9980fa5058b9571f7742818393df66fe63bfc1f5c26066`。原生产 build/install、远端二进制和归档程序尚未替换；只有新测试程序包含本次修复。

## 5. 诊断数据位置与后续使用

本地图片/JSON：`/mnt/d/CorsikaData/corsika_validation_results/waveform_prime_reanalysis_20260909/proton_100PeV_final_100paired_20260917/hadron_layer_diagnosis_20260917/`。

新增 `all_components_layer_320.png`、`all_components_layer_620.png`、`all_components_local_curvature.json`。脚本为工作区 `analysis_jobs/diagnose_all_layer_profiles_20260917.py`；在 PSR 的对应 final 目录读取紧凑原始数组，结果同步回本地，没有搬回大型事件数据。

旧加速样本必须保留版本标签，不能删点、平滑、改 bin 或后处理“校正”为修复后的物理事件。后续需新跑代表性 CUDA/OpenMP 样本，检查层界局部曲率，同时复验全组分、沉积、地面分布及 Ex′/Ey′/Ez′ 射电波形。现有 CPU 样本无需因这项修复而重跑。
