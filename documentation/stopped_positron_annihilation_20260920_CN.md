# 低能正电子停止湮灭补全（2026-09-20）

本次是**物理处理变化**，不是给光子 profile 乘修正系数。补全正电子在能量 cut 处停止后的双光子湮灭；已有的在飞湮灭保留。旧 beta5 的缺失来自继承的标量 ParticleCut 处理，不是 PROPOSAL 不会计算在飞湮灭，也不是 Kokkos 独有的问题。

## 1. 处理规则

当 `e+` 的动能确实低于配置的传播阈值时：

1. 剩余动能按原 ParticleCut 近似在停止点沉积。
2. 用未极化、自由电子—正电子静止双光子模型产生两个反向光子，各带一个电子静质量能量（约 0.511 MeV）。方向各向同性。
3. 两个光子各自继承母粒子的完整统计权重，**不再次薄化**。
4. 光子继续按当前 photon cut 输运；若低于该 cut，其能量直接在停止点沉积。
5. 能量账本增加参与湮灭的介质电子静质量输入；母正电子的静质量已经转换成光子，不能同时计作 cut 静质量损失。

独立的 10 ms 时间截断、几何逃逸或观测面终止不触发停止湮灭。能量 cut 与观测面重合时保留已有终止/观测重叠记账规则。此实现不增加正电子素寿命、束缚修正、三光子支路或额外的低于 cut 精细输运，不能宣称已覆盖这些物理。

质量使用 CORSIKA 生成的 `get_mass(Electron)` / `TransportElectronMassGeV`，没有另引入一个硬编码近似值；在飞过程仍沿用其已有 PROPOSAL 质量约定及显式质量约定修正项。

## 2. 源码与调用

| 层 | 文件 | 职责 |
|---|---|---|
| 公共静止运动学 | `corsika/framework/utility/PositronAtRest.hpp` | CPU 与设备共用的各向同性方向及相反方向 |
| 标量 CPU / fallback | `corsika/detail/modules/ParticleCut.inl` | secondary cut 与 continuous cut 两种入口，端点、时间、权重、沉积和介质质量来源 |
| 共享设备输运 | `corsika/accelerator/em/detail/LeptonTransportStep.hpp` | 真正能量 cut 标记为 `AtRestAnnihilation`，仍保留 cut 的输运记录 |
| 共享设备末态 | `LeptonVertexSelection.hpp`、`LeptonFinalStateStep.hpp` | 不再查随机过程率，直接进入停止湮灭；Philox history-keyed 的独立 draw 3/4；通过既有 scan 分配两个 photon history |
| 空气驻留队列 | `kokkos/KokkosResidentLeptonCascade.hpp`、`src/accelerator/em/kokkos/KokkosBackendInstance.inl` | resident 与参考路径都接纳这一终止末态，沿用既有队列/输出生命周期 |
| Profile / 账本 | `detail/ProfileAccumulationStep.hpp`、`detail/ProfileProjectionStep.hpp`、`PhysicalAcceleratedEmRouter.hpp` | 设备累积与 host/projected 记录都排除重复 cut rest sink，并增加正确的介质 rest source |
| 通用两介质输运 | `corsika/modules/transport/detail/InterfaceEmStep.hpp` | 同一共享末态接入两侧材料，不在山体应用重写湮灭公式 |
| 山体 CPU 账本 | `applications/detail/mountain/TerrainEnergyLedger.hpp` | 将标量 cut 湮灭的介质电子静质量输入纳入总账 |

上述缩写路径中 `detail/` 与 `kokkos/` 相对于 `corsika/accelerator/em/`。本次没有修改射电公式、强子模型、表格参数化、薄化默认值或 CLI。

## 3. 验证结果与边界

- `testStoppedPositron`：202 个断言、2 个测试通过。覆盖不同 cut、权重、时间 cut、两个入口以及 post-step 位置/时间。
- `testKokkosStoppedPositron`：真实 CUDA 和 OpenMP 各 1,000,000 点，错误为 0；检查反向动量、双光子能量、完整权重（即便普通随机反应薄化开启）、history/parent、photon cut、时间 cut 和角分布矩。
- 各向同性检查的方向一阶矩约 `(0.0002263, -0.0004217, -0.0012764)`，二阶矩约 `(0.333086, 0.333884, 0.333030)`，符合百万点抽样量级。
- 实际空气应用：CPU endpoint、CUDA/OpenMP endpoint、光子低于 cut 的 endpoint 均正常关闭。
- 实际 CUDA/OpenMP `-N 32`（1 GeV photon）：32 个 shower 正常关闭，后 31 个复用 backend，profile 分配不随 shower 序号增长。CUDA 此配置 profile 分配恒为 81,704 字节；这不是对所有高能运行的无限期内存证明。
- OpenMP profile 分片／逆向穿越回归：32 次复用和百万穿越检查通过。
- 常规 ParticleCut 原测试：22 个断言通过；PSR Release 的 `c8_air_shower`、`c8_terrain_cascade` 构建通过。本地 CUDA＋OpenMP 组合构建的 `c8_air_shower` 与 CUDA `CORSIKA8InterfaceEm` 库也编译通过，未替换既有 install。
- 两介质的实际 OpenMP session 测试通过：水／岩石两侧停止湮灭、完整权重、光子谱系、动能沉积与介质源项闭合，时间 cut 不生成光子；百万几何求交、6 条跨界轨迹与 resident replay 也通过。这不是山体大样本验收或 CUDA 山体实机验证。
- 没有 AMD/Intel GPU 的本次实机证据，不能把共用设备函数等同于 HIP/SYCL 实机验收。

### 能量账本的实际小例子

总能量 0.7 MeV 的正电子、photon cut 0.25 MeV，实际 CUDA 与 OpenMP 均发生 1 次停止湮灭、6 次 Compton：

| 账本项 | GeV |
|---|---:|
| 初级总能量 | 0.0007000000 |
| 所有反应累计介质电子静质量输入 | 0.0035769925766 |
| 总沉积 | 0.0012109989 |
| 被 cut 的电子静质量 | 0.0030659934 |
| 显式质量约定修正 | 约 −2.766×10⁻¹⁰ |
| 加权源项 − 终态项 | 0（本例输出精度内） |

介质静质量是**总输入账项**，其中 Compton 产生的 recoil 电子最终被 cut 时，其静质量又进入 sink，不能把它当额外沉积。若 photon cut 改为 1 MeV，两个光子直接沉积：`0.7 + 0.5109989 = 1.2109989 MeV`。

该小例子也暴露了原有定点累计量程仅按 `2×primary energy` 估算，对 sub-MeV 初级过小。现将量程下限设为 1 GeV，仅改变计数器比例尺，不分配更多数组、不增加沉积；`Eprimary >= 0.5 GeV` 完全沿用原量程，包括当前 215.4 TeV 生产。

## 4. 新数据必须独立成批

新 cohort 标签为 `stopped-positron-two-photon-v1`。旧 C8 物理样本留存，不从旧 profile 猜测并扣减／补加这部分光子。正常 C7 不关闭停止湮灭、不重跑已合格样本。

当前比较条件不变：215.4 TeV proton，theta 60°、phi 225°，QGSJet-III＋FLUKA，emcut 0.25 MeV，emthin 1e-6，max-weight 2.154，CoREAS-only；种子 `7+6i` 与 C7 标签一致。相同标签不保证相同 shower tree。

- 本地 CUDA：新 cohort 1000 例，70% 显存上限，逐事例进程、保留至少 4 GiB WSL 可用内存和 12 GiB 数据盘空间。
- dirac OpenMP：128 核，偶数全局序号 500 例；保留原 C7 MPI128 逐事例调度与已完成样本。
- PSR OpenMP：130 个相对空闲的物理核，奇数全局序号 500 例。
- 每个主机的精确可执行文件 SHA、输入、模型库、线程数、验证门禁及物理 revision 都冻结记录；只在旧单个事例完成后切换。新样本不与旧 C8 混合平均。

5 例同种子机制复验已完成，种子为 13、25、37、49、61。正常 C7 不关闭停止湮灭：

| 指标 | 正常 C7 | 旧 beta5（已修正反向计数） | 新 beta5 |
|---|---:|---:|---:|
| 逐事例光子／轻子 profile 面积比的均值 | 7.67201 | 6.66897 | 7.29561 |
| 上述比值相对 C7 | — | −13.07% | −4.91% |
| 光子 profile 面积相对 C7 | — | −10.18% | −2.50% |

新版本的光子／轻子比仍低于 C7；配对标签 bootstrap 95% 区间为约 [−5.40%, −4.43%]，独立重采样约 [−5.37%, −4.47%]。小样本机制复验不能代替 1000 例统计，也不能把这个剩余差异宣布为已经解决。其余组分和 profile 形状也仍需大样本判断。本次没有按峰值或积分对图做重新归一化。

图、原数值、报告已同步到本地：

```text
D:\CorsikaData\corsika_validation_results\c7c8_all_acceptance_gallery_20260919\08_stopped_positron_recomparison_20260920
```

## 5. 构建与生产文件

源码补丁在 beta5 当前工作树。运行任务固定使用隔离候选 `build/stopped-positron-20260920/build/applications/c8_corsika7_compare`（服务器为各比较项目的同名子目录），未原地覆盖旧 campaign 的二进制；旧默认 build/install 不应被误认作已更新。

应用没有增加必须填写的新参数，重新构建后默认包含此停止处理。测试目标为 `testStoppedPositron`、`testKokkosStoppedPositron`；两介质实际测试在启用 mountain 构建时由 `testInterfaceTransport` 覆盖。

编译、集成、生产切换及复验脚本保存在工作目录 `78comparison/`：`build_stopped_positron*.sh`、`validate_stopped_positron.py`、`run_stopped_positron_production.py`、`analyze_stopped_positron_comparison.py`。正式统计图不会拿小样本“不显著”冒充 1% 等价。
