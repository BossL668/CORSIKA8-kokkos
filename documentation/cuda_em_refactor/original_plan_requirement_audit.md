# CUDA 电磁级联原始实施计划逐条审计

> 状态更新到 Phase 78 最终生产哈希。工程功能、统计一致性和统计功效分别审计，
> 不把低统计的 raw mean 写成物理失败，也不把“不显著”写成 1% 等价性证明。

状态定义：

```text
PASS       已实现，并有直接测试或生产输出证据
PARTIAL    主路径已实现，但原验收矩阵或某个边界尚未完成
MISSING    尚未实现
IN PROGRESS 正在运行正式验收
```

## 1. 顶层目标与验收

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| CPU 默认路径保持 `Cascade + PROPOSAL` | PASS | `--em-backend` 默认 `proposal`，调度仍为标量 `Cascade + PROPOSAL`；Phase 75 只把原先会丢弃的 SIBYLL/Argon 光核末态改为显式 QGSJet-II 物理回退，因此修复前后的固定 seed 物理输出不应再要求逐字节相同 |
| GPU 处理 \(\gamma,e^-,e^+\) | PASS | physical photon/lepton resident pipelines；完整 application 已运行到 \(10^{18}\) eV |
| 强子、μ、τ 与衰变留在 CPU | PASS | Hybrid route 与 process capability；非 EM 从不进入 device EM queue |
| 五层球形大气 | PASS | `EnvironmentSnapshot`、球面 boundary/grammage 测试 |
| 均匀磁场 | PASS | device leapfrog 与 `testGpuUniformMagneticField` |
| 0.5 MeV cut | PASS | table compatibility hard check；production table 与 application run |
| 现有 EM thinning | PASS | device thinning primitive、完整 shower output |
| 稀有过程指定 CPU fallback | PASS | photoproduction、muon pair 等按 process/reason 统计；SIBYLL/Argon 由 lazy QGSJet-II 接管；Phase 77 又关闭 PROPOSAL Rhode 与 SOPHIA 下限间的窄区，以显式 \(\gamma N\to\pi^0N\) 两体末态守恒闭合。原失败 1 PeV seed 同时触发两类回退并完整结束，`discarded_final_states=0` |
| 表格相对误差 \(\le10^{-3}\) | PASS | tablegen 独立 validation、read-back 与损坏拒绝 |
| 纯 EM 完整 shower 能量闭合 \(\le10^{-4}\) | PASS | Phase 41 严格账本：10 GeV electron 与 100 GeV photon/45° 完整 GPU 覆盖，最大相对残差 \(2.64\times10^{-16}\)；application 对完整无 thinning 事件执行 \(10^{-4}\) hard gate |
| \(X_{\max}\)、profile、沉积、地面分布统计一致 | PASS | Phase 78 当前最终 executable/table 哈希重建 electron、photon、UHE、cut、无 thinning 和 proton 矩阵；1 PeV 首批 200+200 的 3.35 sigma \(X_{\max}\) 异常按功效分析扩展到 800+800 后衰减为 0.827%/2.342 sigma，9/9 curve families 通过。除 20+20 的 \(10^{18}\) eV electron ground radial 仅 11/12 active bins 通过外，其余全部 curve families 通过；未发现稳定且超过 3 sigma 的后端差异 |
| 关键均值偏差 \(\le1\%\) | PARTIAL | 大样本核心纵向/沉积均值通过；1 PeV 800+800 的 charged \(X_{\max}\)、maximum/integral、photon integral、deposit sum 和 closure 均低于 1%，deposit \(X_{\max}\) 与两个 ground totals 低于 3 sigma 但仍超 1%。其他稀疏 ground tail、UHE peak fit 和强子首相互作用量也受相同限制。1 TeV ground tail 实测需要约 \(6.4\times10^5\) 至 \(1.4\times10^6\) events/backend 才能把 3-SE 压到 1%，当前结果不能构成所有 raw mean 的 1% 等价性证明 |
| 纯 EM 端到端加速 \(\ge5\times\) | PASS | Phase 78 最终哈希，热缓存五次 external/shower median ratio 8.60079×/12.62890×，最差 paired external 8.34801×；cold-each shower timing 6.01206×，但包含进程固定初始化的 cold external wall 为 4.85269×。原门禁明确采用热缓存，因此通过 |
| CPU CoREAS 等价 \(e^\pm\) 轨迹段 | PASS | `RadioTrackRecord -> CorsikaOutputSink::onRadioTrack -> doContinuous` |
| GPU 射电不作为首版必要条件 | PASS | `--radio-backend cpu` 保留；另有可选 resident CUDA radio |

## 2. 核心重构

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| 提取 `ScalarCascadeStepper::advance` | PASS | Phase 1 |
| 原 `Cascade` 使用 stepper 且 CPU 固定 seed 等价 | PASS | Phase 1 regression |
| 新增 `HybridCascade` | PASS | Phase 2/4 与 production application |
| `run/forceInteraction/forceDecay` 接口 | PASS | HybridCascade API |
| 不把原 Stack 改为并发容器 | PASS | CPU Stack 保持标量；device queues 独立 |
| GPU 双缓冲/常驻队列 | PASS | workspace 与 resident cross-species queues |
| history/generation/step identity | PASS | transport identity stack 与 scheduler tests |
| 小 batch CPU 前沿展开 | PASS | scalar wavefront expansion |
| 小 batch 展开不会形成无界长尾 | PASS | Phase 40：连续 8 轮预算；同 seed 5 分钟长尾降到 0.553 s |

## 3. CUDA 后端与 wavefront

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| 独立 `CORSIKA8GpuEm` 编译型库 | PASS | CMake target |
| POD double state 与固定单位 | PASS | `EmParticleState`/record types |
| 每线程推进一个粒子到最近限制 | PASS | photon/lepton transport kernels |
| interaction/boundary/deflection/loss/observation 竞争 | PASS | transport tests |
| 稳定 scan 分配次级，避免原子 append | PASS | CUB scan/compaction tests |
| 预分配 workspace，shower 中避免 `cudaMalloc` | PASS | device workspace |
| 多 shower 复用 table/queue/workspace | PASS | Phase 52：production application 跨事件复用 backend；同 seed 26/26 scalar columns 逐位一致，100-shower common timing 再加速 1.606× |
| memory fraction hard limit | PASS | workspace sizing、peak bytes |
| 明确 overflow/failure | PASS | queue、fixed-point、CUDA errors 均 hard fail；Phase 57 起 production router 对 table/LPM/Molière/magnetic/atmosphere 等未预期 fallback 也 hard fail；Phase 73 由真实 CUDA kernel 证明 `unsupported_geometry` 恰好执行一次 scalar step，而 `atmosphere_grammage_failed` 在 strict 模式终止且 CPU step 为 0 |
| 内存不足低能 spill 到 CPU 并统计 | PASS | Phase 46：pending+incoming 全局稳定能量排序；最高能部分留在 device，最低能尾部返回主栈并强制恰好一个 scalar step；1% 显存受控容量测试验证数量和能量边界 |
| `PID × medium × energy` 重排 | PASS | Phase 45：50 bit 三维键、CUB stable radix sort、预分配 gather；独立 8-check 测试及 photon/lepton 完整 oracle；小于 256 的尾部 tile 显式旁路并记录 |

## 4. 随机数与确定性

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| Random123 Philox4x32-10 | PASS | `Philox.hpp` 与官方 reference vector |
| seed/shower/history/step/process/draw 寻址 | PASS | 各过程独立 draw ID 测试 |
| 调度顺序不改变同 history 随机数 | PASS | keyed identity 与 host/device oracle |
| 同 GPU 重复确定性 | PASS | 多阶段完整 Parquet/NPZ byte equality |
| 不使用 `--use_fast_math` | PASS | CUDA build flags |
| double 生产物理 | PASS | particle/table/geometry state double |

## 5. PROPOSAL 拆分与表格

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| rate provider | PASS | Phase 5/6 |
| selected interaction record | PASS | process/component/loss/key records |
| 指定末态 generator 不重抽过程 | PASS | `ProposalCpuFallbackHandler` |
| CPU 用拆分接口重建原 interaction | PASS | split regression/oracle |
| 离线 `gpu_em_tablegen` | PASS | application target |
| rate/component/inverse CDF | PASS | production table |
| continuous range/inverse/dEdX | PASS | table and transport tests |
| LPM metadata | PASS | pair/brems/epair LPM tables |
| metadata/version/hash/medium/cut/domain | PASS | compatibility validation |
| 自适应网格到 \(10^{-3}\) | PASS | independent tablegen validation；Phase 55/56 分别生成 5/50 MeV 表并通过 device query；达到点数上限时也必须由用户容差和独立 validation 双重约束 |
| 热缓存与损坏重建/拒绝 | PASS | rate table strict read；Molière cache corruption test |

## 6. GPU 物理过程

| 过程 | 状态 | 证据或缺口 |
|---|---|---|
| photon pair production | PASS | PROPOSAL oracle、LPM、production |
| Compton | PASS | resident GPU final state、PROPOSAL `NaivCompton` oracle、device statistics |
| photoelectric | PASS | resident GPU final state、binding-energy deposit、device statistics |
| electron/positron bremsstrahlung | PASS | GPU final state、LPM |
| electron/positron pair production | PASS | GPU Epair final state、LPM |
| discrete ionization | PASS | GPU oracle |
| continuous ionization | PASS | range/dEdX transport |
| positron annihilation | PASS | GPU oracle |
| Molière multiple scattering | PASS | mixture-aware interpolation + Newton oracle；Phase 50 复现 CPU `Step` 的磁场增量+散射增量合成 |
| magnetic deflection | PASS | uniform-field device comparison；非零磁场/散射组合有独立方向 oracle |
| ParticleCut | PASS | device endpoint |
| EMThinning | PASS | device primitive与完整 shower |
| photonuclear/photoproduction fallback | PASS | GPU 先固定 process/component/loss 后在 CPU 生成指定末态；SIBYLL 不支持 Argon 时由 QGSJet-II 生成；SOPHIA 下限以下但 \(\pi^0N\) 物理开放时由显式守恒两体模型生成；其余未覆盖区 hard fail |
| photon-induced muon pair fallback | PASS | specified CPU final state |
| 产生 hadron/μ/τ 的末态返回 CPU | PASS | router capability/fallback |

## 7. 环境、轨迹与输出

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| 五层 snapshot | PASS | snapshot builder |
| layer boundary、grammage 与反演 | PASS | spherical atmosphere tests |
| observation/escape records | PASS | production writer |
| `EmStepRecord` | PASS | debug/retaining path |
| `RadioTrackRecord` | PASS | CPU CoREAS/ZHS compatibility |
| `ObservationRecord` | PASS | particle writer |
| GPU resident profile/deposit | PASS | fixed-point accumulators |
| profile 与 writer 合并 | PASS | `CorsikaOutputSink` |
| 自定义未注册 EM Step process 启动失败 | PASS | Phase 44：递归检查普通/切换过程树；未知 continuous process 负向测试会抛错；production CUDA 在 backend 初始化前 hard reject，并把 6/2/4/0 注册统计写入每个 shower provenance |
| CoREAS/ZHS CPU 在线累计 | PASS | `coreas_zhs_online_accumulation.md` |
| 可选 CUDA radio | PASS | resident primitive/production path；Phase 65 的 1 TeV、10-event/8-observer 正式 ensemble 与 Phase 66 的 1 PeV 高轨迹负载均通过；Phase 69 进一步覆盖 proton + photoproduction fallback + radio 完整链路 |

## 8. 构建、CLI 与 provenance

| 要求 | 状态 | 证据或缺口 |
|---|---|---|
| `CORSIKA_ENABLE_CUDA=OFF` 默认 | PASS | CMake |
| CPU build 不需要 CUDA | PASS | Phase 78 最终源码 CPU-only 全树构建与 CTest 10/10；同一源码 CUDA CTest 32/32、Python validation 53/53 |
| CMake/CUDA/C++17 version checks | PASS | top-level CMake |
| CLI backend/device/batch/memory/table/tolerance/deterministic | PASS | `c8_air_shower` |
| 显式 CUDA 配置失败不自动回 CPU | PASS | strict failure |
| incomplete shower marker | PASS | `GpuEmRunOutput` |
| GPU/driver/runtime/table hash metadata | PASS | config/summary |
| CPU/GPU 粒子与 fallback 统计 | PASS | `gpu_em/summary.yaml` |
| 统计样本 executable/table 来源不可混淆 | PASS | Phase 75：每个正式 ensemble root 写 `validation_provenance.json`；运行前后 SHA-256 检查；跨 shard、跨 backend 和增量 pooling 均硬拒绝不同 executable/table，legacy 输出仅可显式诊断 |
| peak memory/kernel/transfer/fallback timing | PASS | detailed stage timing |
| 后端无关逐 shower timing | PASS | Phase 39 `SimulationTiming` |
| Release + sm_89 构建 provenance | PASS | Phase 49：`CMAKE_BUILD_TYPE=Release`、CUDA architecture 89、Release Conan 依赖与干净 imported-target 解析；runner `--require-release-build` hard gate |
| 完整 Release `all` target | PASS | Phase 50：SIBYLL/SOPHIA 混合 C++/Fortran shared link 使用匹配 conda sysroot 的 Fortran driver；全树构建通过 |
| 安装导出与下游链接 | PASS | Phase 72：安装包补齐 CubicInterpolation dependency；独立 downstream `find_package`/link/run 通过；应用使用 `$ORIGIN/../lib`，任意 `--prefix` 安装可找到同前缀 CONEX shared library |

## 9. 测试矩阵

| 矩阵项 | 状态 |
|---|---|
| 单过程 \(10^6\) 级 sampling/oracle | PASS（各过程独立测试） |
| environment boundary/tangent/field | PASS（Phase 73 进一步覆盖最外层边界向外、device `UnsupportedGeometry`、相同 transport identity 单步 CPU bypass，以及实际大气数值失败 hard abort） |
| 1000 × 1 TeV electron | PASS（Phase 78 最终哈希 1000+1000；9/9 curves，通过 7/9 raw means；两个稀疏 ground total 低于 3 sigma，但当前样本不能证明 1% 等价） |
| 200 × 1 PeV | PASS（原定 200+200 首批暴露 charged \(X_{\max}\) 3.35 sigma 异常；没有换 seed 掩盖，而是用同一最终哈希预先扩展到 800+800。最终 9/9 curves、6/9 raw means 通过；其余 deposit \(X_{\max}\) 与两个 ground totals 均低于 3 sigma。新增第二批 200+200 和第三批 400+400 单独分析也各自 9/9 curves 通过） |
| 50 × \(10^{17}\) eV | PASS（electron 扩展到 200+200，9/9 curves；photon 50+50，9/9 curves） |
| 20 × \(10^{18}\) eV | PASS（electron 与 photon 均 20+20；photon 9/9 curves，electron 8/9，剩余 radial family 仅一个低统计 active bin 未过且无超过 3 sigma 的 mean） |
| photon primary | PASS（当前最终哈希覆盖 \(10^{17}\) eV/45° 与 \(10^{18}\) eV/80°，均无 discarded state、overflow 或未知回退） |
| zenith \(0^\circ,45^\circ,80^\circ\) | PASS（当前最终哈希分别由 electron、photon、cut、无 thinning 与 proton 样本覆盖） |
| cut 0.5、5、50 MeV | PASS（三套独立哈希表均通过 table/device validation；0.5 MeV 用于主矩阵，5/50 MeV 各有 200+200 application ensemble） |
| thinning 开/关 | PASS（主矩阵开启；1 TeV electron/80° 的 50+50 无 thinning 样本推进 91,917,669 个 GPU particles，完整结束且无 overflow/spill） |
| 完整 21CMA hadron primary | PASS（当前最终哈希 proton 100+100：29,244,781 个 GPU EM advances、673,128 个 CPU steps、605 个指定 CPU 末态、556 个 SOPHIA 相互作用；无 generic fallback、discard、overflow 或 spill） |
| ground spectrum/lateral/time | PASS（当前最终哈希的 ground curve families 除 20+20 UHE electron 的单个低统计 radial bin 外均过门禁；所有 raw mean 差异低于 3 sigma，但稀疏 total 的 1% 精度仍属统计功效限制） |
| CPU CoREAS trajectory/waveform ensemble | PASS（Phase 65：1 TeV、10-event、8-observer，五类非射电表逐行相同，CoREAS/ZHS 最大归一化差异分别为 \(3.80\times10^{-7}\)/\(9.81\times10^{-7}\)；Phase 66 的 1 PeV 以及 Phase 69 的 proton+fallback 链路也通过） |
| 冷/热 cache performance | PASS（Phase 78 最终哈希：热 external/shower 8.60079×/12.62890×；cold-each external/shower 4.85269×/6.01206×。原验收采用热缓存；冷 external 数字明确保留而不冒充 5× PASS） |
| 5 次性能中位数 | PASS（Phase 78 热/冷各五次；热缓存最差 paired external 8.34801×；runner 对计时前后 artifact hash 做 hard gate） |

## 10. 剩余科研统计工作，不是后端实现阻塞项

原定物理/性能矩阵和最终源码全量回归已经完成。1 PeV 首批异常也已用同一
最终哈希扩展到 800+800，并达到 9/9 curve families 与全部核心纵向均值门禁；
不存在仍在运行或被跳过的原定构建阶段。

若论文需要声明“每个稀疏 ground/UHE/强子 raw mean 的真实后端偏差都小于
1%”，下一步不是继续修改 CUDA 代码，而是先定义等价性检验、置信水平和
方差缩减策略，再按功效分析扩大统计量。当前工程验收已经能够回答代码覆盖、
能量账本、稳定性、分布一致性和热缓存性能问题；不能用事后选择 seed 的方式
关闭统计功效缺口。
