# beta5 Kokkos / 原标量 CPU：物理路径修复与回归

日期：2026-09-06。对象为 beta5 的 Kokkos-OpenMP / Kokkos-CUDA 共享实现，参考为实际标量 CORSIKA 模块及 PROPOSAL 7.6.2；不是根据已有统计图倒推代码正确性。

## 1. 结论与版本边界

本轮发现并修正了真实的物理抽样相关性错误，以及 cut、末态入栈、沉积和射电边界的确定性差异。最重要的不是插值精度，而是 **native PROPOSAL 过程选择与同一步第一次 Molière 散射共用了完全相同的 Philox key**。

当前源码标记为 `rng_domain_version: 2`、`physics_alignment_revision: 2026-09-06`，run-level backend version 为 `kokkos-em-v2-cpu-alignment-rng2`。旧大样本仍是旧实现的历史数据，不得直接标成这次修复的统计验收。修复会有意改变同 seed 的加速 shower。

原标量 CPU 模块、PROPOSAL 参数化／样条、FLUKA/SIBYLL、生产安装和历史数据没有被修改。本轮使用独立 `build/physics_alignment/{cuda,openmp}` 构建，没有推送 GitHub，也没有启动新的大规模生产 campaign。

**阶段结论：确定性错误已修复并进行局部及整事例回归；尚不能据此宣称所有能区、介质和 observable 的大样本物理等价已经完成。**

## 2. 修复清单

| 问题 | 原行为及影响 | 本轮处理 |
|---|---|---|
| 独立物理变量复用随机数 | SampleLoss 选择和第一项散射均使用 `process=0x454d0003, draw=0`；在相同 history/step 上引入不应有的相关性 | 过程选择改为独立 `0x454d0004`；随机域版本升至 2，增加编译期唯一性门禁和真实 selector/scattering 测试 |
| 连续步长的 cut 终点 | 原设备公式 `m + 0.9999*T_cut` 不同于 CPU 的 `0.9999*(T_cut+m)` | 对齐 CPU 公式，动能 cut 单独查询，不能再从目标总能量反推 cut |
| 两套静质量常数混用 | CORSIKA 栈／ParticleCut 与 PROPOSAL 参数化的电子、muon 静质量略有区别 | 显式区分 transport mass 和 native mass，设备值由 host static_assert 对照生成的 CORSIKA 质量表 |
| PROPOSAL 次级进入栈的总能量 | 原 CPU 将 native 总能量减去 native 质量后，以动能加入 CORSIKA 栈；设备直接保留 native 总能量 | 在物化次级边界执行一次 `E_native-m_native+m_CORSIKA`，thinning 使用转换后总能量；角度、LPM 及参数化仍使用原生物理量 |
| cut 能量的空间分箱 | 设备把 cut 剩余动能与连续损失合并沿整段分配 | 连续损失写线段，cut 动能写终点；resident、projected 和完整记录路径同时修正 |
| 观测面与 cut 同步命中 | CPU ProcessSequence 会继续写 cut；设备能量 cut 分支曾跳过 | 保留两种 CPU observable，同时从闭合账本减去重复出口，不重复计算粒子轨迹 |
| 光电结合能记账 | 设备额外写入 CPU legacy `dE/dX` 未写的结合能 | 不改变电子末态；结合能进入独立诊断出口，不静默丢弃、不单侧改变 legacy profile |
| projected 账本遗漏 | 非驻留 projected 轻子路径漏并入原子电子静质量输入 | 与 resident／完整记录保持相同账本；另记录有符号质量约定修正 |
| 零权重 thinning 删除 | 设备按随机选择 mask 留下最终权重为零的“被选中”次级 | 非 multithin 模式按最终权重删除；原 CPU 的早退和 multithin 行为保持不变 |
| 射电时间 bin 边界 | `floor(x+0.5)` 与 CPU 的 `floor(x+0.5L)` 在相邻 double 上可能分到不同 bin | 拆整数／小数部分复刻长双精度舍入语义；不改原 ZHS 明确使用 double 的公式 |
| CoREAS 精确零 Doppler | 原 CPU 有高精度重算分支，设备缺少 | 恰为零时 host 用 long double、GPU 用补偿双 double；不引入 epsilon 钳位 |
| 传播表下边界及轨迹诊断 | 原 CPU 传播器的边界返回值、诊断 kinetic-energy 质量约定与设备不同 | 对齐 legacy 返回约定；诊断减 CORSIKA transport mass |

其中沉积分箱、光电结合能、重复出口和质量账本不是“截面被改了”。它们首先改变 `dE/dX` 或诊断口径，不能直接证明 `N_e(X)` 偏差的来源。随机域错误则会改变相互作用与散射的联合抽样分布，性质不同。

## 3. 随机数问题的代码与证据

调用链：

1. [MoliereStep.hpp](../corsika/accelerator/em/detail/MoliereStep.hpp) 用输运起点的 history/step 取散射随机数。
2. [LeptonTransportStep.hpp](../corsika/accelerator/em/detail/LeptonTransportStep.hpp) 在相互作用限制处把终点复制为 interaction particle，此处尚未增加 step。
3. [LeptonVertexSelection.hpp](../corsika/accelerator/em/detail/LeptonVertexSelection.hpp) 用该粒子的相同 history/step 取原生 SampleLoss 选择随机数。
4. 旧 [RandomDomains.hpp](../corsika/accelerator/em/RandomDomains.hpp) 为两个用途分配了相同 process/draw，故不是偶然相等。

测试调用真实加速 selector 和 Molière 路径，覆盖 65,536 个 history：

| 指标 | OpenMP | CUDA |
|---|---:|---:|
| 用旧 key 重现完全相同的两个随机数 | 65,536 / 65,536 | 65,536 / 65,536 |
| 修复后相等次数 | 0 / 65,536 | 0 / 65,536 |
| 修复后样本相关系数 | 0.00320283 | 0.00320283 |

相关系数是这个控制样本的结果，不是对整个 RNG 的全面统计认证。真正的修复依据是两个独立物理用途不再复用同一完整 key。

本地 beta4 当前源码 `corsika/accelerator/em/RandomDomains.hpp` 也存在相同旧域定义，因此不是本轮 Kokkos 执行空间造成的新错误。本轮只修改 beta5；beta4 及其已完成数据保持原样。它可能影响历史 native 与 c8emrt 的比较，但必须另做控制实验才能量化贡献，不能直接宣布旧 profile 差异全部由它造成。

## 4. 实际测试与覆盖边界

### 4.1 独立 CPU oracle

新增测试的 expected value 来自原 CPU 模块，不是把同一份设备 helper 再在 host 上算一次。

| 测试 | 对照对象与规模 | 本机结果 |
|---|---|---|
| `testKokkosCpuTransportAlignment` | 真实 PROPOSAL 连续步长 72 组；终止条件 12 组，含 4 个 observation/cut 同步命中；真实 StackView 次级适配 30 组；65,536 个 RNG history | OpenMP、CUDA 通过；最大 grammage 相对差分别 `6.4287e-11`、`6.2354e-11` |
| `testKokkosCpuDepositionAlignment` | 原 EnergyLossWriter + ParticleCut，420 组输入；三种输出路径；N/O/Ar 真实光电末态 | OpenMP、CUDA 通过 |
| `testKokkosScalarRadioAlignment` | 实际 CoREAS/ZHS/TimeDomainObserver，19 组轨迹／模式／步长组合，含近 Cherenkov、末端窗口、复用和定点累积 | OpenMP、CUDA 通过 |
| `testKokkosScalarThinningAlignment` | 真实 Stack/StackView + EMThinning，200,000 组输入、Hillas/weight-limited/早退/零权重/multithin | OpenMP、CUDA 均通过；保留／删除决策零差异，存活权重最大 4 ULP（门限 16 ULP） |

连续 grammage 是由 range 差值构成，不能把本测试的最大差异替换成“所有 range/inverse-range 均达到 `1e-11`”的结论。射电 double 累积的最大相对峰值差约为 CoREAS `1.72e-11`；ZHS 矢势在 OpenMP 为 `4.55e-13`、CUDA 为 `2.35e-12`。生产定点路径还存在绝对量化下限，极弱轨迹的相对误差可达约 `9e-5`，不能忽略绝对误差口径。

### 4.2 共享算法回归与测试夹具诊断

既有 native table 测试使用默认 16,384 点，继续检查表、过程选择、轻子输运、末态、驻留 wavefront、profile、radio 等。它混合了直接 PROPOSAL 对照与共享 helper 的 host/device 比较，仍**不是每 PID／过程／组分各一百万点的全矩阵**。

最终 OpenMP、CUDA 两个 Release 构建各自的 15 项 Kokkos/launcher/兼容性测试全部通过。
日志为 `after/openmp_final_ctest.log` 与 `after/cuda_final_ctest.log`。
这不是整个 CORSIKA 的所有模块测试：独立构建未编译／运行全部 `testModules`、`testFramework` 等旧测试目标。

本轮保留两项测试诊断，避免把夹具问题误称物理缺陷：

- 新 thinning 的初始 raw-pointer mock 在 Release 下出现陈旧权重读取；改用真实 CORSIKA Stack/StackView 后排除。未证明具体编译器 bug，不通过降低优化级别或加打印“修复”物理。
- CUDA native-table 测试在一个 Compton 次级的近零方向分量上触发旧逐分量相对门禁。完整单位向量差仅 `1.0804e-14`，能量、PID、history 和 weight 一致。真实 PROPOSAL 同例也显示数 `1e-14` 的方向舍入差。方向改用单位向量范数误差 `<=2e-12` 并检查归一化，保留旧超限数量／首例诊断；能量和离散字段门禁未放宽，物理公式未改。这不是 PROPOSAL 表误差超限。

### 4.3 整事例回归和重复性

控制组采用 10 GeV 的 proton / photon / electron / positron；proton、photon 各 `-N 2`，electron 用 `theta=80°`，其余 `theta=47°, phi=180°`。positron 显式使用 `emthin=0.1, max-weight=100`，确保覆盖活跃 thinning；其他组保留原默认 thinning。统一 IGRF14、2027 年、同一天线文件和固定 seed。

- 原标量 CPU 重构前后：44 个物理输出文件全部一致，包括过程 trace、profile、粒子及射电数组；最终 CUDA 构建内的 scalar 分支也重复通过。
- CLI 帮助仅 `Usage` 的可执行文件路径不同；参数、默认值及帮助文本未改动。
- 新 CUDA 同后端重复运行：40 个物理输出文件全部一致。
- 新 OpenMP 4 线程同后端重复运行：40 个物理输出文件全部一致。
- `-N 2` 实测第二个 shower 复用 backend；每个事件保留自己的完整输出。
- 另用 0.1 GeV electron、`emthin=0, gpu-min-batch=1, -N 2` 检查完整设备输运的闭合：CUDA 两例 residual 为 0；OpenMP 两例最大相对 residual 约 `1.38e-16`，均为 `complete_coverage=true`。

文件一致性比较读取 Parquet 的逻辑列和射电数值数组，不要求压缩容器字节相同；CPU CSV trace 则要求字节相同。计时不是物理输出。上述只是有界回归，不是 CPU/Kokkos 同 seed 长出相同 shower 的证明，也不是跨后端逐位等价承诺。

纯 EM 闭合例未覆盖高能、thinning 或稀有 CPU fallback。对于包含 CPU EM 步、fallback、memory spill 或 thinning 的事件，现有账本仍标记 `complete_coverage=false`，不能把它误读成完整能量守恒验收已通过。

## 5. 尚未完成／不应掩盖的限制

1. Epair 条件 rho 的拒绝采样与 CPU 数值逆 CDF 不是同一随机映射；Molière 等数值近似也不能承诺逐位相同。本轮未完成高能 LPM、所有介质和全部过程的百万点独立矩阵。
2. `final_state_uniform` 对 Epair 仍是遗留的单数诊断，不能代表生产拒绝采样实际消费的整段随机序列；不可据此声称逐 draw tape 已完整。
3. CPU fallback 的末态有 history-keyed 区域，但其后 LPM、thinning、衰变和强子仍可能消费原 CPU 全局 RNG。改变 wavefront 容量或调度也可能改变后代 history 分配。相同 seed 不保证跨容量／跨后端同 tree。
4. Philox 当前 `uniformOpen01` 使用 32-bit 网格，原 CPU double uniform 通常具有更高分辨率；随机分布验收与逐 draw 相等是不同要求。
5. Epair 与 thinning 的部分保留 draw 编号重叠，但当前三次级 Epair 不执行两次级 thinning，没有形成第二个可达碰撞；后续扩展必须保持这种隔离。
6. 原 CPU CoREAS 近 Cherenkov 分支存在清空 vector 后读取的未定义行为；本轮未修改 CPU，也不令设备复刻非法读。应作为单独的 CPU 修复和 sanitizer 工作处理。ZHS 窗口末端尖峰则已经在真实 CPU 上复现，不是本轮应单侧剪掉的“GPU 错误”。
7. 本机没有 HIP/SYCL 硬件验收；本轮只对 OpenMP 和 RTX 4060 Laptop GPU 的 CUDA 实例运行测试。
8. 本轮没有新 500/2000 系综、100 TeV/1 PeV 热缓存性能验收或长时内存压力验收，不发布新的加速比。此前完成的样本不要删除，但比较时须区分旧随机域版本。

建议下一步以修复后版本重做有序的 photon/electron 小系综，再使用已有 CPU 参考进行 1 TeV proton 大样本；分别比较全部粒子 profile、沉积、地面谱和**未经峰值对齐／归一化的平均时域波形及误差阴影**。不能通过选择“更匹配”的种子、丢弃慢事件或平滑波形代替验收。

## 6. 代码、日志与可回退性

主要说明：

- [输运、末态和随机域](BETA5_TRANSPORT_CPU_ALIGNMENT_20260906_CN.md)
- [沉积与能量账本](BETA5_CPU_DEPOSITION_ALIGNMENT_20260906_CN.md)
- [真实 CPU 射电对照](BETA5_SCALAR_RADIO_ALIGNMENT_20260906_CN.md)
- [修复前只读审计](BETA5_KOKKOS_CPU_PHYSICS_AUDIT_20260906_CN.md)

本地验收目录为 `D:\CorsikaData\corsika_validation_results\beta5_cpu_physics_alignment_20260906_v1`，WSL 对应 `/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_physics_alignment_20260906_v1`：

```text
baseline/  修复前源码归档、两个原二进制、CPU/Kokkos 控制事件
after/     新二进制的测试日志、控制事件、重复事件与纯 EM 闭合例
cpu_before_after.json / cpu_final_cuda_build.json
cuda_repeat.json / openmp_repeat.json
FINAL_SUMMARY.json  二进制哈希、逐例版本/复用/闭合及源文件变更清单
```

修复前源码归档 SHA-256：`de2a58ee023ab4ca6013e8dc2a848eabcdf265eba92bd43d80f3f5bded5d2ccc`。独立构建不替换原 `install/`。执行新测试程序需使用 `build/physics_alignment/cuda/applications/c8_air_shower` 或对应 `openmp` 路径，并保持 `corsika_venv` 和原 FLUKA 运行环境；旧统一入口当前仍指向旧安装，不能仅凭源码已修改就认为旧入口的物理版本已更新。
