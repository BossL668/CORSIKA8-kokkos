# Phase 113：PROPOSAL 原生插值表 GPU 接口与生产门禁

## 1. 目标和当前结论

本阶段增加实验性的 `--gpu-physics-source proposal-native`。程序不再把
PROPOSAL 的插值结果重新采样成完整 `.c8emrt` 表，而是从当前 shower 已经
构造的 PROPOSAL calculator 中只读导出原生 CubicInterpolation 样条状态，
转换为只含 POD 的设备布局并一次上传显存。原有 `c8emrt` 仍是默认生产路径，
CPU `--em-backend proposal` 路径没有改变。

截至 2026-08-31，依赖补丁、原生表导出、GPU 求值、最终 C++ 七项生产门禁、
严格百万点 oracle、真实 shower 冷/热缓存、启动能区门禁及旧路径固定种子
回归均已通过。原计划从零运行的 v4 500 vs 500 系综在 125 vs 125 时主动停止：
随后确认已有 1 TeV 和 100 TeV 的 2000 vs 2000 CPU/`c8emrt` 正式数据可复用，
而 v4 的 observer 文件和显式 `max-weight=100` 又与该基准不相同。v4 因此仅保留
为运行链诊断，不进入物理验收。后续只补充配置完全一致的 `proposal-native`
样本；100 TeV/1 PeV 正式性能门禁尚未执行完成。因此目前仍不建议把
`proposal-native` 改成默认值。

当前实现锁定：

- PROPOSAL 7.6.2；
- CubicInterpolation 0.1.5；
- double 精度且不使用 fast-math；
- 原生表 canonical 格式 v6；
- 第一阶段只接受一种介质化学组成。五层大气可有不同密度，但必须使用相同
  组成。密度 profile、磁场和观测面不改变单位 grammage 截面表；组成、cut、
  参数化或 PROPOSAL 版本改变会得到新的缓存与哈希。

## 2. 版本锁定的依赖补丁

补丁保存在项目内，不直接修改 Conan cache：

```text
third_party/conan/cubicinterpolation/
third_party/conan/proposal/
```

`conan-install.sh` 会导出：

```text
cubicinterpolation/0.1.5@c8gpu/stable
proposal/7.6.2@c8gpu/stable
```

CubicInterpolation 补丁只读导出轴类型、范围、节点、函数值、一阶导数和
混合导数。PROPOSAL 补丁只读导出过程、参数化、component/cross-section
hash、cut、dE/dX 原始 component ordinal、range/displacement 和 LPM 参数。
普通 PROPOSAL API、CPU 求值、缓存序列化和随机数状态不受影响；patched 与
vanilla 依赖回归测试的输出逐字节一致。

## 3. CORSIKA 桥接和 canonical 表

主要实现文件为：

```text
corsika/modules/proposal/NativeCalculatorView.hpp
corsika/gpu/em/tables/ProposalNativeTableExporter.hpp
corsika/gpu/em/tables/ProposalNativeTable.hpp
src/gpu/em/ProposalNativeTable.cpp
corsika/gpu/em/tables/CudaProposalNativeTable.hpp
src/gpu/em/CudaProposalNativeTable.cu
```

`InteractionModel::nativeCalculatorViews()` 和
`ContinuousProcess::nativeCalculatorViews()` 暴露 non-owning const view。
应用从实际承担 CPU fallback 的同一组 calculator 导出，不另建一套物理对象。

导出表按以下键组织：

```text
PID × process × component -> NativeDndxColumn
PID × process × cross-section × component-index -> NativeDedxColumn
PID                       -> NativeUtilityColumn
```

格式 v6 具有三项重要性质：

1. 规范化 SHA-256 覆盖依赖版本、物理常数、cut、物理标识、轴和全部系数；
2. dE/dX 按 PROPOSAL 原始 component ordinal 的确定顺序累加，避免求和次序改变；
3. 所有 dN/dX bicubic 单元都在主机端按 CubicInterpolation 0.1.5 的 Eigen
   表达式预展开为 16 个多项式系数。它是同一样条的布局转换，不是重新采样。

上传前会重新计算 canonical SHA-256，并与导出时哈希比较。哈希不符时拒绝
上传，避免主机表在导出后被修改或损坏。当前标准空气完整表包含 58 个 dN/dX
列、550,000 个统计节点，设备占用 78,129,000 bytes。

## 4. GPU 原生样条求值

`ProposalNativeDeviceView` 仅包含 POD 描述符和设备指针。设备端实现与
PROPOSAL/CubicInterpolation 相同的：

1. linear、exponential 和 exponential-minus-one 坐标变换；
2. bicubic Hermite 累计率求值和负率截零；
3. 过程率、process/component 选择；
4. 各过程的物理 `v_min(E)`、`v_max(E)` 和 cut 公式；
5. Newton 反解和保持 bracket 的 bisection 回退；
6. 连续 dE/dX、range 和 inverse range。

PROPOSAL 7.6.2 的 Newton `digits=20` 是二进制有效位语义。CDF 端点附近和
极窄条件分布中，CPU solver 的停止位置可能对等价浮点表达式非常敏感。生产
路径不再把这类点当作 GPU inverse failure，而使用显式
`NativeSelectionReplay`：

- CPU-only 稀有过程继续在 wavefront 结束后批量处理；
- 对已支持 GPU 过程，只在归一化 residual quantile 距离任一端点小于
  `2e-4` 时批量回放；
- 回放复用原 `selection_uniform`，调用同一 live PROPOSAL `SampleLoss`；
- process 和 component 已经确定并必须复核一致，不产生新的随机数；
- 得到精确 `v` 后再进入既有 specified-final-state 路径。

`NativeSelectionReplay` 的剩余 `loss_quantile` 只是诊断量，在累计率边界可因
浮点舍入恰好等于 1；该原因只依赖保留的 `selection_uniform`，所以允许这个
诊断端点。其他 selected-loss fallback 仍严格要求 `loss_quantile` 位于半开
区间 `[0,1)`。另外，endpoint replay 不能把设备端未实现的过程变成可执行
过程：`GpuFinalStateNotImplemented` 始终保持 fail-closed，只有已实现过程或
明确声明为 CPU-only final state 的过程可以进入相应回放路径。

因此 GPU kernel 不会被单粒子同步回退切碎，随机流也不会因为回放多消耗一个
draw。`cpu_completed_native_selection_replays` 会写入 metadata；任何 process 或
component 不一致均为硬错误。

## 5. 辅助缓存

`.c8emaux` 保存 photon-pair/brems 的 LPM snapshot 以及 electron/muon 的
Molière snapshot。Molière 初值多项式仍由独立的
`.moliere-initial-v1.c8cache` 自动管理；Epair `rho` 不写入上述缓存，当前由
有界 device rejection sampler 直接采样。`.c8emaux` 默认位于：

```text
$XDG_CACHE_HOME/corsika8/gpu-em-aux/
```

未设置 `XDG_CACHE_HOME` 时使用：

```text
$HOME/.cache/corsika8/gpu-em-aux/
```

缓存 key 包含辅助算法、依赖版本、介质、粒子和 cut。缺失时加目录锁生成，
验证后用原子 rename 安装；读取时检查 magic、版本、key 和 payload SHA-256。
用户不需要再为 `proposal-native` 手工运行完整 `gpu_em_table_prepare`。

## 6. 启动和兼容性门禁

应用在启动第一个 shower 前检查配置的 primary 能量上界是否位于所有所需原生
轴的公共能区。固定能量使用该能量，`--energy_range` 使用整个区间的 `eMax`，
而不是第一场随机抽到的能量。超范围会终止 shower，并在
`gpu_em/summary.yaml` 中写入 `complete: false` 和明确原因；不 clamp，也不
静默切换到 `c8emrt` 或 CPU。

当前第一阶段还会执行单一介质组成门禁。相同干空气组成但密度随高度变化可以
复用一张原生表；同时含空气、岩石或冰等不同化学组成的 geometry 尚不支持，
必须等多 composition 索引扩展后再启用。

实际链接的 PROPOSAL/CubicInterpolation 版本，而不是硬编码期望值，会写进
metadata。其他关键字段包括：

```text
gpu_physics_source
proposal_native.proposal_version
proposal_native.cubic_interpolation_version
proposal_native.table_sha256
proposal_native.node_count
proposal_native.device_bytes
proposal_native.aux_sha256
proposal_native.aux_cache_hit
proposal_native.proposal_cache_table_count
proposal_native.proposal_cache_hit_count
proposal_native.proposal_cache_all_hit
proposal_native.newton_iterations
proposal_native.bisection_iterations
proposal_native.inverse_failures
cpu_completed_native_selection_replays
```

## 7. 使用方法

默认生产路径保持不变：

```bash
c8_air_shower \
  --em-backend cuda \
  --gpu-physics-source c8emrt \
  --gpu-table-cache /path/to/table.c8emrt \
  ...
```

实验性原生路径不需要介质 YAML 或完整 `.c8emrt`：

```bash
c8_air_shower \
  --em-backend cuda \
  --gpu-physics-source proposal-native \
  --gpu-aux-cache-dir ~/.cache/corsika8/gpu-em-aux \
  ...
```

新介质/cut 第一次运行仍可能由 PROPOSAL 自动生成自身插值 cache；以后直接
加载、导出并上传。同一进程且 canonical hash 相同的后续 shower 复用设备表。
不支持的参数化、轴、介质组成或能区均直接失败。

## 8. 已完成的严格验收

### 8.1 最终 C++ 七项生产门禁审计

在启动大样本系综前，对原生表上传、启动配置、metadata、连续能损、随机数
来源和 fallback 能力边界进行了最后一轮逐代码审计。以下七项均已关闭：

1. `CudaProposalNativeTable` 在上传前先验证表结构、重新计算 canonical
   SHA-256，并拒绝与导出时哈希不一致的 stale/mutated host table，不能把仅
   在导出时计算过一次的哈希当作上传凭证
   （`src/gpu/em/CudaProposalNativeTable.cu:151,160`）。
2. `c8_air_shower` 的原生能区预检使用配置的 `eMax`；在
   `--energy_range` 模式下不会只检查第一场随机抽到的 primary。检查覆盖所有
   可路由到 GPU 的 PID 的公共原生能区和 transport cut
   （`applications/c8_air_shower.cpp:1724`）。
3. metadata 中的 PROPOSAL 和 CubicInterpolation 版本来自实际编译依赖的
   `getPROPOSALVersion()`/`getCubicInterpolationVersion()`，本构建实测分别为
   7.6.2 和 0.1.5，不再写死预期字符串。
4. 连续 dE/dX 保留 PROPOSAL `cross_list` 的原始 component ordinal 求和顺序；
   设备查找索引与数值 accumulation order 分离，避免因重排浮点加法改变 CPU
   语义（`src/gpu/em/ProposalNativeTable.cpp:1071,1117`）。
5. photon fallback 分别保存过程选择 `selection_uniform`、外层接受拒绝和
   final-state split 的随机数 provenance，不能把 final-state draw 误记为选择
   draw（`src/gpu/em/CudaPhotonPairFinalState.cu:497`）。
6. `NativeSelectionReplay` 的 residual quantile 仅作为诊断量时允许舍入到
   `q==1`；其他 selected-loss fallback 仍严格限制为 `[0,1)`
   （`corsika/gpu/em/ProposalFallbackAdapter.hpp:122`）。
7. `GpuFinalStateNotImplemented` 保持 fail-closed，不能借 endpoint replay
   变成可执行过程或静默 CPU 回退
   （`corsika/gpu/em/ProcessCapabilities.hpp:162`）。

回归结果为 `testGpuEmHost` 187 项检查、`testGpuProposalCpuFallback` 30 项检查，
以及 `testGpuProposalNativeTable` 4,617 项检查全部通过。这里的“通过”只表示
上述代码语义和异常门禁已经闭环；独立 shower 的统计等价性仍由后面的正式
v4 系综单独判断。

### 8.2 百万点逐过程 oracle

早期 v6 报告位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_proposal_native_million_oracle_20260831/
  proposal_native_oracle_1m_replay_v6.json
```

该报告使用 PROPOSAL 内置 `Air` fixture；后续审计发现，正式
`c8_air_shower` 的 calculator 是由 CORSIKA `AirDry1Atm` 组分显式构造，且
运行时 medium identity 使用 `NuclearComposition::getHash()`，因此 v6 的表哈希
不是正式 shower 的 `7d6182...`。v6 只保留为 oracle 开发记录，不再作为正式
介质的最终证据。

validation-only calculator fixture 已改为与生产路径相同的
`N=0.78479/O=0.21052/Ar=0.00469` 组成、介质属性、reference density 和 CORSIKA
composition hash；核心 shower、CUDA kernel 和 PROPOSAL CPU 路径均未因此修改。
新的正式介质报告位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_proposal_native_production_air_million_oracle_20260831/
  proposal_native_production_air_oracle_1m_v1.json
```

新报告表 SHA-256 为
`7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f`，与 2,000
个正式 proposal-native shower 完全相同。测试覆盖：

- photon、electron、positron、muon-minus、muon-plus 各 1,000,000 次完整
  process/component/loss 选择；
- 58 个 dN/dX 列各 1,000,000 个随机 `(E,u)`；
- 四种带电粒子的 dE/dX、range、inverse range 各 1,000,000 点。

完整选择器 oracle 直接调用当前 calculator 的 live PROPOSAL
`CrossSection::SampleLoss`，不是拿导出后的另一份 host evaluator 自我比较；
报告同时记录 `production_path_modified: false`，测试没有为通过 oracle 改写
生产 kernel 或 shower 调度路径。

原生标量求值结果为：

```text
process mismatches:                   0 / 5,000,000
component mismatches:                 0 / 5,000,000
completed-selector loss violations:   1 / 5,000,000
rate max normalized error:            2.50630e-14
cumulative max normalized error:      3.81525e-14
inverse-loss max relative error:       2.21497e-12
dE/dX max relative error:              6.17672e-16
range max relative error:              3.92351e-14
inverse-range max relative error:      2.22709e-13
status/live/root exceptions:           0
accepted:                              false
```

唯一严格超限点来自 photon Compton：能量约 `2.06437e7 MeV`，过程和目标组分仍
与 live PROPOSAL 完全相同，但总率末位舍入使过程内 residual quantile 相差
`6.60e-11`，逆 CDF 后 `v` 相对差为 `4.56834e-10`，超过预设 `1e-10` 门槛。
electron、positron 和两个 muon PID 均无 completed-loss 超限。另有 142 个
residual-quantile 诊断点超过 `2e-12`；它们均未改变过程/组分，除上述一个点外也
没有使 `v` 超过最终门槛。

五种粒子的生产回放计数依次为 3033、440、394、7202 和 7107，其中包含本来
就由 CPU 处理的过程以及窄小端点带。另有 37 个仅发生在 exact/`nextafter`
构造边界的 measure-zero 分支差异，最大边界偏移为 1 ULP，仍作为非门禁浮点
诊断单独报告。结论是过程/组分选择门已经通过，但完整 `v` 的严格 oracle 仍为
**warning/fail**，不能再沿用 v6 的 `accepted: true` 结论。

### 8.3 真实 shower、缓存和确定性

冷启动后，同一进程连续两个 shower 正常完成；第二个 shower 报告设备表
`reused: true`。独立新进程的热缓存运行报告 `aux_cache_hit: true`。两次相同
种子的热运行中，CoREAS/ZHS、dE/dX、interaction、particle、production 和
longitudinal profile 等 9 个物理文件全部逐字节一致。

实际记录为：

```text
PROPOSAL version:              7.6.2
CubicInterpolation version:    0.1.5
table SHA-256:                 7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f
node count:                    550000
device bytes:                  78129000
aux SHA-256:                   5d389cde09fb75bf4d53475ef7f8cdebaa2993923c8e9a720df4fd0af7c67c9f
native inverse failures:       0
queue overflow/spill:          0
```

超出公共能区的高能 primary 启动测试以非零状态退出，并留下明确 incomplete
summary。独立回归使用 `--energy_range 1000 200000000000 -N 2`，退出码为 1，
顶层记录 `showers: 0`，`gpu_em/summary.yaml` 记录：

```text
complete: false
status: incomplete
failure_reason: configured maximum primary energy exceeds the common
                PROPOSAL native interpolation domain
```

这证明门禁使用配置的整个能谱上限，并在任何 shower 输运前终止，没有静默
fallback。相关 C++ host/fallback 回归分别通过 187 和 30 项检查。

### 8.4 旧路径回归

修改前后分别执行相同固定种子：

- CPU `--em-backend proposal`：9/9 个物理文件 SHA-256 相同；
- 默认 CUDA `c8emrt`：9/9 个物理文件 SHA-256 相同。

仅配置、计时及新增 metadata 不同。这证明原生表接口和 replay 门禁没有改变
默认 CPU 或既有 `c8emrt` 物理路径。

真实 shower 的现有 hybrid energy ledger 仍未覆盖标量 EM step 和部分 CPU
fallback，因此会报告 `complete_coverage: false`。这是诊断覆盖缺口，不能拿
该字段宣称能量闭合通过；本阶段也没有用它掩盖或否决逐过程 oracle。

## 9. 系综和性能门禁状态

### 9.1 证据等级和旧任务边界

10 vs 10 的 1 TeV proton pilot 已完成：14 条 longitudinal/ground 曲线均通过，
关键标量的绝对 z-score 均小于 1.8，但样本太小，预先定义的均值和 radio
bootstrap 等价区间不能收敛，因此该 pilot 不构成统计验收。该运行还暴露出
省略 `--max-weight` 时自动最大权重小于 1，导致 `1e-6` thinning 实际不
启动；它只能标为未薄化 pilot。

`...20260831_v2` 中保留的 100 vs 100 结果只用于验证比较器、绘图器和射电
分析流水线。它来自最终七项审计前的旧可执行文件，而且 runner 因空 YAML map
解析问题在中途停止，**不得作为 proposal-native 的最终物理验收证据**。该
诊断样本的 14 条正式 shower 曲线均通过；四个关键标量超过 1% 阈值，但
`|z|=0.257--1.185`，属于小样本下统计不确定。射电输入完整性通过，但 1 TeV
弱信号使 bootstrap 比值区间过宽，完整射电等价门禁未通过。这些结果只说明
分析链能够正确暴露“不确定/失败”，不能说明最终新二进制通过或不通过物理
等价性。v3 的单个 25-event `c8emrt` 批次同样因引用审计前旧二进制而主动
停止，并完全排除在正式统计之外。

### 9.2 已停止的 v4 500 vs 500 运行链诊断

唯一正式系综目录为：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  final_beta4_proposal_native_c8emrt_500v500_proton_1TeV_
  emthin1e-6_maxw100_20260831_v4/
```

其 `campaign_manifest.json` 原计划冻结 500 个 `c8emrt` 与 500 个
`proposal-native` 独立 shower、25 events/batch、交替运行、互不重叠的种子段，
以及以下共同物理配置：1 TeV vertical proton、IGRF14/2027、
`emthin=1e-6`、显式 `max-weight=100`、0.5 MeV EM cut、完整 CUDA EM 和 CUDA
radio。两臂都已经确认 thinning 可从单位权重激活且实际出现 thinning vertex。

诊断运行期间使用的冻结 artifacts 为：

```text
c8_air_shower SHA-256:
  71ec7c41d95ab5d719fdd01fb22db212fa598dbc338362ee63cfcb5cf28afa46
runner SHA-256:
  e2247c711ffe4c16b9b4b4b8cb82be0144240b73e83cda30f77949d53b942c97
c8emrt file SHA-256:
  2a81ae0b2c2925171f999863926575c111f3d4fe8c8dacc28c3391215e8aa9e0
proposal-native canonical SHA-256:
  7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f
proposal-native auxiliary SHA-256:
  5d389cde09fb75bf4d53475ef7f8cdebaa2993923c8e9a720df4fd0af7c67c9f
antenna SHA-256:
  067f1dd90eddcbf8b35d75fb999c44bdcc5b15ee2575a418a3676583b211196a
FLUKA library SHA-256:
  6ed44cce8b8dc740053e82ceff963aeaf6bfda3add9beef7873cd7ed1f19a5ad
```

在 manifest 的 `2026-08-31T04:02:49.967702+00:00` 快照中，两臂各完成
5 个批次、125 个 shower，失败条目为 0。随后根据已有数据复用方案主动停止
systemd unit；manifest 中残留的 `running` 状态和未提升的 batch 005 attempt
不表示任务仍在后台运行，也不得计入样本。125 vs 125 canonical 批次的累计检查中
generic fallback、queue overflow、native inverse failure、memory spill 和
cross-species host spill 均为 0；proposal-native 的 6,469 个 completed selected
loss 全部由 6,469 个显式 `NativeSelectionReplay` 闭合。这里的计数只是运行链
诊断，不是最终统计结果。

停止后的配置核查发现，v4 使用的 observer 文件 SHA-256 为
`067f1dd9...`，并显式设置 `max-weight=100`；已完成的 1 TeV/100 TeV
2000 vs 2000 基准使用 observer SHA-256 `238a4818...`，且省略
`--max-weight`。两者会分别改变射电采样位置和 thinning 语义，因此不能直接
混入同一系综。后续 native-only 正式样本必须使用
`antennas_nwu_coordinates_test.txt`，同时省略 `--max-weight`，再分别对已有
CPU 和 `c8emrt` 原始分片做物理与时间比较。

此前 runner 的空 map 问题已经修复：仅当 `generic=specified=0` 时允许 YAML
中的 reason/process map 为空，并强制以下逐批次闭合关系：

```text
sum(reason map) = sum(process map) = generic + specified
specified = deferred queued = deferred flushed
completed selected loss = sum(selected-loss reasons)
completed native replay = native_selection_replay reason count
```

新的配置匹配 native-only 系综完成后，才会比较全部纵向组分、Xmax、能量
沉积、地面粒子和射电脉冲特征，并作出物理验收结论。

### 9.3 已完成的配置匹配 native-only 正式任务

当前只补充既有正式基准中缺少的 `proposal-native` 一臂，不重新运行 CPU 或
`c8emrt`。任务目录为：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  final_beta4_proposal_native2000_proton_1TeV_vertical_
  emthin1e-6_defaultmaxw_igrf14_2027_v1/
```

该任务使用 `--sources proposal-native`，目标为 2,000 个 1 TeV vertical proton
shower、25 events/batch、IGRF14/2027、0.5 MeV EM cut、`emthin=1e-6`、完整
CUDA EM 与 CUDA radio。它不需要也没有传入 `.c8emrt` 表；单源模式禁止非零
`--paired-events`，因为 paired 诊断必须同时存在 `c8emrt` 和
`proposal-native` 两臂。任务省略应用的 `--max-weight`，从而与既有正式数据
一样使用自动 Kobal 上限，而不是 v4 诊断中的显式 100。显存上限设为 0.90，
`gpu-min-batch=4096`，原生表容差为 `5e-4`，native seed 从 2026328001 开始。

冻结 provenance 为：

```text
c8_air_shower SHA-256:
  71ec7c41d95ab5d719fdd01fb22db212fa598dbc338362ee63cfcb5cf28afa46
runner SHA-256:
  d83e1715f79bc777ba6958949ba5db7034a020a9ceffa8d78d10a919b8afe73f
antenna SHA-256:
  238a481851b4d39e9fcc18ed5afefd5ea90a806e235ad7aa6e3bddae0e138668
FLUKA library SHA-256:
  6ed44cce8b8dc740053e82ceff963aeaf6bfda3add9beef7873cd7ed1f19a5ad
```

runner 的恢复单位是完整 batch，而不是单个 shower。每个 batch 先写入私有
`.attempt-proposal-native-NNN`；只有子进程正常退出且输出、GPU integrity 和
provenance 全部通过后，才原子提升为 canonical
`formal/batch_NNN/proposal-native`。中断后必须原样重跑同一命令：runner 先取
非阻塞 campaign lock，并逐项比较 immutable 配置与 executable、antenna、
FLUKA、runner 哈希；任何变化都会拒绝恢复。已经提升的 batch 会重新验证后
跳过；遗留为 `running` 的 attempt 会标为 `interrupted_orphaned`，其私有目录
移入 `failed_attempts/`，然后以递增 attempt 编号从该整个 batch 重新运行。
普通模拟或验收失败同样会归档私有目录、把状态写为 `failed_retriable` 并立即
停止，不会跳过失败 batch 继续向后运行。

该任务现已达到 manifest `status: complete`，formal 计数为 2,000，完整性审计
通过。与配置匹配的 2,000-event CPU 数据比较后，主要纵向 profile、能量沉积
总和和能量闭合通过；固定深度 EM profile 的 global/peak/L1 permutation
`p=0.187/0.338/0.362`，未发现显著均值曲线差异。CPU/GPU 单事例 mean 为
`76.81/3.63 s`，median 为 `74.92/3.36 s`；由于来自不同机器，该比值只描述生产
周转时间。

正式结果仍有三个警告：charged Xmax 方差的 Brown--Forsythe `p=0.0064`；ground
EM weighted count 和 kinetic energy 均值分别为 `-7.14%`、`-12.61%`；部分
CoREAS/ZHS pulse-width 径向门未通过。射电振幅径向曲线均通过，同轨迹的
CPU/CUDA CoREAS/ZHS 投影仍保持约 `1e-7` relative L2。旧 v4 的 125 vs 125 只
保留为配置不兼容的运行链诊断，未混入正式统计。

### 9.4 尚待完成的生产门禁

- 关闭正式介质百万点 oracle 的唯一 Compton `v` 严格超限点；
- 解释 Xmax 方差、ground EM 尾部和 pulse-width 警告；
- 100 TeV 与 1 PeV 热缓存 ABBA 性能对照；
- `proposal-native` 中位时间相对当前 beta4 p2.1 `c8emrt` 不回退超过 3%。

在这些门禁完成前，README 和 CLI 继续把 `c8emrt` 保持为默认生产后端。
