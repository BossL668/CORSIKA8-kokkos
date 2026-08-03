# Phase 81：CPU/CUDA 统计复验、射电采样收敛与当前结论

> 后续 Phase 82 已增加原版 1 TeV 单事例的 CUDA 严格决策回放，并在同一批
> \(e^\pm\) 轨迹上完成 81 天线 CoREAS/ZHS CPU/GPU 波形复刻。逐事例证据
> 与生产 CUDA 独立抽样的边界见
> `phase_82_exact_legacy_event_cuda_decision_replay.md`。
>
> Phase 83 又把本阶段的 500+500 场电子样本重新整理为 \(X_{\max}\)、
> \(N_{\max}\)、纵向粒子含量和地面 EM 粒子数的完整分布、KS/均值/scale
> 检验与图表，见 `phase_83_shower_feature_distribution_validation.md`。

## 1. 先给结论

截至本阶段，证据支持以下结论：

1. CUDA 电磁后端没有复现早期小样本中“辐射能量稳定低约 17%”的现象。
   当前 50 + 50 个电子射电样本中，CPU/CUDA 的原始 radiation-energy
   均值差在四个 CoREAS/ZHS、30--80/50--350 MHz 通道内均小于 1%。
2. 新的 500 + 500 个 1 TeV 电子样本中，charged/shower profile、photon
   profile、能量沉积和能量闭合的主体分布一致；所有经验 KS 距离均低于
   95% 临界值，所有 ensemble 平均曲线均通过逐 bin 检验。
3. 旧的 CoREAS/ZHS 高频巨大差异不是 GPU 电磁反应率错误，而是应用程序
   固定使用 1 ns 射电采样造成的 ZHS 有限差分误差。改用论文的 0.1 ns
   采样、`MaxRad = 0.001 rad` 后，同一 CUDA shower 上 CoREAS/ZHS 的积分
   辐射能量差收敛到 30--80 MHz 的 0.024% 和 50--350 MHz 的 0.617%。
4. 质子 shower 的 300 + 300 个诊断性合并样本没有显示稳定的 CUDA 单向
   偏差；核心均值差约为 \(-0.8\%\) 到 \(-2.7\%\)，全部小于 1.4σ，所有
   KS 距离低于临界值。
5. 质子射电 60 + 60 个样本的未条件化均值仍有约 +16% 到 +21% 的 CUDA
   正偏，但只有 0.7--1.0σ，bootstrap 95% 区间覆盖零。控制每场 shower
   的 EM 沉积和加权 \(e^\pm\) 轨迹长度后，后端系数缩小到低频约 +3%、
   高频约 +5%，仍不显著。

因此，当前 CUDA 电磁级联的核心物理已有较强的正确性证据；电子射电也有
同轨迹、同形式和 ensemble 三层证据。但是仍不能把整个后端标记成“全部生产
验收完成”，原因包括：

- 稀疏地面尾部的 1% 硬门限尚未达到足够统计精度；
- 质子射电的无条件分布仍过宽，60 场不足以关闭严格等价检验；
- 真正的 CUDA 路径不逐事例复刻原版串行随机流；
- 长质子作业会被 UrQMD 1.3.1 的低能中子罕见失败终止；
- UHE、倾斜角矩阵以及最终 21CMA production 配置尚未完成本轮规模的复验。

这里的“正确”是指当前精度下统计物理一致，而不是宣称所有输出逐 bit 相同。

## 2. 比较对象与固定配置

```text
原版 CPU 程序
  /home/yuhanglu/21CMA/corsika-21cma/corsika-build/
  applications/c8_air_shower

CUDA/重构程序
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/
  applications/c8_air_shower

CUDA 物理表
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/
  production_v9_1e-3_1EeV.c8emrt

地磁
  IGRF13.COF，year = 2025
  latitude = 42.5527 deg
  longitude = 86.4153816422 deg
  altitude = 2680.444195 m

天线
  /home/yuhanglu/21CMA/data/antennas.txt
  81 个 NWU 坐标位置

电磁 cut
  0.5 MeV
```

500 场电子样本记录的可执行文件 SHA-256 为：

```text
original CPU
  74a434b059df18b783da5d08c50673e8268e0c1f4b89948614f07329b5bab476

CUDA
  d24d866f6b9e320bc6673fef3e0a5dd0159eae5be136a9a41314489859ccf5c8

rate table file
  140347373b9b5014c8b5bbd6cbcffca5f928eb5190e58c54c725d5843c2b295e
```

这些哈希来自运行时 provenance，不是运行结束后人工猜测。

## 3. 发现并修复的非默认 MaxRad 错误

### 3.1 问题

CPU `TrackingType` 已经读取：

```text
--max-deflection-angle
```

但旧的 resident CUDA lepton transport 在调用
`maximumUniformMagneticStep()` 时仍使用内部默认值 0.2 rad。因此此前任何
“命令行传入 0.001 rad”的 CUDA 扫描实际上是：

```text
CPU tracking     0.001 rad
CUDA tracking    0.2 rad
```

默认 0.2 rad 的历史 CPU/CUDA 比较不受影响；非默认扫描不能作为有效证据。

### 3.2 修复

偏转角现在通过完整配置链传入设备：

```text
applications/c8_air_shower.cpp
  -> EnvironmentSnapshotBuilder
  -> EnvironmentSnapshot.maximum_magnetic_deflection_rad
  -> SphericalAtmosphere / UniformMagneticField
  -> CudaLeptonTransport
```

主要修改文件：

```text
corsika/gpu/em/Types.hpp
corsika/gpu/em/EnvironmentSnapshotBuilder.hpp
corsika/gpu/em/SphericalAtmosphere.hpp
corsika/gpu/em/UniformMagneticField.hpp
src/gpu/em/CudaLeptonTransport.cu
applications/c8_air_shower.cpp
tests/gpu/testGpuSphericalAtmosphere.cpp
tests/gpu/testGpuLeptonTransport.cpp
```

新后端元数据：

```yaml
backend_version: cuda-em-v3-configurable-magnetic-step
environment:
  maximum_magnetic_deflection_rad: 0.001
```

设备测试结果：

```text
GPU spherical-atmosphere validation
  112 checks passed

GPU straight lepton transport validation
  71693 checks passed
```

## 4. 为什么 1 ns 会让 ZHS 高频能量显著偏低

### 4.1 输出算法

当前 time-domain ZHS 先在 observer bin 中累计矢势 \(A\)，写出时使用：

\[
E_i=-\frac{A_{i+1}-A_i}{\Delta t}.
\]

一阶差分相对于连续导数的幅度响应包含：

\[
H(f)=\frac{\sin(\pi f\Delta t)}{\pi f\Delta t}.
\]

原应用程序固定：

```cpp
const TimeType duration_{4e-7_s};
const InverseTimeType sampleRate_{1e+9_Hz};
```

即 \(\Delta t=1\) ns。虽然 500 MHz 的 Nyquist 频率看似覆盖 350 MHz，
但在 350 MHz：

\[
H(350\,{\rm MHz})\simeq 0.81,
\qquad H^2\simeq 0.66.
\]

这已经足以让 ZHS 的高频能量系统下降，而且还没有计入 1 ns binning 对高频
脉冲的 aliasing。对已经写出的 1 ns 波形做解析 `sinc` 反卷积只能把高频
差从约 53% 缩到约 30%，不能恢复已经 alias 的信息；必须重新模拟 0.1 ns
observer。

### 4.2 新增可复现 CLI

应用程序现在支持：

```text
--radio-sampling-rate-ghz
--radio-window-duration-ns
--radio-pretrigger-ns
```

默认仍为原版的 1 GHz、400 ns、10 ns，不改变已有 CPU 基线。正式
50--350 MHz 数值收敛测试使用：

```text
--max-deflection-angle 0.001
--radio-sampling-rate-ghz 10
--radio-window-duration-ns 1000
--radio-pretrigger-ns 100
```

`run_cuda_replay.py` 会拒绝让不支持这些选项的原版二进制与非默认 observer
配置混跑，从而避免静默比较不同采样网格。

## 5. 同一 CUDA shower 的射电收敛

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
cuda_electron_1TeV_radio_maxrad0p001_10GHz_single_v4
```

配置为 1 TeV 垂直电子、IGRF13/2025、81 天线、0.001 rad、10 GHz、
1000 ns。该事例输出：

```text
GPU transport particles     3,591,995
radio track segments        3,582,086
wavefronts                  1,337
queue overflows             0
generic CPU fallbacks       0
complete                    true
```

该 shower 的内部 GPU `energy_ledger` 同时记录了：

```text
complete_coverage           false
relative_closure_error      0.00809
accepted                    false
```

原因是 0.001 rad 产生了 147,047 个 scalar wavefront-expansion step，而当前
设备账本没有覆盖这些 CPU 展开的全部能量通道。这不影响下面“相同轨迹下两种
射电 formalism 是否数值收敛”的用途，但意味着这个单事例不能作为独立的
能量闭合验收。正式能量物理结论来自第 6 节的 500 + 500 ensemble 及其外部
writer 观测量；后续还应让 ledger 覆盖 scalar expansion，或在覆盖不完整时
继续明确拒绝 acceptance。

同 seed 的 1 GHz/400 ns 与 10 GHz/1000 ns 运行具有完全相同的：

```text
profile/profile.parquet
production_profile/profile.parquet
energyloss/dEdX.parquet
particles/particles.parquet
interactions/interactions.parquet
```

这五组文件的 SHA-256 分别逐对相同，证明 observer 设置没有改变 shower
输运。

### 5.1 积分辐射能量

定义：

\[
D=\frac{E_{\rm ZHS}-E_{\rm CoREAS}}{E_{\rm ZHS}}.
\]

| 配置 | 30--80 MHz | 50--350 MHz |
|---|---:|---:|
| 1 ns、0.001 rad | -5.91% | -53.38% |
| 0.1 ns、0.001 rad | -0.024% | -0.617% |

0.1 ns 结果进入 CORSIKA 8 论文给出的 2%/1% formalism 收敛范围。

### 5.2 时域波形

在 50--300 m 强信号区域直接比较同一批轨迹产生的 CoREAS/ZHS 三分量波形：

| 频段 | 最差 cosine | median cosine | 最大相对 L2 | median \(|\Delta f/f|\) |
|---|---:|---:|---:|---:|
| 30--80 MHz | 0.99628 | 0.99931 | 0.0862 | 0.50% |
| 50--350 MHz | 0.99192 | 0.99473 | 0.1276 | 1.24% |

100 m 代表性波形：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
cuda_electron_1TeV_radio_maxrad0p001_10GHz_single_v4/
coreas_zhs_100m_waveforms.png
```

完整逐 observer 数值：

```text
coreas_zhs_waveform_convergence.json
```

## 6. 当前版本 500 + 500 个电子 shower

结果：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_electron_1TeV_500_current_v4
```

原版 CPU 分为 10 个独立 shard，CUDA 为一个 500 shower 库；两臂使用不同
种子。关键分布为：

| 观测量 | CUDA 相对均值差 | z-score | KS / 95% 临界值 | 状态 |
|---|---:|---:|---:|---|
| charged \(X_{\max}\) | -0.696% | 0.71 | 0.072 / 0.086 | 通过 |
| charged 最大值 | -0.256% | 0.40 | 0.036 / 0.086 | 通过 |
| charged 积分 | +0.017% | 0.11 | 0.066 / 0.086 | 通过 |
| photon 积分 | +0.373% | 1.82 | 0.076 / 0.086 | 通过 |
| 总沉积能量 | +0.042% | 0.43 | 0.052 / 0.086 | 通过 |
| 能量闭合 proxy | +0.00065% | 0.07 | 0.048 / 0.086 | 通过 |
| 沉积曲线 \(X_{\max}\) | -1.045% | 0.86 | 0.048 / 0.086 | 超 1%，统计未分辨 |
| 地面 EM weighted count | -1.541% | 0.36 | 0.032 / 0.086 | 超 1%，统计未分辨 |
| 地面 EM kinetic energy | -2.622% | 0.42 | 0.036 / 0.086 | 超 1%，统计未分辨 |

九类平均曲线全部通过：

```text
energy deposit                  100% active bins
charged/electron/EM profiles    100%
photon profile                  98.65%
positron profile                100%
ground energy/radius/time       100%
```

runner 的总状态仍为 `failed`，因为它要求每个核心指标同时满足 1% 和 3σ，
而三个稀疏/峰位置指标超过了 1%。它们的 z-score 只有 0.36--0.86，不能解释
为已确认的 CUDA 系统偏差。正确表述是：

```text
主体电磁级联分布通过；
严格全矩阵 1% production gate 尚未全部通过。
```

旧 200 场电子样本中个别 KS/方差异常在 500 场中没有复现。尤其 charged
最大值的方差比为 0.991，KS 为 0.036；photon 积分方差比为 0.817，KS 为
0.076，均低于临界值。

## 7. 质子 shower 的 300 + 300 诊断合并

来源：

```text
original_vs_cuda_proton_1TeV_200_v1
original_vs_cuda_proton_1TeV_100_repeat_a
```

由于两个历史样本的可执行文件 provenance 不完全相同，这里只作为方向和
分布形状诊断，不冒充正式 guarded pool。汇总文件：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_pooled300_diagnostic_v4.json
```

| 观测量 | CUDA 相对均值差 | z-score | KS / 临界值 |
|---|---:|---:|---:|
| charged \(X_{\max}\) | -0.785% | 0.27 | 0.043 / 0.111 |
| charged 最大值 | -2.721% | 1.18 | 0.077 / 0.111 |
| charged 积分 | -1.299% | 0.69 | 0.070 / 0.111 |
| photon 积分 | -2.000% | 0.89 | 0.060 / 0.111 |
| 总沉积能量 | -1.445% | 1.38 | 0.067 / 0.111 |
| 能量闭合 proxy | -1.533% | 1.49 | 0.090 / 0.111 |

第一组 200 场曾表现为 CUDA 低约 3%--6%，独立 100 场复测则高约 2%--6%。
合并后所有核心量小于 1.5σ，且 KS 全部低于临界值。这不证明质子物理已达到
1% 精度，但排除了一个稳定、固定符号、数个百分点的明显 CUDA 偏差。

两次 400 场尝试及一次 100 场射电尝试被以下上游低能模型错误终止：

```text
UrQMD terminating without collision
iterations = 50000
```

触发粒子是约 0.30--0.315 GeV 的中子与 N/O 核。它不是 CUDA error，但会
使 writer 来不及关闭 parquet，因此不能挽救部分 shower。质子生产验证必须
使用可恢复的小批次。

## 8. 质子射电 60 + 60

成功的三组 20 + 20：

```text
original_vs_cuda_proton_1TeV_radio_20_v1
original_vs_cuda_proton_1TeV_radio_20_current_v4
original_vs_cuda_proton_1TeV_radio_20_current_b_v4
```

诊断性 pool：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_radio_pooled60_v5.json
```

### 8.1 未条件化结果

| 算法 | 频段 | EM-deposit² 归一均值差 | z-score | bootstrap 95% |
|---|---|---:|---:|---:|
| CoREAS | 30--80 MHz | +16.15% | 0.75 | [-20.35%, +70.24%] |
| CoREAS | 50--350 MHz | +21.11% | 1.02 | [-15.14%, +73.78%] |
| ZHS | 30--80 MHz | +16.06% | 0.73 | [-21.28%, +71.28%] |
| ZHS | 50--350 MHz | +21.07% | 1.00 | [-15.06%, +74.03%] |

第三个独立批次自身为：

```text
CoREAS 30--80      +1.10%
ZHS    30--80      -1.03%
CoREAS 50--350    +11.93%
ZHS    50--350     +8.92%
```

全部小于 0.4σ。偏差方向和大小在批次间不稳定。

合并样本的 median 和 KS 提示分布可能仍有形状差异，所以不能只凭均值 z-score
宣布通过。60 场的 radio 分布高度右偏，单场 leave-one-out 可让均值偏差变化
十几个百分点。

### 8.2 条件化结果

对每个算法和频段拟合：

\[
\log E_{\rm radio}
=\alpha+\beta_{\rm backend}I_{\rm CUDA}
+\beta_E\log E_{\rm EM,deposit}
+\beta_L\log L_{e^\pm,\rm weighted}.
\]

得到：

| 算法 | 频段 | 条件 CUDA 系数 | bootstrap 95% |
|---|---|---:|---:|
| CoREAS | 30--80 MHz | +3.23% | [-13.97%, +51.32%] |
| ZHS | 30--80 MHz | +3.14% | [-14.78%, +51.22%] |
| CoREAS | 50--350 MHz | +5.26% | [-11.16%, +54.64%] |
| ZHS | 50--350 MHz | +5.14% | [-11.22%, +54.13%] |

报告：

```text
original_vs_cuda_proton_1TeV_radio_pooled60_conditional_v5.json
```

CoREAS 与 ZHS 给出几乎相同的后端系数，说明它们看到的是同一组底层 shower
涨落，而不是两个独立射电算法同时出现不同的 CUDA 投影 bug。质子射电仍未
达到 production 等价门限，但当前也没有显著的固定 CUDA 辐射增益/缺失证据。

## 9. 早期 “CUDA 辐射低 17%” 的原因

早期 20 + 20 电子样本四个通道均值都低约 17%，但四个通道的 per-shower
相关系数为 0.96--0.98。它们共享相同 shower，不是四次独立发现。

当前 50 + 50 独立复验：

| 算法 | 30--80 MHz | 50--350 MHz |
|---|---:|---:|
| CoREAS | -0.80% | -0.24% |
| ZHS | -0.84% | +0.01% |

从 50 场分布中重复抽取 20 + 20，得到早期同等或更负偏差的概率约为
3.4%--4.3%。这是一个不常见但完全可能的小样本尾部事件。早期均值差大于
median 差，CPU 中少数高信号 shower 对结果影响很强；weighted track length
只差 -0.11%，排除了漏掉约 17% 轨迹的解释。

需要区分两个问题：

```text
CPU vs CUDA 同一 formalism
  早期 -17% 是小样本 shower 尾部，50 场未复现。

CoREAS vs ZHS 同一 shower
  1 ns 时高频差约 53% 是 observer 采样/有限差分误差，
  0.1 ns 后收敛到 0.62%。
```

两者都不是 CUDA 电磁反应率低估，但物理原因不同。

## 10. 相同 seed 能否复刻原版

### 10.1 标量 proposal 控制路径

相同 seed 351001、0.001 rad 下，原版 event 0 与重构
`--em-backend proposal` event 0 的：

```text
profile/profile.parquet                  exact
production_profile/profile.parquet       exact
interactions/interactions.parquet        exact
```

地面粒子数和行顺序也相同，但仍有极小浮点差：

```text
dEdX 5 个 bin
  最大相对差            9.2e-5

一个方向余弦
  最大绝对差            9.3e-10

地面到达时间
  最大绝对差            7.8e-17 s
```

重构程序使用带 history 字段的 `HybridStack`，原版使用普通 `Stack`。物理
shower 相同不等于所有浮点写出逐 bit 相同。若验收标准要求原版文件逐 bit
复刻，需要保留真正的原 `Stack` replay 路径并固定编译器/浮点环境。

### 10.2 真正 CUDA 路径

原版 CPU：

```text
LIFO stack order
global sequential RNG streams
```

CUDA：

```text
wavefront order
history/step/process keyed Philox
```

调度改变后，同一个整数 seed 的随机数被分配给不同物理决定，所以 CUDA
生产路径不会生成同一场 shower。实现完全逐过程复刻需要 decision replay：

```text
interaction/decay distance
selected process and target component
loss fraction v
final-state random draws
multiple scattering
LPM
thinning/cut decisions
```

replay 是调试 oracle；它绕过 GPU 自己的随机采样器，不能代替 ensemble
统计验收。

## 11. 当前推荐命令

### 11.1 30--80 MHz 生产前验证

至少显式记录：

```text
--max-deflection-angle 0.001
--radio-sampling-rate-ghz 10
--radio-window-duration-ns 1000
--radio-pretrigger-ns 100
--antenna-file /home/yuhanglu/21CMA/data/antennas.txt
```

即使只分析 30--80 MHz，也建议保留 0.1 ns，保证 CoREAS/ZHS 数值收敛和后续
高频复用。

### 11.2 统计规则

- 电子/光子固定能量初级：报告 raw radiation energy，或除以固定初级能量²。
- 质子/核初级：同时报告 raw、EM deposit²、track-length²，并对 shower
  development 做条件化；不要只报告一个比值。
- CoREAS、ZHS 和两个频段高度相关，不能当作四个独立显著性检验。
- UrQMD 环境下质子以 10--20 场为一个可恢复 batch，完成后再做 pool。

## 12. 仍需完成的生产验收

优先级从高到低：

1. 在当前固定二进制和表格 provenance 下补足质子 shower 小批次，目标至少
   500 + 500，并保持每批可恢复。
2. 用 0.1 ns observer 对质子射电补充统计；当前 60 场仍不能检验 2% 高频
   门限。
3. 增加 electron/proton 的 PeV、\(10^{17}\) eV、\(10^{18}\) eV 以及
   zenith \(0^\circ,45^\circ,80^\circ\) 矩阵。
4. 对地面稀疏尾部重新定义科研上有意义的等价区域；在样本不足时不能把
   “低 z-score”误写成“通过 1%”。
5. 若逐事例复刻是硬要求，实现独立 `cuda-replay` decision injection；
   不修改生产 Philox 模式。
6. 高精度射电总耗时目前由 CPU CoREAS/ZHS 主导。0.001 rad 单电子事例约
   358 万轨迹，CUDA 级联完成后 CPU 射电求和仍需约 10 分钟；后续性能阶段
   应把射电投影并行化，但不能与电磁输运正确性混为一项验收。

## 13. 主要结果与工具路径

```text
电子 500 + 500
  /home/yuhanglu/21CMA/corsika_validation_results/
  original_vs_cuda_electron_1TeV_500_current_v4/

电子射电 50 + 50
  /home/yuhanglu/21CMA/corsika_validation_results/
  original_vs_cuda_electron_1TeV_radio_50_v1/

0.1 ns 同轨迹收敛
  /home/yuhanglu/21CMA/corsika_validation_results/
  cuda_electron_1TeV_radio_maxrad0p001_10GHz_single_v4/

质子 shower 300 + 300 诊断
  /home/yuhanglu/21CMA/corsika_validation_results/
  original_vs_cuda_proton_1TeV_pooled300_diagnostic_v4.json

质子射电 60 + 60
  /home/yuhanglu/21CMA/corsika_validation_results/
  original_vs_cuda_proton_1TeV_radio_pooled60_v5.json

验证工具
  validation/gpu_em/analyze_cpu_cuda_radio.py
  validation/gpu_em/analyze_radio_bias_attribution.py
  validation/gpu_em/pool_radio_ensembles.py
  validation/gpu_em/run_cuda_replay.py
  validation/gpu_em/run_physics_acceptance.py
```

最终回归：

```text
Python validation tests                         85 / 85 passed
testGpuSphericalAtmosphere                       passed
testGpuLeptonTransport                           passed
```
