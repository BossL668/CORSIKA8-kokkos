# Phase 80：原版 CPU 与 CUDA shower/射电验证

> 后续 500 + 500 电子样本、0.1 ns 射电收敛、MaxRad CUDA 传递修复以及
> 60 + 60 质子射电结果见
> `phase_81_cpu_cuda_validation_sampling_and_verdict.md`。Phase 81 的结论和
> 数值优先于本文件中的阶段性判断。

## 1. 当前结论的边界

这轮验证区分三个不能混为一谈的问题：

1. 原版 CPU 和 CUDA 是否各自在相同 seed 下可重复；
2. 两个后端是否逐事例生成同一场 shower；
3. 两个后端生成的 shower ensemble 是否服从相同的物理分布。

目前的证据支持第 1 项；不支持第 2 项；第 3 项在 1 TeV 电子 shower 和固定
能量电子初级射电的主体观测量上表现良好。两组独立质子样本中的均值差方向
发生反转，没有看到稳定的 CUDA 单向偏差，但质子和质子射电的高波动观测量
仍需用置信区间而不是单个相对差作判断。

这不是降低标准。GPU wavefront 使用按 history 寻址的 Philox，原版标量
`Cascade + PROPOSAL` 使用随 LIFO 执行顺序前进的串行随机流。两者的整数 seed
相同，只表示各自从确定的初始状态开始，并不表示第 \(n\) 个随机数被用于同一
个物理决定。若不增加 decision-replay 输入，生产 CUDA 后端不可能逐事件
复刻原版。

## 2. 固定的比较条件

所有本阶段运行均使用：

```text
原版程序
  /home/yuhanglu/21CMA/corsika-21cma/corsika-build/applications/c8_air_shower

CUDA 程序
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower

GPU 物理表
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/
  production_v9_1e-3_1EeV.c8emrt

地磁模型
  IGRF13.COF，年份 2025

21CMA 天线
  /home/yuhanglu/21CMA/data/antennas.txt
  81 个有效位置

电磁 cut
  0.5 MeV
```

shower-only 比较显式设置 `--ring 0 --antenna-file /dev/null`，避免默认天线使
CoREAS/ZHS 时间混入输运统计。射电比较显式使用上述 81 天线文件，并同时输出
CoREAS 和 ZHS。

## 3. 200 + 200 个电子初级 shower

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_electron_1TeV_200_v1
```

配置为 1 TeV、垂直电子、CPU/CUDA 各 200 场独立 shower、EM thinning
\(10^{-4}\)、最大权重 100。

### 3.1 标量观测量

| 观测量 | 原版 CPU | CUDA | 相对均值差 | z-score | 解释 |
|---|---:|---:|---:|---:|---|
| 总沉积能量 (GeV) | 971.920 | 972.673 | +0.078% | 0.55 | 通过 |
| charged profile 积分 | 395011 | 393674 | -0.339% | 0.83 | 通过 1% 均值门限 |
| photon profile 积分 | 2328616 | 2329905 | +0.055% | 0.12 | 通过 |
| charged \(X_{\max}\) (g/cm²) | 315.577 | 318.915 | +1.058% | 0.75 | 统计未分辨 |
| 沉积 \(X_{\max}\) (g/cm²) | 302.430 | 309.493 | +2.336% | 1.22 | 统计未分辨 |
| 能量闭合 proxy | 0.986794 | 0.986868 | +0.0075% | 0.51 | 通过 |

`charged Xmax` 和 `dE/dX Xmax` 的均值超过 1% 并不等价于已经确认不同。它们的
shower-to-shower 方差很大，bootstrap 95% 相对均值差区间分别为：

```text
charged Xmax       [-1.69%, +3.84%]
dE/dX Xmax         [-1.36%, +6.19%]
```

区间覆盖零，也覆盖 1% 等价区域的一部分，所以准确状态是“当前样本未分辨”，
而不是“证明相同”或“证明不同”。

### 3.2 纵向与地面曲线

所有 active curve bins 的逐 bin 统计通过比例为 100%。ensemble 平均曲线的
相对 L1 差为：

```text
energy deposit       1.82%
charged profile      1.17%
electron profile     1.21%
positron profile     1.26%
photon profile       0.83%
ground energy        3.28%
ground radius        4.59%
ground arrival time  8.85%
```

后面三个地面分布由 shower 尾部少量粒子控制，L1 对空 bin/稀疏尾部很敏感；
其 RMS z-score 分别为 0.82、1.26 和 1.04，没有形成显著的逐 bin 偏差。

## 4. 200 + 200 个质子初级 shower

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_200_v1
```

配置为 1 TeV、天顶角 \(45^\circ\) 的质子，非电磁 cut 为 0.3 GeV，其余
thinning/cut 与电子组相同。两臂使用不重叠随机种子，避免把不同随机算法产生
的事例错误地当成配对样本。

| 观测量 | 原版 CPU | CUDA | CUDA 相对变化 | z-score |
|---|---:|---:|---:|---:|
| charged \(X_{\max}\) (g/cm²) | 360.813 | 356.567 | -1.18% | 0.32 |
| charged 最大值 | 665.87 | 622.96 | -6.44% | 2.33 |
| charged 积分 | 218669 | 209211 | -4.33% | 1.88 |
| photon 积分 | 1148415 | 1083670 | -5.64% | 2.08 |
| 总沉积能量 (GeV) | 735.673 | 713.812 | -2.97% | 2.28 |
| 地面 EM 能量 (GeV) | 5.985 | 4.795 | -19.88% | 0.84 |

这些相关观测量同向偏低，不能忽略；但没有一项达到预定的 3σ 确认条件。尤其
地面 EM 尾部的 20% 表面差异只有 0.84σ。

bootstrap 对 charged 最大值给出 \([-11.57\%,-1.11\%]\)，是本组最值得继续
增加统计量的信号。其他主体量的区间仍较宽。逐 bin 纵向/地面曲线依然全部
通过统计检验，因此当前不能据此修改物理内核或放宽门限。

### 4.1 独立 100 + 100 场方向复测

为了判断上述同向下降是否为稳定的后端偏差，使用完全独立的新种子重复了
100 + 100 场：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_100_repeat_a
```

| 观测量 | CUDA 相对变化 | z-score |
|---|---:|---:|
| charged \(X_{\max}\) | +0.005% | 0.001 |
| charged 最大值 | +5.374% | 1.28 |
| charged 积分 | +5.178% | 1.63 |
| photon 积分 | +5.907% | 1.50 |
| 总沉积能量 | +1.728% | 1.01 |
| 沉积 \(X_{\max}\) | -10.219% | 0.80 |
| 能量闭合 proxy | +1.748% | 1.03 |

第 4 节的 charged/photon 主体量低 4%–6%，本组反而高 5%–6%，而且没有一项
达到 2σ。一个由 GPU 电磁输运引入的固定符号系统偏差不应随独立种子组改变
方向，因此复测显著削弱了“CUDA 稳定低估 shower 规模”的解释。准确结论是：
当前没有稳定方向的后端偏差证据，但质子样本的 combined standard error
仍为数个百分点，尚不能证明严格的 1% 等价。

### 4.2 400 场尝试与 UrQMD 终止

还尝试了两组 400 + 400 场运行，但 CUDA 进程在接近结束时均被 UrQMD 1.3.1
的内部保护终止：

```text
UrQMD terminating without collision
iterations = 50000
```

日志对应低能中子与氮/氧核的强相互作用重采样，不是 CUDA kernel、GPU 表格
或电磁能量守恒错误。运行器按失败策略没有把这两组不完整输出纳入比较。
后续质子大统计应使用较小独立批次，并在批次级保存 provenance 和汇总结果，
避免单个 UrQMD 罕见失败丢掉整组统计。

## 5. 相同 seed 的质子诊断控制

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_50_same_seed_v1
```

本组 CPU/CUDA 均从 seed 292001 开始，并且 CPU 不分片。均值结果为：

```text
charged 最大值       -0.434%
charged 积分         +0.114%
photon 积分          +1.490%
总沉积能量           +0.765%
能量闭合 proxy       +0.983%
```

这组没有复现独立 200 场样本中的 3%–6% 同向下降；第 4.1 节的另一组独立
100 场还出现了反方向变化。两项结果共同支持“强子抽样波动”解释，但现有
样本精度仍不能关闭 1% 等价检查。

更重要的是，同 seed 的 50 对事例没有一对在 shower 标量上逐位相同：

| 观测量 | 完全相同事例 | CPU/CUDA 逐事例相关系数 |
|---|---:|---:|
| 总沉积能量 | 0/50 | 0.194 |
| charged 积分 | 0/50 | 0.092 |
| charged 最大值 | 0/50 | 0.130 |
| photon 积分 | 0/50 | 0.114 |
| charged \(X_{\max}\) | 0/50 | 0.224 |

检查 interaction parquet 可见，第 0 场质子的首次强相互作用末态一致；CPU
随后为 EM 粒子抽取相互作用深度并推进串行 cascade RNG，CUDA 则使用按
history/step 寻址的 Philox。随机流从这里分叉，下一场初级的 cascade 状态也
不再相同。

逐事例报告：

```text
original_vs_cuda_proton_1TeV_50_same_seed_v1/
paired_seed_diagnostics.json
```

## 6. 射电诊断方法

射电比较不能只看一个天线的峰值。验证器对每一场 shower、每个天线、CoREAS
和 ZHS 分别执行：

1. 对时域电场 FFT；
2. 分别保留 30–80 MHz 和 50–350 MHz；
3. 计算三分量 energy fluence；
4. 同半径八方位平均；
5. 积分得到 radiation-energy proxy；
6. 用 writer 新增的 `sum_dEdX_em²` 归一化；
7. 比较均值、median、KS、quantile-Wasserstein、bootstrap 95% 区间；
8. 比较峰值对齐且单位能量归一的脉冲功率模板；
9. 比较真正送入 CoREAS/ZHS 的 \(e^\pm\) segment 数、weighted track
   length、正负电荷 track length 和按动能分箱的 track length。

使用 `sum_dEdX_em` 很重要。质子 shower 的总沉积中包含强子/μ分量，不能拿
总沉积能量平方归一化射电信号。旧输出没有该字段时分析器只为向后兼容才回退
到总 `sum_dEdX`。

## 7. 20 场旧射电样本为什么还不能定案

旧结果位于：

```text
/tmp/c8_cuda_replay_radio_1TeV_20_seed260729_v2
```

四组 CUDA radiation-energy proxy 的均值均低约 17%，z-score 约
1.7–1.8。进一步的分布分析显示：

```text
CoREAS 30–80 MHz
  CPU median       1.491e-28
  CUDA median      1.342e-28       （约 -10%）
  CPU maximum      3.480e-28
  CUDA maximum     1.912e-28

CoREAS 50–350 MHz
  CPU median       2.042e-27
  CUDA median      1.971e-27       （约 -3.5%）
  CPU maximum      4.763e-27
  CUDA maximum     2.846e-27
```

均值差明显大于 median 差，CPU 的少数高信号 shower 对 20 场均值影响很大。
四组 bootstrap 95% 区间中有三组覆盖零；CoREAS 30–80 MHz 的下界只以很小
余量排除零。峰值对齐的平均脉冲功率模板 cosine similarity 通常高于 0.98，
且时间窗边缘能量没有显示 CUDA 特有的截断。

所以目前证据排除了“明显波形错位/输出窗口丢信号”作为首选解释，但还不能在
“小样本高尾波动”和“CUDA 轨迹总量/分段存在偏差”之间定案。

## 8. 50 场带轨迹账本的射电复验

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_electron_1TeV_radio_50_v1
```

本节结果由下列文件给出：

```text
waveform_diagnostics.json
waveform_features.csv
validation_plots/radio_energy_distributions.png
validation_plots/radio_aligned_pulse_templates.png
```

### 8.1 radiation energy

| 算法 | 频段 | 按 `sum_dEdX_em²` 归一的均值差 | z-score | bootstrap 95% 区间 |
|---|---|---:|---:|---:|
| CoREAS | 30–80 MHz | -3.82% | 0.55 | [-16.11%, +10.31%] |
| CoREAS | 50–350 MHz | -3.24% | 0.48 | [-15.16%, +10.10%] |
| ZHS | 30–80 MHz | -3.80% | 0.55 | [-15.95%, +9.84%] |
| ZHS | 50–350 MHz | -3.05% | 0.43 | [-15.61%, +11.00%] |

旧 20 场的 -17% 没有复现，四个区间均覆盖零。若强制使用逐 shower
`sum_dEdX_em²`，30–80 MHz 通过 10% 门限；50–350 MHz 因预设门限为 2%，
formal status 为 failed。这里的 failed 表示“50 场还不能证明该归一化均值
在 2% 内”，不是 CUDA 信号已经显著错误。

本组是固定 1 TeV 电子初级，所以还直接比较了不作 shower 能量归一化的原始
radiation-energy proxy：

| 算法 | 频段 | CUDA 原始均值差 | z-score |
|---|---|---:|---:|
| CoREAS | 30–80 MHz | -0.80% | 0.12 |
| CoREAS | 50–350 MHz | -0.24% | 0.04 |
| ZHS | 30–80 MHz | -0.84% | 0.13 |
| ZHS | 50–350 MHz | +0.01% | 0.002 |

这四个原始信号均值均在 1% 内。归一化后的均值反而移动到 -3% 左右，是因为
原版 CPU 的第 42 场发生了罕见的大能量 photonuclear 转移：

```text
shower 42
  总沉积能量         988.42 GeV
  EM 粒子沉积能量    285.78 GeV
```

其他原版事例的 EM 沉积 median 为 968.40 GeV。把这个事例除以很小的
\(E_{\rm EM}^2\) 会产生高尾。对于固定能量电子初级，原始 radiation energy
是更稳健的后端比较量；对于质子初级，仍必须报告 EM 归一化结果，但要同时
检查低 \(E_{\rm EM}\) 高尾。

因此运行器现在采用显式规则：

```text
electron / positron / photon primary
  默认除以固定 primary energy²

hadron primary
  默认除以逐 shower sum_dEdX_em²
```

按固定 1 TeV primary energy² 的正式报告四个频段全部通过，包括 2% 的高频
门限：

```text
legacy_vs_cuda_primary_normalized.json    status: passed
```

### 8.2 真正送入射电算法的轨迹

CoREAS 和 ZHS 消费同一组轨迹，因此两份轨迹 summary 一致：

| 轨迹量 | CUDA 原始均值差 | z-score | 除以 EM 沉积能后的差 |
|---|---:|---:|---:|
| segment 数 | +1.00% | 0.67 | -0.35% |
| weighted track length | -0.11% | 0.05 | -1.57% |
| energy-weighted track length | -1.03% | 0.83 | -4.92% |
| 最大单段长度 | -6.18% | 1.78 | -10.84% |

总 weighted track length 的原始均值只差 0.11%，直接排除了“CUDA 漏掉约
17% 的 \(e^\pm\) 轨迹”。按 15 个动能区间检查后，没有一个区间达到 2σ；
0.1–1 GeV 区间的 normalized track length 约低 2%–2.5%，高能稀疏区间约低
3%–5%，但全部 bootstrap 区间覆盖零。最大单段长度较短也不会解释信号整体
偏低；如果有影响，它通常意味着 CUDA 分段更细。

### 8.3 时域波形

在 0、100、300、600 m 四个代表半径：

```text
峰值对齐、单位能量脉冲模板 cosine similarity
  CoREAS 30–80 MHz       0.994–0.999
  CoREAS 50–350 MHz      0.981–0.998
  ZHS 30–80 MHz          0.994–0.999
  ZHS 50–350 MHz         0.990–0.999
```

100–600 m 的 pulse-width 均值差约 0%–2.5%，z-score 均小于 1.2。中心天线
高频 pulse width 约低 7%–8%，但只有 1.4–1.6σ；中心峰值高约 15%–22%，也
只有约 1σ。时间窗 edge-energy fraction 没有 CUDA 单向增加，因此没有
看到波形被输出窗口截断的证据。

综合 radiation energy、轨迹和波形，旧 20 场的 -17% 主要是小样本高尾，
不是可重复的 CUDA 射电能量缺失。当前剩余的 2% 高频正式精度问题仍需更多
事例，而不是一个已经定位到内核的确定性 bug。

## 9. 质子初级射电复验

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_proton_1TeV_radio_20_v1
```

本组使用 1 TeV、天顶角 \(45^\circ\) 的质子、81 天线以及 CoREAS/ZHS。由于
射电逐轨迹求和开销很大，首轮为每个后端 20 场，EM thinning 为 \(10^{-3}\)。
其结果用于检查质子 shower 的 EM 能量归一化和时域形状，不与第 4 节
\(10^{-4}\) thinning 的 shower-only 均值混合。

| 算法 | 频段 | EM 归一 radiation-energy 均值差 | z-score | bootstrap 95% 区间 |
|---|---|---:|---:|---:|
| CoREAS | 30–80 MHz | +6.94% | 0.20 | [-43.0%, +98.4%] |
| CoREAS | 50–350 MHz | +4.90% | 0.16 | [-40.7%, +81.3%] |
| ZHS | 30–80 MHz | +9.53% | 0.26 | [-41.6%, +107.3%] |
| ZHS | 50–350 MHz | +6.66% | 0.20 | [-40.3%, +90.1%] |

30–80 MHz 两项在 10% 内；50–350 MHz 两项超过 2% 门限，因此 formal status
为 failed。但所有 z-score 只有 0.16–0.26，bootstrap 区间极宽。这组结果只
能说明 20 场质子 shower 对 2% 射电精度远远不足，不能确认 5%–7% 系统偏差。

轨迹账本给出：

```text
segment 数 / EM deposit               -2.86%   (0.99σ)
weighted track length / EM deposit    +1.93%   (0.28σ)
energy-weighted length / EM deposit   +1.63%   (0.11σ)
```

这里没有出现与 radiation energy 同量级且显著的轨迹缺失。峰值对齐模板的
cosine similarity 在 100–600 m 为 0.986–0.998；中心天线高频最低约
0.960，20 场下仍主要受单 shower 时域结构影响。

结论是：质子射电尚未达到 2% 的定量验收统计量，但现有结果与 CUDA 轨迹漏算
不相容，也没有显著偏差证据。正式收敛需要增加质子射电样本，而不是据此修改
CUDA 物理。

按当前样本方差估算，让三倍 combined standard error 小于门限所需的每后端
事例数约为：

```text
固定能量电子
  30–80 MHz / 10%        约 190
  50–350 MHz / 2%        约 4,600–4,900

质子
  30–80 MHz / 10%        约 2,200–2,400
  50–350 MHz / 2%        约 42,000–48,000
```

这是由当前 TeV shower 的高尾方差外推的量级，不是硬性运行计划；它说明用
20 场质子对 2% 高频门限作肯定/否定结论在统计上不可行。

## 10. 新增与通过的验证代码

```text
validation/gpu_em/analyze_cpu_cuda_radio.py
validation/gpu_em/analyze_paired_seed_control.py
validation/gpu_em/plot_original_cuda_validation.py
validation/gpu_em/run_physics_acceptance.py
validation/gpu_em/run_cuda_replay.py
```

本阶段还在原版和重构版 `RadioProcess` 中加入逐 shower 轨迹账本，在两个
`EnergyLossWriter` 中加入 `sum_dEdX_em`。CUDA resident profile 的沉积通过
`addElectromagneticBin()` 明确进入同一电磁账本，不依赖缺失的逐粒子 PID。

已通过：

```text
Python GPU-EM validation tests
C++ refactor testOutput
C++ original testOutput
refactor Radio module tests
```

## 11. 科研使用建议

当前可以把 CUDA 后端用于继续验证和性能开发，但在以下条件满足前，不应把它
标记成已经完成的 21CMA production physics backend：

1. 用可恢复的小批次独立质子样本把主体 shower 均值的不确定度压到 1% 附近；
2. 增加质子射电统计量；当前 20 场不能检验 2% 高频门限；
3. 对电子中心天线高频 pulse-width 进行 `--max-deflection-angle` 收敛扫描；
4. 在 TeV 验证完成后再扩展至 PeV 和 \(10^{17}\)–\(10^{18}\) eV。

旧 20 场电子射电均值低约 17% 的问题已经由 50 场轨迹账本复验关闭：固定
1 TeV 初级下四个原始 radiation-energy 均值差均小于 1%，weighted track
length 均值差为 -0.11%，没有发现 CUDA 漏算相应比例轨迹的证据。

如果目标是逐过程复刻原版，应单独实现 decision replay；它需要注入原版的
interaction distance、process/component/\(v\)、final-state 随机向量、
multiple scattering、LPM 和 thinning 决定。它是很强的调试 oracle，但绕过
GPU 自己的采样器，因此不能代替生产模式的统计物理验证。
