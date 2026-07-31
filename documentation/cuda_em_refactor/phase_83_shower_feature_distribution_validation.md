# Phase 83：原版 CPU 与生产 CUDA shower 特征分布验证

## 1. 问题与结论

判断生产 CUDA 后端是否与原版 CPU 物理一致，不能要求两个后端在相同 seed
下逐事例产生相同粒子树。正确方法是：

1. 固定初级粒子、能量、方向、大气、地磁、cut、thinning 和观测面；
2. 分别产生足够多的独立 CPU 和 CUDA shower；
3. 将每场 shower 先压缩为一个独立的物理观测量；
4. 比较均值、方差、分位数、完整经验分布和纵向平均曲线。

是的，如果 CUDA 和原版实现的是相同物理模型，两组 shower 的统计分布应当
在有限样本涨落内一致。

本阶段重新分析了已有正式样本：

```text
500 original CPU showers
500 production CUDA showers
```

对 1 TeV 垂直电子 shower，当前结果为：

- charged \(X_{\max}\)、charged \(N_{\max}\)、charged 纵向积分和 photon
  纵向积分全部通过 1% 均值门限与 3σ 门限；
- 8 个选定特征的经验 KS 距离全部低于 95% 临界值；
- 8 个特征的 KS、Welch mean 和 Brown–Forsythe scale 检验均没有发现
  \(p<0.05\) 的差异；
- charged、photon 和能量沉积的 ensemble 平均纵向曲线在 shower 主体区域
  重合；
- 地面 EM 粒子数、地面能量和能损 \(X_{\max}\) 的观测均值超过了严格 1%
  门限，但差异只有 0.36–0.86σ，bootstrap 区间跨过零，属于高方差下的
  “统计未分辨”，不是已确认的 CUDA 偏差。

因此当前准确结论是：

> 这 500+500 场数据支持原版 CPU 与生产 CUDA 电磁 shower 的主体分布
> 一致；没有发现统计显著的分布差异。但有限样本不能证明数学意义上的分布
> 完全相等，地面高方差尾部也尚未达到严格的 ±1% 等价精度。

## 2. 样本与固定配置

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_electron_1TeV_500_current_v4
```

| 参数 | 值 |
|---|---|
| 初级粒子 | electron，PDG 11 |
| 初级总能量 | 1000 GeV |
| zenith / azimuth | \(0^\circ/0^\circ\) |
| 每个后端 shower 数 | 500 |
| 原版 CPU seed streams | 360001–360010，10 shards |
| CUDA base seed | 370001 |
| EM cut | 0.5 MeV |
| EM thinning | \(10^{-4}\) |
| maximum weight | 100 |
| hadron/muon/tau cut | \(10^{13}\) GeV |
| radio observers | disabled |
| CPU 每进程线程数 | 1 |

原版 CPU 与 CUDA 使用不同的 seed 集合，这是正式“两独立样本”检验的设计，
不是配置不一致。

`--seed` 初始化整个随机流。一个 seed 下连续产生 500 场 shower，仍然是
500 个不同随机 realization；CUDA 的随机地址还显式包含 `shower_id`。同一
整数 seed 在两个后端并不会形成有效的 event pair，因为 LIFO 顺序随机流和
history-keyed Philox 在第一次分支后便映射到不同物理决定。

同 seed 应保留为调试控制；正式 KS/均值/方差检验应使用独立 ensemble，避免
错误地把两个本来已经解耦的 shower 当成配对观测。

## 3. “粒子总数”的定义

在启用 thinning 后，“粒子总数”需要明确是哪一种量：

### 3.1 \(N_{\max}\)

```text
profile_charged_max
```

这是 longitudinal profile 中 shower maximum 处的带权
\(e^-+e^+\) 数量。它最接近通常所说的 shower 最大粒子数。

### 3.2 纵向粒子数积分

```text
profile_charged_integral
profile_photon_integral
```

定义为：

\[
I_i=\int N_i(X)\,dX.
\]

它不是“唯一粒子 ID 的计数”，而是整场 shower 的粒子含量/track-length
proxy，通常比单个 bin 的 \(N_{\max}\) 更稳定。

### 3.3 地面粒子数

```text
ground_em_weighted_count
```

这是到达观测面的 \(\gamma,e^-,e^+\) thinning 权重之和：

\[
N_{\rm ground}^{\rm EM}
  =\sum_{j\in\{\gamma,e^-,e^+\}}w_j.
\]

不能直接比较输出 Parquet 的行数，因为 thinning 后每一行可以代表多个
物理粒子，而且两个后端保留的 Monte Carlo representative 数量可以不同。

### 3.4 本阶段没有使用的“内部处理粒子数”

CUDA summary 中的 `gpu_particles` 是计算工作量计数，不是直接物理 observable。
原版 CPU 当前没有完全相同定义的逐 shower 计数，因此不能把二者当成
“物理粒子总数”比较。后续如需比较过程级 multiplicity，应增加统一的
weighted birth/termination ledger。

## 4. 统计方法

每一场 shower 是一个独立样本。不能把纵向 profile 的每个 bin 或地面输出的
每个粒子行当成独立样本，否则会严重低估 shower-to-shower fluctuation。

本阶段对每个 scalar observable 使用：

### 4.1 相对均值差

\[
\delta_\mu=
\frac{\bar{x}_{\rm CUDA}-\bar{x}_{\rm CPU}}
     {\bar{x}_{\rm CPU}}.
\]

同时给出：

\[
z=\frac{|\bar{x}_{\rm CUDA}-\bar{x}_{\rm CPU}|}
        {\sqrt{s_{\rm CPU}^2/N_{\rm CPU}
              +s_{\rm CUDA}^2/N_{\rm CUDA}}}.
\]

### 4.2 两样本 Kolmogorov–Smirnov 检验

经验 KS 距离：

\[
D_{\rm KS}=\sup_x
|F_{\rm CPU}(x)-F_{\rm CUDA}(x)|.
\]

本样本量 \(N_{\rm CPU}=N_{\rm CUDA}=500\) 时，近似 95% 临界值为：

\[
D_{0.05}=1.36
\sqrt{\frac{500+500}{500\times500}}
=0.0860.
\]

KS 检验比较完整经验 CDF，而不只比较均值。

### 4.3 其他检验

- Welch test：检查均值差；
- Brown–Forsythe test：以 median 为中心检查 scale/variance 差；
- 5000 次 bootstrap：估计相对均值差的 95% 区间；
- CUDA/CPU 方差比：作为 shower fluctuation 诊断。

没有拒绝同分布假设不等于证明分布完全相同。若要声称 ±1% 等价，应要求
预定义等价区间内的置信区间整体落入 \([-1\%,+1\%]\)，而不只是
\(p>0.05\)。

## 5. 分布结果

| 观测量 | CPU mean ± SD | CUDA mean ± SD | 均值差 | z | KS / 临界值 | KS p |
|---|---:|---:|---:|---:|---:|---:|
| charged \(X_{\max}\) / g cm\(^{-2}\) | 320.82 ± 49.25 | 318.59 ± 49.61 | -0.696% | 0.71 | 0.072 / 0.086 | 0.150 |
| charged \(N_{\max}\) | 1298.87 ± 132.87 | 1295.55 ± 132.30 | -0.256% | 0.40 | 0.036 / 0.086 | 0.903 |
| charged \(\int N\,dX\) | 395055.9 ± 9856.6 | 395122.1 ± 9160.4 | +0.017% | 0.11 | 0.066 / 0.086 | 0.226 |
| photon \(\int N\,dX\) | 2325007.8 ± 78920.8 | 2333678.4 ± 71345.6 | +0.373% | 1.82 | 0.076 / 0.086 | 0.111 |
| ground EM weighted count | 648.67 ± 469.06 | 638.67 ± 400.30 | -1.541% | 0.36 | 0.032 / 0.086 | 0.960 |
| ground EM kinetic energy / GeV | 15.415 ± 16.053 | 15.011 ± 14.284 | -2.622% | 0.42 | 0.036 / 0.086 | 0.903 |
| \(dE/dX\) \(X_{\max}\) / g cm\(^{-2}\) | 309.67 ± 61.58 | 306.43 ± 57.76 | -1.045% | 0.86 | 0.048 / 0.086 | 0.613 |
| total deposited energy / GeV | 971.44 ± 15.86 | 971.85 ± 14.21 | +0.042% | 0.43 | 0.052 / 0.086 | 0.509 |

所有：

```text
KS p-value                  > 0.05
Welch mean p-value          > 0.05
Brown–Forsythe scale p-value > 0.05
```

其中最接近显著性的量是 photon 纵向积分：

```text
KS p                 0.111
Welch mean p         0.0687
Brown–Forsythe p     0.259
mean shift          +0.373%
```

仍未达到 5% 显著水平，且均值差小于 1%。

## 6. 重点观测量解释

### 6.1 Charged \(X_{\max}\)

```text
CPU mean             320.820 g/cm²
CUDA mean            318.587 g/cm²
mean shift           -0.696%
variance ratio       1.015
bootstrap 95%        [-2.599%, +1.202%]
KS                   0.072 < 0.086
```

观察到的均值差小于 1%，形状检验通过，方差几乎相同。但 bootstrap 区间尚未
完全落入 ±1%，所以可说“统计相容”，不应说“已经以 95% 置信度证明 ±1%
等价”。

### 6.2 Charged \(N_{\max}\)

```text
CPU mean             1298.870
CUDA mean            1295.546
mean shift           -0.256%
variance ratio       0.991
bootstrap 95%        [-1.527%, +1.031%]
KS                   0.036 < 0.086
```

均值、方差、median 和完整 CDF 均非常接近。这是“shower 最大粒子数分布
一致”的直接证据。

### 6.3 Charged 纵向积分

```text
mean shift           +0.0167%
bootstrap 95%        [-0.284%, +0.321%]
KS                   0.066 < 0.086
```

其 bootstrap 区间完整位于 ±1% 内，是当前粒子含量指标中最强的
1% 等价证据。

### 6.4 地面 EM 粒子数

```text
mean shift           -1.541%
z                    0.363
bootstrap 95%        [-9.28%, +7.16%]
KS p                 0.960
Brown–Forsythe p     0.582
```

虽然点估计超过 1%，但分布非常宽，当前差异远小于一倍标准误，完整 CDF 和
scale 检验也没有异常。这里的结论是“样本不足以解析 1% 尾部差异”，不是
“CUDA 已经低估 1.54%”。

## 7. 图与数据产物

分析目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
original_vs_cuda_electron_1TeV_500_current_v4/
feature_distribution_analysis
```

分布图：

![CPU/CUDA shower feature distributions](../../../corsika_validation_results/original_vs_cuda_electron_1TeV_500_current_v4/feature_distribution_analysis/shower_feature_distributions.png)

平均纵向发展：

![CPU/CUDA longitudinal means](../../../corsika_validation_results/original_vs_cuda_electron_1TeV_500_current_v4/feature_distribution_analysis/longitudinal_mean_comparison.png)

机器可读结果：

```text
selected_feature_summary.csv
selected_feature_summary.json
```

可重复生成：

```bash
cd /home/yuhanglu/21CMA/corsika8_gpu_refactor

/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python \
  validation/gpu_em/analyze_shower_feature_distributions.py \
  --ensemble-root \
    /home/yuhanglu/21CMA/corsika_validation_results/original_vs_cuda_electron_1TeV_500_current_v4 \
  --output-dir \
    /home/yuhanglu/21CMA/corsika_validation_results/original_vs_cuda_electron_1TeV_500_current_v4/feature_distribution_analysis
```

## 8. 下一步正式统计矩阵

当前 1 TeV 垂直电子结果只覆盖一个相空间点。建议按以下顺序扩展：

### 8.1 电磁初级

| 初级 | 能量 | zenith | 每后端建议样本 |
|---|---:|---:|---:|
| electron | 1 TeV | \(0^\circ\) | 已完成 500 |
| photon | 1 TeV | \(0^\circ\) | 500 |
| electron/photon | 1 PeV | \(0^\circ,45^\circ,80^\circ\) | 200/配置 |
| electron/photon | \(10^{17}\) eV | \(0^\circ,45^\circ,80^\circ\) | 50–100/配置 |
| electron/photon | \(10^{18}\) eV | \(0^\circ,45^\circ,80^\circ\) | 20–50/配置 |

每一配置将 CPU/CUDA 分别拆成多个 base-seed shards。seed 集合可以相同，
但统计时仍按独立两样本处理；另输出 paired-seed correlation 只作为调试。

### 8.2 cut 与 thinning

至少覆盖：

```text
EM cut       0.5, 5, 50 MeV
thinning     on/off
max weight   当前 production 值和一个更严格值
```

### 8.3 强子初级

21CMA 最终需要 proton/iron shower。除本报告指标外还应比较：

- μ、强子和 EM 地面 multiplicity；
- CPU specified-fallback 次数与能量；
- \(X_{\max}\)；
- 到达时间、横向分布和能谱；
- photohadronic/muon-pair 等稀有末态。

### 8.4 射电分布

射电不能只比较一个 replay event。应对独立 CPU/CUDA shower 比较：

- 30–80 MHz 与 50–350 MHz radiation energy；
- 峰值场强；
- pulse fluence；
- lateral distribution；
- 到达时间；
- CoREAS 和 ZHS 分别检验。

射电计算昂贵，可先做 100+100，再根据 bootstrap 区间决定是否扩到
200+200。

## 9. 验收用语

有限 ensemble 建议使用以下分级：

```text
发现显著差异
  KS/Welch/scale 检验显著，且 effect size 超过物理门限

统计相容
  未发现显著差异，但置信区间尚未完全进入等价区间

通过等价验收
  预定义观测量的置信区间完整落入等价区间，
  且分布/曲线/完整性门禁全部通过
```

当前 500+500 电子 shower：

```text
主体纵向 shower           统计相容，部分指标通过 ±1% 等价验收
charged integral          通过 ±1% 等价证据
Xmax / Nmax               分布检验通过，±1% CI 尚差少量统计量
稀疏地面尾部             统计相容，尚未达到 ±1% 精度
```
