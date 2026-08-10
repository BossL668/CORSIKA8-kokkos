# Phase 104：beta4 100 TeV proton CPU500/CUDA500 原始数据验收

## 1. 数据和配置

本阶段在服务器上直接读取两套原始输出，完成了 beta2 scalar CPU 500 例与 beta4 CUDA 500 例的正式验收。两套样本均为 $10^5$ GeV 质子、垂直入射、`emthin=1e-6`、`emcut=0.5 MeV`、IGRF14 2027；强子、$\mu$ 和 $\tau$ cut 均为 0.3 GeV。beta4 使用 GPU table tolerance $5\times10^{-4}$，GPU EM、GPU CoREAS/ZHS 和 FLUKA process pool 均开启。CPU 和 CUDA 使用互不重叠的独立随机种子集合，因此本测试检验统计分布，而不是逐事件相等。

服务器上的权威结果目录为：

```text
/data/yhlu/CorsikaData/corsika_validation_results/
  final_beta2cpu500_beta4cuda500_proton_100TeV_vertical_emthin1e-6_igrf14_2027_server_raw_v1
```

D 盘仅同步了约 44 MB 的最终 JSON、CSV 和图片；CPU/CUDA shower 原始数据保留在服务器。GPU 原始目录的传输前后均为 1,602 个文件、4,587,122,696 字节，`rsync` dry-run 差异为零。fail-closed finalizer 核对了两个精确种子集合、物理参数、天线布局、可执行文件和表格哈希、时序闭合以及 CUDA 完整性门禁。最终状态为 `complete`，9 个分析步骤全部成功，未报告分析错误。

## 2. Shower profile 结果

纵向 total-EM、$e^-+e^+$、photon、hadron、muon、charged 和 energy-deposit 曲线全部通过逐 bin 统计门禁，active-bin pass fraction 均为 1.0。代表性的 relative $L_1$ 差异为：total-EM 2.87%、hadron 2.07%、muon 1.27%、energy deposit 2.91%；相应 active-bin RMS z-score 分别为 1.30、0.89、0.70 和 1.30。muon-production parent 在按每个 shower 的 charged $X_{\max}$ 对齐后，主导的 pion、kaon、hadron、muon 和 all-parent 曲线也没有错位；heavy/photon parent 的大百分比波动来自接近零的绝对计数。

严格的 1% 标量总门禁没有通过，但未通过项均被分类为 `relative_threshold_failed_but_statistically_inconclusive`，而不是统计不一致。例如：

- charged $X_{\max}$：CPU 503.46、CUDA 513.81 g cm$^{-2}$，均值偏移 +2.06%，$z=1.74$，KS $p=0.129$，bootstrap 95% 区间为 $[-0.29\%,+4.38\%]$；
- charged maximum：均值偏移 $-0.81\%$，通过 1% 门禁；
- charged integral：均值偏移 $-2.13\%$，95% 区间为 $[-4.50\%,+0.20\%]$；
- deposited energy sum：均值偏移 $-1.91\%$，95% 区间为 $[-3.95\%,+0.23\%]$；
- ground EM kinetic energy：均值偏移 $+6.48\%$，但分布很宽，$z=1.54$、KS $p=0.095$，95% 区间为 $[-1.55\%,+15.34\%]$。

需要保留一个诊断警告：charged-profile $X_{\max}$ 的 CPU/CUDA 标准差约为 90.1/99.0 g cm$^{-2}$，Brown--Forsythe scale test 给出未经多重检验校正的 $p=0.030$。这可能是多个诊断量中的统计涨落，也可能提示 CUDA 样本的 $X_{\max}$ 分布稍宽；当前结果不足以二选一，后续应使用新的独立种子样本复核。

## 3. $X_{\max}$ 对齐诊断

固定深度的 ensemble-mean $e^-+e^+$ 峰值比 CPU 低约 3.4%，但其中包含两套独立样本 $X_{\max}$ 均值和宽度不同所造成的展宽效应。按每个 shower 自身的 $e^-+e^+$ $X_{\max}$ 对齐后，峰值相对差异缩小为 $-1.12\%$（$z=-1.02$）。total-EM CUDA/CPU mean ratio 在 $\Delta X=0$ 和 50 g cm$^{-2}$ 分别为 0.9894 和 0.9906，95% bootstrap 区间均覆盖 1；归一化 EM 尾积分的差异仅为 $-0.004\%$，95% 区间为 $[-0.128\%,+0.121\%]$。因此没有看到 $X_{\max}$ 之后 shower 形状的显著算法偏差。

## 4. 射电结果

在 $r_\perp=100$ m、每个 shower 合并八个方位天线后：

- CoREAS amplitude 的 CUDA/CPU 比为 0.955，95% CI 为 [0.901, 1.010]，KS $p=0.370$；
- CoREAS width 的比为 1.014，95% CI 为 [0.999, 1.029]，KS $p=0.339$；
- ZHS amplitude 的比为 0.946，95% CI 为 [0.892, 1.004]，KS $p=0.150$；
- ZHS width 的比为 1.010，95% CI 为 [0.988, 1.034]，KS $p=0.694$。

CoREAS amplitude 通过 $\pm10\%$ 等价门禁。ZHS amplitude 的区间下限比 0.900 低约 0.008，因此没有通过严格等价门禁，但 KS 检验没有发现形状不一致。width 的分布和区间均通过物理统计条件；自动化总门禁仍将它们标为失败，是因为 robust width filter 后只保留了 CoREAS 496/497 和 ZHS 490/492 个有效 shower，少于配置中机械要求的每端 500 个，而有效覆盖率仍为 98.0%--99.4%，高于预设的 80% coverage 门槛。

$r_\perp=1$--600 m 的 40 个 CoREAS/ZHS amplitude/width pointwise bootstrap 区间均覆盖 1，曲线上没有发现随距离单调增长的系统偏差。进一步按 $X_{\max}$ 和 ground EM 规模做诊断性调整后，100 m 的 CoREAS/ZHS amplitude 比分别从 0.955/0.946 变为 0.996/0.989；这说明当前约 5% 的原始振幅差主要与两套独立 shower 样本的发展深度和 EM 规模差异相关，但该回归诊断不能替代同轨迹 radio replay。

## 5. 时间结果

beta2 scalar CPU 每事件平均/中位时间为 8,784.96/8,924.18 s，即 2.44/2.48 h；beta4 CUDA 为 65.45/66.40 s。均值和中位数对应的生产 turnaround ratio 分别为 134.2 和 134.4。500 个 CPU 事件累计约 50.84 CPU-core-days，500 个 CUDA 事件累计约 9.09 h。两套任务运行在不同主机，因此这些数值只能描述本次生产周转时间，不能作为同机受控硬件 speed-up。

## 6. 结论

本次原始数据 500 vs 500 验收的综合判定为：

1. 数据、配置、来源和 CUDA 计算完整性审计通过；
2. shower 各组分纵向曲线和 $X_{\max}$ 对齐后的 EM 尾部统计门禁通过；
3. 没有检出明确的 CPU/CUDA 分布不一致，但当前独立样本还不能证明所有关键标量达到严格 1% 等价；
4. 射电主曲线在统计上总体相容，但完整自动门禁没有通过，主要原因是 ZHS amplitude 的 95% 等价区间下限略越界，以及 width filter 后的机械样本数门槛；
5. 在新的高统计量样本或同轨迹 replay 解决上述两项前，不应宣称 beta4 已满足“所有 observable 均严格等价”的最终生产验收。
