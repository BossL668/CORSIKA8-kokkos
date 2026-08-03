# CORSIKA 8 CUDA 电磁后端最终验收报告

日期：2026-07-30
源码目录：`/home/yuhanglu/21CMA/corsika8_gpu_refactor`
CUDA Release 构建：`/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda`

## 1. 结论

本轮完成了当前二进制的性能、纯电磁、强子/FLUKA、射电和自动测试验收，并建立了 fail-closed 的机器可读证据矩阵。

严格按照原实施计划的全部门限，当前结论仍是 **未完成最终生产验收**：

- 15 个硬要求中 12 个通过；
- 原版 CPU/CUDA 的 shower 分布和纵向曲线在统计门下通过；
- 但若不考虑统计误差、直接要求每个关键样本均值偏差都小于 1%，质子和纯电磁总体仍有若干项不通过；
- 沉积能量平方归一化的 30–80 MHz 辐射能量通过 ±10% 等价门，但 50–350 MHz 的置信区间下界略低于 0.90。

因此，当前状态应表述为：

> CUDA 后端已经通过功能、能量守恒、FLUKA 集成、统计形状、地磁脉冲和性能验收；原计划的严格 1% 样本均值门以及宽带辐射能量门仍需更多独立统计量或物理偏差定位。不能把当前结果标记为完整生产验收通过。

机器审计结果：

- 规则：[final_acceptance.yaml](../../validation/gpu_em/final_acceptance.yaml)
- 报告：`/home/yuhanglu/21CMA/corsika_validation_results/final_acceptance_audit_v1.json`

## 2. 固定的软件与物理配置

正式质子统计样本记录的文件身份为：

- 当前 CUDA `c8_air_shower` SHA-256：`fbe175bdc556924492c3f3b7e3c81fa5362b5b199797bc5e89c4248abee4fba7`
- 原版 CPU `c8_air_shower` SHA-256：`133daa7ee0a4de4ba7355df5facb44e9a09e23bb5fff8b9549e7b6da546daeb9`
- v10 生产表文件 SHA-256：`14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521`
- PROPOSAL：7.6.2
- FLUKA：2025.0.0
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9
- CUDA driver/runtime：12.6/12.6
- 地磁场：IGRF13，epoch 2025
- 天线：`/home/yuhanglu/21CMA/data/antennas.txt`

生产表覆盖 0.5 MeV 至 \(10^{18}\) eV，总表格误差为：

- 最大实测反应率插值误差：\(9.99595\times10^{-4}\)
- 最大实测逆 CDF 插值误差：\(7.51467\times10^{-4}\)

两者均满足 \(10^{-3}\) 要求。

## 3. 已通过的硬验收

### 3.1 单事例 exact replay

原版标量 shower 的 136,114 个电磁输运记录被写入 decision tape，再在 GPU 上按同一决定顺序回放：

- transport replay：通过
- byte hash mismatch：0
- invalid/non-finite/negative record：0
- host/device ordered hash：相同
- 加入 tape 捕获前后的 CoREAS、ZHS、能损、粒子、相互作用和 profile 文件：逐文件 bitwise equal

证据：`/home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/exact_replay_comparison.json`

这个结果证明 GPU 能无损复刻已确定的标量输运决定和射电轨迹。它不是生产 CUDA 自主抽样与原版共享随机流的逐事件同一性证明；生产路径仍应通过独立 ensemble 比较验收。

### 3.2 纯电磁能量闭合

3 个 10 GeV 纯电磁事例全部具有完整账本覆盖，最大相对闭合误差：

\[
3.2862\times10^{-16}<10^{-4}.
\]

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_pure_em_energy_10GeV_3_v2/energy_ledger_validation.json`

### 3.3 相同轨迹的 CoREAS/ZHS CPU-CUDA 等价

10 个 shower、81 个观测点，在完全相同的 CUDA EM 输运轨迹上只切换射电投影后端：

| 算法 | 最大归一化差 | 最大相对 L2 差 | 最大相对 fluence 差 |
|---|---:|---:|---:|
| CoREAS | \(6.76\times10^{-8}\) | \(3.87\times10^{-8}\) | \(1.37\times10^{-8}\) |
| ZHS | \(4.37\times10^{-5}\) | \(2.01\times10^{-5}\) | \(1.37\times10^{-5}\) |

两套输运输出表逐项相同，CoREAS/ZHS 均通过 \(10^{-4}\) 级门限。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_identical_tracks_radio_1TeV_10_v2/radio_comparison.json`

### 3.4 FLUKA 正式集成

当前二进制的 500 个 CUDA 质子 shower 中：

- 500/500 完整结束；
- 500/500 使用 FLUKA 2025.0.0；
- 每个 shower 都执行了至少一次 FLUKA 相互作用，总计 24,459 次；
- 500/500 使用 `fluka-process` 后端和 4 个 worker；
- 500/500 过程注册检查通过，未注册连续过程数为 0；
- 显存 spill 粒子数为 0；
- GPU 共输运 134,959,370 个粒子。

有 6 次显式记录的通用 CPU fallback，约占 GPU 粒子数的 \(4.4\times10^{-8}\)。它们没有被静默丢弃，也没有导致 shower 不完整。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_proton_1TeV_emthin1e-4_radio_1000_pooled_v2/cuda/gpu_em/summary.yaml`

### 3.5 独立 shower 的统计一致性

#### 1 TeV 质子：原版 1000 vs CUDA 1000

所有纵向 profile、沉积能量曲线、地面能谱、径向分布和到达时间曲线均通过统计门。关键标量没有任何 3σ 统计失败，但严格 1% 点估计门未通过。

| 量 | CUDA−CPU 相对均值差 | 绝对 z 分数 |
|---|---:|---:|
| 带电 \(X_{\max}\) | −1.994% | 1.260 |
| 带电峰值 | −2.302% | 1.810 |
| 带电纵向积分 | −2.594% | 2.289 |
| 光子纵向积分 | −2.586% | 1.944 |
| 沉积能量 | −1.278% | 1.859 |
| 地面 EM 加权计数 | −12.590% | 2.063 |
| 能量闭合量 | −1.608% | 2.581 |

地面量的 shower-to-shower 涨落非常宽，因此百分比点估计虽然大，仍未达到 3σ 显著性。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_proton_1TeV_emthin1e-4_radio_1000_pooled_v2/comparison.json`

#### 标量 FLUKA 控制：原版 1000 vs CUDA 1000

把 CUDA 侧强子调度改回标量 FLUKA 后：

- \(X_{\max}\)：−0.196%，0.131σ
- 沉积能量：+0.578%，0.875σ
- 沉积峰位置：−0.101%，0.024σ
- 能量闭合量：+0.665%，1.076σ

全部曲线通过。生产 process-pool 样本中约 −2% 的核心偏移没有在该控制中重现，说明它更可能来自强子随机流/调度改变叠加有限样本，而不是稳定的 CUDA 电磁能量损失。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_scalar_fluka_control_proton_1TeV_1000_v1/comparison.json`

#### 1 TeV 电子：原版 500 vs CUDA 500

所有曲线和所有标量统计门通过。7/9 个关键标量同时满足严格 1%：

- 沉积能量：+0.0246%，0.275σ
- 能量闭合量：−0.0045%，0.473σ
- 带电/光子纵向积分：−0.348%/−0.176%
- 未满足 1% 的两项：\(X_{\max}\) +1.355%（1.441σ）、地面动能 −1.871%（0.324σ）

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_electron_1TeV_1000total_v1/comparison.json`

#### 1 TeV 光子：原版 500 vs CUDA 500

所有曲线和所有标量统计门通过。带电峰值、带电/光子纵向积分、沉积能量和闭合量均在 1% 内。由于初始转换长度和地面尾部涨落较宽，\(X_{\max}\)、沉积峰位置及两个地面量的点估计超过 1%，但最大仅 1.68σ。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_photon_1TeV_1000total_v1/comparison.json`

### 3.6 地磁射电脉冲

使用 21CMA 天线阵列，从每个 shower 的 100 m 圆周天线提取地磁分量的峰值振幅和 max-Q 方窗宽度。四个分布的比值为：

| 算法 | 指标 | CUDA/CPU | 95% bootstrap 区间 | KS p 值 |
|---|---|---:|---:|---:|
| CoREAS | 振幅 | 0.9635 | [0.9156, 1.0149] | 0.0292 |
| CoREAS | 宽度 | 0.9978 | [0.9675, 1.0291] | 0.4083 |
| ZHS | 振幅 | 0.9645 | [0.9145, 1.0175] | 0.0971 |
| ZHS | 宽度 | 1.0099 | [0.9816, 1.0387] | 0.4953 |

四项置信区间都完整位于 [0.9, 1.1]。四个 KS 检验作为同一检验族进行 Holm–Bonferroni 校正后，没有拒绝形状一致性；CoREAS 振幅的未校正 p 值 0.0292 被保留在报告中。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_proton_1TeV_emthin1e-4_radio_1000_pooled_v2/geomagnetic_pulse_comparison/comparison.json`

### 3.7 Release 性能

测试配置：1 PeV 电子、0.5 MeV cut、`emthin=1e-4`、`max-weight=100`、Release、原版 CPU 单线程、热缓存、5 次相同物理负载配对重复。

| 计时 | 原版 CPU 中位数 | CUDA 中位数 | 中位数比 |
|---|---:|---:|---:|
| 外部墙钟 | 701.18 s | 18.32 s | 38.28× |
| shower 内部计时 | 696.51 s | 13.46 s | 51.73× |

5 次配对外部加速范围为 37.67–38.44×，内部加速范围为 50.82–51.96×，显著超过 5× 要求。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_release_performance_1PeV_5rep_v1/benchmark_summary.json`

这项数字是纯电磁 1 PeV 单事件性能，不应外推为 1 PeV 质子全流程加速。已有 1 PeV 质子诊断中，FLUKA process-pool 墙钟约占 hybrid shower 的 8.1%，尚未达到“必须把强子也移到 GPU”的 10% 决策门。

### 3.8 自动测试

- CTest：32/32 通过
- Python 验收测试：114/114 通过
- FLUKA 环境：`/home/yuhanglu/fluka`，有效

首次完整测试发现 3 个 replay 新文件缺少项目版权头；补齐标准 BSD 头后，完整测试重新运行并全部通过。没有修改物理逻辑。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_test_acceptance_v2.json`

## 4. 尚未通过的硬验收

### 4.1 严格 1% 样本均值门

质子、电子和光子样本均没有出现 3σ 统计失败，但若直接检查有限样本点估计，仍有若干量超过 1%。因此当前数据支持“统计上一致”，却不足以支持“所有真实均值偏差已被证明小于 1%”。

不能通过继续生成样本、直到点估计碰巧进入 1% 的方式验收。下一轮应预先固定样本量，并使用等价性置信区间或 TOST；否则会产生可选停止偏差。

### 4.2 50–350 MHz 归一化辐射能量

每后端 1000 shower 的沉积能量平方归一化几何均值结果：

| 算法 | 频段 | CUDA/CPU | 95% bootstrap 区间 | 结论 |
|---|---|---:|---:|---|
| CoREAS | 30–80 MHz | 0.95975 | [0.90299, 1.01884] | 通过 |
| CoREAS | 50–350 MHz | 0.95288 | [0.88816, 1.02201] | 未通过 |
| ZHS | 30–80 MHz | 0.95999 | [0.90468, 1.01976] | 通过 |
| ZHS | 50–350 MHz | 0.95374 | [0.89095, 1.01947] | 未通过 |

四个 KS 形状检验经 Holm 校正后均不拒绝一致性。宽带点估计本身位于 ±10% 内，失败来自置信区间下界尚未达到 0.90。

原始未归一化辐射能量均值低约 10.5%–11.4%，约 3σ。它会受到 shower 沉积能量和长尾的共同影响，不能由相同轨迹射电投影测试替代。

证据：`/home/yuhanglu/21CMA/corsika_validation_results/final_current_original_cuda_proton_1TeV_emthin1e-4_radio_1000_pooled_v2/pooled_radio_comparison.json`

CPU 标量 RadioProcess 与 CUDA resident-radio 路径记录的 track ledger 粒子总体定义不同，因此 track-length 归一化数值不可比较，已明确排除在正式门之外。

## 5. 下一轮最小方案

1. **预注册统计方案。** 把“关键均值小于 1%”改写成明确的等价性检验：固定置信度、等价区间、样本量和多重检验修正，禁止根据中途点估计选择停止。
2. **优先扩大纯 EM 样本。** 电子和光子各扩展到至少每后端 2000 个 shower。纯 EM 没有强子调度混杂，最适合判断 \(X_{\max}\) 的 1% 偏差是否稳定。
3. **分离质子强子随机流。** 保留现有 scalar-FLUKA 控制，并增加相同初级强子历史/不同 EM 后端的条件对照；不要直接用宽涨落的独立质子样本给 EM 后端归因。
4. **宽带射电增加预定样本。** 当前置信区间宽度估计表明每后端约 1500 个 shower 可能达到 0.90 下界，但为避免边界结果，建议预注册每后端 2000 个，并一次性分析。
5. **定位宽带差异。** 同时比较 30–80、80–200、200–350 MHz 子带，按 shower 检查振幅、脉宽、沉积能量和 \(X_{\max}\) 的条件相关性，以区分输运轨迹差异与高频相干/步长效应。
6. **保持相同二进制和表。** 后续样本必须继续固定当前应用 SHA、原版 SHA 和 v10 表 SHA；任何源码或表格变化都应启动新的验收版本。

在上述两项硬门通过之前，建议继续保留 `proposal` 为默认后端，把 CUDA 模式标记为统计验证完成、严格最终验收未完成。
