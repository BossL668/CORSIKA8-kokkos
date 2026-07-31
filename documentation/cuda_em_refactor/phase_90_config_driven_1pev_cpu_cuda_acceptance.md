# 阶段 90：由现有 YAML 驱动的 1 PeV 原版 CPU/CUDA 预验收

## 1. 结论

已经用现有
`/home/yuhanglu/21CMA/python/config.yaml`
中的第一个 task 驱动原版 CPU 和新 CUDA 程序各运行 10 场独立 proton shower。

本次结果同时给出两个不同层次的结论：

1. **性能结论明确**：原版单核 CPU 每场中位数 264.55 s，CUDA 每场中位数
   6.99 s，中位数比值为 37.84 倍；按均值/累计事件时间计算为 30.03 倍。
2. **物理结论仍是预验收**：九个核心标量的 KS 检验都没有拒绝同分布，
   bootstrap 均值偏移区间都包含零，九条归一化/纵向曲线的逐 bin 统计检验都
   通过；但 10+10 的 shower 波动太大，不能证明核心均值已达到 1% 等价。

因此：

```text
没有发现 CPU/CUDA 分布不一致的统计证据；
但严格“关键均值偏差 <= 1%”验收尚未通过。
```

不能把 `passed=false` 写成“CUDA 物理错误”，也不能把“统计上不显著”写成
“已经证明完全一致”。

## 2. YAML 翻译是否忠实

适配器：

```text
validation/gpu_em/run_config_acceptance.py
```

从 task args 读取：

- primary PDG：2212；
- 能量：\(10^6\) GeV，即 1 PeV；
- zenith：27°；
- azimuth：180°；
- shower core：\((0,0)\) m；
- `emcut=0.5 MeV`；
- `hadcut=mucut=taucut=0.3 GeV`；
- task 原始 `emthin=1e-6`；
- 未显式给 maximum weight，因此保留自动 Kobal 模式。

配置顶层 `energy_range=[1e9,1e9]` 与 task 的 `-E 1e6` 不同。适配器 fail-closed
地记录警告，并以实际传给 `c8_air_shower` 的 task args 为权威，不猜测单位或
偷偷改写能量。

这里还有一个必须单独审计的边界：1 PeV 时自动
`maximum_weight = 0.5 * 1e-6 * 1e6 = 0.5`，而初始权重为 1。原版
`EMThinning` 的 `parentWeight >= maxWeight` 保护会使 thinning 无法启动。
所以 task 写法本身合法、参数也被原样保留，但该精确组合实际是未薄化运行。
适配器现在同时输出 `effective_maximum_weight=0.5` 和
`thinning_can_activate_from_unit_weight=false`，并给出显式警告。

## 3. 本次明确的开发档改动

为了在可接受时间内得到多 seed 证据，本次命令显式使用：

```text
--development-emthin 1e-4
--shower-only
```

因此本次 manifest 正确标记：

```text
production_equivalent: false
effective_em_thinning: 1e-4
effective_ring: 0
effective_antenna_file: /dev/null
```

这意味着：

- 不能把它冒充 task 原始 `emthin=1e-6` 的正式结果；
- 不能用于 CoREAS/ZHS 射电验收；
- 可以用于当前 shower transport、强子瓶颈和初步分布一致性判断。

1 PeV、`emthin=1e-6` 仍是正式最高质量档，但应准确称为自动权重下的未薄化
极限；`1e-5` 单 shower 预验收已在阶段 89 记录。

## 4. 可复现输入

输出目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
config_adapter_current_hadronic_proton_1PeV_emthin1e-4_showeronly_10_v1
```

主要证据：

- `config_acceptance_manifest.json`
- `run_manifest.json`
- `comparison.json`
- `proposal_shard_000` 到 `proposal_shard_003`
- `cuda/gpu_em/summary.yaml`
- `cuda/simulation_timing/summary.yaml`

原版二进制：

```text
SHA256:
133daa7ee0a4de4ba7355df5facb44e9a09e23bb5fff8b9549e7b6da546daeb9
```

CUDA 二进制：

```text
SHA256:
7f31b49c0d80510dcdd59a6df30db2e8a47824086b2eb33ce856995e44611fcc
```

GPU 物理表：

```text
SHA256:
14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521
```

原版和 CUDA 使用独立 seed：

```text
proposal seed: 61001（四 shard 依次使用 61001–61004）
CUDA seed:     71001
```

这是独立 ensemble 检验，不是同 seed 单事例 replay。

## 5. 性能

### 5.1 逐事件内部 wall time

原版 CPU，秒：

```text
288.578, 266.167, 244.894, 281.722, 269.079,
208.863, 280.570, 232.748, 262.939, 244.615
```

CUDA，秒：

```text
6.468, 8.651, 7.118, 12.133, 18.982,
6.719, 6.864, 6.114, 5.412, 7.452
```

| 指标 | 原版 CPU | CUDA | CPU/CUDA |
|---|---:|---:|---:|
| mean | 258.018 s | 8.591 s | 30.03× |
| median | 264.553 s | 6.991 s | 37.84× |
| standard deviation | 24.998 s | 4.099 s | — |
| minimum | 208.863 s | 5.412 s | — |
| maximum | 288.578 s | 18.982 s | — |
| 10 场内部时间总和 | 2,580.175 s | 85.911 s | 30.03× |

原版数据集使用四个独立 shard 并行，只是缩短收集 ensemble 的外部等待时间；
每场原版 shower 仍是单核程序。外部最慢 CPU shard 为 806.5 s，CUDA 10 场外部
wall 为 93.3 s。这个 8.64× 是“4 CPU shards 对一个 GPU 作业”的吞吐比较，
不能替代上表的单事件比较。

CUDA 第 5 场耗时 18.98 s，是明显的 shower-size 尾部。这也说明只报一场或只报
最优值会误导，应该同时报告均值、中位数和分布。

## 6. 强子调度

10 场 CUDA 的合计：

| 指标 | 数值 |
|---|---:|
| Hybrid total | 85,911.42 ms |
| FLUKA process-pool wall | 6,920.69 ms |
| process-pool wall fraction | 8.06% |
| FLUKA interactions | 172,690 |
| 四 worker CPU 末态时间总和 | 23,140.42 ms |
| worker 累计末态 max/min | 1.0323 |
| 正常 cost-triggered flush | 821 |
| 平均 interactions/cost flush | 205.47 |
| 预测 worker max/min 均值 | 1.0161 |
| 预测 worker max/min 最差 | 1.0385 |
| 实际 worker max/min 均值 | 1.2932 |
| 实际 worker max/min 最差 | 1.9849 |

强子进程池占比 8.06%，低于当前预设的 10% GPU 强子原型门槛。因此现阶段：

- 保留分类队列、四进程 FLUKA、cost-trigger 和 bulk response；
- 不启动重写 FLUKA/SIBYLL 物理内核的 GPU 项目；
- 继续观察 `emthin=1e-6` 完整射电配置中的分项占比。

## 7. 九个核心标量

相对偏差取绝对值；z score 使用两个独立样本均值差除以合并标准误。

| 量 | CPU mean | CUDA mean | 相对偏差 | \(|z|\) | KS / 95% 临界值 |
|---|---:|---:|---:|---:|---:|
| charged \(X_\max\) (g/cm²) | 564.64 | 589.15 | 4.34% | 0.67 | 0.40 / 0.608 |
| charged max | 627,830 | 671,284 | 6.92% | 1.46 | 0.40 / 0.608 |
| charged integral | \(2.5712\times10^8\) | \(2.5426\times10^8\) | 1.11% | 0.23 | 0.30 / 0.608 |
| photon integral | \(1.4195\times10^9\) | \(1.3949\times10^9\) | 1.73% | 0.30 | 0.30 / 0.608 |
| deposited energy (GeV) | 700,279 | 688,755 | 1.65% | 0.36 | 0.20 / 0.608 |
| deposit \(X_\max\) (g/cm²) | 533.65 | 589.96 | 10.55% | 1.04 | 0.40 / 0.608 |
| ground EM weighted count | \(2.3943\times10^6\) | \(2.7991\times10^6\) | 16.91% | 0.97 | 0.40 / 0.608 |
| ground EM kinetic E (GeV) | 132,630 | 153,097 | 15.43% | 0.70 | 0.30 / 0.608 |
| energy closure fraction | 0.83307 | 0.84204 | 1.08% | 0.73 | 0.20 / 0.608 |

所有 KS 值都低于 95% 临界值 0.608。所有九个 bootstrap signed mean shift 的
95% 区间都跨过零。没有一个量达到 3σ 分布差异证据。

但是九个量都没有同时满足“观测均值偏差 <=1%”和“样本精度足以检验 1%”。
它们被统一归类为：

```text
relative_threshold_failed_but_statistically_inconclusive
```

特别是 ground EM count/energy 的相对均值偏差看起来有 15–17%，但 \(|z|<1\)，
且 bootstrap 区间很宽，不能从 10 场判断这是算法偏差还是 shower 波动。

## 8. 曲线

九条曲线逐 active bin 的统计通过率全部为 100%：

- energy deposit；
- charged/electron/positron/photon/EM longitudinal profile；
- ground energy fraction；
- ground radial fraction；
- ground time-residual fraction。

三条归一化地面曲线的 relative L1 分别为：

```text
energy fraction:       0.39%
radial fraction:       0.64%
time-residual fraction:0.99%
```

六条纵向曲线的 relative L1 约为 10.8–11.4%。它们的 active-bin RMS z score
只有 0.74–0.84，逐 bin 统计检验全部通过；由于 10 场平均 shower 的 \(X_\max\)
偏移和样本波动较大，不能把 10% L1 直接解释为物理模型差异。

## 9. 为什么 10 场无法证明 1%

按本次 pilot 方差、独立样本和 3σ 精度做简单 \(1/\sqrt N\) 外推，某些高方差量
要上千甚至上万场/arm 才能把均值不确定度压到 1%。例如：

- energy closure：约 197 场/arm；
- charged max/integral：约 2,000 场/arm；
- charged \(X_\max\)：约 3,800 场/arm；
- ground EM count：约 27,000 场/arm；
- ground EM energy：约 44,000 场/arm。

这些只是 pilot 外推，不是固定生产配额。更合理的下一步是：

1. 先把 1% 验收量限定为物理上稳定、定义清晰的核心 observable；
2. 对高方差尾量同时使用 KS、Wasserstein、bootstrap 和预先规定的效应量；
3. 使用分层能量/天顶角设计，而不是为每个尾量盲目生成数万场；
4. 射电必须单独在原始天线文件和 `emthin=1e-6` 下验收。

## 10. 下一步

1. 冻结当前强子调度和 v3 IPC，不再仅凭单 shower 调参；
2. 用 `emthin=1e-5` 扩展一个较小的独立 ensemble，检查 thinning 变化是否引入
   系统趋势；
3. 至少重复一个原始 `emthin=1e-6`、81 天线完整事件，记录 CoREAS/ZHS 与
   shower transport 分项时间；
4. 合并已有 1 TeV 500+500 ensemble、1 PeV 10+10 pilot 和射电 ensemble，
   制定分 observable 的最终验收门限；
5. 只有在正式高质量配置中强子 wall 稳定超过 10%，才重新评估强子 GPU 化。
