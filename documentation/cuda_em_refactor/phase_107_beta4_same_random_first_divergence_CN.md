# Phase 107：CPU/CUDA 相同随机数与第一次物理分叉测试

## 1. 问题与结论

本阶段回答两个不同层次的问题：

1. 将同一串原始 uniform random numbers 按顺序交给当前 CPU 和 CUDA 实现，能否生成同一棵 shower tree？
2. 若先固定相同的物理语义，即 `process + component + conditional loss quantile`，两边是否执行相同的物理过程并产生一致的末态？

测试结论是：

- **当前实现仅共享原始随机数时，不能生成同一棵 shower tree。** 第一个电子和正电子测试顶点已经在能量损失处分叉，第一个光子顶点已经可能在过程或靶组分处分叉。
- 分叉的首要原因不是截面分布错误，而是 CPU/CUDA 的随机数消费契约和过程列顺序不同。
- 将 CPU rates 按 CUDA 的规范列顺序排列后，11,462 个有效顶点的过程/组分选择 mismatch 为 0。
- 再将 CPU 选择转换为相同的 `process/component/conditional quantile` 后，11,461/11,462 个 inverse-CDF 点满足 \(5\times10^{-4}\)；唯一超限 photon 点为 \(5.105\times10^{-4}\)。
- 当过程、\(v\) 和末态 random draws 已经显式固定时，主要 GPU 末态与 CPU PROPOSAL oracle 均通过；但 Epair、Molière 和表插值仍是高精度近似，不是逐位相同。

因此，“相同随机数”必须进一步区分为**相同原始 draw 序列**和**相同物理决定**。当前后端只能在第二种意义下得到高度一致的过程结果；要得到完全相同的原 CPU shower tree，必须使用语义 decision tape，单独记录原始 RNG 数值仍然不够。

## 2. 非侵入式实现

新增验证程序：

```text
validation/gpu_em/compare_same_random_processes.cpp
```

对应构建 target：

```text
gpu_em_same_random_process_oracle
```

该 target 具有以下约束：

- `EXCLUDE_FROM_ALL`，普通 `cmake --build` 不会构建；
- 不安装到生产 `bin`；
- 不修改 `c8_air_shower`、PROPOSAL 或 CUDA cascade 的默认行为；
- 只调用已有 `selectInteractionsForValidation()` 和 `reselectLeptonInteractionsAtVertexForValidation()`；
- 工作区显存比例固定为 1%。

因此本测试没有在生产路径中增加 RNG 分支、文件 I/O 或同步点。

## 3. 为什么同一个原始随机数仍会分叉

### 3.1 原 CPU PROPOSAL 契约

原 CPU `Interaction::SampleLoss(E, rates, u)` 只读取一个 uniform：

1. 计算 \(R=\sum_i R_i\)；
2. 用 \(uR\) 在 `Rates()` 的原始顺序中选择 process/component 列；
3. 将该随机点在选中列中剩余的 rate 直接交给 `CalculateStochasticLoss()`；
4. 同一个 \(u\) 因而同时决定过程、靶组分和损失 \(v\)。

对于选中列 \(i\)，其条件分位点是

\[
q_i=\frac{C_i-uR}{R_i},
\]

其中 \(C_i\) 是该列末端的累计 rate。

### 3.2 当前 CUDA 契约

CUDA 使用 history-keyed Philox，并将决定拆成独立地址：

1. `InteractionColumnDrawId` 产生 \(u_{\rm process}\)，选择 process/component；
2. `InteractionLossDrawId` 产生另一个 \(u_{\rm loss}\)；
3. 使用 \(v=F_i^{-1}(E,u_{\rm loss})\) 查询该列 inverse-CDF 表。

因此 CPU 每个顶点消耗一个联合 draw，而 CUDA 消耗两个语义独立的 draw。即使令 \(u_{\rm process}\) 与 CPU 的 \(u\) 数值相同，CUDA 的 \(u_{\rm loss}\) 也不是 CPU 从同一个 \(u\) 推导出的 \(q_i\)。

### 3.3 列顺序也不同

CPU 使用 PROPOSAL `Rates()` 的构造顺序。CUDA 表使用稳定、规范化的序列化列顺序。两种顺序包含相同的物理列和概率，但累计区间排列不同。

所以同一个 \(u\) 可能在 CPU 落入 Compton，在 CUDA 落入 photon pair；这不改变各自系综中的过程概率，却会立即改变单次 shower tree。

## 4. 测试配置

| 项目 | 配置 |
|---|---|
| 介质 | `air_dry_1_atm.yaml` |
| PROPOSAL | 7.6.2，interpolate=true |
| 表 cut | 0.4 MeV，对应用户 0.5 MeV EM cut 的标量优化 cut |
| 表精度目标 | rate/loss 均为 \(5\times10^{-4}\) |
| 表最高能量 | 10.5 GeV |
| 粒子 | electron、positron、photon |
| 请求样本 | 每种 4,096 |
| random seed | 2026081002 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

输出：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_same_random_process_oracle_v1/oracle.json
```

## 5. 第一次分叉结果

### 5.1 原始 draw 契约

| PID | 有效顶点 | process/component mismatch | 同列但 raw loss mismatch | 合计已分叉 |
|---:|---:|---:|---:|---:|
| electron 11 | 3,726 | 1,100 | 2,626 | 3,726（100%） |
| positron -11 | 3,726 | 1,057 | 2,504 | 3,561（95.6%） |
| photon 22 | 4,010 | 2,337 | 548 | 2,885（72.0%） |

未计入的低能输入由正常 ParticleCut 处理；三种粒子均没有 table fallback。

第一个 electron 顶点示例：

```text
E                         = 9.317441306 MeV
same process uniform      = 0.4660743765
CPU/GPU process           = ionization / ionization
CPU conditional quantile  = 0.8885810033
GPU independent loss draw = 0.3263314791
CPU v                     = 0.2604628125
GPU raw-contract v        = 0.0624047733
```

两边选择了相同过程，但因为 loss draw 的契约不同，第一步的次级粒子能量已经不同，后续 tree 不可能继续逐项相同。

第一个 photon 顶点还出现了相同过程但不同靶组分；在其他点会直接选择不同过程。这来自累计 rate 列顺序，而不是新增物理过程。

## 6. 规范列顺序对照

oracle 额外保留 CPU 直接计算的每一列 rate，但按照 CUDA 表中的列顺序重新排列，再使用完全相同的 process uniform 选择列。

结果：

| PID | canonical-order process/component mismatch |
|---:|---:|
| 11 | 0 / 3,726 |
| -11 | 0 / 3,726 |
| 22 | 0 / 4,010 |

这说明在本样本中：

- CPU 与 CUDA 包含的物理过程和靶组分集合一致；
- 表中各列 rate 的 \(5\times10^{-4}\) 插值误差没有把任何测试点推过过程选择边界；
- 原始模式的大量 process mismatch 由列的排列契约造成，而不是过程概率明显错误。

## 7. 语义 decision 对齐

接着不再使用 CUDA 独立的 loss draw，而是从 CPU 联合 draw 精确恢复：

```text
CPU process + CPU component + CPU conditional quantile
```

然后用该三元组查询 CUDA inverse-CDF 表。

| PID | 比较点 | 超过 \(5\times10^{-4}\) | 最大相对误差 |
|---:|---:|---:|---:|
| 11 | 3,726 | 0 | \(4.9614\times10^{-4}\) |
| -11 | 3,726 | 0 | \(4.8895\times10^{-4}\) |
| 22 | 4,010 | 1 | \(5.1051\times10^{-4}\) |

唯一轻微超限点为约 198.878 MeV 的 Compton 查询：

```text
CPU v          = 0.9076001329822617
CUDA-table v   = 0.9071367963631760
relative error = 5.105074385e-4
```

该点只比目标高约 2.1%，属于局部表插值精度问题，不是过程身份或末态拓扑错误；但在声明“所有点严格不超过 \(5\times10^{-4}\)”之前，应继续细化对应 Compton inverse-CDF 网格或在 tablegen 验收点中加入该坐标。

## 8. 逐过程末态测试

使用显式相同的 final-state/LPM/thinning uniforms，运行以下 CPU PROPOSAL 与 GPU oracle。10/10 逐过程测试通过，随后完整 `testGpu*` 回归 27/27 通过。

| 过程 | 主要结果 |
|---|---|
| photon pair / Compton / photoelectric | 1,409,955 项检查；4,097 个 interaction；最大次级能量相对误差 \(1.39\times10^{-17}\)，Compton/PROPOSAL 最大方向差 \(1.40\times10^{-14}\) |
| bremsstrahlung | 81,391 项检查；3,010 accepted、1,086 LPM suppressed、3 explicit fallback |
| ionization | 118,796 项检查；8,192 个 event |
| positron annihilation | 77,834 项检查；4,096 个 event |
| electron/positron pair | 122,528 项检查；1,024 个 PROPOSAL interaction；最大 normalized-rho error \(5.26914\times10^{-4}\) |
| photon-pair LPM | 450 个点；最大相对误差 \(8.38\times10^{-16}\) |
| brems LPM | 540 个点；最大相对误差 \(1.41\times10^{-15}\) |
| Epair LPM | 750 个点；最大相对误差 \(5.72\times10^{-11}\) |
| EM thinning | 8,192 个 host/device 精确案例和 200,000 个统计样本 |
| Molière | 相对当前 CPU `MoliereInterpol` 最大误差 \(9.83\times10^{-6}\) |

连续能损、球形大气、均匀磁场、photon/lepton wavefront 和相同轨迹射电测试也全部通过。完整回归耗时 47.01 s。

这些结果表明：当过程身份和所需 random draws 已经一一对应时，两边的物理规则、次级粒子数和能量方向结果总体一致；已知差异主要是表插值和少数数值近似，而不是缺失过程。

## 9. 能否长出完全相同的 shower tree？

### 9.1 当前 CPU 与当前 CUDA，只共享原始随机数

不能。本测试在第一个有效顶点已经观察到分叉，因此不需要继续演化整棵树就可以否定“完整相同”。

### 9.2 强制共享语义 decision tape

可以使**树的过程拓扑**相同，但 tape 必须至少记录：

- history/parent/child ordinal；
- interaction distance 或已经接受的顶点；
- process、component、条件 loss quantile 和最终 \(v\)；
- final-state、LPM、Molière 和 thinning 的每个语义 draw；
- 次级粒子的稳定排序。

若 tape 同时直接记录 CPU 的最终 \(v\) 和次级粒子状态，GPU 可以像 phase 106 一样严格回放同一棵树。此模式适合验证，但 CPU 已经完成了主要随机物理决定，不能代表生产加速性能。

### 9.3 让 CPU 与 GPU 自主运行仍产生相同 tree

需要建立新的共享契约，而不是沿用当前两套实现：

1. CPU 和 GPU 使用相同的规范列顺序；
2. CPU 也拆成独立 process draw 和 loss draw，或 GPU 完全复刻 PROPOSAL 的单 draw 联合映射；
3. 两边使用同一张 inverse-CDF 表或同一个直接函数；
4. 所有末态、LPM、散射和 thinning 使用相同 history-keyed draw 地址；
5. 固定浮点、FMA、累计和次级排序语义。

这有机会让“新 CPU reference”和 CUDA 生成相同 tree，但不会复刻公开原版 CPU 对既有 seed 的随机流。

## 10. 最终判断

本测试将“物理不一致”和“随机映射不同”区分开了：

- 当前相同 raw draws 不能产生同一 shower tree；
- 将列顺序统一后，过程/组分选择在 11,462 个点上完全一致；
- 将过程语义和条件分位点统一后，inverse-CDF 基本达到设定精度；
- 将末态 draws 统一后，主要物理过程的子粒子数、能量和方向 oracle 通过。

因此当前 beta4 的主要问题不是 CPU/GPU 实现了不同的物理过程，而是它们没有共享同一个“随机数到物理决定”的确定性映射。对于生产后端，继续使用大样本统计一致性是合理标准；对于严格逐事例验证，应使用扩展的语义 decision tape，而不是只记录原始随机数。
