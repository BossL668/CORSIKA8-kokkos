# Phase 114：PROPOSAL-native decision-tape replay 与首次分叉诊断

## 1. 目的与结论

本阶段针对以下问题做受控测试：如果 CPU、`c8emrt` CUDA 和
`proposal-native` CUDA 使用相同 seed 或相同物理决定，能否重放同一个
shower；此前 2000 例系综中看到的差别是否可能来自原生 PROPOSAL 表本身。

结论分为三层：

1. **固定完整 CPU 输运轨迹时可以精确回放。** 57,313 条 CPU 输运记录上传
   GPU 后 host/device ordered hash 完全相同，53,550 条 lepton 轨迹产生的
   CoREAS/ZHS 波形通过验收。
2. **只固定相同 raw random numbers 时不能得到同一棵树。** 原 CPU
   PROPOSAL 使用一个联合 draw 同时决定过程、组分和条件能损；`c8emrt`
   使用独立的过程 draw 与能损 draw，因此第一个有效顶点即可分叉。
3. **固定语义 decision 时，`proposal-native` 与 CPU PROPOSAL 一致。** 在
   20,480 个 \(\gamma,e^-,e^+,\mu^-,\mu^+\) 顶点中，过程、组分和 live
   `SampleLoss` mismatch 均为 0；支持的 \(v\) 最大相对差为
   \(7.97\times10^{-13}\)。`c8emrt` 在相同语义 decision 下的局部
   inverse-CDF 差异约为其 \(5\times10^{-4}\) 制表精度。

因此，当前结果没有显示 `proposal-native` 原生样条造成过程缺失或错误过程
选择。相同 seed 的 `c8emrt` 与 `proposal-native` shower 不会逐事件相同，原因
首先是随机数到物理决定的映射不同；即使强制相同语义 decision，`c8emrt`
二次制表的有限精度也会使能量在后续级联中逐渐分叉。

## 2. 测试范围与非侵入性

测试使用已有的验证旁路和 `EXCLUDE_FROM_ALL` oracle：

- `--cuda-replay-tape-out`：在标量 CPU shower 中记录已经接受的输运段；
- `cuda_decision_replay`：逐字节检查 tape 并在 GPU 上重算射电投影；
- `gpu_em_same_random_process_oracle`：比较 CPU 与 `c8emrt` 的 raw-random
  及语义 decision；
- `gpu_em_proposal_native_decision_oracle`：将 `proposal-native` GPU 选择、
  原生样条反解和 live CPU PROPOSAL `SampleLoss` 逐点比较。

本阶段没有修改 `c8_air_shower`、CPU PROPOSAL、CUDA 生产级联或射电内核。

结果目录：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_proposal_native_decision_tape_replay_20260831_v1/
```

## 3. 第一层：完整输运轨迹强制回放

CPU oracle 使用一个 100 GeV 电子初级、IGRF14/2027、0.5 MeV EM cut、
\(10^{-3}\) EM thinning 和 81 个 21CMA observer。未记录与记录 tape 的两次
CPU 运行使用相同 seed `2026081001`。

### 3.1 tape 记录不改变 CPU shower

未记录运行耗时 9.46 s，记录二进制 tape 与文本过程 trace 的运行耗时
11.68 s。`compare_exact_cuda_replay.py` 对两次 CPU 输出执行逐文件 SHA-256
比较，profile、energy deposit、ground particles、CoREAS 和 ZHS 文件均逐字节
相同，`capture_noninvasive=true`。

二进制 tape 大小约 12 MiB，SHA-256 为：

```text
2dd35f1c39c347b5d0e77b0f4f7c944b07cd2d3b6b66c5607bf8fbdac9b3a3f2
```

过程 trace 包含 13,404 个 CPU PROPOSAL 离散决定：

| 过程 | 次数 |
|---|---:|
| ionization | 5,249 |
| bremsstrahlung | 4,850 |
| Compton | 2,332 |
| photon pair | 894 |
| positron annihilation | 75 |
| electron/positron pair | 3 |
| photoproduction | 1 |

### 3.2 CUDA 轨迹和射电回放

| 指标 | 结果 |
|---|---:|
| transport records | 57,313 |
| lepton / photon records | 53,550 / 3,763 |
| invalid / non-finite records | 0 / 0 |
| byte-hash mismatch | 0 |
| host ordered hash | `450064fdedea100c` |
| device ordered hash | `450064fdedea100c` |
| fixed-point overflow | 0 |

射电逐 observer、逐分量最坏差异：

| 指标 | CoREAS | ZHS |
|---|---:|---:|
| peak-normalized maximum | \(1.02\times10^{-7}\) | \(2.18\times10^{-7}\) |
| relative \(L_2\) | \(1.14\times10^{-7}\) | \(1.91\times10^{-7}\) |
| relative fluence | \(2.23\times10^{-8}\) | \(3.19\times10^{-8}\) |
| 最大绝对场差 / V m\(^{-1}\) | \(8.57\times10^{-18}\) | \(8.35\times10^{-17}\) |

这一层证明：当 shower 轨迹已经固定时，GPU 可以消费完全相同的粒子历史，
CUDA 射电模块不会造成系综 profile 的偏差。它不测试 CUDA 自主随机采样。

## 4. 第二层：相同 raw random number 的第一次分叉

`c8emrt` 生产表使用与正式验收一致的介质、0.4 MeV PROPOSAL stochastic cut
和 \(5\times10^{-4}\) 表格精度。每个 PID 请求 4,096 个顶点。

| PID | 有效顶点 | raw process/component mismatch | 同过程但 raw loss mismatch |
|---:|---:|---:|---:|
| \(e^-\) | 3,726 | 1,100 | 2,626 |
| \(e^+\) | 3,726 | 1,057 | 2,504 |
| \(\gamma\) | 4,010 | 2,337 | 548 |

所有 CPU rates 按 CUDA 的 canonical column order 重排后，三种 PID 的过程和
组分 mismatch 都变为 0。这说明 raw-random 分叉不是漏过程，而是：

- 原 CPU PROPOSAL 按自身列顺序使用一个联合 draw；
- `c8emrt` 按 canonical 列顺序用一个 draw 选过程，再用独立 draw 选 \(v\)。

所以“相同 seed”不能作为逐事件等价条件。

## 5. 第三层：相同语义 decision

语义 tape 固定：

```text
particle state + process + component + conditional loss quantile
```

### 5.1 c8emrt

| PID | semantic points | 超过 \(5\times10^{-4}\) | 最大相对误差 |
|---:|---:|---:|---:|
| \(e^-\) | 3,726 | 0 | \(4.9614\times10^{-4}\) |
| \(e^+\) | 3,726 | 0 | \(4.8895\times10^{-4}\) |
| \(\gamma\) | 4,010 | 1 | \(5.1051\times10^{-4}\) |

该结果符合二次制表的预期精度；唯一轻微超限点是已知的局部 Compton
inverse-CDF 点。它不是错误过程，但如果各路径继续使用各自求得的 \(v\)，则
次级能量不会逐位相同，完整 shower 最终仍会分叉。

### 5.2 proposal-native

| PID | 顶点 | process mismatch | live `SampleLoss` mismatch | 最大 \(v\) 相对差 |
|---:|---:|---:|---:|---:|
| \(\gamma\) | 4,096 | 0 | 0 | \(6.76\times10^{-14}\) |
| \(e^-\) | 4,096 | 0 | 0 | \(5.32\times10^{-13}\) |
| \(e^+\) | 4,096 | 0 | 0 | \(7.97\times10^{-13}\) |
| \(\mu^-\) | 4,096 | 0 | 0 | \(6.09\times10^{-13}\) |
| \(\mu^+\) | 4,096 | 0 | 0 | \(1.82\times10^{-13}\) |

endpoint 与 CPU-only process 产生的 79 个显式 fallback 均按既定原因处理，
没有过程/组分重抽样。oracle 总状态为 `accepted=true`，原生表哈希为：

```text
7d618286c1acf3832ac8f6a4c8219ed02b94a204eea9b0cd16775d473b9ba72f
```

因此，就 `SampleLoss` 的随机数到物理决定映射而言，当前 proposal-native 比
`c8emrt` 二次制表更接近原 CPU PROPOSAL。

## 6. 后续过程回归

使用当前源码和当前链接的 CUDA 库补跑 13 项门禁，全部通过：

- native table 与 CPU fallback；
- interaction selection；
- photon-pair、brems、ionization、annihilation、epair 末态；
- 球形大气、lepton transport、photon wavefront；
- CUDA radio projection。

总计 `13/13 passed`。临时构建目录的独立 `.cu` 单测在重新编译时遇到 Conda
旧 sysroot 与 NVCC 的 `_Float32/_Float64` 头文件冲突；这是测试目标的编译环境
问题，不是运行时物理或 GPU oracle 失败。主 CUDA 库、主程序、两个 decision
oracle 和上述 13 个当前源码测试均已成功编译运行。

## 7. 对 2000 例 profile 差异的含义

本测试不支持“proposal-native 原生样条存在百分比级物理偏差”这一解释：

- 过程和组分没有错配；
- 原生 \(v\) 与 live CPU `SampleLoss` 的局部误差约 \(10^{-13}\)；
- 末态、输运和射电的定向回归通过；
- `c8emrt` 本身只要求约 \(5\times10^{-4}\) 的局部逆 CDF 精度，却可能在有限
  系综中偶然看起来更接近某一组独立 CPU 样本。

因此，现有 2000 vs 2000 图中 `c8emrt` 曲线视觉上更接近 CPU，不能单独证明
它的物理实现更准确。下一项最有效的检验是正在准备的同 seed 配对系综：逐事件
比较 `c8emrt` 与 proposal-native 的相关差值，再与独立 CPU 系综共同判断。

## 8. 能否得到完全相同的 shower？

- **完整 transport tape：可以。** CPU 直接指定轨迹，GPU 精确消费；适用于
  输运记录与射电模块验证，不代表生产加速。
- **语义 decision tape：可以固定过程树拓扑。** 若再强制使用 CPU 的最终
  \(v\)、散射结果和次级状态，可得到相同 tree。
- **只给相同 seed：不可以。** 除非 CPU 与两种 CUDA 后端统一列顺序、draw
  地址、浮点累计、末态 draw 和次级排序契约。

正式生产后端仍应以大样本统计一致性作为最终验收标准，而 decision tape 用于
定位第一处局部物理或数值分叉。

## 9. 同 seed 自主演化 profile 诊断图

为避免把旧可执行文件差异误认为表源差异，补跑了一个受控的 25 对诊断批次：

- 同一个当前 beta4 可执行文件；
- 相同的种子 `2026180001--2026180025`；
- 1 TeV 垂直质子、`emthin=1e-6`、默认 maximum weight；
- IGRF14（2027）、相同 cuts、CUDA EM/radio 与 FLUKA 配置；
- 唯一主动改变的物理开关是 `c8emrt` / `proposal-native`。

结果与脚本分别位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_proposal_native_decision_tape_replay_20260831_v1/diagnostic_plots/

validation/gpu_em/plot_physics_source_paired_profiles.py
```

生成了三张图：

1. `same_seed_single_event_profile_seed2026180001.png`：第一对事件的
   (e^-+e^+)、光子、强子、缪子和能量沉积 profile；
2. `same_seed_single_event_shape_seed2026180001.png`：同一事件的峰值归一化
   profile 形状；
3. `same_seed_pair_summary_25.png`：25 对事件的 EM profile 相对 (L_2)、
   (X_{\max}) 相关性和 (N_{\max}) 比值。

第一对事件恰好较接近：EM profile 相对 (L_2=0.136)，
\(\Delta X_{\max}=-10\ \mathrm{g\,cm^{-2}}\)。但是 25 对样本的中位相对
\(L_2=0.623\)，中位 EM profile 相关系数为 0.915；个别事例在早期分叉后产生
很大的差别。这个结果不是 proposal-native 局部样条精度失败。它验证的是前述
结论：两条自主路径对 random draw 的地址和消费顺序不同，相同 seed 并不等价于
强制相同 decision tape。单事例图适合显示分叉如何被级联放大，但不能代替独立
大样本的物理验收。

## 10. 新同 seed native 2000 例的本地阶段验收

与 c8emrt 正式样本共用 seed 起点 `2026180001` 的 proposal-native 2000 例
已经完成。80 个批次均为 canonical complete，manifest 记录 `2000/2000`；没有
queue overflow、native inverse failure、显存 spill 或 cross-species host spill。
15 次 generic fallback 全部有明确的 `unsupported_geometry` 原因。

由于 PSR 停机，CPU 2000 例原始 Parquet 暂时不可访问。本地先完成两层检查：

1. 使用完整原始数据，比较上一批 native 2000 与新 native 2000；
2. 使用此前 CPU/native 正式比较保存的 CPU mean、standard deviation 和
   standard error，重新计算新 native 相对 CPU 的均值偏移和合并误差。

第二层通过严格 bridge 校验：上一批 native 在两个独立比较 JSON 中的 events、
mean、standard deviation 和 standard error 逐指标一致。因此没有混用不同
native 样本。

### 10.1 相对 CPU 摘要的变化

| Observable | 旧 native/CPU | 新 native/CPU |
|---|---:|---:|
| charged \(X_{\max}\) | -2.060% | -0.114% |
| charged-profile integral | +0.546% | +0.293% |
| photon-profile integral | +0.354% | +0.056% |
| \(e^-+e^+\) profile integral | +0.511% | +0.333% |
| total-EM profile integral | +0.377% | +0.097% |
| deposited energy | +0.366% | +0.067% |
| energy-deposit \(X_{\max}\) | -3.093% | +0.812% |
| ground EM count | -7.140% | +0.788% |
| ground EM kinetic energy | -12.611% | -1.405% |
| energy closure | -0.175% | +0.006% |

列出的 11 个指标全部减小了绝对均值偏移，新样本相对 CPU 摘要的所有合并
\(|z|<0.52\)。因此，针对用户关注的“新同 seed 样本是否看起来更好”，答案是：
**均值层面明确更好**。这也表明上一批 native 图中的视觉偏移至少包含明显的有限
样本涨落，不能直接归因为 proposal-native 物理表误差。

### 10.2 必须保留的未通过项

新旧 native 的直接 2000 vs 2000 比较中：

- key-scalar 3-sigma gate：通过；
- 所有 longitudinal/ground curve gate：通过；
- 9 个 key-scalar KS 中 8 个通过；
- `ground_em_kinetic_energy_GeV` 的 KS 距离为 0.0530，略高于 95% 临界值
  0.0430，因此严格 pooling 总判定仍为失败。

两批数据使用完全相同的 executable、原生表、辅助表和物理配置，这个单项失败
不是版本差异，说明低能地面 EM 尾部具有较强 seed 敏感性。不能为了得到“通过”
结论而删除该门禁。

完整本地报告和图位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  final_beta4_proposal_native2000_proton_1TeV_vertical_emthin1e-6_
  defaultmaxw_igrf14_2027_seed2026180001_pairedc8_v1/
    local_stored_cpu_crosscheck_2000/
    independent_seed_stratum_acceptance_vs_seed2026328001/plots/
```

PSR 恢复后仍需用 CPU 原始数据补做新 native 的直接逐深度 profile、KS、ground
timing 和射电脉冲比较；在此之前，本节结论应称为“本地阶段验收”，不能标成完整
CPU 2000 vs native 2000 正式通过。
