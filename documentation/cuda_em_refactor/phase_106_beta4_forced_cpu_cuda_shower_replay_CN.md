# Phase 106：CPU 决策带到 CUDA 的强制 shower 回放测试

## 1. 测试目的

本测试检查以下问题：先让原标量 CPU 路径生成一个确定的电磁 shower，记录其输运决定和轨迹，再由 CUDA 消费这份记录，能否重放同一个 shower，并得到几乎相同的 CoREAS/ZHS 射电信号。

为避免影响生产物理路径，本次测试只使用 beta4 已有的验证接口：

- `--cuda-replay-tape-out` 仅在 CPU `proposal` 运行中旁路记录；
- `--cuda-replay-trace` 额外输出逐过程审计 CSV；
- 独立程序 `cuda_decision_replay` 读取决策带并在 GPU 上验证输运记录和计算射电波形；
- `compare_exact_cuda_replay.py` 比较未记录 CPU、记录 CPU 和 CUDA replay 三份结果；
- 没有修改 CORSIKA cascade、PROPOSAL、CUDA 生产输运或射电核心代码。

## 2. 测试配置

本次使用一个较快但包含完整电磁级联和射电计算的纯电子事例：

| 参数 | 数值 |
|---|---:|
| 初级粒子 | electron，PDG 11 |
| 初级能量 | 100 GeV |
| 事例数 | 1 |
| seed | 2026081001 |
| zenith / azimuth | 0 deg / 0 deg |
| 地磁场 | IGRF14，2027 年 |
| observation level | 2680.44 m |
| injection height | 112750 m |
| EM cut | 0.5 MeV |
| EM thinning | \(10^{-3}\) |
| 天线 | `antennas_nwu_coordinates_test.txt`，81 个 observer |
| CPU EM backend | PROPOSAL |
| CPU radio backend | CPU CoREAS + ZHS |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9 |

为了隔离电磁路径，hadron、muon 和 tau cut 均设为远高于初级能量的值。本测试不是强子 shower 验收。

结果目录为：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  beta4_forced_replay_electron_100GeV_v1/
```

## 3. 测试步骤

### 3.1 检查记录功能是否干扰 CPU shower

相同参数和 seed 分别运行两次 CPU：

1. `cpu_no_tape`：不启用任何 replay 记录；
2. `cpu_with_tape`：输出 `event.c8rpt` 和 `cpu_process_trace.csv`。

比较两次运行产生的全部 Parquet 文件。以下 7 类物理输出的 SHA-256 均逐字节相同：

- CoREAS waveform；
- ZHS waveform；
- energy loss；
- interactions；
- ground particles；
- production profile；
- longitudinal profile。

因此，当前旁路记录器没有消耗额外物理随机数，也没有改变 stack 顺序、粒子轨迹、profile 或射电结果。

CPU 未记录运行时间为 8.535 s；同时写二进制 tape 和 4.6 MiB CSV trace 时为 9.885 s。后者包含文本过程跟踪的 I/O 开销，不能解释为单独的二进制 tape 开销。

### 3.2 CPU 过程覆盖

`cpu_process_trace.csv` 共记录 13,404 次离散/连续过程结果：

| 过程 | 记录数 |
|---|---:|
| bremsstrahlung | 4,850 |
| ionization | 5,249 |
| electron/positron pair production | 3 |
| Compton scattering | 2,332 |
| positron annihilation | 75 |
| photon pair production | 894 |
| photoproduction | 1 |
| 合计 | 13,404 |

这说明该事例不是单一步长的 toy test，而是覆盖了电子、正电子和光子的主要级联过程。photoproduction 仍属于 CPU 指定过程；它被如实保留在 CPU shower 中。

### 3.3 CUDA 强制回放

为了不干扰当时正在执行的 CUDA 生产任务，replay 只设置 `--memory-fraction 0.01`。执行命令的核心形式为：

```bash
cuda_decision_replay \
  --tape event.c8rpt \
  --output cuda_forced_replay \
  --device 0 \
  --memory-fraction 0.01 \
  --fixed-point-field-limit 1 \
  --deterministic true
```

CUDA 接收并验证：

- 57,313 条输运记录；
- 53,550 条 \(e^\pm\) 轨迹段；
- 3,763 条 photon 轨迹段；
- invalid、non-finite、负能量、负权重、时间倒退和非法方向记录均为 0；
- host/device ordered hash 均为 `450064fdedea100c`；
- byte hash mismatch 为 0；
- 最大方向模长误差为 \(3.16\times10^{-13}\)。

诊断中有 526 条 `superluminal_records`，最大 \(v/c=1.0132\)。这是记录中由已有 leapfrog 离散轨迹端点和时间差计算出的诊断量，并未造成非法记录、哈希差异或验收失败；它不表示 GPU 改写了这些 CPU 轨迹。

## 4. 射电信号比较

CUDA 对同一批 53,550 条 lepton 轨迹进行了 CoREAS 和 ZHS 投影：

- track-observer pairs：8,675,100；
- CoREAS contributions：7,873,445；
- ZHS contributions：20,002,292；
- ZHS subtracks：4,523,069；
- fixed-point overflow：0；
- GPU radio kernel 时间：93.58 ms。

81 个 observer、三个电场分量的逐波形比较全部通过。各算法最坏误差如下：

| 指标 | CoREAS | ZHS |
|---|---:|---:|
| 最大时间差 | \(2.84\times10^{-14}\) ns | \(1.42\times10^{-14}\) ns |
| 最大绝对场差 | \(8.57\times10^{-18}\) V/m | \(8.35\times10^{-17}\) V/m |
| 峰值归一化最大差 | \(1.02\times10^{-7}\) | \(2.18\times10^{-7}\) |
| relative \(L_2\) | \(1.14\times10^{-7}\) | \(1.91\times10^{-7}\) |
| relative fluence | \(2.23\times10^{-8}\) | \(3.19\times10^{-8}\) |

因此，在输入轨迹严格相同时，CUDA 射电路径和 CPU 路径已经达到约 \(10^{-7}\) 的相对一致性，远低于当前 `compare_exact_cuda_replay.py` 使用的 \(10^{-4}\) 波形阈值。

## 5. 本测试能够证明什么

本次验收通过，可以直接证明：

1. CPU 决策带记录是非侵入的，开启记录不会改变原 CPU shower；
2. CPU 产生的完整输运轨迹能够被 CUDA 逐记录、逐字节接收；
3. 强制输入同一条 shower 轨迹时，CPU 与 GPU 的 CoREAS/ZHS 波形几乎一致；
4. “同一 shower 的 CPU/GPU 确定性模块比较”是可行而且诊断能力很强的测试方法。

## 6. 本测试尚不能证明什么

当前 `event.c8rpt` 是 **transport decision/trajectory tape**，并不是“每次物理过程的所有原始随机数 tape”。它记录已经由 CPU 决定的粒子种类、步长限制、起止状态、能量、时间和权重，然后让 GPU 消费这些确定结果。当前 CUDA replay 的 `production_cuda_sampling` 明确为 `false`。

所以本结果不能证明：仅给 CPU 和 CUDA 相同 seed，或仅把少量 uniform random numbers 交给 GPU，CUDA 的自主表格采样就会生成同一个粒子树。即使随机数完全相同，CPU PROPOSAL 与 CUDA 表格在反应率、逆 CDF、浮点运算和次级粒子排序上的微小差异，也可能让第一次边界判断或过程选择发生分叉，之后整个 shower 会不同。

若要严格测试“原始随机数驱动的自主重演”，还需要一个验证专用的第二版 tape，至少记录并校验：

- interaction/decay distance 的每个 draw；
- process 和 target component 选择；
- energy-loss \(v\)；
- LPM、角分布与 multiple scattering；
- final-state 的全部随机数和次级粒子顺序；
- cut/thinning 的每次随机决定；
- history ID、parent ID、step ID 与随机数消费位置。

GPU 必须在每一步报告“实际消费的随机数、得到的决定和 CPU oracle 是否一致”，并在第一个分叉点停止。这可以做成独立 validation executable 或测试构建开关，不应默认进入生产二进制。

## 7. 结论

答案是：**在 CPU 先固定完整决定和轨迹的 forced replay 模式下，CPU 与 GPU 可以得到同一个 shower 表示，并产生约 \(10^{-7}\) 相对差异的射电波形。** 当前测试对核心程序零修改，所有自动验收均通过。

但这是一项“同轨迹确定性回放”验证，不是 CUDA 自主电磁级联与原 CPU 的逐随机数等价证明。后者需要扩展验证 tape 和建立逐 draw 的第一次分叉诊断；在生产模拟中，物理正确性的主要验收仍应使用独立 CPU/CUDA shower 的大样本统计分布。

逐 draw/逐过程的后续测试和第一次分叉结果见
[`phase_107_beta4_same_random_first_divergence_CN.md`](phase_107_beta4_same_random_first_divergence_CN.md)。
