# Phase 82：原版单事例的 CUDA 严格决策回放与射电复刻

## 1. 结论

已经完成一次原版 `c8_air_shower` 的 1 TeV 电子 shower 严格回放。
本次验收同时满足：

1. 原版 CPU 在关闭和开启决策记录器时，7 个物理 Parquet 输出的
   SHA-256 全部逐位相同；
2. CUDA 接收了原版 CPU 产生的全部 136,114 个标量输运段，CPU 和 GPU
   的有序字节散列完全相同；
3. CUDA 使用同一批 127,174 个 \(e^\pm\) 轨迹段重新计算了 81 个天线的
   CoREAS 和 ZHS 波形；
4. 所有天线、三分量和两种射电算法均通过波形比较门限；
5. CUDA 固定点累加没有溢出，新增 C++ 与 Python 回归测试全部通过。

总比较报告给出的最终状态为：

```json
{
  "accepted": true
}
```

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
exact_cuda_replay_original_electron_1TeV_v1
```

机器上的正式比较报告：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
exact_cuda_replay_original_electron_1TeV_v1/
exact_replay_comparison.json
```

需要准确理解这里的“严格复刻”：

> CUDA 回放程序消费原版 CPU 已经做出的逐步输运决定，并在设备端校验这些
> 状态、重新计算射电信号。它是单事例兼容性 oracle，不是生产 CUDA 后端
> 从同一个 seed 独立重新抽样得到同一 shower。

因此，本阶段已经直接证明“同一条原版 shower 轨迹输入 CUDA 射电内核时，
能得到与原版 CPU 相同的 CoREAS/ZHS 信号”，也证明了记录功能没有扰动原版
shower；但它不应被描述成“wavefront CUDA 蒙特卡洛已经逐事例复制串行
PROPOSAL 的随机抽样”。

## 2. 为什么必须采用决策回放

原版 CPU cascade 使用：

```text
LIFO stack
  -> 深度优先完成一条粒子支路
  -> 全局随机流按实际调用先后消耗
  -> PROPOSAL 抽取反应距离、过程和末态
```

生产 CUDA 后端使用：

```text
wavefront queues
  -> 同一前沿中的大量粒子并行推进
  -> history/step/process keyed Philox
  -> GPU 表格和设备端过程采样
```

二者即使输入同一个整数 seed，随机数消费顺序和随机数映射也不同。第一次
分支只要不同，后续粒子树便不再具有一一对应关系。这是蒙特卡洛调度语义的
差异，不是“相同 seed 应当自动得到相同 shower”的保证失效。

若要逐事例定位差异，必须先固定原版所做出的决定，再让 CUDA 消费同一组
决定。严格回放的数据流为：

```text
原版 CPU c8_air_shower
  │
  ├─ 正常写出 profile、dEdX、ground particles、CoREAS、ZHS
  │
  └─ 旁路记录每个标量 Step
       │
       ▼
   event.c8rpt
       │
       ├─ CUDA：逐记录字节散列与物理健全性检查
       │
       └─ CUDA：提取相同 e± Step，重新运行 CoREAS/ZHS kernel
             │
             ▼
       GPU observers.parquet
             │
             ▼
       CPU/GPU 逐天线、逐分量波形比较
```

## 3. 本阶段实现

### 3.1 与 CUDA 无关的原版 CPU 记录器

原版源码树：

```text
/home/yuhanglu/21CMA/corsika-21cma/corsika
```

增加：

```text
corsika/validation/CudaDecisionReplayTypes.hpp
corsika/validation/CudaDecisionReplayTape.hpp
```

并在原版：

```text
corsika/detail/framework/core/Cascade.inl
```

的普通标量 `Step` 完成 `doContinuous()` 后记录：

- shower 编号与全局 step ordinal；
- PDG；
- step limiter；
- `ProcessReturn`；
- 前后位置；
- 前后方向；
- 前后时间；
- 前后总能量；
- 粒子权重。

原版应用新增：

```text
--cuda-replay-tape-out PATH
```

该选项只写旁路二进制记录，不改变 stack、PROPOSAL、RNG 或输出过程。

### 3.2 回放格式

格式 magic 和版本为：

```text
magic           C8RPT001
format_version  1
```

文件头保存：

- seed 和 shower 数量；
- 每条记录的固定字节数；
- CoREAS/ZHS observer 位置、窗口和采样率；
- 原版 flat-atmosphere 射电传播表及其外推参数。

记录对象是可直接复制到设备的 trivially-copyable POD。读取时严格检查：

- magic、版本和 endian marker；
- header/record 大小；
- 文件长度和 record count；
- observer 与传播表数量；
- 流截断和尾部多余字节。

### 3.3 CUDA 输运记录校验

新增：

```text
corsika/gpu/em/CudaDecisionReplayVerifier.hpp
src/gpu/em/CudaDecisionReplayVerifier.cu
```

完整 tape 一次上传到 GPU。每个 CUDA thread 校验一条记录并计算该 POD
全部字节的 FNV-1a 散列。主机对完全相同的字节再计算一次散列，最后分别
形成保持原顺序的整条 tape 散列。

以下情况会立即拒绝回放：

- CPU/GPU 单记录字节散列不同；
- 有序总散列不同；
- NaN 或 infinity；
- 负能量或负权重；
- 时间倒退；
- 方向向量没有归一化。

`|dx|/dt > c` 单独作为诊断计数而不是致命错误。原因是原版
`TrackingLeapFrogCurved` 的第二个位置半步使用临时未归一化速度，随后只对
输出方向归一化。在较粗的磁偏转步长下，原版 `Step` 自身可能出现这一离散
化特征。严格回放必须保留原版实际传给射电过程的 `Step`，不能在 CUDA 侧
静默修改端点。

### 3.4 CUDA 射电回放

新增应用：

```text
applications/cuda_decision_replay.cpp
```

并为：

```text
corsika/gpu/radio/CudaRadioAccumulator.hpp
src/gpu/em/CudaRadioAccumulator.cu
```

增加 exact host-track injection 接口。应用从 tape 中选择电子和正电子，
保持原顺序上传，并调用现有 CUDA CoREAS/ZHS 内核。射电计算不是复制 CPU
输出文件，而是 GPU 对原版轨迹段进行新的数值累加。

输出包括：

```text
cuda_replay/
  CoREAS/config.yaml
  CoREAS/observers.parquet
  ZHS/config.yaml
  ZHS/observers.parquet
  replay_summary.json
```

`replay_summary.json` 明确写入：

```json
{
  "production_cuda_sampling": false
}
```

防止把决策注入回放误报成生产 CUDA 独立抽样。

### 3.5 自动比较

新增：

```text
validation/gpu_em/compare_exact_cuda_replay.py
validation/gpu_em/tests/test_compare_exact_cuda_replay.py
```

比较器执行三组门禁：

1. 原版 CPU 关闭/开启 tape 的物理 Parquet SHA-256 必须逐位相同；
2. replay summary 必须没有致命记录、散列差异或固定点溢出；
3. 对每个 shower、observer、算法和 Ex/Ey/Ez 分量比较时间轴、最大绝对
   误差、peak-normalized 最大误差、相对 L2 和相对 fluence。

当前门限：

| 指标 | 门限 |
|---|---:|
| peak-normalized 最大误差 | \(10^{-4}\) |
| 相对 L2 | \(10^{-4}\) |
| 相对 fluence | \(5\times10^{-4}\) |
| 弱信号绝对误差 | \(10^{-18}\ {\rm V/m}\) |
| 时间轴误差 | \(10^{-12}\ {\rm ns}\) |

## 4. 正式事例配置

| 项目 | 值 |
|---|---|
| 原版程序 | `corsika-21cma/corsika-build/applications/c8_air_shower` |
| 初级粒子 | electron，PDG 11 |
| 初级总能量 | 1000 GeV |
| shower 数量 | 1 |
| seed | 260731 |
| zenith / azimuth | \(0^\circ/0^\circ\) |
| injection height | 112,750 m |
| observation level | 2,680.444195 m |
| 地磁 | 程序自带 `IGRF13.COF`，year 2025 |
| 天线 | `/home/yuhanglu/21CMA/data/antennas.txt`，81 个有效位置 |
| EM cut | 0.5 MeV |
| EM thinning | \(10^{-3}\) |
| maximum weight | \(10^6\) |
| maximum magnetic deflection | 0.2 rad |
| CPU threads | 1 |
| CUDA device | NVIDIA GeForce RTX 4060 Laptop GPU，CC 8.9 |
| CUDA deterministic accumulation | enabled |
| fixed-point field limit | 0.001 V/m |

本次产物的 SHA-256：

| 产物 | SHA-256 |
|---|---|
| 原版 `c8_air_shower` | `133daa7ee0a4de4ba7355df5facb44e9a09e23bb5fff8b9549e7b6da546daeb9` |
| `cuda_decision_replay` | `b3b5fa4bd77b18aa665237f2a690bc2dbe5e332ed8d493f76f03af31b1333c50` |
| `event.c8rpt` | `eeffd612a84efbb51e27628967569901afc4a5c9dcf07d1f189ab72b8fdc6a12` |
| `exact_replay_comparison.json` | `92b1f14852f99c03e1aadf815c59be3cff22c5bf6be6f01e9cd02ecc80ad25de` |

这里保留了原版程序的默认 1 GHz、400 ns 射电 observer 设置。该试验的
目的不是比较 CoREAS 与 ZHS 两种 formalism，而是分别比较：

```text
CPU CoREAS vs CUDA CoREAS
CPU ZHS    vs CUDA ZHS
```

## 5. 可复现命令

### 5.1 原版 CPU：不记录 tape

```bash
cd /home/yuhanglu/21CMA/corsika-21cma/corsika-build

FLUPRO=/home/yuhanglu/fluka \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
./applications/c8_air_shower \
  -p 11 -E 1000 -N 1 \
  -f /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/cpu_no_tape \
  --seed 260731 \
  --zenith 0 --azimuth 0 \
  --observation-level 2680.444195 \
  --injection-height 112750 \
  --ring 0 \
  --antenna-file /home/yuhanglu/21CMA/data/antennas.txt \
  --emcut 0.0005 \
  --emthin 0.001 \
  --max-weight 1000000 \
  --hadcut 10000000000000 \
  --mucut 10000000000000 \
  --taucut 10000000000000 \
  --max-deflection-angle 0.2 \
  --disable-interaction-histograms \
  --verbosity warn
```

### 5.2 原版 CPU：只增加 tape 记录

命令参数完全相同，只把输出改为 `cpu_with_tape` 并增加：

```bash
--cuda-replay-tape-out \
  /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/event.c8rpt
```

本次 tape 大小：

```text
25,786,600 bytes
```

### 5.3 CUDA 回放

```bash
cd /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda

./applications/cuda_decision_replay \
  --tape /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/event.c8rpt \
  --output /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/cuda_replay \
  --device 0 \
  --memory-fraction 0.7 \
  --fixed-point-field-limit 0.001 \
  --deterministic true
```

### 5.4 自动验收

```bash
cd /home/yuhanglu/21CMA/corsika8_gpu_refactor

/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python \
  validation/gpu_em/compare_exact_cuda_replay.py \
  --untaped-reference /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/cpu_no_tape \
  --reference /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/cpu_with_tape \
  --replay /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/cuda_replay \
  --report /home/yuhanglu/21CMA/corsika_validation_results/exact_cuda_replay_original_electron_1TeV_v1/exact_replay_comparison.json \
  --require-pass
```

## 6. 正式结果

### 6.1 记录器没有改变原版 shower

| 物理输出 | SHA-256 | 逐位相同 |
|---|---|---:|
| `CoREAS/observers.parquet` | `ff130212…57c781e` | 是 |
| `ZHS/observers.parquet` | `8d998dc9…ed27775` | 是 |
| `energyloss/dEdX.parquet` | `58c9fce3…4cd49c` | 是 |
| `interactions/interactions.parquet` | `c645e641…552e9f` | 是 |
| `particles/particles.parquet` | `92dcb709…ce8471` | 是 |
| `production_profile/profile.parquet` | `c245976b…b32575` | 是 |
| `profile/profile.parquet` | `7dd956d6…e069` | 是 |

这里比较的是原版 CPU 的 tape-off 与 tape-on 两次独立运行。包括原版 CPU
射电信号在内的物理数据逐字节一致，直接证明旁路记录没有改变 RNG 消费或
shower 发展。

`config.yaml` 和 timing summary 不作为逐位比较对象，因为输出路径和运行
时间本来就不同。

### 6.2 CUDA 接收了完全相同的输运记录

| 项目 | 结果 |
|---|---:|
| 标量输运记录 | 136,114 |
| EM 记录 | 136,114 |
| \(e^\pm\) 记录 | 127,174 |
| photon 记录 | 8,940 |
| 致命非法记录 | 0 |
| CPU/GPU 单记录散列不匹配 | 0 |
| CPU 有序总散列 | `1e23215251464505` |
| GPU 有序总散列 | `1e23215251464505` |
| 最大方向模长误差 | \(1.301\times10^{-12}\) |

这意味着设备看到的每个 PID、位置、方向、时间、能量、权重和 limiter 的
原始字节，与 CPU tape 中的记录相同，并且顺序也相同。

额外记录到：

```text
legacy leapfrog |dx|/dt diagnostic count = 2662
maximum |dx|/(c dt)                         = 1.0072568168
```

它是第 3.3 节所述原版 leapfrog 大步长离散化诊断，不是 CPU/GPU 差异。

### 6.3 CUDA 射电执行统计

| 项目 | 结果 |
|---|---:|
| GPU \(e^\pm\) tracks | 127,174 |
| track-observer pairs | 20,602,188 |
| CoREAS contributions | 19,641,525 |
| ZHS contributions | 49,828,384 |
| ZHS subtracks | 10,516,349 |
| fixed-point overflows | 0 |
| host-to-device | 81,189,552 bytes |
| device-to-host | 1,559,128 bytes |
| GPU radio device time | 142.025 ms |

这里的计时只描述回放射电内核，不可当作完整 CPU/CUDA shower 加速比。

### 6.4 CPU 与 CUDA 射电波形

每种算法包含：

```text
81 observers × 3 components = 243 waveform comparisons
```

最坏结果：

| 算法 | 时间轴最大差 / ns | 最大绝对差 / V/m | peak-normalized max | relative L2 | relative fluence |
|---|---:|---:|---:|---:|---:|
| CoREAS | \(2.842\times10^{-14}\) | \(4.763\times10^{-18}\) | \(3.646\times10^{-10}\) | \(1.983\times10^{-10}\) | \(6.290\times10^{-11}\) |
| ZHS | \(1.421\times10^{-14}\) | \(1.098\times10^{-15}\) | \(1.663\times10^{-7}\) | \(7.304\times10^{-8}\) | \(2.334\times10^{-8}\) |

ZHS 的最大绝对误差来自较强样本，归一化误差仍只有
\(1.66\times10^{-7}\)，远低于 \(10^{-4}\) 的验收门限。两种算法的所有
observer 和三分量均通过。

CPU 与 CUDA 不是逐 bit 相同的浮点波形，因为：

- CPU 和 GPU 的浮点指令及求和顺序不同；
- CUDA 确定性路径使用 64-bit 固定点原子累加；
- 最终再转换回 double。

但误差已经处于数值舍入/量化量级，且比当前物理验收门限小数个数量级。

## 7. 回归测试

### 7.1 CUDA 射电 C++ 测试

```text
ctest --output-on-failure -R '^testGpuRadioProjection$'

1/1 passed
```

### 7.2 GPU 验证 Python 测试

```text
python -m unittest discover \
  -s validation/gpu_em/tests \
  -p 'test_*.py'

Ran 87 tests
OK
```

其中包含 exact replay 比较器的通过和故意扰动失败用例。

## 8. 这份证据证明了什么

已经证明：

1. 原版 scalar `c8_air_shower` 在 seed 260731 下可重复产生同一 shower；
2. 打开决策记录器不会改变原版 shower 或射电输出；
3. CUDA 可以无损接收原版该事例的逐步输运状态；
4. GPU 使用原版同一批 \(e^\pm\) 轨迹时，CoREAS 和 ZHS 波形与 CPU
   数值一致；
5. 因此，若未来生产 CUDA shower 与 CPU 的射电不同，可以把问题进一步
   分解为“轨迹不同”或“射电投影不同”；本阶段已经基本排除后者。

尚未证明：

1. 生产 `--em-backend cuda` 只靠 seed 260731 就会生成同一棵粒子树；
2. CUDA 已经独立重算并逐位复制每个 PROPOSAL 反应率、过程选择和末态；
3. 当前 tape 已覆盖离散反应选择、所有次级粒子关系、thinning/cut 决定和
   RNG draw 的完整决策图；
4. 回放的 142 ms 能代表完整 shower 性能。

当前 tape 的核心是 `doContinuous()` 完成后的标量输运段。它足以验证轨迹
传输与射电，但还不是完整的离散相互作用 replay protocol。

## 9. 下一步：从轨迹回放扩展到完整物理决策回放

若目标是让 CUDA 在每个反应边界逐步复刻原版，而不只是消费最终 Step，
下一阶段应将 tape 扩展为以下事件：

```text
ParticleBirth
TransportCompetition
ContinuousLoss
InteractionSelection
InteractionFinalState
DecaySelection
ThinningDecision
ParticleCutDecision
BoundaryCrossing
ParticleTermination
```

每个事件必须包含：

- 稳定的 history ID、parent ID、generation 和 step ID；
- 原版 RNG stream 名称、draw ordinal 和原始随机字；
- interaction/decay/geometry/continuous 的候选距离；
- 选中的 process、medium component 和 \(v\)；
- 相互作用前后四动量及全部次级粒子；
- thinning 的选择、权重变化和被丢弃分支；
- 能量沉积及终止原因。

CUDA replay kernel 应在每个阶段执行对应的状态转移，再与 tape 中的期望
状态逐字段比较，第一次不一致时输出：

```text
history_id
step_id
process_id
field
CPU value
CUDA value
absolute/relative difference
RNG key/draw
```

这会把当前的“相同轨迹输入下射电一致”进一步推进为“同一原版决策图下，
CUDA 逐过程状态转移一致”。最后才适合尝试一个代价较高的
`--cuda-legacy-rng-order` 兼容模式，让 CUDA 按原版 LIFO 顺序消费同一随机
流；该模式主要用于验证，不会具有 wavefront 并行性能。
