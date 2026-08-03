# CUDA 电磁后端重构记录：阶段 25，驻留 GPU CoREAS/ZHS 射电投影

## 1. 本阶段结论

本阶段把 CUDA 电磁输运产生的 \(e^\pm\) 轨迹段直接送入设备端
CoREAS/ZHS 内核。对于驻留 GPU 的电磁级联，不再把全部轨迹逐段回传到
CPU 后再调用标量射电过程。

实现后的数据流为：

```text
resident lepton transport
  -> device LeptonTransportRecord[]
  -> GPU CoREAS/ZHS track × observer projection
  -> device-resident waveform accumulator
  -> shower 结束时一次 waveform D2H
  -> 原 TimeDomainObserver
  -> 原 RadioProcess::endOfShower()
  -> observers.parquet
```

这回答了“射电计算与粒子传播是同步还是事后计算”的实现问题：

- 原 CORSIKA 8 `RadioProcess` 是连续过程，每个粒子 step 形成后立即调用
  `simulate(step)`，所以射电贡献与粒子输运同步生成；
- 贡献立即累加到观测器波形，但 Parquet 写出发生在 shower 结束时；
- 新 CUDA 路径保持相同语义：每个 resident wavefront 输运结束后立即投影，
  只是波形一直驻留显存，到 shower 结束才统一回传和写出；
- 它不是先保存完整 shower 的全部粒子，再离线重放。

当前阶段通过：

- `testGpuRadioProjection`：19/19 checks；
- 10 GeV electron 完整 shower；
- CPU/GPU 同种子波形比较；
- 两次独立 GPU 进程的完整 Parquet SHA-256 一致；
- 设备定点累加无溢出；
- 有效射电轨迹计数与 CPU 完全一致。

本阶段仍不是 \(10^{17}\)–\(10^{18}\,\mathrm{eV}\) 的最终生产验收。

## 2. 新增与修改的核心组件

### 2.1 设备射电类型

新增：

```text
corsika/gpu/radio/Types.hpp
```

主要类型为：

```cpp
RadioObserverSnapshot
FlatAtmosphereRadioSnapshot
GpuRadioConfig
RadioWaveform
GpuRadioWaveforms
GpuRadioStatistics
```

所有设备输入使用固定单位：

```text
位置       m
时间       s
采样率     Hz
电场       V/m
```

`GpuRadioConfig` 同时记录：

```cpp
bool deterministic;
double fixed_point_field_limit_V_per_m;
```

默认保留 \(\pm1\,\mathrm{V/m}\) 的目标电场范围，并额外留有一个二进制
headroom bit。

### 2.2 从原环境建立只读快照

新增：

```text
corsika/gpu/radio/RadioSnapshotBuilder.hpp
```

它从应用当前使用的：

```text
Environment
TabulatedFlatAtmospherePropagator 的高度范围
1 m 折射率步长
CoREAS/ZHS TimeDomainObserver
```

生成设备快照。折射率查找、离散积分、上下边界斜率和最近网格点规则与现有
标量 propagator 保持一致，而不是另建一套近似大气。

### 2.3 持久 GPU 累加器

新增：

```text
corsika/gpu/radio/CudaRadioAccumulator.hpp
src/gpu/em/CudaRadioAccumulator.cu
```

`CudaRadioAccumulator` 在 shower 开始时一次性分配并上传：

- CoREAS/ZHS 观测器；
- 折射率和积分折射率表；
- 两套三偏振波形数组；
- 设备统计和错误计数。
- 一个 non-blocking radio stream；
- 两个 EM workspace 各自的 input-ready/start/done events。

shower 传播期间不为每个 step 调用 `cudaMalloc`。每个 lepton wavefront
直接传入后端已经存在的 `LeptonTransportRecord*` 设备指针。

## 3. 设备端 CoREAS

每个 CUDA thread 处理一个：

```text
track × observer
```

内核逐项复现原 CoREAS endpoint 算法：

1. 从轨迹端点求平均 \(\boldsymbol{\beta}\)；
2. 对起点和终点传播到观测器；
3. 计算
   \[
   1-n\,\boldsymbol{\beta}\cdot\hat{\mathbf r};
   \]
4. 生成两个 endpoint 电场贡献；
5. 在 Cherenkov 小分母区域采用原实现的 midpoint approximation；
6. 当两端落入同一时间 bin 时使用相同的 bin separation 与幅度缩放；
7. 将贡献加入对应观测器、时间 bin 和偏振。

电荷常数有意使用当前 CORSIKA 核心中的

```text
1.6021766208e-19 C
```

而不是直接换成 2019 SI exact value。这样设备输出与本项目当前标量
CoREAS 参考完全一致。

## 4. 设备端 ZHS

ZHS 内核同样以 `track × observer` 并行。它实现：

- 中点传播；
- \(\boldsymbol{\beta}_\perp\)；
- 起止接收时间；
- 同 bin 与跨 bin 的 vector-potential 累加；
- 标量实现中间 bin 的符号约定；
- Fraunhofer 条件；
- 超过条件时的等长 subtrack 划分。

GPU 中保存的是 vector potential。shower 结束时它被合并回原
`TimeDomainObserver`，随后仍由原 `RadioProcess::endOfShower()` 执行：

\[
\mathbf E_i =
 -\left(\mathbf A_{i+1}-\mathbf A_i\right)f_s .
\]

因此输出格式、时间轴定义和 Parquet writer 没有分叉。

## 5. 确定性波形累加

### 5.1 为什么不能保留 `atomicAdd(double)`

大量 thread 可能同时写入同一个 waveform bin。浮点加法不满足结合律，
`atomicAdd(double)` 的到达顺序取决于 warp/block 调度。即使同一机器上
通常看起来稳定，它也不能构成逐位可重复保证。

### 5.2 生产默认：checked fixed-point atomic sum

确定性模式使用 64 位有符号定点整数：

```text
CoREAS scale = 2^62 / field_limit
ZHS scale    = 2^62 * observer_sample_rate / field_limit
```

每个贡献先执行 round-to-nearest，再用 CAS 循环进行带符号溢出检查。
整数加法是精确、可交换和可结合的，所以最终结果不依赖 CUDA 调度顺序。

若单个贡献超出范围或累计和溢出：

```text
fixed_point_overflows > 0
```

后端立即拒绝输出并终止当前 shower，禁止产生静默回绕的波形。应用提供：

```text
--gpu-radio-field-limit <V/m>
```

用于研究超常强信号时显式增大范围。默认值为 `1.0`。

`--gpu-deterministic false` 仍保留直接 double atomic 路径，主要用于性能
剖析；科研生产默认值为 `true`。

## 6. HybridCascade 集成

### 6.1 驻留 GPU 轨迹

`CudaEmBackend` 在 lepton transport kernel 完成、记录仍驻留显存时调用：

```cpp
radio_accumulator_.accumulateLeptonTracksOnDevice(
    pipeline.transport.records, pipeline.transport.count);
```

GPU 射电启用时，紧凑 profile projection 可以继续使用，完整
`EmStepRecord/RadioTrackRecord` 不再为了 CPU 射电而回传。

### 6.2 CPU 标量与 fallback 轨迹

以下轨迹仍通过原 CPU `RadioProcess`：

- 主栈中的非 GPU 标量步骤；
- 稀有过程 CPU fallback 后在 CPU 上形成的步骤；
- 当前 GPU 能力边界外的介质或过程。

shower 结束时：

1. GPU waveform 下载一次；
2. 与 CPU 已累计 waveform 相加；
3. 调用原 CoREAS/ZHS `endOfShower()`；
4. 写入同一套 `observers.parquet`。

这样不会丢失混合级联中的 CPU 分支。

### 6.3 双缓冲 stream overlap

射电内核不再和 EM transport 串行放在默认 stream 中。每个 wavefront 使用：

```text
default EM stream
  -> record input-ready event

radio stream
  -> wait input-ready
  -> valid-track + CoREAS + ZHS
  -> record radio-done
```

EM 级联有两个轮换 workspace。第 \(n+1\) 轮可以写另一个 workspace，同时
radio stream 读取第 \(n\) 轮记录。只有某个 workspace 即将再次作为输出
缓冲时，host 才等待该槽对应的 `radio-done`；它不会执行 device-wide
synchronization。radio stream 自身保持批次顺序，但两个槽的 event 允许
EM 与射电并行。

### 6.4 有效轨迹语义

resident transport 会产生少量零时长或零位移 bookkeeping record。原
`CorsikaOutputSink::onRadioTrack()` 在调用 CPU RadioProcess 前过滤它们。
GPU 新增设备端有效轨迹计数 kernel，并采用相同条件：

```text
duration > 0 && squared_displacement > 0
```

10 GeV 对照中：

```text
resident lepton records: 14,378
有效 radio tracks:       13,942
CPU radio tracks:        13,942
```

436 条无效记录不产生任何波形贡献。

## 7. 数值验证

### 7.1 单轨迹测试

`tests/gpu/testGpuRadioProjection.cpp` 使用：

- 10 GeV electron；
- 已知直线轨迹；
- 固定折射率；
- 一个观测器；
- CoREAS 与 ZHS；
- 会触发 ZHS Fraunhofer subdivision 的几何。

测试覆盖：

- 三偏振 CPU/GPU 波形；
- 有效/无效轨迹过滤；
- track-observer pair 数；
- CoREAS/ZHS contribution 数；
- ZHS subtrack 数；
- fixed-point overflow 为零；
- reset 后第二次执行逐位相同。

结果：

```text
GPU radio projection: 19 checks passed
```

单轨迹定点量化后的最大相对差约为：

```text
4.1e-10
```

### 7.2 10 GeV 完整 shower

共同配置：

```text
primary: electron
energy: 10 GeV
seed: 24025
observers: 8 CoREAS + 8 ZHS
GPU EM min batch: 1
radio field limit: 1 V/m
```

设备统计：

```text
photon steps:          1,718
lepton steps:         14,378
valid radio tracks:   13,942
track-observer pairs: 230,048
CoREAS contributions: 32
ZHS contributions:    3,729
ZHS subtracks:        124,182
fixed-point overflow: 0
waveform D2H:          154,024 bytes
```

相同 shower 的 CPU/GPU 波形差：

| 算法/分量 | relative \(L_2\) | max absolute difference |
|---|---:|---:|
| CoREAS Ex | \(2.54\times10^{-6}\) | \(1.04\times10^{-19}\) V/m |
| CoREAS Ey | \(3.55\times10^{-8}\) | \(1.06\times10^{-19}\) V/m |
| CoREAS Ez | \(3.64\times10^{-6}\) | \(9.90\times10^{-20}\) V/m |
| ZHS Ex | \(1.51\times10^{-5}\) | \(3.03\times10^{-19}\) V/m |
| ZHS Ey | \(2.43\times10^{-7}\) | \(2.97\times10^{-19}\) V/m |
| ZHS Ez | \(4.11\times10^{-5}\) | \(2.95\times10^{-19}\) V/m |

较大的相对数出现在本身接近零的弱偏振；绝对误差保持在
\(3.1\times10^{-19}\,\mathrm{V/m}\) 以下。

### 7.3 完整进程可重复性

相同 executable、GPU、配置和 seed 运行两个独立进程：

```text
CoREAS SHA-256
4a019063cd2e5e16111e774ca7522367101a705b35c593fa4603ebb851e6ad6a

ZHS SHA-256
d47ad47a1a9cb9daf5f5dfcfac04de3b0f6c2e943b37c7393bb5c05c8a6d69c6
```

两次运行中相应 Parquet 文件的哈希完全相同，Arrow 表和全部数值列也逐
元素相等。

### 7.4 1 TeV 百万轨迹验证

1 TeV electron、相同 GPU EM history、8+8 个观测器的结果为：

```text
GPU particles:          1,549,634
valid radio tracks:     1,337,590
track-observer pairs:  22,094,560
CoREAS contributions:  19,344,691
ZHS contributions:     52,159,256
ZHS subtracks:         10,755,399
radio device time:        338.5 ms
fixed-point overflow:          0
```

CPU/GPU 全波形比较：

| 算法 | 全三分量 relative \(L_2\) | max absolute difference |
|---|---:|---:|
| CoREAS | \(1.08\times10^{-8}\) | \(3.28\times10^{-17}\) V/m |
| ZHS | \(1.61\times10^{-8}\) | \(9.50\times10^{-17}\) V/m |

同种子第二次独立 GPU 运行得到完全相同的文件：

```text
CoREAS SHA-256
8d4ae9141de986978f490d472cd372d96c9b5db0c7665368e139b7af1d87b4a2

ZHS SHA-256
ca663863c470734deb657fb7cf05032cb69505bfc5ea9cb204f3cd359477b39d
```

## 8. 阶段性能

在很小的 10 GeV shower 中：

```text
CPU radio hybrid run:             1407 ms
deterministic GPU, serial radio:    949 ms
deterministic GPU, stream overlap:  704 ms
```

stream overlap 后相对 CPU 约为 \(2.0\times\)。小 shower 只有约 14k
条轨迹，初始化、781 个细小 lepton wavefront 和 CAS 开销占比较高，因此
这个数字不是高能级联的最终加速比。

1 TeV 百万轨迹对照为：

| 路径 | Hybrid run | 完整进程 wall time | peak RSS |
|---|---:|---:|---:|
| CUDA EM + CPU radio | 31.09 s | 34.68 s | 597 MB |
| CUDA EM + GPU radio | 2.44 s | 6.14 s | 294 MB |
| speedup | \(12.7\times\) | \(5.65\times\) | — |

相对于 CPU 射电路径，GPU 路径还消除了约 14k 条完整轨迹记录的 CPU
后处理；在 1 TeV 中消除了约 134 万条记录的 CPU 射电重放。两种能量最终
都只回传约 154 kB 波形。

`kernel_launch_time_ms` 是 host launch 开销；`device_time_ms` 由 radio
stream 的 CUDA events 测得。后者不再混入默认 EM stream 的 elapsed。

## 9. 当前边界与下一阶段

当前实现明确限定为：

- `TabulatedFlatAtmospherePropagator`；
- `TimeDomainObserver`；
- 当前应用中的 CoREAS/ZHS 算法；
- resident \(\gamma/e^\pm\) 级联，CPU fallback 与 CPU 波形合并；
- double 物理计算，确定性定点仅用于最终波形和。

下一阶段仍需：

1. 运行 1 TeV、1 PeV、\(10^{17}\) 和 \(10^{18}\,\mathrm{eV}\) 射电矩阵；
2. 覆盖多个天顶角、观测器数量和 thinning 配置；
3. 进一步用 CUDA event 分离 CoREAS、ZHS 和 CAS contention；
4. 根据高能信号验证默认 fixed-point range，保留 overflow hard failure；
5. 对高占用时间 bin 评估 block-local aggregation，减少全局 CAS 冲突；
6. 将当前同步 summary/D2H 进一步改为 pinned async transfer；
7. 完成全应用 \(10^{17}\)–\(10^{18}\,\mathrm{eV}\) 物理和性能验收。
