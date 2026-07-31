# CORSIKA 8 中 CoREAS/ZHS 与粒子输运的执行时序

## 结论

CORSIKA 8 的 CoREAS 和 ZHS 射电计算采用：

```text
粒子每产生一个有效 e-/e+ 轨迹段
  -> 立即计算该轨迹段对所有 observer 的贡献
  -> 累加到 observer 的内存 waveform

整个 shower 结束
  -> 将已经累加完成的 waveform 写入 Parquet
  -> 清空 observer，准备下一个 shower
```

因此它不是：

```text
先保存 shower 的全部粒子
  -> shower 结束后再遍历所有粒子/轨迹
  -> 重新执行射电计算
```

需要区分“射电贡献的计算时间”和“结果写盘时间”：贡献随输运在线计算，
完整波形在 shower 结束时统一写盘。

## 1. RadioProcess 在 process sequence 中的位置

`c8_air_shower.cpp` 构造过程序列：

```cpp
auto sequence =
    make_sequence(
        stackInspect,
        neutrinoPrimaryPythia,
        hadronSequence,
        decaySequence,
        emCascade,
        prodprof,
        emContinuous,
        coreas,
        zhs,
        longprof,
        observationLevel,
        inter_writer,
        thinning,
        cut);
```

`coreas` 和 `zhs` 是普通的 `ContinuousProcess`，与连续能损、纵向 profile
和 observation process 一起参与每一个 transport step，而不是位于
`Cascade::run()` 之后的独立后处理阶段。

对应文件：

```text
applications/c8_air_shower.cpp
corsika/modules/radio/RadioProcess.hpp
```

## 2. 每个 transport step 如何触发射电

`ProcessSequence::doContinuous()` 顺序调用所有注册的 continuous
process：

```cpp
ret |= A_.doContinuous(step, ...);
ret |= B_.doContinuous(step, ...);
```

对应文件：

```text
corsika/detail/framework/process/ProcessSequence.inl
```

RadioProcess 的入口为：

```cpp
template <typename Particle>
ProcessReturn RadioProcess::doContinuous(
    Step<Particle> const& step,
    bool const);
```

实现先检查 observer 是否为空，再只接收 electron/positron：

```cpp
if (observers_.size() == 0) {
  return ProcessReturn::Ok;
}

auto const particleID =
    step.getParticlePre().getPID();

if (particleID == Code::Electron ||
    particleID == Code::Positron) {
  return implementation().simulate(step);
}
```

对应文件：

```text
corsika/detail/modules/radio/RadioProcess.inl
```

也就是说，触发单位是一个已经完成几何传播的 `Step`，其中包含：

```text
start/end position
start/end time
particle PID
direction/trajectory
energy change
thinning weight
```

photon 不直接产生轨迹电流贡献；它产生的 \(e^\pm\) 在后续 step 中触发
射电过程。

## 3. CoREAS 在线计算什么

`CoREAS::simulate(step)` 对当前一个轨迹段：

1. 读取 start/end time 和 position；
2. 由位移与时间计算速度 \(\boldsymbol{\beta}\)；
3. 读取粒子电荷和 thinning weight；
4. 对每个 observer 传播 start/end endpoint；
5. 计算接收时刻、Doppler factor 和 endpoint electric field；
6. 在 Cherenkov 附近使用 ZHS-like approximation；
7. 调用 observer 接口，把贡献加到相应时间 bin。

外层结构是：

```text
one e-/e+ step
  for every observer
    propagate start endpoint
    propagate end endpoint
    calculate endpoint fields
    observer receives contribution
```

对应文件：

```text
corsika/detail/modules/radio/CoREAS.inl
```

因此 CPU CoREAS 的主要计算代价就在 shower transport 循环内部。

## 4. ZHS 在线计算什么

`ZHS::simulate(step)` 同样立即处理当前轨迹段：

1. 读取 start/end spacetime；
2. 计算速度、电荷和 thinning weight；
3. 对每个 observer 传播轨迹中点；
4. 检查 Fraunhofer 条件；
5. 必要时把一个长 step 再划分为多个 subtrack；
6. 计算 vector potential；
7. 立即累加到 observer 的时间 bin。

对应文件：

```text
corsika/detail/modules/radio/ZHS.inl
```

ZHS 在这一阶段主要累加矢势。把矢势离散差分成电场发生在
`endOfShower()` 写盘路径，但这不等于在 shower 末尾重新遍历轨迹。

## 5. shower 末尾做什么

`RadioProcess::endOfShower()`：

```cpp
for (auto& observer : observers) {
  getAxis();
  getWaveformX/Y/Z();

  if (ZHS) {
    E = -difference(vector_potential) * sample_rate;
    write rows;
  } else if (CoREAS) {
    write accumulated E rows;
  }

  observer.reset();
}
```

最后关闭当前 shower 的 streamer，并把 shower ID 加一。

对应文件：

```text
corsika/detail/modules/radio/RadioProcess.inl
```

所以：

- CoREAS：step 中计算并累计 electric field；末尾写盘。
- ZHS：step 中计算并累计 vector potential；末尾做离散导数并写盘。

## 6. 标量 Cascade 的完整时间线

```text
取出一个 particle
  |
  v
计算 interaction/decay/boundary/continuous step 竞争
  |
  v
生成这一段 Step
  |
  +--> continuous energy loss
  +--> CoREAS::simulate(step)
  +--> ZHS::simulate(step)
  +--> longitudinal profile
  +--> observation plane
  +--> particle cut/thinning
  |
  v
处理 interaction 和 secondaries
  |
  v
取下一 particle

所有 particle 结束
  |
  v
RadioProcess::endOfShower()
  |
  v
写 observers.parquet
```

即使 CORSIKA 8 默认使用深度优先 LIFO stack，射电波形的物理和是线性的，
不同粒子轨迹段可以按到达计算器的顺序累加。CPU 实现的浮点相加顺序仍
可能造成极小的末位差异。

## 7. 当前 CUDA EM + CPU radio 路径

当：

```text
--em-backend cuda
--radio-backend cpu
```

GPU 每个 lepton transport record 生成一个 `RadioTrackRecord`，包含：

```text
start/end position
start/end time
start/end direction
start/end energy
PID
weight
```

`CorsikaOutputSink::onRadioTrack()` 把它转换成 CORSIKA `Step`，随后立即
调用：

```cpp
coreas_.doContinuous(step, false);
zhs_.doContinuous(step, false);
```

对应文件：

```text
corsika/gpu/em/CorsikaOutputSink.hpp
corsika/gpu/em/PhysicalCudaEmRouter.hpp
```

这个调用发生在 GPU wavefront 的 host postprocess 中。因此时序是：

```text
GPU 推进一批 e-/e+
  -> 返回/投影轨迹段
  -> CPU 立即将每段交给 CoREAS/ZHS
  -> observer 在线累计
  -> 下一 GPU wavefront
```

它仍不是整个 shower 后处理，只是粒度从“标量 particle step”变成
“GPU wavefront 返回的一批 step”。

这也是第一版生产合同中“保留现有 CPU CoREAS 等价轨迹段”的实现。

## 8. 当前 CUDA radio 路径

当：

```text
--em-backend cuda
--radio-backend cuda
```

且 `ring != 0` 时，GPU 在 lepton wavefront 中直接把轨迹对 CoREAS/ZHS
observer 的贡献累加到 resident device waveform。

shower 结束时：

```text
PhysicalCudaEmRouter::endOfShower()
  -> backend.downloadRadioWaveforms()
  -> CorsikaOutputSink::onGpuRadioWaveforms()
  -> observer.addWaveform()
  -> RadioProcess::endOfShower()
  -> Parquet
```

这条路径在 shower 末尾下载最终 waveform，但贡献已经随 GPU transport
在线累计。末尾没有重新遍历完整粒子历史。

## 9. 与 GPU EM 性能设计的关系

CPU radio 兼容路径的优势：

- 与现有 CoREAS/ZHS 实现共用同一代码；
- 容易做物理等价验证；
- 不需要保存完整 shower history；
- GPU wavefront 结束后即可释放该批轨迹记录。

代价：

- 每个轨迹段仍需 CPU 对所有 observer 计算；
- GPU/CPU 之间需要轨迹数据交换；
- observer 数量多时，radio 可能重新成为端到端瓶颈；
- GPU EM kernel 与 CPU radio 的重叠调度需要额外 stream/pipeline。

GPU radio 路径可以减少轨迹回传并提高 observer 并行度，但必须独立验证：

```text
CoREAS waveform
ZHS waveform
arrival time
polarization
amplitude
energy fluence
fixed-point overflow
deterministic accumulation
```

因此当前重构把“EM shower 物理验收”和“radio waveform 物理验收”分为
两个测试层。关闭 ring 的纯 EM 性能数字不包含 radio；打开 ring 后必须
单独报告 radio 的 kernel、transfer 和 CPU fallback 时间。
