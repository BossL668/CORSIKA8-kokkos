# 阶段 34：Lepton pipeline 剖析与 Epair 末态拒绝采样

## 1. 本阶段结论

阶段 33 已经证明 CUDA EM backend 能在
\(10^{17}\)–\(10^{18}\,\mathrm{eV}\) 完成多 seed shower，但详细计时表明，
电子/正电子的末态分类内核仍包含一个不适合 GPU 的逐粒子数值积分热点：

```text
electron/positron pair production
  -> Kelner--Kokoulin--Petrukhin (KKP) rho distribution
  -> repeated Romberg integration
  -> inverse-CDF root search
```

这个精确标量算法在 CPU 上很稳健，但会造成：

- 单线程执行大量分支、对数和积分；
- 同一 warp 内不同粒子的积分路径不同；
- classification kernel 使用 202 个寄存器和 272 B local stack；
- \(10^{17}\,\mathrm{eV}\) 代表 shower 的该阶段耗时约 1.12 s。

本阶段保留 PROPOSAL 的 KKP 微分分布，改用有安全上界的变换拒绝采样。最终
production kernel：

- 使用 120 个寄存器；
- local stack 降为 8 B；
- 没有 register spill；
- 四个 UHE shower 的 Epair sampler 均为零 envelope violation、
  零 CPU fallback；
- classification 降至 0.16–0.24 s；
- CUDA EM 主链相对精确积分基线缩短 24.6%–30.0%；
- 相同 GPU、配置和 seed 的完整输出仍逐字节可重复。

这不是用近似分布替代物理模型。目标密度仍是 KKP 公式；改变的是从该密度
抽样的 Monte Carlo 算法。

## 2. 如何定位热点

### 2.1 设备端分阶段计时

`CudaEmBackend` 的 detailed timing 将 lepton pipeline 拆为：

```text
selection
transport
  - transport physics
  - Moliere scattering
  - transport control
  - compaction
interaction extraction
vertex selection
final state
  - classification
  - scan
  - summary
  - write
endpoint compaction
post endpoint
```

代表性的 \(10^{17}\,\mathrm{eV}\), seed 32017 精确积分基线为：

```text
CUDA EM core total             3332.598 ms
lepton final-state classification
                               1116.425 ms
Moliere scattering              843.650 ms (approximately)
```

一次只把 Epair 的 \(\rho\) 暂时固定为零的诊断运行，使 classification
从约 1116 ms 降至约 55 ms。该实验只用于归因，代码随后被完全移除。它证明
热点来自 KKP 逆 CDF，而不是分类 kernel 的通用控制流或输出 compaction。

### 2.2 编译器资源报告

对 CUDA target 启用 `--ptxas-options=-v` 后，精确积分分类 kernel 的资源
报告为：

```text
registers = 202
stack frame = 272 bytes
spill stores/loads = 0
```

即使没有显式 spill，过高的寄存器数也会限制 occupancy；local stack
还会产生 local-memory traffic。变换拒绝采样版本为：

```text
registers = 120
stack frame = 8 bytes
spill stores/loads = 0
```

当前 WSL 环境中的 `ncu` 对 device 0 返回 `Unknown Error`，因此本阶段没有
伪造 Nsight Compute 指标，而是使用：

1. CUDA event 分阶段计时；
2. `ptxas` 静态资源报告；
3. 控制变量诊断；
4. 四个 UHE shower 端到端计时。

## 3. 尝试过但没有进入 production 的方案

### 3.1 仅拆 kernel、`noinline` 或建立 worklist

这些改动能改变寄存器分配，但不能删除每个 Epair interaction 内部的重复
积分和 root search。实测没有形成稳定的 UHE 收益，因此没有保留。

### 3.2 固定阶数 Gauss 积分

固定阶数能减少 warp divergence，但最佳测试误差仍约为
\(2.4\times10^{-3}\)，高于 \(10^{-3}\) 物理验收阈值，因此被拒绝。

### 3.3 规则三维 \(\rho\) 逆 CDF 表

KKP 允许区间和阈值行为随能量、\(v\) 与介质组分变化。直接在规则三维网格
插值会跨越这些不连续边界，压力测试最大归一化误差约为 0.84，因此不能作为
生产算法。

表格式 version 10 仍保留可选的实验性 Epair rho table schema，便于后续
研究，但默认 table generator 不生成它，运行时 production sampler 也不
依赖它。已有 version 9 production rate table 继续受支持。

### 3.4 精确积分加有限 refinement

以粗略逆解为初值、再进行六次精确 refinement，可把最大归一化
\(\rho\) 误差压到 \(7.30\times10^{-4}\)。但初值和 refinement 本身仍需
昂贵积分，在 UHE shower 中没有取得足够收益，因此没有进入 production。

## 4. 变换拒绝采样

### 4.1 原目标分布

令 PROPOSAL KKP 微分权重为

\[
w(\rho; E,v,Z),
\]

且允许区间为

\[
-\rho_{\max}\leq\rho\leq\rho_{\max}.
\]

分布关于零对称，所以算法先抽取
\(|\rho|\in[0,\rho_{\max}]\)，最后用独立随机 bit 选择符号。

### 4.2 为什么对变量做变换

直接对 \(\rho\) 均匀 proposal，在部分能量和 \(v\) 区域会有较差的
acceptance。production 实现令：

\[
\rho(u)=\rho_{\max}\sin^2\left(\frac{\pi u}{2}\right),
\qquad u\sim U(0,1).
\]

Jacobian 为：

\[
\frac{d\rho}{du}
=\frac{\pi\rho_{\max}}{2}\sin(\pi u).
\]

去掉与 \(u\) 无关的常数后，变换空间的目标密度为：

\[
g(u)\propto
w\!\left(\rho_{\max}\sin^2\frac{\pi u}{2}\right)\sin(\pi u).
\]

这使区间端点自然衰减，并显著改善拒绝采样效率。

### 4.3 确定性 envelope

`estimateTransformedKkpMaximum()` 使用固定工作量：

1. 8 个 midpoint 扫描点定位单峰区间；
2. 12 次固定 golden-section refinement；
3. 对所得最大值乘 1.05 safety factor。

固定循环次数避免不同 thread 因收敛条件不同而产生额外 divergence。
测试同时对参数网格做 dense scan；若运行时发现：

```text
g(u) > envelope
```

该 interaction 不会被错误接受，而是记录 envelope violation 并返回
CPU PROPOSAL 指定末态生成器。最多 256 次 trial；若全部拒绝，同样显式
fallback。

### 4.4 随机数映射

候选点与接受随机数使用 Philox 的独立 draw ID 空间：

```text
candidate draw IDs: 0x100 + trial
acceptance draw IDs: 0x200 + trial
```

键仍由：

```text
seed, shower_id, history_id, step_id, process_id, draw_id
```

唯一确定。因此改变 GPU wavefront 排序不会改变某个 history 的抽样结果。
同一个 interaction 的 trial 数不会消耗或移动其他粒子的随机流。

### 4.5 失败策略

以下情况全部是显式、可统计的 CPU fallback：

- envelope 无法构造；
- 实际密度越过安全 envelope；
- 256 次 trial 耗尽；
- 非有限值或非法输入；
- medium component 不存在。

禁止静默夹断概率、强制接受或丢弃 interaction。summary 新增：

```yaml
epair_sampler:
  rejection_trials: ...
  zero_weight_samples: ...
  cpu_fallbacks: ...
  envelope_violations: ...
```

## 5. 实现位置

核心文件：

```text
corsika/gpu/em/EpairFinalState.hpp
src/gpu/em/CudaBremsFinalState.cu
corsika/gpu/em/Types.hpp
corsika/gpu/em/CudaBremsFinalState.hpp
corsika/gpu/em/CudaLeptonSelectionTransport.hpp
src/gpu/em/CudaLeptonSelectionTransport.cu
src/gpu/em/CudaEmBackend.cu
tests/gpu/testGpuEpairFinalState.cpp
applications/c8_air_shower.cpp
```

主要入口为：

```cpp
sampleEpairRhoRejection(...)
```

它返回：

```cpp
EpairRejectionSample {
  EpairFinalStateSample sample;
  uint32_t trials;
  uint32_t envelope_violations;
};
```

classification kernel 只在 `status == Success` 时在 GPU 构造
\(e^-e^+e^\pm\) 末态；其他状态沿现有指定过程 fallback 通道返回 CPU。

## 6. 单元与统计验证

### 6.1 精确参考实现

原 Romberg + inverse-CDF 实现仍作为测试参考，而不是生产 hot path。
1024 个跨能量、\(v\) 和介质组分的 interaction 与 PROPOSAL 参考比较：

```text
maximum normalized rho error = 5.26914e-4
```

这先确认了用于评价新 sampler 的 KKP 参考实现本身满足 \(10^{-3}\)。

### 6.2 Host/device 确定性

同一组 Philox key 在 host 和 CUDA device 上产生相同：

- sampler status；
- trial count；
- \(\rho\)；
- \(\rho_{\max}\)；
- envelope violation count。

### 6.3 Envelope 网格

测试覆盖：

- dry-air 三个介质组分；
- \(20\,\mathrm{MeV}\) 至 \(10^{14}\,\mathrm{MeV}\)；
- 11 个从 \(v_{\min}\) 附近到 \(v_{\max}\) 附近的坐标；
- dense \(u\) grid 与随机参数点。

结果：

```text
maximum observed local maxima = 1
minimum sampled acceptance     = 0.0719313
minimum systematic acceptance = 0.194215
envelope violations           = 0
```

低 acceptance 的随机极端点仍远离 256 trial 上限；系统物理网格的最小
acceptance 更高。

### 6.4 分布检验

三个代表性物理配置，每个配置 49,152 个样本，与数值积分得到的变换 KKP
CDF 比较：

```text
maximum KS distance = 0.00837101
acceptance limit    = 0.025
total trials        = 81117
```

此外，故意移除 medium component 的测试会稳定产生两个 sampler fallback，
证明失败路径可达、可重复且不会越界写入。

### 6.5 GPU 测试套件

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  cmake --build /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuEpairFinalState -j2

/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  ctest --test-dir /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -R '^testGpu' --output-on-failure
```

最终结果：

```text
24/24 GPU tests passed
```

## 7. UHE 性能与安全性

生产算法使用 8-point envelope scan 与 12-step refinement。四个 shower
使用阶段 33 的相同 UHE 配置：

| 能量 | seed | GPU 粒子 | Epair | rejection trials | zero weight | fallback / violation | classification | 新 core | 精确基线 | 加速 | 时间缩短 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(10^{17}\) eV | 32017 | 962,167 | 2,432 | 4,434 | 24 | 0 / 0 | 189.448 ms | 2512.379 ms | 3332.598 ms | 1.3265x | 24.61% |
| \(10^{17}\) eV | 32018 | 979,916 | 2,380 | 4,522 | 31 | 0 / 0 | 163.869 ms | 2269.002 ms | 3132.173 ms | 1.3804x | 27.56% |
| \(10^{18}\) eV | 33018 | 2,107,997 | 3,929 | 7,059 | 42 | 0 / 0 | 242.616 ms | 3318.377 ms | 4738.753 ms | 1.4280x | 29.97% |
| \(10^{18}\) eV | 33019 | 2,294,524 | 3,981 | 7,275 | 53 | 0 / 0 | 242.628 ms | 3661.619 ms | 4927.163 ms | 1.3456x | 25.69% |

四个输出均满足：

```text
complete = true
CUDA errors = 0
queue overflows = 0
Epair CPU fallback = 0
Epair envelope violations = 0
```

代表性加权能量沉积与 radio track 数为：

| 能量/seed | weighted deposit | radio tracks |
|---|---:|---:|
| \(10^{17}\) eV / 32017 | 79,965,936.922669 GeV | 903,298 |
| \(10^{17}\) eV / 32018 | 85,326,186.382307 GeV | 919,154 |
| \(10^{18}\) eV / 33018 | 719,335,329.573479 GeV | 1,920,147 |
| \(10^{18}\) eV / 33019 | 788,530,688.167737 GeV | 2,083,370 |

seed 32017 的新 sampler production run 完整重复一次，对全部 `*.parquet`
与 `*.npz` 比较：

```text
binary_differences = 0
```

精确 inverse-CDF 与拒绝 sampler 使用不同的随机数映射，所以两种算法之间
不要求同 seed 逐事件相同；验收对象是物理分布一致。拒绝 sampler 自身在
相同 GPU 和配置上的重复运行则要求逐字节一致。

## 8. CoREAS/ZHS 与粒子输运的时序

### 8.1 原 CPU CORSIKA 8

`corsika/detail/modules/radio/RadioProcess.inl` 的
`RadioProcess::doContinuous(step)`：

1. 在每个带电粒子输运 step 形成后被 process sequence 调用；
2. 过滤掉非电子和非正电子；
3. 对 \(e^-\) 或 \(e^+\) 立即调用 `implementation().simulate(step)`；
4. CoREAS/ZHS 把该轨迹段对各 observer 的贡献累加到内存波形。

因此射电计算依赖的是完整轨迹段：

```text
pre position/time/direction/energy
post position/time/direction/energy
particle charge and weight
```

它不是只看 shower 最后剩下哪些粒子。若只保留最终粒子状态，轨迹的加速、
减速、转向和相位信息已经丢失，不能重建等价射电信号。

`RadioProcess::endOfShower()` 才执行：

- 读取已累计的 waveform；
- 对 ZHS vector potential 做时间差分；
- 写 `observers.parquet`；
- reset observer。

所以“shower 结束时写出”不等于“shower 结束时才计算”。

### 8.2 当前 CUDA 路径

GPU 数据流为：

```text
resident lepton wavefront
  -> device LeptonTransportRecord[]
  -> CudaRadioAccumulator::accumulateLeptonTracksOnDevice(...)
  -> device-resident CoREAS/ZHS waveform
  -> next wavefront
  -> shower end: one waveform D2H + writer flush
```

调用位于 `src/gpu/em/CudaEmBackend.cu` 的 lepton pipeline 中。轨迹记录仍在
设备上时就被射电 kernel 消费，不需要把百万条 track 先传回 CPU，也没有
等全部 shower 完成后重新遍历粒子。

CPU fallback 或标量主栈形成的轨迹仍通过
`corsika/gpu/em/CorsikaOutputSink.hpp` 构造原 `Step`，随后调用现有
CoREAS/ZHS `doContinuous()`。最终 GPU 和 CPU 累计的波形在 shower 结束
时合并，并使用原 writer 写出。

因此更准确的表述是：

```text
物理贡献：随轨迹段 / wavefront 在线累计
磁盘输出：shower 结束后统一 flush
```

## 9. 当前瓶颈与下一步

Epair classification 已从最大热点降为 0.16–0.24 s。新的主要瓶颈是
Molière multiple scattering：

| 能量/seed | transport total | Moliere |
|---|---:|---:|
| \(10^{17}\) eV / 32017 | 1135.350 ms | 840.862 ms |
| \(10^{18}\) eV / 33018 | 1597.937 ms | 1210.051 ms |

下一阶段应：

1. 分解 Molière kernel 的采样、插值、旋转与状态写回时间；
2. 检查不同 scatter multiplicity 引起的 warp divergence；
3. 评估按 scatter regime 分桶，或建立误差受控的采样表；
4. 保留当前 exact CPU/host reference；
5. 对角分布、横向展宽、地面分布与 radio waveform 重新做统计验收；
6. 优化后再次执行四个 UHE shower，而不能只依据 microbenchmark。
