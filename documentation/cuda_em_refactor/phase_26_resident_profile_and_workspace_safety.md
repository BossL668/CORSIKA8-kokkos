# 阶段 26：GPU 常驻纵向剖面、确定性累加与工作区安全边界

## 1. 本阶段解决的问题

阶段 25 已经把 CoREAS/ZHS 的波形投影放在 GPU 上，但电磁输运的每一个
step 仍被压缩成 `ProjectedEmStepRecord` 并返回 CPU。CPU 随后逐条调用
`LongitudinalWriter::writeProjected()` 和
`EnergyLossWriter::writeProjected()`。

该路径物理上正确，但数据量仍然与总 step 数成正比。固定 seed 的 1 PeV
electron shower 有 17,051,645 个 GPU step，旧路径产生了约 1.966 GB
device-to-host 数据。它同时造成三个问题：

1. PCIe/WSL 映射传输与 shower 粒子数线性增长；
2. CPU 仍需逐条遍历 GPU 已经算完的轨迹；
3. 主机保存一个 wavefront 的投影记录，峰值 RSS 较高。

本阶段把纵向粒子数、能量沉积和相关统计改成常驻 GPU 直方图。shower
结束前只下载约 `O(number_of_bins)` 的结果，然后合并到原 CORSIKA 8
writer。

## 2. 设备端直方图

### 2.1 配置

`GpuEmConfig::ProfileProjection` 新增：

```cpp
bool accumulate_on_device;
std::size_t output_bin_count;
double output_bin_width_g_per_cm2;
double energy_loss_threshold_g_per_cm2;
double fixed_point_weight_limit;
double fixed_point_energy_limit_GeV;
```

`c8_air_shower` 从现有 `ShowerAxis`、`dX`、EM cut 和初级总能量构造这些
参数。GPU 与 CPU writer 使用相同 bin 数、相同 \(dX\)，初始化时若二者
不一致立即终止。

### 2.2 纵向粒子数

设备 kernel 先用原有 `ShowerAxis` 支撑点插值，把 step 两端投影成
\(X_\mathrm{start}\) 和 \(X_\mathrm{end}\)。随后严格复现
`LongitudinalWriter::writeProjected()`：

```text
first = ceil(Xstart / dX)
last  = floor(Xend / dX)
for bin in [first, last]:
    profile[pid][bin] += weight
```

GPU 只负责 photon、electron 和 positron 三列。合并到 CPU writer 时，
electron/positron 会通过原 writer 的 `addBin()` 同时增加 `charged` 列；
强子和 μ 的 CPU 路径保持不变。

### 2.3 能量沉积

设备端复现 `EnergyLossWriter::writeProjected()`：

- 若 step 反向传播，先交换 \(X_\mathrm{start}\) 和
  \(X_\mathrm{end}\)；
- 若 \(\Delta X < 10^{-4}\,\mathrm{g\,cm^{-2}}\)，按 point-like loss
  放入起点 bin；
- 否则按每个 bin 覆盖的 grammage 比例分配能量；
- bin 范围使用与 CPU writer 相同的端点截断。

累加量已经乘以粒子权重。photoelectric 的壳层结合能不在 transport
record 的 cut deposit 中，因此单独对 compact final-state records
执行 kernel：

1. 根据稳定的 `input_index` 二分查找对应 transport record；
2. 验证 parent history 和 process ID；
3. 用
   \(E_\mathrm{deposit}=E_\mathrm{end}(1-f_\mathrm{electron})\)
   重建局域沉积；
4. 写入同一能量直方图。

任何找不到 transport record、非法能量分数或 PID 都增加
`invalid_records`，并在当前 shower 输出结束前硬失败。

## 3. 确定性和溢出保护

浮点 `atomicAdd` 的加法顺序依赖 warp 调度，不能保证逐次相同。本阶段
沿用 GPU radio 后端的策略：

- 所有 profile bin 用有符号 64-bit fixed-point 保存；
- 使用 CAS 实现带符号溢出检查的原子加法；
- 保留两个 headroom bit；
- 粒子权重与能量沉积使用独立 scale；
- 任意单次转换越界或累计溢出都会记录
  `fixed_point_overflows` 并终止 shower。

应用默认范围由初级能量确定：

```text
weight limit = 2 * primary_energy / emcut
energy limit = 2 * primary_energy
```

因此在 0.5 MeV cut 下仍保留足够的整数范围，同时对小 shower 提供很高的
小数精度。该算法的整数加法满足交换律和结合律，CUDA block/warp
调度变化不会改变结果。

## 4. 异步 profile stream

最初的正确性版本在默认 stream 上同步执行 profile kernel。1 PeV 中
profile kernel 约占 2.38 s，虽然省去了大量 D2H，却阻塞了下一次物理
wavefront。

当前版本建立一个独立的 non-blocking CUDA stream，并为两个物理
workspace 分别建立 start/done event：

```text
default stream:  EM pipeline 0 ---- EM pipeline 1 ---- EM pipeline 2
profile stream:       profile 0 -------- profile 1 -------- profile 2
radio stream:         radio 0 ---------- radio 1 ---------- radio 2
```

只有当物理 pipeline 准备复用某个 workspace 时，才等待仍在读取该
workspace 的 profile/radio event。profile、radio 和下一 wavefront 的
物理计算因此可以重叠。

为避免每个 wavefront 下载 profile counters，photon endpoint compaction
现在直接在原有 16-byte summary 中返回 cut 数。全部 profile counters
只在 shower 末尾下载一次。

## 5. 输出合并

新增：

```cpp
bool CudaEmBackend::gpuProfileEnabled() const;
GpuProfileResult CudaEmBackend::downloadProfile();
```

`HybridCascade` 的顺序是：

```text
router.endOfShower()
output.endOfShower()
```

所以 router 可以在 Parquet writer flush 前下载 GPU histogram。
`CorsikaOutputSink::onGpuProfile()` 验证 bin 数、有限性、非负性和错误
counter，然后调用：

```cpp
LongitudinalWriter::addBin(...)
EnergyLossWriter::addBin(...)
```

CPU 粒子已经写入的贡献不会被清零；GPU EM 贡献只是加到同一 profile。

## 6. 工作区硬上限与安全 checkpoint

两个物理 arena 都有显式 `byteLimit()`。初始化阶段通过与真实
`appendPhotonDevicePipelineWorkspace()` 和
`appendLeptonDevicePipelineWorkspace()` 相同的 sizing 函数，二分求出：

```text
maximum_resident_photon_batch
maximum_resident_lepton_batch
```

router 在上传前按二者较小值分块。若内部 continuation front 仍超过
arena 上限，resident cascade 返回：

```cpp
workspace_limit_checkpoint = true;
remaining_photons / remaining_leptons;
```

checkpoint 是无损的，不会静默丢粒子。生产 thinning
`emthin=1e-4, max-weight=100` 的 1 PeV 测试中：

```yaml
workspace_limit_checkpoints: 0
input_batch_splits: 0
maximum_input_batch: 214424
maximum_resident_photon_batch: 约 584000
maximum_resident_lepton_batch: 约 370000
```

此前近乎不 thinning 的压力测试确实超过工作区容量。安全 checkpoint
可以继续运行，但大量 front 返回主机会造成很差的性能；它是失败保护，
不是生产调度策略。后续仍需用统一的 device-resident photon/lepton
cross-queue 处理该极端情形。

## 7. 验证结果

### 7.1 10 GeV

相同 seed、相同 GPU 物理路径，比较：

- resident GPU histogram；
- `--gpu-full-step-records` 的逐 step CPU writer 路径。

结果：

```text
profile.parquet: 每个 bin 完全相同
dEdX.parquet:    每个 bin 完全相同
D2H:             9.94 MB -> 0.54 MB
host postprocess:约 4.8 ms -> 0.08 ms
```

### 7.2 1 TeV

样本包含 575,938 个 GPU step 和 2 次 photoelectric：

```text
profile.parquet: 每个 bin 完全相同
dEdX.parquet:    每个 bin 完全相同
D2H:             381.75 MB -> 7.22 MB
shower runtime:  2.000 s -> 1.606 s（同步 histogram）
async runtime:   1.500 s
peak RSS:        约 357 MiB -> 约 253 MiB
```

同步版与异步版文件哈希：

```text
profile:
48452692fd19a3960b17e666295aef53635586c2c488abe545ff391106cee21b

dEdX:
978f946a3ff160ca8a9e05add0135fcf5604f6c74d2d27bcf166d7645e68e091
```

### 7.3 1 PeV + GPU CoREAS/ZHS

固定 seed 24027，electron primary，`emthin=1e-4`，
`max-weight=100`，8 个 CoREAS + 8 个 ZHS observer：

| 路径 | shower runtime | D2H | 进程墙钟 | peak RSS |
|---|---:|---:|---:|---:|
| 旧 projected-step D2H | 23.504 s | 1.966 GB | 29.55 s | 约 611 MiB |
| 同步 resident profile | 23.347 s | 316.21 MB | 28.89 s | 约 353 MiB |
| 异步 resident profile | 20.599 s | 315.89 MB | 26.07 s | 约 352 MiB |

同步与异步版本的四类科研输出逐字节相同：

```text
profile:
ff708c819c4ca99b5a33863adbe38586dbbc3b9c771ae3aea903d310a2115f98

dEdX:
3e36a259717755c259d7f362c8c4131743425d9f3fba8a7e67719f377f941c1b

CoREAS:
aa435abc26dca07fd81942fec86b6023ddf12d1115c0d435474d2b2673a5a517

ZHS:
13851763c3b228cc3492ab47da9dd972e9050fb49e80df7f4da807b95521940f
```

profile 与 radio 的 `fixed_point_overflows` 均为 0，
profile `invalid_records` 为 0。

## 8. 自动测试

`testGpuPhotonWavefront` 新增：

- 工作区最大 batch API 与统计一致性；
- 超过工作区上限时的无损 photon checkpoint；
- resident profile 不返回逐 step/final-state records；
- profile step/bin/counter 一致性；
- photon track 和 photoelectric deposit 非零；
- resident profile 的 D2H 小于 host-projected oracle。

全部 GPU 测试当前为：

```text
24/24 passed
```

## 9. 下一步瓶颈

1 PeV 异步版本仍有约 315.9 MB D2H。主体不再是 profile，而是：

- photon final state 产生的 \(e^-/e^+\) 返回主机后重新分批；
- lepton brems/annihilation 产生的 photon 返回主机后重新分批；
- CPU-only 稀有末态、observation 和 checkpoint 的小量记录。

下一阶段应建立 GPU 常驻的 photon/lepton cross-species 双队列，使
`gamma -> e+/e- -> gamma` 循环不经过主机。CPU 只接收真正的
PROPOSAL 指定末态 fallback、非 EM 粒子和最终 observation。该重构也是
支持弱 thinning UHE shower 而不触发 host checkpoint 的关键。
