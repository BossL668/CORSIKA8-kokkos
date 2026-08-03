# Phase 45：稳定三维 GPU wavefront 分桶

## 1. 目标

原始计划要求每个 GPU wavefront 按

```text
PID × medium_id × energy_bin
```

分桶。这样同一个 warp 中的粒子更可能执行相同的物理分支，并访问相邻的反应率
和逆 CDF 表格区域。分桶只能改变设备执行顺序，不能改变粒子的
`history_id`、随机数键、物理状态或次级粒子稳定编号。

## 2. 实现

新增文件：

- `corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp`
- `src/gpu/em/CudaWavefrontBucketing.cu`
- `tests/gpu/testGpuWavefrontBucketing.cpp`

分桶键是一个 50 bit 无符号整数：

```text
[ PID:2 ][ signed medium_id:32 ][ logarithmic energy bin:16 ]
```

能量使用每 octave 16 个 bin。无效、零或负能量映射到最低 bin；超出可表示
范围时饱和到端点，因此键生成本身不会产生 NaN、溢出或未定义转换。

设备端流程为：

1. 为原 wavefront 生成稳定输入索引 `0..N-1`；
2. 为每个粒子生成三维组合键；
3. 用 CUB `DeviceRadixSort::SortPairs` 对“键—输入索引”稳定排序；
4. 根据排序后的索引把完整 `EmParticleState` gather 到另一块预分配缓冲区；
5. 交换输入/输出缓冲区，进入原有 selection/transport/final-state pipeline。

没有使用全局原子 append，也没有在 shower 期间按 wavefront 调用
`cudaMalloc`。相同键粒子保留原相对次序，次级粒子的 history 分配仍由原有
exclusive scan 和 keyed identity 完成。

## 3. 小 wavefront 阈值

实测表明，对只有几十个粒子的 cascade 尾部执行完整 radix sort，其固定 kernel
和同步开销高于分桶收益。因此生产路径设置：

```text
MinimumWavefrontRadixSortSize = 256
```

小于 256 的 wavefront 本身已经是一个很小的调度 tile，保持原稳定顺序直接进入
物理 pipeline。输出 provenance 分别记录：

- `wavefront_bucketing.batches/particles`；
- `wavefront_bucketing.small_batches/small_particles`；
- key 定义、每 octave bin 数、稳定性和阈值。

这不是物理近似；它只决定是否执行一次纯排列优化。

## 4. 验证

`testGpuWavefrontBucketing` 覆盖：

1. PID 排序；
2. signed `medium_id` 排序；
3. 对数能量 bin 排序；
4. 相同键的稳定性；
5. 同输入重复执行的逐元素一致性；
6. 空 wavefront；
7. 单粒子 wavefront；
8. 超过生产阈值的真实 CUDA 执行。

受影响的 photon/lepton resident oracle 已改为先执行相同稳定分桶，再比较完整
物理记录。以下测试通过：

```text
testGpuWavefrontBucketing: 8 checks
testGpuPhotonWavefront:    52900 checks
testGpuLeptonTransport:    2697 checks
testGpuHybridRoute:        passed
```

真实 application 冒烟输出：

```text
/tmp/c8_phase46_spill_normal_smoke_v2
```

其中执行了 8 个大 wavefront 分桶和 422 个小 wavefront 旁路；严格能量账本
完整，相对闭合误差为

\[
6.5848822917282037\times10^{-16}.
\]

## 5. 性能解释

10 GeV 的小 shower 主要由短尾组成，不适合用来评价分桶收益。生产阈值把该场景
相对“所有 wavefront 都排序”的 kernel 时间从约 411 ms 降到约 328 ms。最终
性能结论仍以 Release、热缓存、1 PeV 及以上的五次中位数测试为准；分桶统计会
随正式性能报告一并保存。
