# CUDA 电磁后端重构记录：阶段 23，低能终止与小批次前沿调度

## 1. 本阶段结论

本阶段解决了生产型混合级联中两个会在低能 shower 尾部集中出现的问题：

1. 低于 0.5 MeV cut 的光子和带电轻子不再因为超出反应率表格能区而回退到
   CPU；它们在 GPU 上生成显式 `ParticleCut` 终点，并把剩余动能精确计入
   能量沉积。
2. `--gpu-min-batch` 不再只是显存预分配参数。小于阈值的电磁前沿会先返回
   `ScalarCascadeStepper` 推进一步，以便 CPU 继续展开前沿；若 CPU 在精确
   几何边界没有取得进展，则下一次必须把该粒子送入 GPU，防止 CPU/GPU
   路由形成无限循环。

这两项改动已经接入真实 `c8_air_shower --em-backend cuda` 路径。完整 CUDA
回归为：

```text
23/23 tests passed
```

阶段结果仍不是最终性能验收。1 GeV 固定种子测试显示，小 shower 的 CUDA
运行时间仍由初始化、小 kernel 和输出开销主导；当前不能据此宣称相对 CPU
有加速。

## 2. 为什么低能粒子以前会错误回退

GPU 反应率表只在有限能区内定义。旧流程先查询反应率，再决定输运限制：

```text
particle
  -> rate-table lookup
  -> sample interaction
  -> transport
```

当 shower 中产生一个低于表格最小能量的粒子时，查询首先返回
`RateEnergyOutOfRange`。因此，物理上本应由 `ParticleCut` 吸收的粒子被标成
通用 CPU fallback。

正确的判定顺序应当是：

```text
particle
  -> compare with the configured physical cut
     -> below cut: GPU ParticleCut
     -> otherwise: rate-table lookup and transport
```

这里必须使用物理 cut，而不能简单把“表格最小能量”当作 cut。否则表格网格
设置会无意中改变 shower 物理。

## 3. cut 元数据从文件传播到设备

`FlatRateTable` 和 `FlatRateTableView` 现在携带

```cpp
double energy_cut_MeV;
```

该值来自版本化 `.c8emrt` 文件的 metadata，并在上传反应率数据时一起写入
设备 view。传播链为：

```text
.c8emrt metadata
  -> FlatRateTable
  -> CudaRateTable
  -> FlatRateTableView
  -> interaction selector kernel
```

相关实现位于：

- `corsika/gpu/em/tables/FlatRateTable.hpp`
- `src/gpu/em/tables/FlatRateTable.cpp`
- `src/gpu/em/CudaRateTable.cu`
- `src/gpu/em/CudaInteractionSelector.cu`

选择器采用严格的小于关系：

```text
E < cut  -> ParticleCut
E = cut  -> 继续正常输运
```

这使 CPU/GPU cut 边界含义明确，也为“恰好等于 cut”的回归测试保留了正常
物理路径。

## 4. GPU ParticleCut 状态机

### 4.1 带电轻子

`EmInteractionStatus::ParticleCut` 告诉后续轻子输运：这个输入不是查表失败，
而是一个正常物理终点。

`CudaLeptonTransport` 随后生成：

```text
limit                 = LeptonTransportLimit::ParticleCut
distance              = 0
traversed grammage    = 0
end.step_id            = start.step_id + 1
deposited kinetic E   = E_total - m_e
fallback              = none
continuation          = none
```

电子和正电子都使用剩余动能，而不是总能量。静止质量不被重复写入连续能损。

### 4.2 光子

光子路径新增：

```cpp
PhotonTransportLimit::ParticleCut
PhotonTransportRecord::cut_deposited_energy_GeV
```

由于光子没有静止质量，其终止沉积就是完整光子能量：

```text
deposited E = E_gamma
```

resident photon endpoint compaction 不为该终点创建下一轮光子，因此 sub-cut
光子在一个 wavefront 内终止。`ResidentPhotonCascadeResult::particle_cuts`
和 `GpuEmStatistics::photon_transport_cuts` 分别保存级联结果和累计统计。

### 4.3 输出能量账本

`PhysicalCudaEmRouter::recordPhotonStep()` 和轻子对应路径把 cut 沉积乘权重后
传入现有 `CorsikaOutputSink`。同一份沉积只在输出边界乘一次权重。

需要区分两个统计量：

- `gpu_em/summary.yaml::weighted_deposit_GeV` 是 GPU sink 直接收到的沉积；
- `energyloss/summary.yaml::sum_dEdX` 是 GPU 路径与 CPU 小前沿扩展路径合并
  后的 CORSIKA 总沉积。

当 `--gpu-min-batch > 1` 时，后者才是完整 shower 的能量闭合量。CPU
`ScalarCascadeStepper` 中产生的沉积不会被重复加到 GPU 专属计数中。

## 5. 表格能区与 cut 的启动契约

`gpu_em_tablegen` 的默认最小能量已由 0.6 MeV 改为 0.5 MeV，并拒绝：

```text
energy_min_MeV > cut_MeV
```

`c8_air_shower` 在 CUDA 初始化前执行同样的运行时检查。表格若不能覆盖所选
光子 cut，shower 会以 incomplete 状态终止，不能静默转入 CPU，也不能让
网格空洞改变物理。

本阶段生成的 cut 覆盖 smoke 表为：

```text
path:
  gpu_em_tables/smoke_cut_covered_100GeV.c8emrt
energy domain:
  [0.5 MeV, 100 GeV]
cut:
  0.5 MeV
requested tolerance:
  0.1
measured maximum rate error:
  0.09540860
measured maximum inverse-CDF error:
  0.07480778
payload SHA-256:
  a0cf4d9c7683c810aaefcbd549d0df5bb92b9f95bf72520eb969f825ec18eb3a
```

这个文件只用于集成 smoke test。误差阈值 0.1 不满足科研生产要求；生产表
必须通过 \(10^{-3}\) 验证。

## 6. `--gpu-min-batch` 的真实调度语义

### 6.1 小前沿扩展

`CudaEmBackend::minimumBatchSize()` 向混合路由器公开配置阈值。路由器取得
暂存电磁粒子后执行：

```text
batch size >= min_batch_size
  -> run GPU resident wavefront

batch size < min_batch_size
  -> put particles back on the CORSIKA stack
  -> advance each particle by exactly one ScalarCascadeStepper step
  -> collect newly produced EM front
  -> reconsider the threshold
```

它不是把整个小 shower 永久切换到 CPU。CPU 每次只推进一层，使强子主栈、
现有 process sequence、writer 和 CPU 稀有末态语义保持不变；前沿一旦足够
宽就重新进入 GPU。

新增统计量为：

```text
cpu_wavefront_expansion_steps_executed
particles_returned_for_cpu_wavefront_expansion
small_batch_expansions
stalled_boundary_gpu_flushes
```

### 6.2 为什么精确球面边界会形成循环

首次实现暴露了一个真实的混合调度边界条件。粒子恰好位于球形大气层边界
时，CPU tracking 可以产生一个 0 m 的 node transition：

```text
GPU staging
  -> small batch returned to CPU
  -> CPU 0 m boundary transition
  -> same EM state staged again
  -> small batch returned to CPU
  -> ...
```

这不是 CUDA kernel 死锁，而是两个各自合法的调度器组合后没有“取得进展”
的全局保证。

### 6.3 零进展保护

路由器只保存最近一次小批次扩展的有界状态，并按 `history_id` 比较：

- 位置；
- 时间；
- 能量；
- 权重；
- 方向。

如果同一 history 被重新暂存且上述状态完全相同，本批次即使小于阈值也会
强制送入 GPU。统计量 `stalled_boundary_gpu_flushes` 记录这种情况。

缓存只保留最近一个扩展批次，执行下一次 CPU 扩展或 GPU wavefront 后即
清空，因此不会随 shower 历史增长。

这个保护不是数值容差“推开”粒子，也不修改物理状态；它只改变哪一个后端
处理下一步。

## 7. 测试覆盖

### 7.1 低能终止

`testGpuLeptonTransport` 覆盖：

- sub-cut 电子；
- sub-cut 正电子；
- 低于 rate-table 网格但属于 cut 的输入；
- 直接 device pipeline；
- resident 一轮终止；
- 精确剩余动能沉积；
- 零 fallback。

`testGpuPhotonWavefront` 覆盖：

- 0.4 MeV 和 0.499 MeV 光子；
- `distance = grammage = 0`；
- 完整光子能量沉积；
- `step_id` 单调增加；
- resident 一轮终止；
- 无次级、无 observation、无 fallback；
- 等于 cut 时不被提前吸收。

### 7.2 小批次和边界保护

`testGpuHybridRoute` 覆盖：

- 小批次返回 CPU 并且只推进一步；
- 前沿重新进入 GPU；
- 统计量与实际推进次数一致；
- CPU 零进展后强制 GPU flush；
- 不重复或丢失 history。

### 7.3 完整 CUDA 回归

在 `corsika_venv`、CUDA 12.6、NVIDIA GeForce RTX 4060 Laptop GPU 上：

```text
Test project: corsika8_gpu_refactor_build_cuda
23/23 tests passed
total test time: 6.54 s
```

## 8. 固定种子真实应用对照

共同配置为：

```text
primary       = photon
energy        = 1 GeV
seed          = 8675309
EM thinning   = disabled
GPU table     = smoke_cut_covered_100GeV.c8emrt
```

结果如下：

| 后端/阈值 | 端到端时间 | GPU 粒子推进 | CPU 前沿步 | 小批次扩展 | 边界强制 GPU | kernel 时间 | 总 dEdX |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU PROPOSAL | 2.894 s | - | - | - | - | - | 0.988758 GeV |
| CUDA, min=1 | 6.185 s | 1610 | 0 | 0 | 0 | 346.9 ms | 0.988758 GeV |
| CUDA, min=16 | 3.660 s | 1478 | 74 | 14 | 2 | 471.4 ms | 0.990802 GeV |

`min=16` 相对 `min=1` 的端到端时间缩短约 1.69 倍，证明阈值已经真正改变
执行策略。但它仍比单核 CPU 慢约 26.5%，而且 kernel 时间没有降低。这符合
小 shower 的预期：GPU 初始化、表格上传、输出建库和数量很少的 device
wavefront 无法被足够并行工作摊薄。

两个 CUDA 调度模式不要求逐事件 bitwise 相同。Philox 保证同一 history 的
设备随机数不依赖 GPU 队列排列，但把一部分步交给 CPU 会改变 history 展开
和后续随机流。这里的 0.2% 总沉积差远低于 smoke 表允许的 10% 插值误差，
不能代替后续 \(10^{-3}\) 表上的统计验证。

剩余 11 次 reason 7 fallback 都是“GPU 已精确选定 photon-pair 过程后，由
CPU 生成指定末态”。当前 pair 辅助 \(\rho(E,u)\) 表从 10 GeV 开始，低能
区域仍使用这条显式且可审计的指定末态 fallback；它不是通用能力失败，也
没有删除该物理过程。

## 9. 下一阶段

下一阶段按以下顺序推进：

1. 生成并验证覆盖 0.5 MeV–100 GeV 的 \(10^{-3}\) 表，先测量 tablegen
   收敛、文件大小和初始化成本；
2. 将相同误差标准扩展到 UHE 生产能区；
3. 建立 device-resident 的 photon/electron/positron 混合队列，减少物种间
   host checkpoint；
4. 把非射电的纵向与能损直方图移到 GPU 累计，降低详细 step record 的
   D2H 流量；
5. 在 \(10^{17}\)–\(10^{18}\) eV thinning 配置下完成 CPU/GPU 物理统计和
   端到端 \(5\times\) 性能验收。

只有第 5 项满足既定标准后，才可把这个后端描述为完成科研生产加速。
