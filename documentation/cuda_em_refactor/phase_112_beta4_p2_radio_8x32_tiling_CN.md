# 阶段 112：beta4 P2 射电 8×32 tile 优化

## 1. 目标与回滚点

本阶段继续优化 `--radio-backend cuda`，但不修改 CoREAS/ZHS 物理公式、折射率模型、
波形定点累计、EM 输运、PROPOSAL 表或随机流。修改前的 P1 提交为 `f39bc855`，并
建立了本地 tag：

```text
beta4-pre-p2-propagation-20260819
```

P1 使用 `4 tracks × 64 observers = 256 threads/block`。P2 的目标是在保持 256 个线程
和所有浮点表达式不变的前提下，降低每个 block 的 shared-memory 占用并提高 track
方向的并行度。

## 2. 最终采用的修改

生产代码只修改了 `src/gpu/em/CudaRadioAccumulator.cu` 中两个编译期常量：

```cpp
constexpr unsigned int RadioTrackTileSize = 8;
constexpr unsigned int RadioObserverTileSize = 32;
static_assert(
    RadioTrackTileSize * RadioObserverTileSize ==
    ThreadsPerBlock);
```

即每个 block 从 4 条 track、64 个 observer 改为 8 条 track、32 个 observer。线程总数
仍为 256；每个合法 `track × observer` 对仍恰好执行一次相同的 CoREAS/ZHS 设备函数。
`testGpuRadioProjection.cpp` 同步更新预期 tile 数，原有标量参考、reset、定点溢出和
逐位重复性检查保持不变。

这项修改没有新增命令行参数。`gpu_em/summary.yaml` 中已有的 `track_tile_size` 与
`observer_tile_size` 会分别记录 8 和 32。

## 3. 为什么这项修改安全

- track 与 observer 的逻辑笛卡尔积不变，只改变它们映射到 block/thread 的方式；
- 每个线程执行的 CoREAS/ZHS 数学表达式和输入数据不变；
- 波形仍使用原来的 checked 64-bit fixed-point atomic add，线程调度顺序不会改变最终
  整数累计结果；
- shower tree、粒子随机数和 EM/hadronic 路由完全不经过这两个 tile 常量；
- P1 的双缓冲 track workspace、显存预算和输入槽事件依赖均未改变。

CUDA 资源检查显示寄存器数不变，而 shared memory 明显下降：

| kernel | P1 4×64 | P2 8×32 | 变化 |
|---|---:|---:|---:|
| fused CoREAS/ZHS | 10,848 B | 6,336 B | -41.59% |
| CoREAS-only | 5,728 B | 3,776 B | -34.08% |
| ZHS-only | 5,728 B | 3,776 B | -34.08% |

对应寄存器数分别保持为 109、84 和 108；track precompute kernel 仍为 66 个寄存器。

## 4. 被否决的两个候选方案

### 4.1 传播状态预计算

曾尝试在 track precompute 阶段缓存起点、终点和中点的传播状态。四个固定 seed 的
输出仍逐字节一致，但相对 P1：radio device time 平均增加 0.229%，projection time
增加 0.184%，precompute time 增加 5.94%，且 precompute kernel 寄存器从 66 增至
98。因此该修改已经完全从源代码撤回，仅保留原始测试数据作为否决证据。

### 4.2 4×64 / 8×32 运行时自适应

还测试过按“哪一种布局启动的 block 更少”逐 batch 选择 4×64 或 8×32。它把 tile
数相对 P1 平均减少 16.61%，但 radio device time 只缩短 2.52%；固定 8×32 在同一组
seed 上缩短 4.24%。这说明最少 block 数不是该 kernel 的充分代价模型，运行时分支
没有超过简单的固定布局，因此也没有进入生产代码。

## 5. 构建与回归验收

在 `corsika_venv`、Release/CUDA 构建和 NVIDIA GeForce RTX 4060 Laptop GPU 上：

- `testGpuRadioProjection`：577 项检查通过；
- 完整 CTest：34/34 通过，无失败、无 skip，总用时 238.76 s；
- `testModules`（含 FLUKA/PROPOSAL）通过，用时 218.73 s；
- `git diff --check` 通过。

尝试运行 Compute Sanitizer memcheck 时，WSL/NVIDIA 驱动返回
`Failed to initialize WDDM debugger interface` 和 `Device not supported`。测试程序本身
仍完成 577 项检查，但本机不能把该次 sanitizer 运行声明为通过；这属于当前 WSL
调试接口限制，而不是测试发现了 device memory error。

## 6. 固定 seed 物理一致性

测试配置为 10 GeV electron、zenith 80 degree、azimuth 180 degree、IGRF14/2027、
`emthin=1e-4`、`--gpu-min-batch 1`、GPU EM 和 GPU CoREAS/ZHS，使用 81 个外部
observer。

四个独立详细计时 seed 和三个 5-event 生产批次共覆盖 19 对 shower。对每个独立
seed 或批次比较以下七类 Parquet 文件：

1. longitudinal profile；
2. production profile；
3. energy deposit；
4. ground particles；
5. interaction records；
6. CoREAS waveform；
7. ZHS waveform。

共 49 对文件全部逐字节相同。该结论比“统计相容”更强：在这些固定输入上，P2 没有
改变 shower tree、profile、地面粒子或任一射电脉冲样本。

## 7. 性能结果

### 7.1 详细计时模式

四个 paired seed（8119001、8120001、8120002、8120003）的均值结果如下；正数表示
P2 用时减少：

| 指标 | 平均缩短 | paired median |
|---|---:|---:|
| shower run wall time | 2.99% | 3.32% |
| radio device time | 4.24% | 4.01% |
| projection device time | 4.26% | 4.05% |
| radio input-slot wait | 6.70% | 5.85% |
| projection tile 数 | 7.46% | 7.66% |

`--gpu-detailed-stage-timing` 会增加 CUDA event 同步，只用于分解设备阶段，不作为最终
生产端到端指标。

### 7.2 生产模式交错顺序测试

关闭详细计时后，使用两个互不重叠的 5-event seed 组。第一轮按 P2→P1 运行，第二轮
按 P1→P2 运行，以降低热身和执行顺序偏差。10 个事件合计：

| 指标 | P1 4×64 | P2 8×32 | 缩短 |
|---|---:|---:|---:|
| 外部端到端 wall time | 97.59 s | 93.88 s | 3.80% |
| shower 内部 wall time | 83.452 s | 80.289 s | 3.79% |
| radio device time | 67.859 s | 64.183 s | 5.42% |
| projection device time | 67.185 s | 63.478 s | 5.52% |
| radio input-slot wait | 46.835 s | 43.575 s | 6.96% |

两种执行顺序中 P2 的外部 wall time 均更短：反向轮次缩短 2.41%，正向轮次缩短
5.46%。因此采用固定 8×32，而不采用传播缓存或运行时自适应布局。

## 8. 数据位置与结论边界

原始输出保存在：

```text
/mnt/d/CorsikaData/corsika_validation_results/
beta4_p2_radio_8x32_tiling_benchmark_v1
```

目录同时保留 P1 executable、最终 P2、被否决的传播缓存和自适应布局结果。上述性能
数字只适用于本机 RTX 4060、该 observer 数量和测试 shower；它证明这是安全且有
实测收益的局部优化，但不替代 100 TeV--100 PeV 生产负载上的后续基准。
