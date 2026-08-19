# 阶段 113：P2 生产显存回退与 100 TeV 性能验收

## 1. 目的和配置

本阶段把阶段 112 的 radio `8 tracks × 32 observers` tile 放到 100 TeV 质子
生产负载上测试。正式接受的样本为 100 个垂直 shower，配置为：

- primary：proton（PDG 2212），能量 `100000 GeV`；
- zenith/azimuth：`0/0 degree`；
- IGRF14，epoch 2027；
- EM cut：`0.0005 GeV`；输入 `emthin=1e-6`；
- hadron/muon/tau cut：`0.3 GeV`；
- CUDA EM 和 CUDA CoREAS/ZHS；
- `gpu-min-batch=4096`，PROPOSAL 表容差 `5e-4`；
- FLUKA process backend，4 workers；
- 81 个外部 observer，ring 0。

前 50 个 seed 使用 P2 可执行文件和 70% 显存预算；后 50 个 seed 使用本阶段增加的
有界显存回退和 90% 显存预算。两部分使用同一物理表、天线文件和物理参数，但不是同一
可执行文件，因此时间结果同时给出分层值和合并值。

## 2. 生产测试发现的显存生命周期问题

原 P2 在前两个 25-event batch 完成后，于第三批报错：

```text
current CUDA radio allocation exceeds the updated device memory budget
```

失败不是波形数组过大，而是 EM 的两个 physical arena 和 radio 的双缓冲 track
precompute workspace 都保留各自的历史峰值容量。某个大 shower 扩大 EM arena 后，radio
在共享预算内已没有空间保留旧的可选 workspace，于是严格门禁终止。失败 batch 留下了
11 个单 shower 完成记录，但 manifest 未把该 batch 标为完成；正式统计只接受 manifest
中的完整 batch，因此这些残片没有进入 100-event 样本，也没有重复 seed。

## 3. 采用的有界 GPU 回退

修改集中在 `CudaRadioAccumulator.cu`，没有改变 EM 反应率、末态、随机流、CoREAS/ZHS
公式或 fixed-point waveform accumulation：

1. 更新显存预算时先 drain radio stream，再释放两个输入槽中可重建的 track cache；
2. 几何扩容不能满足预算时，改为只申请当前 batch 精确需要的容量；
3. 有部分容量时按有界 chunk 依次执行 track precompute 和 tiled projection；
4. 连一条预计算 track 都放不下时，使用纯 GPU direct projection kernel。该 kernel 从
   resident transport record 重建 track kinematics，再调用同一 `accumulateCoREAS()` 和
   `accumulateZHS()`；它不是 CPU radio fallback；
5. 若前一个 shower 使用过 direct path，在下一个 shower 开始时只释放并重建 EM arena
   的容量 cache，避免历史峰值跨 shower 传播；物理状态和队列上限不变；
6. 新增 `direct_projection_batches/records` 输出，令回退可审计。

曾测试预留最多 256 MiB radio workspace、相应缩小 EM arena 的方案。相同 seed 的运行
时间由约 86.6 s 增至 109.6 s，并改变 wavefront 切分和 shower tree；该方案已完整撤回。

## 4. 正确性验收

- `testGpuRadioProjection` 强制把 radio 预算压到只有静态波形能驻留，确认 direct 与
  tiled 路径的 CoREAS 和 ZHS 波形逐位相同，并检查新的统计字段；
- 阶段 112 的 P1/P2 固定 seed 回归仍覆盖 19 对 shower、49 对 Parquet，全部逐字节
  相同；
- 100 个正式生产 shower 全部 `complete`；queue overflow、profile/radio fixed-point
  overflow、CPU memory spill、cross-species spill 和 invalid profile record 均为 0；
- 设置 `FLUPRO=/home/yuhanglu/fluka` 后完整 CTest 为 34/34 通过，其中
  `testModules` 实际覆盖 FLUKA/PROPOSAL；总用时 214.53 s；
- `git diff --check` 通过。

因此本阶段验收的是“显存压力下不丢失 radio 物理且不静默切到 CPU”。它不改变阶段
112 已经验证的 8×32 tile 数学结果。

## 5. 100 TeV 时间结果

### 5.1 当前 100-event 样本

| 样本 | N | wall mean | wall median | radio device mean | radio ns/logical pair |
|---|---:|---:|---:|---:|---:|
| P2 8×32，70% | 50 | 69.480 s | 69.058 s | 54.342 s | 4.405 |
| P2.1 有界回退，90% | 50 | 67.376 s | 67.209 s | 50.460 s | 4.389 |
| 合并，仅作生产周转统计 | 100 | 68.428 s | 68.319 s | 52.401 s | 4.397 |

第二组相对第一组 wall mean 短 3.03%，radio device mean 短 7.14%。但第二组平均
track-observer pair 较少，所以按工作量归一化后只短 0.38%；不能把 7.14% 全解释成
算法加速。重要结果是 direct fallback 没有造成可测的单位 pair 性能崩溃，并消除了
原来的硬失败。

5 秒一次的遥测结果为：

| 样本 | GPU util mean | GPU util median | GPU util p95 | memory max |
|---|---:|---:|---:|---:|
| P2 70% | 91.31% | 99% | 100% | 4995 MiB |
| P2.1 90% | 87.47% | 99% | 100% | 6919 MiB |

均值包含事件边界、文件输出和 CPU 强子阶段；median=99% 表明主要 GPU 阶段长期接近
满载。

### 5.2 与已有标量 CPU 数据的生产周转比较

已有同一物理参数的 beta2 scalar CPU 2000-event 数据为 mean 11127.06 s、median
10276.88 s。当前 100-event CUDA 合并数据为 mean 68.43 s、median 68.32 s，对应：

- mean turnaround ratio：162.61；
- median turnaround ratio：150.42。

CPU 数据来自服务器单核任务，CUDA 数据来自本地 RTX 4060 Laptop GPU，因此这是跨机器
生产周转比，不是同主机受控硬件 speed-up。

### 5.3 为什么不能用早期 500 例直接判定 P2 退化

较早 beta4/P0-era 500 例的 wall mean 为 65.45 s，比当前合并均值短 4.55%。但旧
executable 早于 beta4 的 observation plane、连续轨迹、10 ms cut 和首相互作用对齐
修复；相同 seed 的 shower tree 和粒子工作量已经不同。当前样本的平均 GPU particle
和 track-observer pair 均约高 1.77%。所以该数字只能描述两个生产版本，不能隔离
8×32 tile 的效果。

隔离 tile 本身的受控证据仍是阶段 112 的 P1 4×64 对 P2 8×32 测试：10 个生产事件
端到端缩短 3.80%，radio device time 缩短 5.42%，且 49 对输出逐字节相同。

## 6. thinning 语义限制

本配置虽然输入 `emthin=1e-6`，但在 100 TeV 下自动得到 threshold `0.1 GeV` 和
maximum weight `0.05`。后者不高于初始粒子单位权重，`EMThinning` 的
`parentWeight >= maxWeight` 门禁使 thinning 实际不能启动；100 个事件的
`hillas_vertices/statistical_vertices/particles_discarded` 均为 0。

因此本结果必须表述为“参数匹配但实际未薄化”的 100 TeV 性能点。若要验收真正的
`1e-6` thinning，应显式给出大于 1 的 maximum weight，并另跑独立数据集。

## 7. 数据位置

最终时间表、逐 shower CSV 和图片：

```text
/mnt/d/CorsikaData/corsika_validation_results/
final_beta4_p2p1_cuda100_proton_100TeV_vertical_emthin1e-6_igrf14_2027_runtime_acceptance_v1/
```

主要文件为：

- `runtime_summary.json`；
- `per_shower_runtime.csv`；
- `runtime_and_gpu_comparison.png`。

原始前 50 例和显存回退后 50 例保留在各自 campaign 目录；失败 batch 的残片不属于
正式样本。
