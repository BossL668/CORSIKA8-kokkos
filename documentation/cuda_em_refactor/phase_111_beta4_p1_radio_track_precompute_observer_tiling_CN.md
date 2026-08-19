# beta4 P1：射电 track 预计算与 observer tiling

## 1. 目标与基线

P1 只优化 `--radio-backend cuda` 的设备执行方式，不修改 CoREAS/ZHS 公式、折射率
快照、波形定点累计、粒子随机流或 EM 输运。P1 前基线为 Git tag
`beta4-pre-p1-20260819`，对应提交 `0826afa2`。

P0 在 RTX 4060 Laptop GPU 上发现，10 GeV 倾斜电子事例的真实 CUDA copy 仅约
14 ms，而 radio input-slot wait 约 489 ms。因此 P1 优先减少射电 projection kernel
的重复计算和全局内存读取，而不是优化 PCIe 传输。

## 2. P1 前的问题

旧 fused kernel 的逻辑线程对应一个 `track × observer` 组合。每个线程都从完整
`LeptonTransportRecord` 重新计算同一条 track 的：

- 起止位置、位移和长度；
- 时间间隔、速度向量和 beta；
- 电荷符号、权重及公共常数。

若一条 track 对应 83 个 observer，这些与 observer 无关的量会重复计算 83 次。
旧的一维 pair index 还需要整数除法和取模恢复 track/observer 索引。相邻线程虽然
读取相邻 observer，却没有把 observer 元数据显式放入 shared memory。

## 3. 实现

### 3.1 每个输入槽独立的 track workspace

双缓冲 `RadioInputSlot` 新增：

```cpp
RadioTrackKinematics* precomputed_tracks;
std::size_t track_capacity;
cudaEvent_t track_precompute_done;
```

`precomputeRadioTracksKernel` 每条输入 record 只执行一次
`makeRadioTrackKinematics()`。无效 record 写入 `valid=0` 的零记录；有效 record
写入只读 `RadioTrackKinematics`。原 valid-track 计数和可选 diagnostics 也在这个
kernel 中完成，避免再遍历一次输入。

workspace 按当前 batch 动态扩容，每个双缓冲槽独立持有，容量采用不小于需求的
二倍增长策略。扩容前把新容量计入统一 GPU memory budget；超过预算直接终止
shower，不允许静默回退或突破 `--gpu-memory-fraction`。

### 3.2 二维 observer tile

projection 使用固定的二维 tile：

```text
4 tracks × 64 observers = 256 CUDA threads/block
grid.x = ceil(track_count / 4)
grid.y = ceil(observer_count / 64)
```

每个 block 把 4 个预计算 track 放入 shared memory；同时把最多 64 个 observer
放入 shared memory。fused CoREAS/ZHS kernel 分别缓存两套 observer 元数据，但共享
同一套 track。这样每个 block 对全局内存的 track 读取从 256 次降为最多 4 次，
observer 读取从 256 次降为最多 64 次；线程不再执行 pair-index 整数除法。

波形样本仍使用原 64-bit checked fixed-point atomic add。因此线程/块顺序变化不会
改变确定性波形，overflow 仍然使 shower 失败。

### 3.3 新统计字段

`gpu_em/summary.yaml` 的 `radio` 节点新增：

- `track_precompute_enabled`；
- `track_tile_size`、`observer_tile_size`；
- `track_precompute_batches`、`track_precomputed_records`；
- `projection_tiles`；
- `track_workspace_bytes`、`maximum_track_batch`；
- `track_precompute_device_time_ms`；
- `projection_device_time_ms`。

`device_time_ms` 仍为两者之和，并包含在原有 radio device time 语义中。

## 4. 物理一致性验收

### 4.1 单元测试

`testGpuRadioProjection` 从 1 个 observer 扩展到 70 个相同 observer，强制跨越
64-observer tile 边界。测试检查：

1. GPU CoREAS/ZHS 与原标量公式的相对误差仍不超过原阈值；
2. precompute batch、record 和 tile 数量正确；
3. reset 后统计不残留；
4. 低于当前 allocation 的动态显存预算会被拒绝；
5. 两次运行的确定性 CoREAS/ZHS 波形逐位一致。

最终 Release/CUDA 全树构建通过；真实 RTX 4060 环境中 CTest 34/34 通过，无
失败、无 skip，总用时 209.54 s，其中 `testModules`（含 FLUKA/PROPOSAL）用时
191.30 s。

### 4.2 P0 与 P1 固定 seed 比较

使用 seed 8119001、10 GeV electron、zenith 80 degree、azimuth 180 degree、
IGRF14/2027、`emthin=1e-4`、GPU EM 和 GPU CoREAS/ZHS。P0 与 P1 的以下输出
SHA-256 全部逐字节相同：

1. longitudinal profile；
2. production profile；
3. energy deposit；
4. ground particles；
5. interaction records；
6. CoREAS waveform；
7. ZHS waveform。

5 个 paired seed（8120001 起）的全部 Parquet 文件也得到 0 个哈希差异。因此 P1
改变的是设备执行映射，不是 shower tree 或射电数值。

## 5. RTX 4060 性能结果

测试条件为同一 Release binary 配置、同一 GPU、同一 warm table cache、同一进程
连续运行 5 个 10 GeV electron shower，83 个外部 observer，zenith 80 degree，
azimuth 180 degree，`emthin=1e-4`。P0/P1 使用相同 seed。

| 指标（5 事件合计） | P0 | P1 | 缩短 | 加速比 |
|---|---:|---:|---:|---:|
| 外部端到端 wall time | 393.94 s | 348.82 s | 11.45% | 1.129× |
| shower `total_run` | 386.23 s | 342.07 s | 11.43% | 1.129× |
| radio device time | 43.94 s | 38.14 s | 13.22% | 1.152× |
| radio input-slot wait | 41.16 s | 35.24 s | 14.38% | 1.168× |

各事件的 `total_run` 缩短 7.31%–14.55%，radio device time 缩短
7.19%–14.63%。轨迹最多的事件含 144508 条 radio track；其 precompute 用时
343.22 ms，projection 用时 13014.65 ms，说明预计算本身远小于被优化的 projection
成本。

较小的 seed 8119001 事例中：

- radio device time：661.16 ms → 604.14 ms，缩短 8.62%；
- input-slot wait：489.31 ms → 427.47 ms，缩短 12.64%；
- shower run time：6931.66 ms → 6384.33 ms，缩短 7.90%。

这些结果表明 P1 有稳定的设备和端到端收益，但没有消除全部射电等待。ZHS
Fraunhofer subdivision、传播路径计算、原子波形累计和小 batch 调度仍是后续主要
候选。正式生产性能仍应关闭 `--gpu-detailed-stage-timing`，并用相同 seed 的多事件
paired benchmark 报告。
