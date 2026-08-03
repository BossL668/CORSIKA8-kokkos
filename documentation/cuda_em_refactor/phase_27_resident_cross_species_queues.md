# 阶段 27：GPU 常驻 photon/lepton 跨种类队列

## 1. 本阶段解决的问题

阶段 26 已把纵向 profile、能量沉积和 CoREAS/ZHS 投影留在 GPU，但一次
电磁相互作用产生另一类粒子时，仍会发生主机往返：

```text
photon pair/Compton/photoelectric
    -> 下载 e-/e+
    -> CPU router 分类
    -> 再上传 lepton wavefront

lepton brems/annihilation
    -> 下载 photon
    -> CPU router 分类
    -> 再上传 photon wavefront
```

这不是物理上的 CPU fallback，只是旧 scheduler 的数据路由方式。1 PeV
生产样本在已经启用 resident profile 后，仍有约 225.7 MB H2D 和
315.9 MB D2H。

本阶段增加两个持久设备队列，使正常的
\(\gamma\leftrightarrow e^\pm\) 级联不再经过主机。CPU 现在只接收：

- 明确定义的 PROPOSAL 指定末态 fallback；
- observation/escape 记录；
- 工作区或跨队列容量保护触发的显式 spill；
- 需要 CPU 处理的非 EM 粒子。

## 2. 配置和设备内存

`GpuEmConfig` 新增：

```cpp
bool resident_cross_species;
```

`c8_air_shower` 的 CUDA 路径默认开启，并提供验证/回退开关：

```text
--gpu-resident-cross-species true|false
```

关闭时不分配任何跨种类队列，旧路径及其显存上限保持不变。这一行为已用
固定 seed 验证：10 GeV 的 profile、dEdX 和 ground particles 与阶段 26
基线逐字节相同。

开启时初始化两个等容量的 `EmParticleState` POD 数组：

```text
device_pending_photons
device_pending_leptons
```

同时分配：

- 一个 device-side selected-count；
- CUB `DeviceSelect::If` 临时存储；
- 不超过 512 MiB 的跨队列总预算；
- 两个物理 arena 所需的最小保留空间。

当前机器上每个 PID 队列的容量为 2,396,745 个粒子，总设备开销约
537.0 MB。全部分配都计入 `memory_fraction` 和 `peak_device_bytes`；
CUDA 后端关闭该功能时这些字段严格为 0。

## 3. photon 到 lepton 的稳定选择

photon final-state kernel 的输出同时包含：

- pair production 的 \(e^-+e^+\)；
- Compton 的 photon+electron；
- photoelectric 的 electron；
- thinning 后保留的子粒子。

不能用全局原子 append，因为它会使队列顺序依赖 warp 调度。本阶段使用：

```cpp
cub::DeviceSelect::If(
    source_secondaries,
    device_pending_leptons,
    IsChargedEmParticle{});
```

CUB 的稳定 select 保留原 secondary 数组顺序。只下载一个
`std::size_t` 计数，不下载粒子本身。队列统计记录：

```text
peak_pending_leptons
cross_species_particles_kept_on_device
cross_species_device_to_device_bytes
```

如果 source 数量保守上界超过剩余队列容量，完整 secondary batch 会走
原 host 路径，并增加 `cross_species_host_spills`；禁止截断或静默丢失。

## 4. lepton 到 photon 的设备追加

lepton endpoint compaction 已经生成稳定排序的
`generated_photons`。因此不需要第二次筛选，只需执行有边界检查的 D2D
追加：

```text
generated_photons
    -> device_pending_photons[pending_count : pending_count + count]
```

空间不足时同样返回 host，并记录一次显式 spill。

## 5. resident cascade 的输入合并

`CudaEmBackend` 新增：

```cpp
std::size_t pendingPhotonCount() const noexcept;
std::size_t pendingLeptonCount() const noexcept;
```

每次 photon/lepton resident cascade 的输入顺序固定为：

```text
已有 device pending front
随后是本次 host staging
```

两部分在设备上合并到当前物理 arena。被消费的 pending count 随后清零；
粒子的 history、parent、generation 和 step ID 全部原样保留。

router 的 batch 切分同时考虑 host 和 device 数量。photon 运算后新产生的
lepton 可能使“device pending + 本轮 host lepton”超过 lepton arena，
因此多出的 host lepton 会留到下一轮，而不是触发越界。

shower 结束前执行硬检查：

```text
host staged == 0
pending photons == 0
pending leptons == 0
```

任一条件不满足都拒绝 finalize，输出按现有异常路径标记为 incomplete。

## 6. 修复的设备纯前沿死循环

初版集成在 10 GeV 和 1 TeV 均通过，但 1 PeV 出现：

```text
CPU core: 100%
GPU utilization: 0%
无输出进展
```

根因是 router 的两个接口对“未完成”的定义不一致：

```cpp
pending()
    = !host_staged.empty()
      || pendingPhotonCount() != 0
      || pendingLeptonCount() != 0;

advanceOneWavefrontAndReturn()
    // 旧错误
    if (host_staged.empty()) return 0;
```

当 shower 首次只剩设备队列时，`HybridCascade` 正确判断尚未结束，但
router 每轮返回 0，从而形成 CPU 空循环。

修复后的入口条件是：

```cpp
if (host_staged.empty() &&
    pendingPhotonCount() == 0 &&
    pendingLeptonCount() == 0) {
  return 0;
}
```

小 batch 的 CPU scalar expansion 也只在设备 pending 为 0 时允许触发。
`testGpuHybridRoute` 现在构造真实的
lepton -> photon -> lepton 设备纯前沿，并要求 HybridCascade 自动排空两
个设备队列。该测试可直接防止上述死循环回归。

## 7. 输出 metadata

每个 CUDA shower 的 `gpu_em/summary.yaml` 新增：

```yaml
cross_species:
  enabled: true
  queue_device_bytes: ...
  queue_capacity_per_pid: ...
  peak_pending_photons: ...
  peak_pending_leptons: ...
  particles_kept_on_device: ...
  device_to_device_bytes: ...
  host_spills: ...
  final_pending_photons: 0
  final_pending_leptons: 0
```

这些字段使科研批量作业可以区分真正的 resident 运行和发生过 host
spill 的运行。

## 8. 验证结果

### 8.1 10 GeV

固定 seed 25001：

```text
GPU particles:            15,392
particles kept on device: 1,944
peak pending photons:     234
peak pending leptons:     431
host spills:              0
final pending queues:     0 / 0
H2D:                      474,040 -> 241,976 bytes
D2H:                      518,608 -> 193,016 bytes
```

新路径运行两次，以下文件逐字节一致：

```text
profile:
e25f29f5026f0f6d38d137a4ae431d0b95e71db5a30598c9a9b747892f32000d

dEdX:
f17236ef4668e40d570f27d593daf41008f6b5f54ab5d6d9b1be878528b280e0

ground particles:
8e9dcf20e4b3637f97c0abd9595724b4b3b26f56dd6ab17fee23add45824ba3f
```

### 8.2 1 TeV

固定 seed 25002，`emthin=1e-4`，`max-weight=100`：

| 指标 | 阶段 26 host cross-route | resident cross-queue |
|---|---:|---:|
| GPU particles | 575,938 | 576,015 |
| resident photon wavefronts | 51 | 35 |
| resident lepton wavefronts | 709 | 442 |
| H2D | 6.28 MB | 0.233 MB |
| D2H | 7.11 MB | 0.182 MB |
| core shower runtime | 1.500 s | 1.152 s |

新路径有 53,298 个粒子留在 GPU，两个队列峰值分别为 7,529 和
8,154，host spill 为 0。H2D/D2H 分别降低约 96.3% 和 97.4%。

两次独立进程的 profile、dEdX、ground particles、interactions、
production profile、CoREAS 和 ZHS 七类 Parquet 输出全部逐字节一致。

### 8.3 1 PeV + GPU CoREAS/ZHS

固定 seed 24027，`emthin=1e-4`，`max-weight=100`，8 个 CoREAS 和
8 个 ZHS observer：

```text
GPU particles:                 17,645,186
particles kept on device:       2,022,970
peak pending photons:             152,035
peak pending leptons:             245,790
host spills:                              0
final pending queues:                   0/0
H2D:                       225.73 MB -> 7.34 MB
D2H:                       315.89 MB -> 1.42 MB
GPU router time:             12.141 s -> 11.690 s
GPU kernel time:             14.044 s -> 12.347 s
```

D2H 降低约 99.55%。profile/radio 的 fixed-point overflow 均为 0，
profile invalid record 为 0，CUDA queue overflow 为 0。

端到端 core runtime 是 26.33 s，而阶段 26 的另一个固定事件样本为
20.60 s。这个数不能直接解释为 GPU 退化：cross-queue 改变了次级
history ID 的分配顺序，因此相同顶层 seed 会对应另一个合法 Monte
Carlo shower。本次事件产生了更多 photonuclear 强子，CPU proton/neutron
输运比旧样本多约 6 s。可直接比较的 router 和 kernel 计时均下降；正式
性能结论仍需多 seed ensemble，而不能用两个不同 shower 的单事件墙钟。

新路径两次独立运行的七类输出逐字节一致。关键 SHA-256：

```text
profile:
3db76ad9089708601dacf98464a062285858c89553dc50f36a3840d3a323a792

dEdX:
37ea203f3faa372f9e6651dbe31e9cec3d3b639f21695547b46e9d7ab6be1986

CoREAS:
6b2ad3f1b5dbbb000e1e3fcb5743f244a70f841abe7479bcf30f122382b68cee

ZHS:
047973c559a6d642eb36ef19e734699e0f5a910045f066d6721bef46a7539692
```

## 9. 自动测试

新增覆盖：

- cross-queue 关闭时不分配显存，旧结果逐字节不变；
- photon final-state 的稳定 charged select；
- lepton generated-photon 的 D2D append；
- device-only photon/lepton front 的 HybridCascade 自动推进；
- 设备队列最终归零；
- 队列计数、D2D 字节和 spill 统计；
- oversize host front 仍返回原有无损 workspace checkpoint。

全部 GPU 测试：

```text
24/24 passed
```

## 10. 当前边界和下一步

1. 跨队列当前是固定容量。超过容量会显式 spill 到 host，物理无损，但
   可能降低性能。
2. device pending front 已支持 head/tail 分块消费；front 大于 resident
   arena 时只消费当前 chunk，未消费尾部继续驻留设备。队列尾端仍不是
   环形结构，长时间不归零时可能产生连续空间碎片。
3. photon-to-lepton 的独立 8-byte count 同步已在阶段 28 合入 endpoint
   summary；更大的剩余问题是 selection、transport 和 final-state 各阶段
   仍分别由 host count 驱动。
4. 固定 512 MiB 队列对小 shower 偏大。可在实现分块消费后按物理 arena
   容量自适应设置，或改成有上限的增长式显存池。
5. 跨队列改变 history ID 的全局分配顺序。相同配置在同一 GPU 上仍逐位
   可重复，但与关闭跨队列的单事件不应逐 bin 比较。物理验收必须使用多
   seed ensemble 比较 \(X_{\max}\)、profile、ground 和 radio 分布。
