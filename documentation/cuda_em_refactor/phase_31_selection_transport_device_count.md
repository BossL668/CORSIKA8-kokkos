# 阶段 31：selection count 直接驱动 transport

## 1. 原有同步

每个 photon 和 lepton wavefront 的第一段原为：

```text
interaction selection
    -> count selection fallback
    -> 4-byte D2H
    -> host 决定 raw/compact interaction 指针和 transport launch count
transport
    -> count transport fallback
    -> 4-byte D2H
```

selection 和 transport 之间没有 CPU 物理计算，因此第一次同步只服务于
GPU kernel 调度。

## 2. 为什么不能无条件复制大记录

最简单的异步化方法是每次都 scan selection flags 并把所有
`EmInteractionRecord` compact 到第二个数组。生产表正常覆盖时 selection
fallback 通常为 0，这会无条件复制整个大 interaction batch，可能比删除
一次同步更昂贵。

当前方案只无条件扫描轻量 `uint32` flag：

1. selection kernel 写 raw interaction/fallback 和 fallback flag；
2. CUB exclusive scan 生成 stable offset；
3. 单线程 device kernel 写 fallback count；
4. compact kernel 检查 device fallback count：
   - count=0：全部线程立即返回，不复制 interaction；
   - count>0：稳定 compact interaction 和 fallback；
5. transport kernel 在 device 上读取同一个 count：
   - count=0：直接从 raw interaction 读取；
   - count>0：从 compact interaction 读取；
   - index 超过精确 selected count：写零 flag 后返回。

因此常规无 fallback 路径只增加一次小 flag scan 和一个空操作 kernel，
不增加大记录复制。

## 3. 合并 transport control summary

transport 的 host summary 现在包含两个 word：

```text
[0] selection fallback count
[1] transport fallback count
```

selection interaction count 和 transport record count由输入总数推导：

\[
N_\mathrm{selected}=N_\mathrm{input}-N_\mathrm{selection\ fallback},
\]

\[
N_\mathrm{record}=N_\mathrm{selected}-N_\mathrm{transport\ fallback}.
\]

旧路径总共下载 \(4+4=8\) byte，新路径仍下载 8 byte，因此：

```text
control D2H bytes: 不变
blocking host synchronization: 每个 wavefront 减少 1
```

`DeviceInteractionSelectionBatch` 保留 raw 和 compact 指针、device count
指针及 `counts_deferred` 状态。transport 完成后恢复公开 host counts，
后续 fallback 下载和统计接口无需改变。

## 4. Lepton/Molière 安全性

lepton transport 后还有独立 Molière kernel。对于 selection fallback
形成的无效 launch suffix：

- transport kernel 将其 fallback flag 明确写为 0；
- Molière kernel用 device selection count 先检查精确前缀；
- count 和 compact kernel只处理有效 selected prefix。

因此不会读取未初始化 `LeptonTransportRecord`，也不会把 selection
fallback 重复计为 transport fallback。

## 5. 测试

除原有 staged-vs-pipeline 比较外，新增两类 mixed batch：

- photon：64 个输入中 2 个故意超出表格能区；
- electron/positron：64 个输入中 2 个故意超出表格能区。

每类测试都要求：

- selection fallback 恰好为 2；
- selected interaction 稳定保持原输入相对顺序；
- transport records/fallbacks 与旧 staged 路径逐项一致；
- photon final state 或 lepton vertex/final state 逐项一致；
- lepton Molière 路径不访问无效 suffix。

完整 CUDA 回归：

```text
24/24 passed
```

## 6. 生产验证

### 6.1 10 GeV

| 指标 | 阶段 30 | 阶段 31 |
|---|---:|---:|
| photon / lepton wavefronts | 27 / 107 | 27 / 107 |
| selection→transport fusions | 0 | 27 / 107 |
| total eliminated synchronizations | 132 | 266 |
| eliminated control D2H | 5,376 B | 5,376 B |

### 6.2 1 TeV

| 指标 | 阶段 30 | 阶段 31 |
|---|---:|---:|
| photon / lepton wavefronts | 35 / 442 | 35 / 442 |
| selection→transport fusions | 0 | 35 / 442 |
| total eliminated synchronizations | 476 | 953 |
| eliminated control D2H | 20,384 B | 20,384 B |

### 6.3 1 PeV 与 GPU CoREAS/ZHS

| 指标 | 阶段 30 | 阶段 31 |
|---|---:|---:|
| GPU particles | 17,645,186 | 17,645,186 |
| photon / lepton wavefronts | 107 / 1,194 | 107 / 1,194 |
| selection→transport fusions | 0 | 107 / 1,194 |
| total eliminated synchronizations | 1,299 | 2,600 |
| eliminated control D2H | 55,444 B | 55,444 B |
| core runtime | 26.164 s | 26.223 s |
| router runtime | 11.501 s | 11.499 s |
| photon backend | 0.797 s | 0.850 s |
| lepton backend | 10.603 s | 10.550 s |
| outer wall time | 29.84 s | 29.86 s |

photon/lepton 分项时间的约 53 ms 反向变化互相抵消，router 和外部墙钟
基本不变；没有观察到额外 flag scan 导致端到端回退。单次运行仍不用于
宣称统计显著加速。

10 GeV、1 TeV、1 PeV 的 profile、dEdX、ground particles、
interaction、production profile、CoREAS 和 ZHS 均与阶段 30 逐字节
一致。1 PeV 的七类 SHA-256 继续保持阶段 27–30 的固定基线。

所有 fixed-point overflow、profile invalid record、queue overflow、
host spill 和最终 pending queue 都为 0。

## 7. 下一步

当前每个 wavefront 的下一组中间同步是：

```text
transport records
    -> extract interaction count
    -> photon final state

transport records
    -> extract interaction candidates
    -> lepton vertex selection count
    -> lepton final state
```

下一阶段将使 transport-interaction extraction 返回 device count，并让
photon final-state / lepton vertex kernel用固定上界加 device 精确前缀。
对 lepton 还需把 vertex interaction、continuation、fallback 三类 count
合并到后续 final-state/endpoint summary。
