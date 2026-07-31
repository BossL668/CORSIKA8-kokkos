# 阶段 29：延迟 lepton final-state 计数并融合 endpoint 同步

## 1. 动机

阶段 28 清除了 photon charged-secondary 选择后的独立计数同步，但
lepton pipeline 仍在每个非空 final-state batch 中执行两次阻塞回传：

```text
22 个 scan 数组的末元素
    -> 88 byte D2H，主机计算 11 类计数
    -> 主机按计数决定 endpoint scatter 的 launch 大小
endpoint compaction
    -> 5 × uint32 = 20 byte D2H
```

这两个同步之间没有 CPU 物理计算。后一个阶段只需要知道各 compact
数组的有效前缀长度，因此可以在 device 上直接读取 scan summary，以
`input_count` 作为安全 launch 上界，并让 kernel 自行跳过有效前缀之外的
线程。

## 2. Device summary

`DeviceBatchStages.hpp` 新增固定布局
`BremsFinalStateSummaryLayout`：

```text
[0]  secondary count
[1]  accepted GPU final-state count
[2]  CPU fallback count
[3]  continuation count
[4]  LPM suppression count
[5]  bremsstrahlung count
[6]  annihilation count
[7]  ionization count
[8]  electron-pair count
[9]  bremsstrahlung LPM suppression count
[10] electron-pair LPM suppression count
[11] error code
```

final-state 的 22 个 flag/offset 数组在 workspace 中连续分配。
`finalizeBremsFinalStateSummaryKernel<<<1,1>>>` 在 device 上读取每对数组
的最后一个 flag 和 offset，生成 12 项 summary，并同时验证：

- GPU、fallback、continuation、suppression 分类总数等于输入数；
- 四种 GPU 过程计数之和等于 GPU final-state 数；
- 未启用 thinning 时，二体/三体末态的 child 数严格闭合；
- LPM suppression 分类不丢失、不重复；
- secondary history ID 不溢出。

末态方向归一化错误仍由 write kernel 原子写入同一个 error word，禁止
静默继续。

## 3. 延迟 host count

`launchBremsFinalStateOnDevice()` 新增内部参数：

```cpp
bool defer_count_download = false;
```

- 独立 final-state 验证入口保持默认 `false`，立即下载 12-word summary，
  因而原 API 和错误行为不变；
- production lepton pipeline 传入 `true`，返回 device summary 指针并把
  `counts_deferred` 置真，不进行 D2H；
- endpoint scatter 以 `input_count` 为 launch 上界，同时把 device
  summary 中的精确 count 指针传给 kernel；
- 超出精确 compact 前缀的线程立即返回，不能读取未初始化记录。

这样没有引入原子 append，也没有改变 CUB stable scan 的输出顺序；
history ID、generation、step ID 和同一顶点的 child 顺序保持确定。

## 4. 融合 endpoint summary

lepton endpoint summary 从 5 word 扩展为 16 word：

```text
[0..3]   next lepton、generated photon、observation、endpoint error
[4..15]  完整 BremsFinalStateSummaryLayout
```

endpoint finalizer 在 device 上复制 final-state summary，然后只执行一次
64-byte D2H。主机收到该 summary 后恢复所有公开统计计数，并在使用输出
数组或增长 history ID 之前检查错误。

每个非空 final-state batch 的同步和流量变化为：

\[
B_\mathrm{before}=88+20=108\ \mathrm{byte},
\qquad
B_\mathrm{after}=16\times4=64\ \mathrm{byte},
\]

\[
\Delta B_\mathrm{D2H}=44\ \mathrm{byte},
\qquad
\Delta N_\mathrm{sync}=1.
\]

新增统计：

```text
lepton_final_state_endpoint_summary_fusions
pipeline_host_synchronizations_eliminated
pipeline_device_to_host_bytes_eliminated
```

并写入 `gpu_em/summary.yaml` 的 `pipeline_control` 节点。已有
`device_to_host_bytes` 只统计物理记录传输，不包含内部 stage summary，
所以它不会因本阶段变化；新增字段专门审计控制流传输。

## 5. 测试

以下目标重新编译并重新链接：

```text
CORSIKA8GpuEm
testGpuBremsFinalState
testGpuLeptonTransport
testGpuHybridRoute
c8_air_shower
```

`testGpuLeptonTransport` 额外要求一次非空 device-chained pipeline：

- 恰好记录一次 final-state/endpoint fusion；
- 恰好消除一次 host synchronization；
- 恰好记录 44 byte D2H control traffic reduction。

完整 CUDA 回归：

```text
24/24 passed
```

## 6. 固定种子生产验证

### 6.1 10 GeV

seed 25001、ring=0：

| 指标 | 结果 |
|---|---:|
| GPU particles | 15,392 |
| photon wavefronts | 27 |
| lepton wavefronts | 107 |
| fused non-empty final-state batches | 105 |
| eliminated synchronizations | 105 |
| eliminated control D2H | 4,620 B |
| host spill / final pending queues | 0 / 0 |

profile、dEdX、ground particles、interaction、production profile、
CoREAS 和 ZHS 七类文件全部与阶段 28 逐字节一致。

### 6.2 1 TeV

seed 25002、`emthin=1e-4`、`max-weight=100`：

| 指标 | 结果 |
|---|---:|
| GPU particles | 576,015 |
| photon wavefronts | 35 |
| lepton wavefronts | 442 |
| fused non-empty final-state batches | 441 |
| eliminated synchronizations | 441 |
| eliminated control D2H | 19,404 B |
| host spill / final pending queues | 0 / 0 |

七类输出全部与阶段 28 逐字节一致。

### 6.3 1 PeV 与 GPU CoREAS/ZHS

seed 24027、`emthin=1e-4`、`max-weight=100`、8 个 CoREAS 和 8 个 ZHS
observer：

| 指标 | 阶段 28 | 阶段 29 |
|---|---:|---:|
| GPU particles | 17,645,186 | 17,645,186 |
| photon wavefronts | 107 | 107 |
| lepton wavefronts | 1,194 | 1,194 |
| fused non-empty final-state batches | - | 1,192 |
| eliminated synchronizations | - | 1,192 |
| eliminated control D2H | - | 52,448 B |
| core runtime | 26.238 s | 26.208 s |
| router runtime | 11.669 s | 11.587 s |
| host spill / final pending queues | 0 / 0 | 0 / 0 |

单次时间差仍处于运行噪声范围，不把它解释为统计显著的端到端加速。
可以严格确认的是 1,192 个 host blocking point 已从控制流中删除。

以下七类输出与阶段 28 逐字节一致：

```text
profile
3db76ad9089708601dacf98464a062285858c89553dc50f36a3840d3a323a792

dEdX
37ea203f3faa372f9e6651dbe31e9cec3d3b639f21695547b46e9d7ab6be1986

ground particles
7ab4b8c11d16ec819dd17cc0c381acc956180c1ee30cd758c09f55fcb663a204

interactions
6b529acf3d26f1349f298da3f9c5b7e8d50e7aa452939698212bf969d435e06d

production profile
14f0fc114c4671c25d80cf58607115b85f3b83803dd21078f78d1255de95a54a

CoREAS
6b2ad3f1b5dbbb000e1e3fcb5743f244a70f841abe7479bcf30f122382b68cee

ZHS
047973c559a6d642eb36ef19e734699e0f5a910045f066d6721bef46a7539692
```

radio/profile fixed-point overflow、profile invalid record、CUDA queue
overflow、host spill 和最终 pending queue 均为 0。

## 7. 下一步

photon final-state 仍有同构的 scan-count D2H，应采用相同模式：

1. 在 device 上生成 photon final-state summary；
2. endpoint scatter 用输入上界加 device 精确计数；
3. 把 final-state summary 融合进 photon endpoint summary；
4. 再逐步把 selection、transport extraction 和 vertex selection 的 host
   count 决策收敛到统一 pipeline control block。

完成 photon 对称路径后，再评估 CUDA Graph 或 persistent scheduler，
避免在尚有 stage-level host dependency 时过早固定 graph 结构。
