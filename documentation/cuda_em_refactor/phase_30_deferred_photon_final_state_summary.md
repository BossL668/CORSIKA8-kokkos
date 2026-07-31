# 阶段 30：延迟 photon final-state 计数并融合 endpoint 同步

## 1. 对称控制流

阶段 29 已让 lepton final-state 和 endpoint 共用一次 device summary。
photon 路径原先仍执行：

```text
16 个 final-state scan 数组末元素
    -> 64 byte D2H
endpoint + charged-secondary selection
    -> 6 × uint32 = 24 byte D2H
```

本阶段将这两个同步融合，同时保留阶段 28 的 stable charged-secondary
选择和 persistent cross-species queue。

## 2. Photon final-state summary

新增 `PhotonFinalStateSummaryLayout`：

```text
[0] secondary count
[1] accepted GPU final-state count
[2] CPU fallback count
[3] continuation count
[4] LPM suppression count
[5] photon-pair count
[6] Compton count
[7] photoelectric count
[8] error code
```

`finalizePhotonFinalStateSummaryKernel<<<1,1>>>` 直接读取 8 对连续
flag/offset 数组，检查：

- pair、Compton、photoelectric 之和等于 accepted record 数；
- accepted、fallback、continuation、suppression 之和等于输入数；
- thinning 关闭时 child 数严格等于过程定义的二体/一体末态数；
- thinning 开启时 child 数不超过未 thinning 上界；
- secondary history ID 不溢出。

write kernel 中原先理论上可能静默返回的 Compton direction 构造失败，
现在会写入同一 error word，并在任何输出被主机使用前终止 batch。

独立 final-state API 仍立即下载 9-word summary；production pipeline 设置
`counts_deferred=true`，直到 endpoint 才下载。

## 3. 用 device count 启动 endpoint

suppression 和 accepted-record scatter 以 `input_count` 为安全 launch
上界，同时接收 device summary 中的精确 count 指针。每个线程先检查：

```text
index < launch upper bound
index < *device_exact_count
```

因此未初始化 compact suffix 永远不会被读取。

### 3.1 Charged-secondary stable select

CUB `DeviceSelect::If` 的 item count 是 host 参数，不能直接传 device
count。一个直接但低效的方案是清零整个 `2 × input_count` 粒子缓冲；
`EmParticleState` 较大，这会引入显著显存写流量。

当前实现改用：

```text
CountingInputIterator<size_t>
    -> TransformInputIterator<EmParticleState>
       index < device_secondary_count: load real state
       otherwise: return PID=0 sentinel
    -> CUB stable DeviceSelect::If(charged)
```

这样：

- 不读取有效前缀之外的 device 粒子存储；
- 不清零大粒子缓冲；
- 不增加逐粒子 flag/offset workspace；
- CUB 输出顺序仍与 compact secondary 顺序相同；
- history ID 和同一顶点的 \(e^-/e^+\) 顺序不变。

设备 queue 容量仍使用保守上界 `2 × input_count`。若连续尾区不足则按
已有规则显式 spill 到 host；它只可能损失性能，不会截断物理输出。

## 4. 融合流量

photon endpoint summary 由 6 word 扩展为 15 word：

```text
[0..5]  endpoint、cut、charged-selection summary
[6..14] PhotonFinalStateSummaryLayout
```

\[
B_\mathrm{before}=64+24=88\ \mathrm{byte},
\qquad
B_\mathrm{after}=15\times4=60\ \mathrm{byte},
\]

\[
\Delta B_\mathrm{D2H}=28\ \mathrm{byte},
\qquad
\Delta N_\mathrm{sync}=1
\]

每个非空 photon final-state batch 都更新：

```text
photon_final_state_endpoint_summary_fusions
pipeline_host_synchronizations_eliminated
pipeline_device_to_host_bytes_eliminated
```

## 5. 测试

增加 deferred pipeline history-overflow 错误注入，要求错误从 device
summary 经融合 endpoint 回传为 `std::overflow_error`，不能被普通 endpoint
错误覆盖。

重新编译并链接 GPU 库、photon/lepton/hybrid 测试和 `c8_air_shower`。
完整结果：

```text
24/24 CUDA tests passed
```

## 6. 固定种子生产验证

### 6.1 10 GeV

| 指标 | 结果 |
|---|---:|
| photon / lepton wavefronts | 27 / 107 |
| photon / lepton fusions | 27 / 105 |
| total eliminated synchronizations | 132 |
| total eliminated control D2H | 5,376 B |
| host spill / final pending queues | 0 / 0 |

### 6.2 1 TeV

| 指标 | 结果 |
|---|---:|
| photon / lepton wavefronts | 35 / 442 |
| photon / lepton fusions | 35 / 441 |
| total eliminated synchronizations | 476 |
| total eliminated control D2H | 20,384 B |
| host spill / final pending queues | 0 / 0 |

### 6.3 1 PeV 与 GPU CoREAS/ZHS

| 指标 | 阶段 29 | 阶段 30 |
|---|---:|---:|
| GPU particles | 17,645,186 | 17,645,186 |
| photon / lepton wavefronts | 107 / 1,194 | 107 / 1,194 |
| photon / lepton fusions | 0 / 1,192 | 107 / 1,192 |
| total eliminated synchronizations | 1,192 | 1,299 |
| total eliminated control D2H | 52,448 B | 55,444 B |
| core runtime | 26.208 s | 26.164 s |
| router runtime | 11.587 s | 11.501 s |
| host spill / final pending queues | 0 / 0 | 0 / 0 |

墙钟单次运行分别约 29.83 s 和 29.84 s，因此不根据单次测量声称端到端
显著加速；严格成立的是又删除了 107 个 blocking control transfers。

10 GeV、1 TeV 和 1 PeV 的 profile、dEdX、ground particles、
interaction、production profile、CoREAS、ZHS 均与阶段 29 逐字节一致。
1 PeV 哈希保持：

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

所有 fixed-point overflow、profile invalid record、queue overflow、
host spill 和最终 pending count 都为 0。

## 7. 下一步

final-state 与 endpoint 之间的 host dependency 已从 photon 和 lepton
两条路径消失。余下同步主要位于：

1. interaction selection count；
2. transport record/fallback count；
3. transport-interaction extraction count；
4. lepton vertex interaction/continuation/fallback count；
5. wavefront 结束时的 endpoint control summary。

下一阶段将优先融合 selection 与 transport：selection 返回的 compact
interaction 数由 device control block 传给固定上界 transport kernel，
并把 selection fallback/count 延迟到后续 summary。完成后再对
transport extraction 和 lepton vertex 应用同一模式。
