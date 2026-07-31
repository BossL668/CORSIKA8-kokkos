# 阶段 33：\(10^{17}\)–\(10^{18}\,\mathrm{eV}\) UHE 验收

## 1. 验收范围

本阶段验证 CUDA EM backend 在目标能区的：

- 表格能区上边界；
- LPM 长级联；
- EM thinning；
- resident photon/lepton cross-species queue；
- CPU 指定稀有末态回退；
- GPU profile、CoREAS 和 ZHS；
- 显存上限与 workspace增长；
- 同 seed逐字节确定性；
- 多 seed稳定性。

验收同时发现：完整 photonuclear hadron cascade 会触发一个与 CUDA 无关
的 UrQMD 进程级退出。为避免把外部强子模型问题混入 EM backend验收，
最终矩阵对 hadron、muon 和 tau child 使用高 cut；所有稀有过程仍被 CPU
显式生成和记录，非 EM child 随后由 cut处理。

## 2. 可复现配置

### 2.1 \(10^{17}\,\mathrm{eV}\)

```bash
./applications/c8_air_shower \
  -p 11 -E 100000000 -N 1 \
  -f OUTPUT \
  --em-backend cuda \
  --radio-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 64 \
  --gpu-table-cache gpu_em_tables/production_v9_1e-3_1EeV.c8emrt \
  --gpu-table-tolerance 1e-3 \
  --ring 1 \
  --seed 32017 \
  --emthin 1e-3 \
  --max-weight 1000000 \
  --hadcut 10000000000000 \
  --mucut 10000000000000 \
  --taucut 10000000000000 \
  --verbosity warn
```

第二个独立 seed 为 `32018`。

### 2.2 \(10^{18}\,\mathrm{eV}\)

只把能量改为：

```text
-E 1000000000
```

两个 seed 为 `33018` 和 `33019`。

这是 version-9 production table 的标称上边界，因此也验证了闭区间端点
查询、LPM 参数与 rate interpolation不会错误地落入 CPU generic
fallback。

## 3. 多 seed 结果

| 能量 | seed | GPU particles | photon/lepton waves | CPU specified final states | max input batch | core runtime |
|---|---:|---:|---:|---:|---:|---:|
| \(10^{17}\) eV | 32017 | 1,013,284 | 57 / 1,020 | 184 | 6,718 | 3.438 s |
| \(10^{17}\) eV | 32018 | 1,021,308 | 78 / 987 | 192 | 7,136 | 3.132 s |
| \(10^{18}\) eV | 33018 | 2,257,755 | 80 / 1,278 | 245 | 18,631 | 4.739 s |
| \(10^{18}\) eV | 33019 | 2,093,092 | 90 / 1,320 | 235 | 16,222 | 4.927 s |

外层进程墙钟包含 Pythia/TAUOLA/UrQMD 初始化与 writer：

```text
10^17 eV: 7.07 s, 6.88 s
10^18 eV: 8.22 s, 9.18 s
```

## 4. 安全性结果

四个 shower 均满足：

```text
complete = true
CUDA errors = 0
queue_overflows = 0
cross-species host_spills = 0
final pending photons/leptons = 0 / 0
profile fixed-point overflows = 0
profile invalid records = 0
radio fixed-point overflows = 0
workspace limit checkpoints = 0
input batch splits = 0
```

两个代表性显存统计：

| 能量/seed | peak device bytes | workspace bytes |
|---|---:|---:|
| \(10^{17}\) eV / 32017 | 688,968,654 | 134,217,728 |
| \(10^{18}\) eV / 33018 | 823,186,382 | 268,435,456 |

RTX 4060 Laptop 8 GiB 上没有接近配置的 70% memory budget，也没有触发
spill。

## 5. 射电验收

\(10^{17}\,\mathrm{eV}\), seed 32017：

```text
track-observer pairs: 15,265,600
CoREAS contributions: 14,979,375
ZHS contributions: 16,797,092
ZHS subtracks: 7,667,140
radio fixed-point overflows: 0
```

四个 UHE shower 都使用 GPU radio backend 和八个 CoREAS + 八个 ZHS
observer。射电 accumulator与 lepton transport wavefront 在线消费同一批
track segment；没有保存完整 shower后重算。

## 6. 同 seed 确定性

seed 32017 的 \(10^{17}\,\mathrm{eV}\) shower和 seed 33018 的
\(10^{18}\,\mathrm{eV}\) shower各重复一次。

比较范围为输出目录内全部：

```text
*.parquet
*.npz
```

两个能量的结果都是：

```text
binary_differences = 0
```

因此 profile、dEdX、ground particles、interactions、production profile、
CoREAS、ZHS 和 interaction histograms在同 GPU、同配置、同 seed下全部
逐字节可重复。

## 7. Thinning 参数压力测试

最初按 1 PeV 回归配置直接使用：

```text
emthin = 1e-4
max_weight = 100
```

在 \(10^{17}\,\mathrm{eV}\) 下运行 494.49 s后仍保持约 95% GPU
utilization，没有死锁或显存错误，但粒子量继续增长，故主动终止。

原因不是 CUDA backend退化，而是固定 `max_weight=100` 在主能量提高
100 倍后限制过严，使 thinning无法有效压缩统计粒子数。UHE production
必须把 thinning fraction 和 max weight视为物理/性能配置的一部分，
不能直接复制 PeV 参数。

本阶段采用 `emthin=1e-3, max_weight=10^6` 后：

- 仍保留加权 shower observable；
- GPU 粒子数稳定在约 \(10^6\)–\(2.3\times10^6\)；
- 单 shower CUDA 主链为 3–5 s；
- radio fixed-point仍无 overflow。

正式论文分析需要另外研究该 thinning设置对 profile、ground 和 radio
方差的影响，不能只依据运行时间选择。

## 8. 完整光核末态的外部阻塞

使用默认 hadron cut、`emthin=1e-4`、`max_weight=100` 的
\(10^{17}\,\mathrm{eV}\), seed 32017 完整运行在 217.09 s 后终止：

```text
UrQMD terminating without collision !?
ebeam = 0.31192533609280393
projectile mass = 1
projectile charge = 0
target mass = 16
target charge = 8
iterations = 50000
exit status = 77
```

源码位置：

```text
modules/urqmd/urqmd.f
```

Fortran 中实际调用：

```fortran
call exit(333)
```

Linux shell看到 `333 mod 256 = 77`。

该事件来自 GPU 正确返回 CPU 的 photonuclear末态，随后低能中子进入
UrQMD。它证明稀有过程 fallback 没有被静默删除，但也说明完整 UHE
production目前受低能强子模型失败策略阻塞。

曾验证 UrQMD 的 `iflb=1` “empty event直接传播”模式。它会产生大量
正常几何空事件并改变同顶点重采样语义，不可作为科研等价修复；相关实验
代码已全部移除，CPU 默认路径保持不变。

后续正确方案应在 UrQMD wrapper中实现：

1. 可检测、有限次数的同顶点 retry；
2. 明确的 failure record，而不是 Fortran process exit；
3. 物理上定义的 fallback模型或弹性传播策略；
4. 独立 CPU/UrQMD统计验证；
5. output manager对外部模型异常写 incomplete summary。

在完成这些工作前，不能声称“默认完整 photonuclear UHE shower”已通过
生产验收。

## 9. Phase 32 control fusion 在 UHE 的收益

| 能量/seed | eliminated host synchronizations | eliminated control D2H |
|---|---:|---:|
| \(10^{17}\) eV / 32017 | 4,251 | 58,944 B |
| \(10^{17}\) eV / 32018 | 4,180 | — |
| \(10^{18}\) eV / 33018 | 5,346 | 73,960 B |
| \(10^{18}\) eV / 33019 | 5,550 | — |

UHE 下 lepton wavefront约 1,000–1,320 个，说明 Phase 32删除的不是少量
启动开销，而是每个 shower数千次潜在 device-wide host同步。

## 10. 下一步

GPU EM backend 本身已经通过目标能区多 seed验收。下一阶段应并行推进：

1. 用 Nsight Systems分析约 1,000 个 lepton wavefront的 launch gap；
2. 评估 CUDA Graph或 persistent lepton scheduler；
3. 建立 UHE thinning 系统误差矩阵；
4. 单独修复 UrQMD process-exit失败策略；
5. 在完整 hadronic/photonuclear配置下重新执行 UHE acceptance。
