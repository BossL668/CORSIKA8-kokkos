# 阶段 87：CUDA μ 子输运、原参数表适配、性能与当前统计验收

## 1. 阶段结论

强子分类 FLUKA worker 接入后，细粒度 profiling 显示：在 100 TeV 测试里，scalar
stepper 的 94.2% 时间来自 \(\mu^\pm\)，而不是 FLUKA。因此本阶段把
\(\mu^-\) 和 \(\mu^+\) 纳入 CUDA wavefront，同时保留稀有过程和衰变的显式 CPU
回退。

当前得到三个不同层次的结论：

1. 性能冒烟测试中：
   - 100 TeV shower 内部加速 14.64×；
   - 1 PeV shower 内部加速 17.31×。
2. 接近用户参数的 1 PeV、`emthin=1e-5`、完整 CoREAS/ZHS 事例已经跑通，显存
   无溢出，但瓶颈转移到 GPU profile 和 radio 投影。
3. 当前 1 TeV proton、500+500 shower 的 CPU/CUDA ensemble：
   - 九条关键平均曲线全部通过；
   - 关键 scalar 分布的 KS 检验均未拒绝同分布；
   - 九个关键 scalar 中五个已经通过严格 1% 均值门限；
   - 但其余四个没有通过，所以正式 physics acceptance 当前状态仍是
     **未通过/统计上尚不充分**，不能宣称已经完成生产验收。

## 2. μ 子如何进入 CUDA 路径

### 2.1 生产选择

CUDA table 必须同时包含 PDG `13` 和 `-13`。如果只包含一个电荷态，启动立即失败。
应用还检查：

- 表中 μ 子 continuous range 的 transport cut；
- 运行时 `--mucut`；
- 物理表 tolerance 和内容 hash。

当前组合表：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/
  production_v10_muons_1e-3_1EeV.c8emrt
```

文件 SHA-256：

```text
14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521
```

内容 hash：

```text
c662d8d49a8b6f2a3a23987a2d7652b8de63f0a3da292a8fff01360ec3728e08
```

该文件由已经分别验证的 EM production table 和 μ-only table 合并得到。实验性的
直接 Epair-\(\rho\) 表再生因为最大误差 0.9769 而被严格拒绝，没有为了让表通过而
放宽 \(10^{-3}\) tolerance。

### 2.2 当前 GPU 覆盖

当前 μ 子 CUDA 路径处理：

- 连续 ionization；
- 连续 range；
- multiple scattering；
- 磁场与五层球形大气 tracking；
- 边界、观测面和 transport cut；
- 离散过程和衰变距离竞争；
- GPU 端 ionization 选择。

当前仍由 CPU 精确完成：

- μ 子 bremsstrahlung；
- μ 子 pair production；
- photonuclear；
- 其他尚未登记的稀有离散末态；
- μ 子实际 decay final state。

对于稀有离散过程，GPU 返回已经选定的过程记录，CPU 不允许重新随机选择另一个
过程。对于 decay，GPU 把 μ 子推进到选定的衰变顶点，随后主进程对同一个
`history_id/step_id` 调用 `ScalarCascadeStepper::forceDecay()`。

这是一条 lossless fallback 边界，不是静默忽略尚未 GPU 化的 μ 子物理。

### 2.3 射电语义

当前 scalar CoREAS/ZHS 只观察 \(e^\pm\) 轨迹。CUDA μ 子轨迹同样不直接送入
radio accumulator，因而没有因为 μ 子搬到 GPU 而额外增加原版不存在的 radio
源项。μ 衰变产生的电子回到正常 EM/CUDA 路径，并照常产生射电贡献。

## 3. 单元和集成验证

组合表下的关键测试：

| 测试 | 结果 |
|---|---:|
| `testGpuHybridRoute` | passed |
| `testGpuInteractionSelection` | 245,783 checks passed |
| interaction selection 粒子数 | 8192 |
| unexpected fallback | 0 |
| `testGpuLeptonTransport` | 97,809 checks passed |
| lepton particles transported | 8192 |
| selected vertices | 3404 |
| GPU ionization | 3309 |
| specified CPU fallbacks | 95 |
| forced CPU decays | 725 |

1 GeV \(\mu^-\) 的端到端 smoke 输出：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor/validation/
  muon_cuda_force_decay_smoke_v1
```

该事例记录一个 GPU decay candidate，并且恰好执行一次 CPU forced decay：

```text
cpu_decay_particles = 1
forced_cpu_decays_executed = 1
```

当前 GPU energy ledger 尚未计入 CPU decay 产生的 neutrino，因此这个极小测试的
ledger residual 不能用于声称完整能量闭合；这是输出语义待补项，不是把 μ 子
neutrino 删除了。

## 4. 性能结果

### 4.1 100 TeV

设置：

- proton；
- 100 TeV；
- seed 24680；
- `emthin=1e-4`；
- `max-weight=100`；
- 无 radio；
- 热 PROPOSAL fallback cache。

输出：

```text
CPU:
/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_scaling_proton_E100000GeV_seed24680_original_scalar_hot_v1

CUDA μ + FLUKA workers:
/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_scaling_proton_E100000GeV_seed24680_gpu_mu_fluka_process_w4_hot_v2
```

| 指标 | CPU scalar | CUDA μ + FLUKA-4 | 加速 |
|---|---:|---:|---:|
| shower internal | 20.550 s | 1.404 s | 14.64× |
| 整个进程 | 24.74 s | 6.38 s | 3.88× |

新路径的附加统计：

- GPU particles：2,618,763；
- CPU scalar steps：16,549；
- FLUKA interactions：2352；
- forced μ CPU decays：1534/1534；
- FLUKA pool execute：114.35 ms；
- router：1048.14 ms；
- scalar stepper：154.01 ms。

整个进程的加速比低于 shower internal，是因为约 5 秒的一次性应用、模型和输出
初始化在这个小事例中占比很高。

### 4.2 1 PeV 性能冒烟

设置仍为 `emthin=1e-4`、`max-weight=100`、无 radio。这是性能开发测试，不是
用户 `config.yaml` 的正式质量参数。

输出：

```text
CPU:
/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_fluka_fixedseed_1PeV_original_cpu_v1

CUDA μ + FLUKA workers:
/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_fluka_fixedseed_1PeV_gpu_mu_fluka_process_w4_v1
```

| 指标 | CPU scalar | CUDA μ + FLUKA-4 | 加速 |
|---|---:|---:|---:|
| shower internal | 147.474 s | 8.521 s | 17.31× |
| 整个进程 | 约 154 s | 13.58 s | 约 11× |

CUDA 事例统计：

- GPU particles：14.35 million；
- CPU scalar steps：114,355；
- FLUKA interactions：19,571；
- forced μ CPU decays：9988/9988；
- FLUKA pool execute：925.68 ms；
- router：5.861 s；
- scalar stepper：1.082 s。

四个 worker 的 FLUKA 物理负载为：

```text
473.108, 479.209, 481.088, 482.175 ms
```

最大差 9.067 ms，约为平均负载的 1.9%。这说明在 1 PeV 下，强子分类调度没有
形成明显长尾；强子末态已经不是当前主要门槛。

## 5. 接近 `config.yaml` 的完整 radio 事例

已完成的较高质量 smoke 使用：

- proton，1 PeV；
- zenith 27°；
- azimuth 180°；
- core `(0, 0)`；
- ring 2，共 81 个天线；
- `emthin=1e-5`；
- 不显式指定 `--max-weight`，保留自动计算；
- seed 0；
- CUDA CoREAS/ZHS；
- CUDA μ；
- 四个 FLUKA worker。

输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  config_like_proton_1PeV_TH27_PH180_ring2_emthin1e-5_gpu_mu_fluka4_v1
```

结果：

| 指标 | 数值 |
|---|---:|
| 整个进程 | 335.53 s |
| shower internal | 330.268 s |
| GPU particles | 231,928,454 |
| peak device memory | 5.03 GB |
| memory overflow/spill | 0 |
| GPU internal radio tracks | 201,555,148 |
| radio device time | 202.354 s |
| profile kernel | 148.842 s |
| router wall | 273.573 s |
| scalar stepper | 40.275 s |
| FLUKA interactions | 16,561 |
| FLUKA pool execute | 0.793 s |

radio device time、profile kernel 和总 kernel time 有流重叠，不能简单相加。但
结果已经清楚表明：

- FLUKA 不再是门槛；
- 高质量 thinning 下的主要开销是 profile/radio 和大量 EM wavefront；
- 小于 GPU minimum batch 的前沿还会导致 CPU 展开，当前有 84,153 次 CPU
  expansion step；
- 继续 GPU 化高能强子事件生成器不会解决当前这个完整事例的主要耗时。

本次没有运行完整 `emthin=1e-6`。用户已经验证 `1e-6` 比 `1e-5` 更慢并提供更好
数据质量；后续正式验收必须保留 `1e-6`，不能用这个 `1e-5` smoke 代替。后续
代码审计进一步确认：在精确的 1 PeV、自动 maximum-weight 配置下，
`maxWeight=0.5`，所以该档实际退化为未薄化极限；这与“更慢、质量更高”的观测
一致。

## 6. `emthin` 参数的兼容原则

只指定：

```text
--emthin 1e-6
```

是原版支持的有效用法。未指定 `--max-weight` 时，应用沿用原有自动推导。CUDA
适配层不会：

- 强迫用户额外给出 `--max-weight`；
- 把 `1e-6` 自动改成更大的值；
- 根据运行速度猜测是否发生 thinning。

开发阶段若使用 `1e-5` 或 `1e-4`，必须明确标注为性能 smoke。正式结果需要按原
参数重新运行。同时，适配器现在会审计自动权重是否大于初始 unit weight；在
1 PeV、`1e-6` 时会报告 `maxWeight=0.5`、thinning 无法启动，而不会仅根据
命令行存在 `--emthin` 就声称实际发生了 thinning。

## 7. 直接读取原有 `config.yaml`

脚本：

```text
/home/yuhanglu/21CMA/python/run_script/parallel_corsika.py
```

新增：

```text
--backend-profile original|cuda-hybrid
```

`cuda-hybrid` 只在每个现有 task 后追加实现相关参数，不会改写：

- particle；
- energy；
- zenith/azimuth；
- shower core；
- ring；
- `emthin`；
- seed；
- 输出名字。

因此 `/home/yuhanglu/21CMA/python/config.yaml` 中的 `--emthin 1e-6` 会原样进入
新程序。注意该 YAML 顶层 `energy_range` 写的是 `1e9` GeV，但实际 task 参数是：

```text
-E 1e6
```

所以当前这个 task 实际模拟 1 PeV。输出名字里仍含 `E1e9`，这是原 YAML 的命名
不一致，适配层不会擅自修改物理参数或输出名。

可复现命令：

```bash
env FLUPRO=/home/yuhanglu/fluka \
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python \
/home/yuhanglu/21CMA/python/run_script/parallel_corsika.py \
  --corsika-exec \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --config /home/yuhanglu/21CMA/python/config.yaml \
  --output-dir \
    /home/yuhanglu/21CMA/corsika_validation_results/config_yaml_cuda_hybrid_emthin1e-6 \
  --jobs 1 \
  --backend-profile cuda-hybrid \
  --gpu-table-cache \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --gpu-min-batch 4096 \
  --radio-backend cuda \
  --hadronic-workers 4 \
  --force
```

当前脚本在单 GPU 模式下要求 `--jobs 1`，避免多个 shower 同时争用 RTX 4060 的
8 GB 显存。一个 CUDA task 若使用四个 FLUKA worker，CPU affinity 应至少分配
主进程加四个 worker 共五个核。

## 8. 当前 CPU/CUDA ensemble 验收

### 8.1 先导 200+200 样本

组合表和 μ GPU 路径先完成了 200 个原版 CPU proton shower 与 200 个 CUDA
hybrid shower 的独立随机样本比较：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_mu_hybrid_vs_original_proton_1TeV_200_v2
```

参数：

- 1 TeV proton；
- zenith 27°、azimuth 180°；
- EM cut 0.5 MeV；
- hadron/muon cut 0.3 GeV；
- `emthin=1e-4`、`max-weight=100`；
- 无 radio。

关键结果：

| observable | CPU mean | CUDA mean | 相对差 | \(|z|\) | KS / 95% 临界值 |
|---|---:|---:|---:|---:|---:|
| profile \(X_{\max}\) (g/cm²) | 337.759 | 357.013 | +5.70% | 1.719 | 0.110 / 0.136 |
| charged \(N_{\max}\) | 622.979 | 651.954 | +4.65% | 1.586 | 0.105 / 0.136 |
| charged profile integral | 199950 | 201990 | +1.02% | 0.386 | 0.075 / 0.136 |
| photon profile integral | 1,043,234 | 1,060,520 | +1.66% | 0.531 | 0.065 / 0.136 |
| total energy deposit (GeV) | 691.594 | 692.560 | +0.140% | 0.094 | 0.060 / 0.136 |
| dE/dX peak depth | 160.624 | 192.337 | +19.74% | 2.007 | 0.135 / 0.136 |
| ground EM weighted count | 318.426 | 397.188 | +24.73% | 1.719 | 0.080 / 0.136 |
| ground EM kinetic energy | 10.183 | 15.553 | +52.73% | 2.073 | 0.105 / 0.136 |
| energy closure fraction | 0.70180 | 0.70814 | +0.903% | 0.637 | 0.070 / 0.136 |

正确解释是：

- 九条关键平均曲线全部通过逐 bin 统计门限；
- 表中关键 scalar 的经验 KS 距离都低于 95% 临界值；
- 没有关键均值达到 3σ 差异；
- energy deposit 和 closure 通过 1% 门限；
- 其余关键均值没有通过预先设定的严格 1% 门限；
- ground tail 和 dE/dX peak depth 方差很大，其中两个约为 2σ，仍需更多事例。

因此当前报告状态为：

```text
failed: relative threshold failed but statistically inconclusive
```

不能把它写成“CPU/CUDA 已经统计等价”，也不能据此判定存在确定的物理偏差。正式
结论需要扩大样本，并在用户真正关心的 1 PeV、`emthin=1e-6` 和 radio observable
上重复检验。

### 8.2 独立 500+500 样本

为了检查 200+200 中约 2σ 的高尾偏移是否会增强，随后用全新、互不重叠的 seed
运行 500 个原版 CPU shower 和 500 个当前 CUDA hybrid shower：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  current_mu_hybrid_vs_original_proton_1TeV_500_v3
```

物理参数和验收门限与 200+200 相同。CPU 使用十个各 50 shower 的 shard，CUDA
使用一个 500-shower 进程；二者重叠执行，因此墙钟时间只用于完成统计样本，不作为
正式性能测量。

| observable | CPU mean | CUDA mean | CUDA−CPU | \(|z|\) | KS / 95% 临界值 |
|---|---:|---:|---:|---:|---:|
| profile \(X_{\max}\) (g/cm²) | 351.272 | 347.688 | −1.020% | 0.470 | 0.040 / 0.086 |
| charged \(N_{\max}\) | 652.046 | 646.772 | −0.809% | 0.463 | 0.048 / 0.086 |
| charged profile integral | 203871 | 204061 | +0.093% | 0.058 | 0.042 / 0.086 |
| photon profile integral | 1,067,930 | 1,064,896 | −0.284% | 0.150 | 0.026 / 0.086 |
| total energy deposit (GeV) | 693.496 | 697.251 | +0.542% | 0.565 | 0.038 / 0.086 |
| dE/dX peak depth | 192.105 | 175.224 | −8.787% | 1.597 | 0.066 / 0.086 |
| ground EM weighted count | 413.061 | 390.438 | −5.477% | 0.672 | 0.044 / 0.086 |
| ground EM kinetic energy | 16.608 | 12.837 | −22.705% | 2.002 | 0.066 / 0.086 |
| energy closure fraction | 0.710130 | 0.710112 | −0.0026% | 0.003 | 0.038 / 0.086 |

500+500 的判读：

- 九条平均曲线继续全部通过；
- 九个关键 scalar 的 KS 距离都低于 95% 临界值；
- `Nmax`、charged/photon integral、总沉积能和 closure 共五项通过 1% 门限；
- \(X_{\max}\) 只比门限多 0.020 个百分点，且差异只有 0.47σ；
- 200+200 中多数正向偏移在独立 500+500 中反向或缩小，证明小样本 shower
  fluctuation 对均值方向影响很大；
- ground EM kinetic energy 仍约为 2.00σ，是当前最值得继续追踪的 observable；
- 没有关键量达到预设 3σ 拒绝条件。

因此更大样本的正式状态仍是：

```text
failed: relative threshold failed but statistically inconclusive
```

它比 200+200 更接近验收，但仍不足以签署生产统计等价。下一批样本应优先增加
ground tail 的统计量，并检查该量对 observation boundary、CPU geometry fallback
和低能 cut 的敏感性。

作为方向性诊断，把 200+200 与 500+500 的逐 shower observable 合并后，得到
700+700 样本。两批之间重编译只移除了 thinning 警告和一项诊断 metadata，但严格
provenance 规则仍把 executable hash 视为不同，所以这个合并结果不能替代正式门限
报告。它仍有助于判断 fluctuation：

| pooled diagnostic | CUDA−CPU | \(|z|\) | KS \(p\) |
|---|---:|---:|---:|
| profile \(X_{\max}\) | +0.847% | 0.466 | 0.720 |
| dE/dX peak depth | −1.637% | 0.340 | 0.630 |
| ground EM weighted count | +1.643% | 0.231 | 0.880 |
| ground EM kinetic energy | −7.847% | 0.753 | 0.381 |

这解释了为什么不能把任一批约 2σ、且方向相反的 tail 均值直接当成物理偏差；同时
也说明正式生产签署仍应使用同一个冻结 binary 获得更大的独立样本。

## 9. 当前瓶颈与下一步

按优先级：

1. 优化 GPU profile 累计与 radio projection：
   - 减少 2.0 亿级 radio track 的中间表示；
   - 检查 profile 与 radio 是否能进一步融合；
   - 分析 transfer、fixed-point accumulation 和天线维度 scaling。
2. 降低小前沿 CPU expansion：
   - 保留 `gpu-min-batch=4096` 作为当前较优点；
   - 已测 `64` 在 100 TeV 反而把 1.404 s 增至 2.148 s；
   - 不能仅靠减小 batch threshold 解决。
3. 补齐 CPU fallback/decay 的统一 energy ledger。
4. 用独立非零种子运行 1 PeV 多事例：
   - 先用 `emthin=1e-5` 做较快的趋势和 radio scaling；
   - 再用 `emthin=1e-6` 做未薄化高质量验收，并记录自动
     `maxWeight=0.5` 的边界。
5. 比较射电脉冲的地磁分量幅度和宽度分布，而不只比较累计轨迹数。
6. 只有 profiling 再次显示 high-energy hadronic generator 成为门槛时，才评估
   SIBYLL/QGSJet/EPOS 的 GPU 或更多进程后端。

当前证据不支持“先把全部强子放到 GPU、再算轻子”的两阶段 shower。强子会不断
产生 EM/μ 次级，而后者的反馈和衰变也参与后续发展；生产实现仍应使用混合 wavefront
调度，而不是把物理级联强行切成两个互不交互的整段。
