# Phase 43：相同轨迹的 CoREAS/ZHS CPU–CUDA 验收

## 1. 为什么不能直接比较两个独立 shower

射电脉冲保留了 shower 的相位、到达时间和几何结构。即使 CPU 与 CUDA 的
粒子分布统计完全正确，两个独立随机 shower 的逐 bin 电场也不会相同。因此
不能用两个独立 ensemble 的波形逐点差值判断射电内核。

本阶段采用受控的 A/B 设计：

```text
A: CUDA EM transport + CPU CoREAS/ZHS
B: CUDA EM transport + CUDA CoREAS/ZHS
```

A 与 B 使用：

- 相同 CUDA rate table；
- 相同初级和几何；
- 相同 cut/thinning；
- 相同 Philox transport seed；
- deterministic 模式；
- 相同 observer 配置；
- 唯一差异是 `--radio-backend cpu|cuda`。

这样每个 CPU/GPU radio waveform 都来自同一组 \(e^\pm\) 轨迹。

## 2. 新增工具

```text
validation/gpu_em/run_radio_acceptance.py
validation/gpu_em/compare_radio_backends.py
validation/gpu_em/tests/test_run_radio_acceptance.py
validation/gpu_em/tests/test_compare_radio_backends.py
```

`run_radio_acceptance.py` 负责：

1. 检查可执行文件、table 和 Molière 热缓存；
2. 生成相同 CUDA 输运命令；
3. 分别执行 CPU radio 和 CUDA radio；
4. 调用严格比较器；
5. 保存命令、环境、wall time、输出和比较报告。

`compare_radio_backends.py` 也可以单独重分析已经完成的输出。

## 3. 防止错误配对

比较器不会只相信调用者声称“seed 相同”，而是检查：

```text
summary.yaml: showers 相同
summary.yaml: seed 相同
gpu_em/config.yaml: deterministic == true
CPU 输入: radio_backend == cpu
GPU 输入: radio_backend == cuda
移除 radio_backend 后的 GPU/transport 配置完全相同
```

随后要求下列非射电输出的 Arrow table 完全相等：

```text
energyloss/dEdX.parquet
profile/profile.parquet
particles/particles.parquet
production_profile/profile.parquet
interactions/interactions.parquet
```

任一表缺失、行数不同或值不同，整个 radio comparison 失败。这样不能用两个
不同 shower 的“看起来接近的波形”获得通过。

## 4. 波形重建

当前 `observers.parquet` 包含：

```text
shower, Time, Ex, Ey, Ez
```

但没有 observer name。`RadioProcess::endOfShower()` 与 config writer 都以
同一个 `ObserverCollection` 顺序遍历天线，因此比较器从
`CoREAS|ZHS/config.yaml` 读取 observer 顺序和 `number of bins`，按每个
shower 重建：

```text
shower × observer × algorithm × component
```

它同时要求 CPU/GPU：

- shower row identity 完全相同；
- 时间轴逐项相同；
- observer 配置逐项相同；
- 每个 shower 的总行数等于所有 observer bins 之和。

## 5. 数值指标与门禁

对每个三维电场分量计算：

\[
\delta_\infty =
\frac{\max_t|E_{\rm cuda}(t)-E_{\rm cpu}(t)|}
     {\max_t\left(|E_{\rm cuda}(t)|,|E_{\rm cpu}(t)|\right)},
\]

\[
\delta_2 =
\frac{\|E_{\rm cuda}-E_{\rm cpu}\|_2}
     {\max(\|E_{\rm cuda}\|_2,\|E_{\rm cpu}\|_2)},
\]

以及 summed-squared-field proxy：

\[
\delta_F =
\frac{|\sum E_{\rm cuda}^2-\sum E_{\rm cpu}^2|}
     {\max(\sum E_{\rm cuda}^2,\sum E_{\rm cpu}^2)}.
\]

默认门禁：

```text
peak-normalized maximum     1e-4
relative L2                 1e-4
relative fluence            5e-4
absolute weak-signal bound  1e-18 V/m
```

绝对 bound 是必要的：对理论上接近零的交叉偏振，\(10^{-19}\) V/m 的定点量化
误差可能产生较大的相对数，但它远低于任何实际物理信号。

## 6. 现有 1 TeV 相同轨迹证据

对此前保存的相同 seed 24026 输出运行新比较器：

```text
CPU radio:
  /tmp/c8_cpu_radio_1TeV_reference
CUDA radio:
  /tmp/c8_gpu_radio_1TeV_repeat
report:
  /tmp/c8_radio_compare_existing_v2.json
```

配置：

```text
primary             electron
energy              1 TeV
showers             1
observers           8 CoREAS + 8 ZHS
bins/observer       400
GPU min batch       64
deterministic       true
```

所有五个 transport tables 完全相同。48 个
`algorithm × observer × component` 比较全部通过：

| 算法 | max normalized | max relative L2 | max relative fluence | max absolute |
|---|---:|---:|---:|---:|
| CoREAS | \(5.94\times10^{-7}\) | \(7.43\times10^{-7}\) | \(1.91\times10^{-7}\) | \(3.28\times10^{-17}\) V/m |
| ZHS | \(4.66\times10^{-7}\) | \(1.18\times10^{-6}\) | \(2.18\times10^{-7}\) | \(9.50\times10^{-17}\) V/m |

所有相对指标都比默认门禁至少低约两个数量级。

## 7. 自动测试

验证脚本测试集现在覆盖：

- 相同 transport 和接近波形通过；
- transport table 任一值不同则失败；
- 时间轴不同则失败；
- 全零波形；
- CPU/CUDA 命令使用同一个 seed；
- 两条命令都选择 CUDA EM；
- 只有 radio backend 不同；
- ring 0 拒绝；
- 非有限 tolerance 拒绝；
- Phase 41 energy ledger；
- Phase 40 ensemble analysis；
- Phase 42 五次性能中位数。

结果：

```text
Ran 22 tests
OK
```

## 8. 尚未完成

1 TeV 单 shower 是明确的端到端 same-track oracle，但原计划需要更广的统计
矩阵。需求审计保持 `PARTIAL`，后续仍需：

- 正式运行默认 10-event 1 TeV radio acceptance；
- 至少一个 1 PeV radio pair；
- 对 GPU radio fixed-point overflow、显存和 wall time 形成正式报告；
- 完整强子初级中的 CPU fallback 轨迹与 GPU resident waveform 合并验收。
