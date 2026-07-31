# Phase 65–69：多 shower 射电生命周期与强子混合验收

## 1. 本阶段覆盖的生产边界

本阶段集中验证此前单事件测试没有覆盖的两个边界：

1. 一个 application/library 内连续运行多个 shower 时，CoREAS/ZHS 输出流
   必须保持可写；
2. 强子初级必须能沿

   ```text
   CPU hadronic cascade
     -> CUDA gamma/e-/e+ cascade
     -> specified CPU PROPOSAL final state
     -> CUDA/CPU particle reroute
     -> CoREAS/ZHS track projection
   ```

   完整闭环运行。

所有射电比较都使用相同 seed 和完全相同的 CUDA EM 输运，只切换
`--radio-backend cpu|cuda`。因此非射电输出应逐行相同，波形差异只测量射电
投影器的数值差异，不含 shower 统计涨落。

## 2. 修复多 shower Parquet 生命周期

正式 10-event 射电运行首先暴露：

```text
ParquetStreamer not initialized
```

原 `RadioProcess::endOfShower()` 在第一场 shower 后调用
`output_.closeStreamer()`。但一个 Parquet 文件本来就用 `shower` 列区分同一
library 的多个事件；关闭后第二场 shower 无法再写入。

修复后的生命周期为：

- `endOfShower()`：写出当前 observer 波形、reset observer、增加 shower ID；
- `endOfLibrary()`：只在整个 library 结束时关闭 Parquet streamer。

`testRadio` 对 CoREAS 和 ZHS 都连续调用两次 `endOfShower()`，确认第二次不会
抛异常，并在 `endOfLibrary()` 后确认文件已写出。

## 3. 1 TeV、10-event 射电正式 ensemble

输出：

```text
/tmp/c8_phase65_radio_acceptance_1TeV_10_v2
```

配置：

```text
electron primary            1 TeV
events                      10
observers                   8
EM cut                      0.5 MeV
EM thinning                 1e-4
maximum weight              100
GPU minimum batch           64
```

五类非射电输运表全部逐行相同：

| 表 | 行数 | CPU-radio/CUDA-radio |
|---|---:|---|
| `energyloss/dEdX.parquet` | 12430 | exact |
| `profile/profile.parquet` | 12430 | exact |
| `particles/particles.parquet` | 369 | exact |
| `production_profile/profile.parquet` | 12430 | exact |
| `interactions/interactions.parquet` | 21 | exact |

波形比较：

| 算法 | shower | observer | component | 最大归一化差异 | 最大相对 L2 | 最大 fluence 差异 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| CoREAS | 10 | 8 | 240 | \(3.8033\times10^{-7}\) | \(3.2119\times10^{-7}\) | \(7.3527\times10^{-8}\) | PASS |
| ZHS | 10 | 8 | 240 | \(9.8139\times10^{-7}\) | \(7.6028\times10^{-7}\) | \(1.3117\times10^{-7}\) | PASS |

归一化/L2 门限均为 \(10^{-4}\)，fluence 门限为 \(5\times10^{-4}\)。

## 4. 1 PeV 射电高轨迹负载

输出：

```text
/tmp/c8_phase66_radio_acceptance_1PeV_1_v1
```

这是同样的 8-observer 双后端比较，但 primary energy 提高到 1 PeV：

| 算法 | 最大归一化差异 | 最大相对 L2 | 最大 fluence 差异 | 状态 |
|---|---:|---:|---:|---|
| CoREAS | \(4.7271\times10^{-10}\) | \(3.7843\times10^{-10}\) | \(1.3168\times10^{-10}\) | PASS |
| ZHS | \(3.9413\times10^{-7}\) | \(2.7480\times10^{-7}\) | \(1.0733\times10^{-7}\) | PASS |

五类非射电输运表继续逐行相同。这证明高能长轨迹集合不会改变输运结果，也不会
破坏 CPU-compatible `RadioTrackRecord`。

## 5. 1 TeV proton 正式强子 ensemble

### 5.1 20+20 诊断样本

输出：

```text
/tmp/c8_phase67_proton_1TeV_45deg_20_v1
```

所有 9 类曲线通过，但 `profile_photon_integral` 在 20 场样本中出现
28.0%、3.02 sigma 的边界偏差。由于强子首相互作用涨落很大，不能据此修改
实现或放宽门限，因此扩展独立样本。

### 5.2 100+100 扩展样本

输出：

```text
/tmp/c8_phase68_proton_1TeV_45deg_100_v1
```

配置为 proton、1 TeV、45 degree、0.3 GeV hadron/muon/tau cut、0.5 MeV
EM cut、\(10^{-4}\) thinning。

- 全部 9 类纵向/沉积/地面 shape gate 通过；
- `profile_charged_integral` 均值差异 0.733%；
- `profile_photon_integral` 从 20-event 的 28.0% 降为 1.306%，只有
  0.312 sigma；
- `energy_deposit_sum` 差异 1.311%，只有 0.649 sigma；
- 其余超过 1% 的高方差标量均小于 1.81 sigma。

因此 20-event 的边界失败没有在扩展样本中重现。严格的“所有 raw mean 小于
1%”仍不能由 100 场高方差强子样本证明，所以状态保留为统计精度不足，而不是
宣称完整 1% 验收已完成。

## 6. proton + fallback + radio 交叉验收

输出：

```text
/tmp/c8_phase69_proton_radio_acceptance_1TeV_2_v1
```

两场 proton shower 均完整结束：

| shower | GPU EM 粒子 | CPU steps | 指定 CPU 末态 | generic fallback | radio tracks |
|---|---:|---:|---:|---:|---:|
| 0 | 263801 | 7827 | 8 | 0 | 247072 |
| 1 | 175518 | 7711 | 1 | 0 | 164502 |

9 次指定 CPU 末态全部是可审计的
`photoproduction / cpu_only_process`。CPU-radio 与 CUDA-radio 两次运行的
上述统计完全一致，且非射电输出逐行相同。

CUDA radio 实际累计：

```text
track-observer pairs     6,624,336
CoREAS contributions    5,685,149
ZHS contributions      15,041,280
fixed-point overflows           0
```

波形最大归一化差异：

```text
CoREAS    5.3190e-8
ZHS       1.8403e-6
```

两者均通过 \(10^{-4}\) 门限。这直接覆盖了 CPU 强子、CUDA EM、指定
PROPOSAL fallback 和射电轨迹合并的完整生产路径。

## 7. 结论

- 多 shower 射电输出生命周期错误已经修复并有单元测试与 10-event 生产运行
  双重证据；
- CPU-compatible CoREAS/ZHS 轨迹接口在 TeV ensemble 与 PeV 高轨迹负载下
  通过；
- 可选 resident CUDA radio 在相同轨迹上通过严格波形比较；
- 强子混合链路已有 100+100 场 shape 验收和含实际 photoproduction fallback
  的射电交叉验收；
- 强子 raw means 的统一 1% 结论仍受 shower 方差限制，不能把当前结果扩大解释
  为原计划所有统计门禁均完成。
