# 阶段 91：1 PeV `emthin` 边界、完整射电事例与局部能量账本

## 1. 结论

本阶段完成了三个容易混淆但必须分开的验证：

1. 只指定 `--emthin` 是原版 `c8_air_shower` 支持的合法方式，CUDA 路径保持
   了相同接口和自动 maximum-weight 公式。
2. 对精确的 1 PeV、`emthin=1e-6`、未显式指定 `--max-weight` 组合，自动
   `maxWeight=0.5`，小于初始粒子权重 1，所以原版和 CUDA 的保护条件都会让
   thinning 无法启动。该组合实际是未薄化的高质量极限。
3. 当前 cost-triggered FLUKA process pool、CUDA EM 和 CUDA CoREAS/ZHS 已经
   联合完成一场 1 PeV、81 天线的完整事例。强子进程池只占 Hybrid wall 的
   0.052%，主要门槛是十亿量级 EM 输运及射电投影，不是 FLUKA。

“`1e-6` 比 `1e-5` 更慢并提供更好数据质量”仍然正确。这里修正的是原因：
在这个精确的 1 PeV 自动权重边界上，`1e-6` 不是发生了更弱的 thinning，而是
没有发生 thinning。在更高初能或显式 `--max-weight > 1` 时，
`emthin=1e-6` 会正常实际薄化。

## 2. 原版与 CUDA 的代码证据

原版应用和当前重构版都使用：

```cpp
maxWeight =
    0.5 * emthinfrac * primaryTotalEnergy / 1_GeV;
```

原版 CPU `EMThinning`：

```cpp
double const parentWeight = projectile.getWeight();
if (parentWeight >= maxWeight_) { return; }
```

CUDA `applyEmThinning` 也有同样的判定：

```cpp
if (parent_weight >= config.maximum_weight || ...) {
  return not_applied;
}
```

当

\[
E_0=10^6\ \mathrm{GeV},\qquad
\epsilon_{\rm thin}=10^{-6}
\]

时，

\[
E_{\rm thin}=1\ \mathrm{GeV},\qquad
w_{\max}=0.5.
\]

初始权重为 1，满足 \(1\ge0.5\)，所以 thinning 在第一个满足能量阈值的二体
EM 顶点就会提前返回；因为没有产生大于 1 的 thinned weight，后续也不会自行
启动。

相关实现：

- `applications/c8_air_shower.cpp`
- `corsika/detail/modules/thinning/EMThinning.inl`
- `corsika/gpu/em/EmThinning.hpp`

## 3. 新增的 fail-visible 审计

应用现在在 shower 初级能量确定后记录：

```yaml
thinning:
  em_fraction: ...
  maximum_weight: ...
  automatic_maximum_weight: true|false
  can_activate_from_unit_weight: true|false
```

若 `emthin > 0` 但有效 `maximum_weight <= 1`，程序明确警告：

```text
EM thinning is configured ..., but maximum weight ... is not above
the unit initial particle weight; the parentWeight >= maxWeight guard
prevents thinning from starting.
```

YAML 验收适配器
`validation/gpu_em/run_config_acceptance.py` 使用同一公式，在真正启动昂贵运行前
给出：

```json
{
  "effective_em_thinning": 1e-6,
  "effective_maximum_weight": 0.5,
  "automatic_maximum_weight": true,
  "thinning_can_activate_from_unit_weight": false
}
```

它不会替用户改写参数。若希望 1 PeV 下实际薄化，可提高 `emthin`，例如：

| `emthin` | 自动 `maxWeight` | unit-weight 粒子能否启动 thinning |
|---:|---:|---|
| `1e-6` | 0.5 | 否 |
| `1e-5` | 5 | 是 |
| `1e-4` | 50 | 是 |

也可以显式给 `--max-weight > 1`，但这已经是与原 task 不同的物理配置，必须在
manifest 中记录，不能静默替换。

## 4. 当前完整 1 PeV 事例

输出目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
current_hadronic_cost_bulk_proton_1PeV_emthin1e-6_radio_seed72001_v1
```

主要配置：

```text
primary              proton
energy               1e6 GeV
zenith/azimuth       27 deg / 180 deg
seed                 72001
emcut                0.5 MeV
emthin               1e-6
maximum weight       automatic = 0.5
antennas             /home/yuhanglu/21CMA/data/antennas.txt
radio                CUDA CoREAS + CUDA ZHS
hadronic             SIBYLL-2.3d + FLUKA 2025.0.0
FLUKA workers         4
```

运行使用的 `c8_air_shower` SHA-256：

```text
7f31b49c0d80510dcdd59a6df30db2e8a47824086b2eb33ce856995e44611fcc
```

运行之后只增加了 thinning 边界警告和 metadata，未改变物理路径。当前重新
构建二进制 SHA-256：

```text
fbe175bdc556924492c3f3b7e3c81fa5362b5b199797bc5e89c4248abee4fba7
```

FLUKA worker 和物理表 SHA-256：

```text
fluka_batch_worker:
89402364853bfd5a5c7b637c880b6677de0736bd2b955b0e8fd06318991892ec

PROPOSAL CUDA table:
14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521
```

### 4.1 完成性和 thinning 实证

```text
status / complete           complete / true
external runtime            992.43 s
Hybrid total                986.109 s
GPU transport particles     1,063,155,468
GPU final states            157,042,426
physical secondaries        314,104,125
wavefronts                  6,450
queue overflow              0
memory spill                0
```

thinning 统计为：

```yaml
em_fraction: 1e-6
maximum_weight: 0.5
hillas_vertices: 0
statistical_vertices: 0
particles_discarded: 0
```

`particles/particles.parquet` 共 2,056,139 行，`weight` 的最小值、最大值和
唯一值都为 1。这同时排除了“只是在输出末端碰巧看不到 thinning 计数”的解释。

### 4.2 强子后端

```text
SIBYLL high-energy interactions       1,131
SIBYLL final-state CPU time           51.91 ms
FLUKA interactions                    15,705
FLUKA four-worker CPU-time sum        1,777.55 ms
FLUKA process-pool wall               516.50 ms
FLUKA pool / Hybrid                   0.05238%
```

新的 cost trigger 形成 73 个正常批次和 7 个 drain 尾批，没有 capacity flush。
正常批次的预测 worker max/min 均值为 1.0153、最差 1.0489；实测 final-state
耗时 max/min 均值为 1.2182、最差 1.6263。四个 worker 的累计时间为
436.67–455.33 ms，累计 max/min 为 1.0427。

bulk-response wire protocol v3 共处理 313 个 IPC super-batch，只需 626 次 payload
read；接收时间 11.72 ms，等待 FLUKA 完成时间 493.70 ms。当前没有性能证据支持
把 SIBYLL/FLUKA 物理内核迁移到 GPU。

### 4.3 CUDA EM 与射电

```text
GPU radio tracks                       920,928,374
logical track-observer pairs           154,063,658,070
fused physical pairs                    77,031,829,035
CoREAS contributions                   131,043,064,952
ZHS contributions                      270,385,335,877
ZHS subtracks                           76,500,136,195
radio device time                      665,267.79 ms
fixed-point overflows                  0
```

CoREAS 和 ZHS 都写出 32,400 行，即 81 个天线乘 400 个时间 bin；所有
`Time, Ex, Ey, Ez` 有限。

| formalism | Ex 范围 (V/m) | Ey 范围 (V/m) | Ez 范围 (V/m) |
|---|---:|---:|---:|
| CoREAS | \([-1.175,1.795]\times10^{-6}\) | \([-1.347,14.324]\times10^{-6}\) | \([-0.922,0.592]\times10^{-6}\) |
| ZHS | \([-1.026,1.311]\times10^{-6}\) | \([-1.505,14.567]\times10^{-6}\) | \([-1.484,1.092]\times10^{-6}\) |

这证明完整输出和数值有限，不等于单个 seed 已经证明原版 CPU/CUDA 射电统计
一致。地磁振幅、宽度和 scaling law 仍必须使用多个独立 seed 形成分布。

## 5. 为什么局部 energy ledger 没有通过完整 shower gate

独立验证器重算得到：

```text
complete_coverage              false
internally_consistent          true
source                         1,052,228.897 GeV
terminal                         960,780.532 GeV
residual                          91,448.365 GeV
relative residual                     8.6909%
```

这不是发现了 8.69% 的物理能量不守恒。当前 ledger 的严格 \(10^{-4}\) gate
只适用于完全驻留的纯 EM shower。应用源码只有同时满足以下条件才设置
`complete_coverage=true`：

- 初级是 \(\gamma/e^\pm\)；
- 没有 EM thinning；
- 没有 scalar EM step；
- 没有 CPU fallback、memory spill 或 CPU 指定末态。

本事件是 proton 初级，含 139,100 个 scalar EM step、10,267 个 CPU 指定末态
以及完整的 hadron/μ/decay 路径。这些通道没有全部进入当前 GPU 局部 terminal
定义，所以验证器正确地拒绝把它当作严格完整账本。`internally_consistent=true`
只表示存储的 primitive terms 与重算公式一致。

在补齐统一的 hadron/μ/decay/scalar ledger 前，完整强子 shower 应使用与原版
相同定义的 writer observable 做 ensemble 比较；不能把 partial ledger 的
residual 当作 CUDA 能量损失，也不能把它标成 \(10^{-4}\) closure 通过。

## 6. 下一步验收策略

1. 以 `emthin=1e-5` 做较快的 CPU/CUDA 多 seed 输运和射电分布预验收；它在
   1 PeV 下自动 `maxWeight=5`，会实际薄化。
2. 保留少量 `emthin=1e-6` 作为未薄化高质量参考，用于检查薄化偏差和关键
   scaling law；不再把它笼统称为“较弱 thinning”。
3. 每个档位同时记录阈值、有效 maximum weight、Hillas/statistical vertex
   数、discard 数和落地权重范围。
4. 用 `pulse_analysis_modular` 的同一定义提取地磁分量振幅与宽度；以独立
   seed 分布比较 CPU/CUDA，而不是要求不同调度和随机流的单 shower 逐位相同。
5. 继续优先优化射电投影和剩余 scalar EM 路径。只有强子 process-pool 在新的
   representative ensemble 中稳定超过总 wall 的 10%，才启动强子 GPU 内核。
