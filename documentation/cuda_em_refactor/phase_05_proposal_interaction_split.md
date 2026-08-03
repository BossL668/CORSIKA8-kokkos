# CUDA 电磁后端重构记录：阶段 5，PROPOSAL 相互作用 CPU 拆分

## 1. 本阶段目标

阶段 5 不增加 CUDA 电磁物理，而是在 CPU 上把原先耦合在
`proposal::InteractionModel::doInteraction()` 中的操作拆成三个明确阶段：

```text
PROPOSAL::Interaction::Rates()
  │
  ▼
ProposalRateProvider
  ├─ 公开每个 process × component 的 rate
  └─ 只采样一次 type、component_hash 和 v
        │
        ▼
ProposalInteractionRecord
        │
        ▼
ProposalFinalStateGenerator
  └─ 对指定 record 调用 CalculateSecondaries()
        │
        ▼
原有 LPM rejection 和 CORSIKA stack 写入
```

拆分后的生产 `doInteraction()` 已经使用这些新接口。原有默认
`Cascade + PROPOSAL` 路径不需要修改调用方式。

## 2. 拆分前的真实调用链

实现位置：

```text
corsika/detail/modules/proposal/InteractionModel.inl
```

PROPOSAL 7.6.2 原调用顺序是：

1. 根据 CORSIKA 粒子的核组成 hash 和 PID 查找 calculator；
2. `PROPOSAL::Interaction::Rates(E)` 计算所有过程和靶组分的 rate；
3. 从 CORSIKA `"proposal"` RNG stream 抽取一个均匀随机数；
4. `SampleLoss(E, rates, u)` 同时选出：
   - `InteractionType`；
   - target component hash；
   - fractional loss \(v\)；
5. `RequiredRandomNumbers(type)` 查询末态需要的随机数数量；
6. 从同一个 RNG stream 依次抽取末态随机数；
7. 构造 `PROPOSAL::StochasticLoss`；
8. `CalculateSecondaries(loss, target, randoms)` 生成末态；
9. 对 Photopair、Brems 和 Epair 执行 CORSIKA 额外的 LPM rejection；
10. 将电磁末态写入 CORSIKA stack，或把 hadron placeholder 交给原有强子模型。

LPM rejection 位于末态生成之后，并且可能再消耗一个 `"proposal"` 随机数。
本阶段没有移动它，否则固定 seed 下的 RNG 消费位置会改变。

## 3. `ProposalInteractionRecord`

定义文件：

```text
corsika/modules/proposal/ProposalInteractionRecord.hpp
```

record 保存已经确定的相互作用，而不是一个可以重新采样的请求：

```cpp
struct ProposalInteractionRecord {
  ProposalInteractionContext context;
  std::size_t interaction_hash;
  PROPOSAL::InteractionType type;
  std::size_t component_hash;
  double v_loss;
  double selection_uniform;
  std::optional<ProposalRandomKey> random_key;
};
```

`ProposalInteractionContext` 使用 PROPOSAL 原生单位：

| 字段 | 单位或含义 |
|---|---|
| `projectile_id` | CORSIKA `Code` |
| `medium_hash` | CORSIKA 核组成 hash |
| `projectile_energy_MeV` | MeV，总能量 |
| `position_cm[3]` | cm |
| `direction[3]` | 无量纲方向 |
| `time_s` | s |

`interaction_hash` 标识产生这个 rate table 的完整 PROPOSAL calculator。
CPU fallback 生成末态前会同时检查：

- medium hash；
- projectile code；
- calculator hash；
- interaction type 是否属于该 calculator。

因此不能把一个 electron/氧介质的 record 静默交给 photon/氮介质 calculator。

### 3.1 随机数键

`ProposalRandomKey` 已固定未来设备回退需要的字段：

```text
seed, shower_id, history_id, step_id, process_id, draw_id
```

当前 legacy CPU 路径仍使用顺序型 `"proposal"` RNG stream，因此
`random_key` 为空。未来 GPU 选择过程后可以附带 Philox 地址，不需要修改
record 的物理字段或 CPU final-state 接口。

## 4. `ProposalRateProvider`

定义文件：

```text
corsika/modules/proposal/ProposalRateProvider.hpp
```

### 4.1 rate table

`ProposalRateTable` 同时保存两种视图：

- `entries()`：`type + component_hash + rate`，供验证和未来表格生成；
- `nativeRates()`：PROPOSAL 7.6.2 的 cross-section shared pointer，供当前
  CPU `SampleLoss()` 使用。

rate 使用 PROPOSAL 原生 \(dN/dX\) 约定。本阶段没有偷偷加入单位换算；
正式 `gpu_em_tablegen` 必须显式定义磁盘单位和 schema。

provider 对以下情况立即报错：

- 能量非有限或不大于零；
- rate 缺少 cross-section；
- 单项 rate 为负或非有限；
- total rate 非有限；
- rate table 和 vertex 能量不一致；
- calculator hash 不一致；
- 选择随机数不在 `[0, 1)`；
- PROPOSAL 返回非法 \(v\)。

### 4.2 只采样一次

`sample()` 仍调用 PROPOSAL 原生：

```cpp
interaction.SampleLoss(energy, native_rates, selection_uniform);
```

结果立即冻结到 `ProposalInteractionRecord`。后续 final-state API 没有
`Rates()` 或 `SampleLoss()` 入口，因此不能在 CPU fallback 中重新选择过程。

## 5. `ProposalFinalStateGenerator`

定义文件：

```text
corsika/modules/proposal/ProposalFinalStateGenerator.hpp
```

接口分成：

```cpp
std::size_t requiredRandomNumbers(
    SecondariesCalculator const&, ProposalInteractionRecord const&);

ProposalFinalState generate(
    SecondariesCalculator&, ProposalInteractionRecord const&,
    std::vector<double> random_numbers);
```

`generate()` 只使用 record 中的：

- interaction type；
- component hash；
- \(v\)；
- projectile energy；
- vertex position、direction 和 time。

它随后直接调用：

```cpp
CalculateSecondaries(loss, target, random_numbers);
```

不会重新调用 `Rates()` 或 `SampleLoss()`。

输入检查包括：

- interaction 不能是 `Undefined`；
- \(v\) 必须位于 `[0, 1]`；
- vertex 能量、位置、方向和时间必须有限；
- 方向不能为零；
- 随机数数量必须精确匹配 `RequiredRandomNumbers()`；
- 每个末态随机数必须位于 `[0, 1)`。

## 6. `InteractionModel` 的公开分层接口

`proposal::InteractionModel` 新增：

```cpp
getRateTable(particle, projectile_id);

sampleInteraction(
    particle, projectile_id, rate_table, selection_uniform, random_key);

requiredFinalStateRandomNumbers(record);

generateFinalState(record, random_numbers);
```

当前 `doInteraction()` 的随机数顺序为：

```text
draw selection u
  -> sampleInteraction()
draw N final-state randoms
  -> generateFinalState()
possible LPM draw
```

与拆分前完全相同。`getRateTable()` 和 calculator/hash 检查不抽取随机数。

## 7. 测试

### 7.1 PROPOSAL 接口测试

`tests/modules/testProposal.cpp` 现在完成 67 个断言，新增覆盖：

- 公开 rate entry 与 PROPOSAL native rate 逐项相同；
- total rate 为正且有限；
- 手工复现 PROPOSAL 7.6.2 的 rate 累减选择算法；
- type、component hash 和 \(v\) 与 `sampleInteraction()` 完全一致；
- vertex 的 MeV/cm/s 单位转换；
- random key 无损保存；
- 相同 record 和相同随机数组产生逐项相同的 `ParticleState`；
- target hash 保持一致；
- 错误随机数数量、非法选择随机数、`Undefined` record 和错误 calculator
  hash 均抛出异常；
- 完整 `doInteraction()` 通过新路径产生 CORSIKA secondaries。

### 7.2 完整 CPU 回归

```text
testFramework  passed
testModules    passed（完整 41 个 module test cases）
c8_air_shower  compiled and linked
```

测试最初在 `ProposalInterface` 内设置了全局 seed，导致后续 SOPHIA 测试使用
不同随机路径。该测试副作用已删除；固定 seed 等价性改由独立 shower 进程
验证，不再污染聚合测试进程。

### 7.3 拆分前后 shower 黑盒比较

使用：

```text
primary  electron
energy   10 GeV
seed     12345
```

分别运行阶段 5 之前保存的旧 `em_shower` 二进制和当前新二进制。
以下物理文件 SHA-256 逐字节一致：

```text
energyloss/dEdX.parquet
profile/profile.parquet
particles/particles.parquet
tracks/tracks.parquet
```

其余 config 和 primary 输出也一致。差异只有：

- 顶层 `summary.yaml` 的 runtime；
- `energyloss/summary.yaml` 的末位浮点求和。

例如 `sum_dEdX`：

```text
old  9.8983111853391978
new  9.8983111853391925
```

绝对差约 \(5.3\times10^{-15}\)，而底层 `dEdX.parquet` 本身逐字节一致。
这说明物理 step、过程选择、末态和 RNG 消费顺序保持一致；summary 差异是
聚合求和的机器精度末位，不是 shower 内容变化。

### 7.4 CUDA 回归

PROPOSAL CPU 拆分后重新验证：

```text
testGpuEmHost        passed
testGpuEmCppLink     passed
testGpuHybridRoute   11 checks passed
testGpuEmCuda        11 deterministic outputs
```

## 8. PROPOSAL 缓存说明

第一次运行纯氧测试环境时，PROPOSAL 在数据子模块中生成了 54 个缺失的
临时插值表。测试完成后已经只删除这 54 个未跟踪文件，保留原有 1142 个
受 Git 管理的表；数据子模块最终没有未提交修改。

这些临时表可由 PROPOSAL 重新生成，因此删除不可从回收站恢复，但不会丢失
源码或唯一数据。未来 `gpu_em_tablegen` 应把生成缓存移到显式
`--gpu-table-cache`，避免测试向源码数据目录写入。

## 9. 当前限制

- rate table 仍包含 host-only PROPOSAL shared pointer，不能上传 GPU；
- `ProposalRateEntry` 还不是磁盘格式；
- legacy CPU record 没有 Philox key；
- LPM rejection 仍在 CORSIKA 层、末态生成之后；
- photo-hadronic、photo-muon-pair 等 CPU fallback 尚未批处理；
- 没有实现版本化 metadata、内容 hash 或 \(10^{-3}\) 插值误差验证；
- continuous range、\(dE/dX\) 和 multiple scattering 尚未进入表格接口；
- `c8_air_shower` 仍未公开 `--em-backend cuda`。

因此当前成果是“CPU 物理接口拆分完成”，不是 GPU 电磁物理完成。

## 10. 下一步

下一阶段应实现 `gpu_em_tablegen` 的最小版本，而不是立刻编写 pair/brems
kernel：

1. 定义固定宽度、带版本号的 table metadata；
2. 记录 PROPOSAL 7.6.2、参数化名称、介质组成、cut 和能区；
3. 首先导出 photon/electron/positron 的
   `process × component × energy` rate；
4. 增加 CPU reader、内容 hash 和损坏/版本不匹配拒绝策略；
5. 用直接 `ProposalRateProvider` 计算验证表格插值误差；
6. rate 表稳定后，再加入 \(v(E,u)\) 逆 CDF；
7. 最后才让 toy CUDA kernel 查询第一张真实物理表。
