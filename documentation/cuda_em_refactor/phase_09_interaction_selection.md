# CUDA 电磁后端重构记录：阶段 9，离散相互作用选择

> 本文末尾规划的 process capability map 与首个 photon pair GPU 末态已经在
> 阶段 10 完成。实现和验证见
> [phase_10_photon_pair_final_state.md](phase_10_photon_pair_final_state.md)。

## 1. 本阶段完成了什么

阶段 8 已经把 PROPOSAL v3 物理表转换为只读 GPU SoA，并实现了设备端 rate
与 inverse-CDF 查询。本阶段第一次用这些表格完成了真实的随机物理选择：

```text
EmParticleState batch
        │
        ├─ 查询总离散反应率
        ├─ 抽取相互作用 grammage
        ├─ 抽取 process × target component
        └─ 查询该过程能损分数 v
        │
        ▼
raw interaction / fallback records
        │
        │ CUB exclusive scan
        ▼
稳定有序的 interaction queue + CPU fallback queue
```

这里的“真实”是指 rate、过程概率和 \(v\) 都来自阶段 7 生成的 PROPOSAL
7.6.2 表，而不再是 toy branching 概率。

本阶段仍不是完整的电磁 shower：选择完成后尚未生成 pair production 或
bremsstrahlung 次级粒子，也尚未把相互作用距离与大气边界、连续步长和观测面
进行竞争。现有 `advanceWavefront()` 因此仍保留 toy 行为，真实选择通过独立
验证接口调用，避免把未完成的物理路径误接入生产模拟。

## 2. `CudaEmBackend` 现在拥有物理表

`CudaEmBackend::initialize()` 现在读取：

```cpp
GpuEmConfig config;
config.table_cache = "/path/to/table.c8emrt";
config.table_tolerance = 1.e-3;

ProposalTableSet expected;
expected.format_version = 3;
expected.process_count = ...;
expected.content_hash = ...;

backend.initialize(environment, expected, config);
```

初始化严格按以下顺序执行：

1. 检查设备编号、batch 大小、显存比例和误差阈值；
2. `readRateTable()` 检查文件 envelope、格式版本与内容 SHA-256；
3. 检查 rate 和 inverse-CDF 的请求误差及实测最大误差；
4. 若调用方给出非零 process/column 数，检查它与文件一致；
5. 若调用方给出非零 hash，检查它与文件 payload hash 一致；
6. 扁平化并上传 `CudaRateTable`；
7. 在剩余配置预算内建立粒子双缓冲队列。

任何一步失败都会释放本次已经分配的资源并终止初始化，不会静默换用 CPU
表格，也不会继续使用不匹配的缓存。

表格和粒子队列现在共享同一个：

```text
memory_fraction × initialize 时的可用显存
```

预算。`GpuEmStatistics::table_device_bytes` 单独记录物理表占用；
`peak_device_bytes` 至少包含物理表和常驻 toy 队列。当前验证接口自身的临时
AoS、raw record 和 compaction buffer 尚未计入该峰值，这也是它不能直接作为
生产 wavefront 的原因之一。

## 3. 相互作用抽样算法

### 3.1 总反应率

对粒子种类 \(p\)、能量 \(E\)，GPU 遍历该粒子的全部
`process × component` column：

\[
\lambda_{\mathrm{tot}}(E)
  = \sum_c \lambda_c(E),
\]

其中 rate 单位为：

\[
[\lambda] = \mathrm{cm^2/g}.
\]

这里的 \(\lambda\) 是单位 grammage 的反应率，而不是单位几何长度的反应率。
因此在当前层只抽取 \(X\)，不依赖大气密度或轨迹几何。

如果粒子不在表中或能量超出表格能力域，则直接生成带原因的 fallback event。
如果查询成功但 \(\lambda_{\mathrm{tot}}=0\)，记录状态为
`NoDiscreteInteraction`，并令相互作用 grammage 为无穷大。它不是错误，也
不应送给 CPU 重新抽取。

### 3.2 相互作用 grammage

离散相互作用满足 Poisson 过程。若

\[
u_X \sim U(0,1),
\]

则到下一次相互作用的柱深为：

\[
X_{\mathrm{int}}
  = -\frac{\ln u_X}{\lambda_{\mathrm{tot}}}.
\]

单位自然为 \(\mathrm{g/cm^2}\)。代码保留 `distance_uniform`、
`total_rate_cm2_per_g` 和 `interaction_grammage_g_per_cm2`，所以 CPU
可以逐事件复核抽样，而不只看到最终距离。

注意：这里得到的是候选离散相互作用柱深。将来输运 kernel 还要把它与：

- 大气层边界；
- 最大磁偏转步长；
- 连续能损步长；
- 观测面；

进行竞争，选择最近限制。

### 3.3 过程与介质组分

第二个独立均匀数 \(u_c\) 给出阈值：

\[
T = u_c \lambda_{\mathrm{tot}}.
\]

GPU 按物理表中固定的 column 顺序累加：

\[
S_k = \sum_{c \le k}\lambda_c(E),
\]

第一个满足 \(T<S_k\) 的 column 被选中。因此：

\[
P(c)=\frac{\lambda_c(E)}{\lambda_{\mathrm{tot}}(E)}.
\]

记录中保存 `process_id` 和完整 64 位 `component_hash`。最后一个正 rate
column 还作为浮点累积误差的保护：若理论上 \(T<\lambda_{\mathrm{tot}}\)，
但累计舍入没有越过阈值，选择最后一个正 column，而不是丢失粒子。

### 3.4 能损分数

第三个均匀数 \(u_v\) 由已选中的 `process_id` 参与随机数 key。GPU 查询该
`process × component` 的逆累计分布：

\[
v = F^{-1}_{p,c}(E,u_v).
\]

结果保存在 `energy_fraction`。若该 column 没有 inverse-CDF，或当前能量、
分位点超出它的能力域，则不夹紧输入，而是产生 CPU fallback。fallback 保存：

- 原始粒子；
- 原 batch 的 `input_index`；
- 已选定的 process ID；
- component hash；
- 已抽取的 \(u_v\)；
- draw ID；
- 精确失败原因。

因此未来 CPU `ProposalFinalStateGenerator` 只需生成这个已经选定的末态，
不得重新抽取过程。

## 4. 随机数为什么不依赖 GPU 调度顺序

三个随机变量均使用设备端 Random123 Philox4x32-10。key 由物理身份构造，而
不是从一个共享全局 RNG 状态递增：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

当前分工为：

| 随机变量 | process key | draw ID |
|---|---:|---:|
| 相互作用 grammage | `0x454d0001` | 0 |
| column 选择 | `0x454d0002` | 0 |
| \(v\) 选择 | 已选 `process_id` | 0 |

因此同一 history 的结果只取决于粒子身份和 shower 配置。测试把输入粒子顺序
完全反转后，再按 `history_id` 对齐比较，过程、组分、三个均匀数、\(v\) 和
grammage 都保持不变。

这保证了同一 GPU 和相同配置下的可重复性，也允许将来按 PID、介质和能段
重新分桶，而不改变单个 history 的随机流。

## 5. 输出记录

新增两个 batch 类型：

```cpp
struct EmInteractionRecord {
  EmParticleState particle;
  std::uint64_t input_index;
  std::int32_t process_id;
  EmInteractionStatus status;
  std::uint64_t component_hash;
  double total_rate_cm2_per_g;
  double interaction_grammage_g_per_cm2;
  double energy_fraction;
  double distance_uniform;
  double process_uniform;
  double loss_quantile;
  std::uint64_t distance_draw_id;
  std::uint64_t process_draw_id;
  std::uint64_t loss_draw_id;
};

struct EmInteractionBatchResult {
  std::size_t input_particles;
  std::vector<EmInteractionRecord> interactions;
  std::vector<ProposalFallbackEvent> fallback_events;
};
```

这两个设备记录都是 standard-layout、trivially-copyable POD。成功记录和
fallback 记录都保留输入身份，便于检查没有粒子丢失、重复或乱序。

`GpuEmStatistics` 同时增加：

```text
interaction_selection_batches
interactions_selected
proposal_fallbacks
table_device_bytes
```

## 6. 为什么需要 CUB scan

每个 GPU 线程处理一个输入粒子，但成功和 fallback 的数量事先未知。直接用
全局 `atomicAdd` 追加会让输出顺序依赖 warp 调度。

本实现先令：

\[
F_i =
\begin{cases}
1,& i\text{ 是 fallback},\\
0,& i\text{ 是 success}.
\end{cases}
\]

CUB exclusive scan 给出：

\[
O_i = \sum_{j<i}F_j.
\]

于是线程 \(i\) 的输出位置唯一确定：

```text
fallback index = O_i
success index  = i - O_i
```

这只需要一次 scan，就能同时得到两个稳定 compaction 队列。两个输出队列
内部都严格保持输入顺序，不使用不确定的全局原子追加。

## 7. 代码位置

| 功能 | 文件 |
|---|---|
| 设备选择 API 与随机流常量 | `corsika/gpu/em/CudaInteractionSelector.hpp` |
| rate、grammage、column、\(v\) kernel | `src/gpu/em/CudaInteractionSelector.cu` |
| CUB scan 与双队列 compaction | `src/gpu/em/CudaInteractionSelector.cu` |
| 物理记录与 fallback schema | `corsika/gpu/em/Types.hpp` |
| 查询错误到 fallback 原因的映射 | `corsika/gpu/em/ProposalFallback.hpp` |
| 表格所有权、校验、统计 | `src/gpu/em/CudaEmBackend.cu` |
| 独立 CPU 参考和 GPU 对照测试 | `tests/gpu/testGpuInteractionSelection.cpp` |

对外验证接口是：

```cpp
EmInteractionBatchResult
CudaEmBackend::selectInteractionsForValidation(
    std::vector<EmParticleState> const&);
```

它不读取、不推进也不清空现有 toy wavefront 队列。

## 8. 验证结果

### 8.1 小型 ragged fixture

fixture 故意包含缺失粒子、能区外粒子、零 rate column 和无 inverse-CDF
column。结果：

```text
4098 input particles
4081 interaction/no-interaction records
17 fallback events
77,781 checks passed
492 device table bytes
```

额外拒绝测试覆盖：

- descriptor hash 不一致；
- descriptor process/column 数不一致；
- descriptor 格式版本不一致；
- 表格误差大于配置阈值；
- 未上传表格时调用物理选择。

### 8.2 完整 PROPOSAL v3 表

输入表：

```text
/tmp/c8_gpu_em_full_strict_v3.c8emrt
dry air
0.5 MeV cut
0.6 -- 1e12 MeV
1e-3 rate/loss tolerance
gamma, electron, positron
```

结果：

```text
8192 input particles
8116 selected interactions
76 fallback events
155,207 checks passed
15,118,340 device table bytes
elapsed 0.73 s
peak host RSS 176,100 KiB
```

76 个 fallback 是明确的 inverse-CDF 能力域或表格状态结果，不会被静默
忽略。当前阶段只验证它们的稳定生成，还没有交给 CPU 末态生成器消费。

除逐事件与独立 CPU 参考一致外，测试还检查：

- 同一个 batch 连续执行两次完全一致；
- 反转输入顺序不改变每个 history 的结果；
- 两个 compact 输出队列均保持输入顺序；
- success 加 fallback 等于输入总数；
- \(X_{\mathrm{int}}\lambda_{\mathrm{tot}}\) 的样本均值与指数分布期望 1
  相符；
- backend 的 batch、success、fallback 计数正确累计。

### 8.3 回归

| 测试 | 结果 |
|---|---|
| CUDA 相关 CTest | 8/8 通过 |
| 纯 CPU 相关 CTest | 3/3 通过 |
| `testGpuEmHost` | 156 checks 通过 |
| `testGpuEmRateTable` | 32 checks 通过 |
| `testGpuEmFlatRateTable` | 102 checks 通过 |
| `testGpuEmRateTableCuda` fixture | 6,450 checks 通过 |
| `[ScalarCascadeStepper]` + `[HybridCascade]` | 15 assertions 通过 |
| `testGpuEmCuda` | 11 个确定性 toy 粒子 |
| `testGpuHybridRoute` | 11 checks 通过 |

## 9. 构建与复现

构建：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO \
  cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuInteractionSelection -j2
```

运行 fixture：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/gpu/\
testGpuInteractionSelection
```

运行完整表：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/gpu/\
testGpuInteractionSelection /tmp/c8_gpu_em_full_strict_v3.c8emrt
```

运行 CUDA 回归：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO \
  ctest --test-dir \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --output-on-failure -R 'Gpu(Em|Hybrid|Interaction)'
```

## 10. 当前限制

当前实现有意保留以下边界：

1. `selectInteractionsForValidation()` 每次调用都上传 AoS 输入，并临时分配 raw
   record、flag、scan 和 compact buffer；
2. 它尚未直接读取 `CudaEmBackend` 的常驻粒子 SoA；
3. 它只抽取候选离散相互作用，没有与 tracking 限制竞争；
4. 它没有生成任何物理次级粒子，也没有计算角分布；
5. photonuclear、光致 muon pair 等 CPU-only 过程尚未通过 capability map
   自动路由；
6. fallback 队列尚未批量调用 `ProposalFinalStateGenerator`；
7. `advanceWavefront()` 仍是阶段 3 的 toy branching kernel。

因此本阶段证明的是“GPU 能可靠地选择下一离散物理记录”，而不是“GPU 已经
能够运行完整的电磁级联”。

## 11. 下一阶段

下一阶段应实现第一种真实 GPU 末态，并开始消除验证桥：

1. 定义 GPU process capability map；
2. 把 photonuclear、光致 muon pair 和其他 CPU-only 记录稳定路由到
   `ProposalFallbackEvent`；
3. 实现 photon pair-production 的
   \(\gamma\rightarrow e^-+e^+\) 末态 kernel；
4. 为每个选定记录计算次级数量，并用 CUB scan 分配稳定 history ID；
5. 检查逐事件能量守恒和四动量方向；
6. 把选择和末态 kernel 接到常驻双缓冲 SoA，避免 batch 内
   `cudaMalloc/cudaFree`；
7. 在这一物理路径通过后，再加入 bremsstrahlung 和连续 ionization。
