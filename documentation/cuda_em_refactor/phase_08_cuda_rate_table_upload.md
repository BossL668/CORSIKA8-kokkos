# CUDA 电磁后端重构记录：阶段 8，物理表 SoA 上传与设备查询

> 阶段 8 末尾列出的“离散过程选择 kernel”已经在阶段 9 完成。实现、随机数
> 设计和验证结果见
> [phase_09_interaction_selection.md](phase_09_interaction_selection.md)。

## 1. 本阶段目标

阶段 7 已经能够生成经过 PROPOSAL 独立验证的 v3 物理表，但表格仍由
`std::string`、`std::vector` 和多层 C++ 对象组成，CUDA kernel 不能直接安全
访问这些对象。

本阶段完成以下中间层：

```text
RateTableSet（有字符串、嵌套 vector）
        │
        │ validateRateTable + flattenRateTable
        ▼
FlatRateTable（host owning SoA）
        │
        │ CudaRateTable::initialize
        ▼
CUDA device arrays + FlatRateTableView
        │
        ├─ queryRate
        ├─ queryTotalRate
        └─ queryLossFraction
```

本阶段没有生成真实次级粒子，也没有替换当前 toy branching kernel。它提供的
是之后所有电磁过程 kernel 共用的只读物理数据层。

## 2. 为什么需要单独的扁平层

原始 `RateTableSet` 适合磁盘读写和人工检查，因为它保留：

- 粒子、过程、参数化和介质组分名称；
- 每个粒子独立的能量网格；
- 每个过程独立的 ragged inverse-CDF；
- 生成器版本、PROPOSAL 版本和误差 metadata。

但它不适合复制到设备：

1. `std::vector` 内含 host 指针；
2. `std::string` 的 ABI 不能在 host/device 间共享；
3. 每个 column 的小块独立分配会造成大量指针追踪；
4. kernel 无法使用异常表达“能区超界”。

因此新增：

```cpp
FlatRateTable flattenRateTable(RateTableSet const&);
FlatRateTableView makeFlatRateTableView(FlatRateTable const&);
```

`FlatRateTable` 拥有所有 host 数组，`FlatRateTableView` 只包含设备可复制的
指针、计数和绝对 offset。

## 3. SoA 布局

### 3.1 粒子 metadata

每个粒子使用五个并行数组：

```text
particle_pdg_ids
particle_energy_offsets
particle_energy_counts
particle_column_offsets
particle_column_counts
```

例如 photon 的 `particle_energy_offset=0`、`energy_count=1149`，表示它的
rate 能量网格位于：

```text
rate_energies_MeV[0 ... 1148]
```

### 3.2 column metadata

每个 `process × component` column 使用：

```text
column_process_ids
column_component_hashes
column_rate_offsets
column_inverse_energy_offsets
column_inverse_energy_counts
column_inverse_row_offsets
```

PROPOSAL component 使用 64 位 hash，不能缩成原占位结构中的 32 位
`component_id`。`ProposalFallbackEvent` 因此也改为保存
`std::uint64_t component_hash`。

### 3.3 数值数组

所有数值被连续保存：

```text
rate_energies_MeV
rates_cm2_per_g
inverse_energies_MeV
inverse_row_offsets
inverse_quantiles
inverse_v_loss
```

`inverse_row_offsets` 已从每个 column 内的局部 offset 转换为
`inverse_quantiles` / `inverse_v_loss` 的全局绝对 offset，所以设备端不需要
二次指针解引用。

所有 offset 使用无符号 32 位整数。扁平化时每次数量转换和 offset 加法都检查
溢出；超过 \(2^{32}-1\) 个元素的表格会在上传前失败。当前正式表远小于该限制。

## 4. 校验顺序

扁平化不是无条件复制，顺序为：

1. `validateRateTable(source)` 校验 v3 层级结构；
2. `calculateContentHash(source)` 保存源 payload 身份；
3. 构建所有 SoA 数组；
4. `validateFlatRateTable(flat)` 再检查扁平 offset；
5. 只有全部通过后才允许 CUDA 上传。

扁平校验覆盖：

- 粒子和 column 并行数组长度；
- 粒子能量和 column 范围；
- rate 数组范围；
- inverse-energy 和 row-offset 范围；
- 每行 quantile 严格递增；
- 同一 column 所有 ragged row 的 quantile 端点一致；
- \(v\) 有限、位于 \([0,1]\) 且在 \(10^{-14}\) 数值容差内单调。

最后一项与 v3 reader 的规则一致。完整生产表测试曾发现，如果扁平层采用
严格的 `v[i] >= v[i-1]`，会错误拒绝 PROPOSAL 求根器允许的
\(10^{-14}\) 量级抖动；当前两层已使用相同判断。

## 5. host/device 共用查询

设备查询函数定义在 `FlatRateTable.hpp` 中，并使用
`__host__ __device__`：

```cpp
TableQueryResult queryRate(...);
TableQueryResult queryTotalRate(...);
TableQueryResult queryLossFraction(...);
TableQueryResult executeTableQuery(...);
```

因此同一查询代码可以先在 host flat view 上运行，再在 CUDA view 上运行。

### 5.1 rate 查询

1. 按 PDG ID 找粒子；
2. 在粒子的 column 范围内按 `process_id × component_hash` 找 column；
3. 对粒子能量网格二分查找；
4. 两端 rate 都大于零时执行 `log(E) / log(rate)` 插值；
5. 含零阈值区间使用 `log(E) / linear(rate)`。

该规则与 `RateTable.cpp::interpolateRate()` 相同。

### 5.2 inverse-CDF 查询

1. 找到粒子和已选定 column；
2. 检查该 column 是否存在 inverse-CDF；
3. 检查 energy 和 quantile 是否在该 column 的能力域；
4. 在上下 energy row 内分别二分 quantile；
5. 行内执行 `logit(u) / log(v)` 插值；
6. 行间执行 `log(E) / log(v)` 插值。

编译仍未启用 `--use_fast_math`。CPU/GPU 不要求逐位相同，但测试要求相对/绝对
差异不超过 \(2\times10^{-13}\) 的 double 精度比较尺度。

## 6. 非抛异常状态和 CPU fallback

CUDA kernel 不能抛出 C++ 异常，因此查询返回：

```cpp
struct TableQueryResult {
  TableLookupStatus status;
  std::uint32_t reserved;
  double value;
};
```

状态包括：

```text
Success
ParticleNotFound
ColumnNotFound
RateEnergyOutOfRange
InverseCdfUnavailable
LossEnergyOutOfRange
LossQuantileOutOfRange
NonFiniteInput
InvalidQueryKind
InvalidTableView
```

任何非 `Success` 状态都不会把 energy 或 quantile 夹紧到端点。

新增 `ProposalFallbackReason` 和 `makeTableFallbackEvent()`，把查询状态转换为
稳定的 fallback 原因，并保留：

- 完整 `EmParticleState`；
- 已选定的 process ID；
- 64 位 component hash；
- random draw ID。

这保证未来 CPU PROPOSAL 可以继续生成“已经选定的指定末态”，而不是重新抽取
过程。目前 fallback schema 和映射已经实现并测试；wavefront kernel 尚未批量
输出这些事件。

## 7. CUDA 所有权

新增 RAII 类：

```cpp
class CudaRateTable {
 public:
  void initialize(RateTableSet const&, int device,
                  std::size_t maximum_device_bytes);
  FlatRateTableView deviceView() const;
  std::size_t deviceBytes() const;
  Sha256Digest const& sourceContentHash() const;
  void reset() noexcept;
};
```

初始化时：

1. 构建并校验 host flat table；
2. 检查用户给定的显存上限；
3. 检查目标 CUDA 设备和当前空闲显存；
4. 为每个只读 SoA 数组分配 device memory；
5. 上传全部数组；
6. 保存源 v3 payload SHA-256；
7. 仅在所有步骤成功后标记 initialized。

任何分配或复制失败都会释放此前已经分配的数组。`reset()` 和析构函数也会在
正确设备上释放内存。

`queryForValidation()` 会临时分配 query/result buffer，它只用于测试。
生产 wavefront 必须直接把 `deviceView()` 传入已有 kernel，不能在每个 batch
重复分配这些 buffer。

## 8. 测试

### 8.1 小型 ragged fixture

fixture 包含：

- 一个 photon 粒子；
- 三个 column；
- 正 rate、含零阈值 rate 和全零 rate；
- 三行不同长度的 ragged inverse-CDF；
- column-specific quantile 能力域；
- 无 inverse-CDF 的零 rate column。

CPU flat 测试结果：

```text
testGpuEmFlatRateTable
102 checks passed
```

覆盖层级表到 flat table 的数值一致性、offset、hash、损坏结构拒绝和所有
fallback 状态。

CUDA fixture 结果：

```text
1291 device queries
6450 checks passed
492 device bytes
```

同一 query batch 连续运行两次，CUDA 输出逐项确定。

### 8.2 完整生产表

测试文件：

```text
/tmp/c8_gpu_em_full_strict_v3.c8emrt
```

其物理配置为阶段 7 的：

```text
dry air
0.5 MeV cut
0.6 -- 1e12 MeV
1e-3 rate/loss tolerance
gamma, electron, positron
```

完整验证同时比较：

```text
层级 RateTableSet CPU 查询
        ↕
FlatRateTable host 查询
        ↕
CudaRateTable device 查询
```

结果：

```text
2528 device queries
12649 checks passed
15,118,340 device bytes
elapsed 0.70 s
peak host RSS 172,780 KiB
```

测试机器：

```text
GPU       NVIDIA GeForce RTX 4060 Laptop GPU
显存      8188 MiB
driver    560.94
CUDA      12.6 / nvcc 12.6.85
```

这里的 0.70 s 包含读取约 15 MiB 文件、两次 host 结构校验、扁平化、CUDA
上传、两轮 device query 和结果回传，不代表未来 wavefront 的单次 kernel
耗时。

### 8.3 原有回归

| 测试 | 结果 |
|---|---|
| CPU `testGpuEmHost` | 150 checks 通过 |
| CPU/CUDA `testGpuEmRateTable` | 32 checks 通过 |
| CPU/CUDA `testGpuEmFlatRateTable` | 102 checks 通过 |
| `[ScalarCascadeStepper]` | 3 assertions 通过 |
| `[HybridCascade]` | 12 assertions 通过 |
| `testGpuEmCppLink` | 通过 |
| `testGpuEmCuda` | 11 个确定性 toy 粒子 |
| `testGpuHybridRoute` | 11 checks 通过 |

CPU 的三个 GPU 基础设施 CTest 和 CUDA 的七个相关 CTest 均为 100% 通过。

## 9. 构建和复现

CPU：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO \
  cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  --target testGpuEmFlatRateTable -j2

/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean/tests/gpu/testGpuEmFlatRateTable
```

CUDA：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO \
  cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuEmRateTableCuda -j2

/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/gpu/testGpuEmRateTableCuda

/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/gpu/testGpuEmRateTableCuda \
  /tmp/c8_gpu_em_full_strict_v3.c8emrt
```

## 10. 当前边界和下一阶段

本阶段仍未完成：

- `CudaEmBackend::initialize()` 尚未从 `GpuEmConfig::table_cache` 读取并持有
  `CudaRateTable`；
- GPU 尚未根据所有 column rate 抽取 process/component；
- fallback event 尚未经过 scan/compaction 批量输出；
- 尚未调用 `ProposalFinalStateGenerator` 消费 GPU fallback；
- 尚未生成 photon pair-production 的两个次级粒子。

下一阶段应先实现“离散过程选择 kernel”：

1. `CudaEmBackend` 在 shower 开始时读取、兼容性校验并上传 v3 表；
2. 用 total \(dN/dX\) 抽取相互作用距离；
3. 用 Philox key 抽取 process/component；
4. 对已选 column 查询 \(v(E,u)\)；
5. 成功记录 `ProposalInteractionRecord`；
6. 越界事件通过 CUB scan 稳定写入 `ProposalFallbackEvent`；
7. 在该选择层通过后，再实现 photon pair-production 末态 kernel。
