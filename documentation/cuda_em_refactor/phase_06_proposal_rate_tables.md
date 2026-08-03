# CUDA 电磁后端重构记录：阶段 6，版本化 PROPOSAL 相互作用率表

> 历史说明：本文件记录 rate-only v1 格式。阶段 7 已将磁盘格式升级为
> `C8EMRT03`，并加入 ragged inverse-CDF、参考模式和新的误差 metadata。
> 当前实现与命令应以
> `phase_07_inverse_cdf_tables.md` 为准。

## 1. 本阶段目标和边界

本阶段把阶段 5 暴露出的 `process × target component` 相互作用率第一次变成
独立、可校验的磁盘数据：

```text
PROPOSAL 7.6.2 cross-section configuration
  │
  ├─ gamma
  ├─ electron
  └─ positron
       │
       ▼
ProposalRateProvider::rates(E)
       │
       ▼
adaptive log-energy grid
       │
       ├─ process type
       ├─ target component hash
       ├─ parameterization name
       └─ dN/dX [cm²/g]
       │
       ▼
versioned little-endian payload + SHA-256
       │
       ▼
strict CPU reader and interpolation
```

这一阶段仍不是 GPU 电磁 shower：

- CUDA kernel 尚未读取这些率表；
- 尚未生成能损 \(v(E,u)\) 的逆 CDF；
- 尚未生成 continuous \(dE/dX\)、range、multiple scattering 或 LPM 表；
- 尚未在 GPU 上抽取过程或生成末态。

因此本阶段完成的是“设备物理数据的第一层”，不能用于科研 shower 或报告
GPU 加速比。

## 2. 新的构建目标

新增静态库：

```text
CORSIKA8GpuEmTables
CORSIKA8::GpuEmTables
```

实现位置：

```text
corsika/gpu/em/tables/Sha256.hpp
corsika/gpu/em/tables/RateTable.hpp
src/gpu/em/tables/Sha256.cpp
src/gpu/em/tables/RateTable.cpp
src/gpu/em/tables/CMakeLists.txt
```

该库是纯 C++17 主机库，不包含 CUDA header，也不链接 CUDA runtime。
因此：

```text
CORSIKA_ENABLE_CUDA=OFF
```

时仍能生成、读取和测试物理表；原 CPU 默认构建不会因为本阶段要求 `nvcc`。

## 3. 磁盘格式

### 3.1 固定 envelope

文件开头使用固定宽度、小端序字段：

| 字段 | 大小 | 含义 |
|---|---:|---|
| magic | 8 byte | `C8EMRT01` |
| format version | `uint32` | 当前为 1 |
| endian marker | `uint32` | `0x01020304` |
| payload size | `uint64` | 后续 payload 的精确字节数 |
| SHA-256 | 32 byte | 仅覆盖 payload |
| payload | variable | metadata、网格和 rate |

没有直接把 C++ struct 写入文件，因此格式不依赖：

- struct padding；
- host 对齐；
- `std::string` 或 `std::vector` 内部布局；
- host 字节序；
- CUDA toolkit。

reader 会拒绝错误 magic、未知版本、错误 endian marker、截断文件、尾随字节和
SHA-256 不匹配。字符串、vector 和总文件大小也有上限，损坏的 count 不能导致
无限内存分配。

### 3.2 metadata

payload 保存：

- PROPOSAL 版本；
- table generator 版本；
- medium 名称；
- absolute energy cut；
- relative \(v_\text{cut}\)；
- 能量范围；
- 请求的插值容差；
- 实测最大抽查误差；
- 固定单位 `MeV` 和 `cm2/g`；
- 每个介质组分的 CORSIKA code、PROPOSAL component hash、number fraction
  和名称。

每个粒子保存：

- PDG ID；
- 粒子名称；
- 严格递增的能量网格；
- 每个过程和靶组分的 rate column。

每个 column 保存：

- PROPOSAL `InteractionType` 的固定整数值；
- target component hash；
- 人类可读过程名；
- PROPOSAL parameterization 名；
- target 名；
- 与能量网格等长的 \(dN/dX\)。

光核和 photon-induced muon-pair 等未来必须走 CPU fallback 的稀有过程没有
删除。它们的 rate 也写入表，未来 GPU 才能正确选择并产生带指定过程的
`ProposalFallbackEvent`。

## 4. 内存不变量和严格兼容检查

`validateRateTable()` 在写文件前和读文件后都检查：

- metadata 字符串非空；
- cut、范围、容差和误差有限且合法；
- 介质 fraction 之和为 1；
- component hash、PDG 和 process/component key 不重复；
- 能量严格递增；
- 每个 rate column 与能量网格等长；
- rate 有限且非负；
- 每个粒子网格的首尾值与 metadata 范围一致。

仅通过结构校验还不代表缓存适合当前 shower。为此增加：

```cpp
validateCompatibility(table, requirements);
```

它另外要求以下项目匹配：

- PROPOSAL 版本；
- medium 名称和完整组分；
- absolute cut 和 relative \(v_\text{cut}\)；
- 表格范围必须覆盖请求范围；
- 请求容差和实测误差都不能大于运行所允许的误差。

所以旧版本、错误介质、错误 cut 或精度不足的缓存不能被静默使用。

## 5. CPU 插值规则

公开接口：

```cpp
findParticle(table, pdg_id);

interpolateRate(
    particle, process_id, component_hash, energy_MeV);

interpolateTotalRate(particle, energy_MeV);
```

能量坐标始终使用 \(\log E\)：

1. 区间两端 rate 都大于零时，对 \(\log(dN/dX)\) 线性插值；
2. 任一端为零时，对 rate 本身线性插值；
3. 精确落在网格点时返回原值；
4. 禁止超出网格外推；
5. 找不到 process/component 时立即抛出异常。

第二条专门处理物理阈值。对零取对数既没有定义，也容易把阈值以下的零 rate
变成伪小正数。

## 6. `gpu_em_tablegen`

实现文件：

```text
applications/gpu_em_tablegen.cpp
```

当前生成器固定使用 21CMA 五层大气采用的 dry-air 组成：

```text
N  0.78479
O  0.21052
Ar 0.00469
```

它复用 `ProposalProcessBase.hpp` 中现有 CORSIKA 8 过程配置，因此生成的列包括：

- photon：Photopair、Compton、Photoproduction、Photoeffect、PhotoMuPair；
- electron：Brems、Epair、离散 Ioniz、Photonuclear；
- positron：electron 的过程加 Annihilation。

过程是否按 component 拆列由 PROPOSAL 自身的 cross-section 定义决定；例如
非 component-wise 的 ionization 使用 medium hash，不能伪装成氮、氧、氩
三个独立 target。

### 6.1 自适应网格

每个粒子有独立能量网格：

1. 先建立均匀的 log-energy 初始区间；
2. 对每个区间的 1/4、1/2 和 3/4 log 位置重新调用
   `ProposalRateProvider::rates(E)`；
3. 用与运行时 reader 完全相同的插值规则计算预测值；
4. 对所有 process/component column 取最大相对误差；
5. 超过容差的 probe 插入网格；
6. 重复直到所有 probe 通过，或超过 `--max-points` 后失败。

相对误差定义为：

\[
\epsilon =
\frac{\lvert r_\text{PROPOSAL}-r_\text{interp}\rvert}
{\max(r_\text{PROPOSAL},r_\text{interp},10^{-300})}.
\]

这会对很小但非零的稀有过程同样执行相对误差约束，而不是只保证 total rate。

这里的“PROPOSAL 参考值”是当前 CPU 生产路径实际调用的
`Interaction::Rates(E)`。该调用内部可能使用 PROPOSAL 自己生成的 cross-section
插值缓存；本阶段验证的是“GPU 文件插值相对现有 CPU PROPOSAL 路径”的额外
误差。若要验证相对非插值积分器的总误差，还需在后续增加
`interpolate=false` 的慢速 oracle，不能把两个误差来源混为一谈。

### 6.2 缓存位置

PROPOSAL 自身第一次构造 cross-section 时仍需生成底层插值缓存。
`gpu_em_tablegen` 默认把它放在输出目录旁：

```text
proposal_rate_cache/
```

也可显式指定：

```bash
--proposal-cache /path/to/cache
```

它不会再默认向源码的 `modules/data/PROPOSAL` 写未跟踪文件。

## 7. 构建和运行

当前 Conan 依赖包为 `RelWithDebInfo` 配置生成了完整 imported-target 属性，
所以开发构建保持同一配置：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  --target gpu_em_tablegen testGpuEmRateTable -j2
```

正式 dry-air、0.5 MeV cut、\(10^{18}\) eV 上限的命令为：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean/applications/gpu_em_tablegen \
  /path/to/gpu_em_rates_air_cut0p5.c8emrt \
  --proposal-cache /path/to/proposal_rate_cache \
  --energy-min-MeV 0.6 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --tolerance 1e-3 \
  --initial-intervals 16 \
  --max-points 20000
```

第一次冷启动可能需要数分钟。之后复用 PROPOSAL cache，表格自适应采样会快
很多。输出已经存在时默认拒绝覆盖，只有显式传 `--overwrite` 才会替换。

## 8. 测试

新增：

```text
tests/gpu/testGpuEmRateTable.cpp
```

当前 21 项主机检查覆盖：

- SHA-256 `"abc"` 标准向量；
- binary write/read 往返和重编码 hash 稳定性；
- 人为翻转 payload 最后一个 bit 后必须拒绝；
- log-log rate 插值；
- 阈值零端点；
- total rate；
- 禁止外推和缺失 process；
- 非法介质 fraction；
- column 长度错误；
- 非单调能量网格；
- 正确 compatibility requirements；
- 错误 PROPOSAL 版本、cut、范围、容差和介质组分均拒绝。

运行结果：

```text
GPU EM rate-table validation passed: 21 checks
```

### 8.1 真实 PROPOSAL 端到端烟雾测试

使用独立 `/tmp` cache、dry air、0.5 MeV cut 和 \(10^{-3}\) 容差，对
1–1000 MeV 范围实际运行生成器。结果为：

| 粒子 | 网格点 | rate columns | 最大 probe 相对误差 |
|---|---:|---:|---:|
| gamma | 474 | 15 | \(9.909500\times10^{-4}\) |
| electron | 695 | 10 | \(9.972252\times10^{-4}\) |
| positron | 714 | 13 | \(9.972252\times10^{-4}\) |

输出大小约 201 KiB。payload SHA-256 为：

```text
3ae40c0480dd0994dcde64efd3656dac78f71d42550cc353d951f4ee717a2509
```

payload hash 与整个文件的 hash 不应相同，因为整个文件还包含 envelope 和
payload hash 字段本身。完整文件 SHA-256 为：

```text
7fd7f9ff153bf6db01d1775c7e7a8db1ba90ee631a43c9ae08ffa8773fc5955a
```

使用热 PROPOSAL cache 再运行一次，两个 `.c8emrt` 文件逐字节相同。这验证了
当前生成顺序、序列化和 hash 的重复性。

### 8.2 完整 UHE 能区生成

底层 PROPOSAL cache 热启动后，又实际运行了上一节的完整
0.6–\(10^{12}\) MeV 命令：

| 粒子 | 网格点 | rate columns | 最大 probe 相对误差 |
|---|---:|---:|---:|
| gamma | 1149 | 15 | \(9.981565\times10^{-4}\) |
| electron | 822 | 10 | \(9.982391\times10^{-4}\) |
| positron | 824 | 13 | \(9.982391\times10^{-4}\) |

完整文件约 308 KiB，没有接近 20000 点上限。payload SHA-256：

```text
864ec25e0b4c9342edf35a5e56fe5bb5181c0f6d85d0e405cc4418ec964836a2
```

完整文件 SHA-256：

```text
1bb0bf1cbc50b9824cf66c18aec8d803e3c3346342f8a74b3d9bc416b0a14df4
```

完整能区也重复生成一次并通过逐字节比较。当前验证文件位于：

```text
/tmp/c8_gpu_em_rates_air_cut0p5_full.c8emrt
```

它是本地验证产物，未加入 Git，也还不是可直接用于科研 shower 的完整物理
表集，因为 \(v\) inverse-CDF、continuous loss 和 scattering 仍未实现。

CPU 和 RTX 4060 CUDA 构建回归结果：

```text
CUDA=OFF
  testGpuEmHost       passed
  testGpuEmRateTable  passed

CUDA=ON
  testGpuEmRateTable  passed
  testGpuEmCppLink    passed
  testGpuHybridRoute  passed
  testGpuEmCuda       passed
```

一次额外的旧 `ProposalInterface` 冷缓存回归会向
`modules/data/PROPOSAL` 写测试表。确认其行为后终止该重复测试，并只清除了
Git 标记为未跟踪的 20 个新生成文件；数据子模块最终保持干净。阶段 5 已经
完成过该接口和完整 `testModules` 回归，本阶段没有修改它的生产路径。

## 9. 下一步

紧接本阶段的工作应按以下顺序继续：

1. 增加非插值 PROPOSAL 慢速 oracle，分别报告 PROPOSAL 内部插值误差与
   GPU 文件插值误差；
2. 生成并校验 \(v(E,u)\) inverse-CDF；
3. 生成 continuous \(dE/dX\)、range \(R(E)\) 和 inverse range；
4. 把 host `RateTableSet` 压平为 device SoA，上传常量 metadata 和 rate
   arrays；
5. 在均匀 dry air 中实现第一个 photon pair-production 过程；
6. 对 GPU 抽取的 process/component/\(v\) 与 CPU 分布做至少
   \(10^6\) 次单过程统计比较。

在第 2 项完成前，GPU 不能仅凭 rate 表生成正确离散能损；在第 3 项完成前，
也不能正确竞争连续步长。
