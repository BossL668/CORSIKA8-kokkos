# CUDA 电磁后端重构记录：阶段 7，随机能损逆 CDF 表

> 后续进展：本文件结尾提出的 host-to-device SoA 上传和设备查询已在
> [阶段 8](phase_08_cuda_rate_table_upload.md) 完成。本文件的“当前仍未完成”
> 保留为阶段 7 结束时的历史状态。

## 1. 本阶段完成了什么

阶段 6 只能给出每个
`projectile × process × target component` 的相互作用率 \(dN/dX(E)\)。
这足以抽取“发生哪一种过程”，但还不能抽取该过程损失的能量比例
\(v=\Delta E/E\)。

本阶段增加条件逆累计分布：

\[
v = F^{-1}_{p,c}(E,u),\qquad u\in(0,1),
\]

其中 \(p\) 是已选定的过程，\(c\) 是已选定的介质组分。运行时的完整离散抽样
因而被拆成：

```text
rate table dN/dX(E)
  │
  ├─ 抽取 process
  ├─ 抽取 target component
  └─ 得到该 column 的条件随机数 u
        │
        ▼
ragged inverse-CDF table v(E,u)
        │
        ├─ 行内：logit(u) / log(v) 插值
        └─ 行间：log(E) / log(v) 插值
```

本阶段仍然是主机端物理数据基础设施；CUDA kernel 尚未读取表格，也没有生成
最终的电磁次级粒子。

## 2. PROPOSAL 的实际抽样语义

PROPOSAL 7.6.2 的 `Interaction::SampleLoss()` 先用所有 rate 的总和抽取一条
`Interaction::Rate`，随后调用：

```cpp
crosssection->CalculateStochasticLoss(
    component_hash, energy, conditional_rate);
```

这里的 `conditional_rate` 在所选 column 的
\([0,dN/dX_{p,c}(E))\) 内均匀。因此表格生成器计算：

```cpp
v = crosssection->CalculateStochasticLoss(
    component_hash, E, u * reference_column_rate);
```

而不是再次调用 `Interaction::SampleLoss()`。这样过程和 target 只抽取一次，
也与阶段 5 的 `ProposalInteractionRecord` / 指定末态接口一致。

## 3. 为什么不是规则二维网格

最初尝试让所有能量共享同一个 quantile 网格：

```text
energy[E_index] × quantile[u_index] -> v
```

该方法对 Compton、Brems 和 Ioniz 可用，但在电子/正电子 Epair 阈值附近失败。
Epair 的运动学结构会随能量移动；把每个能量行发现的节点合并成全局 quantile
网格，会使两个方向互相触发细化，调试范围内就需要约 2000 个能量点和 1500
个 quantile 点。

最终格式采用 ragged rows：

```cpp
struct InverseCdfTable {
  std::string reference_mode;
  std::vector<double> energies_MeV;
  std::vector<std::uint64_t> quantile_offsets;
  std::vector<double> quantiles;
  std::vector<double> v_loss;
};
```

能量行 \(i\) 的数据范围为：

```cpp
[quantile_offsets[i], quantile_offsets[i + 1])
```

所以每个能量节点只保存自身需要的 quantile 节点。该布局既能直接压平上传
GPU，也避免 `std::vector<std::vector<...>>` 的 host/device ABI 问题。

## 4. 坐标和插值

### 4.1 quantile 坐标

均匀 \(u\) 网格会把绝大多数节点浪费在分布中部，无法同时解析
\(u\rightarrow0\) 和 \(u\rightarrow1\) 的尾部。本阶段使用：

\[
z=\operatorname{logit}(u)=\log\frac{u}{1-u}.
\]

自适应初始节点在 \(z\) 上均匀分布；行内插值也在 \(z\) 上进行。
除表格能力边界外，新增节点会对齐到设备 Philox 的实际开区间输出：

\[
u_k=\frac{k+1/2}{2^{32}},\qquad
k\in\{0,\ldots,2^{32}-1\}.
\]

这避免为设备随机数永远无法产生的相邻实数无限细化。

### 4.2 loss 和能量坐标

若相邻 \(v\) 都大于零，对 \(\log v\) 线性插值；包含零时退回普通线性插值。
能量方向使用 \(\log E\)。查询一共执行三步：

1. 在下能量行按 `logit(u)` 求 \(v_\mathrm{lower}\)；
2. 在上能量行按 `logit(u)` 求 \(v_\mathrm{upper}\)；
3. 沿 `log(E)` 在两者之间插值。

主机接口为：

```cpp
interpolateLossFraction(
    particle, process_id, component_hash, energy_MeV, quantile);
```

energy 或 quantile 超出该 column 的已存能力域时抛出 `std::out_of_range`。
未来 `HybridCascade` 必须把这类事件变成有统计计数的 CPU fallback，不能夹紧
到端点继续运行。

## 5. 两级自适应生成

### 5.1 每个能量行的 quantile 细化

对每个固定能量：

1. 在 logit 坐标建立初始 quantile 网格；
2. 对每个区间的 1/4、1/2、3/4 位置调用 PROPOSAL；
3. 与运行时相同的 `logit(u)` / `log(v)` 插值比较；
4. 超过内部容差的 probe 加入该行；
5. 重复直至通过或超过 `--loss-max-quantile-points`。

### 5.2 能量细化

建立若干完整 ragged rows 后，对每对相邻能量：

1. 合并上下两行的 quantile 节点；
2. 在这些节点的 logit 中点增加验证 quantile；
3. 在能量区间的 1/4、1/2、3/4 log 位置直接调用 PROPOSAL；
4. 与“两次行内插值 + 一次能量插值”的结果比较；
5. 任一 quantile 超差就插入该能量并为它单独生成一行。

用户请求容差为 \(\epsilon_\mathrm{requested}\) 时，内部自适应目标使用：

\[
\epsilon_\mathrm{refine}=0.75\epsilon_\mathrm{requested}.
\]

这不是改变验收标准，而是为没有参与网格细化的独立验证点保留裕量。

loss 的相对误差定义为：

\[
\epsilon_v =
\frac{|v_\mathrm{PROPOSAL}-v_\mathrm{table}|}
{\max(v_\mathrm{PROPOSAL},v_\mathrm{table},10^{-6})}.
\]

\(10^{-6}\) 只在 \(v\) 本身趋近零时防止无意义的无限相对误差；正常离散能损
仍使用严格相对误差。

## 6. 独立验证

自适应 probe 通过不等于表格已经被独立验证。生成器另用两组无理数步长构造
确定性的低差异样本：

```text
energy fraction   ~ multiples of 0.6180339887...
quantile fraction ~ multiples of 0.4142135623...
```

这些点没有参加建表。每个有效 column 默认复算 64 个点，可通过：

```text
--loss-validation-samples
```

调整。任一点超过用户要求的 `--loss-tolerance`，生成器在写文件前失败。
metadata 中的 `measured_max_loss_relative_error` 取自适应 probe 和独立样本
两者的最大值。

## 7. PROPOSAL 7.6.2 数值边界和显式 fallback

### 7.1 发现的问题

测试揭示了两个上游数值问题：

1. 插值截面的 `CalculateStochasticLoss()` 在低能 Epair 阈值附近可能让
   Newton 求根跳到错误分支；固定精度 bisection fallback 又会产生假台阶。
2. `interpolate=false` 的直接积分器在 Epair 刚过阈值、极小累计概率处会报告
   “Precision 1e-06 has not been reached”，并可产生明显非单调的 \(v(u)\)。

生成器不会把这些结果静默写入一张声称达到 \(10^{-3}\) 的表。

### 7.2 当前能力边界

当前按 column 保存以下可用域：

- 非 Epair：复现生产 CPU 的 `proposal_interpolated` 路径，覆盖完整 Philox
  开区间；其中 Brems 的可靠能量域从 1 MeV 开始，Ioniz 的 quantile
  上限为 \(1-10^{-5}\)；
- Epair：同样使用 `proposal_interpolated` 参考，但要求
  \(E\ge20\,\mathrm{MeV}\) 且
  \(u\in[10^{-4},0.98]\)；
- Brems 的 1 MeV 以下、Ioniz 最高 10 ppm 尾部、Epair 低能区和 Epair 两端
  quantile 尾必须显式 CPU fallback。

调试中还发现，`proposal_direct` 的 Epair 积分在数百 MeV、非极端 quantile
也可能不单调，因此直接积分不进入本阶段的生产表。底层截面工厂仍支持
`interpolate=false`，供后续诊断使用。

每个 inverse-CDF column 在 v3 格式中保存 `reference_mode`，reader 仅接受：

```text
proposal_interpolated
proposal_direct
```

quantile 和能量边界则直接保存在 column 内。未来 capability 检查不需要依赖
生成器硬编码猜测范围。

这个 fallback 是已知并可计数的能力边界，不是删除 Epair 物理过程。Epair 的
rate 在全能区仍然存在；GPU 选中过程后若状态不在 inverse-CDF 域内，应把
指定过程、component、随机数键和粒子状态交回 CPU。

## 8. 磁盘格式 v3 和校验

阶段 6 的 rate-only v1 格式已由本阶段替代。当前 envelope 为：

```text
magic          C8EMRT03
format version 3
endian marker  0x01020304
payload bytes
payload SHA-256
```

除阶段 6 的 rate metadata 外，payload 还保存：

- 请求和实测的 inverse-CDF 相对误差；
- 每个 column 的 `reference_mode`；
- inverse-CDF energy rows；
- ragged offsets；
- quantile 和 \(v\) 数组。

`validateRateTable()` 另外拒绝：

- 正 rate column 缺失 inverse CDF；
- 未注册的 reference mode；
- energy 行少于两个点或不递增；
- ragged offsets 数量、首尾或范围错误；
- 不同行使用不同 quantile 能力端点；
- quantile 越过 Philox 支持域或不递增；
- \(v\) 非有限、超出 \([0,1]\) 或不单调。

SHA-256 仍覆盖完整 payload，因此修改 reference mode、能力边界或任意表值都会
使缓存 hash 改变。

## 9. 构建和运行

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  --target gpu_em_tablegen testGpuEmRateTable -j2
```

正式命令：

```bash
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean/applications/gpu_em_tablegen \
  /path/to/gpu_em_air_cut0p5_v3.c8emrt \
  --proposal-cache /path/to/proposal_cache \
  --energy-min-MeV 0.6 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --initial-intervals 16 \
  --max-points 20000 \
  --loss-initial-energy-intervals 8 \
  --loss-initial-quantile-intervals 8 \
  --loss-max-energy-points 4096 \
  --loss-max-quantile-points 2048 \
  --loss-validation-samples 64
```

PROPOSAL 的重复积分 warning 在生成器中降为 `critical` 日志级别；真正的非法
值、非单调、点数上限和独立验证失败仍通过异常终止。

## 10. 当前验证结果

主机单元测试目前 32 项，覆盖：

- v3 序列化、读回和 SHA-256 损坏检测；
- ragged offsets 和 value count；
- log-energy、logit-quantile、log-loss 插值；
- column-specific quantile capability；
- 非法 reference mode；
- 非单调 \(v\)；
- inverse-CDF compatibility tolerance。

最终实际运行了 0.6–\(10^{12}\) MeV、dry air、0.5 MeV cut 和正式
\(10^{-3}\) 容差的完整 v3 构建：

| 粒子 | rate 网格点 | columns | 独立样本 | 最大 rate 误差 | 最大 inverse-CDF 误差 |
|---|---:|---:|---:|---:|---:|
| gamma | 1149 | 15 | 960 | \(9.981565\times10^{-4}\) | \(7.498586\times10^{-4}\) |
| electron | 822 | 10 | 640 | \(9.982391\times10^{-4}\) | \(7.499968\times10^{-4}\) |
| positron | 824 | 13 | 832 | \(9.982391\times10^{-4}\) | \(7.499968\times10^{-4}\) |

其他结果：

```text
输出文件                       /tmp/c8_gpu_em_full_strict_v3.c8emrt
输出大小                       约 15 MiB
payload SHA-256                570f9d6a2a3bccb3a33470cca71637ba
                               9f9a88c473cc0b86a5a56cb51307f4e1
完整文件 SHA-256               6fa9078db4a23e78bf854c47f2969e24
                               f511d01bb30a7d853c2fae5995f477d0
热 cache 单核运行时间          28.38 s
峰值 RSS                       98,772 KiB
```

payload hash 与整个文件 hash 不同是预期行为：envelope 还包含 magic、版本、
payload 长度和 payload hash 本身。

随后用相同配置、相同热 cache 生成第二个文件：

```text
/tmp/c8_gpu_em_full_strict_v3_repeat.c8emrt
```

两次输出通过 `cmp` 逐字节相同，完整文件 SHA-256 也均为
`6fa9078db4a23e78bf854c47f2969e24f511d01bb30a7d853c2fae5995f477d0`。
第二次运行耗时 29.12 s、峰值 RSS 98,480 KiB。因此当前 host 端表格生成在
同一 PROPOSAL 版本、配置和平台上具备确定性。

本阶段收尾时执行的回归结果为：

| 测试 | 结果 |
|---|---|
| `testGpuEmHost`（CPU/CUDA build） | 150 checks 通过 |
| `testGpuEmRateTable`（CPU/CUDA build） | 32 checks 通过 |
| `testGpuEmCppLink` | 通过 |
| `testGpuEmCuda` | 11 个 toy 粒子确定性 wavefront 通过 |
| `testGpuHybridRoute` | 11 checks，通过 5 个 EM step / 4 个 wavefront |
| `[ScalarCascadeStepper]` | 3 assertions 通过 |
| `[HybridCascade]` | 12 assertions 通过 |
| `[Cascade]` | 24 assertions 通过 |
| `ProposalInterface` | 67 assertions 通过 |

`ProposalInterface` 首次运行生成了 54 个未跟踪的 PROPOSAL cache 文件；测试
完成后仅清除了这 54 个新文件，`modules/data` 中原有的受版本控制表格未修改。

## 11. 当前仍未完成

- inverse-CDF 尚未压平上传 CUDA；
- 尚未在 GPU 上用 rate 抽取 process/component；
- 尚未生成 continuous \(dE/dX\)、range 和 inverse range；
- 尚未生成多重散射和 LPM 设备表；
- 尚未把 inverse-CDF 能区/quantile 越界接到
  `ProposalFallbackEvent` 统计；
- 尚未比较 \(10^6\) 次 GPU 与 CPU 的过程和 \(v\) 分布。

下一阶段应先实现 host-to-device 的只读 SoA view 和设备查询单元测试，再移植
第一个均匀介质 photon pair-production kernel。CPU fallback 路由必须在启用
Epair GPU 抽样前先完成。
