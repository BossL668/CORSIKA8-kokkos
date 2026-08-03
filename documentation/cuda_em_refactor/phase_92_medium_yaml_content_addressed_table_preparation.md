# 阶段 92：介质 YAML、规范化哈希与自动物理表准备

## 1. 目标

本阶段把“修改介质后如何获得正确的 CUDA 物理表”从手工流程变成一个独立、
可重复、并发安全的准备步骤：

```text
介质 YAML
  -> 严格解析和物理域检查
  -> 组分排序、分数归一化、规范化 YAML
  -> 介质 SHA-256
  -> 解析能区、cut、精度和生成器合同
  -> 查找兼容的已有表
       | 命中：返回已有表
       + 未命中：加文件锁 -> gpu_em_tablegen -> 回读校验
  -> 写入表、规范化介质和 manifest
  -> 返回可直接传给 c8_air_shower 的路径
```

独立程序是 `gpu_em_table_prepare`。它不会改变
`c8_air_shower` 的 fail-closed 语义：生产 shower 仍只读显式传入的
`.c8emrt`，不会在事件运行过程中临时制表。

## 2. 介质 YAML 合同

仓库内标准干空气模板为
[`configs/media/air_dry_1_atm.yaml`](../../configs/media/air_dry_1_atm.yaml)：

```yaml
schema_version: 1
name: air_dry_1_atm

proposal:
  mean_excitation_energy_eV: 85.7
  density_correction_C: -10.5961
  density_correction_a: 0.10914
  density_correction_m: 3.3994
  density_correction_x0: 1.7418
  density_correction_x1: 4.2759
  density_correction_delta0: 0.0
  reference_mass_density_g_per_cm3: 0.0012048

components:
  - name: nucleus
    corsika_pid: 1000070140
    nuclear_charge: 7
    atomic_mass_g_per_mol: 14
    number_fraction: 0.78479
```

实际模板还包含氧和氩。字段约束如下：

- `density_correction_C` 是直接传给 PROPOSAL 的带符号 \(C\)，不是正的
  CORSIKA `Cbar`；
- `number_fraction` 是靶核数目分数，不是质量分数，总和必须在
  \(10^{-8}\) 内等于 1；
- `corsika_pid` 是写入表 metadata 的 CORSIKA/PDG 核编码；
- 元素电荷必须是 1–118 的整数，原子质量必须不小于电荷；
- 所有字段必填，未知键会被拒绝，避免拼写错误被静默忽略；
- schema 1 最多允许 64 个不重复 PID 的组分。

YAML 只描述生成 PROPOSAL 表所需的材料常数和组成。高度—密度 profile、磁场、
观测面及三维几何不进入介质哈希；只改变这些运行时量不要求重新制表。

## 3. 规范化与介质哈希

`MediumConfig` 在哈希前执行：

1. 检查 schema、必填键、有限数和物理取值域；
2. 检查 PID 唯一性和数目分数总和；
3. 用总和重新归一化数目分数，消除无害的十进制舍入；
4. 按 PID、核电荷、原子质量和名称稳定排序组分；
5. 用 `max_digits10` 科学计数法输出所有浮点数；
6. 对完整规范化 UTF-8 文本计算 SHA-256。

因此，改变注释、空白、YAML 键顺序或组分输入顺序不会产生新介质身份；改变任一
实际材料参数会产生新哈希。标准干空气 schema 1 的固定哈希是：

```text
054323b00f3ae72fab27ca4a46ee510c5578e65f63fc0eaff41cf93251a857fa
```

这个哈希是本工具的内容地址，不替代 PROPOSAL 自己的 medium hash。表生成和
加载时两者都会检查。

## 4. 请求身份与兼容表查找

请求 SHA-256 包含：

- 介质规范化哈希；
- PROPOSAL 版本、`.c8emrt` 格式和生成器合同版本；
- 能区、EM cut、\(e^\pm\)/\(\mu^\pm\) transport cut；
- PROPOSAL 相对随机损失阈值 `v_cut=0.01`；
- rate 和 inverse-CDF 容差；
- 全部自适应网格上限和独立验证样本数；
- 是否包含 \(\mu^\pm\) 和实验性 Epair rho 表。

工具首先扫描同一介质目录，并通过 `validateCompatibility()` 回读检查候选表的
内容哈希、PROPOSAL identity、组分、cut、能区、误差和粒子列。更宽能区或更严
精度的表可以作为当前请求的兼容超集复用；有多个候选时选择满足要求的最小能区
表。损坏或不兼容候选只会给出原因，不会被使用。

缓存布局为：

```text
CACHE/
  media/<medium-sha256>.yaml
  proposal/<medium-sha256>/*.dat
  tables/<medium-sha256>/
    <request-sha256>.lock
    table-<request-sha256>.c8emrt
    table-<request-sha256>.c8emrt.manifest.yaml
```

manifest 保存源 YAML、两级哈希、版本、能区、cut、容差、表内容哈希和生成器
实测误差。相同请求由 POSIX `flock` 串行化；等待者在获得锁后重新查找缓存，
不会重复制表。表、manifest 和规范化 YAML 均通过临时文件原子发布。

## 5. \(10^{19}\) eV 质子实例

已有 Release 表只到 \(10^{18}\) eV，不能用于 \(10^{19}\) eV 初级。先检查将要
生成的请求，不写任何文件：

```bash
export C8_SOURCE=/path/to/corsika8_gpu_refactor
export C8_BUILD=/path/to/corsika8_gpu_refactor_build_cuda
export C8_TABLE_CACHE=/path/to/c8_gpu_table_cache

"$C8_BUILD/applications/gpu_em_table_prepare" \
  --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
  --cache-dir "$C8_TABLE_CACHE" \
  --primary-energy-eV 1e19 \
  --dry-run
```

默认安全系数为 1.05，所以输出应包含：

```text
energy_max_MeV=10500000000000
```

正式生成并取得路径：

```bash
export C8_TABLE="$(
  "$C8_BUILD/applications/gpu_em_table_prepare" \
    --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e19 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 1e-3 \
    --loss-tolerance 1e-3 \
    --print-path-only
)"
```

`--print-path-only` 把生成器进度写到标准错误，标准输出只保留最终表路径，因而
可以安全用于 shell 命令替换。第一次运行可能需要较长时间和大量 PROPOSAL
计算；后续相同或被该表覆盖的请求直接命中缓存。

在 shower 中，\(10^{19}\) eV 等于 \(10^{10}\) GeV：

```bash
"$C8_BUILD/applications/c8_air_shower" \
  -p 2212 \
  -E 1e10 \
  -N 1 \
  -f /path/to/output \
  --emcut 0.0005 \
  --mucut 0.3 \
  --em-backend cuda \
  --gpu-table-cache "$C8_TABLE"
```

新表通过数值生成门禁不等于 \(10^{19}\) eV shower 已获得 production 物理验收
身份。正式科研数据仍需执行 CPU/CUDA 系综、能量闭合、纵向 profile、地面粒子
和射电统计验收。

## 6. 查找、自动化与失败策略

只允许查找、不允许在计算节点制表：

```bash
gpu_em_table_prepare \
  --medium-yaml medium.yaml \
  --primary-energy-eV 1e19 \
  --lookup-only \
  --print-path-only
```

命中返回 0；未命中返回 2。`--force` 只重建当前内容寻址请求，不能与
`--lookup-only` 或 `--dry-run` 同用。以下情况返回失败而不是猜测：

- YAML 缺字段、含未知字段、物理域非法或组分分数错误；
- 请求上限高于当前验证合同 \(10^{14}\) MeV，即 \(10^{20}\) eV；
- PROPOSAL/格式/介质/cut/能区/精度或 PID 列不兼容；
- 锁等待超时、生成器异常退出、输出损坏或回读校验失败。

## 7. 当前能力边界

`gpu_em_tablegen --medium-yaml` 和 `gpu_em_table_prepare` 已能为 schema 1 的
自定义材料建立 PROPOSAL 介质和 CUDA 表。当前
`c8_air_shower` 的五层 `EnvironmentSnapshot` 仍是标准干空气。山体、土壤、
月壤或冰的端到端模拟还必须加入：

- 对应材料的 CORSIKA environment node 和运行时 medium ID；
- 边界/几何及 grammage 积分；
- snapshot 到表中介质身份的一一映射；
- 新介质上的 CPU/GPU 物理与射电验收。

所以本阶段解决的是“新介质表怎样规范、自动、可追溯地准备”，没有把任意介质
几何自动接入现有空气 shower。

## 8. 验证证据

本阶段完成：

- 标准干空气 YAML 与历史硬编码 PROPOSAL 合同逐字段一致；
- 固定介质 SHA、组分重排不变性和物理参数变化敏感性测试；
- 未知 YAML 键和错误分数总和拒绝测试；
- PROPOSAL medium 构造及表组分 PID 映射测试；
- 一张 0.5–20,000 MeV、EM-only、宽松 \(0.2\) 容差的真实端到端冒烟表；
- 同一请求的缓存命中；
- \(10^{19}\) eV 请求解析和 path-only 输出；
- 旧标准干空气 production 表的兼容查找。

冒烟表只用于证明制表流程，不是 production 物理表。

## 9. 主要实现位置

| 文件 | 职责 |
|---|---|
| `corsika/gpu/em/tables/MediumConfig.hpp` | schema、规范化和哈希接口 |
| `src/gpu/em/tables/MediumConfig.cpp` | 严格 YAML 解析与规范序列化 |
| `corsika/gpu/em/tables/ProposalMedium.hpp` | PROPOSAL medium 适配接口 |
| `src/gpu/em/tables/ProposalMedium.cpp` | YAML 到 PROPOSAL 和表 metadata |
| `applications/gpu_em_table_prepare.cpp` | 查找、锁、生成、校验和 manifest |
| `applications/gpu_em_tablegen.cpp` | `--medium-yaml` 通用制表 |
| `tests/gpu/testGpuMediumConfig.cpp` | 规范化与兼容合同回归 |
