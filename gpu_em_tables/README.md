# GPU 物理表获取、自动准备与介质配置

此目录只保存说明和校验值，不把大型二进制 `.c8emrt` 提交进 Git 历史。

## 1. 两类缓存

```text
proposal/<medium-hash>/*.dat   PROPOSAL 生成器内部插值缓存
tables/<medium-hash>/*.c8emrt CUDA 后端的版本化物理表
```

二者不能互换。`c8_air_shower --gpu-table-cache` 必须指向一个 `.c8emrt` 文件。

## 2. 推荐入口：`gpu_em_table_prepare`

独立准备工具从介质 YAML 计算稳定哈希，先查找兼容表，未命中才调用
`gpu_em_tablegen`。它还负责并发锁、回读校验和 manifest：

```bash
export C8_SOURCE=/path/to/corsika8_gpu_refactor
export C8_BUILD=/path/to/corsika8_gpu_refactor_build_cuda
export C8_TABLE_CACHE=/path/to/c8_gpu_table_cache

export C8_TABLE="$(
  "$C8_BUILD/applications/gpu_em_table_prepare" \
    --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
    --cache-dir "$C8_TABLE_CACHE" \
    --primary-energy-eV 1e18 \
    --em-cut-MeV 0.5 \
    --electron-transport-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 1e-3 \
    --loss-tolerance 1e-3 \
    --print-path-only
)"
```

工具默认包含 \(\mu^-\) 和 \(\mu^+\) 表；只准备电磁表时使用 `--no-muons`。
`--primary-energy-eV` 默认乘 1.05 安全系数。也可用
`--energy-max-MeV` 直接给出表上限，两者必须且只能指定一个。

先查看请求而不写文件：

```bash
"$C8_BUILD/applications/gpu_em_table_prepare" \
  --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
  --cache-dir "$C8_TABLE_CACHE" \
  --primary-energy-eV 1e19 \
  --dry-run
```

只查找、禁止在 batch 节点制表：

```bash
"$C8_BUILD/applications/gpu_em_table_prepare" \
  --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
  --cache-dir "$C8_TABLE_CACHE" \
  --primary-energy-eV 1e19 \
  --lookup-only \
  --print-path-only
```

未命中时退出码为 2。完整 schema、缓存算法和失败策略见
[阶段 92 文档](../documentation/cuda_em_refactor/phase_92_medium_yaml_content_addressed_table_preparation.md)。

## 3. \(10^{19}\) eV 为什么需要新表

Release 中的标准表上限是 \(10^{12}\) MeV，即 \(10^{18}\) eV。表查询禁止在
能区外外推，因此 \(10^{19}\) eV 质子不能复用它。以下命令准备上限
\(1.05\times10^{13}\) MeV 的新表：

```bash
export C8_TABLE_1E19="$(
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

首次严格制表可能耗时很长；相同请求和兼容的较小请求会复用缓存。新表仍需经过
\(10^{19}\) eV CPU/CUDA shower、能量和射电验收，不能仅凭生成成功就标记为
production。

## 4. 介质 YAML

标准模板是
[`configs/media/air_dry_1_atm.yaml`](../configs/media/air_dry_1_atm.yaml)。
安装后位于 `share/corsika/media/air_dry_1_atm.yaml`。最小结构为：

```yaml
schema_version: 1
name: material_name
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
    number_fraction: 1.0
```

`density_correction_C` 是传给 PROPOSAL 的带符号值；`number_fraction` 是数目
分数，不是质量分数。工具拒绝缺失键、未知键、重复 PID、非有限数和错误的分数
总和。

规范化会排序组分、归一化分数并统一浮点表示。注释、空白和输入顺序不会改变
SHA-256，任何实际材料参数改变都会进入新缓存目录。

## 5. 已验证的标准干空气 Release 表

不需要更高能区或新介质时，优先下载已经验收的表：

```bash
mkdir -p gpu_em_tables

gh release download gpu-table-dry-air-v10 \
  --repo BossL668/corsika8-gpu-hybrid \
  --pattern 'production_v10_muons_1e-3_1EeV.c8emrt' \
  --dir gpu_em_tables

(cd gpu_em_tables && sha256sum --check SHA256SUMS)
```

合同为：

- 标准 `AirDry1Atm` 干空气；
- \(\gamma,e^-,e^+,\mu^-,\mu^+\)；
- EM stochastic/transport cut 0.5 MeV；
- muon transport cut 300 MeV；
- 最大总能量 \(10^{12}\) MeV，即 \(10^{18}\) eV；
- rate 和 inverse-CDF 最大容许误差 \(10^{-3}\)。

私有仓库协作者需先执行 `gh auth login`。

## 6. 直接使用底层生成器

调试或精确控制输出文件名时可以直接调用：

```bash
"$C8_BUILD/applications/gpu_em_tablegen" \
  /path/to/output.c8emrt \
  --medium-yaml "$C8_SOURCE/configs/media/air_dry_1_atm.yaml" \
  --proposal-cache /path/to/proposal-cache \
  --energy-min-MeV 0.5 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --transport-cut-MeV 0.5 \
  --muon-transport-cut-MeV 300 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --include-muons
```

省略 `--medium-yaml` 会保留历史标准干空气合同。直接生成器不负责内容寻址查找
和跨进程锁，日常准备流程应优先用 `gpu_em_table_prepare`。

## 7. 改变介质后的运行边界

制表工具已支持 schema 1 自定义介质；当前 `c8_air_shower` 的 GPU
`EnvironmentSnapshot` 仍固定为五层标准干空气。

- 只改变同一干空气的密度—高度 profile、磁场或观测面：可复用表；
- 改变材料组成或 PROPOSAL 参数：用新 YAML 生成并验收新表；
- 岩石、土壤、月壤和冰：制表可以先完成，但端到端 GPU 模拟还需实现对应
  environment node、边界/grammage、medium ID 映射及物理验收。

不要把为自定义材料生成的表直接传给仍使用标准干空气 snapshot 的应用。
