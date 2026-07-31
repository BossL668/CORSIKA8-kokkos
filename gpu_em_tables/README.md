# GPU 物理表获取与生成

此目录只保存表格说明和校验值，不把二进制 `.c8emrt` 提交进 Git 历史。

## 推荐方式：下载已验证的标准干空气表

```bash
mkdir -p gpu_em_tables

gh release download gpu-table-dry-air-v10 \
  --repo BossL668/corsika8-gpu-hybrid \
  --pattern 'production_v10_muons_1e-3_1EeV.c8emrt' \
  --dir gpu_em_tables

(cd gpu_em_tables && sha256sum --check SHA256SUMS)
```

该 Release 属于私有仓库。协作者需要先运行 `gh auth login`，或者从 GitHub
Release 页面手工下载表文件。

`production_v10_muons_1e-3_1EeV.c8emrt` 的合同为：

- CORSIKA `AirDry1Atm` 标准干空气；
- \(\gamma,e^-,e^+,\mu^-,\mu^+\)；
- EM stochastic/transport cut：0.5 MeV；
- muon transport cut：300 MeV；
- 最大总能量：\(10^{12}\) MeV，即 \(10^{18}\) eV；
- rate 和 inverse-CDF 最大容许误差：\(10^{-3}\)。

## 哪些内容会自动生成

运行 `gpu_em_tablegen` 时，PROPOSAL 会在 `--proposal-cache` 目录自动生成其自身
的插值缓存。`c8_air_shower` 不会自动生成 `.c8emrt`，必须通过
`--gpu-table-cache` 显式传入已有文件。

两者不能互换：

```text
proposal_rate_cache/   PROPOSAL 内部缓存目录
*.c8emrt               CUDA 后端的版本化物理表
```

## 自行生成标准干空气表

以下命令生成包含 EM 和 μ 子列的标准干空气表：

```bash
export C8_BUILD=/path/to/corsika8_gpu_refactor_build_cuda
export C8_TABLE_DIR="$C8_BUILD/gpu_em_tables"
mkdir -p "$C8_TABLE_DIR"

"$C8_BUILD/applications/gpu_em_tablegen" \
  "$C8_TABLE_DIR/dry_air_em_muons_0p5MeV_1EeV.c8emrt" \
  --proposal-cache "$C8_TABLE_DIR/proposal_rate_cache" \
  --energy-min-MeV 0.5 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --transport-cut-MeV 0.5 \
  --muon-transport-cut-MeV 300 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --include-muons
```

生成器会自适应细化 rate 和 inverse-CDF 网格、验证误差、写入内容哈希并立即
回读检查。输出已存在时默认拒绝覆盖；确认目标正确后才使用 `--overwrite`。

上述命令生成一张数值上自洽的新表，但不自动获得本项目 production 表的物理
验收身份。正式科研使用前仍需运行 `validation/gpu_em` 中的 CPU/CUDA ensemble、
能量闭合和射电验收。

## 改变介质

当前 `gpu_em_tablegen` 的介质工厂在
`applications/gpu_em_tablegen.cpp::makeDryAirMedium()` 中固定为
`AirDry1Atm`，`c8_air_shower` 的五层环境也固定使用同一介质。因此当前没有
`--medium` 或材料配置文件能够自动生成岩石、土壤、月壤或冰的表。

- 只改变干空气密度随高度的 profile、磁场或观测面：复用当前表；
- 改变元素组成、数密度比例或 PROPOSAL 材料参数：必须扩展介质工厂和
  `EnvironmentSnapshot`，生成新表并重新验收；
- 不要只修改 `c8_air_shower` 中的介质而继续传入干空气表。

在通用介质配置和运行时环境—表格身份校验实现之前，非干空气介质应视为尚未
支持，而不是让程序自动猜测。
