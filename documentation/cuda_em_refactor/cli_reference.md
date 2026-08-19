# CORSIKA 8 CUDA/FLUKA 命令行参数参考

本文记录本研究分支新增的生产参数、辅助程序参数和主要验收驱动参数。参数名称、
默认值和约束以当前源码及当前构建的 `--help` 输出为准。

相关文档：

- [项目中文说明](../../README_CN.md)
- [CUDA 后端生产使用指南](cuda_em_backend_user_guide.md)
- [验证工具说明](../../validation/gpu_em/README.md)
- [重构阶段索引](README.md)

## 1. 使用约定

- 未指定 `--em-backend` 时，`c8_air_shower` 使用原标量
  `Cascade + PROPOSAL` 路径。
- `--em-backend cuda` 只在 `CORSIKA_ENABLE_CUDA=ON` 的构建中可用，并且
  必须显式提供兼容的 `.c8emrt` 表。
- 布尔 option 使用 `true` 或 `false`，例如
  `--gpu-deterministic true`。不带值的 flag 只需写参数名。
- 表格、设备、介质、输出或进程池检查失败时，程序终止当前 shower；不会静默
  切换到另一套物理。
- `--gpu-detailed-stage-timing`、`--gpu-full-step-records` 和
  `--gpu-radio-track-diagnostics` 会增加诊断开销，不应在正式性能计时中开启。
- 天线文件默认值带有本地环境路径。迁移服务器时应始终显式传入
  `--antenna-file /absolute/path/to/antennas.txt`。

查看实际构建支持的参数：

```bash
c8_air_shower --help
gpu_em_table_prepare --help
gpu_em_tablegen --help
cuda_decision_replay --help
fluka_batch_worker --help
```

## 2. `c8_air_shower` 新增参数

### 2.1 CUDA 电磁输运

| 参数 | 默认值 | 功能与使用边界 |
|---|---:|---|
| `--em-backend proposal\|cuda` | `proposal` | 选择原标量 PROPOSAL 或 HybridCascade CUDA 电磁输运。 |
| `--gpu-device INT` | `0` | CUDA device index。多 GPU 节点上每个进程应指定自己的 device。 |
| `--gpu-min-batch UINT` | `4096` | CUDA 执行的最小前沿规模。更小前沿先做有界标量展开；该值需要按新 GPU 实测。 |
| `--gpu-memory-fraction FLOAT` | `0.70` | 后端最多使用初始化时空闲显存的比例，范围为 0.01–1.0。 |
| `--gpu-table-cache PATH` | 空 | 版本化 `.c8emrt` 表。CUDA 后端必需；不是 PROPOSAL 自身 cache 目录。 |
| `--gpu-table-tolerance FLOAT` | `1e-3` | 可接受的最大表格相对误差。表 metadata 超过此值会拒绝启动。 |
| `--gpu-deterministic BOOL` | `true` | 启用按 history/step/process 寻址的 Philox 随机数。 |
| `--gpu-detailed-stage-timing` | 关闭 | 记录融合 lepton pipeline、真实 device-copy 及 host-wait 分解；会增加 event/synchronization 开销，仅用于 profiler。 |
| `--gpu-full-step-records` | 关闭 | 返回完整 GPU transport records，而不是紧凑 profile 投影，仅用于验证和调试。 |
| `--gpu-resident-cross-species BOOL` | `true` | 让 photon→lepton 和 lepton→photon 次级粒子保留在常驻 device 队列。 |
| `--cuda-replay-trace PATH` | 空 | 写出过程级 CSV trace，用于标量/CUDA 过程序列诊断。 |
| `--cuda-replay-tape-out PATH` | 空 | 记录标量 transport segments 和 radio observer snapshot，供离线严格 replay。 |

`--gpu-min-batch` 不是越小越快。过小会增加 kernel launch、同步和小批次尾部
开销；过大则会让更多前沿留在 CPU。推荐在目标 GPU 上扫描
`64, 256, 1024, 4096, 8192`，以相同物理表和热缓存的五次中位数选择。

启用详细计时后，`gpu_em/summary.yaml` 使用 timing schema 2，并新增：

- `transfer_timing.device_copy_time_ms`：紧贴 copy 前后的 CUDA event 时间；
- `transfer_timing.host_api_time_ms`：主机调用 CUDA copy API 的 wall time；
- `transfer_timing.host_wait_upper_bound_ms`：两者的非负差，包含同步等待和少量 API/计时开销；
- `synchronization_timing.physical_pipeline_wait_time_ms`：主线程等待物理 pipeline 完成的时间；
- `synchronization_timing.profile_input_wait_time_ms`：复用 profile 输入槽前的等待；
- `radio.input_slot_host_wait_time_ms`：复用 radio 输入槽前的等待。

旧 `transfer_time_ms` 为兼容既有分析脚本而保留，语义仍是选定同步 copy 外层的
host wall time，不能解释为纯 PCIe 时间。`kernel_time_ms` 是多个 CUDA stream 的
event duration 之和；profile、radio 与物理 pipeline 可以重叠，因而它也不能直接
与 shower wall time相加比较。

### 2.2 射电计算

| 参数 | 默认值 | 功能与使用边界 |
|---|---:|---|
| `--radio-backend cpu\|cuda` | `cpu` | CUDA EM 轨迹使用 CPU 或 resident CUDA CoREAS/ZHS 投影。 |
| `--gpu-radio-field-limit FLOAT` | `1.0` V/m | 确定性 fixed-point 波形累加器的检查范围；溢出会终止，不会环绕。 |
| `--gpu-radio-track-diagnostics` | 关闭 | 收集与标量端一致的 \(e^\pm\) 轨迹统计，增加 GPU reduction 开销。 |
| `--radio-sampling-rate-ghz FLOAT` | `1.0` | 时域采样率。50–350 MHz 精确波形比较建议至少 10 GHz。 |
| `--radio-window-duration-ns FLOAT` | `400` ns | 每个 observer 的时间窗长度。 |
| `--radio-pretrigger-ns FLOAT` | `10` ns | 时间窗相对几何直达时间提前开始的长度。 |
| `--antenna-file PATH` | 本地 21CMA 路径 | NWU 坐标文件，三列依次为 North、West、Up，单位 m。 |
| `--ring INT` | `0` | 原应用的星形同心环 observer 数；0 表示只使用天线文件。 |

CPU 和 CUDA radio 都随 \(e^\pm\) 轨迹段在线累计，不是在 shower 完成后重新读取
整棵粒子树。`--radio-backend cuda` 不改变强子或电磁相互作用模型，只改变射电
投影和波形累加所在设备。

### 2.3 强子调度和 FLUKA 进程池

| 参数 | 默认值 | 功能与使用边界 |
|---|---:|---|
| `--hadronic-plan-workers INT` | `4` | 回顾性 batch oracle 的假定 worker 数；只写计划，不启用并行。 |
| `--hadronic-plan-target-ms FLOAT` | `5` ms | 回顾性 homogeneous batch 的目标实测末态耗时。 |
| `--hadronic-plan-max-batch UINT` | `256` | 回顾性 batch 的最大 interaction 数。 |
| `--hadronic-backend scalar\|fluka-process` | `scalar` | 选择进程内标量 FLUKA 或隔离的持久 FLUKA worker pool。 |
| `--hadronic-workers INT` | `4` | 持久 FLUKA worker 进程数，范围 1–256。 |
| `--hadronic-min-batch UINT` | `64` | 执行预计代价 flush 检查前至少停放的 FLUKA vertices。 |
| `--hadronic-target-batch-ms FLOAT` | `5` ms | 每个 homogeneous worker batch 的在线预计计算时间。 |
| `--hadronic-max-batch UINT` | `256` | 单次 binary IPC batch 的最大请求数。 |
| `--hadronic-initial-cost-ms FLOAT` | `0.1` ms | 某个工作类别尚无计时样本时的单 interaction 初始代价。 |
| `--hadronic-worker-executable PATH` | 空 | 指定 `fluka_batch_worker`；空值时使用 `c8_air_shower` 同目录程序。 |
| `--cpu-detailed-step-timing` | 关闭 | 记录 CPU cross-section、tracking、continuous 和 discrete 阶段耗时。 |

`fluka-process` 仍使用相同 FLUKA 物理，并不是强子 CUDA kernel。加速来自持久
进程、工作类别分组和批量 IPC。启用条件：

1. 构建时 `WITH_FLUKA=ON`；
2. `FLUPRO` 指向含 `libflukahp.a` 的合法安装；
3. worker executable 与主程序版本一致；
4. 正式性能测试期间没有另一套 CPU shower 与 worker 抢占核心。

### 2.4 与原参数的关键组合

CUDA/CPU 验收时还必须保持以下原参数一致：

| 原参数 | 作用 |
|---|---|
| `-p/--pdg` 或 `-Z/-A` | 初级粒子或核素。两种写法互斥。 |
| `-E/--energy` | 初级总能量，单位 GeV。 |
| `--emcut` | photon、electron、positron 的 kinetic-energy cut，单位 GeV。 |
| `--hadcut`, `--mucut`, `--taucut` | 非 EM 粒子 cut，单位 GeV。 |
| `--emthin` | EM thinning 的初级能量比例。 |
| `--max-weight` | thinning 最大权重；0 使用应用的自动值。 |
| `--seed` | 初始随机种子。相同 seed 保证各自后端可复现，不保证两种调度逐事例相同。 |
| `--zenith`, `--azimuth` | 初级方向，单位 degree。 |
| `--geomagnetic-model IGRF13\|IGRF14` | 地磁系数文件版本；本分支默认 `IGRF14`。文件从已安装的 `GeoMag/` 数据目录读取。 |
| `--geomagnetic-year` | IGRF 计算年份；默认 `2027`，必须位于 1900–2030。模型和年份都会写入 GPU 输出 metadata。 |
| `--observation-level`, `--injection-height` | 局部水平观测面的高度和球形环境的注入高度，单位 m。 |
| `--shower-core-x`, `--shower-core-y` | NWU 平面中的 shower core，单位 m。 |
| `--force-interaction` | 强制调度器下一次取出的初级粒子在注入位置立即相互作用。CUDA 模式先执行一个标量顶点，再正常路由次级。 |
| `--force-decay` | 强制调度器下一次取出的初级粒子在注入位置立即衰变。与 `--force-interaction` 互斥。 |

当 `--max-weight` 省略或为 0 时：

```text
maxWeight = 0.5 * emthin * E_primary[GeV]
```

如果自动值小于初始 history weight 1，thinning 不会从未加权 history 开始。
CPU/CUDA 质量比较应显式使用同一个 `--max-weight`，或者明确记录两端都使用自动
值。

## 3. `gpu_em_table_prepare`

这是日常使用的独立物理表准备入口。它执行“介质 YAML → 规范化哈希 → 兼容表
查找 → 缓存未命中时加锁生成 → 回读校验 → manifest”，最后返回 `.c8emrt`
路径。它不在 shower 运行期间调用。

### 3.1 输入、能区和物理合同

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--medium-yaml PATH` | 必填 | schema 1 材料文件；未知键和非法物理值会被拒绝。 |
| `--cache-dir PATH` | `$XDG_CACHE_HOME/corsika8/gpu_em_tables` 或 `$HOME/.cache/corsika8/gpu_em_tables` | 内容寻址缓存根目录。 |
| `--tablegen PATH` | 同目录的 `gpu_em_tablegen`，否则从 `PATH` 查找 | 指定底层生成器。 |
| `--primary-energy-eV FLOAT` | 与下项二选一 | 初级总能量；表上限为该值乘安全系数。 |
| `--energy-max-MeV FLOAT` | 与上项二选一 | 直接指定表的总能量上限。 |
| `--energy-margin FLOAT` | `1.05` | `--primary-energy-eV` 的上限安全系数，范围 1–10。 |
| `--energy-min-MeV FLOAT` | `0` | 0 表示使用 EM cut。 |
| `--em-cut-MeV FLOAT` | `0.5` | PROPOSAL absolute stochastic cut。 |
| `--electron-transport-cut-MeV FLOAT` | `0` | 0 表示使用 EM cut。 |
| `--muon-transport-cut-MeV FLOAT` | `300` | \(\mu^\pm\) kinetic transport cut。 |
| `--tolerance FLOAT` | `1e-3` | 最大 rate 插值误差。 |
| `--loss-tolerance FLOAT` | `1e-3` | 最大 inverse-CDF 插值误差。 |
| `--no-muons` | 关闭 | 只生成/接受 \(\gamma,e^-,e^+\) 表；默认要求成对 \(\mu^\pm\) 列。 |
| `--enable-epair-rho-table` | 关闭 | 要求实验性 dense Epair rho 表。 |

必须且只能指定 `--primary-energy-eV` 或 `--energy-max-MeV`。当前生成器验证合同的
硬上限是 \(10^{14}\) MeV，即 \(10^{20}\) eV；更高请求直接失败。

### 3.2 缓存、并发和自动化

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--lookup-only` | 关闭 | 只查找；未命中返回退出码 2，不生成。 |
| `--dry-run` | 关闭 | 解析并打印计划路径，不加锁、不写文件、不生成。 |
| `--force` | 关闭 | 重建当前精确请求；不能与前两项同用。 |
| `--print-path-only` | 关闭 | 标准输出只写最终表路径，适合 shell 命令替换。 |
| `--lock-timeout-seconds INT` | `7200` | 等待另一个相同请求制表的最长时间。 |

`--initial-intervals`、`--max-points`、全部 `--loss-*` 网格参数与下节生成器同义，
默认值也相同，并全部进入请求哈希。

\(10^{19}\) eV 示例：

```bash
export C8_TABLE="$(
  gpu_em_table_prepare \
    --medium-yaml configs/media/air_dry_1_atm.yaml \
    --cache-dir /path/to/table-cache \
    --primary-energy-eV 1e19 \
    --em-cut-MeV 0.5 \
    --muon-transport-cut-MeV 300 \
    --tolerance 1e-3 \
    --loss-tolerance 1e-3 \
    --print-path-only
)"
```

默认安全系数使表上限成为 \(1.05\times10^{13}\) MeV。生成成功只证明表自身
数值门禁通过；新能区仍需 shower 和射电物理验收。

## 4. `gpu_em_tablegen`

底层生成器从 YAML 建立 PROPOSAL medium，并生成版本化 rate、inverse-CDF、
continuous-range、LPM 和散射表。省略 `--medium-yaml` 时保留历史
`AirDry1Atm` 合同。通常应通过上一节准备工具调用它。

### 4.1 文件和组合参数

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `output` | 必填 | 输出 `.c8emrt` 文件。 |
| `--medium-yaml PATH` | 空 | schema 1 材料定义；空值使用历史标准干空气。 |
| `--proposal-cache PATH` | 空 | PROPOSAL 自身插值 cache；不同于运行时 `.c8emrt`。 |
| `--epair-rho-source PATH` | 空 | 复用 v9/v10 rate table，只追加 dense Epair rho 表。 |
| `--merge-em-source PATH` | 空 | 复用已验证 EM production table。 |
| `--merge-muon-source PATH` | 空 | 将已验证 muon-only table 追加到 `--merge-em-source`。 |
| `--overwrite` | 关闭 | 允许替换已存在的输出文件；默认拒绝覆盖。 |

### 4.2 能区、cut 和精度

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--energy-min-MeV FLOAT` | `0.5` | 最小总能量；不得高于 stochastic cut。 |
| `--energy-max-MeV FLOAT` | `1e12` | 最大总能量，即 \(10^{18}\) eV。 |
| `--cut-MeV FLOAT` | `0.5` | PROPOSAL absolute stochastic energy cut。 |
| `--transport-cut-MeV FLOAT` | `0.5` | \(e^\pm\) kinetic transport cut 和 zero-range anchor。 |
| `--muon-transport-cut-MeV FLOAT` | `300` | \(\mu^\pm\) kinetic transport cut 和 zero-range anchor。 |
| `--photon-pair-final-state-min-MeV FLOAT` | `1e4` | normalized GPU photon-pair final-state 表的最低 photon 能量。 |
| `--tolerance FLOAT` | `1e-3` | 最大 \(dN/dX\) 相对插值误差。 |
| `--loss-tolerance FLOAT` | `1e-3` | 最大 inverse-CDF \(v(E,u)\) 相对插值误差。 |

### 4.3 自适应网格

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--initial-intervals UINT` | `16` | rate table 初始对数能量区间数。 |
| `--max-points UINT` | `20000` | 每个粒子的最大 energy-grid 点数。 |
| `--loss-initial-energy-intervals UINT` | `8` | 每个 inverse-CDF column 的初始对数能量区间。 |
| `--loss-initial-quantile-intervals UINT` | `8` | 每个 inverse-CDF column 的初始 quantile 区间。 |
| `--loss-max-energy-points UINT` | `4096` | 单个 inverse-CDF column 的最大能量点数。 |
| `--loss-max-quantile-points UINT` | `2048` | 单个 inverse-CDF column 的最大 quantile 点数。 |
| `--loss-validation-samples UINT` | `64` | 每个 column 的独立直接 PROPOSAL 验证样本数。 |

### 4.4 Epair rho 和 μ 子

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--epair-rho-min-energy-MeV FLOAT` | `20` | Epair rho inverse-CDF 的最低 parent energy。 |
| `--epair-rho-energy-points UINT` | `65` | dense rho 表的对数能量点数。 |
| `--epair-rho-v-points UINT` | `65` | threshold-excess \(v\) 点数。 |
| `--epair-rho-quantile-points UINT` | `129` | logit rho-quantile 点数。 |
| `--epair-rho-validation-samples UINT` | `4096` | 每个 target component 的独立 PROPOSAL 样本。 |
| `--epair-rho-tolerance FLOAT` | `1e-3` | \(\lvert\rho\rvert/\rho_{\max}\) 的最大绝对插值误差。 |
| `--enable-epair-rho-table` | 关闭 | 生成 dense Epair rho inverse-CDF；仍属于显式实验选项。 |
| `--include-muons` | 关闭 | 同时生成 \(\mu^\pm\) rate、inverse-CDF 和 continuous-range 表。 |
| `--muons-only` | 关闭 | 只生成 \(\mu^\pm\) 表，用于快速开发验证。 |

最小示例：

```bash
gpu_em_tablegen production.c8emrt \
  --medium-yaml configs/media/air_dry_1_atm.yaml \
  --proposal-cache /path/to/proposal-cache \
  --energy-min-MeV 0.5 \
  --energy-max-MeV 1e12 \
  --cut-MeV 0.5 \
  --transport-cut-MeV 0.5 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --include-muons
```

该示例不自动等价于当前 production table。正式表还需要保存完整 metadata、
内容哈希、实测误差和对应构建身份，并通过 shower 验收。

## 5. `cuda_decision_replay`

该程序不重新抽样 shower。它读取 scalar decision tape，在 GPU 上逐条验证记录，
并对 tape 中完全相同的 \(e^\pm\) 轨迹执行 CUDA CoREAS/ZHS 投影。

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--tape PATH` | 必填 | `c8_air_shower --cuda-replay-tape-out` 生成的 tape。 |
| `--output PATH` | 必填 | replay 输出目录；已存在时拒绝覆盖。 |
| `--device INT` | `0` | CUDA device index。 |
| `--memory-fraction FLOAT` | `0.70` | radio buffer 可使用的当前空闲显存比例。 |
| `--fixed-point-field-limit FLOAT` | `1.0` V/m | checked deterministic waveform 的场强范围。 |
| `--deterministic BOOL` | `true` | 使用确定性 fixed-point 累加。 |

示例：

```bash
c8_air_shower \
  -p 11 -E 1000 -N 1 -f scalar-output \
  --seed 10001 \
  --cuda-replay-tape-out event.c8rpt

cuda_decision_replay \
  --tape event.c8rpt \
  --output cuda-replay-output \
  --device 0 \
  --deterministic true
```

严格 replay 证明“相同 transport tape 可在 CUDA 上逐条复核并产生一致射电
投影”，不等同于 production CUDA Monte Carlo 与标量全局随机流逐事例相同。

## 6. `fluka_batch_worker`

该程序通常由 `c8_air_shower --hadronic-backend fluka-process` 自动启动。手工
参数主要用于协议测试、生成测试输入和持久 server 调试。

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--input PATH` | 空 | binary `HadronicBatchProtocol` 请求文件。 |
| `--output PATH` | 空 | binary 响应文件或生成的测试输入。 |
| `--self-test UINT` | `0` | 在单进程中执行指定数量的 deterministic 请求。 |
| `--pool-self-test UINT` | `0` | 通过持久 worker pool 将指定数量请求运行两遍。 |
| `--pool-workers UINT` | `4` | `--pool-self-test` 使用的 worker 数。 |
| `--server-fd INT` | `-1` | 在给定 socket descriptor 上运行持久 binary protocol server。 |
| `--worker-id UINT` | `0` | 持久 server 的诊断 worker ID。 |
| `--generate-test-input UINT` | `0` | 向 `--output` 写出指定数量的 deterministic 测试请求。 |

常用自检：

```bash
export FLUPRO=/path/to/fluka

fluka_batch_worker --self-test 100
fluka_batch_worker --pool-self-test 1000 --pool-workers 4
```

生产运行不要手工设置 `--server-fd`；主进程负责创建、继承和关闭协议 socket。

## 7. `run_physics_acceptance.py`

该驱动创建独立 CPU/CUDA ensembles，核对 provenance，然后比较 profile、
energy deposit、ground particles、radio 和稳定性统计。

### 7.1 输入、样本和恢复

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--executable PATH` | 必填 | CUDA 分支 `c8_air_shower`。 |
| `--proposal-executable PATH` | 同 `--executable` | 可指定独立原版 scalar executable。 |
| `--table PATH` | 必填 | CUDA `.c8emrt` 表。 |
| `--output-root PATH` | 必填 | 新验收根目录。 |
| `--additional-proposal PATH` | 空，可重复 | 合并已有独立 scalar shards；严格核对配置。 |
| `--additional-cuda PATH` | 空，可重复 | 合并已有独立 CUDA shards；严格核对配置。 |
| `--label TEXT` | `custom` | 写入 manifest 和报告的实验标签。 |
| `--energy-gev FLOAT` | `1000` | 初级总能量。 |
| `--events INT` | `1000` | 每个新建 backend ensemble 的事件数。 |
| `--proposal-seed INT` | `41001` | scalar ensemble 起始 seed。 |
| `--cuda-seed INT` | `51001` | CUDA ensemble 起始 seed。 |
| `--paired-seed-control` | 关闭 | 使用相同 seed 做诊断对照；不要求逐事例相同。 |
| `--proposal-shards INT` | `1` | scalar ensemble 拆分进程数。 |
| `--proposal-parallelism INT` | `1` | 同时运行的 scalar shard 上限。 |
| `--resume-completed-proposal` | 关闭 | 严格验证 seed、事件数和命令后复用已完成 scalar shards。 |
| `--resume-completed-cuda` | 关闭 | 在 post-processing 中断后严格核对并复用已闭合 CUDA 输出；要求同时启用 `--resume-completed-proposal`。 |
| `--skip-proposal-run` | 关闭 | 不新建 scalar 事例，只池化至少一个 `--additional-proposal`，并运行新的 CUDA supplement。 |
| `--allow-mixed-proposal-builds` | 关闭 | 显式允许不同机器分别构建的原版 scalar executable；每个来源仍必须有 provenance，且规范化物理配置和实际写出的天线布局必须完全一致。CUDA provenance 不放宽。 |
| `--allow-mixed-cuda-builds` | 关闭 | 显式允许不同 CUDA executable 构建分层池化；rate-table SHA-256 与规范化物理配置仍必须完全一致，并保留全部 executable fingerprints。 |
| `--overlap-backends` | 关闭 | 同时运行 CPU/CUDA；只用于物理统计，禁止用于性能比较。 |

### 7.2 物理配置

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--primary-pdg INT` | `11` | 非核初级 PDG。 |
| `--primary-z INT` | 空 | 核初级 Z；设置后要求 `--primary-a`。 |
| `--primary-a INT` | 空 | 核初级 A；与 `--primary-z` 配套。 |
| `--zenith-deg FLOAT` | `0` | 天顶角。 |
| `--azimuth-deg FLOAT` | `0` | 方位角。 |
| `--em-cut-gev FLOAT` | `5e-4` | EM cut。 |
| `--em-thinning FLOAT` | `1e-4` | EM thinning fraction。 |
| `--maximum-weight FLOAT` | `100` | 显式最大权重；0 表示不向应用传 `--max-weight`。 |
| `--non-em-cut-gev FLOAT` | 空 | 同时设置 hadron/muon/tau cut 的旧简写。 |
| `--had-cut-gev FLOAT` | `0.3` | hadron cut。 |
| `--mu-cut-gev FLOAT` | `0.3` | muon cut。 |
| `--tau-cut-gev FLOAT` | `0.3` | tau cut。 |
| `--shower-core-x-m FLOAT` | `0` | shower core North/x。 |
| `--shower-core-y-m FLOAT` | `0` | shower core West/y。 |
| `--ring INT` | `0` | ring observer 配置。 |
| `--antenna-file PATH` | `/dev/null` | 验收天线文件；ring=0 时 `/dev/null` 关闭 radio。 |

### 7.3 CUDA arm 和统计门禁

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--cuda-hadronic-backend scalar\|fluka-process` | `scalar` | 只控制 CUDA arm 的低能强子执行后端。 |
| `--cuda-hadronic-workers INT` | `4` | CUDA arm 的 FLUKA workers。 |
| `--cuda-hadronic-min-batch INT` | `64` | FLUKA 最小 parked vertices。 |
| `--cuda-hadronic-target-batch-ms FLOAT` | `5` | 目标 batch 代价。 |
| `--cuda-hadronic-max-batch INT` | `256` | 最大 batch。 |
| `--gpu-device INT` | `0` | CUDA device。 |
| `--gpu-min-batch INT` | `4096` | CUDA 最小 batch。 |
| `--gpu-memory-fraction FLOAT` | `0.70` | 可使用空闲显存比例。 |
| `--gpu-table-tolerance FLOAT` | `1e-3` | 表格误差上限。 |
| `--cuda-detailed-stage-timing` | 关闭 | 开启 CUDA event stage profiling。 |
| `--cuda-radio-backend cpu\|cuda` | `cpu` | CUDA arm 的射电投影后端。 |
| `--gpu-radio-field-limit FLOAT` | `1.0` V/m | checked fixed-point radio 范围。 |
| `--gpu-radio-track-diagnostics` | 关闭 | 收集 GPU 轨迹诊断。 |
| `--radio-sampling-rate-ghz FLOAT` | `1.0` | 两个 ensemble 的射电采样率。 |
| `--radio-window-duration-ns FLOAT` | `400` | 两个 ensemble 的射电窗口。 |
| `--radio-pretrigger-ns FLOAT` | `10` | 两个 ensemble 的 pretrigger。 |
| `--relative-tolerance FLOAT` | `0.01` | 关键均值相对偏差门限。 |
| `--sigma-limit FLOAT` | `3` | 统计 z-score 门限。 |
| `--active-fraction FLOAT` | `1e-4` | 纵向曲线 active-bin 判定比例。 |
| `--minimum-bin-pass-fraction FLOAT` | `0.95` | active bins 最低通过比例。 |
| `--stability-bootstrap-repetitions INT` | `20000` | scalar 稳定性 bootstrap 次数；0 关闭。 |
| `--stability-seed INT` | `8052026` | bootstrap 随机种子。 |
| `--key-scalar NAME` | 内置核心列表，可重复 | 替换同时接受 1% 和统计门禁的关键标量列表。 |
| `--require-pass` | 关闭 | 门禁失败时返回 exit code 2。 |

### 7.4 旧版硬编码地磁配置的比较

直接调用 `compare_ensembles.py` 比较不具备地磁 CLI 的旧版 scalar 输出时，
必须显式记录旧二进制中硬编码的系数文件与年份：

```bash
python validation/gpu_em/compare_ensembles.py \
  --proposal ORIGINAL_CPU_OUTPUT \
  --cuda CUDA_OUTPUT \
  --output COMPARISON_OUTPUT \
  --proposal-implicit-geomagnetic-model IGRF13 \
  --proposal-implicit-geomagnetic-year 2025 \
  --allow-cross-build-reference
```

两个 `--proposal-implicit-*` 参数必须成对给出。它们不会忽略地磁配置，也不会
覆盖输出中已有的显式参数；若声明值与 CUDA 输出中的
`--geomagnetic-model/--geomagnetic-year` 不一致，比较会在计算统计量前失败。
`diagnose_longitudinal_mean_difference.py` 提供同名参数，使固定深度与 Xmax
对齐诊断遵循相同的配置门禁。

## 8. `run_remote_cpu_ensemble.py`

该脚本在固定 CPU 集合上运行独立 scalar showers。当前调度器使用全局动态队列：
每个 worker 完成一个事例后立即领取下一个事例；只要剩余任务不少于 worker
数，就保持全部指定核心忙碌。单个事例失败会被记录，但不会停止其他核心领取
后续任务。

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--executable PATH` | 必填 | scalar `c8_air_shower`。 |
| `--output-root PATH` | 必填 | ensemble 输出根目录。 |
| `--antenna-file PATH` | 必填 | NWU observer 文件。 |
| `--events INT` | `500` | 总独立 shower 数。 |
| `--jobs INT` | `130` | 同时工作的固定核心数。 |
| `--cpu-list SPEC` | `120-249` | 允许的逻辑 CPU，例如 `0-31,64-95`。 |
| `--seed-start INT` | `10100051` | 第 0 个事例 seed；第 \(i\) 个使用 `seed-start+i`。 |
| `--seed-list-file PATH` | 空 | 每行一个显式 seed；忽略空行和 `#` 注释，并取代 `--events/--seed-start`。用于修复后原 seed 精确重跑，重复、负值或非法 seed 会被拒绝。 |
| `--primary-pdg INT` | `2212` | 初级 PDG。 |
| `--energy-gev FLOAT` | `1e5` | 初级总能量。 |
| `--zenith-deg FLOAT` | `0` | 天顶角。 |
| `--azimuth-deg FLOAT` | `0` | 方位角。 |
| `--shower-core-x-m FLOAT` | `0` | shower core x。 |
| `--shower-core-y-m FLOAT` | `0` | shower core y。 |
| `--ring INT` | `0` | ring observers。 |
| `--em-cut-gev FLOAT` | `5e-4` | EM cut。 |
| `--em-thinning FLOAT` | `1e-6` | EM thinning fraction。 |
| `--had-cut-gev FLOAT` | `0.3` | hadron cut。 |
| `--mu-cut-gev FLOAT` | `0.3` | muon cut。 |
| `--tau-cut-gev FLOAT` | `0.3` | tau cut。 |
| `--maximum-weight FLOAT` | `0` | 0 表示省略应用的 `--max-weight`，保留自动值。 |
| `--flupro PATH` | `/home/yuhanglu/fluka` | 含 `libflukahp.a` 的 FLUKA 安装。 |
| `--resume` | 关闭 | 只复用 command、seed、artifact hash 和 provenance 完全一致的 shards。 |
| `--dry-run` | 关闭 | 只验证输入并输出 immutable configuration。 |

推荐先检查：

```bash
python validation/gpu_em/run_remote_cpu_ensemble.py \
  --executable /path/to/c8_air_shower \
  --output-root /path/to/new-output \
  --antenna-file /path/to/antennas_nwu.txt \
  --flupro /path/to/fluka \
  --events 500 \
  --jobs 130 \
  --cpu-list 120-249 \
  --dry-run
```

再移除 `--dry-run`。不要让两套 runner 使用相同输出目录或重叠 CPU 集合。

## 9. `run_local_cuda_ensemble.py`

该脚本把大型 CUDA 样本拆为可验证、可恢复的批次，并在每个批次后检查输出
闭合、executable/table/天线/FLUKA 哈希以及 CUDA 完整状态。关键参数包括：

| 参数 | 默认值 | 功能 |
|---|---:|---|
| `--target-events INT` | `500` | 最终需要的 CUDA shower 数。 |
| `--batch-events INT` | `25` | 每个 `c8_air_shower` 进程内的事件数。 |
| `--cuda-seed-start INT` | 必填 | 第一个 CUDA seed；后续连续递增。 |
| `--geomagnetic-model IGRF13\|IGRF14` | `IGRF14` | 显式传给每个 CUDA 批次并写入 immutable manifest。 |
| `--geomagnetic-year FLOAT` | `2027` | 与模型成对构成地磁物理配置。 |
| `--defer-reference-comparison` | 关闭 | 只完成 CUDA 样本；等远端 CPU 数据同步后再统一比较。 |
| `--existing-cuda PATH` | 空，可重复 | 将严格验证过的既有 CUDA shards 计入目标事件数。 |

正式运行应始终显式传入 `--geomagnetic-model` 和 `--geomagnetic-year`，即使采用
默认值。跨机器比较时，旧版 CPU 二进制的硬编码值需要按 7.4 节登记。

当一个高能 shower 的 CPU 强子阶段较长、GPU wavefront 呈间歇 burst 时，可以
用两个独立 runner 提高整批吞吐量，但必须同时满足：

1. 两个 `--cuda-seed-start` 区间完全不重叠；
2. 两个 `--output-root` 和进程工作目录各自独立；FLUKA 会在当前工作目录创建
   `fort.11` 与 `.timer.out`，不能让独立主进程共享这些 scratch 文件；
3. 各进程的 `--gpu-memory-fraction` 总和保留足够的 CUDA runtime/table 余量，
   例如两个进程各用 `0.35`；
4. 每个输出仍分别通过 queue overflow、CPU spill、radio fixed-point overflow、
   表哈希和完整性检查，最后才按 provenance 池化；
5. 并发总吞吐量不能冒充单 shower 隔离性能。

## 10. 生产示例

```bash
export C8_BUILD=/path/to/corsika8-gpu-build
export C8_TABLE=/path/to/production.c8emrt
export C8_ANTENNAS=/path/to/antennas_nwu_coordinates_test.txt
export FLUPRO=/path/to/fluka

"$C8_BUILD/applications/c8_air_shower" \
  -p 2212 \
  -E 100000 \
  -N 1 \
  -f /path/to/output \
  --seed 10400001 \
  --zenith 0 \
  --azimuth 0 \
  --geomagnetic-model IGRF14 \
  --geomagnetic-year 2027 \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --antenna-file "$C8_ANTENNAS" \
  --em-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_TABLE" \
  --gpu-table-tolerance 1e-3 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true \
  --radio-backend cuda \
  --gpu-radio-field-limit 1 \
  --hadronic-backend fluka-process \
  --hadronic-workers 4 \
  --hadronic-min-batch 64 \
  --hadronic-target-batch-ms 5 \
  --hadronic-max-batch 256
```

运行后至少核对：

1. 顶层 `summary.yaml` 和 `simulation_timing/summary.yaml`；
2. `gpu_em/config.yaml` 中的 executable、table、GPU、IGRF14/2027 和参数 identity；
3. `gpu_em/summary.yaml` 中的完整状态、fallback、overflow、显存和阶段计时；
4. `CoREAS`/`ZHS` 的 config、summary 和 observers；
5. 没有 NaN、负能量、未知 PID、表 hash 不匹配或未注册过程。
