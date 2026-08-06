# CORSIKA 8 CUDA 电磁后端生产使用指南

> 本文说明生产运行流程和物理边界。所有新增 executable 参数、默认值、参数间
> 约束以及辅助工具参数见
> [CUDA/FLUKA 命令行参数参考](cli_reference.md)。

## 1. 当前能力边界

本后端用于在 NVIDIA GPU 上输运：

```text
gamma, electron, positron
table-enabled muon minus / muon plus transport
```

CPU 继续负责：

```text
hadron, tau
muon decay final state and rare muon discrete final states
photoproduction / photonuclear
photon-induced muon pair
其他会产生 hadron/muon/tau 的指定末态
```

μ 子 GPU 路径覆盖连续电离、range、Molière 散射、磁场/大气 tracking、边界、
观测面、cut 以及离散过程/衰变距离竞争。GPU 把 μ 子推进到已选衰变顶点后由
CPU `forceDecay()` 生成实际衰变末态；bremsstrahlung、pair production 和
photonuclear 等稀有末态也按已选 process/component/\(v\) 返回 CPU。由于原版
radio process 不观察 μ 轨迹，CUDA μ 轨迹同样不作为额外射电源项。

当前生产环境快照支持：

- CORSIKA 五层球形大气；
- 每层干空气组成及唯一 medium ID；
- 均匀磁场；
- 球形观测面；
- 0.5、5、50 MeV 已生成并验证的 EM cut 表。

山体、月壤、冰和一般三维介质尚不属于当前 snapshot 能力，不能仅更换输入几何就
认为 GPU 后端支持。此类扩展需要增加介质表、device geometry 和独立物理验收。

## 2. 运行架构

```text
CPU CORSIKA main stack
        |
        | gamma/e-/e+
        v
HybridCascade + PhysicalCudaEmRouter
        |
        v
host staging -> resident device photon/lepton queues
        |
        +-- stable PID x medium x energy bucketing
        +-- interaction selection
        +-- transport to nearest physical limit
        +-- final-state generation + thinning
        +-- resident cross-species reroute
        |
        +--> ObservationRecord -> CORSIKA particle writer
        +--> profile/deposit accumulator -> CORSIKA writers
        +--> RadioTrackRecord -> CPU CoREAS/ZHS
        |                    or resident CUDA radio
        |
        +--> specified rare-process record
              -> CPU PROPOSAL final-state generator
              -> gamma/e-/e+ back to GPU
              -> other species to CPU main stack
```

原 `Stack` 没有改成并发容器。GPU 使用独立的 SoA、CUB stable scan/compaction
和常驻 photon/lepton 队列。

## 3. 构建

当前本机配置：

```text
GPU                       NVIDIA GeForce RTX 4060 Laptop GPU
driver                    560.94
CUDA toolkit              12.6.85
CUDA architecture         89
CMake                     >= 3.24
C++/CUDA                  17
build type                Release
```

在 `corsika_venv` 中：

```bash
export C8_SOURCE=~/corsika-21cma-cuda/corsika8_gpu_refactor
export C8_BUILD=~/corsika-21cma-cuda/corsika8_gpu_refactor_build_cuda

cmake -S "$C8_SOURCE" \
  -B "$C8_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build "$C8_BUILD" --parallel 8
```

上述命令适合已有 Conan/CMake 配置的本机增量构建。迁移到新服务器时还必须
重建 Conan Release 依赖、选择新 GPU 的 compute capability、校验 FLUKA，并
使用全新的 CMake cache；完整的从零部署和多 GPU/多核调优步骤见项目主
[`README.md`](../../README.md#deploying-on-a-new-nvidia-server)。

默认仍是：

```text
CORSIKA_ENABLE_CUDA=OFF
```

CPU-only build 不要求 CUDA toolkit。当前 CPU-only 全树及 CTest 10/10、CUDA
Release 全树及 CTest 34/34 均通过；包含 FLUKA 的模块测试需要设置 `FLUPRO`。

## 4. 物理表

先区分两类文件：

| 文件 | 是否自动生成 | 用途 |
|---|---|---|
| PROPOSAL 插值 cache 目录 | 是，由准备/生成工具按需创建 | 仅供生成器调用 PROPOSAL |
| 版本化 `.c8emrt` 文件 | 可由独立 `gpu_em_table_prepare` 生成 | CUDA 后端运行时物理表 |

这里的“自动”是显式运行独立准备步骤，不是 shower 启动时的隐式动作。生产
运行不能在缺少 `.c8emrt` 时生成或切换 CPU。`c8_air_shower` 要求显式传入
`--gpu-table-cache`，文件不存在、损坏或不匹配都会立即终止。

### 4.1 默认表合同

旧的 `production_v10_muons_1e-3_1EeV.c8emrt` Release 表产生于 split-cut
合同之前，不能用于当前应用。默认标准干空气请求使用 0.5 MeV CORSIKA EM
输运 cut、按标量缓存规则解析出的 0.4 MeV PROPOSAL stochastic cut、300 MeV
muon cut 和所需的明确能区。表中同时存在 PDG `13/-13` 时启用 CUDA μ 子输运；
没有 μ 子列时 EM CUDA 仍可使用，而 μ 子保留在 CPU。

### 4.2 自动查找或生成标准干空气表

推荐用介质 YAML 和初级能量准备内容寻址表：

```bash
export C8_TABLE="$(
  gpu_em_table_prepare \
  --medium-yaml configs/media/air_dry_1_atm.yaml \
  --cache-dir /path/to/c8_gpu_table_cache \
  --primary-energy-eV 1e18 \
  --em-cut-MeV 0.5 \
  --electron-transport-cut-MeV 0.5 \
  --muon-transport-cut-MeV 300 \
  --tolerance 1e-3 \
  --loss-tolerance 1e-3 \
  --print-path-only
)"
```

工具对 YAML 规范化并计算 SHA-256，扫描介质、cut、能区、精度和制表器合同
版本均兼容的现有表；未命中时通过跨进程锁
调用 `gpu_em_tablegen`，随后回读校验并写 manifest。`--dry-run` 只显示请求，
`--lookup-only` 禁止生成。已有 \(10^{18}\) eV 表不能外推到 \(10^{19}\) eV；
后者应将 `--primary-energy-eV` 改为 `1e19`，默认表上限会解析为
\(1.05\times10^{13}\) MeV。

生成器会建立 rate、inverse-CDF、continuous range、LPM 和散射数据，自适应
细化并验证误差。但一张新生成的表不会自动继承 production 验收身份；正式科研
使用前仍应重新运行 CPU/CUDA shower、能量闭合和射电统计验收。表文件记录：

- PROPOSAL 版本与参数化；
- medium 组成；
- cut 与能区；
- rate/inverse-CDF/continuous-table 最大实测误差；
- schema 版本与内容哈希。

表不匹配、损坏、哈希错误或误差大于 `--gpu-table-tolerance` 都会终止当前
shower。PROPOSAL 原生插值与 GPU 平坦表插值的区别见
`proposal_and_cuda_interpolation.md`。

### 4.3 改变介质时的边界

`gpu_em_tablegen --medium-yaml` 和 `gpu_em_table_prepare` 已支持 schema 1
材料配置；`c8_air_shower` 的五层环境仍固定为 CORSIKA `AirDry1Atm`。

- 改变同一干空气的密度—高度 profile、磁场、观测面或 GPU：复用现有表；
- 改变元素组成、组分比例、电离/材料常数或 PROPOSAL 版本：用新 YAML 生成表；
- 岩石、土壤、月壤和冰：现在可以规范化制表，但端到端模拟仍需扩展环境快照、
  device geometry、GPU medium ID 和运行时环境—表格身份校验。

不要把自定义介质表直接交给仍使用标准干空气 snapshot 的应用。完整 schema、
自动查找、并发和 \(10^{19}\) eV 示例见
[`gpu_em_tables/README.md`](../../gpu_em_tables/README.md)。

## 5. 基本运行

### 5.1 CPU 默认路径

```bash
c8_air_shower \
  -p 11 -E 1000 -N 10 \
  -f OUTPUT_CPU \
  --seed 10001
```

省略 `--em-backend` 等价于：

```text
--em-backend proposal
```

不需要 GPU device 或 table 参数。

### 5.2 CUDA EM + CPU CoREAS/ZHS

这是首个生产合同推荐路径：

```bash
c8_air_shower \
  -p 11 -E 1000000 -N 1 \
  -f OUTPUT_CUDA \
  --seed 10001 \
  --emcut 0.0005 \
  --emthin 1e-4 \
  --max-weight 100 \
  --em-backend cuda \
  --radio-backend cpu \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache \
    "$C8_TABLE" \
  --gpu-table-tolerance 1e-3 \
  --gpu-deterministic true
```

`C8_TABLE` 应来自第 4.2 节的当前内容寻址准备请求。接口默认值为
`--gpu-min-batch 4096`；当前 RTX 4060 的 100 TeV 强子 shower 实测也优于
64。这个阈值不是跨 GPU 常数：迁移到新卡后应固定物理配置和热缓存，对
64、256、1024、4096、8192 等候选值做至少五次重复的中位数扫描。小于阈值时
scheduler 执行有界的 CPU 前沿展开，避免大量很小的 kernel launch。

### 5.3 可选 CUDA radio

```text
--radio-backend cuda
--gpu-radio-field-limit 1
```

它使用固定点 deterministic accumulator。任何 overflow 都是 hard failure。
CPU radio 保持生产默认兼容路径；CUDA radio 已通过同轨迹 TeV、PeV 和强子
fallback 波形验收，但不是使用 CUDA EM 的必要条件。

### 5.4 可选 FLUKA 多进程末态

CUDA HybridCascade 可把低能强子末态交给进程隔离的持久 FLUKA worker：

```text
--hadronic-backend fluka-process
--hadronic-workers 4
--hadronic-min-batch 64
--hadronic-target-batch-ms 5
--hadronic-max-batch 256
```

这要求 `WITH_FLUKA=ON`，并要求 `fluka_batch_worker` 位于
`c8_air_shower` 同一目录，或通过 `--hadronic-worker-executable` 指定。
它并未把强子物理移植成 CUDA kernel；加速来自独立 FLUKA 进程、按物理工作类
分组和批量 IPC。新多核服务器必须独立扫描 worker 数量，性能计时时不得同时
运行另一组 CPU shower。

`fluka-process` 还有一个仅影响 CUDA 主进程的安全处理：FLUKA 在构造进程内
相互作用对象时注册的周期性 `SIGALRM` quota timer 会被主进程关闭。此时 worker
已经通过 `fork/exec` 独立启动，仍保留标准 FLUKA 行为。不要在应用层重新打开
这个主进程 timer；其 signal handler 会执行非 async-signal-safe 的 stdio，在
CUDA helper thread 存在时可能等待同一线程已经持有的 libc 锁。

### 5.5 21CMA 地磁场

当前应用默认读取项目数据中的 `GeoMag/IGRF14.COF` 并使用 2027 年，也可通过
`--geomagnetic-model IGRF13|IGRF14` 和 `--geomagnetic-year` 显式选择；
仓库中的 `resources/GeoMag/IGRF14.COF` 是 IGRF14 的可复现后备文件，并会随
安装复制到共享数据目录；
21CMA 位置固定为纬度 42.5527 度、经度 86.4153816422 度、海拔
2680.444195 m。CPU 与 CUDA
共享同一个计算后场矢量，具体值写入 `gpu_em/config.yaml`，因此服务器迁移不能
遗漏 CORSIKA data 安装。

## 6. 随机数与可重复性

设备端使用 Philox4x32-10，随机地址包含：

```text
seed, shower_id, history_id, step_id, process_id, draw_id
```

同一 GPU、同一表、同一配置必须可重复，wavefront 重排不能改变某个 history 的
随机数。不同 GPU 架构只要求统计一致，不承诺逐位一致。生产构建不使用
`--use_fast_math`，粒子状态、物理表与几何均使用 double。

这不表示生产 CUDA 与原版 LIFO/全局顺序随机流只靠相同 seed 就会产生同一棵
粒子树。需要逐事例调试时，应使用原版 scalar decision tape 和
`cuda_decision_replay`；其证据范围、命令和 1 TeV 正式结果见
`phase_82_exact_legacy_event_cuda_decision_replay.md`。

## 7. CPU fallback 与失败语义

允许的指定 CPU 末态包括：

- photoproduction/photonuclear；
- photon-induced muon pair；
- GPU 已选定 Epair 过程后，原 PROPOSAL 逆 CDF 的低分位/包络能力边界；
- 产生非 EM 物种的末态。

GPU 已经选定 process、component 和随机 key，CPU 不能重新抽过程。

generic scalar fallback 只允许预声明的能力边界：

```text
unsupported_particle
unsupported_medium
unsupported_geometry
```

表格、LPM、Molière、磁场、大气积分、非法末态、NaN、负能量、queue overflow
和 CUDA error 均为 hard failure。应用会把当前 shower 标为 incomplete，不会
自动切换成 CPU 并伪装成成功。

内存不足时只有受控的最低能尾部 spill 可以回 CPU；它使用全局稳定能量排序并
记录数量、能量边界和恰好一次 scalar step。

Phase 73 的运行时回归直接从 photon CUDA transport 触发了两条完整路径：
最外层边界的 `unsupported_geometry` 以相同 history/step 返回并只执行一个
scalar step；人为构造的 `atmosphere_grammage_failed` 在 strict 模式抛错，
CPU tracking 和 continuous-process 调用数都保持为零。它们不是只检查
fallback 枚举的单元测试。

## 8. 输出与审计

`gpu_em/config.yaml` 记录：

- GPU 型号、driver/runtime；
- backend/table schema 与内容哈希；
- table 实测误差和用户容差；
- device、memory fraction、batch、deterministic 配置。

`gpu_em/summary.yaml` 每个 shower 记录：

- complete/status；
- GPU 粒子、CPU steps、wavefront 数；
- process/reason 的 fallback 数字 ID 和可读名称；
- overflow、spill 和队列峰值；
- peak device bytes；
- kernel/transfer/CPU fallback/host postprocess timing；
- profile/radio/thinning/process/Molière/Epair 统计；
- pure-EM 完整覆盖时的严格能量账本。

调试时才使用：

```text
--gpu-detailed-stage-timing
--gpu-full-step-records
```

它们会增加同步、记录或输出开销，不应用于正式性能结论。

## 9. 验证命令

完整 C++/CUDA 回归：

```bash
ctest \
  --test-dir "$C8_BUILD" \
  --output-on-failure --parallel 8
```

Python 验收工具：

```bash
python -m unittest discover \
  -s validation/gpu_em/tests -p 'test_*.py'
```

正式性能：

```bash
python validation/gpu_em/run_performance_acceptance.py \
  --executable BUILD/applications/c8_air_shower \
  --table TABLE.c8emrt \
  --output-root OUTPUT \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --cache-mode warm \
  --require-release-build \
  --minimum-speedup 5
```

物理 ensemble：

```bash
python validation/gpu_em/run_physics_acceptance.py \
  --executable BUILD/applications/c8_air_shower \
  --table TABLE.c8emrt \
  --output-root OUTPUT \
  --energy-gev 1000 \
  --events 1000 \
  --proposal-shards 8 \
  --proposal-parallelism 8 \
  --overlap-backends \
  --require-pass
```

已有独立 ensemble 可以用重复的 `--additional-proposal` 和
`--additional-cuda` 与新样本合并。工具会从每个输出的 `config.yaml` 提取
物理命令，只忽略 seed、事件数、输出路径和 backend 实现参数；能量、初级、
方向、cut、thinning、环境/跟踪或相互作用选项有任何差异都会在计算均值前
拒绝合并。

若 CPU 样本已经在本机或服务器上完成，可用 `--skip-proposal-run` 只生成
新的 CUDA supplement；此时至少要给出一个 `--additional-proposal`。不同
机器分别编译的原版 scalar binary 默认仍会被拒绝，只有显式加入
`--allow-mixed-proposal-builds` 才允许分层池化。该模式保留每个来源的
provenance，并继续严格核对物理参数和 CoREAS/ZHS 实际写出的天线位置与
采样网格。CUDA executable 默认也不允许混用；只有显式加入
`--allow-mixed-cuda-builds`、rate-table SHA-256 与规范化物理配置完全一致，
并额外通过 `compare_cuda_build_strata.py` 的独立构建分层检验后，才能把
两个 CUDA 构建的样本作为同一统计集合使用。rate-table 哈希始终不允许混用。

同轨迹射电比较：

```bash
python validation/gpu_em/run_radio_acceptance.py \
  --executable BUILD/applications/c8_air_shower \
  --table TABLE.c8emrt \
  --output-root OUTPUT \
  --energy-gev 1000 \
  --events 10 \
  --ring 1 \
  --require-pass
```

性能测试不能并行重叠 CPU/CUDA；`--overlap-backends` 只允许用于物理 ensemble。

原版单事例 CUDA 决策回放：

```bash
BUILD/applications/cuda_decision_replay \
  --tape event.c8rpt \
  --output replay_output \
  --device 0 \
  --fixed-point-field-limit 0.001 \
  --deterministic true

python validation/gpu_em/compare_exact_cuda_replay.py \
  --untaped-reference cpu_no_tape \
  --reference cpu_with_tape \
  --replay replay_output \
  --report exact_replay_comparison.json \
  --require-pass
```

该模式验证 CUDA 消费的是否为原版精确输运记录，并用 GPU 重新计算同轨迹
CoREAS/ZHS；它明确不是 production CUDA 独立抽样。

## 10. 当前正式证据

```text
CUDA Release all target                 PASS
CUDA CTest                              34/34
CPU-only all target                     PASS
CPU-only CTest                          10/10
Python validation                       53/53
installed downstream GPU target link    PASS
hot-cache 1 PeV speedup                 8.60x external / 12.63x shower timing
cold-each 1 PeV speedup                 4.85x external / 6.01x shower timing
1 PeV final-hash physics ensemble       800+800, 9/9 curve families PASS
1 TeV 10-event CoREAS/ZHS               PASS
1 PeV CoREAS/ZHS                        PASS
proton + specified fallback + radio     PASS
original 1 TeV exact decision replay    PASS, 136114 records, 0 hash mismatch
same-track CPU/CUDA CoREAS and ZHS      PASS, 81 antennas x 3 components
1 TeV electron feature distributions   500+500, all selected KS tests PASS
```

完整构建哈希、最终物理矩阵、冷热缓存定义以及仍受高方差限制的 raw-mean 项见
`phase_78_terminal_observation_sophia_threshold_and_final_matrix.md` 和
`original_plan_requirement_audit.md`。后者是判定当前科研结论范围的滚动索引，
不能用单个 smoke test 替代。
