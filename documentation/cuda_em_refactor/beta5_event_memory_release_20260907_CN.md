# beta5：逐 shower 输出内存释放

## 修改范围

本轮只调整输出和已消费的诊断数据生命周期，不改变 EM/强子/射电物理、随机数流、cut、thinning 或 PROPOSAL 表。已有生产 campaign 使用归档程序，本轮不替换它。

- `ParquetStreamer` 为每个 writer 设置 16 MiB row-group 目标，并提供 `flushStreamer()`。所有 Parquet writer 在事件结束后提交已完成的行，文件仍保持打开以接收后续 shower。
- 新增 `ShowerSummary`：把 `shower_N` YAML 映射依次写入输出目录内的私有 journal，每次写成功后释放当前 YAML 图；上一例可重写为 incomplete，禁止重写更早的事件。完整 `summary.yaml` 的字段、顺序和层级不变。
- `OutputManager` 通过虚拟 `writeSummary()` 输出报告，不在结束时把完整 journal 再解析回内存。显式调用旧 `getSummary()` 仍可取得全库快照，但调用者需承担全量加载成本。
- GPU 报告、radio track 诊断、simulation timing、primary、ground、interaction、能损和production摘要使用流式存储。
- `c8_air_shower` 在报告和相互作用直方图写完后释放强子计时样本的容量；累计相互作用数、物理直方图、能量账本及总计时不重置。
- 物理表、Kokkos 工作区、observer 波形容量保持复用，不在每例后清空显卡或重新制表。

## 边界与兼容性

16 MiB 是每个 Parquet 数据块的目标，不是进程 RSS 硬上限。Arrow 的文件 footer 保留少量数据块元数据，分配器也可能缓存已经释放的页；因此不承诺 RSS 每例都下降或任意大的 N 下完全恒定。此修改针对此前每例保留大量输出对象的累积，而不是通过 `malloc_trim` 掩盖保留来源。

Parquet 的物理分块和文件字节会改变；读出的 schema、行序和数值必须相同。Arrow 可能在末尾保留一个零行数据块，验收不得把它误计为额外 shower。

Journal 位于该输出子目录，正常析构会清理。模拟运行中不要删除隐藏 journal；突然断电或 SIGKILL 后可能留下可用于诊断的片段，它本身不是 complete 标志，也不是粒子级 checkpoint。写入/序列化错误抛出异常，不静默丢弃报告。

## 验收入口

- `tests/output/testShowerSummary.cpp`：2000条流式记录、顺序和精度、最后一例覆盖、复制隔离、存储失败、完整性状态、OutputManager 不全量加载。
- `tests/output/testParquetStreamer.cpp`：逐事件分块、空 flush、部分行拒绝、文件行数。
- `tests/framework/testInteractionCounter.cpp`：释放样本而不清除累计物理计数。
- `validation/accelerator/run_event_memory_regression.py`：前后程序各自同 PID 多 shower，逐项比较 Parquet/NPZ，并使用已有 RSS/可用内存保护。

## 2026-09-07 实测结果

在 `corsika_venv`、Release、CUDA/OpenMP 组合二进制上，固定 seed `26090711`。两种后端各自用修复前/后程序运行同一序列：1 GeV 光子、垂直、`-N 256`、完整 CoREAS/ZHS、81天线；OpenMP 4线程，CUDA 主机1线程。诊断固定 `min-batch=16`、resident容量4096和显存预算15%，避免与正在进行的高能生产任务争用大量显存；这不是默认容量性能测试。如此低能下自动 max-weight 小于1，所指定的 `emthin=1e-6` 实际不触发 thinning。

每个256例序列从头到尾是**同一个 PID**，没有用每例启动新进程来隐藏增长。

| 后端 | 修复前，第8 → 256例 RSS | 修复后，第8 → 256例 RSS | 后半段斜率：前 → 后 |
|---|---:|---:|---:|
| OpenMP | 658.1 → 1217.1 MiB | 595.8 → 608.4 MiB | 2.146 → 约0 MiB/例 |
| CUDA | 611.3 → 1182.3 MiB | 542.1 → 564.5 MiB | 2.128 → 0.0088 MiB/例 |

修复后第256例的 RSS 分别降低约50%和52%。这说明此前主要的逐事件积累已在本轮测试中消除，不等于证明任意初级能量、任意 N 下严格常量内存。读数来自结束日志附近的 `/proc` 采样，不是精确插入的 C++ 事件边界；启动期或事件内峰值可能更高。

验收覆盖：

- CUDA / OpenMP：各自光子 `N=256`、10 GeV质子 `N=4`，修复前后各运行一次。
- 原标量 PROPOSAL + CPU radio：光子 `N=16`、10 GeV质子 `N=4`，修复前后各一次。
- 12个进程共1080个 shower 正常结束；6组比较的 **1122个 Parquet/NPZ 文件读回数组完全一致**，含两种射电波形和相互作用直方图。不是要求 CPU 与 GPU 相互逐位一致，而是每条路径分别与自己的修复前版本比较。
- 所有6组 YAML 对比通过：字段/顺序、物理计数、能量账本、PROPOSAL/aux哈希及 backend复用状态一致。仅排除明确列出的墙钟计时及其派生统计、编译 revision；粒子飞行时间和 radio time residual 不排除。比较器由 `compare_event_memory_metadata.py` 实现，有独立单测防止放过物理时间差异。
- 两种后端的原生表分配均稳定为78,665,960字节，第二例起 `reused=true`。光子样本中 OpenMP 的251例、CUDA 的252例有非零且有限的 CoREAS/ZHS 波形，因此波形回归并非只比较全零数组。
- `testOutput`、`testFramework` 全部通过；报告流式2000条记录、错误覆盖、存储失败和 Parquet 行数均覆盖。`--help` 除被比较程序的文件名外完全一致。
- 测试进程自身 swap 最大值为0；全系统可用内存始终约6.4 GiB以上。监控服务设4 GiB cgroup限制、3 GiB单模拟RSS/系统剩余门禁，未触发终止。

本轮能量账本对包含 scalar EM 步的 shower 仍会如旧版一样报告 `complete_coverage=false`；本轮验证其**未因内存修复改变**，不是重新宣布全物理能量闭合验收通过。

### 时间代价

在这次低能 `N=256` 运行中，OpenMP 总墙时83.2 → 100.0 s，CUDA 101.1 → 131.0 s。输出直接写WSL挂载的D盘，逐事件提交增加文件操作；测试时另有高能GPU生产任务，没有做独占设备交错重复，因此不能把差值全部归因于修复，也不能声称性能无退化。内存稳定优先；高能生产性能需另行固定条件复测。

### 结果与安装位置

数据及曲线位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  diagnostic_beta5_event_memory_release_20260907_v1/
    event_memory_before_after.png
    event_memory_before_after.pdf
    memory_review.json
    regression.json
    metadata_comparison.json
```

`build/cuda-openmp` 已完成修复后的 Release 编译，`install/cuda-openmp` 已更新。可直接使用 `install/cuda-openmp/bin/c8_air_shower`，通过 `--kokkos-execution cuda|openmp` 选择运行模式。原独立 `install/cuda`、`install/openmp` 和统一启动器本轮未替换；使用它们需要按 README 重新编译，不能假设已经获得此修复。正在运行的高能 campaign 继续用其归档程序，不受本轮安装影响。

安装二进制 SHA-256：`9761440a8bab417799e9a6ea211b5d0264b8824b2b89bfe070a86b6c5de28441`。安装会调整 RPATH，所以与构建目录的二进制哈希不同。

安装后另在 `installed_N2/` 对安装程序与已验收的构建程序做了CUDA/OpenMP光子 `N=2`、质子 `N=4` 对照：4组数组和YAML比较全部通过，共24个 shower；两种模式均正常完成并在后续 shower 复用 backend。该测试也由限内存的独立 systemd 用户服务运行，服务已正常结束。

尚未完成的矩阵不标记通过。大数据和监控曲线保留在 D 盘，不写入源码仓库。
