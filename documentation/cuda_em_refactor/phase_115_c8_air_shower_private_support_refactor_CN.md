# Phase 115：`c8_air_shower` 应用私有支持层重构

## 1. 结论

本阶段将 `applications/c8_air_shower.cpp` 中的 CUDA 命令行、运行级资源、HybridCascade 接线和 YAML 报告实现迁移到应用私有支持层。主文件由重构前的 3316 行缩减为 1122 行，保留了大气、初级粒子、物理模型、`ProcessSequence`、CPU/CUDA 分支和 shower 生命周期等高层结构。

本次改动不改变物理算法、随机数键、CLI、默认值、输出 schema、CUDA kernel、PROPOSAL 表格、fallback、tracking 或射电算法。固定种子回归表明，CPU、CUDA `c8emrt` 和 CUDA `proposal-native` 三条路径的物理输出均与重构前逐文件一致。

## 2. 文件职责

```text
applications/c8_air_shower.cpp
  ├─ 构造环境、初级粒子和物理模型
  ├─ 保留原 ProcessSequence 顺序
  ├─ 运行标量 Cascade
  └─ 调用 runCudaAirShower(...)

applications/detail/air_shower_cuda/
  ├─ GpuCliOptions.{hpp,cpp}       CLI 注册、解析后校验和 replay 文件
  ├─ CudaRunSession.{hpp,cpp}      表格、CUDA 设备、运行 metadata 和 backend 复用
  ├─ CudaAirShowerRunner.hpp       原生表导出、registry、fallback、router 和 HybridCascade
  └─ CudaShowerReport.hpp          原有 YAML、计时、显存、fallback 和能量闭合报告
```

这些文件只通过 `applications/CMakeLists.txt` 中的私有静态目标 `c8_air_shower_support` 使用，不安装为 CORSIKA 公共头文件。CPU-only 构建只编译 `GpuCliOptions.cpp`；开启 CUDA 时再追加 `CudaRunSession.cpp` 和 CUDA 相关链接。`c8_air_shower` 与 parent-profile 验证程序共享该支持目标。

## 3. 行为保持方法

- CLI 选项从旧代码机械迁移，注册顺序、名称、默认值和帮助文本保持不变。
- 运行级 `CudaRunSession` 在原来的位置构造，首次调用 `initialize()`、后续 shower 调用 `beginShower()` 的时机不变。
- `CudaEventConfig` 只保存已在主程序中计算出的单事例状态，不重新计算能量、cut、thinning、几何或随机种子。
- `runCudaAirShower(...)` 接收主程序已构造的过程对象，保持 process registry、router、fallback 和 `HybridCascade` 的原接线顺序。
- `CudaShowerReportBuilder` 保留旧 YAML 节点的字段名称、层级和插入顺序。
- CPU 标量分支和原 `ProcessSequence` 仍直接留在主程序，未通过新支持层执行。

## 4. 验收结果

基线保存在 `/tmp/c8-air-refactor-baseline-fZKpPC`。重构前源码 SHA-256 为 `b9badad85a6c5480beef0fa5df9b47a6b9ca862e94e87244a959400772142ab0`，重构前二进制 SHA-256 为 `71ec7c41d95ab5d719fdd01fb22db212fa598dbc338362ee63cfcb5cf28afa46`。

| 门禁 | 结果 |
|---|---|
| 主文件规模 | 3316 行降至 1122 行 |
| CUDA Release 构建 | 通过 |
| CPU-only Release 构建 | 通过；生成的应用无 CUDA 动态依赖 |
| 独立安装 | CPU-only 与 CUDA 前缀均安装通过；应用私有支持头文件未进入公共 include |
| parent-profile 验证程序 | 与同一支持目标编译、链接通过 |
| `--help` | 去除可执行文件路径和日志源码行号后完全一致 |
| 失败门禁 | 缺表、radio/backend 不兼容、replay/CUDA 不兼容、FLUKA worker 路径、无效设备等退出状态和错误文本一致 |
| 固定种子 CPU | profile、地面粒子、相互作用记录、CoREAS/ZHS 数组的文件哈希全部一致 |
| 固定种子 CUDA `c8emrt` | 所有物理输出和射电输出的文件哈希全部一致 |
| 固定种子 CUDA `proposal-native` | 所有物理输出和射电输出的文件哈希全部一致；summary 非计时字段无差异 |
| 多 shower `-N 2` | 第一个 shower `reused: false`，第二个 `reused: true`，输出均完整关闭 |
| decision-tape replay | 两份 tape 均为 2,088,040 bytes，SHA-256 相同；1559 条记录、0 个 byte-hash mismatch，未出现新分叉 |
| GPU/Hybrid/Native/Radio 单元测试 | 5/5 通过 |
| 格式检查 | `git diff --check` 通过 |

`c8emrt` 回归中有少量显存容量统计不同，是因为测试期间另一生产进程占用 GPU，队列容量按当时可用显存计算；物理文件、过程计数和射电数组不受影响。

## 5. 100 TeV 性能门禁

在用户授权暂停本地 production campaign 后，使用 RTX 4060 Laptop GPU 对重构前后版本进行了五个相同 seed 的交替顺序测试。配置为 100 TeV 垂直质子、`emthin=1e-6`、`gpu-min-batch=4096`、70% 显存上限、native-PROPOSAL、CUDA EM 与 CUDA CoREAS/ZHS，并为两版显式指定同一个 FLUKA worker。

| 指标 | 重构前 | 重构后 |
|---|---:|---:|
| 五事件中位 wall time | 106.72 s | 106.21 s |
| 相对变化 | — | -0.47% |

五个 seed 的 9 类 `.parquet`/`.npz` 物理输出均逐字节一致。第四个 seed 首轮出现 104.57 s 对 157.65 s 的计时离群值；反转顺序复跑时，同一未修改基线变为 148.53 s，重构后为 149.36 s，证明该变化来自机器/GPU运行状态而非支持层。正式判据采用预定的交替五样本中位数，满足“性能退化不超过 3%”门禁。测试数据位于 `/tmp/c8-air-refactor-perf-100tev-ZD1DJm`。

为释放 GPU，外层 production shell 和 Python 调度器保持 `SIGSTOP`；当前未完成的 `batch_021` 子进程被终止，恢复调度器后会从该批次起点重跑，先前完成的批次不受影响。

## 6. 后续维护规则

- 新增应用参数优先放入 `GpuCliOptions`，不要重新堆回主文件。
- 新增运行级 CUDA 资源放入 `CudaRunSession`，单事例数据放入 `CudaEventConfig`。
- 新增 YAML 字段集中在 `CudaShowerReportBuilder`，并用固定种子比较字段顺序和内容。
- 物理过程的构造和 `ProcessSequence` 顺序继续留在主文件中，便于审计 CPU/CUDA 使用的物理配置。
- `c8_ice_cascade.cpp` 未在本阶段修改；后续复用支持层时应另做独立回归。
