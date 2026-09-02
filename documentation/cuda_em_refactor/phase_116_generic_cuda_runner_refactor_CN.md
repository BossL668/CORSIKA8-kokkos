# Phase 116：CORSIKA 8 通用 CUDA 运行器重构

## 1. 结论

本阶段把与具体应用无关的 CUDA 物理源生命周期和 `HybridCascade` 接线从
`applications/detail/air_shower_cuda/` 下沉到 CORSIKA 源码目录。空气 shower
仍负责选择模型、建立五层大气 snapshot、定义 process registry、注册 CLI 及生成
原有 YAML；通用层不认识 CLI11、YAML、空气/冰模型、天线、FLUKA 或具体 writer。

这是一次无物理变化重构。CUDA kernel、PROPOSAL calculator/表格、随机数键、
fallback、tracking、thinning、CoREAS/ZHS 算法、`ProcessSequence` 顺序和输出
schema 均未修改。

## 2. 最终分层

```text
corsika/gpu/em/detail/
  CudaEmRunSession.hpp          通用物理源与 backend 生命周期接口
  CudaHybridCascadeRunner.hpp   通用 process-registry/router/cascade 接线

src/gpu/em/
  CudaEmRunSession.cpp          c8emrt/native 门禁、导出、上传与复用实现

applications/detail/air_shower_cuda/
  GpuCliOptions.hpp             纯应用参数数据
  CudaRunSession.{hpp,cpp}      设备信息和既有运行级 YAML 的薄包装
  CudaAirShowerSetup.hpp        五层大气、观测平面、profile/radio 和 registry
  CudaAirShowerRunner.hpp       空气应用 factories 与完成回调
  CudaShowerReport.hpp          既有逐 shower 报告
```

CLI 注册和解析后门禁直接保留在 `c8_air_shower.cpp`，便于从应用入口审计所有
参数。`c8_air_shower.cpp` 为 1339 行；空气适配 runner 为 233 行，满足本轮
1100--1400 行和 200--300 行的目标。

## 3. 通用 session 的行为

`CudaEmRunSession` 在构造时固定物理源。

- `c8emrt`：立即读取文件、验证格式与生成器合同，并拒绝含运行期
  selected-loss CPU fallback 的旧表；每个 shower 再校验 cut、连续表、muon
  能力与能区。
- `proposal-native`：第一个 shower 从当前 PROPOSAL calculator 只读导出原生
  spline，校验共同能区并加载/生成小型 `.c8emaux`；后续 shower 不再导出和
  上传，只调用 `beginShower()`。
- `loadedRateTable()` 只在 `c8emrt` 模式返回表，供应用生成原有 metadata。
- 请求错误物理源、未初始化就访问 backend、表/cut/能区不兼容时均明确失败，
  不自动切换到 CPU 或另一种表。

## 4. 通用 HybridCascade 执行器

`runCudaHybridCascade(...)` 先执行 process registry 门禁，再调用 backend factory。
这样未登记过程仍在 CUDA 初始化或显存分配之前失败，与重构前时机一致。调用方以
factory/callback 提供 fallback、output sink、强制初级动作、强子进程池配置和完成
报告；通用层只负责按原顺序构造 router、`HybridCascade` 并执行 `run()`。

完成回调在 router、fallback、output sink 和 cascade 尚存活时执行，所以原报告
读取的统计量和对象生命周期没有改变。

## 5. 架构门禁

新增 `testGpuGenericRunnerArchitecture`，静态扫描三个通用源文件，禁止直接包含或
使用：

- CLI11 或 YAML；
- `applications`、air-shower 或 ice-cascade 头文件；
- CoREAS/ZHS detector 类型；
- FLUKA 或具体强子应用接口。

因此该层可以由后续 `c8_ice_cascade` 或其他环境适配器复用，但本阶段没有修改
冰应用。

## 6. 验收结果

| 门禁 | 结果 |
|---|---|
| CUDA Release 构建 | 通过；`c8_air_shower` 与 parent-profile 程序均链接成功 |
| CPU-only Release 构建 | 通过；不要求 CUDA toolkit |
| `--help` | 归一化可执行路径和源码行号后与重构前完全一致 |
| 架构边界测试 | 通过 |
| 完整 CTest | 36/36 等效通过；proposal-native 主测试耗时 657.93 s |
| Python validation 测试 | 309/309 通过，耗时 12.77 s |
| 固定种子 CPU `-N 2` | 所有物理文件逐字节一致；仅路径和计时 metadata 不同 |
| 固定种子 proposal-native `-N 2` | profile、粒子、能损、相互作用、CoREAS/ZHS 全部逐字节一致 |
| backend 跨 shower 复用 | 第一个 shower `reused: false`，第二个 `reused: true` |
| decision-tape replay | 57,313 条输运记录与 53,550 条射电记录通过；CoREAS/ZHS parquet 逐字节一致 |
| 失败门禁 | 缺 c8emrt、CUDA radio/CPU EM 不兼容、CPU-only 请求 CUDA、无效设备均明确失败 |
| `git diff --check` | 通过 |

完整 CTest 首轮只报告一个版权头格式失败：原有诊断文件
`validation/gpu_em/inspect_c8emrt_native_aux.cpp` 尚未采用仓库标准头。修正后该测试
单独复测通过；这不涉及运行时或物理代码。

当前源码仓库按设计不追踪大型 `.c8emrt` 文件，`gpu_em_tables/` 只保存说明。
因此本轮没有为结构重构重复生成物理表；`c8emrt` 的表读写和 GPU 查询由完整测试
覆盖，应用级固定种子回归应在取回合同版本 0.18 的既有生产表后补做。不能用旧
合同表冒充当前表，也不能为了结构测试临时生成并宣称为生产表。

## 7. 100 TeV 热缓存性能门禁

在 RTX 4060 Laptop GPU 上使用 6 个相同 shower、相同 seed 和相同旧版 FLUKA
worker 比较重构前后的二进制。配置为垂直 100 TeV 质子、`emthin=1e-6`、默认
maximum-weight、`gpu-min-batch=4096`、70% 显存上限、proposal-native、CUDA EM
与 CUDA CoREAS/ZHS；两轮之间冷却 20 s，测试期间无制表、安装或生产任务并发。

| 指标 | 重构前 | 重构后 | 变化 |
|---|---:|---:|---:|
| 6 shower 总 wall time | 511.48 s | 502.64 s | -1.73% |
| 去掉首个 warm-up 后的中位数 | 81.183 s | 79.917 s | -1.56% |
| 后五个 shower 平均值 | 81.848 s | 81.147 s | -0.86% |

重构后没有性能退化，满足预设的不超过 3% 门禁。两轮均达到约 97%--100% 的稳定
GPU 利用率和约 5.67 GiB 常驻显存。

两轮文件集合完全一致；只有根 config、GPU summary、simulation timing 和总
summary 四个 metadata 文件不同。profile、能量沉积、地面粒子、相互作用、parent
profile、interaction histograms 以及 CoREAS/ZHS parquet 均逐字节一致。GPU
config 也逐字节一致。GPU summary 中除计时外仍可看到四个 FLUKA worker 的 batch
分配和基于实测耗时的预测值不同；这是并行 worker 的运行时调度 provenance，不是
物理输出差异。

原始测试目录为 `/tmp/c8-generic-runner-perf-100tev-clean.r3caAS`，逐 shower 数值
另存为其中的 `performance_summary.json`。

## 8. 运行保护

本地 100 TeV 正式 campaign 的外层 shell 和 Python scheduler 在测试期间保持
`SIGSTOP`。测试结束后两者已恢复；被终止的当前子批次没有被记为完成，调度器先
重新扫描已有 manifest，再由原重试逻辑从同一 seed 范围执行。测试使用独立 build
和 `/tmp` 输出，没有覆盖正式二进制或已有数据。
