# beta5 跨设备统一入口：实现与审查

日期：2026-09-05。范围是构建、运行时、设备选择、共享算法边界、安装和
用户入口；不是重新证明所有粒子过程的物理正确性，也不是 HIP/SYCL 硬件验收。
beta4 未改动，未启动新生产样本，未推送远端。

## 1. 结论与设计选择

本版采用 **同一源码、独立后端二进制、统一启动入口**。一次
`build_kokkos.sh openmp,cuda` 可以构建并安装 CPU 与 NVIDIA 两个版本。
AMD/Intel 对应 `openmp,hip`、`openmp,sycl`，仍需要目标平台工具链。

这不是把同一 shower 拆到 CPU 和 GPU，也不是一个二进制内含两个算法
实例。`KokkosEmBackend.cpp` 和 `KokkosRuntime.cpp` 仍通过编译宏选择一个
execution space；模板 kernel 共用一套源码，内存空间随实例类型选择。
启动器只负责在 shower 之前选择程序，之后 `exec`，不参与粒子调度。

Kokkos 官方允许一个设备后端和一个主机并行后端共存，例如 CUDA/OpenMP。
官方没有规定应用必须拆成多个二进制。我们保留独立构建是为了无 GPU
服务器不依赖 GPU 驱动/运行库，而不是宣称合并编译一定更慢。
参考：[后端配置](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)、
[初始化](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Initialization.html)。

## 2. 端到端审查结果

| 层级 | 对应位置 | 结论 / 处理 |
|---|---|---|
| 构建依赖 | `conanfile.py`、`dependencies/kokkos/` | 每个包一个选定执行后端；补充 SYCL 基础 `-fsycl` 编译/链接标志。HIP/SYCL 完整工具链仍需实机检查 |
| 项目构建 | `tools/build_kokkos.sh` | 同次调用可顺序构建 OpenMP 和一个 GPU 版本；依赖构建和 CMake 均默认单任务；不并行争抢 WSL 内存 |
| 硬件架构 | `dependencies/kokkos/profiles/` | CUDA 自动识别 75/80/86/89/90；未知/混合架构失败并要求显式 profile。HIP/SYCL 不再无提示套用别的设备示例 |
| 运行初始化 | `src/accelerator/em/kokkos/KokkosRuntime.cpp` | OpenMP 无 GPU 初始化；GPU 程序用 Serial host。没有做单二进制双实例改造 |
| 后端/会话 | `KokkosEmBackend.cpp`、`KokkosEmRunSession.cpp` | 同源模板实例化、原生 PROPOSAL 导出和缓存复用，本轮不调整其语义 |
| 粒子过程 | `corsika/accelerator/em/common/`、`detail/`、`kokkos/` | Philox、物理函数、cut/thinning、波前/队列与 fallback 继续共享；没有新增另一套公式 |
| 射电 | `corsika/accelerator/radio/kokkos/` | 与 EM 使用同一个选定执行后端；CoREAS/ZHS 公式和累积方式不变 |
| 设备专用代码 | `KokkosWavefrontBucketing.hpp`、`KokkosCompositeExclusiveScan.hpp`、profile atomics | CUB/CUDA 快路径有编译/类型门禁；其他执行后端用 Kokkos 路径，不把独立 native CUDA 后端加回来 |
| 内存 | `KokkosResidentCapacity.hpp`、memory budget/projection | 不更改原预算/容量策略。SYCL 使用 total global memory 估算预算，不是实际空闲显存，不能宣称四后端内存语义完全相同 |
| 应用/CLI | `applications/c8_air_shower.cpp`、应用支持层 | 本轮不修改物理参数、ProcessSequence、模型构造或输出 schema；入口仅补默认 kokkos/kokkos 参数 |
| 安装 | `applications/CMakeLists.txt`、`tools/CMakeLists.txt` | 安装 probe、版本/后端清单和统一入口；应用模型库改用相对 `$ORIGIN` 查找 |
| 信号/错误 | `tools/c8_air_shower_launcher.py` | 参数数组原样传递，无 shell；exec 交接 PID/信号/退出码，启动后不重试、不换后端 |

## 3. 选择规则

- `--backend openmp`：不探测 GPU；`--kokkos-num-threads N` 指定 EM/射电线程数。
- `--backend cuda|hip|sycl`：只探测该后端；不可用立即报错，不换 CPU。
- `--backend auto`（默认）：探测已安装 GPU 后端，一个可用就选择；多个
  可用要求显式指定；没有可用 GPU 时提示原因后尝试 OpenMP。
- auto 下 `--kokkos-num-threads >1` 明确选择 OpenMP。显式 device index
  不能被忽略后换 CPU。GPU 模式限制 host 线程环境为 1。
- 指定 `proposal/cpu` 时保留标量算法，不伪装成 OpenMP 多核 EM。
- 探针只用 1024 项 scan + 4096 项队列测试；超时 30 s，单 host 线程，
  禁止 core dump。探针成功不是 shower 物理验收或性能保证。
- `--list-backends` 不访问硬件；`--check-backends` 会初始化探针中的执行空间；
  `--dry-run` 只打印选择及实参，不创建 shower。

## 4. 本轮测试

- Python 启动器/构建助手共 **23 项测试通过**：涵盖 CPU-only、CUDA/HIP/SYCL
  分发协议、路径空格、重复/冲突参数、错误清单、探针超时、设备失败、
  多 GPU 歧义、明确 CPU 时不访问 GPU、退出码与 SIGTERM、成对构建隔离。
  其中 HIP/SYCL 使用假探针，只证明调度协议，不冒充实机结果。
- CUDA 与 OpenMP 的 Release 应用、探针、调优工具及 FLUKA worker 已在新
  `build/cuda`、`build/openmp` 完成构建，并安装到相应前缀。
- 真实安装路径 CUDA 探针：RTX 4060 Laptop，`gpu=true/openmp=false`，
  scan/队列顺序/回传检查通过。
- 真实安装路径 OpenMP 探针：`gpu=false/openmp=true`，单线程小探针通过。
  `ldd` 确认 OpenMP 应用链接 libgomp，不链接 CUDA/HIP/SYCL 运行库；
  SIBYLL 和 FLUKA 保留。
- 同样的实际 argv[0] 下比较应用 `--help`；统一入口不改应用选项。相对路径
  与绝对路径调用时，Usage 的程序名会不同，不应将其当作 CLI 变化。
- 构建设置 CPUQuota=100%、MemoryMax=4 GiB、MemoryHigh=3 GiB、无 swap；
  安装数据也计入 cgroup 文件缓存。没有更改 IDE 设置或停止其主线程。

日志位于项目 `build/audit/launcher-*`；CTest 的 `Beta5UnifiedLauncher`
可以重跑全部启动器测试。构建助手的单元测试使用假编译命令，不代替真实
编译；实际 CUDA/OpenMP 构建记录独立保存。

## 5. 尚不能作出的承诺

1. **HIP/SYCL 尚未实机编译和跑完整 shower。** oneAPI 的 AOT/RDC/原子支持
   等编译选项及设备 FP64 能力需要与选定 Kokkos 4.7.03 配置联合验收。
2. 相同源码不等于一个二进制能在所有机器上运行。CUDA 架构、CPU ISA、
   驱动、Conda/Fortran 运行库、FLUKA 许可均须在目标机器匹配。
3. `$ORIGIN` 仅改善项目共享库定位；模型数据和外部 Pythia 配置仍可能包含
   构建路径。不能仅复制 install 就承诺完全可迁移，推荐迁移源码后重建。
4. beta4 的既有统计差异、beta5 提取后逐过程/固定种子/多 shower 的物理
   验收不由启动器测试替代。没有在本轮宣称通过新 500/2000 例验收。
5. auto 不做性能调优、不保证最优设备，也不承诺跨设备逐位一致。正式
   campaign 应显式记录 backend、seed、capacity 和 tuning 信息。
