# WSL / RTX 5070 安装反馈与修复

本次核查基于生产分支 `97336691359b0f1f9d9ee7fee88f410a92721e81`。
反馈环境为 WSL2 / Ubuntu 24.04、RTX 5070、系统 GCC/G++/GFortran 13、
CUDA 12.8.1、Conan 2.11、Kokkos 4.7.03，采用独立 CUDA 构建。
下面区分仓库问题、文档不足和环境问题，避免把一次安装成功等同于完整物理验收。

## 1. 确认的仓库问题

| 问题 | 根因 | 修复 |
|---|---|---|
| Miniforge 校验下载 404 | 不带版本号的安装包别名没有对应的同名校验文件 | 中英文 README 使用同一 release 的带版本安装包及 `.sha256`，仍要求校验通过 |
| RTX 5070 架构无法识别 | 缺少计算能力 12.0 的脚本分支、profile 和 CMake 架构映射 | 增加 `cuda-blackwell120`、实验组合 profile、`BLACKWELL120 → 120`；明确 CUDA 12.8+ |
| `RadioTrackParticle` 缺少 `addSecondary`，测试目标编译失败 | `if constexpr (...) return false;` 只控制这一条语句，不会丢弃后面的模板实例化代码 | 将整个停止正电子光子生成实现放入 `else` 分支；默认关闭时不要求只读适配器支持生成次级粒子 |
| CUDA 清单出现 `architecture: NONE` | Conan 的 `cache_variables` 在 preset 中，但 shell helper 只传 `CMAKE_TOOLCHAIN_FILE`，没有读取 preset 中的架构选择 | 在 toolchain 导出包身份，CMake 同步架构和编译设置；显式冲突报错，修复旧 `NONE/native` 占位值，不仅修改 JSON |

架构配置的具体依据是 [Conan CMakeToolchain 文档](https://docs.conan.io/2/reference/tools/cmake/cmaketoolchain.html)：
`cache_variables` 由 preset / Conan CMake helper 应用，`variables` 才写入 toolchain。
仓库里的 Kokkos 依赖配方本身使用 Conan CMake helper；需要修复的是根项目的直接 toolchain 配置路径。

`BLACKWELL120` 仅对应计算能力 12.0，不把计算能力 10.0 的设备错误映射为 120。
CUDA 要求参见 [NVIDIA CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。
新增组合 profile 不代表已完成 RTX 5070 的 CUDA/OpenMP 组合运行验收。

## 2. 文档补充，不擅自修改物理或第三方包

- **FLUKA**：检查授权 Linux 包、Fortran 兼容性、库文件、链接脚本及动态依赖；
  CMake 必须报告启用 FLUKA。可以显式传入 `C8_FLUKALIB`，但运行仍需完整的
  `FLUPRO` 数据目录。反馈中的包标签与程序报告版本不同，尚不足以确定安装包或
  链接库的具体问题，不硬编码“正确版本”。本项目优先读取库中的 `flkavr`，
  老库才使用提取的版本信息。
- **GPU 状态**：区分后端已安装、设备探测成功和实际事件推进。
  `host_execution: serial` 是独立 CUDA 构建的主机执行空间，不是 GPU 被关闭。
  应结合退出码、事件输出、CUDA EM 实际步数与 backend wall time 判断。
- **提示信息**：可选 tuning 文件缺失使用默认值；低能 thinning 提示不是安装失败。
  不完整能量账本的 residual 不应作为完整能量守恒判据，零值的 kernel/transfer
  计时也不能单独说明设备未工作。示例增加 `pipefail` 和日志保存。
- **不是仓库缺陷**：WSL1/WSL2 选择、网络超时、未检测 Conan profile、
  首次编译间接依赖、仅安装 CUDA 后未安装 HIP/SYCL/OpenMP，需按各自配置处理。
  `nvidia-smi` 的 CUDA 版本与 `nvcc` 的 Toolkit 版本含义不同，数值不同不必然冲突。

停止正电子默认开关、在飞湮灭、cut、薄化、PROPOSAL 表及采样算法均保持不变。
不通过关闭测试或给只读粒子伪造 `addSecondary` 来掩盖编译错误。

## 3. 验证与边界

- 11 项离线安装测试：自动/显式架构选择、混合和未知架构拒绝猜测、OpenMP 无 GPU
  路径、Conan 生成配置、CMake cache 与实际 `backend.json` 一致性、显式冲突检测。
- 从本次源码重新生成 OpenMP 依赖并编译：
  `testStoppedPositron`、`testKokkosCpuDepositionAlignment`；与安装配置测试一起通过。
  新增无 `addSecondary` 的只读适配器用例，覆盖 cut 下的电子、正电子和光子。
- 本机 RTX 4060 Laptop / CUDA 12.6 / GCC 13：重新生成 CUDA 依赖并编译，
  `testKokkosCpuDepositionAlignment` 与安装配置测试均通过；设备探针的
  `scan_valid`、`queue_stable_order`、`queue_roundtrip_exact` 均为 true。
  配置仅传 Conan toolchain 和后端、不手填架构，实际得到 `ADA89` 清单及
  `compute_89/sm_89` 编译参数。
- 官方 Miniforge `26.7.2-0` 安装包及校验文件已实际下载并通过 SHA256 校验，
  未执行安装、未覆盖本机 Conda。
- 本机授权 FLUKA 包的临时链接与动态库检查通过。其包装脚本自动识别编译器时
  曾错误选择 `g77`，所以教程对匹配的 GNU 64 位包显式指定 GFortran；
  不修改、重新分发 FLUKA 源码或库，也不推断其他安装包必然有同样问题。

可重复执行：

```bash
python tests/tools/test_install_configuration.py -v
ctest --test-dir ../build/openmp \
  -R '^(testInstallationConfiguration|testStoppedPositron|testKokkosCpuDepositionAlignment)$' \
  --output-on-failure
```

实际 C++ 回归使用本机现有依赖缓存以及同版本 Pythia/TAUOLA/FLUKA，
不是另一台全新机器的全依赖下载验收。RTX 5070 的实机小事件来自安装者反馈，
不能冒充本机独立复测，也不构成全能区/全模型统计物理验收。

更新后重新运行对应构建脚本，使 Conan、CMake、目标和安装清单一起更新。
更换编译器或 CUDA Toolkit 时使用新构建目录；不要手工把旧 JSON 中的 `NONE`
替换为目标架构来伪装修复。
