# beta5 三种构建与 kernel 入口复核（2026-09-07）

## 范围

针对 ZHS 修复后的 CUDA extended-lambda 入口故障，检查当前源码的
`CUDA`、`OPENMP`、`CUDA_OPENMP` 三种 Release 构建。
双后端指同一个程序在启动时选择 CUDA 或 OpenMP，不是把同一 shower
同时交给 CPU 多核和 GPU。

本轮重新编译第一方 Kokkos 后端库、应用和相关测试；复用既有 Conan
依赖及 PROPOSAL 缓存，不重新制表。不更新 `install/`，不推送仓库，
不修改物理公式、随机流或生产参数，也不重启生产 campaign。

## 最终结论

在本机 `corsika_venv`、GCC 12.4、NVCC 12.6.85、Kokkos 4.7.03
及 RTX 4060 Laptop GPU 上，三种构建及所测运行路径全部通过，
未再次复现未注册的 lambda 入口或 `cudaErrorInvalidDeviceFunction`。

| 构建目录 | Release 后端库 / 相关应用 | 相关 CTest | 实际 shower 路径 |
|---|---|---|---|
| `build/cuda` | 通过 | 15/15 | CUDA：4 photon + 2 proton，通过 |
| `build/openmp` | 通过 | 15/15 | OpenMP：4 photon + 2 proton，通过，屏蔽 GPU |
| `build/cuda-openmp` | 通过 | 17/17 | 同一二进制分别选择 CUDA、OpenMP，各 6 个，通过 |

合计 24 个小型 shower。8 组同执行空间比较的 104 个 Parquet / NPZ
文件数值均相同；没有新增的物理输出或非豁免 metadata 差异。
这不表示 CPU 多核与 GPU 的独立 shower 被要求跨后端逐位相同：
比较的是同一执行空间在独立 / 双后端构建以及修复后基线之间的结果。

三种新构建的 `--help` 参数区相同。`ldd` 确认独立 OpenMP 不链接
CUDA runtime，独立 CUDA 不链接 OpenMP runtime；双后端链接两者。
当前独立 `install/cuda` 和 `install/openmp` 仍是此前版本，帮助中
尚无后加入的 `--kokkos-execution`，本轮没有把它们当作新构建使用。

构建进程树采样 RSS 峰值约 2.61 GiB；8 次多 shower 运行的采样
RSS 峰值为 510–627 MiB，运行时主机可用内存均高于 8.9 GiB。
这些短测试验证初始化 / 复用 / 结束流程，不替代长时内存泄漏测试，
也不用于比较后端加速比。

审核开始与结束的 99 个 accelerator 文件 SHA-256 一致：本轮
保留已完成的具名 functor / ZHS 修复，后续改动只有测试和诊断文档。

## 为什么不能只检查“编译成功”

先前的 CUDA-only 故障发生在内核启动时：host 请求的扩展 lambda
入口与实际注册的入口不一致。独立实验已将它定位到双后端适配后的
条件编译 / `if constexpr` / 函数内多个 extended lambda 组合；
旧、新 ZHS 公式都能复现同一故障。GPU tiled 入口现使用具名
`RadioTileProjection<Coreas, Zhs>` functor，不再依赖这一匿名入口。

因此，本次同时要求重新编译、实际启动直接与分块内核、选择双后端的
两个分支，以及 `-N > 1` 的完整 shower；不能用旧二进制的成功运行
代替当前源码验收。

## 源码检查

| 层次 | 主要文件 | 检查重点 |
|---|---|---|
| 构建及实例化 | `src/accelerator/em/kokkos/CMakeLists.txt`、`KokkosEmBackend.cpp` | 独立后端依赖边界；双后端分别创建 CUDA / OpenMP 实例 |
| runtime 与调优 | `src/accelerator/em/kokkos/KokkosRuntime.cpp` | 实际执行空间、线程门禁、GPU 可见性、调优中的两套 launch |
| EM 执行层 | `corsika/accelerator/em/kokkos/` | policy 与 View 的执行 / 内存空间一致；输运、末态、驻留波前内核 |
| scan 与分桶 | `KokkosCompositeExclusiveScan.hpp`、`KokkosWavefrontBucketing.hpp` | CUB 与 portable 分支、64 位计数、零长度、稳定顺序及 workspace 复用 |
| profile | `KokkosProfileAccumulator.hpp` | 固定点累计、显式 policy、host/device atomic 分支 |
| 射电 | `corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp` | direct / tiled；CoREAS-only、ZHS-only、fused；定点 / 浮点；reset 后复用 |

没有在此次检查的生产 EM / radio 执行代码中发现硬编码
`Kokkos::DefaultExecutionSpace` 导致双后端误选设备的问题；这些模板
使用显式 `ExecutionSpace`。剩余 `if constexpr` 不一概视为错误：
重点检查是否存在非类型依赖的被丢弃分支与匿名 GPU 入口组合，
并以实际实例化、launch 和结果检查验证已覆盖的路径。

## 本轮补齐的测试问题

1. 原 radio oracle 在双后端构建中使用默认执行空间，只测到 CUDA。
   新增 `testKokkosScalarRadioAlignmentOpenMP`，以同一份测试显式
   实例化 OpenMP；驱动中的两个默认 policy 同时改为显式 policy，
   避免 CUDA 默认空间访问 OpenMP 的 HostSpace 输入。
2. 增加 CoREAS-only、ZHS-only 对 fused 输出的比较，及浮点 tiled
   场景。断言实际进入 tiled 或 direct 路径，避免测试意外绕过目标入口。
3. `testKokkosCpuTransportAlignment` 旧夹具无条件请求两个 host
   线程，与 GPU 单线程调度门禁冲突。现在 OpenMP 使用两个线程，
   GPU 使用一个；没有放宽程序门禁。
4. `testKokkosCpuDepositionAlignment` 旧夹具直接调用 writer 的
   `startOfShower/endOfShower`，没有建立新版磁盘 summary 所需的
   library。现使用独占临时目录并调用完整 library 生命周期；
   CPU oracle、binning、能量和误差阈值均未改变。

首次完整 CUDA 测试为 13/15，OpenMP 为 14/15；上述旧夹具导致的
失败日志保留。修复后重新编译并重跑，不能将初次失败隐藏为通过。

## 验收数据及比较方法

运行记录存放在：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  acceptance_beta5_build_matrix_20260907_v2/
```

每条执行路径运行 4 个 1 GeV 光子和 2 个 10 GeV 质子，seed 为
`26090729`，垂直入射，`emthin=1e-6`，IGRF14 / 2027，81 个
天线，CoREAS 与 ZHS 均开启。低能、受限 batch 用于入口与生命周期
测试，不作为高能性能基准。显式保留 FLUKA 与 SIBYLL 2.3d。

检查内容：

- 所有 shower 完整、表和统计有效、射电 fixed-point overflow 为零；
- 首次 backend 初始化、后续 shower 复用；
- 两套射电数组尺寸为 `N × 81 × 400 × 3` 且全部有限；
- 独立构建与此前已修复 ZHS 的同执行空间基线逐项比较；
- 双后端各分支与相应独立构建逐项比较；
- Parquet 数值表与 NPZ 数组严格相同；YAML 只排除列明的计时、
  项目 revision 及由计时推导的 retrospective worker-oracle batch
  诊断。该 oracle 不派发实际粒子；实际物理计数不排除。

独立 OpenMP 的运行与最终回归额外设置 `CUDA_VISIBLE_DEVICES=""`。
双后端另行验证 CUDA、OpenMP 1/2/4 线程及非法设备 / 线程拒绝门禁。
注意：当前 Kokkos 双后端初始化仍要求可用 NVIDIA 设备；无 GPU
机器应使用独立 OpenMP 构建。这是已知运行约束，不是本次修复内容。

## 重现命令

在 `corsika_venv` 内、源码根目录运行：

```bash
python validation/accelerator/run_backend_build_matrix.py \
  --output /path/to/new-build-audit \
  --force-backend-recompile

python validation/accelerator/run_backend_entry_acceptance.py \
  --mode cuda-openmp --output /path/to/new-dual-runtime-audit
```

第一条对三个既有独立 build 目录逐一配置，清理的只是后端目标的
编译产物，随后 `--parallel 1` 重编译并运行相关 CTest；不清理依赖、
源码、安装或用户数据。第二条对指定构建实际执行 probe、合成调优
及多 shower 测试，调优缓存只写到本次诊断目录，不替换生产配置。
独立运行改用 `--mode cuda` 或 `--mode openmp`。

构建与运行均保留命令、返回码及采样内存数据，并设主机可用内存
保护门限。本轮不使用高能长任务来判断内核能否启动。

## 结果边界

本轮针对本机编译器 / NVIDIA 设备的构建、入口、执行空间、输出和
生命周期回归，不替代全部物理过程的百万点 oracle 或大样本统计
验收，也不能外推为 HIP、SYCL、任意 NVCC 版本均已通过。

本次数据目录中的最终入口为 `FINAL_SUMMARY.json`。最初的
`summary.json` 与 `cuda_ctest.log`、`openmp_ctest.log` 保留旧夹具
失败记录；修复后的最终日志分别为 `cuda_ctest_retry.log`、
`openmp_ctest_retry.log`、`cuda-openmp_ctest.log`。
`runtime_*/` 保存实际事件，`*_comparison.json` 保存逐文件检查，
`build_dependency_audit.json` 保存三种新二进制的 SHA-256、链接依赖、
帮助文本和实际编译 flags。
