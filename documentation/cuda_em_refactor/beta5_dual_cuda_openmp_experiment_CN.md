# beta5：CUDA/OpenMP 单二进制实验

## 目标与边界

同一 ELF 可执行文件同时包含两种提前编译的 EM 和 CoREAS/ZHS 实例，启动时选择其中一种；不是包装器再启动其他后端文件，也不是一个 shower 同时由 CPU 多核与 GPU 执行 EM。

本实验不替代已有独立后端，不修改强子模型、PROPOSAL 表或随机流算法。2026-09-06 已完成本机 Release 编译、安装和下述小样本回归；本页不是大样本生产验收声明。

## 调用结构

```text
c8_air_shower --kokkos-execution cuda|openmp
  └─ KokkosEmBackend（主机端选择与转发）
       ├─ KokkosBackendInstanceImpl<Kokkos::Cuda>
       └─ KokkosBackendInstanceImpl<Kokkos::OpenMP>
            └─ 同一份 KokkosBackendInstance.inl + EM/radio kernel 模板
```

- `src/accelerator/em/kokkos/KokkosBackendInstance.inl`：迁移后的共享实现，原计算方法的函数体保持一致。
- `KokkosCudaBackendInstance.cpp` / `KokkosOpenMPBackendInstance.cpp`：独立编译单元分别实例化同一模板，链接进同一个库和应用。
- `KokkosBackendSelection.cpp`：只校验请求，不初始化设备；不支持的选择明确失败。
- `KokkosRuntime.cpp`：进程级初始化与释放；GPU 模式把 OpenMP host 实例限定为 1 线程。OpenMP 模式使用指定线程数。
- `KokkosRadioAccumulator.hpp`：按实际 `ExecutionSpace` 而不是整个构建的宏选择线程私有累积或 GPU tiling。波形公式不变。
- `KokkosMemorySpace.hpp`：OpenMP 的主机暂存仍使用普通 `HostSpace`；不会因为组合 Kokkos 的默认设备为 CUDA，就给 OpenMP 粒子队列分配 CUDA pinned memory。

## 构建

先按主 README 配置 `corsika_venv`、CUDA toolkit、只读补丁依赖和 FLUKA。实验新增组合类型 `CUDA_OPENMP`，仍锁定 Kokkos 4.7.03。

```bash
conda activate corsika_venv
cd ~/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5
C8_BUILD_JOBS=1 bash tools/build_kokkos_dual.sh -DWITH_FLUKA=ON
```

默认 profile 为 RTX 40 系列 `cuda-openmp-ada89`。其他 NVIDIA 架构复制该 profile 并修改 `kokkos_architecture`，通过 `C8_KOKKOS_DUAL_PROFILE=/absolute/path/to/profile` 指定。

```text
corsika-21cma-kokkos-beta5/
  corsika8_kokkos_beta5/
  build/cuda-openmp/deps/          组合依赖的 CMake 描述
  build/cuda-openmp/               独立实验构建
  install/cuda-openmp/bin/c8_air_shower
```

不设置 `CORSIKA_LAUNCHER_PREFIX`，不替换 `install/bin/c8_air_shower`。

## 使用同一个程序

以下两个命令使用完全相同的可执行文件。`-f` 输出路径必须尚不存在。

```bash
conda activate corsika_venv
# 从源码目录执行。显式使用本安装的数据，避免继承其他版本的 CORSIKA_DATA。
export CORSIKA_DATA="$(cd ../install/cuda-openmp/share/corsika/data && pwd)"

../install/cuda-openmp/bin/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos --kokkos-execution cuda \
  --pdg 22 -E 1 -s 26090611 --antenna-file /path/to/antennas.txt -f photon_cuda

../install/cuda-openmp/bin/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos --kokkos-execution openmp \
  --kokkos-num-threads 2 \
  --pdg 22 -E 1 -s 26090611 --antenna-file /path/to/antennas.txt -f photon_openmp
```

组合版省略 `--kokkos-execution` 时默认 CUDA；默认 `--em-backend` 仍是标量 PROPOSAL，故加速运行需明确指定上述 EM/radio 参数。组合版本不改变物理参数默认值。

## 重要限制

1. Kokkos 全局初始化会初始化已编译的 CUDA runtime。实验版在 OpenMP 模式也要求可见 NVIDIA 设备；`CUDA_VISIBLE_DEVICES` 不能为空。无驱动/无 GPU 时明确失败，请用独立 OpenMP 版。
2. “只使用 OpenMP 计算”不等于“完全不加载 CUDA 库或创建 CUDA context”。独立 OpenMP 版才具有后一种部署隔离。
3. 只在进程启动时选择后端，不在多个 shower 之间动态切换后端。`-N 2` 在同一选定后端内复用 workspace。
4. 当前只增加 CUDA/OpenMP 组合；不提供 CUDA/HIP/SYCL 多 GPU runtime 混合二进制。
5. 不宣称合并后自动更快。必须分别对照原 CUDA、原 OpenMP；跨硬件、跨后端不承诺完整 shower 逐位相等。

## 本地验收方法

`validation/accelerator/run_dual_backend_acceptance.py` 对两条新路径分别与重构前对应独立二进制比较：1 GeV 光子、10 GeV 质子、相同 seed，每进程 `-N 2`，完整 CoREAS/ZHS。逐文件比较 Parquet 表和 NPZ 数组，同时检查 backend metadata、完成标记和第二例复用。低能测试显式固定波前容量；它不是生产性能测试。

测试设有 3 GiB RSS 上限、3 GiB 系统剩余可用内存门限和进程超时；只会终止由测试自身启动的进程组。大样本 Fe 生产进程不在测试停止范围内。

## 本地验收结果（2026-09-06）

环境：`corsika_venv`、GCC 12.4、CUDA 12.6、Kokkos 4.7.03、RTX 4060 Laptop GPU。依赖编译、应用编译均限制为单个编译任务；未停止、替换当前 Fe 生产任务。

| 检查 | 结果 |
|---|---|
| Release 编译和完整安装 | 通过；独立安装到 `install/cuda-openmp` |
| 单 ELF / 动态库 | ELF 同时链接 `libcudart`、`libcuda`、`libgomp`；不是二次启动其他程序的脚本 |
| CUDA / OpenMP 1、2、4 线程探针 | 通过；百万元素 scan 校验和 `366504574976` 一致，队列顺序及回传完全一致 |
| 不匹配后端、GPU 多 host 线程、隐藏 GPU | 明确拒绝；共 8 项运行时测试通过 |
| 1 GeV 光子、10 GeV 质子；各自 CUDA/OpenMP 对独立旧版 | 4 组比较全部通过；每进程 `-N 2`，共 8 个进程 / 16 个 shower |
| Parquet、NPZ 及 YAML 物理计数 | 每组 79 项检查通过，共 316 项；排除明确列出的耗时字段 |
| CoREAS / ZHS | 光子两种后端均有非零有限波形，与对应旧版数组完全相同 |
| 原生 PROPOSAL | 表哈希、辅助哈希、反解计数一致；74/74 原生缓存命中 |
| `-N 2` 生命周期 | 第二例 `reused=true`，均正常完成；没有测试内存门禁触发 |
| 独立 OpenMP 源码兼容性 | 使用原 OpenMP-only 依赖和 GCC 对新实例、转发层、runtime 做语法编译，通过；未覆盖旧二进制 |
| 原独立启动器及源码边界 | 23 项启动器测试、Kokkos-only 边界检查及 `git diff --check` 通过 |

上述比较是**新 CUDA 对原 CUDA、新 OpenMP 对原 OpenMP**，不是要求 CUDA 和 OpenMP 两套完整 shower 彼此逐位相同。这里的低能质子在默认观测时间窗中恰好可以出现全零波形，旧版也是如此；非零射电回归由光子样本验证，未为使测试通过而修改时间窗或物理参数。

安装后测试的组合版进程峰值 RSS 为约 544–610 MiB。OpenMP 组合进程相比独立 OpenMP 多出约 100 MiB 的加载/初始化开销，与同时初始化 CUDA 的部署方式有关。这只验证短程 `-N 2`，不能据此声称所有高能或长期多事件运行均无内存问题。正在运行的 Fe 任务也在使用 GPU，本次秒级耗时不用于受控性能比较。

最终安装二进制 SHA-256：

```text
382d5b9c4b6e18b7108e5d45dbd2bef5a4f32a3b7b04798f161501e3dbdd0053
```

验收数据和最终报告：

```text
/mnt/d/CorsikaData/corsika_validation_results/
  diagnostic_beta5_dual_cuda_openmp_installed_20260906_v2/
    acceptance_review.json
```

该目录保留原始 `acceptance.json`：初次 YAML 比較把强子 `final_state_time_ms` 也当作必须相同，因计时差异报失败。明确排除此时间字段后，用同一批完整数据重新审计，结果保存在 `acceptance_review.json`；物理数据、二进制均未因这项测试判据修正而改变。重审命令：

```bash
python validation/accelerator/run_dual_backend_acceptance.py \
  --recheck /path/to/acceptance.json
```

构建日志与 8 项运行时门禁报告保存在 `build/cuda-openmp/`。尚未进行组合版 500/2000 例统计验收、高能性能验收、长程内存验收或其他 GPU 架构验收；原独立后端继续作为生产路径。本轮未推送远端仓库。
