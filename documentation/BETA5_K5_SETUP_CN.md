# beta5 在 k5 的部署与 GPU 验证

检查日期：2026-09-05。这里只记录 k5 的部署，不改变本地或 dirac 的生产任务。

## 硬件与驱动

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA A100-PCIE-40GB，compute capability 8.0 |
| CUDA 可见显存 | 42,404,806,656 bytes |
| CPU | 2 × Xeon Gold 6248R，48 物理核 / 96 逻辑线程 |
| RAM | 约 503 GiB |
| CUDA Toolkit | 13.0，nvcc 13.0.88 |
| C/C++/Fortran 编译器 | 系统 GCC 13.3.0 |
| CMake | corsika_venv 中的 3.31.0 |

`nvidia-smi` 当前报告 **NVML driver/library version mismatch**：运行中的
内核模块为 580.159.03，磁盘上的驱动与用户态库为 580.173.02。没有修改驱动、
安装驱动或重启服务器。后续维护时仍应请管理员统一这些版本。

不能仅凭这个监控错误判断 CUDA 计算不可用。本次独立探针通过了 `cuInit`、
设备查询，以及真实 CUDA 内核：分配显存、计算、同步、回传并精确检查
1,048,576 个 double 全部成功。驱动 API 与 CUDA runtime 均报告 13000。
这不表示 NVML 已修复；后端的显存查询使用 CUDA runtime，不依赖 NVML。

beta5 的实际 Kokkos 探针也已通过：`gpu=true`、`openmp=false`，扫描、稳定
队列和数据往返检查均成功。该次查询空闲显存为 9,594,929,152 bytes（约
8.94 GiB）；这只是当时的可用量，不能把显卡标称总容量当作全部空闲容量。
程序中的 70% 预算依据初始化时 **空闲显存** 计算。

## 目录与依赖隔离

```text
~/21CMA/corsika-21cma-kokkos-beta5/
├── corsika8_kokkos_beta5/       同步的 beta5 源码
├── build/
│   ├── cuda/                  k5 本机编译，显式 sm_80
│   ├── external/              Pythia 8.315、TAUOLA 1.1.8
│   ├── tools/                 独立 Conan Python 工具及依赖缓存
│   └── audit/k5_setup/         锁文件、部署脚本、日志和验证数据
└── install/
    ├── bin/c8_air_shower       统一启动器
    └── cuda/                  CUDA + Serial 后端
```

k5 的 `corsika_venv` 保留原样；其原有 Conan Python 依赖不能正常导入，因此
构建使用项目内独立 Conan 2.20.1（系统 Python 3.12），不重装共享虚拟环境。
包装器 `build/audit/k5_setup/conan` 独立设置 `PYTHONPATH` 和 `CONAN_HOME`。

k5 不能直接访问 GitHub，故使用 Conan 官方 `cache save/restore` 从 dirac
传入已验证的 GCC 13 CPU 依赖。压缩包 SHA-256 为：

```text
4097c49d88436ec1b2eb55023871c3be222069d876735366a544e71b87bad77f
```

初次向旧用户缓存恢复时遇到只读文件，已改用独立缓存；没有修改旧缓存权限。
离线依赖图使用归档的 `dirac-full.lock`。Kokkos 固定 4.7.03，配方 revision
为 `5108eb695b39bc35ad36a28fe98c5410`；与当前源码配方的差别仅为后续添加的
SYCL `-fsycl` 传播，不涉及本次 CUDA/OpenMP 分支。PROPOSAL 与
CubicInterpolation 使用锁文件中带只读导出补丁的 7.6.2 / 0.1.5。

**没有复制本地 RTX 4060 的 CUDA 可执行文件**。Kokkos CUDA 库和 beta5
均在 k5 使用 CUDA 13 重新编译，显式选择 `AMPERE80` / `80`。
GPU 使用 Serial host，未构建 OpenMP 混合 shower。

CUDA 13 的 CCCL 3 已移除旧 `cub::TransformInputIterator`。本次在
`KokkosCompositeExclusiveScan.hpp` 加入头文件可用性分支：CUDA 12 保留旧
CUB 路径，CUDA 13 使用 `thrust::transform_iterator`；loader、uint64 前缀
求和、输入输出次序和 stream 均不变。这是工具链兼容修复，不是物理算法更改。
依据 [NVIDIA CCCL 3 迁移说明](https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html)。

FLUKA 使用 k5 已有授权库 `~/fluka/libflukahp.a`，不打包传播；构建启用
`WITH_FLUKA=ON`，高能模型保留默认 SIBYLL-2.3d。Pythia/TAUOLA 使用同版本、
同 GCC 主版本的 CPU 库。已有 PROPOSAL 缓存随项目数据保留，辅助缓存由程序
按哈希查找，不要求手动重新制表。

## 重建与运行

这是 k5 离线部署的复现入口，使用 4 个编译任务，不让高并行编译占满机器：

```bash
cd ~/21CMA/corsika-21cma-kokkos-beta5
bash build/audit/k5_setup/build_k5.sh
```

运行时激活已有环境即可，不必把构建用的 Conan Python 路径加入环境：

```bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
cd ~/21CMA/corsika-21cma-kokkos-beta5
install/bin/c8_air_shower --backend cuda --check-backends

install/bin/c8_air_shower --backend cuda -p 2212 -E 1000 \
  -f /path/to/new_output \
  --antenna-file ~/21CMA/data/antennas_nwu_coordinates_test.txt
```

统一启动器自动添加 Kokkos EM 与 Kokkos radio；物理源为 proposal-native。
显存预算默认 70%，不是保证任何低能事件都占满 70%。指定 `--backend cuda`
失败时不切换 CPU；未指定时启动器用真实计算探针选择可用后端，不使用
`nvidia-smi` 作为唯一判断条件。以上输出路径须替换为尚不存在的实际目录。

本轮安装的是 CUDA + Serial 后端；没有安装 OpenMP、HIP/SYCL 后端，也未承诺
旧原生 CUDA 二进制可用。CUDA 默认 batch 为 4096；尚未生成 A100 专用调优
缓存，使用现有保守默认值，不将这次部署测试称为最优性能测试。

## 验证记录

**Release 构建、安装及 11/11 项测试全部通过。** 包含启动器、旧参数拒绝门禁、
调优缓存、粒子队列、分桶、显存预算投影、复合扫描、profile 和原生表测试。

原生表测试通过 16,384 个率查询、8,192 次反应选择、8,154 次光子传播及
4,096 次轻子传播，并覆盖末态、驻留 wavefront 和射电投影。最大率相对差异
为 `2.2703e-14`。这是部署级测试，不是每个过程百万点的完整物理验收矩阵。
其日志耗时包含为完整 shower 测试让出的暂停时间，不能当作性能基准。

完整事例参数：质子 1 TeV，垂直，seed 2026090501，emthin=1e-6，IGRF14/2027，
81 个天线，Kokkos EM + CoREAS/ZHS，FLUKA + SIBYLL-2.3d；其余物理参数默认。

| 验证运行 | 端到端时间 | 进程峰值 RSS | 结果 |
| --- | ---: | ---: | --- |
| `-N 1` | 14.48 s | 1.83 GiB | 1/1 完整 |
| `-N 2` | 15.79 s（两例合计） | 1.87 GiB | 2/2 完整，第二例复用后端 |

- 三个 shower 均报告 `backend=cuda`、`architecture=sm_80`、`host_threads=1`、
  `gpu=true`、`openmp=false`，原生逆解失败、队列溢出和射电定点溢出均为 0。
- 每次运行的 PROPOSAL 缓存均为 **74/74 命中**，辅助缓存命中，无需重新制表。
- 两次运行各 7 个 Parquet 文件均完整可读，所有浮点数据有限。
- `-N 1` 与 `-N 2` 的首例，同种子的 profile、production profile、能量沉积、
  地面粒子、相互作用记录以及 CoREAS/ZHS 数组 **逐值完全一致**。
- 后端记录峰值显存为 28,761,081,966 bytes（26.79 GiB）。测试时空闲显存已
  恢复到约 39 GiB；与前面的探针时刻不同，不应把瞬时空闲量当成固定值。
- `-N 2` 仅验证基本复用与输出生命周期，不等价于长序列内存稳定性验收。
- 这三个 shower 不是独立的三事件统计样本（首例重复），不用于声称大样本
  物理一致性或相对其他机器的加速比。

安装程序 SHA-256：

```text
8f286f0d2dad81c721de913d3ce88608358e9ef86993d82b845a6e76b9756ad4
```

复查入口均在 `build/audit/k5_setup/`：`build_k5.sh`、`run_gpu_smoke.sh`、
`check_gpu_smoke.py`、`build.log`、`smoke.log`、`proton_1TeV_cuda_N{1,2}.time.txt`。
原始小事例保留在 k5 的同名 `proton_1TeV_cuda_N{1,2}/`，不传回 WSL。
`run_gpu_smoke.sh` 遇到已有目录会退出以避免覆盖；再次运行须使用新的输出目录。
