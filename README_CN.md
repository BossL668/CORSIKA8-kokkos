# CORSIKA 8 Kokkos beta5：安装与使用

[English](README.md)

本程序模拟大气粒子级联及 CoREAS/ZHS 射电信号，通过 Kokkos 让同一套电磁输运与射电算法面向多核 CPU 或不同 GPU 编译。这是基于 CORSIKA 8 的独立软件分支，不是官方发布版；安装不需要旧 beta 项目、已有构建目录或手工生成的 `.c8emrt` 表。

本文面向初学者，以 **Ubuntu 24.04 x86-64、Bash、空 Conda 环境**为例：先完成 OpenMP 安装和一个小事件，再按需增加 CUDA。HIP/SYCL 工具链单独说明，不能将配置支持等同于硬件验收通过。

## 1. 功能与构建逻辑

- Kokkos 处理光子、电子、正电子及当前支持的 muon 输运，包含传播、能损、相互作用、cut、thinning、profile 和 CoREAS/ZHS 累积。
- PROPOSAL 原生样条经只读接口导出到 Kokkos 数据结构，保留版本、介质、cut、能区和哈希检查；不要求用户先运行完整制表工具。
- 强子模型、衰变及不支持的末态仍走 CPU，保留单核标量 PROPOSAL 对照。本文科研配置采用 SIBYLL-2.3d + 用户合法安装的 FLUKA。
- 一次 shower 选择 OpenMP **或** GPU，不叠加多核强子调度，不做 MPI/多卡粒子栈分发。不是所有物理过程都已 GPU 化。

```text
               同一套源码：c8_air_shower.cpp
                + Kokkos 电磁输运 / 射电算法
                             |
               选择编译器、后端和目标架构
                             |
        +--------------+--------------+--------------+
        |              |              |              |
      OpenMP          CUDA           HIP            SYCL
     多核 CPU      NVIDIA GPU      AMD GPU       受支持的设备
        |              |              |              |
  build/openmp    build/cuda      build/hip      build/sycl
        |              |              |              |
 install/openmp  install/cuda    install/hip    install/sycl
        +--------------+--------------+--------------+
                             |
                 install/bin/c8_air_shower
                 读清单 → 检查 → 选择 → exec
                             |
                    运行一个后端二进制
```

这是“同一源码、独立二进制、统一入口”：编译时生成后端实例，运行时选择已经安装的程序，不临时编译。OpenMP 与 GPU 可同时安装，但每个 GPU 程序使用 Kokkos Serial host，OpenMP 程序不需要 GPU runtime。这是本项目的组织方式，不是声称 Kokkos 只能这样使用。

```text
corsika-21cma-kokkos-beta5/
├── corsika8_kokkos_beta5/  源码、依赖配方、示例、文档、测试
├── build/                编译产物，不是第二份项目
│   ├── openmp/deps/       OpenMP 的 Conan/CMake 工具链
│   ├── cuda/deps/         CUDA 的独立工具链
│   └── ...               其他后端或审计记录
└── install/
    ├── bin/c8_air_shower  统一启动器
    ├── openmp/           CPU 二进制、库、模型数据、安装清单
    └── cuda/             GPU 对应产物；HIP/SYCL 按需增加
```

`deps` 描述依赖位置与编译选项，实际 Conan 包在用户缓存；它不是物理插值表。配置好的 `CMakeCache.txt` 含绝对路径，不能跨后端或跨机器复制使用。所有后端使用同一个应用源码文件。

## 2. 从空环境开始

需要可写的用户目录、联网下载依赖的条件和兼容的 C++17/Fortran 编译器。建议为源码、缓存及构建预留 30–50 GiB 空间，这是规划建议而非严格最低值；CUDA、多后端及模拟数据另需空间。首次编译使用 1 个任务，确认内存余量后再增加。WSL 用 `free -h` 检查 Linux 的实际可用内存，不要直接按电脑物理内存估算。

### 2.1 系统开发工具

以下只安装主机开发工具，不安装 GPU 驱动：

```bash
sudo apt-get update
sudo apt-get install -y build-essential gfortran git curl ca-certificates \
  pkg-config autoconf automake libtool bison flex patch unzip bzip2 xz-utils
```

无 `sudo` 请管理员提供这些工具或加载服务器已有编译器模块。不要自行替换共享服务器驱动。教程使用系统 GCC/G++/GFortran，不再混装另一套 Conda C++ 编译器；其他 Linux 发行版使用对应包管理器。

### 2.2 安装 Conda（已有则跳过）

下面用 Miniforge 提供 `conda`。命令仅适用 Linux x86-64，目标安装目录必须不存在；已有 Miniconda/Anaconda 用户直接使用自己的安装。[Miniforge 官方说明](https://github.com/conda-forge/miniforge#install)

```bash
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
curl -fLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
curl -fLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh.sha256
sha256sum -c Miniforge3-Linux-x86_64.sh.sha256
# 只有校验成功后才执行：
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

### 2.3 新建环境与检查

以下选择项目已使用的工具/粒子数据库版本，不要求已有 `corsika_venv`：

```bash
conda create -n corsika_venv --override-channels -c conda-forge \
  python=3.11 pip cmake=3.31 ninja numpy=1.26.4 pyyaml lxml
conda activate corsika_venv
python -m pip install "conan==2.11.0" "particle==0.25.1" "hepunits==2.4.1"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran

python -c 'import particle, hepunits, numpy, yaml; print("Python dependencies OK")'
cmake --version
conan --version
gcc --version
g++ --version
gfortran --version
```

`particle` 和 `hepunits` 用于**构建时生成粒子属性代码**，不是可省略的绘图包。粒子数据库版本会影响物理常数，需要记录。分析结果时可额外安装 `pandas pyarrow scipy matplotlib`，不必先运行分析程序才能编译。

已有同名环境不要覆盖，先检查或换环境名。每次新终端都需 `source` 实际 Conda 的 `etc/profile.d/conda.sh`，然后 `conda activate corsika_venv`。

## 3. 获取源码，配置 Conan

**beta5 源码**位于[项目私有仓库的 `kokkos-beta5` 分支](https://github.com/BossL668/corsika8-gpu-hybrid/tree/kokkos-beta5)。先取得仓库访问权限并配置 GitHub 认证。beta2、beta4 保留在其他分支；不要把仓库默认分支当作 beta5。

```bash
mkdir -p "$HOME/corsika-21cma-kokkos-beta5"
cd "$HOME/corsika-21cma-kokkos-beta5"
git clone --branch kokkos-beta5 --single-branch --recurse-submodules \
  https://github.com/BossL668/corsika8-gpu-hybrid.git corsika8_kokkos_beta5
cd corsika8_kokkos_beta5
test -f CMakeLists.txt
test -f conanfile.py
test -f modules/data/CMakeLists.txt
test -f resources/GeoMag/IGRF14.COF
git submodule status --recursive
```

模型数据和可选 CONEX 源码以锁定版本的上游子模块提供，不将大文件重复放进 beta5 仓库；子模块使用绝对地址，可从 GitHub 正常定位。GitHub 自动生成的源码 ZIP **不包含子模块内容**。若采用离线归档，应向维护者索取包含两个子模块、依赖配方、资源及示例的完整包。源码仓库不包含本机的 build/install、自动生成的缓存或模拟数据集。

Conan 负责 C++ 依赖，CMake 负责装配应用。保持第 2 节环境和编译器设置，在源码目录执行：

```bash
conan profile detect
conan profile show -pr:h default -pr:b default
conan remote list
conan export third_party/conan/cubicinterpolation
conan export third_party/conan/proposal
conan export dependencies/kokkos
```

核对 profile：`compiler=gcc`、主版本号与 `g++ --version` 一致，C++ 标准 17（`gnu17` 也可）、`compiler.libcxx=libstdc++11`。`detect` 只是候选配置，需要核对；已有 `default` 不要随意 `--force` 覆盖。生产复现需保存 profile 和依赖锁文件。[Conan profile 文档](https://docs.conan.io/2/reference/commands/profile.html)

默认 ConanCenter 应已启用；**只有远程列表为空时**才添加：

```bash
conan remote add conancenter https://center2.conan.io
```

三个 `export` 注册配方，依赖编译由下一节自动完成。项目使用 `kokkos/4.7.03@c8gpu/stable`、`proposal/7.6.2@c8gpu/stable`、`cubicinterpolation/0.1.5@c8gpu/stable`；后两者带只读样条导出补丁，不能与无补丁库/旧头文件混用。完整二进制图仍受间接依赖版本影响。

## 4. 准备物理模型

### 4.1 SIBYLL + FLUKA 科研配置

FLUKA 有独立许可，不通过源码或 Conan 分发。先按 [FLUKA 官方安装说明](https://fluka.cern/documentation/installation)取得匹配系统/Fortran 编译器的授权包并完整安装，保留其运行数据，不是只复制一个库。假设安装到 `~/fluka`：

```bash
export FLUPRO="$HOME/fluka"
test -r "$FLUPRO/libflukahp.a"
```

检查失败先修正安装或路径。高能 SIBYLL-2.3d 源码已在项目内；Pythia 8.315 和 TAUOLA 1.1.8 默认由 CMake 下载固定来源并编译。**空环境不用先准备 `build/external`，不要复制旧机器的 Pythia/TAUOLA 绝对路径。** 首次需访问 ConanCenter、GitHub、Pythia GitLab 和 CERN 下载站点。

### 4.2 暂无 FLUKA 许可

仅为学习安装，可以在下节显式改用 `-DWITH_FLUKA=OFF`，低能模型变为 UrQMD。它不等价于 SIBYLL+FLUKA，不能混入后者的科研验收样本。不要将更换物理模型视为单纯的编译/性能调整。

## 5. 构建 OpenMP（不需要 GPU）

当前在源码目录，已激活环境、检查 `FLUPRO`：

```bash
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp -DWITH_FLUKA=ON
```

脚本顺序执行：

```text
conan install --build=missing
    → build/openmp/deps/conan_toolchain.cmake
    → CMake Release 配置
    → 编译模型、应用和测试
    → 安装 install/openmp 和统一入口 install/bin/c8_air_shower
```

首次下载和编译可能较久，不是单个 shower 的时间。`C8_BUILD_JOBS=1` 限制编译和 Conan 构建并行，不限制模拟线程数；内存充足可改为 2 或 4，不建议初次 `-j128`。脚本遇错停止，先看第一条错误，不将“出现 build 目录”当作成功。

构建成功后，回到项目根目录检查：

```bash
cd ..
install/bin/c8_air_shower --list-backends
install/bin/c8_air_shower --backend openmp --check-backends
install/bin/c8_air_shower --backend openmp -- --help
```

OpenMP 应显示 `installed: true`，探针 `available: true`、`gpu=false`、`openmp=true`。清单只读安装信息，探针验证基础操作，不是完整 shower 验收。

## 6. 首次完整运行与输出

仍在项目根目录，使用随源码提供的三个演示天线，不依赖私人数据：

```bash
mkdir -p "$HOME/CorsikaData"
install/bin/c8_air_shower --backend openmp --kokkos-num-threads 4 \
  -p 22 -E 10 -s 12345 -f "$HOME/CorsikaData/first_photon_openmp" \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

这是 10 GeV 垂直光子、一个 shower；统一入口默认开启 Kokkos EM 和 Kokkos CoREAS/ZHS。**`-f` 最终目录必须不存在**，只创建上级目录。重复运行请换名称，不删除旧科研数据。低能信号可能很弱或个别天线无有效脉冲，此例用于安装检查而非性能评估。

天线每行 `north_m west_m up_m`，允许 `#` 开头注释。当前空气应用将第三列放在 `EarthRadius + up_m`，所以它是相对参考地球半径的高度，**不是离当地地面的高度**；示例使用默认观测高度 `2680.444195`。没有有效天线且 `--ring 0` 时可能没有射电观测点，退出成功并不代表验证了射电输出。

首次配置 PROPOSAL 仍可生成自身缓存，再导出样条复用。辅助缓存默认在 `~/.cache/corsika8/gpu-em-aux` 或 XDG 目录；PROPOSAL 自身缓存来自 `corsika_data("PROPOSAL")`，对应目录需要可写。`--gpu-aux-cache-dir` 只改辅助缓存，不改所有缓存。无需手工制表不等于没有插值/缓存，也不自动扩大介质、几何支持范围。

检查 `profile/`、`energyloss/`、`particles/`、`CoREAS/`、`ZHS/`、`gpu_em/`、`simulation_timing/`；内容可能位于各自 shower 子目录。确认日志无异常、summary 完成、Parquet 可读，再扩大样本。CPU/OpenMP/GPU 同 seed 不保证整个 shower 逐位相同，需要逐过程与统计验收。

## 7. 增加 NVIDIA CUDA 后端

**只在获准使用 GPU 的机器执行本节**；无 GPU 用户完成上节即可。驱动让系统访问显卡，Toolkit 提供 `nvcc` 和开发库；`nvidia-smi` 的 CUDA 版本不是已安装的 `nvcc` 版本。驱动由管理员配置；WSL2 使用 Windows NVIDIA 驱动，不能在 WSL 安装 Linux 显示驱动。[NVIDIA WSL 说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

已有可用 Toolkit 不重复安装。空开发环境可在 Conda 中安装 CUDA 12.6.3，示例与 Ubuntu 24.04 的 GCC 13 系列配套；其他组合需核对 [CUDA 12.6 编译器支持及安装说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html)。

```bash
conda activate corsika_venv
conda install --override-channels -c nvidia/label/cuda-12.6.3 -c conda-forge cuda
nvcc --version
nvidia-smi
```

这不安装/修复系统驱动。不要用只有运行库的老 `cudatoolkit` 包代替完整开发工具。已有系统 Toolkit 时将其 `bin` 放入 PATH，别混用不同版本头文件和库。

```bash
cd "$HOME/corsika-21cma-kokkos-beta5/corsika8_kokkos_beta5"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda -DWITH_FLUKA=ON
```

脚本通过 `nvidia-smi` 查询 compute capability，选择对应 profile：

| 目标示例 | profile | CUDA 架构 |
|---|---|---|
| Turing | `cuda-turing75` | 75 |
| A100 | `cuda-ampere80` | 80 |
| Ampere 8.6 | `cuda-ampere86` | 86 |
| RTX 4060 等 Ada 8.9 | `cuda-ada89` | 89 |
| H100 | `cuda-hopper90` | 90 |

混合/未知架构不猜测；已知目标可显式选择，例如 A100：

```bash
C8_KOKKOS_PROFILE=cuda-ampere80 C8_BUILD_JOBS=1 \
  bash tools/build_kokkos.sh cuda -DWITH_FLUKA=ON
```

这只绕过架构探测，不修复驱动。`CORSIKA_ENABLE_CUDA=ON` 不是 beta5 开关，应选择 Kokkos CUDA。也可以 `bash tools/build_kokkos.sh openmp,cuda -DWITH_FLUKA=ON` 顺序构建安装两套。

```bash
cd ..
install/bin/c8_air_shower --backend cuda --check-backends
install/bin/c8_air_shower --backend cuda \
  -p 22 -E 10 -s 12345 -f "$HOME/CorsikaData/first_photon_cuda" \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

这两条命令会访问 GPU。GPU 模式不要加 `--kokkos-num-threads 16`。默认显存预算为初始化时可用显存的 70% 上限，不要求小事件占满。

## 8. 最简命令、参数与正式模拟

项目根目录的最简加速调用：

```bash
install/bin/c8_air_shower --backend cuda -p 2212 -E 1000 \
  -f "$HOME/CorsikaData/proton_1TeV_cuda" \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

换成 `--backend openmp --kokkos-num-threads 16` 就用多核 CPU，其他物理参数不变。标量对照用 OpenMP 安装的程序，但不开 Kokkos EM：

```bash
install/bin/c8_air_shower --backend openmp \
  --em-backend proposal --radio-backend cpu \
  -p 2212 -E 1000 -s 12345 -f "$HOME/CorsikaData/proton_1TeV_scalar" \
  --antenna-file corsika8_kokkos_beta5/examples/beta5/antennas_minimal_nwu.txt
```

`proposal/cpu` 不是 OpenMP 加速。直接调用 `install/<backend>/bin/c8_air_shower` 的默认值也仍是标量；统一入口才补上 `--em-backend kokkos --radio-backend kokkos`。加速物理源只有 `proposal-native`，不用另填 YAML 或 `.c8emrt`。

| 参数 | 含义 / 默认 |
|---|---|
| `--backend openmp/cuda/hip/sycl` | 选择已安装后端；省略为 `auto` |
| `-p`、`-E` | 初级 PDG、总能量 GeV；光子 22、电子 11、质子 2212 |
| `-z`、`-a` | 天顶角/方位角，度；默认 0/0 |
| `-s`、`-N` | seed / 同进程顺序事件数；默认自动 seed / 1 |
| `--emthin`、`--max-weight` | 默认 `1e-6` / `0`（自动计算权重上限） |
| `--emcut` | 默认动能 cut `0.0005` GeV |
| `--geomagnetic-model`、`--geomagnetic-year` | 默认 IGRF14 / 2027 |
| `--kokkos-num-threads N` | OpenMP 线程数；GPU 不接受大于 1 |
| `--kokkos-device N` | GPU 编号，不是多卡并行 |
| `--gpu-min-batch` | 默认 4096，不是 shower 数量 |
| `--gpu-memory-fraction` | 默认 0.70，预算上限非填充目标 |
| `--gpu-resident-batch-limit` | 默认 0 自动容量；固定容量用于回放/诊断 |
| `--radio-sampling-rate-ghz` | 默认 1 GHz |
| `--radio-window-duration-ns`、`--radio-pretrigger-ns` | 默认 400 ns / 10 ns |
| `--kokkos-tuning-cache`、`--kokkos-require-tuning` | 可选调优缓存；require 模式下不匹配即失败 |

`auto` 探测已装 GPU，不可用时提示并选 OpenMP；多个可用 GPU 后端要求明确选择，不保证自动找最快者。显式 OpenMP 不探测 GPU；显式后端失败不换后端重跑。`--list-backends` 只读清单，`--check-backends` 和 `--dry-run` 会运行探针。跨机器迁移的是源码，不是让 NVIDIA 二进制直接在 AMD 上运行。

正式实验固定 seed 清单、天线、模型、能量、角度、cut、窗口。高能/倾斜事件不能假设默认 400 ns 足够：先画脉冲，扩大窗口检查边缘能量及重叠区收敛，再冻结设置。低能自动最大权重可能小于 1，非零 `--emthin` 不一定实际薄化。

`C8_BUILD_JOBS` 控制编译；`--kokkos-num-threads` 控制 OpenMP 计算；`-N` 控制同进程顺序事件数，三者不同。多进程 × 线程数不超过分配资源并留内存余量；长批次先验证 `-N 1`、`-N 2` 和 RSS 趋势，一次探针不证明长期稳定性。

## 9. HIP / SYCL：目标机器上的进阶构建

OpenMP/CUDA 有已编译测试的配置；HIP/SYCL 仍需目标机器的编译、逐过程 oracle、shower 和射电验收。AMD/Intel CPU 不等于拥有 HIP/SYCL 可用 GPU。

HIP：按 [ROCm 安装说明](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html)检查 GPU/系统兼容性并装好开发工具与驱动，检查 `hipcc --version`。`hip-vega90a` 是 MI200/VEGA90A 模板，不是任意 Radeon 配置。先匹配其架构和 Conan Clang 版本，在源码目录执行：

```bash
C8_KOKKOS_PROFILE=hip-vega90a bash tools/build_kokkos.sh hip -DWITH_FLUKA=ON
```

SYCL：按 [oneAPI 安装说明](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/overview.html)安装 DPC++ 和目标设备 runtime/驱动，初始化工具链，检查 `icpx --version`、`sycl-ls`。`sycl-intel-pvc` 是 PVC/oneAPI 模板，不适合原样套到任意 Intel 核显：

```bash
C8_KOKKOS_PROFILE=sycl-intel-pvc bash tools/build_kokkos.sh sycl -DWITH_FLUKA=ON
```

自定义 profile 可用绝对路径传给 `C8_KOKKOS_PROFILE`，OpenMP 的覆盖变量为 `C8_KOKKOS_OPENMP_PROFILE`。每次调用最多一种 GPU 工具链，可附加 OpenMP；不同调用的安装可共存。工具链版本、ABI、依赖一起匹配，不关闭门禁或启用 fast-math 绕过错误。

## 10. 验证、维护与常见错误

项目根目录先列测试，再运行对应后端：

```bash
ctest --test-dir build/openmp -N
OMP_NUM_THREADS=4 ctest --test-dir build/openmp \
  -R 'testKokkos|Beta5' --output-on-failure
# 仅在允许使用 GPU 时：
ctest --test-dir build/cuda -R 'testKokkos|Beta5' --output-on-failure
```

测试可能生成缓存并耗时；通过基础测试不是所有能量/介质的生产验收。同 seed 的结果还取决于 RNG 域、物理修订和容量，记录二进制 SHA-256、参数、依赖和 `gpu_em` 元数据，不能仅凭“beta5”文件名认定相同版本。

| 现象 | 检查/处理 |
|---|---|
| `conda` 找不到 | source 实际 Conda 安装下的 `etc/profile.d/conda.sh` |
| 缺少 `particle` / `hepunits` | 激活环境、检查 python，按 2.3 节安装，不用系统 pip |
| 找不到 `@c8gpu/stable` | 检查三个配方已导出到当前 Conan 缓存 |
| profile/编译器/ABI 不匹配 | 核对 CC/CXX/FC 与 profile，换工具链后独立重配 |
| FLUKA 找不到或链接失败 | 检查授权安装、FLUPRO、Fortran 兼容性，不静默换模型 |
| 下载失败 | 检查日志 URL/网络；离线需受控依赖归档，不能关闭哈希验证 |
| 编译被 Killed / WSL 重启 | `C8_BUILD_JOBS=1`，检查 `free -h` 和磁盘空间 |
| nvcc 找不到 | 需要开发工具而不只是驱动，检查 PATH |
| NVML driver/library mismatch | 管理员协调驱动；指定架构不能修复运行时 |
| 清单存在但探针失败 | 检查环境、设备权限、动态库，不随意拼接陌生库 |
| 输出目录已存在 | 换 `-f`，不要删除科研数据为测试让路 |
| 射电为空/截断 | 检查有效天线数、高度、时间窗口；低能本身也可能无信号 |
| 显存不到 70% | 是上限，取决于粒子量、工作区和当时可用显存 |

更新源码后重跑对应 `build_kokkos.sh` 才会编译安装，编辑 README 不改变二进制。新机器重新构建，不复制旧 CMakeCache。复用预构建 Pythia/TAUOLA 仅限同版本/ABI 的高级部署，不是从零安装前提。

实现/审计另见[目录边界与验收记录](documentation/BETA5_EXTRACTION_CN.md)、[统一入口审查](documentation/BETA5_PORTABLE_ENTRY_AUDIT_CN.md)。核心许可见 [LICENSE](LICENSE)，模型与依赖许可独立；不向仓库提交 FLUKA、构建产物、缓存和私人数据。
