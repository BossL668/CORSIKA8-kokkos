# CORSIKA8-kokkos

[English installation guide](README.md) · [后端与场景说明](documentation/kokkos.rst)

基于 CORSIKA 8 的大气簇射与射电模拟程序，可选择 Kokkos CPU/GPU 后端。
`c8_air_shower` 应用通过库接口接入输运、调度、环境转换和输出。
本仓库是独立研究项目，不是 CORSIKA 官方发行版。

## 最新更新（2026-10-29）

- CPU 前缀的自动拆分阈值改为按 `--prefix-workers` 缩放，与 GPU 数量独立。
  显式指定正值 `--prefix-split-energy` 仍优先；默认串行前缀行为不变。
- 在 `CONFIG.json` 中记录实际使用的 CPU 拆分能量与 GPU 交接能量。

## 更新（2026-10-28）

- 重新组织代码：公共头文件、内联实现和编译源文件分开存放，应用接入与计算模块统一整理到 Kokkos 库中。
- 调整应用接入方式：通过统一适配接口选择后端，保留原生过程序列及 `EAS.run()`；安装脚本同步适配新的目录和原生二进制入口。

## 目录与应用

```text
applications/c8_air_shower.cpp         大气簇射入口
corsika/kokkos/application/cascade/    共享应用接入
  atmosphere/                         大气场景
corsika/kokkos/{environment,em,radio,scheduling}/  公共库模块
corsika/detail/kokkos/                对应的 .inl 实现
src/kokkos/                           编译源文件
externals/conan/                      依赖配方及后端配置
```

应用构建环境、模型和天线，`CascadeExtension` 将这些对象接入所选后端。
逐粒子计算和调度留在库中。具体职责见 [Kokkos 目录说明](corsika/kokkos/README.md)。

## 从零编译与安装

下面按 Ubuntu 22.04/24.04 或 WSL2 的空环境编写，所有命令在 Bash 中执行。
先准备系统工具和 FLUKA，再选择 **Miniconda/Miniforge 或 venv**，最后选择
**OpenMP 或 CUDA**。无需先编译 OpenMP 才能编译 CUDA。
**安装脚本默认开启 FLUKA**，缺少依赖时会提前报错，不会静默更换物理模型。

需要 C++17 编译器（GCC 11 及以上）、兼容的 GFortran、CMake 3.24–3.x、Conan 2
和构建用 Python 环境。Kokkos 固定为 4.7.03。源码、依赖缓存和构建应预留数十 GiB，
多个后端和模拟数据需要更多空间。WSL 除了检查 `df -h` 和 `free -h`，还要检查
存放虚拟磁盘的 Windows 分区；Linux 显示空间充足不代表 C 盘充足。
首次编译保持 **`C8_BUILD_JOBS=1`**。

### 1. 系统工具和 FLUKA（先准备）

Ubuntu 系统工具：

```bash
sudo apt-get update
sudo apt-get install -y build-essential gfortran git curl ca-certificates \
  python3-venv python3-dev pkg-config autoconf automake libtool bison flex \
  patch unzip bzip2 xz-utils
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FC=/usr/bin/gfortran
```

没有 sudo 权限时，使用服务器提供的编译器模块或请管理员安装。
本教程即使使用 Conda，也沿用系统的 GCC/G++/GFortran，不再混装另一套 Conda 编译器。
上面命令不会安装显卡驱动。

**FLUKA 需要独立许可，脚本不会代为下载。**
请先从 [FLUKA collaboration 官方安装说明](https://www.fluka.eu/Fluka/www/html/fluka.php?id=ins_run&mm2=3)
取得与 Linux/GFortran 匹配的包并完成安装。当前 C8 接口需要包含 `ndmhep.o` 的
`libflukahp.a`；另一套 CERN FLUKA 的库不能直接替换。必须保留完整安装和运行数据，
不能只复制静态库。假设已经安装在 `$HOME/opt/fluka`：

```bash
export FLUPRO="$HOME/opt/fluka"   # 改成实际的完整安装目录
export FLUFOR=gfortran
export GFORFLU="$FC"
test -r "$FLUPRO/libflukahp.a"
ar t "$FLUPRO/libflukahp.a" | grep -x ndmhep.o
# 如果安装包提供了独立可执行文件：
ldd "$FLUPRO/flukahp"
```

先解决缺文件、`not found` 或 Fortran 版本不兼容，再进行 C8 编译。
运行 C8 时也保留 `FLUPRO`。可用 `-DC8_FLUKALIB=/绝对路径/libflukahp.a`
显式指定链接库，但它不替代运行数据。脚本默认传入 `-DWITH_FLUKA=ON`，并在
下载大量依赖前检查库是否存在、是否符合接口要求。

若仅想学习安装、暂时不用 FLUKA，需要在后面的**检查命令和构建命令**中均显式加
`-DWITH_FLUKA=OFF`，此时大气应用的低能强子模型使用 UrQMD。这属于物理配置改变，
不是单纯的编译优化。默认高能模型为 SIBYLL-2.3d，其他模型仍可通过运行选项选择。

### 2. 构建环境二选一

Miniconda/Miniforge 与 Python `venv` 都可以，不要求必须使用 `venv`。
这里仍需要 Python 来运行 Conan、生成粒子属性代码；模拟程序使用原生二进制，
没有 Python 启动器。不要同时激活 Conda 和 venv。

#### 方案 A：Miniconda / 已有 Conda

已有可用 Conda 可跳过安装器。否则，以下适用于 **Linux x86-64**，包括 x86-64 WSL：

```bash
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

将校验值与 [官方安装包目录](https://repo.anaconda.com/miniconda/)中同名文件的
SHA256 对照，确认一致后安装：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
# 若使用安装器默认目录：
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

按安装器提示完成操作，不覆盖已有 Conda 目录。ARM 等平台从
[官方安装指南](https://www.anaconda.com/docs/getting-started/installation)选择对应安装器。
也可使用 [Miniforge](https://github.com/conda-forge/miniforge#install)，只需改为
source 实际 Miniforge 目录中的 `etc/profile.d/conda.sh`。随后创建环境：

```bash
conda create -n corsika8 --override-channels -c conda-forge \
  python=3.11 pip cmake=3.31 ninja numpy=1.26.4 pyyaml lxml
conda activate corsika8
python -m pip install 'conan>=2,<3' particle==0.25.1 hepunits==2.4.1
```

若已有同名环境，请检查后复用或换个名称，不要直接覆盖。

#### 方案 B：Python venv，不安装 Conda

```bash
python3 -m venv "$HOME/venvs/corsika8"
source "$HOME/venvs/corsika8/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'conan>=2,<3' 'cmake>=3.24,<4' ninja \
  particle==0.25.1 hepunits==2.4.1 numpy==1.26.4 pyyaml lxml
```

无论选择哪种方式，检查当前环境：

```bash
python -c 'import particle, hepunits, numpy; print("Build Python packages OK")'
cmake --version
conan --version
"$CXX" --version
"$FC" --version
```

每次打开新终端，重新激活所选环境并设置编译器、FLUKA 变量。
`particle` 和 `hepunits` 是构建依赖；画图或读取 Parquet 的 `matplotlib`、
`pyarrow` 等可以之后再装。

### 3. 下载源码，检查 Conan

```bash
mkdir -p "$HOME/corsika8-workspace"
cd "$HOME/corsika8-workspace"
git clone --recursive --branch CORSIKA8-kokkos \
  https://github.com/BossL668/CORSIKA8-kokkos.git
cd CORSIKA8-kokkos
git submodule status --recursive
conan remote list
```

子模块前出现 `-` 表示尚未初始化，执行 `git submodule update --init --recursive`。
GitHub 自动生成的源码 ZIP 不包含子模块，初学者应使用上述递归克隆方式。
模型数据和 CONEX 子模块都应完整获取。IGRF14 已随源码提供在
`resources/GeoMag/IGRF14.COF`，脚本自动放入并安装到标准 `GeoMag` 数据目录，
不用另外下载。

脚本仅在默认 Conan profile **不存在**时自动执行 `conan profile detect`，
不会覆盖已有配置。也可以先手动检查：

```bash
# 只在默认 profile 不存在时创建：
conan profile path default || conan profile detect
conan profile show -pr:h default -pr:b default
```

确认 profile 中的 GCC 主版本与 `g++ --version` 一致，GCC 使用 `libstdc++11`。
脚本为目标依赖请求 C++17。更换编译器后需要检查旧 profile；不要直接强制覆盖。
只有 remote 列表没有 ConanCenter 时，才添加一次：

```bash
conan remote add conancenter https://center2.conan.io
```

脚本会导出 `externals/conan/` 的依赖配方，构建缺少的 Kokkos、PROPOSAL 等库。
这里“导出配方”不是生成物理表。Pythia、TAUOLA 默认也由 CMake 下载并构建，
不需要复制他人的 `build/external` 或模型绝对路径。首次构建需要访问依赖和模型
下载站点；不要混用旧 PROPOSAL 库和新头文件。

### 4. NVIDIA 开发工具（仅 CUDA 用户）

仅编译 OpenMP 可跳过。本步骤需要可用 NVIDIA 驱动和包含 `nvcc` 的完整开发工具，
分别检查：

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` 显示的 CUDA 版本是驱动能力，不是已安装编译器版本。
有兼容 Toolkit 就直接使用；否则参考
[NVIDIA 安装及编译器兼容说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)。
前面的 Conda 环境可以选择安装 CUDA 12.8.1：

```bash
conda install --override-channels -c nvidia/label/cuda-12.8.1 -c conda-forge cuda
nvcc --version
```

此命令不安装系统驱动。WSL 使用 Windows NVIDIA 驱动，参见
[NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
只有 CUDA 运行库的包不能代替 `nvcc`，不要混用不同 Toolkit 的头文件和库。
V100/V100S 编译需要 CUDA 12.x，CUDA 13 已移除这些目标；RTX 50 需要 CUDA 12.8
或更新版本。脚本会在开始编译前检查 `nvcc` 是否支持目标架构。

### 5. 自动检查、编译和安装

在源码目录运行，保持环境已激活、FLUKA 已配置。
`--check` 只检查前置条件、显示所选 profile 和安装路径，不下载依赖、不编译。
按需要选择后端：

```bash
# 纯 CPU：不需要 NVIDIA 驱动或 CUDA
bash tools/build_kokkos.sh openmp --check
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh openmp

# NVIDIA：自动识别本机 GPU 架构，完成编译和安装
bash tools/build_kokkos.sh cuda --check
C8_BUILD_JOBS=1 bash tools/build_kokkos.sh cuda
```

完整构建命令自动执行：**准备数据 → 安装 Conan 依赖 → CMake 配置 → 编译 → 安装**，
完成后打印二进制和数据目录。无需再单独运行安装命令，也不需要 `sudo make install`。
`C8_BUILD_JOBS` 控制编译并发，不是模拟线程数；确认内存足够后再增加。

自动检测与对应 profile：

| GPU 示例 | 计算能力 | profile |
|---|---|---|
| V100 / V100S | 7.0 | `cuda-volta70` |
| T4 / RTX 20 | 7.5 | `cuda-turing75` |
| A100 | 8.0 | `cuda-ampere80` |
| A10 / RTX 30 | 8.6 | `cuda-ampere86` |
| L20 / RTX 40 | 8.9 | `cuda-ada89` |
| H100 | 9.0 | `cuda-hopper90` |
| RTX 50 | 12.0 | `cuda-blackwell120` |

同机有不同架构 GPU 时，用 `nvidia-smi` 的物理编号或 UUID 指定**编译目标**：

```bash
C8_CUDA_DEVICE=0 bash tools/build_kokkos.sh cuda
```

这个变量不指定之后模拟使用哪张卡，运行时仍使用 `--device`。
编译机没有目标 GPU 时，也可以直接指定 profile，仍需兼容的 CUDA Toolkit：

```bash
C8_KOKKOS_PROFILE=cuda-volta70 bash tools/build_kokkos.sh cuda
```

未知或混合架构不会被随意猜测。附带的 GPU profile 继承默认编译器配置；
`C8_KOKKOS_PROFILE` 也可指向自定义文件，`C8_KOKKOS_OPENMP_PROFILE` 和
`C8_BUILD_PROFILE` 分别覆盖 CPU 目标和构建工具 profile。
HIP/SYCL 需准备对应工具链并显式选 profile；这里的自动检测针对 NVIDIA CUDA。

默认工作区：

```text
corsika8-workspace/
├── CORSIKA8-kokkos/                 源码
├── build/{openmp,cuda}/deps/        各后端独立的依赖描述
└── install/{openmp,cuda}/
    ├── bin/c8_air_shower            原生可执行文件
    └── share/corsika/data/          模型和地磁数据
```

若需放到更大磁盘，在每次脚本调用前设置 `C8_WORKSPACE=/目标工作区`。
Conan 缓存默认在 `~/.conan2`，它不跟随 `C8_WORKSPACE` 移动；需要迁移时，
在创建 profile/安装依赖**之前**设置 `CONAN_HOME=/目标缓存目录`。
修改编译器、GPU 架构或依赖 ABI 时使用新的构建目录，避免旧 CMake 缓存混用。

`openmp,cuda` 会依次安装两个独立后端。可选的 `cuda-openmp` 组合版放在另一前缀，
即使选择 OpenMP 运行也仍需可用 NVIDIA runtime；无 GPU 服务器使用独立 OpenMP 版。
手动 CMake 接入见 [后端指南](documentation/kokkos.rst)，若采用这里的默认科研配置，
手动配置也需传入 `-DWITH_FLUKA=ON`。

## 第一次运行与检查

先跑一个 **10 GeV 光子**检查安装和输出，不作为物理验收或性能测试。
回到工作区父目录，保持环境已激活、`FLUPRO` 已设置，选择自己安装的后端：

```bash
cd "$HOME/corsika8-workspace"   # 自定义了 C8_WORKSPACE 则改成对应路径
mkdir -p results

# OpenMP
export CORSIKA_DATA="$PWD/install/openmp/share/corsika/data"
install/openmp/bin/c8_air_shower --help
install/openmp/bin/c8_air_shower --pdg 22 --energy 10 --nevent 1 \
  --seed 18641 --filename results/smoke-openmp \
  --em-backend kokkos-proposal --kokkos-execution openmp \
  --kokkos-num-threads 1 --radio-backend kokkos

# CUDA
export CORSIKA_DATA="$PWD/install/cuda/share/corsika/data"
install/cuda/bin/c8_air_shower --help
install/cuda/bin/c8_air_shower --pdg 22 --energy 10 --nevent 1 \
  --seed 18641 --filename results/smoke-cuda \
  --em-backend kokkos-proposal --kokkos-execution cuda --device 0 \
  --gpu-memory-fraction 0.70 --radio-backend kokkos
```

最终的 `--filename` 目录不能预先存在，重复运行时换一个新名字。
检查进程是否成功退出、输出目录和日志是否正常。低能射电很弱或没有有效脉冲，
不一定是安装错误。第一次运行会准备原生 PROPOSAL 表及辅助缓存，之后复用兼容数据，
不需要手工制作 `.c8emrt` 文件。
可用 `ldd install/cuda/bin/c8_air_shower`（或 OpenMP 路径）检查是否存在 `not found`。

程序默认 EM/射电为标量 PROPOSAL/CPU，示例明确选择 Kokkos。
将入射参数改成 `--pdg 2212 --energy 1e4` 就是 10 TeV 质子。
cut 处正电子额外湮灭光子默认关闭，飞行中正电子湮灭仍保留。
物理条件及输出参数见 `--help`。

空气程序支持 `--device 0,1,2,3` 让四卡共同推进一个 shower，
`--prefix-workers 16` 启用独立 CPU 并行前缀。对于 `P` 个前缀进程，
自动 CPU 拆分能量为 `primary_energy / (96 * P)`，可用正值
`--prefix-split-energy <GeV>` 覆盖。该值与 GPU 交接能量独立；
改变阈值可能改变子簇分组及后续随机历史。`--gpu-min-batch` 和
`C8_STATIC_FRONTIER_IMPORT_BATCH` 控制派发和静态导入粒度，不改变物理 cut。
后端组合及限制见 [后端说明](documentation/kokkos.rst)。

## 测试、常见问题和更新

```bash
# 在源码目录；自定义了 C8_WORKSPACE 则使用实际构建路径
ctest --test-dir ../build/openmp --output-on-failure -j1
# CUDA 测试使用 CUDA 构建目录，且需要可用 GPU。
```

| 问题 | 检查方式 |
|---|---|
| 找不到 conan/cmake 或 Python 模块 | 重新激活选定 Conda/venv 环境，补齐上述构建依赖。 |
| FLUKA 缺失或不兼容 | 检查 FLUPRO、libflukahp.a、ndmhep.o 与 GFortran 版本。 |
| 找不到 nvcc / 不支持目标架构 | 安装兼容的完整 Toolkit，分别检查 nvcc 与 nvidia-smi。 |
| 同机 GPU 架构不同 | 使用 C8_CUDA_DEVICE 或显式 profile，换目标时使用独立构建目录。 |
| Conan 下载失败 | 检查 remote、网络和代理；修复后重跑相同命令。 |
| 编译 Killed / OOM / 磁盘满 | 先用 C8_BUILD_JOBS=1，检查内存、构建目录及 Conan 缓存空间；WSL 还需看 Windows 分区。 |
| 运行库 not found | 激活原构建环境并检查 ldd，不从其他安装随意复制库。 |
| 缺少模型数据或 IGRF 文件 | 检查子模块及 CORSIKA_DATA 是否指向当前后端的安装数据目录。 |
| 输出目录已经存在 | 换一个新的 --filename，不覆盖已有模拟结果。 |

更新前先用 `git status` 检查本地改动，然后：

```bash
git pull --ff-only
git submodule update --init --recursive
bash tools/build_kokkos.sh openmp   # 或已经安装的其他后端
```

编译器、架构或依赖 ABI 改变时使用新 build/install 目录。
模拟数据保存在源码和构建目录之外。引用与许可证要求见
[英文说明](README.md#corsika-references-and-license)；FLUKA 不随仓库分发。

## 历史更新记录

以下保留以前发布的时间线，描述对应日期的版本；当前安装方法以上文为准。

最近更新（2026-10-23）：

- GPU 射电的分块 ZHS / CoREAS+ZHS 路径先在本地累计诊断计数，再在团队内归约，减少全局原子操作。仅 CoREAS、小轨迹批次及不足一个天线块的路径保留原计数方式，无需新增命令行参数。
- 射电公式、传播、波形累加、物理 cut 和薄化均未改变；CPU/OpenMP 射电保持原实现。

更新（2026-10-20）：

- 修复安装流程：Miniforge 安装包及校验文件使用同一版本的文件名；新增 RTX 50 系列计算能力 12.0 的 `BLACKWELL120` profile，要求 CUDA 12.8 及以上。
- 将 Conan 选择的后端和架构传入直接使用 toolchain 的 CMake 配置，保证编译参数与安装清单一致。
- 修复关闭停止正电子湮灭时，只读粒子适配器仍被要求实现 `addSecondary` 的编译错误；不改变默认开关和物理算法。
- 补充 FLUKA 检查与首次 CUDA 运行的判断方法，问题分类见[安装反馈与修复记录](documentation/installation_feedback_20261020_CN.md)。

更新（2026-10-19）：

- 新增可选的**静态 CPU 并行 prefix**：通过 `--prefix-workers N`，先由独立 CPU 进程推进子簇，再将汇总后的 EM/μ 前沿交给所选 GPU。CPU 进程数与 GPU 数量独立，支持后续单卡或多卡；目前用于 `kokkos-proposal`。默认仍为串行 prefix，不引入动态再分配。
- `--prefix-split-energy` 控制 CPU 分支拆分能量上限，不是物理 cut；可选环境变量 `C8_STATIC_FRONTIER_IMPORT_BATCH` 独立控制静态前沿导入块，不等于 `--gpu-min-batch`，未设置时沿用原规则。
- 交接保留粒子种类（含核与强子）、动能、权重、输运身份和逻辑环境节点；CPU/GPU 后代编号范围互不重叠。合并包括初始 bootstrap 和所有完成的 worker，profile 与带符号射电场不漏计、不重复计数。
- 协调进程与大型物理栈帧隔离，SHA256 文件读取缓冲改用堆内存；新增交接、编号和小栈回归测试。物理模型、CDF 算法、cut、薄化及停止正电子默认策略均未改动。并行拆分会改变后续随机历史，低能事件也可能更慢，不自动开启。

参数和示例见[静态并行前缀说明](documentation/native_multigpu_CN.md#静态-cpu-并行前缀2026-10-19)。

更新（2026-10-15）：

- 大气 Kokkos 路径初始化时导出 C8 环境节点树，包含父子和排除节点关系，复用原 C8 的有序边界搜索逻辑；观测面仍是独立的步长限制。
- 曲线与球面求交采用稳定的四次多项式因子分解，修正平行磁场投影；不引入逐步残差扫描或二分兜底。这不意味着与原 C8 逐位相同的轨迹。
- 逻辑环境节点编号在设备队列、CPU 回退和静态多卡交接中保持。完整环境树使用 V2 前沿格式；旧 V1 前沿需重新生成后才能用于该路径。
- 标量 CPU 与共享 Kokkos 切能路径均**默认关闭停止正电子在能量 cut 处额外生成的双光子**，与原 C8 对齐；**在飞正电子湮灭仍保留**。CPU/GPU 二进制均需重新构建，不同停止湮灭策略的样本不要混用。
- 保留既有原生 CDF 策略：光子产对和 μ 子轫致辐射在加速路径使用原生表求逆；电子/μ 子直接产电子对因缺少条件不对称分能表，由 CPU PROPOSAL 完成末态。不启用产对分能拒绝采样、候选解析 CDF 导数或连续能损逆表优化。

沿用上一版应用接口：

- 空气应用的电磁选择器为 `--em-backend proposal`、`kokkos-proposal`、`kokkos-egs4`；物理后端与执行硬件分别选择。
- 统一 `--device`：一个 GPU 编号走单卡，多个编号由原生 C++ 协调器共同推进同一个 shower。
- 可选 EGS4 C++/Kokkos 后端已按 CORSIKA 模块结构整理；AIR-NTP 表随项目提供并在构建时内嵌，运行时不依赖 C7 安装或表路径参数。
- 主应用通过封装的 bindings 和 session 装配、复用后端；PROPOSAL 和 EGS4 共用 `c8_air_shower` 入口。

极小批次 CPU 回退及长尾重分配也仍是实验，不属于生产默认行为；不能把模块微基准加速等同于整例 shower 加速。

空气程序更新（2026-09-23）：`c8_air_shower` 默认采用原 C8 的正向闭端点纵向计数 `--profile-crossings original-c8`。可显式选择 `forward`（仅正向半开区间）或 `both`（双向半开区间）；CUDA、OpenMP、标量 writer 共用同一规则。**不是禁止向上输运，不影响射电轨迹、cut 或能量沉积。** 旧二进制需重新 build/install，旧双向计数样本勿混用。详见 [计数选项与最简运行](documentation/profile_crossings_CN.md)。

历史说明（2026-09-20）：曾加入能量 cut 处的停止正电子双光子模型；实现仍保留，但从 2026-10-15 起默认关闭，以对齐原 C8。[停止湮灭说明](documentation/stopped_positron_annihilation_20260920_CN.md) 记录的是之前开启时的实现，不代表当前默认策略。
