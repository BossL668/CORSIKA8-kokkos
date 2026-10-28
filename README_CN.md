# CORSIKA8-kokkos

[English installation guide](README.md) · [后端与场景说明](documentation/kokkos.rst)

基于 CORSIKA 8 的粒子级联与射电模拟程序，可选择 Kokkos CPU/GPU 后端。
空气、均匀冰和 firn 应用共用库中的输运、调度、环境转换和输出接口。
本仓库是独立研究项目，不是 CORSIKA 官方发行版。

## 最新更新（2026-10-28）

- 重新组织代码：公共头文件、内联实现和编译源文件分开存放；大气、冰和 firn 场景使用平级目录，共用 Kokkos 库模块。
- 调整应用接入方式：通过统一适配接口选择后端，保留原生过程序列及 `EAS.run()`；安装脚本同步适配新的目录和原生二进制入口。

## 目录与应用

```text
applications/                         三个应用入口
corsika/kokkos/application/cascade/    共享应用接入
  atmosphere/                         大气场景
  ice/                                均匀冰场景
  firn/                               firn 场景
corsika/kokkos/{environment,em,radio,scheduling}/  公共库模块
corsika/detail/kokkos/                对应的 .inl 实现
src/kokkos/                           编译源文件
externals/conan/                      依赖配方及后端配置
```

应用构建环境、模型和天线，`CascadeExtension` 将这些对象接入所选后端。
逐粒子计算和调度留在库中。具体职责见 [Kokkos 目录说明](corsika/kokkos/README.md)。

## 安装

完整的系统工具、Python 虚拟环境、Conan 2、C++17/Fortran 和 CUDA 准备步骤见
[英文安装说明](README.md#install-on-linux)。首次编译保持 `C8_BUILD_JOBS=1`，
不要按 CPU 核数直接开满编译进程。Kokkos 固定为 4.7.03。

```bash
git clone --recursive --branch CORSIKA8-kokkos \
  https://github.com/BossL668/CORSIKA8-kokkos.git
cd CORSIKA8-kokkos
# 二选一；各后端拥有独立构建和安装目录
bash tools/build_kokkos.sh openmp
bash tools/build_kokkos.sh cuda
# V100/V100S 可显式指定；L20 使用 cuda-ada89
C8_KOKKOS_PROFILE=cuda-volta70 bash tools/build_kokkos.sh cuda
```

IGRF14 系数已随源码提供在 `resources/GeoMag/IGRF14.COF`，会自动安装到标准
`GeoMag` 数据目录，不需要另行下载或单独安装。

脚本会导出本仓库的依赖配方，准备标准 C8 数据目录（含 IGRF14），并安装
`c8_air_shower`、`c8_ice_cascade`、`c8_firn_cascade` 三个二进制。
现有 Conan 编译器配置不会被脚本覆盖。`C8_WORKSPACE` 可指定 `build/` 与
`install/` 的父目录；默认使用源码的父目录。多个后端顺序安装可用 `openmp,cuda`。
需要 FLUKA 时先设置合法安装的 `FLUPRO`，再在构建命令末尾加 `-DWITH_FLUKA=ON`。

## 运行

在工作区父目录使用对应后端的二进制，不经过 Python 启动器：

```bash
export CORSIKA_DATA="$PWD/install/cuda/share/corsika/data"
install/cuda/bin/c8_air_shower --pdg 2212 --energy 1e4 --nevent 1 \
  --seed 18641 --filename shower-cuda \
  --em-backend kokkos-proposal --kokkos-execution cuda --device 0 \
  --gpu-memory-fraction 0.90 --radio-backend kokkos
```

默认是原生标量 PROPOSAL/CPU 射电，使用 Kokkos 时显式设置上述后端选项。
原生物理表由程序自动准备和复用。cut 处正电子额外湮灭光子默认关闭，飞行中
正电子湮灭仍保留。空气应用可用 `--device 0,1,2,3` 和 `--prefix-workers 16`
分别选择四卡输运和 CPU 并行前缀；调度参数与物理 cut 分开。
冰和 firn 的介质、天线、射电模式及支持范围见 [后端说明](documentation/kokkos.rst)。

```bash
ctest --test-dir ../build/openmp --output-on-failure -j1
```

更新此目录布局时使用新的 build/install 目录，避免旧缓存中的绝对路径和依赖 ABI
混用。引用与许可证要求见 [README](README.md#corsika-references-and-license)。

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
