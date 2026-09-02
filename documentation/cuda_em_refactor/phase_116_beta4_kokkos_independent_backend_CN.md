# Phase 116：beta4 Kokkos 独立 EM/射电后端

日期：2026-09-03

## 1. 结论

beta4 已建立一条不替换现有生产路径的实验性 Kokkos 后端：同一套物理 device
函数可以编译为 OpenMP 或一个 GPU execution space，但构建和运行严格互斥。

```text
OpenMP build -> Kokkos::OpenMP -> 多核 CPU EM + radio，无 GPU runtime
CUDA build   -> Kokkos::Cuda   -> GPU EM + radio，Serial host，无 OpenMP
HIP build    -> Kokkos::HIP    -> GPU EM + radio，Serial host，无 OpenMP
SYCL build   -> Kokkos::SYCL   -> GPU EM + radio，Serial host，无 OpenMP
```

当前已经在本机完成 OpenMP 和 NVIDIA CUDA 的编译、逐过程/队列/射电门禁和完整
应用冒烟测试。HIP/SYCL 只完成版本锁定 recipe、profile、编译选择及 fail-closed
边界，尚未在对应硬件上测试。因此本阶段是“实现主体完成、生产验收未完成”，不能
把它表述为四类硬件均已验证，也没有改变 native CUDA 的生产基准地位。

## 2. 架构实现

### 2.1 host-wavefront 公共接口

`IAcceleratedEmBackend` 定义 host wavefront 边界，包含 shower 初始化、能力查询、
photon/lepton 常驻级联、profile 和 radio 下载。逐粒子 kernel 中没有虚函数；
concrete backend 在编译期选择 execution space。

`NativeCudaBackendAdapter` 包装原 `CudaEmBackend`，而
`KokkosEmBackend` 通过 PIMPL 隔离 Kokkos/CUDA/HIP/SYCL 头文件。原
`PhysicalCudaEmRouter` 的实现已经泛化为 backend template，并新增
`PhysicalAcceleratedEmRouter` 名称；旧 include/name 保留，避免破坏 native CUDA。

### 2.2 proposal-native 数据

Kokkos session 直接复用 beta4 的 `ProposalNativeTableExporter` 和 `.c8emaux`：

1. 从实际 shower 的 PROPOSAL calculator 只读导出轴、Hermite 系数、过程和组分；
2. 验证版本、介质、cut、过程覆盖、能区和 canonical SHA-256；
3. 扁平化为 `Kokkos::View`；
4. OpenMP 使用 HostSpace，GPU 使用 execution space 的 memory space；
5. 同进程后续 shower 只重置事件状态，复用不变数据和 allocation。

Kokkos 不接受 `.c8emrt`，也不会在失败时静默切换物理源。

### 2.3 粒子波前

photon、electron、positron 和已有表/辅助数据支持的 muon 使用双缓冲 SoA：

- flag + `parallel_scan` + scatter 执行稳定 compaction；
- exclusive scan 分配次级位置；
- history/parent/generation/step 字段随粒子保留；
- Philox key 与线程结束顺序无关；
- selection、球形环境/局部观测面传播、连续损失、Molière/LPM、末态、cut 和
  thinning 使用 beta4 共享 POD 物理函数；
- CPU-only 或异常末态在 wavefront 边界形成带过程/组分/hash 的显式 fallback。

调优 batch 只能改变 workspace/capacity hint，不能覆盖 router 的
`minimumBatchSize()`；后者会改变 host checkpoint、history ID 分配和随机流。

### 2.4 profile 与 CoREAS/ZHS

profile 和 waveform 使用 checked 64-bit fixed-point accumulation。射电核心公式被
提取到共享的 `RadioProjectionStep`，native CUDA 与 Kokkos 调用同一个数值合同。

- GPU 使用 `TeamPolicy` 和 track/observer tiling；
- OpenMP 使用同步 phase 和 tile-local 工作，不建立第二个并发线程池；
- Kokkos EM 轨迹只能进入相同 Kokkos execution space 的 radio accumulator；
- CLI 禁止 `kokkos EM + cpu/native-CUDA radio` 以及反向组合。

## 3. 构建与运行门禁

根 CMake 新增：

```text
CORSIKA_ENABLE_KOKKOS=OFF
CORSIKA_KOKKOS_BACKEND=OPENMP|CUDA|HIP|SYCL
CORSIKA_KOKKOS_ARCHITECTURE=...
```

门禁包括：

- native CUDA 与 Kokkos 不能同时启用；
- Kokkos Conan package 的 backend/architecture 必须与 CMake 一致；
- GPU translation unit 若看到 `KOKKOS_ENABLE_OPENMP` 会编译失败；
- OpenMP translation unit 必须看到 `KOKKOS_ENABLE_OPENMP`；
- GPU 运行拒绝 `--kokkos-num-threads > 1`；
- Kokkos 运行拒绝 FLUKA shower-internal worker pool；
- Kokkos 只接受 `proposal-native` 且 EM/radio 必须同时选择 Kokkos；
- tuning cache 在 `--kokkos-require-tuning` 下缺失或 key 不匹配即失败。

Kokkos 4.7.03 recipe、四套 profile 和四套 lockfile位于
`dependencies/kokkos/`。不同后端必须使用独立 dependency/build/install 目录。

## 4. 调优实现

`c8_kokkos_tune` 对候选进行一次 warm-up 和默认五次计时，并校验输出 checksum：

- OpenMP：batch 256--2048、chunk 1--64、track tile 16--64、observer tile 4--16；
- GPU：batch 1024--8192、team 64--256、track tile 4--16、observer tile 16--64。

缓存 key 包含 backend、设备名、架构、driver/runtime、Kokkos/编译器版本、项目
revision、线程数和 proposal-native hash。脏的开发工作树会在 commit 后附加 tracked
source diff 摘要。当前生产中实际应用的是 radio team/track/observer tile；batch 是
容量 hint，device queue 首版固定为 1。尚未把 chunk 参数接入所有物理 RangePolicy，
所以不能把 microbenchmark 的最优 chunk 直接解释成 shower 加速。

## 5. 已完成证据

### OpenMP

- `testKokkosTuningCache`：写入、哈希、精确匹配、非严格 mismatch、严格 mismatch；
- `testKokkosProposalNativeTable`：常规门禁为 16,384 次 rate query，并另行通过了
  1,000,000 点聚合 rate 抽样；同时包含 8,192 次 interaction
  selection、8,154 次 photon transport、4,096 次 lepton transport、photon/lepton
  final states 和常驻 cascade；最大相对差 `0`；
- stable queue/scan probe：稳定顺序和 POD roundtrip 完全一致；
- 以 1、2、4、8、16 线程分别启动 probe 时，Kokkos execution-space
  concurrency 精确等于请求值，且五次 stable scan/queue 门禁均通过；这只证明线程
  配置有效，不是受控的 scaling 性能结论；
- 10 GeV photon、81 observer 的 Kokkos EM + CoREAS/ZHS 完整应用运行完成；
- tuned/untuned 固定 seed 的 profile、ground particles、interactions、CoREAS、ZHS、
  dEdX 和 production profile 文件 SHA-256 完全相同；
- 动态依赖包含 OpenMP runtime，不包含 CUDA runtime。

### Kokkos CUDA（RTX 4060 Laptop GPU）

- proposal-native 常规 oracle 最大相对差 `2.2703e-14`；1,000,000 点聚合
  rate 抽样的最大相对差为 `3.04007e-14`；
- stable queue/scan 的顺序和 POD roundtrip 完全一致；
- 10 GeV photon、81 observer 的 Kokkos EM + CoREAS/ZHS 完整应用运行完成；
- `-N 2` 连续 shower 运行完成，第二例明确记录 `reused: true`；
- tuned/untuned 固定 seed 的上述七类物理输出 SHA-256 完全相同；
- missing required tuning cache 会终止并将输出标记为 incomplete；
- 动态依赖包含 CUDA runtime/driver，不包含 OpenMP runtime。

### 安装与下游消费

- 安装包配置会在导入 `CORSIKA8KokkosEm` 之前恢复 Kokkos 4.7.03；
- Kokkos-CUDA 安装包还会恢复 CUDA Toolkit 12 runtime target；
- 独立的最小下游工程 `tests/gpu/kokkos_install_consumer` 已从临时安装前缀
  `find_package(corsika)`、链接并运行成功。

### 既有路径回归

- native CUDA `testGpuRadioProjection`：579 checks passed；
- native CUDA proposal-native、10 GeV photon、81 observer 完整应用运行完成；
- CPU-only 干净构建及 scalar PROPOSAL + CPU radio 10 GeV photon 冒烟完成；
- `--help` 保留已有选项并增加 Kokkos 组选项。

以上是开发门禁，不等价于用户计划中的百万点逐列、500/2000 shower 系综和性能
验收。

## 6. 尚未完成的生产门禁

1. 当前一百万点是跨列聚合抽样；仍需每 PID/过程/组分各一百万 `(E,u)` 的完整
   矩阵和 16 ULP 报告；
2. Kokkos-CUDA 对 native CUDA、Kokkos-OpenMP 对 CPU PROPOSAL 的完整 decision-tape
   第一次分叉报告；
3. photon/electron/proton、1/100 TeV、垂直/倾斜的 500 与 2000 例系综；
4. OpenMP 1/2/4/8/16 核 scaling；
5. 100 TeV/1 PeV 热缓存同机性能，验证 native CUDA 回归不超过 3%、Kokkos-CUDA
   与 native CUDA 差距；
6. HIP 与 SYCL 的真实编译器、设备、物理和性能验收；
7. memory fraction 驱动的 portable pool、所有 kernel 的 tuning-policy 接线、
   以及一个/两个 GPU queue 的正式比较。

在这些门禁完成前，推荐关系保持：标量 PROPOSAL 是物理 reference，native CUDA
是 NVIDIA 生产加速后端，Kokkos 是可移植性与 OpenMP scaling 的实验后端。
