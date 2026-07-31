# CORSIKA 8 GPU 混合级联分支：中文说明

> 状态：本仓库是面向 21CMA 科研模拟的本地研究分支，不是 CORSIKA 8
> 官方发布版，也尚未经过 CORSIKA 8 上游评审。默认 CPU 路径仍是原来的
> `Cascade + PROPOSAL`；CUDA、CUDA 射电和 FLUKA 进程池都必须显式启用。

文档导航：

- [新增命令行参数完整参考](documentation/cuda_em_refactor/cli_reference.md)
- [CUDA 后端生产使用指南](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md)
- [验证脚本和复现实验](validation/gpu_em/README.md)
- [架构文档与阶段记录索引](documentation/cuda_em_refactor/README.md)
- [英文主 README 与新服务器部署](README.md#deploying-on-a-new-nvidia-server)

本文回答四个问题：

1. 相对原版 CORSIKA 8，这个分支加入了什么？
2. CPU 粒子蒙特卡洛为什么可以改写为 GPU wavefront？
3. 电磁、μ 子、强子和射电分别在哪一端计算？
4. 与当前官方公开版本相比，哪些方向是本分支的特色，哪些方面官方版本更强？

从零迁移、CUDA 架构选择和多核服务器调优的完整命令见
[英文主 README](README.md#deploying-on-a-new-nvidia-server)；物理覆盖、表格、
fallback 和输出字段见
[CUDA 后端生产指南](documentation/cuda_em_refactor/cuda_em_backend_user_guide.md)。

## 1. 项目定位

原版 CORSIKA 8 的空气簇射主循环是标量调度：

```text
从 Stack 取一个粒子
  -> 决定下一相互作用/衰变/边界/连续步长
  -> 推进粒子
  -> 生成次级粒子并压回 Stack
  -> 继续处理同一深度优先粒子树
```

这种方式物理逻辑清楚，而且同一随机数种子可以复现同一标量 shower；但同一时刻
只处理一条粒子历史，难以把一个大电磁前沿变成 GPU 上成千上万个相互独立的线程。

本分支保留原 Stack 和 CPU 物理模块，在旁边增加一条混合路径：

```text
                 CORSIKA CPU Stack
                         |
              +----------+-----------+
              |                      |
        hadron / tau           gamma / e± / μ±
              |                      |
    scalar step + FLUKA       HybridCascade router
              |                      |
              |              resident GPU queues
              |                      |
              |       select -> transport -> final state
              |              -> thinning -> reroute
              |                      |
              +----------+-----------+
                         |
          CPU rare final state / decay / output
                         |
              CoREAS/ZHS: CPU 或 CUDA
```

这里的“GPU μ 子”是有边界的：GPU 处理其连续损失、range、Molière 散射、磁场/
大气输运、边界、cut 和衰变距离竞争；实际衰变末态以及 μ bremsstrahlung、
pair production、photonuclear 等稀有末态仍交给 CPU 精确生成。

## 2. 本分支新增内容

### 2.1 标量输运与调度解耦

新增
[`ScalarCascadeStepper`](corsika/framework/core/ScalarCascadeStepper.hpp)，把原
`Cascade::step()` 中的单粒子物理推进抽成可复用组件。原 CPU Cascade 仍可调用
同一 stepper，因此默认路径不需要变成并发 Stack。

新增
[`HybridCascade`](corsika/framework/core/HybridCascade.hpp)，负责：

- CPU 主栈和 GPU staging 队列之间的物种路由；
- 小前沿的有界 CPU 展开；
- GPU wavefront 的提交与回收；
- 指定 PROPOSAL fallback、μ 衰变和 FLUKA worker 的返回；
- history、parent history、generation 和 step 标识；
- 输出、能量沉积、纵向 profile 和 shower 结束条件。

### 2.2 编译型 CUDA 后端

新增静态库目标 `CORSIKA8GpuEm`，核心接口位于
[`CudaEmBackend.hpp`](corsika/gpu/em/CudaEmBackend.hpp)，实现位于
[`src/gpu/em`](src/gpu/em/)。

设备粒子使用可直接复制的 POD 状态：

```text
PID, medium ID, generation
energy [GeV]
position [m], direction
time [s], weight
history ID, parent history ID, step ID
```

GPU 不持有 CORSIKA 的模板 Stack，也不在 device code 中使用带单位的复杂 C++
对象。主机/设备边界显式完成单位转换，device 内固定使用 GeV、m、s、
g/cm² 和 Tesla。

### 2.3 PROPOSAL 物理表

新增 `gpu_em_tablegen` 和版本化 `.c8emrt` 文件。表格保存：

- 每个粒子、介质组分和过程的相互作用率；
- 过程与目标组分概率；
- 能损比例 $v$ 的逆累计分布；
- 连续 $dE/dX$、range $R(E)$ 和逆 range；
- photon-pair、brems 和 electron-pair 的 LPM 参数；
- Molière 散射参数；
- PROPOSAL 版本、参数化、cut、能区、误差、schema 和内容哈希。

当前格式和数据结构见
[`RateTable.hpp`](corsika/gpu/em/tables/RateTable.hpp)，设备平坦视图见
[`FlatRateTable.hpp`](corsika/gpu/em/tables/FlatRateTable.hpp)。

表查询不在能区外静默 clamp。缺列、超范围、无逆 CDF 或哈希/误差不匹配都会
成为明确 fallback 或 hard failure。

这里有两个容易混淆的“表”：

- PROPOSAL 自身的插值 cache：运行 `gpu_em_tablegen` 时会在
  `--proposal-cache` 指定目录中自动创建；
- CUDA 运行时使用的 `.c8emrt`：`c8_air_shower` 不会自动生成，必须下载一个
  已验证表，或预先显式运行 `gpu_em_tablegen`。

当前 `c8_air_shower` 和 `gpu_em_tablegen` 都固定使用 `AirDry1Atm` 标准干空气。
仅改变五层大气的密度随高度分布、观测高度或磁场时，可以复用同一干空气表；
改变元素组成、组分比例、介质电离参数，或者改为岩石、土壤、月壤、冰时不可以
复用。当前版本也没有通用介质配置参数：这种改动需要同时扩展环境快照、介质
工厂和 table generator，重新生成并重新验收。

### 2.4 GPU 粒子过程

当前 GPU 末态覆盖：

- photon pair production；
- Compton scattering；
- photoelectric effect；
- $e^\pm$ bremsstrahlung；
- $e^\pm$ electron-pair production；
- 离散和连续 ionization；
- positron annihilation；
- Molière multiple scattering；
- pair/brems/epair 的 LPM 抑制；
- ParticleCut 和 EMThinning。

当前 GPU μ 子覆盖：

- 连续电离与 range；
- Molière 散射；
- 五层球形大气和均匀磁场 tracking；
- 离散相互作用、连续步长、边界、观测面、cut 和衰变距离竞争；
- GPU 端 ionization 选择。

CPU 指定末态包括：

- photoproduction/photonuclear；
- photon-induced muon pair；
- μ 子 bremsstrahlung、pair production 和其他稀有离散末态；
- μ 子实际 decay final state；
- 产生 hadron/μ/τ 的非 GPU 末态；
- 预先声明的介质或几何能力边界。

过程能力表见
[`ProcessCapabilities.hpp`](corsika/gpu/em/ProcessCapabilities.hpp)。

### 2.5 GPU CoREAS/ZHS

新增
[`CudaRadioAccumulator`](corsika/gpu/radio/CudaRadioAccumulator.hpp)，可用：

```text
--radio-backend cuda
```

设备端对每条 $e^\pm$ 轨迹、每个天线和时间 bin 计算 CoREAS endpoint 与 ZHS
贡献。为使并行累加顺序不改变结果，波形使用带范围检查的定点累加器；超过
`--gpu-radio-field-limit` 会终止 shower，而不是产生未标记的环绕误差。

原始 CPU 射电路径仍可用：

```text
--radio-backend cpu
```

无论 CPU 还是 GPU，射电贡献都随粒子轨迹段产生而在线累计；并不是等所有粒子
存盘以后再重新读取整棵 shower。最终波形在 shower 结束时下载或写出。μ 轨迹
不直接加入射电源项，因为原标量 `RadioProcess` 同样只观察 $e^\pm$；μ 衰变
产生的电子会正常进入射电计算。

### 2.6 FLUKA 多进程末态

新增 [`fluka_batch_worker`](applications/fluka_batch_worker.cpp) 和：

```text
--hadronic-backend fluka-process
```

这不是“把强子模型写成 CUDA kernel”。HybridCascade 把低能强子相互作用顶点
按工作类别和预估代价稳定分组，再交给多个隔离的持久 FLUKA CPU 进程批量计算，
减少重复初始化和逐顶点 IPC。默认起点为：

```text
--hadronic-workers 4
--hadronic-min-batch 64
--hadronic-target-batch-ms 5
--hadronic-max-batch 256
```

新服务器应单独扫描 worker 数量；GPU 性能计时时不能并行运行另一套 CPU shower。

### 2.7 确定性、回放和审计

设备随机数使用 Random123 Philox4x32-10。随机地址为：

```text
(seed, shower_id, history_id, step_id, process_id, draw_id)
```

实现见 [`Philox.hpp`](corsika/gpu/em/Philox.hpp)。队列位置、block 大小和
wavefront 重排不进入随机地址，因此同一 GPU、同一表和配置应可重复。

新增 `cuda_decision_replay`，可读取原版标量 decision tape，在 GPU 上消费指定
的输运记录并重新计算同轨迹射电。这能回答“GPU 能否精确消费原版某一事例”，
但它不是 production CUDA 独立抽样。

生产 CUDA 与原版 CPU 使用不同的随机流地址和调度顺序，所以相同 seed 一般不会
生成逐粒子完全相同的独立 shower。正确的两层验收是：

1. decision replay：同一指定粒子树/轨迹的逐过程调试；
2. independent ensemble：不同独立 seed 下 $X_{\max}$、纵向粒子数、能量
   沉积、地面分布和射电脉冲统计一致。

## 3. 粒子蒙特卡洛算法

### 3.1 相互作用距离

设粒子在能量 $E$ 和介质 $m$ 中，各离散过程的质量厚度反应率为
$\lambda_p(E,m)$，总率为：

$$
\lambda_{\mathrm{tot}}(E,m)=\sum_p\lambda_p(E,m).
$$

抽取 $u_X\in(0,1)$，下次离散相互作用的质量厚度为：

$$
\Delta X_{\mathrm{int}}
=-\frac{\ln u_X}{\lambda_{\mathrm{tot}}}.
$$

在非均匀大气中，$\Delta X$ 不是几何长度。tracking 必须沿轨迹积分
$\rho(\mathbf{x})\,ds$，并求逆得到实际顶点位置。

### 3.2 竞争限制

一次 step 不是无条件走到相互作用点，而是比较：

```text
离散相互作用距离
衰变距离
大气层/介质边界
磁场最大偏转步长
连续能损允许步长
观测面或环境逃逸面
```

取最近限制推进。到达边界时粒子保留并进入新介质；到达相互作用点才生成离散
末态；到达 cut 时沉积剩余允许能量；到达观测面时写 observation record。

### 3.3 过程、目标和能损抽样

用第二个随机数 $u_p$ 在累计率中选择过程和目标组分：

$$
\sum_{q<p}\lambda_q
\le u_p\lambda_{\mathrm{tot}}
<\sum_{q\le p}\lambda_q.
$$

对有随机能损的过程，用 $u_v$ 查询逆累计分布：

$$
v=F^{-1}_{p,m}(E,u_v).
$$

CPU PROPOSAL 原本就会使用自己的插值/逆累计计算。本分支为了 GPU 设备访问，
额外生成平坦、自适应细化且带实测误差的表；它不是声称原 PROPOSAL 完全没有
插值。二者区别见
[PROPOSAL 与 CUDA 插值说明](documentation/cuda_em_refactor/proposal_and_cuda_interpolation.md)。

### 3.4 连续能损

对带电粒子，表中 range 定义为从 transport cut 到当前能量的累计质量厚度。
走过 $\Delta X$ 后：

$$
E_1=R^{-1}\!\left(R(E_0)-\Delta X\right).
$$

差值 $E_0-E_1$ 进入连续能量沉积。若 $\Delta X$ 跨过 cut，则 step 在 cut
附近终止，避免查询表外能量。

### 3.5 次级粒子与薄化

每个输入粒子先计算会产生多少 continuation、次级、observation、fallback 和
记录。设备端用 CUB exclusive scan 得到稳定写入偏移：

```text
child_count:  [2, 0, 1, 2]
exclusive:    [0, 2, 2, 3]
```

这样每个线程写入自己的确定区间，不使用“哪个线程先抢到全局原子计数器”的不
确定追加顺序。history ID 按 scan 偏移稳定分配。

EM thinning 在 GPU 端使用同样的能量阈值、权重和保留概率语义。被保留粒子的
权重补偿保证 ensemble 期望值不变；单事例会有由 thinning 引入的统计涨落。

一个容易误判的细节是 `c8_air_shower` 的自动最大权重。未显式指定
`--max-weight` 时，CPU 原版和本分支都使用

```text
maxWeight = 0.5 * emthin * E_primary[GeV]
```

而 `EMThinning` 在 `parentWeight >= maxWeight` 时直接返回。初级和第一代未加权
粒子的权重为 1，所以自动 `maxWeight` 必须大于 1，薄化才可能从这些 history
启动。例如 $E_\mathrm{primary}=10^5$ GeV、`emthin=1e-6` 时，能量阈值是
0.1 GeV，但自动 `maxWeight=0.05`，因此该配置实际上不会启动 EM thinning。
$10^6$ GeV、`emthin=1e-6` 对应的 `maxWeight=0.5`，同样如此。若确实需要薄化，
应显式给出物理上合适且大于 1 的 `--max-weight`，并把该值纳入 CPU/CUDA
共同配置和验证；不能只根据运行时间判断薄化是否生效。程序会把
`can_activate_from_unit_weight` 写入 summary，并在不可能启动时给出 warning。

## 4. GPU wavefront 为什么成立

同一 shower 内不同粒子 history 在给定当前状态后可以独立推进一个物理限制。
因此可以把深度优先树改写为“同一轮推进许多粒子一次”：

1. host staging 收集可路由粒子；
2. 按 `PID × medium_id × energy_bin` 生成 bucket key；
3. stable radix sort 形成相似工作负载；
4. 每个 CUDA thread 推进一个粒子到最近限制；
5. exclusive scan 分配输出；
6. $\gamma/e^\pm/\mu^\pm$ 在设备常驻队列交叉路由；
7. observation、profile、radio、fallback 批量回收；
8. 队列为空后 shower 才结束。

稳定分桶实现见
[`CudaWavefrontBucketing.cu`](src/gpu/em/CudaWavefrontBucketing.cu)，设备工作区
和双缓冲见
[`DeviceWorkspace.hpp`](corsika/gpu/em/detail/DeviceWorkspace.hpp)。

这种调度的主要收益来自：

- 一次 kernel 启动处理大量相似粒子；
- photon/lepton 次级不必每一步往返主机；
- 物理表连续读取更适合 GPU cache；
- profile 和 radio 可在设备上归约；
- Philox 随机数不依赖执行顺序。

主要代价是：

- 小前沿无法占满 GPU；
- 稀有 CPU 末态会形成同步边界；
- 不同过程分支造成 warp divergence；
- 高质量低 thinning shower 会产生很大的队列和射电工作量；
- 强子、输出或 CPU fallback 可能成为 Amdahl 瓶颈。

`--gpu-min-batch` 控制小前沿何时先做有界 CPU 展开。它不是越小越好；不同 GPU
应对 64、256、1024、4096、8192 等值做固定配置的中位数扫描。

## 5. 环境与 21CMA 配置

当前 CUDA snapshot 包含：

- 五层球形大气；
- 四层指数密度和一层线性密度；
- 干空气介质及 medium ID；
- 球面边界求交；
- grammage 积分和逆积分；
- 均匀磁场 leapfrog；
- 球形观测面。

当前 `c8_air_shower` 从 `GeoMag/IGRF13.COF` 读取模型，固定为：

```text
year       2025
latitude   42.5527 deg
longitude  86.4153816422 deg
altitude   2680.444195 m
```

CPU 与 CUDA 使用同一个计算后磁场矢量，值写入 `gpu_em/config.yaml`。

山体、月壤、冰和一般三维介质尚未进入当前 GPU snapshot。官方 CORSIKA 8 的
通用环境框架能够表达跨介质问题，不等于本地 CUDA kernel 已支持这些几何。

## 6. 输出和失败策略

CUDA 模式写出：

- `gpu_em/config.yaml`：GPU、driver/runtime、表哈希、环境和配置；
- `gpu_em/summary.yaml`：粒子数、wavefront、fallback、显存、溢出和分阶段时间；
- profile、能量沉积和 observation 输出；
- CPU 或 CUDA CoREAS/ZHS 波形；
- 验证 provenance 和独立 ensemble 报告。

以下情况必须停止当前 shower：

- CUDA error、NaN、负能量或非法 PID；
- 表格 schema/hash/cut/能区/误差不匹配；
- 非法未登记过程；
- 大气积分、LPM、Molière 或磁场计算失败；
- 队列/定点射电累加溢出；
- 能量或记录计数不一致。

只有预先声明的稀有指定末态、能力边界和受控低能显存 spill 可以回 CPU。禁止
自动切换成 CPU 后把输出伪装成完整 CUDA shower。

## 7. 新服务器构建和运行

CUDA 构建要求 CMake 3.24+、CUDA toolkit 12.x、C++/CUDA 17。下面是一套可
迁移流程；更详细的故障排查和多 GPU 调优见
[英文主 README](README.md#deploying-on-a-new-nvidia-server)。

### 7.1 检查 driver、GPU 和编译器

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total \
  --format=csv
nvcc --version
cmake --version
c++ --version
gfortran --version
ldd --version
```

NVIDIA driver 属于宿主系统。Conda 中只有 toolkit 而 `nvidia-smi` 失败时，
需要先修复宿主 driver；在 WSL2 中应安装 Windows NVIDIA driver，不应在 WSL
内再安装第二套显示驱动。

### 7.2 建立参考 Conda 环境

```bash
conda create -n corsika_venv \
  -c conda-forge \
  python=3.9 \
  cmake=3.31 \
  conan=2.11 \
  particle=0.25.1 \
  numpy pandas scipy matplotlib pyyaml

conda activate corsika_venv

conda install \
  -c nvidia/label/cuda-12.6.3 \
  cuda-toolkit=12.6.3
```

这组版本复现当前开发环境。HPC 上也可以使用管理员提供的 CUDA/GCC module；
关键是 Conan 建 profile、CMake 和 NVCC host compiler 必须选择同一套 C++
工具链。

### 7.3 配置 FLUKA

```bash
export C8_FLUPRO=/path/to/fluka
export FLUPRO="$C8_FLUPRO"
export FLUFOR=gfortran
test -f "$C8_FLUPRO/libflukahp.a"
```

FLUKA binary 必须匹配服务器的 glibc 和 Fortran runtime。
`-DWITH_FLUKA=ON` 在本分支是 fail-closed：找不到库时 CMake 直接失败，不会
静默换成 UrQMD。

### 7.4 自动取得 GPU architecture

CMake 把 compute capability `8.9` 写成 `89`：

```bash
export C8_CUDA_ARCHS="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
  sed 's/\.//g' |
  sort -u |
  paste -sd';' -
)"

printf 'CUDA architectures: %s\n' "$C8_CUDA_ARCHS"
```

若 NVCC 不认识新 GPU 的 architecture，应升级 toolkit，不能把一个无关旧架构
当成正式性能构建。不要添加 `--use_fast_math`，否则现有数值验收失效。

### 7.5 安装 Release 依赖

```bash
export C8_SOURCE=/path/to/corsika8_gpu_refactor
export C8_BUILD=/path/to/corsika8_gpu_refactor_build_cuda
export C8_INSTALL=/path/to/corsika8_gpu_refactor_install_cuda

"$C8_SOURCE/conan-install.sh" \
  --source-directory "$C8_SOURCE" \
  --release
```

不要复用另一台机器、另一 CUDA toolkit 或另一 GPU 架构产生的
`CMakeCache.txt`。

### 7.6 配置、编译和安装

```bash
cmake \
  -S "$C8_SOURCE" \
  -B "$C8_BUILD" \
  -DCONAN_CMAKE_DIR="$C8_SOURCE/conan_cmake" \
  -DCMAKE_TOOLCHAIN_FILE="$C8_SOURCE/conan_cmake/conan_toolchain.cmake" \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=$C8_CUDA_ARCHS" \
  -DWITH_FLUKA=ON \
  -DC8_FLUKALIB="$C8_FLUPRO/libflukahp.a" \
  -DCMAKE_INSTALL_PREFIX="$C8_INSTALL"

cmake --build "$C8_BUILD" --parallel 16
cmake --install "$C8_BUILD"
```

CUDA 编译会消耗较多主机内存，`--parallel` 应同时考虑核心数和 RAM。预期得到：

```bash
test -x "$C8_BUILD/applications/c8_air_shower"
test -x "$C8_BUILD/applications/gpu_em_tablegen"
test -x "$C8_BUILD/applications/cuda_decision_replay"
test -x "$C8_BUILD/applications/fluka_batch_worker"
```

### 7.7 构建后测试

```bash
ctest \
  --test-dir "$C8_BUILD" \
  --output-on-failure \
  --parallel 16

cd "$C8_SOURCE"
python -m unittest discover \
  -s validation/gpu_em/tests \
  -p 'test_*.py'
```

### 7.8 准备默认干空气物理表

物理表不是 GPU 架构文件。换显卡或 CUDA 架构不需要重新生成。最简单的方式是
从本项目的私有 GitHub Release 下载已经验收的默认表：

```bash
export C8_TABLE_DIR="$C8_BUILD/gpu_em_tables"
mkdir -p "$C8_TABLE_DIR"

gh release download gpu-table-dry-air-v10 \
  --repo BossL668/corsika8-gpu-hybrid \
  --pattern 'production_v10_muons_1e-3_1EeV.c8emrt' \
  --dir "$C8_TABLE_DIR"

export C8_TABLE="$C8_TABLE_DIR/production_v10_muons_1e-3_1EeV.c8emrt"
echo '14eb8d7fe38c8046e3f6e38935a107e08e5e07e45d11496f6cda9e8a41cd9521  '"$C8_TABLE" |
  sha256sum --check
```

私有仓库的协作者需要先运行 `gh auth login`。也可以从 Release 网页手工下载，
但仍应执行同一个 SHA-256 检查。该表适用于：

- CORSIKA `AirDry1Atm` 标准干空气；
- `--emcut 0.0005` GeV；
- `--mucut 0.3` GeV；
- 初级总能量不高于 \(10^{18}\) eV；
- CUDA 电磁和可选 CUDA \(\mu^\pm\) 输运。

如果不需要 CUDA μ 子输运，也可以使用只含 \(\gamma/e^\pm\) 的已验证表；
表中没有成对的 PDG `13/-13` 时，μ 子保留在 CPU 路径。

`c8_air_shower` 不会在缺表时自动执行生成器。这样设计是为了避免生产任务在
计算节点上意外花费很长时间制表，也避免多个任务并发写同一缓存。若只改变 GPU
型号、GPU 数量、磁场、天线、观测高度或同一干空气的密度 profile，直接复用表。
以下变化必须生成并重新验收表：

- PROPOSAL 版本或物理参数化；
- 介质组成、组分比例或材料常数；
- `--emcut`、`--mucut`；
- 所需最大能量或表格式版本。

当前生成器只支持标准干空气。对默认干空气自行制表的完整参数见
[`gpu_em_tables/README.md`](gpu_em_tables/README.md)，全部开关见
[`cli_reference.md`](documentation/cuda_em_refactor/cli_reference.md)。

### 7.9 典型全加速运行

典型全加速运行参数：

```bash
c8_air_shower \
  -p 2212 -E 100000 -N 50 \
  -f /path/to/output \
  --seed 10200001 \
  --emcut 0.0005 \
  --emthin 1e-6 \
  --em-backend cuda \
  --gpu-device 0 \
  --gpu-min-batch 4096 \
  --gpu-memory-fraction 0.70 \
  --gpu-table-cache "$C8_TABLE" \
  --gpu-table-tolerance 1e-3 \
  --gpu-deterministic true \
  --gpu-resident-cross-species true \
  --radio-backend cuda \
  --gpu-radio-field-limit 1 \
  --hadronic-backend fluka-process \
  --hadronic-workers 4 \
  --hadronic-min-batch 64 \
  --hadronic-target-batch-ms 5 \
  --hadronic-max-batch 256
```

性能模式不要启用：

```text
--gpu-detailed-stage-timing
--gpu-full-step-records
--gpu-radio-track-diagnostics
```

它们用于诊断，会增加同步、归约或输出开销。

## 8. 与官方最新公开状态的比较

### 8.1 比较基线和方法

查询日期为 2026-07-31。

- 官方标签页把
  [`corsika8-v1.0-beta1`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/tags/corsika8-v1.0-beta1)
  标为 2024-12-24 的首个公开 beta；
- 其后有 2025-06-24 的
  [`icrc2025-v1`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/tags)
  研究基线，但标签页没有把它写成新的正式 release；
- 查询时官方 `main` 为
  [`77bedbaf`](https://gitlab.iap.kit.edu/AirShowerPhysics/corsika/-/commit/77bedbafec10782018390e144bef258de919fc24)，
  提交日期 2026-07-08；
- 对该 commit 的公开源码树检索
  `CORSIKA_ENABLE_CUDA`、`CORSIKA8GpuEm`、`--em-backend`、
  `--radio-backend`、`cuda_decision_replay` 和 `fluka-process`，未发现本分支
  对应的 CUDA EM/射电构建目标和 CLI；
- 官方 2025 射电论文确认 CORSIKA 8 原生支持同一 shower 同时计算 CoREAS
  endpoint 与 ZHS：
  [arXiv:2409.15999](https://arxiv.org/abs/2409.15999)；
- 官方 2026 总览论文展示通用框架、空气簇射验证、跨介质和冰中射电：
  [arXiv:2604.01850](https://arxiv.org/abs/2604.01850)；
- 官方生态也有 GPU 光学 Cherenkov/fluorescence 方向，但它与本分支的
  $\gamma/e^\pm/\mu^\pm$ 粒子输运不是同一问题：
  [arXiv:2405.04229](https://arxiv.org/abs/2405.04229)。

“官方公开源码未发现”只描述上述 commit 和检索范围，不能排除未公开分支或其他
合作组原型。因此下表使用“公开主线中未见同类接口”，不使用“世界首个”等无法
证明的表述。

### 8.2 能力矩阵

| 能力 | 官方公开版/main | 本地 GPU 分支 | 判断 |
|---|---|---|---|
| C++17、单位、模块化 ProcessSequence | 官方核心能力 | 完整继承 | 官方基础优势 |
| 标量 Cascade + PROPOSAL | 支持 | 默认路径保留 | 等价基础 |
| CoREAS endpoint + ZHS 同 shower | 官方论文和源码支持 | CPU 路径继承 | 官方基础能力 |
| 通用环境、跨介质、冰中级联/射电 | 2026 官方论文展示 | CPU 框架继承；CUDA snapshot 尚限五层大气 | 官方更完整 |
| GPU 光学 photon 传播研究 | 官方生态已有研究 | 不是本分支目标 | 方向不同 |
| $\gamma/e^\pm$ CUDA 粒子输运 | 公开 main 未见同类构建/CLI | 已集成 HybridCascade | 本分支特色 |
| $\mu^\pm$ GPU 连续输运和顶点竞争 | 公开 main 未见同类接口 | v10 表下已集成，末态回 CPU | 本分支特色 |
| resident CUDA CoREAS/ZHS | 公开 main 为通用 RadioProcess，未见 CUDA backend CLI | 可选定点 CUDA accumulator | 本分支特色 |
| 调度无关 Philox history RNG | 未见对应公开设备接口 | 已实现 | 本分支特色 |
| 原版 decision tape → CUDA 精确回放 | 未见对应公开工具 | 已实现 | 本分支特色 |
| 版本化 PROPOSAL GPU 表和 fail-closed hash/tolerance | 未见对应公开设备表 | 已实现 | 本分支特色 |
| FLUKA 低能强子 | 支持标量 FLUKA | 继承并增加进程池批处理 | 本分支调度扩展 |
| 单 shower 跨多 GPU | 未公开声明 | 尚不支持 | 双方均非当前能力 |
| 上游 CI、用户群、通用维护和官方物理结论 | 官方具备 | 本地工作树，尚未上游评审 | 官方显著更强 |

### 8.3 可以合理称为“领先”的具体维度

在“已公开官方 main 与本地当前代码”的限定比较下，本分支的前沿点是：

1. 把真实 PROPOSAL 电磁和部分 μ 子输运，而不是仅光学 photon，集成进
   CORSIKA 8 shower 主循环的 resident CUDA wavefront；
2. 通过 history-keyed Philox、stable radix sort 和 exclusive scan，使并行
   调度具备可审计的确定性；
3. 用版本化物理表、指定末态 fallback 和 hard-failure 边界避免“GPU 加速但
   静默少算物理”；
4. 让 CoREAS/ZHS 可在同一设备端随轨迹在线累计，并保留 CPU 同轨迹验收路径；
5. 把强子瓶颈拆成独立 FLUKA 进程池，与 GPU 轻子前沿形成混合流水；
6. 提供 exact replay、能量账本、provenance、独立 ensemble 和射电脉冲统计
   组成的多层验证体系。

这些是技术实现维度的领先，不等于总体成熟度领先。官方版本在通用几何、跨介质、
冰中级联、上游维护、协作评审和可引用物理结论方面仍有明显优势。

## 9. 当前限制

- CUDA 环境仅覆盖当前五层球形大气；山体/月壤/冰需要新 snapshot 和新表；
- 强子物理没有变成 GPU kernel，FLUKA 加速仍依赖 CPU 多进程；
- μ 稀有离散末态和实际衰变末态仍在 CPU；
- 单 shower 不能跨多 GPU；多 GPU 服务器应一块卡运行一个独立进程；
- 不同 GPU 架构只承诺统计一致，不承诺浮点逐位一致；
- production CUDA 与原版 CPU 相同 seed 不承诺逐粒子同一 shower；
- `.c8emrt` 只在记录的粒子、介质、cut、能区和误差范围内有效；
- CUDA radio 的定点范围必须预先配置并检查 overflow；
- 本地工作树包含尚未上游合并的实现，迁移前必须保存精确 commit/patch、二进制
  和表哈希；仅从官方仓库重新 clone 不会得到这些 GPU 功能。

## 10. 验证路线

迁移或修改 kernel 后至少执行：

```text
1. CUDA/CPU CTest
2. Python validation 单元测试
3. 同 GPU 固定 seed 重复性
4. 单过程 rate/v/角度/末态抽样
5. tracking 边界、grammage、时间和磁场对照
6. energy ledger 与 illegal fallback 检查
7. CPU/CUDA independent shower ensemble
8. Xmax、纵向粒子数、能量沉积和地面分布
9. CoREAS/ZHS 地磁振幅和脉冲宽度分布
10. 冷/热缓存分开的五次中位数性能
```

主要工具位于 [`validation/gpu_em`](validation/gpu_em/)，说明见
[`validation/gpu_em/README.md`](validation/gpu_em/README.md)。

必须区分三种“相同”：

- 同一 CUDA 配置重复：应确定性相同；
- 原版 decision tape 与 CUDA replay：指定轨迹应逐记录一致；
- 原版 CPU 与 production CUDA 独立抽样：比较统计分布，不要求逐事例相同。

## 11. 代码阅读顺序

建议按以下顺序理解实现：

1. [`ScalarCascadeStepper.hpp`](corsika/framework/core/ScalarCascadeStepper.hpp)
   和对应 `.inl`：原标量单步蒙特卡洛；
2. [`HybridCascade.hpp`](corsika/framework/core/HybridCascade.hpp)：CPU/GPU
   调度边界；
3. [`PhysicalCudaEmRouter.hpp`](corsika/gpu/em/PhysicalCudaEmRouter.hpp)：
   粒子转换、fallback、输出和强制衰变；
4. [`Types.hpp`](corsika/gpu/em/Types.hpp)：设备 POD 和记录格式；
5. [`Philox.hpp`](corsika/gpu/em/Philox.hpp)：调度无关随机数；
6. [`CudaEmBackend.cu`](src/gpu/em/CudaEmBackend.cu)：常驻队列和 wavefront；
7. photon/lepton selection、transport 和 final-state CUDA 文件；
8. [`RateTable.hpp`](corsika/gpu/em/tables/RateTable.hpp) 与
   [`gpu_em_tablegen.cpp`](applications/gpu_em_tablegen.cpp)；
9. [`CudaRadioAccumulator.cu`](src/gpu/em/CudaRadioAccumulator.cu)；
10. [`fluka_batch_worker.cpp`](applications/fluka_batch_worker.cpp)；
11. [`validation/gpu_em`](validation/gpu_em/) 的物理与性能验收工具。

## 12. 引用与科研表述

使用本分支生成科研结果时，应分别引用：

- CORSIKA 8 框架/官方总览论文；
- CORSIKA 8 射电论文；
- PROPOSAL 和所用强子模型；
- FLUKA（若启用）；
- 对本地 CUDA 分支，应记录源码 commit/patch、编译器、GPU/driver/runtime、
  `.c8emrt` SHA-256、CLI、seed、天线文件和验证报告。

在本分支尚未上游发表前，论文中宜写成“基于 CORSIKA 8 的本地 CUDA 研究后端”，
而不是“CORSIKA 8 官方 CUDA 后端”。
