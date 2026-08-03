# 阶段 84：恢复 FLUKA、强子工作分类与固定种子性能基线

## 1. 本阶段的目的

本阶段处理两个直接相关的问题：

1. CUDA 版本必须和原版
   `/home/yuhanglu/21CMA/corsika-21cma/corsika/applications/c8_air_shower.cpp`
   一样，使用 FLUKA 作为低能强子模型，不能以 UrQMD 替代；
2. 在实施强子并行前，先按模型、粒种和能区测量真实工作量，确认强子是否已经成为端到端瓶颈。

本阶段没有改变粒子栈的处理顺序，也没有并行调用 FLUKA。这样可以先保留原有
LIFO 调度和随机数消费顺序，建立可信基线。

## 2. 原版和 CUDA 版本的强子模型配置

原版构建缓存给出的配置是：

```text
C8_FLUKALIB=/home/yuhanglu/fluka/libflukahp.a
WITH_FLUKA=ON
CMAKE_C_COMPILER=/usr/bin/cc
CMAKE_CXX_COMPILER=/usr/bin/c++
CMAKE_Fortran_COMPILER=/usr/bin/gfortran
```

原版应用在 `WITH_FLUKA` 下实例化：

```cpp
corsika::fluka::Interaction leIntModel{all_elements};
InteractionCounter leIntCounted{leIntModel};
```

并在 79.4328 GeV 的默认转换能量处，通过 `EnergySwitch` 在低能 FLUKA 和高能
SIBYLL 之间选择。

此前 CUDA 构建在 Conda Fortran sysroot 下链接 FLUKA 时失败，随后临时配置成了
UrQMD。1 PeV 质子测试在约 29 s 时出现：

```text
UrQMD terminating without collision !?
iterations = 50000
```

因此，这个 UrQMD 结果不能作为原版与 CUDA 版本的物理或性能对照。

## 3. FLUKA 构建修复

### 3.1 工具链

当前唯一 CUDA 构建目录为：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda
```

重新配置时固定：

```text
Build type       Release
C compiler       /usr/bin/cc 13.3
C++ compiler     /usr/bin/c++ 13.3
Fortran compiler /usr/bin/gfortran 13.3
CUDA             12.6.85
CUDA host        /usr/bin/c++
CUDA arch        89
FLUKA library    /home/yuhanglu/fluka/libflukahp.a
WITH_FLUKA       ON
```

完整配置命令为：

```bash
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/cmake --fresh \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/yuhanglu/21CMA/corsika8_gpu_refactor_install_cuda \
  -DCMAKE_TOOLCHAIN_FILE=/home/yuhanglu/21CMA/corsika-21cma/corsika/conan_cmake/conan_toolchain.cmake \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
  -DCMAKE_CUDA_COMPILER=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++ \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPYTHON_EXECUTABLE=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python3 \
  -DCORSIKA_ENABLE_CUDA=ON \
  -DC8_FLUKALIB=/home/yuhanglu/fluka/libflukahp.a \
  -DWITH_FLUKA=ON
```

### 3.2 防止静默选错模型

`applications/CMakeLists.txt` 现在执行两项检查：

- 如果发现有效的 `fluka` target 且用户没有显式设置 `WITH_FLUKA`，应用默认选择
  FLUKA；
- 如果用户要求 `WITH_FLUKA=ON`，但没有有效 FLUKA target，配置立即失败，不能静默
  编译成 UrQMD 版本。

显式的 `-DWITH_FLUKA=OFF` 仍然可以用于专门的 UrQMD 实验。

### 3.3 动态链接证据

当前 CUDA 应用包含：

```text
-DWITH_FLUKA
```

`ldd` 显示：

```text
libfluka.so    => .../corsika8_gpu_refactor_build_cuda/modules/fluka/libfluka.so
libgfortran.so.5 => /lib/x86_64-linux-gnu/libgfortran.so.5
libm.so.6        => /lib/x86_64-linux-gnu/libm.so.6
```

没有 `not found`，也不再从 Conda 旧 sysroot 的 `/lib64` linker script 解析
`libm`。

## 4. FLUKA 运行验证

运行必须进入 `corsika_venv`，因为该环境保存：

```text
FLUPRO=/home/yuhanglu/fluka
```

裸 shell 中未设置 `FLUPRO` 时，FLUKA 初始化会显式失败，不会退回其他模型。

测试命令：

```bash
/home/yuhanglu/miniconda3/bin/conda run --no-capture-output \
  -n corsika_venv \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/tests/modules/testModules \
  'FLUKA*' --reporter compact
```

结果：

```text
FLUKA version 2025.0.0
All tests passed (269 assertions in 2 test cases)
```

覆盖内容包括：

- CORSIKA/FLUKA 粒子编号转换；
- H、N、O、Ar 靶材料初始化；
- π、K、Λ、Σ、质子和反质子的截面；
- 1、20、100、1000 GeV 末态生成；
- 次级动量和能量守恒；
- 非法入射粒子和未初始化靶核的拒绝路径。

框架回归结果为：

```text
All tests passed (1447 assertions in 49 test cases)
```

其中新增的 `InteractionCounter::getCount()` 也有回归断言。

## 5. 可审计的模型元数据

每个完成的 CUDA shower 现在在 `gpu_em/summary.yaml` 中记录：

```yaml
hadronic_models:
  high_energy:
    name: SIBYLL-2.3d
    interactions: 1891
  low_energy:
    name: FLUKA
    version: 2025.0.0
    interactions: 24697
  transition_energy_GeV: 79.4328
```

相互作用次数来自包裹实际 `doInteraction()` 的 `InteractionCounter`；只有底层模型
成功返回后才增加计数。因此这比“链接了 `libfluka.so`”更强：它证明该 shower
确实调用 FLUKA 生成了低能强子末态。计数在每个 shower 开始时保存基线，输出写入
增量而不是进程累计值；`-N 2` 回归分别得到 28 和 76 次 FLUKA 相互作用，证明第二
个 shower 没有把第一个 shower 的 28 次重复计入。

旧的质子 CUDA 输出若没有这一段元数据，不能再单独作为 FLUKA 验收证据。尤其是
此前在无 FLUKA 的 CUDA 构建下产生的 hadronic ensemble，应当用当前构建重新生成。

## 6. 强子分类存储

新增：

```text
corsika/framework/core/HadronicWorkQueue.hpp
```

分类键是：

```text
低能/高能模型
× 核子/π/K/其他介子/其他重子/原子核
× log2 动能区间
× 核质量数区间
```

默认每个 octave 分两个能量 bin，模型转换能量使用应用的
`--hadronModelTransitionEnergy`。

`ClassifiedHadronicWorkQueue<T>` 提供：

- 每类独立 FIFO；
- 单调 sequence ID；
- 按当前估计成本最大的类别优先取任务；
- 以目标总成本而不是固定粒子数组成 batch；
- 稳定、可重复的 batch 顺序；
- 不使用并发 CORSIKA `Stack` 或 `SecondaryView`。

当前 `HybridCascade` 只启用了同一分类器的观测模式：每个原有 LIFO 标量步骤完成后，
记录分类、时间、均值、标准差和极值。尚未改变执行顺序，因此这一步不会因为重排而
改变 CORSIKA/FLUKA 随机数消费。

## 7. 固定种子确定性

### 7.1 `seed=0` 的含义

在 `c8_air_shower.cpp` 中：

```cpp
if (seed == 0) {
  std::random_device rd;
  seed = rd();
}
```

所以 `/home/yuhanglu/21CMA/python/config.yaml` 当前的 `-s 0` 表示自动生成种子，
不是固定的数值 0。CPU/CUDA 逐事例对照必须改用相同的非零种子。

`emthin` 越小，thinning 阈值越低，通常会保留更多粒子，因此运行更慢并提供更高
统计质量。用户已经用完整数据集验证：只指定 `emthin` 是有效配置，
`1e-6` 比 `1e-5` 更慢、同时提供更好的数据质量。因此 CUDA 适配层必须把
`emthin` 原样交给 CORSIKA，不能要求用户额外指定 `--max-weight`。后续精确代码
审计补充了一个能量相关边界：1 PeV、自动权重、`1e-6` 得到
`maxWeight=0.5`，所以原版 `parentWeight >= maxWeight` 保护会让该精确组合
实际未薄化；这与更慢、更高质量的观测一致。

如果没有显式指定 `--max-weight`，原版和 CUDA 版都沿用应用已有的自动计算：

```text
Wmax = 0.5 * emthin * E0[GeV]
```

这属于原版 thinning 配置的一部分，而不是 CUDA 后端新增的行为。性能开发阶段可以
明确使用 `emthin=1e-5` 或 `1e-4` 缩短周转时间，但必须在结果中标成性能冒烟测试，
不能把它冒充 `/home/yuhanglu/21CMA/python/config.yaml` 中 `emthin=1e-6` 的
正式质量验收。正式验收仍需保持配置参数不变。

### 7.2 两次 CUDA 重复

采用质子、1 PeV、天顶角 27°、方位角 180°、种子 12345、FLUKA、0.5 MeV
EM cut、`emthin=1e-4`、`max-weight=100`，重复两次。

下列文件逐字节 SHA-256 一致：

- `particles/particles.parquet`
- `profile/profile.parquet`
- `production_profile/profile.parquet`
- `energyloss/dEdX.parquet`
- `interactions/interactions.parquet`
- `interaction_hist/inthist_lab_1.npz`
- `interaction_hist/inthist_cms_1.npz`
- 空天线配置下的 CoREAS/ZHS observer 文件

两次都得到：

```text
FLUKA interactions     24697
SIBYLL interactions     1891
GPU particles       20713175
physical secondaries 4712242
scalar steps         1881885
```

墙钟分别约 62.2 s 和 59.6 s。墙钟抖动不影响物理产物。

结果目录：

```text
/home/yuhanglu/21CMA/corsika_validation_results/hadronic_fluka_fixedseed_1PeV_cuda_a_v1
/home/yuhanglu/21CMA/corsika_validation_results/hadronic_fluka_fixedseed_1PeV_cuda_b_v1
```

## 8. 原版 CPU 与 CUDA 性能

相同的 1 PeV、种子 12345 和物理参数下：

| 后端 | 墙钟 |
|---|---:|
| 原版单核 CPU + PROPOSAL + FLUKA | 154.36 s |
| CUDA EM + CPU FLUKA，重复 A | 62.15 s |
| CUDA EM + CPU FLUKA，重复 B | 59.59 s |

以 CUDA 两次中值计，当前端到端加速约：

```text
154.36 / 60.87 = 2.54×
```

原版结果：

```text
/home/yuhanglu/21CMA/corsika_validation_results/hadronic_fluka_fixedseed_1PeV_original_cpu_v1
```

原版的 `interactions/interactions.parquet` 与 CUDA 版本逐字节相同，说明早期主强子
相互作用链在这个固定种子下得到保留。EM profile 和地面粒子不逐事例相同，这是
当前表驱动 CUDA 电磁抽样与标量 PROPOSAL 使用不同随机变量映射的预期结果；它们仍
必须通过多种子 ensemble 检验统计一致性，不能用这一个 shower 宣称物理验收完成。

## 9. 真正的当前瓶颈

固定种子 CUDA 运行的 `HybridCascade` 计时为：

| 部分 | 时间 | 占 HybridCascade |
|---|---:|---:|
| 总运行 | 53.188 s | 100% |
| CPU scalar stepper | 44.972 s | 84.6% |
| CUDA router/backend wall section | 7.145 s | 13.4% |
| 分类为强子的 scalar steps | 6.377 s | 12.0% |
| 其中 μ± scalar steps | 38.588 s | 72.5% |

最大的单项是：

```text
mu-: 886896 steps, 19.425 s
mu+: 873343 steps, 19.162 s
```

低能核子、π、K 等强子步骤合计远小于 μ 输运。

因此，在这一参数点：

- 强子/FLUKA 还不是端到端门槛；
- 即使把全部分类强子步骤理想地降为零，Amdahl 上限也只有
  `1 / (1 - 0.120) = 1.14×` 的额外加速；
- 下一项最有价值的工作是 μ 输运批处理/并行，而不是立即重写 FLUKA GPU 内核。

这一结论不代表更高能量、不同 cut 或重核初级下强子永远不成为瓶颈。分类统计将对
能量、初级和 cut 做 scaling 扫描；只有强子占比显著上升后，才值得启用强子专用
并行后端。

### 9.1 FLUKA 质子/铁核 scaling 扫描

当前先用 `emthin=1e-4`、`max-weight=100`、0.5 MeV EM cut 和无天线输出建立
性能剖面。以下时间均来自 `summary.yaml` 内的 `HybridCascade` 计时，而不是只计
CUDA kernel：

| 初级 | 能量 | 总时间 | 强子步骤 | μ± 步骤 | FLUKA 相互作用 | SIBYLL 相互作用 |
|---|---:|---:|---:|---:|---:|---:|
| proton | 1 TeV | 0.364 s | 0.003 s (0.9%) | 0.034 s (9.4%) | 11 | 2 |
| proton | 10 TeV | 1.040 s | 0.078 s (7.5%) | 0.430 s (41.3%) | 448 | 33 |
| proton | 100 TeV | 4.178 s | 0.388 s (9.3%) | 2.651 s (63.5%) | 2206 | 160 |
| proton | 1 PeV | 53.188 s | 6.377 s (12.0%) | 38.588 s (72.5%) | 24697 | 1891 |
| iron | 100 TeV | 7.567 s | 0.729 s (9.6%) | 4.920 s (65.0%) | 4017 | 280 |
| iron | 1 PeV | 45.299 s | 4.852 s (10.7%) | 32.175 s (71.0%) | 25270 | 2079 |

铁核的两次运行都正常结束，分别实际调用 FLUKA 4017 和 25270 次。因每个能量和
初级目前只有一个 shower，这张表用于瓶颈定位，不能作为物理 scaling law 的最终
拟合。它已经足以说明：FLUKA 工作量随能量和重核初级增加，但当前参数下 μ± 标量
输运仍比分类强子步骤大约高 6–7 倍。

## 10. 为什么不能直接对 FLUKA 开线程

当前 FLUKA 接口存在明确的进程全局状态：

- `evtxyz_` 写入全局 HEPEVT COMMON block；
- C++ 随后从全局 `hepevt_` 读取次级粒子；
- FLUKA RNG 通过全局函数指针连接到 CORSIKA 的 `fluka` 随机流；
- 测试源码明确说明 FLUKA 在单进程中只能初始化一次；
- `cumsgx_` 和材料初始化也与这套全局状态配套。

因此，同一进程内让多个线程并发调用 `getCrossSection()`/`doInteraction()` 会产生
数据竞争、随机数交错和末态覆盖，不能作为科研级实现。

可接受的并行边界是：

1. 每个 FLUKA worker 使用独立进程，拥有独立 COMMON block 和 RNG；
2. 主进程按本阶段的工作分类形成近等成本 batch；
3. 输入和输出都使用 POD 消息，次级粒子返回后由主进程按稳定 sequence ID 合并；
4. 在进程并行投入生产前，必须先把 FLUKA 随机数键绑定到
   `(seed, shower, history, step, process, draw)`，否则调度改变仍会改变单事例；
5. worker 崩溃或超时必须终止当前 shower，不能静默切回 UrQMD。

GPU 化 FLUKA 本身意味着重写或替换大规模 Fortran 事件生成器，并验证全部截面和
末态，不是当前最短的生产路径。

## 11. 下一步

1. 用固定非零种子，对 proton/iron、1 TeV–1 EeV、多个 cut 扫描
   `hadronic time / total time`；
2. 重新生成明确记录 `FLUKA 2025.0.0` 的 CPU/CUDA proton ensemble；
3. 先实现进程隔离的强子 batch 原型，只用于可控的小规模回放测试；
4. 并行推进 μ± 的分类、批处理和 PROPOSAL 线程安全审计，因为当前它占总时间
   72.5%；
5. 最终以相同参数表、相同显式种子集合比较 Xmax、粒子数、地面分布、能损和射电
   脉冲统计，并报告端到端中值加速。
