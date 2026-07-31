# Phase 49：Release 构建与五次正式性能验收

## 1. 本阶段解决的问题

此前 Phase 39 的一次性性能验收已经超过 5 倍，Phase 42 也实现了五次独立
进程重复和中位数门禁，但原计划要求的以下证据仍不完整：

- 真正的 `CMAKE_BUILD_TYPE=Release`；
- NVIDIA RTX 4060 Laptop GPU 对应的 `sm_89` 构建；
- 热缓存条件下 CPU PROPOSAL 与 CUDA 各运行五次；
- 单 CPU 数值线程；
- 以双方中位数计算端到端加速比；
- Release 产物完成设备测试和严格能量账本测试。

本阶段补齐这些证据。性能测试只评价速度，不代替独立 seed 的物理系综验收。

## 2. Release 依赖与构建

在 `corsika_venv` 中为 Conan 依赖补充 `Release` 配置后，使用：

```bash
cmake -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target c8_air_shower -j4
```

构建缓存的最终 provenance 为：

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_ARCHITECTURES=89
CORSIKA_ENABLE_CUDA=ON
C++ compiler=/usr/bin/c++
CUDA compiler=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/nvcc
```

切换构建类型时发现旧 build cache 中曾残留空的
`CMAKE_MAP_IMPORTED_CONFIG_RELEASE`。空映射会同时屏蔽 Conan 的
config-specific Release 库和 CUDA/TAUOLA 的 config-less imported target。
清除该历史缓存项后，CMake 的默认查找顺序可同时解析两类依赖；不需要修改
物理代码，也不需要保留特殊映射。

## 3. Release 回归测试

所有 GPU 专项 target 在 Release 下重新编译。设备测试必须在可访问 WSL
NVIDIA 设备的上下文中执行；受限环境中的 `Skipped` 不计为通过。

真实设备结果：

```text
25/25 C++/CUDA tests passed
0 failed
0 skipped
real time: 9.71 s
```

覆盖范围包括：

- host ABI、表格读取与 CPU fallback；
- Philox、stable scan/compaction 和三维稳定分桶；
- photon/lepton transport 与全部 GPU 末态；
- pair、brems、epair LPM；
- thinning、Molière、均匀磁场和五层球形大气；
- resident photon/lepton wavefront；
- CPU CoREAS/ZHS 轨迹投影兼容路径。

验证工具测试：

```text
25/25 Python tests passed
```

## 4. Release 严格能量账本 smoke

配置：

```text
primary              electron
energy               10 GeV
zenith               0 degrees
EM cut               0.5 MeV
thinning             off
gpu-min-batch         1
radio ring            0
```

输出：

```text
/tmp/c8_phase49_release_strict_smoke_v1
```

关键结果：

```text
GPU particles                         16135
CPU scalar EM steps                       0
CPU generic fallbacks                     0
CPU specified final states                0
memory spill particles                    0
queue overflows                           0
profile fixed-point overflows             0
process registry accepted              true
energy-ledger complete coverage        true
relative closure error                    0
energy-ledger accepted                  true
```

独立 `validate_energy_ledger.py` 从 primitive terms 重新计算 source、terminal
和 residual，结果同样为 `status=passed`，而不是信任 application 已存储的
residual。

## 5. 五次正式性能配置

执行：

```bash
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python \
  validation/gpu_em/run_performance_acceptance.py \
  --executable \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v9_1e-3_1EeV.c8emrt \
  --output-root \
    /tmp/c8_phase49_release_performance_1PeV_5rep_v1 \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --seed 24949 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --gpu-min-batch 64 \
  --require-release-build \
  --minimum-speedup 5
```

共同条件：

```text
primary                      electron
energy                       1 PeV
EM cut                       0.5 MeV
EM thinning                  1e-4
maximum weight               100
hadron/muon/tau cut          1e13 GeV
radio ring                   0
OMP/BLAS/MKL/NUMEXPR threads 1
Molière cache                hot, 539032 bytes
GPU                          RTX 4060 Laptop, sm_89
```

五轮按 `proposal -> cuda`、`cuda -> proposal` 交替，避免固定顺序偏置。
每一轮都是两个全新的进程；相同 seed 用来保持性能工作负载一致。

## 6. 正式结果

输出：

```text
/tmp/c8_phase49_release_performance_1PeV_5rep_v1/benchmark_summary.json
```

后端样本：

| 指标 | PROPOSAL 五次样本 | CUDA 五次样本 |
|---|---|---|
| 外部 wall time / s | 96.762, 97.169, 95.746, 91.869, 93.255 | 12.292, 12.335, 10.413, 10.405, 10.342 |
| 公共 shower timing / s | 93.442, 93.968, 92.592, 88.680, 90.213 | 8.525, 8.568, 6.688, 6.711, 6.683 |

双方中位数及主门禁：

| 指标 | PROPOSAL 中位数 | CUDA 中位数 | ratio-of-medians |
|---|---:|---:|---:|
| 外部 wall time | 95.746 s | 10.413 s | **9.1949×** |
| 公共 shower timing | 92.592 s | 6.711 s | **13.7978×** |

逐轮配对的外部 wall-time 加速比为：

```text
7.8721, 7.8776, 9.1949, 8.8290, 9.0167
median = 8.8290
```

逐轮配对的公共 shower timing 加速比为：

```text
10.9614, 10.9679, 13.8442, 13.2148, 13.4986
median = 13.2148
```

最终机器门禁：

```text
status = passed
minimum required speedup = 5
require_release_build = true
```

因此原计划“热缓存、Release、单 CPU 核、每项五次取中位数、纯 EM shower
端到端至少 5 倍”的性能要求已由正式真实设备证据满足。物理统计矩阵仍按
滚动审计中的剩余项目继续执行。
