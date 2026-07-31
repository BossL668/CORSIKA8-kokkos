# Phase 50：CPU Step 方向等价、完整 Release 构建与 1 PeV 复验

## 1. 本阶段的触发条件

Phase 49 已证明 Release CUDA 后端在 1 PeV 纯电磁 shower 上达到正式五次
中位数性能门禁，但重新生成的 200+200 独立 1 PeV 系综暴露出一组相关的
地面偏差：

```text
ground radius mean       -1.851%
ground radius RMS        -1.887%
ground arrival-time RMS  -2.073%
```

旧 CUDA 系综的径向直方图也只有 11/12 个 active bin 通过三标准误差门禁。
Molière 角度采样本身已经逐点通过 PROPOSAL `MoliereInterpol` oracle，因此
问题更可能位于散射角与磁场轨迹的组合，而不是角分布。

## 2. 找到的方向合成差异

标量 CORSIKA 8 的 `Step` 在构造时已经保存曲线 tracking 的方向增量：

```cpp
track.getDirection(1) - particle.getDirection()
```

`proposal::ContinuousProcess::scatter()` 又相对 step 起点方向生成散射方向，
并只追加：

```cpp
scattered_from_start - start
```

最终 `Step::getDirectionPost()` 的标量语义因此是：

\[
\hat u_\mathrm{CPU} =
\operatorname{normalize}\left(
  \hat u_\mathrm{tracked}
  + \hat u_\mathrm{scattered\ from\ start}
  - \hat u_\mathrm{start}
\right).
\]

旧 CUDA 路径采用的是另一种、看似自然但不等价的顺序：

```text
先得到 magnetic endpoint，再围绕 endpoint 旋转 Molière 角
```

两种写法在零磁场或一阶小角近似下相同，但在低能电子的有限磁偏转和多重
散射同时发生时并不相同。原验收要求是复现当前 CPU CORSIKA 8 的统计输出，
所以生产 CUDA 路径必须采用标量 `Step` 的合成语义。

## 3. 实现

新增设备/主机共用函数：

```cpp
applyMoliereDirectionCpuStepCompatible(
    start_direction,
    tracked_direction,
    angle_rad,
    azimuth_uniform)
```

它先用现有、已通过 PROPOSAL oracle 的 `applyMoliereDirection()` 相对起点
方向生成散射方向，再执行上述增量相加和一次归一化。

`CudaLeptonTransport.cu::applyMoliereScattering()` 已切换到该函数。随机数 key、
角度采样、位置、飞行时间、能损和 wavefront 顺序都没有改变。

集成测试不调用该组合 helper 来构造期望值，而是独立执行：

1. 相对起点的 Molière 旋转；
2. `tracked + scattered - start`；
3. 独立归一化；
4. 逐分量比较设备输出；
5. 证明至少一个非零磁场样本与旧的 endpoint 顺序旋转确实不同。

真实 RTX 4060 结果：

```text
GPU lepton transport validation:
2465 checks, 6 records, 1 intentional fallback, PASS
```

## 4. 严格事件与完整回归

修正后重新运行 10 GeV、0.5 MeV cut、无 thinning、`gpu-min-batch=1`
的纯电子事件：

```text
/tmp/c8_phase50_cpu_step_moliere_strict_smoke_v1
```

独立能量账本结果：

```text
complete coverage             true
internally consistent         true
relative closure error        0
accepted                      true
```

所有 25 个 C++/CUDA 专项测试在真实设备上通过；7 个非 GPU CTest 也通过。
验证脚本测试扩展到 30 个并全部通过。

本阶段还修复了完整 `all` target 的一个混合工具链问题。SIBYLL 和 SOPHIA
共享库同时含 C++ 与 Fortran 对象；在 `corsika_venv` 中若由系统 C++ driver
链接，conda Fortran 的 `libm` linker script 会错误地在 host `/lib64` 查找。
两个共享 target 现在显式使用 Fortran linker driver，从而继承匹配的 conda
sysroot，同时由 CMake 补入 C++ runtime。结果：

```text
Sibyll shared target       PASS
Sophia shared target       PASS
complete Release all build PASS
32/32 total CTest          PASS
30/30 Python tests         PASS
```

## 5. 200+200 个 1 PeV 独立系综

CPU 沿用 Phase 49 后保存的 16 个 PROPOSAL shard，共 200 个事件。修正后的
CUDA 使用同一正式配置重新生成 200 个事件：

```text
primary                  electron
energy                   1 PeV
zenith                   0 degrees
EM cut                   0.5 MeV
EM thinning              1e-4
maximum weight           100
CUDA seed                96001
gpu-min-batch            64
```

输出：

```text
CUDA:
/tmp/c8_phase50_cpu_step_moliere_physics_1PeV_cuda200_v1

comparison:
/tmp/c8_phase50_cpu_step_moliere_physics_1PeV_200_compare_v1
```

完整性统计：

```text
complete CUDA showers                  200/200
process registry accepted              200/200
GPU particles                    3,401,628,077
CPU scalar/fallback particle steps       124,487
CPU specified final states               104,197
CPU generic fallbacks                          8
memory spill / queue overflow                  0
profile overflow / invalid record              0
peak device memory                  4,848,267,238 bytes
summed common shower time                 1321.70 s
```

所有 CPU fallback 均有 process 和 reason provenance。主要指定末态为
photoproduction、epair、photonuclear 和 ionization；没有静默删除过程。

## 6. 复验结果

所有纵向曲线及三类地面直方图均通过：

```text
charged/e-/e+/photon/EM profiles      100% active bins PASS
energy deposit                        100% active bins PASS
ground radial fraction                100% active bins PASS
ground energy fraction                100% active bins PASS
ground time-residual fraction         100% active bins PASS
```

方向修正前后的相关地面量如下：

| observable | 修正前 CUDA-CPU | 修正后 CUDA-CPU | 修正后 z |
|---|---:|---:|---:|
| ground radius mean | -1.851% | **-0.492%** | -0.483 |
| ground radius RMS | -1.887% | **-0.677%** | -0.788 |
| ground time RMS | -2.073% | **-1.080%** | -1.486 |

径向直方图从 11/12 active bins 通过改善为 12/12，说明该修正不仅改变一个
汇总量，而且关闭了原先一致的 lateral-shape 信号。

原计划的 9 个 key scalar 中 7 个通过 1% 和统计双门禁：

```text
charged Xmax                     +0.646%  PASS
charged maximum                           PASS
charged longitudinal integral             PASS
photon longitudinal integral              PASS
deposited energy                           PASS
deposit-curve Xmax               +0.843%  PASS
approximate energy closure                 PASS
```

两个 ground-tail 均值仍未达到 1% 的原始样本均值门禁，但都远低于一个标准
误差，不能解释为已分辨的物理偏差：

| observable | CUDA-CPU | z | 95% bootstrap relative interval |
|---|---:|---:|---:|
| weighted ground EM count | +2.773% | 0.618 | [-5.83%, +11.94%] |
| weighted ground EM kinetic energy | +4.584% | 0.649 | [-8.72%, +19.12%] |

因此严格总状态仍为 `FAIL`，原因明确标记为
`relative_threshold_failed_but_statistically_inconclusive`，而不是把有限样本
波动误写成物理失败或物理通过。按当前方差，三标准误差带缩小到 1% 分别约需
36,282 和 89,826 个独立事件/后端。

## 7. 新增统计稳定性工具

新增：

```text
validation/gpu_em/analyze_scalar_stability.py
```

它从 `per_shower_observables.csv` 计算：

- 有符号相对均值差；
- 独立样本 z-score；
- 可重复 bootstrap 置信区间；
- 达到指定 \(k\sigma\) 相对精度所需的每后端事件数；
- `equivalent`、`different` 或 `inconclusive` 诊断。

该工具不覆盖 `compare_ensembles.py` 的严格门禁，只用于区分“已分辨偏差”和
“统计精度不足”。4 个专门单元测试已加入 30 项 Python 回归。

## 8. 当前结论

磁场与 Molière 散射的方向合成曾是一个真实的 CPU/GPU 实现差异，现已修复，
并由点级独立 oracle、严格能量闭合、完整回归和 200+200 shower 的地面分布
共同验证。1 PeV 的所有 shape gate 与 7/9 key mean gate 已通过；剩余两个
高方差 ground-tail 均值仍保持“统计不充分”，不能宣告完整 1 PeV 验收通过。
