# Phase 100：beta4 GPU 10 ms `ParticleCut` 与 CPU 时间语义对齐

## 1. 问题确认

标量 `ParticleCut` 不只检查粒子动能，还在次级粒子产生后以及每个连续传播步
结束后执行严格条件：

```cpp
timePost > 10_ms
```

这里的时间是 shower 事件内的物理传播时间，不是程序墙钟时间或 CPU 核时。
GPU 粒子状态虽然一直保存并推进 `time_s`，但 beta4 此前只实现了能量 cut，
没有执行 10 ms 时间 cut。因此，超过 10 ms 的 GPU 生成次级粒子仍可能查表，
一次传播跨过 10 ms 后也可能继续执行离散末态。该问题来自早期 CUDA
`ParticleCut` 只移植能量条件，并非本阶段之前的观测平面修复引入。

## 2. 对齐后的精确语义

设备端使用与 CPU 相同的常量和严格比较：

```cpp
inline constexpr double ParticleCutMaximumTimeS = 10.e-3;
return time_s > ParticleCutMaximumTimeS;
```

- 恰好 `10 ms` 不切除；只有大于 `10 ms` 才切除。
- 次级粒子继承顶点的全局事件时间；若产生时已经超过阈值，在任何 PROPOSAL
  表查询和 Philox 随机抽样之前终止。
- 若一个在阈值内开始的传播步在终点超过 10 ms，保留完整的物理步，不把轨迹
  人工截短到 10 ms；连续能损、多重散射、纵向 profile 和射电轨迹仍处理该步，
  随后的离散相互作用或衰变被取消。
- photon cut 沉积其剩余总能量；带电轻子 cut 沉积连续损失后的剩余动能，不把
  静质量混入 `dEdX`，与 CPU `ParticleCut` 一致。
- `c8_air_shower` 的标量过程序列中 `ObservationPlane` 位于 `ParticleCut` 之前。
  因此，同一步若先到达观测面且终点时间超过 10 ms，GPU 同时输出 ground
  observation 并执行 cut，而不是丢失观测记录。

## 3. 修改位置

- `corsika/gpu/em/Types.hpp`：统一的 10 ms 常量、严格判断函数，以及
  observation-before-cut 的 POD 标志。
- `src/gpu/em/CudaInteractionSelector.cu`：迟发次级粒子在查表和抽取随机数前
  进入 `ParticleCut`。
- `src/gpu/em/CudaPhotonTransport.cu`：photon 相互作用、层边界、逃逸和观测面
  步后的时间检查。
- `src/gpu/em/CudaLeptonTransport.cu`：连续能损和带质量飞行时间更新后的时间
  检查；优先于离散相互作用和衰变末态。
- `src/gpu/em/CudaPhotonSelectionTransport.cu` 与
  `src/gpu/em/CudaLeptonSelectionTransport.cu`：保留观测面先于 cut 的双重结果。
- `corsika/gpu/em/detail/DeviceBatchStages.hpp` 与
  `src/gpu/em/CudaEmBackend.cu`：把 observation+cut 标记为一个输运终点，避免
  endpoint accounting 把同一个源粒子重复相减。
- `src/gpu/em/CudaProfileProjection.cu`：resident profile 同时登记 terminal
  observation energy、cut deposit 和带电轻子的 cut rest-mass ledger。

## 4. 验收

新增边界测试覆盖：

1. `time_s == 10 ms` 与普通选择结果相同；
2. `nextafter(10 ms, +inf)` 必须直接得到 `ParticleCut`，不产生 fallback；
3. photon 和 electron 从 10 ms 以内开始，在传播步终点跨过阈值；
4. 跨阈值的同一步到达观测平面时，既有 `ObservationRecord`，又有剩余动能
   cut deposit；
5. observation+cut 在 endpoint accounting 中只对应一个源粒子，同时保留
   两类输出和各自的能量账本；
6. cut 后不产生离散末态、continuation 或重复队列粒子；
7. 重复运行仍保持确定性。

原标量 `testParticleCut` 也增加了同一阈值两侧的控制：`10 ms` 保留，
`10 ms + 1 ns` 删除；筛选运行得到 1 个测试用例、15 个断言全部通过。

Release CUDA 全树重新构建成功，完整 GPU 矩阵为 27/27 通过。CTest 中其余
测试也直接通过；`testModules` 在裸 shell 中因没有设置运行期 `FLUPRO`
中止，显式设置
`FLUPRO=<FLUKA_ROOT>` 和 `FLUFOR=/usr/bin/gfortran` 后单独通过。因此
代码测试合计为 34/34 通过。Python 验证工具为 241/241 通过。

另运行了一个 1 GeV、`theta=80 deg` electron 的生产级 CUDA 冒烟事例：
shower 标记为 `complete`，858 个 GPU 粒子步、零 CPU fallback、零显存 spill、
零非法 profile 记录，能量账本和全部输出正常结束。该普通事例的时间远小于
10 ms，用来确认新判断不会改变阈值以内的正常运行；真正的阈值两侧行为由
上述确定性合成测试严格覆盖。

## 5. 结论

beta4 的 GPU `ParticleCut` 现在覆盖原版 CPU 的能量条件和 10 ms 物理时间
条件，并保持 `c8_air_shower` 中连续输出、观测面和 cut 的调用顺序。该修复
不修改 CPU 默认路径、不改变 PROPOSAL 表格式，也不增加运行期 CPU fallback。
