# CUDA 电磁后端重构记录：阶段 4，端到端 toy 混合路由

## 1. 本阶段完成了什么

本阶段第一次把 CORSIKA 8 的 CPU 主栈和阶段 3 的 CUDA wavefront 后端接到
同一个 `HybridCascade::run()` 循环中。当前执行关系为：

```text
CPU 主栈
  │
  ├─ 非 EM 粒子
  │    └─ ScalarCascadeStepper::advance()
  │
  └─ gamma / electron / positron
       ├─ 显式转换单位和 transport identity
       ├─ CudaEmBackend::enqueue()
       ├─ toy CUDA wavefront
       ├─ 下载当前 GPU 输出
       └─ 以原 identity 导回 CPU 主栈
```

这里的 CUDA kernel 仍然是阶段 3 的 toy branching process，不包含真实
bremsstrahlung、pair production、ionization 或 multiple scattering。
因此本阶段证明的是调度、数据所有权和身份守恒，不是电磁物理正确性，也不会
带来生产性能提升。

## 2. 为什么目录看起来像多个 GPU 项目

清理前 `/home/yuhanglu/21CMA` 中的相似名称实际上分为三类：

| 路径 | 类型 | 处理 |
|---|---|---|
| `corsika8_gpu_refactor` | 唯一的 Git 源码工作树，分支 `cuda-em-refactor` | 保留 |
| `corsika8_gpu_refactor_build_clean` | `CUDA=OFF` 的 CPU CMake 构建缓存 | 保留 |
| `corsika8_gpu_refactor_build_cuda` | `CUDA=ON` 的 CUDA CMake 构建缓存 | 保留 |
| `corsika8_gpu_refactor_build` | 已被 clean CPU build 取代的旧缓存 | 删除 |
| `corsika8_gpu_refactor_system_build` | 已被 Conda/Conan build 取代的旧缓存 | 删除 |
| `cuda_beginner_tutorial` | 独立的 CUDA 入门教程源码 | 保留 |

因此现在不是三个 CORSIKA GPU 项目，而是“一份源码 + 两套互斥配置的构建
缓存”。CPU 与 CUDA 必须使用不同 build tree，因为 CMake 的语言、编译器和
target 集合不同，混用同一个缓存很容易产生错误结果。

本次删除的两个旧 build tree 共释放约 `1.16 GiB`：

```text
corsika8_gpu_refactor_build         约 712 MiB
corsika8_gpu_refactor_system_build  约 452 MiB
```

删除的是可由 CMake 重新生成的编译产物，不是源码；它们不能从回收站恢复，
但随时可以重新配置构建。原始项目
`/home/yuhanglu/21CMA/corsika-21cma/corsika` 没有被修改或清理。

## 3. `HybridCascade` 的可选 GPU router

实现文件：

```text
corsika/framework/core/HybridCascade.hpp
corsika/detail/framework/core/HybridCascade.inl
```

模板增加了第五个、带默认值的参数：

```cpp
template <
    typename TTracking,
    typename TProcessList,
    typename TOutput,
    typename TStack,
    typename TEmRouter = DisabledHybridEmRouter>
class HybridCascade;
```

不传 router 的旧四参数实例仍编译为 CPU-only 路径，没有虚函数、运行时类型
判断或 CUDA 链接依赖。这保证阶段 2 的固定 seed 等价测试仍然有效。

传入 router 后，每次 scheduler 从 CPU 栈取出粒子：

1. `canRoute()` 判断是否为 gamma、electron 或 positron；
2. 对 EM 粒子调用 `stage()`；
3. 只有 staging 成功后才从 CPU 栈删除该粒子；
4. 非 EM 粒子继续调用同一个 `ScalarCascadeStepper`；
5. CPU 栈暂时为空而 GPU 仍有粒子时，推进一个 GPU wavefront；
6. 下载的粒子进入 CPU 栈，再重新设置环境 node；
7. CPU 栈和 GPU 队列都为空后，才调用 cascade-equation 过程。

`CpuOnlyWavefrontScheduler` 的 acquire/complete 配对仍然覆盖每一个被路由的
EM step，便于检查丢粒子、重复粒子和未完成 step。

## 4. CPU/GPU 边界和固定单位

开发适配器位于：

```text
corsika/gpu/em/ToyCudaEmRouter.hpp
```

它只路由 `is_em(pid)` 为真的粒子，并在边界执行以下显式转换：

| CORSIKA CPU 状态 | `EmParticleState` |
|---|---|
| `Code` / PDG enum | `int32_t pid` |
| total energy | `double energy_GeV` |
| position with physical units | `double position_m[3]` |
| coordinate-aware direction | `double direction[3]` |
| time with physical units | `double time_s` |
| thinning weight（若 stack 支持） | `double weight` |
| transport lineage | 固定宽度 ID 字段 |

回到 CPU 时，router 将 PDG 转回 `Code`，从总能量减去静质量得到
`Stack::addParticle()` 所需的动能，并重建位置、方向、时间和可选权重。
若 GPU 返回总能量低于静质量、未知 PID 或非 unit weight 而目标 stack 没有
weight 字段，会立即抛出异常。

## 5. transport identity 回导

GPU 分支会创建新的 history，不能在回到 CPU 时重新自动编号，否则 Philox key
和完整级联谱系都会改变。为此新增：

```cpp
particle.setTransportIdentity(TransportIdentity{
    history_id,
    parent_history_id,
    generation,
    step_id
});
```

实现文件：

```text
corsika/stack/TransportIdentityStackExtension.hpp
corsika/detail/stack/TransportIdentityStackExtension.inl
```

导入时验证：

- `history_id` 不能为 0；
- primary 必须同时满足 `generation == 0` 和 `parent_history_id == 0`；
- secondary 必须同时满足 `generation > 0` 和 `parent_history_id > 0`；
- CPU 的下一 ID 分配器必须推进到所有已导入 ID 之后，防止未来 secondary
  与 GPU history 冲突；
- history ID 达到 `uint64_t` 上限后进入 exhausted 状态，而不是回绕。

测试显式导入 `{history=42, parent=7, generation=3, step=9}`，随后在 CPU
创建的 secondary 获得 history 43、parent 42、generation 4。

## 6. 测试专用下载接口

`CudaEmBackend` 暂时增加：

```cpp
std::vector<EmParticleState> extractActiveParticlesForTesting();
```

它下载当前 active SoA、清空设备 current queue 并转移其所有权。若 host
staging 尚未上传则拒绝执行，避免把两组不同阶段的粒子混在一起。

该接口名称明确包含 `ForTesting`，因为生产 scheduler 不应在每个 wavefront
后把全部 EM 粒子拉回 CPU。它只是让现阶段可以完整验证：

- CPU stack → GPU SoA；
- GPU branching 和 identity 分配；
- GPU SoA → CPU stack；
- 再次路由；
- 最终双端队列排空。

## 7. 集成测试和实机结果

新增测试：

```text
tests/gpu/testGpuHybridRoute.cpp
```

初始 CPU 栈放入一个 proton 和一个 photon：

- proton 必须且只能执行一个 CPU tracking/continuous step，然后被吸收；
- photon 必须进入 GPU，不能调用 CPU scalar step；
- toy GPU 输出继续通过 CPU 栈重新进入后续 wavefront；
- 结束时 CPU stack 和 GPU queue 必须同时为空；
- scheduler acquire/complete、router staging/return 和 backend
  advanced/produced 计数必须守恒。

RTX 4060 Laptop GPU 上的结果：

```text
GPU hybrid route passed 11 checks:
  5 EM particle steps
  4 wavefronts
  maximum batch 2
```

同时通过：

```text
TransportIdentity  26 assertions
HybridStack         9 assertions
HybridCascade      12 assertions（固定 RNG 的 CPU 等价路径）
testGpuEmHost       passed
testGpuEmCppLink    passed
testGpuEmCuda       11 deterministic toy outputs
```

## 8. 当前构建方式

CUDA build 使用 `corsika_venv`，并显式移除环境中的 `FLUPRO`，防止 CMake
因为环境变量存在而意外启用 FLUKA。当前 Conan 依赖为 `RelWithDebInfo`
提供了完整 target 属性，所以持续开发使用同一 build type：

```bash
/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  -DCORSIKA_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPYTHON_EXECUTABLE=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python3

/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv env -u FLUPRO cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuHybridRoute -j2
```

指定 `PYTHON_EXECUTABLE` 还不够：粒子头文件生成脚本通过 shebang 启动
Python，因此整个 build 命令也必须运行在 `corsika_venv` 的 `PATH` 中。

## 9. 当前限制

- 仍然使用 toy branching，没有真实 EM 相互作用率或末态；
- 每个 GPU wavefront 后都把所有输出下载回 CPU，性能上是刻意低效的；
- 尚未按 `PID × medium_id × energy_bin` 分桶；
- 没有 min-batch 策略、异步 stream、pinned staging 或 CPU 稀有过程回退；
- GPU 返回后，CPU stack 的 erased storage 会在 purge 前累计；
- `EnvironmentSnapshot` 和 `ProposalTableSet` 仍只有骨架；
- 当前 router 不接入 `c8_air_shower` CLI，默认 `Cascade + PROPOSAL`
  完全不变。

因此本阶段不能用于 shower 科研结果，也不能用于报告 GPU 加速比。

## 10. 下一步重构

下一阶段进入 PROPOSAL 的“率采样”和“指定末态生成”拆分，但应先在 CPU 上
完成，不立即移植 CUDA kernel：

1. 找出当前 `PROPOSAL::doInteraction()` 中反应率、过程/组分选择、`v`
   采样和 `CalculateSecondaries()` 的准确调用链；
2. 定义 `ProposalInteractionRecord`，保存过程、component hash、`v`、顶点
   状态和随机 key；
3. 实现 `ProposalRateProvider` 和 `ProposalFinalStateGenerator`；
4. 用新接口在 CPU 上重建旧 `doInteraction()`；
5. 固定 seed 比较旧路径和新路径的过程选择、末态、能量守恒及 RNG 消费；
6. 等 CPU 拆分等价后，再设计 `gpu_em_tablegen` 的版本化表格 schema。

这一顺序能先隔离最敏感的物理语义，再把经过验证的数据边界移到 GPU。

上述 CPU 拆分已经完成，见
[phase_05_proposal_interaction_split.md](phase_05_proposal_interaction_split.md)。
