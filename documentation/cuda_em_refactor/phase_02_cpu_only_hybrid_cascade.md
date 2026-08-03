# CUDA 电磁后端重构记录：阶段 2，CPU-only HybridCascade

## 1. 本阶段目标

本阶段仍不执行 CUDA 物理计算。目标是在不改变默认 CPU 路径的前提下建立三个后续 GPU 后端必需的边界：

1. 每个粒子 history 都有稳定的 `history_id`、`parent_history_id`、
   `generation` 和 `step_id`；
2. 栈选择与单粒子输运之间增加 wavefront scheduler 接口；
3. 新增与 `Cascade` 外部接口一致的 `HybridCascade`，但所有粒子暂时仍由
   `ScalarCascadeStepper` 处理。

当前执行关系为：

```text
HybridCascade
  │
  ├─ CpuOnlyWavefrontScheduler::acquireNext()
  │      ├─ 保持原 LIFO getNextParticle()
  │      ├─ 读取 history / generation
  │      └─ 分配本次 step_id
  │
  ├─ ScalarCascadeStepper::advance()
  ├─ scheduler.completeParticleStep()
  └─ ProcessSequence::doStack()
```

wavefront 宽度当前固定为 1，因此随机数调用顺序和原 `Cascade` 一致。

## 2. Transport identity

实现文件：

```text
corsika/stack/TransportIdentityStackExtension.hpp
corsika/detail/stack/TransportIdentityStackExtension.inl
```

每个栈条目增加以下 POD 元数据：

```cpp
struct TransportIdentity {
  std::uint64_t history_id;
  std::uint64_t parent_history_id;
  std::uint32_t generation;
  std::uint64_t step_id;
};
```

该结构经过 `standard_layout` 和 `trivially_copyable` 编译期检查，后续可以直接映射到设备粒子状态。

### 2.1 ID 语义

- `history_id`：一个 shower 内单调递增，从 1 开始；
- `parent_history_id`：primary 为 0，secondary 指向直接母粒子的
  `history_id`；
- `generation`：primary 为 0，每一代 secondary 加 1；
- `step_id`：该 history 已经开始的输运推进次数。

调度器调用 `beginTransportStep()` 时，返回本次推进使用的旧 `step_id`，随后把粒子上的计数加 1。因此第一次推进使用 step 0，第二次使用 step 1。这个返回值以后可以直接进入 Philox key。

### 2.2 稳定性

identity 是 `CombinedStack` 的数据扩展，而不是以 stack index 为键的外部 map。因此：

- 创建 primary 时自动分配新 ID；
- `SecondaryView::addSecondary()` 和直接 `addSecondary()` 都自动继承谱系；
- stack `swap()`、`copy()`、`purge()` 时 identity 与粒子其他数据一起移动；
- `stack.clear()` 后 ID 计数器重新从 1 开始；
- 被 thinning/cut 立即删除的 secondary 仍然消耗一个 ID，因而 ID 可以有空洞，但不会复用。

这避免了 stack compaction 后 sidecar map 指向错误粒子的风险。

## 3. 默认 Stack 保持不变

`setup::detail::StackGenerator` 新增：

```cpp
StackWithTransportIdentity
StackWithTransportIdentityAndHistory
```

公开别名新增：

```cpp
setup::HybridStack<TEnvironment>
setup::HybridStackView<TEnvironment>
```

现有 `setup::Stack<TEnvironment>` 和 `setup::StackView<TEnvironment>` 没有修改。因而当前 `c8_air_shower` 仍使用原 CPU Stack，不会因为本阶段重构增加默认事件的内存占用。

当 CORSIKA history 功能开启时，`HybridStack` 的组合顺序为：

```text
Vector data
  + geometry node
  + thinning weight
  + transport identity
  + full cascade history
```

带 history 和不带 history 的两种组合均有编译和 secondary 谱系测试。

## 4. CPU-only wavefront scheduler

实现文件：

```text
corsika/framework/core/CpuOnlyWavefrontScheduler.hpp
```

调度器返回一个 `ScheduledParticle`：

```cpp
struct ScheduledParticle {
  particle_type particle;
  HistoryId history_id;
  HistoryId parent_history_id;
  Generation generation;
  StepId step_id;
};
```

它目前严格调用原来的 `stack.getNextParticle()`，一次只允许一个 in-flight scalar step，并记录：

- acquired particle steps；
- completed particle steps；
- maximum wavefront size。

这一接口刻意没有修改现有 `Stack` 类，也没有让 Stack 变成并发容器。未来真正的 hybrid scheduler 可以在这个边界把 EM history 转入独立的双缓冲 SoA 队列。

## 5. HybridCascade

实现文件：

```text
corsika/framework/core/HybridCascade.hpp
corsika/detail/framework/core/HybridCascade.inl
```

公开接口与当前 `Cascade` 对齐：

```cpp
HybridCascade(env, tracking, processes, output, stack);
void setNodes();
void run();
void forceInteraction();
void forceDecay();
```

另外提供只读的 `schedulerStatistics()`。

当前 `run()` 保留：

- `startOfShower()` / `endOfShower()`；
- 原两层 cascade-equation 循环；
- 原 `initCascadeEquations()`、`doStack()`、`doCascadeEquations()` 调用位置；
- 原 LIFO 粒子顺序；
- 原 `ScalarCascadeStepper` 物理推进；
- 原 `"cascade"` RNG stream。

唯一新增的运行时操作是 identity/step 计数，不抽取随机数，也不修改粒子物理状态。

## 6. 验证结果

### 6.1 Identity 与 scheduler

- transport lineage：18 个断言通过；
- CPU-only scheduler：12 个断言通过；
- `setup::HybridStack`，包含显式 full-history 组合：9 个断言通过。

覆盖了 primary、两代 secondary、`SecondaryView`、step 递增、stack purge、
clear 后 ID 重置、空栈错误和 scheduler 完成状态。

### 6.2 Cascade 与 HybridCascade 等价性

使用固定 CORSIKA seed `0x5eed1234`，分别运行旧 `Cascade` 和
CPU-only `HybridCascade` 的 100 GeV Heitler shower，结果为：

- tracking calls：2047，两者相同；
- interaction calls：2047，两者相同；
- cut particles：2048，两者相同；
- cut calls：2047，两者相同；
- 最终 RNG engine state：逐字符串完全相同；
- Hybrid acquired/completed steps：2047/2047；
- maximum wavefront size：1。

这不仅比较了统计计数，还验证了完整随机数消费位置没有变化。

### 6.3 默认应用回归

- `testFramework`：全部通过；
- `c8_air_shower`：完整模板重新编译和链接通过；
- `corsika_venv` 中执行 CTest：通过；
- 固定 seed 12345 的 10 GeV electron shower 与阶段 1 输出比较：
  除 `config.yaml` 中的输出路径，以及 `summary.yaml` 中的开始时间、结束时间和运行时外，其余所有文件逐字节相同。

## 7. 当前限制

- 没有 CUDA target、device allocation 或 kernel；
- wavefront size 固定为 1，尚未改变深度优先执行；
- transport ID 分配器当前是 CPU-only，不支持设备并发分配；
- 尚未实现把指定 ID 的 GPU fallback history 重新导入主栈；
- `HybridCascade` 尚未接入 `c8_air_shower` CLI；
- 默认 `proposal` 路径仍使用旧 `Cascade`，这是刻意保留的行为。

## 8. 下一阶段

下一阶段进入 GPU 基础设施，但仍先使用 toy physics：

1. 增加默认关闭的 `CORSIKA_ENABLE_CUDA`；
2. 增加编译型静态库 `CORSIKA8GpuEm`；
3. 定义固定单位、标准布局的 `EmParticleState` 和批量结果 POD；
4. 实现 CUDA/CPU 共用的 counter-based Philox key；
5. 建立 host staging、双缓冲 device queue 和容量检查；
6. 实现 toy branching kernel、稳定 secondary count 和 scan/compaction；
7. 在进入 PROPOSAL 表格或真实 EM 物理前，先验证确定性、溢出和至少
   `10^7` 个 toy 粒子的队列完整性。

实现记录见
[phase_03_cuda_wavefront_infrastructure.md](phase_03_cuda_wavefront_infrastructure.md)。
