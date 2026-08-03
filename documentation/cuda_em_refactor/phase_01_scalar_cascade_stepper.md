# CUDA 电磁后端重构记录：阶段 1，拆分标量单粒子推进器

## 1. 本阶段目标

本阶段不引入 CUDA，也不改变任何物理过程。唯一目标是把原来同时承担“调度”和“单粒子输运”的 `Cascade` 拆成两个职责明确的层：

```text
Cascade
  ├─ 保留：LIFO 主栈调度、doStack、cascade equations、输出生命周期
  └─ 委托：ScalarCascadeStepper::advance(particle)
                ├─ interaction / decay / geometry / continuous-step 竞争
                ├─ tracking 与介质边界
                ├─ doContinuous / doSecondaries / doBoundaryCrossing
                └─ 原有 cascade 随机数流
```

这是 GPU wavefront 调度的最低层前置重构。如果单粒子物理推进和 LIFO 栈调度没有分开，未来就无法在保持 CPU 路径不变的同时，把一批电磁粒子送入 GPU。

## 2. 隔离工作区

为避免覆盖原项目中已有的大量本地修改，本阶段使用独立 Git worktree：

- 原工作树：`/home/yuhanglu/21CMA/corsika-21cma/corsika`
- 重构工作树：`/home/yuhanglu/21CMA/corsika8_gpu_refactor`
- 重构分支：`cuda-em-refactor`
- 基准提交：`64b7c8606497ed18933fa8f67152cd3341a88863`
- 独立构建目录：`/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean`

原工作树的修改没有被重置、格式化或复制到重构分支。

## 3. 代码变化

### 3.1 新增 `ScalarCascadeStepper`

公开接口位于：

```text
corsika/framework/core/ScalarCascadeStepper.hpp
```

模板参数为：

```cpp
template <typename TTracking, typename TProcessList, typename TStack>
class ScalarCascadeStepper;
```

核心接口为：

```cpp
void setNodes();
void advance(particle_type& particle);
void forceInteraction();
void forceDecay();
```

`advance()` 只把一个粒子推进到最近的一个限制点。它不会从主栈选择下一个粒子，也不会调用 `doStack()` 或 `doCascadeEquations()`。因此，推进器既可以被当前 LIFO `Cascade` 使用，也可以在后续被 CPU wavefront scheduler 使用。

实现位于：

```text
corsika/detail/framework/core/ScalarCascadeStepper.inl
```

原 `Cascade::step()`、`decay()`、`interaction()`、`setEventType()` 和节点初始化代码被原样迁移到这里。随机流仍然是：

```cpp
RNGManager<>::getInstance().getRandomStream("cascade")
```

没有增加随机抽样，也没有改变抽样顺序。

### 3.2 `Cascade` 变为调度器

`Cascade::run()` 的两层循环保持不变：

1. 从原有栈中通过 `getNextParticle()` 取粒子；
2. 调用 `stepper_.advance(pNext)`；
3. 调用原有 `sequence_.doStack(stack_)`；
4. 栈清空后调用原有 `doCascadeEquations()`；
5. 如果 cascade equations 又产生粒子，则重新进入内层循环。

因此当前 CPU 后端仍然是原来的深度优先 LIFO 执行顺序。

`Cascade::setNodes()`、`forceInteraction()` 和 `forceDecay()` 保持原公开 API，只在内部委托给 stepper，已有应用代码不需要修改。

### 3.3 新增最低层单元测试

`tests/framework/testCascade.cpp` 新增了一个直接构造 `ScalarCascadeStepper` 的用例。该用例验证：

- `setNodes()` 可以为粒子设置环境节点；
- `advance()` 确实调用一次 tracking；
- 连续过程能够吸收该粒子；
- stepper 不需要拥有或替换现有 `Stack`。

## 4. 必须保持的物理不变量

本阶段明确保持以下条件：

- 主栈仍为原有 `Stack`，没有改成并发容器；
- 粒子选择顺序仍为原有 LIFO 顺序；
- interaction、decay、geometry 和 continuous limit 的比较关系不变；
- 边界相交后的 `setNode()` 和 `doBoundaryCrossing()` 顺序不变；
- `doContinuous()`、粒子状态写回、`doSecondaries()` 和 `erase()` 顺序不变；
- `"cascade"` 随机流的获取和每一步随机抽样顺序不变；
- `proposal` CPU 路径和所有物理模块均未修改；
- `Cascade` 的外部构造函数、`run()`、`setNodes()`、`forceInteraction()`、`forceDecay()` 接口不变。

## 5. 构建环境说明

原项目成功构建使用 Ubuntu 系统 GCC/G++/GFortran 13.3。Conan 缓存中的 Boost 也是按系统 glibc 构建的。如果在 `conda run` 内重新配置整个 C++ 工程，CMake 会选择 Conda GCC 12 和 Conda sysroot，链接时会出现：

```text
undefined reference to __isoc23_sscanf
```

因此当前采用以下边界：

- CORSIKA C/C++/Fortran：系统 GCC 13；
- CORSIKA 代码生成脚本：`corsika_venv` 中的 Python；
- CUDA Toolkit：继续由 `corsika_venv` 提供；
- 编译完成后的测试和程序：可以在 `corsika_venv` 中运行。

配置时应显式指定系统编译器和 Conda Python：

```bash
/usr/bin/cmake \
  -S /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  -B /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  -DCMAKE_TOOLCHAIN_FILE=/home/yuhanglu/21CMA/corsika-21cma/corsika/conan_cmake/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/home/yuhanglu/21CMA/corsika8_gpu_refactor_install_clean \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
  -DPYTHON_EXECUTABLE=/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python3
```

## 6. 验证结果

重构前基线：

- `testFramework`：通过；
- `[Cascade]`：2 个用例、24 个断言全部通过。

重构后：

- `testFramework`：通过；
- 原 `[Cascade]`：仍为 2 个用例、24 个断言全部通过；
- 新 `[ScalarCascadeStepper]`：1 个用例、3 个断言全部通过；
- 完整 `c8_air_shower` 模板实例：编译并链接通过；
- 在 `corsika_venv` 中执行 CTest：通过。
- 在 `corsika_venv` 中运行固定种子 `10 GeV` 电子 shower：正常结束，并完整写出
  particle、profile、energy-loss、interaction 和 radio 占位输出。

验证命令：

```bash
/usr/bin/cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  --target testFramework c8_air_shower -j2

/home/yuhanglu/miniconda3/bin/conda run \
  --no-capture-output -n corsika_venv \
  /usr/bin/ctest \
  --test-dir /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean \
  -R '^testFramework$' --output-on-failure
```

## 7. 下一步

下一步仍然不应直接写 CUDA 物理 kernel，而应继续完成调度边界：

1. 给粒子历史增加稳定的 `history_id`、`generation` 和 `step_id`；
2. 新增 CPU-only 的 wavefront scheduler；
3. 让 scheduler 可以把粒子分类为 scalar CPU 路径或未来的 EM batch 路径；
4. 新增空实现 `HybridCascade`，先只调用 `ScalarCascadeStepper`；
5. 对固定种子事件比较旧 `Cascade` 与 CPU-only `HybridCascade` 的粒子数、调用序列和输出。

只有这个 CPU-only 混合调度层通过等价验证后，才应增加 `CORSIKA8GpuEm`、设备 POD 和 CUDA toy branching kernel。

阶段 2 已完成上述 CPU-only 混合调度层，详见
[`phase_02_cpu_only_hybrid_cascade.md`](phase_02_cpu_only_hybrid_cascade.md)。
