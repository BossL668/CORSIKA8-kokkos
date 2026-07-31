# Phase 44：CUDA EM 过程序列兼容性启动门禁

## 1. 问题

resident CUDA 电磁级联绕过 scalar `ProcessSequence::doContinuous()`。因此，一个
用户新增的 `ContinuousProcess` 如果观察或修改了电子、正电子或光子的
`Step`，而 CUDA 路径既没有在设备上实现它，也没有根据设备轨迹记录重放它，
模拟就会静默改变物理或输出。

仅为 `c8_air_shower` 当前固定过程列表写一个布尔标志不足以解决这个问题；
门禁必须递归检查任意模板过程树，并且在第一个 CUDA shower 开始之前拒绝未知
连续过程。

## 2. 实现

新增：

```text
corsika/gpu/em/ProcessSequenceCompatibility.hpp
```

其中定义两种明确策略：

```cpp
enum class GpuEmStepProcessPolicy {
  ReplacedOnDevice,
  ReplayedFromDeviceRecord,
};
```

`GpuEmStepProcessRegistry` 递归展开：

- 普通 `ProcessSequence`；
- `SwitchProcessSequence` 的两个分支；
- 任意叶节点 process。

每个 `ContinuousProcess` 类型必须显式注册为：

- `ReplacedOnDevice`：功能已由 resident CUDA transport 实现；
- `ReplayedFromDeviceRecord`：依据 GPU 产生的 step、radio 或 observation
  record 在 CPU 端重放。

未注册计数非零时，`validateOrThrow<TSequence>()` 抛出
`std::runtime_error`。检查发生在 `CudaEmBackend::initialize()` 之前，因此未知
过程不能运行到第一个 CUDA kernel，也不会自动切换到 PROPOSAL 后端。

interaction、decay、secondaries 与 stack process 不直接观察 scalar transport
`Step`，不属于此门禁的检查对象；它们继续由已有的 route/capability/fallback
合同处理。

## 3. `c8_air_shower` 的生产注册表

当前 CUDA 路径显式注册 6 个连续过程类型：

| 过程 | 策略 | 保留方式 |
|---|---|---|
| PROPOSAL EM continuous | device-replaced | GPU range、连续电离及散射 |
| ParticleCut | device-replaced | GPU endpoint/cut accounting |
| CoREAS | record-replayed | 等价 \(e^\pm\) 轨迹段或 CUDA radio accumulator |
| ZHS | record-replayed | 等价 \(e^\pm\) 轨迹段或 CUDA radio accumulator |
| LongitudinalProfile | record-replayed | resident profile/step record 合并 |
| ObservationPlane | record-replayed | `ObservationRecord` 路由 |

启动日志与每个 shower 的 `gpu_em/summary.yaml` 同时记录：

```yaml
process_registry:
  registrations: 6
  device_replaced: 2
  record_replayed: 4
  unregistered_continuous_processes: 0
  accepted: true
```

这使生产输出可以证明所使用的过程树确实经过了门禁，而不只是证明源代码中存在
检查器。

## 4. 自动化负向测试

`testGpuEmHost` 建立一个已注册 continuous process 和一个未知 continuous
process，并验证：

1. 普通嵌套 `ProcessSequence` 能递归发现未知过程；
2. `validateOrThrow()` 确实拒绝启动；
3. 完整注册后同一过程树被接受；
4. device-replaced 与 record-replayed 数量正确；
5. 显式注册整个 wrapper 的合同有效；
6. `SwitchProcessSequence` 两个分支也会递归发现并拒绝未知过程。

## 5. 实际验证

构建命令：

```bash
cmake --build \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target c8_air_shower testGpuEmHost -j2
```

结果：两个 target 均成功。

host 负向测试结果：

```text
GPU EM host validation passed: 178 checks
```

真实 CUDA 冒烟输出：

```text
/tmp/c8_phase44_registry_smoke_v1
```

配置为 10 GeV electron、0.5 MeV cut、无 thinning、CUDA EM、无射电天线。
结果：

- process registry：6/2/4/0，accepted；
- GPU particles：16053；
- CPU fallbacks：0；
- scalar EM steps：0；
- 严格能量账本相对闭合误差：
  \(8.239687289990963\times10^{-16}\)；
- CUDA 进程退出码：0。

因此“未知的、可观察或修改 EM Step 的自定义连续过程必须在 CUDA 模式启动
失败”这一要求已经从固定 capability 判断提升为通用、递归、具有负向测试与
运行 provenance 的启动门禁。
