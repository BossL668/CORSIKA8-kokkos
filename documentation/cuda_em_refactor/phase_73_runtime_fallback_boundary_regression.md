# Phase 73：运行时几何回退与数值失败边界回归

## 1. 目的

Phase 57 已经建立 production strict fallback 策略，但原
`testGpuHybridRoute` 主要验证：

- `UnsupportedGeometry` 位于允许列表；
- `InvalidMassDensity` 和 `AtmosphereGrammageFailed` 不在允许列表；
- 一般 GPU fallback 数量与 scalar bypass 数量相等。

这些检查没有直接证明两个完整运行时行为：

1. device kernel 实际返回 `UnsupportedGeometry` 后，粒子是否以相同
   history/step 回到 `HybridCascade`，并且只执行一个 CPU scalar step；
2. device kernel 实际返回大气数值错误时，production strict 模式是否在
   `HybridCascade::run()` 中停止，而不是静默回到 CPU。

Phase 73 为这两条路径增加确定性的端到端回归。

## 2. `UnsupportedGeometry` 单步回退

测试把 10 MeV photon 精确放在五层大气最外边界，并令方向沿径向向外：

```text
radius       = outermost atmosphere radius
direction    = (0, 0, +1)
energy       = 10 MeV
min batch    = 1
strict mode  = true
```

`queryAtmosphereLayer()` 在统一的浮点边界保护内仍把该状态识别为最外层，
所以它可以进入 GPU。随后 `distanceToAtmosphereBoundary()` 找不到大于
0.1 mm 保护的前向球面交点，photon transport kernel 明确返回：

```text
reason = unsupported_geometry
```

完整调用链为：

```text
HybridCascade::canRoute
  -> PhysicalCudaEmRouter::stage
  -> resident photon CUDA pipeline
  -> ProposalFallbackEvent(UnsupportedGeometry)
  -> PhysicalCudaEmRouter::returnFallbacks
  -> importParticle with the original history_id/step_id
  -> canRoute consumes cpu_fallback_steps_ marker
  -> ScalarCascadeStepper::advance exactly once
```

测试要求同时满足：

```text
particles_staged                         1
GPU wavefronts                           1
particles_returned_for_cpu_fallback      1
cpu_fallback_steps_executed              1
fallback reason UnsupportedGeometry      1
tracking calls                           1
scalar continuous calls                  1
scheduler acquired/completed steps       2 / 2
remaining CPU/GPU particles              0 / 0
```

这里 scheduler 的两个 step 分别是首次 GPU ownership 和返回后的单次 CPU
ownership。任何重复回退、重复推进或粒子丢失都会使计数不成立。

## 3. 大气数值失败必须终止

第二个测试保留一个结构上合法的五层 snapshot，但把第一层指数密度参数设为会在
查询点产生浮点上溢的值。snapshot schema 仍通过
`validEnvironment()`，因此错误只能在真实 transport 数值求值时暴露。

10 MeV photon 进入 device 后，大气柱深计算返回：

```text
reason             = atmosphere_grammage_failed
diagnostic_status  = InvalidSnapshot
diagnostic density = inf
```

router 启用：

```cpp
setFailOnUnexpectedFallback(true);
```

测试要求 `HybridCascade::run()` 抛出包含 reason 和诊断信息的
`std::runtime_error`，并验证：

```text
particles_returned_for_cpu_fallback = 0
cpu_fallback_steps_executed         = 0
tracking calls                      = 0
scalar continuous calls             = 0
```

所以这个错误不能被 CPU PROPOSAL 重试掩盖。

## 4. 验证

目标测试：

```bash
cmake --build /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --target testGpuHybridRoute --parallel 8

ctest --test-dir \
  /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --output-on-failure -R '^testGpuHybridRoute$'
```

结果：

```text
testGpuHybridRoute  PASS
```

运行日志同时出现预期的两条可审计记录：

```text
reason=unsupported_geometry
reason=atmosphere_grammage_failed, diagnostic density=inf
```

在加入回归前后的完整基线仍为：

```text
CUDA Release build all target   PASS
CUDA CTest                      32/32
CPU-only build all target       PASS
CPU-only CTest                  10/10
Python validation               40/40
```

## 5. 结论

允许的环境能力边界和禁止静默接受的数值错误现在都具有真实 CUDA kernel 到
`HybridCascade` 的运行时证据。该测试不是只检查枚举 allow-list，而是覆盖
device fallback、host import、transport identity、scheduler ownership 和
scalar stepper 的完整路径。
