# Phase 52：跨 shower 常驻 CUDA 后端

## 1. 性能问题

此前 `c8_air_shower` 在事件循环内部执行：

```cpp
CudaEmBackend backend;
backend.initialize(environment, table, config);
```

这意味着每个 shower 都重复：

- 读取并校验 PROPOSAL 表；
- 上传 rate、inverse-CDF、continuous、LPM 和 Molière 数据；
- 分配粒子 SoA、scan storage、cross-species queue；
- 分配两个 physical workspace；
- 分配并清零 profile/radio accumulator。

这不影响单个 UHE shower 的高加速比，但会严重拖慢 1 TeV 这类短 shower
系综，而且已经扩容的 workspace 在每个事件结束后都会被释放。

## 2. 新生命周期

`CudaEmBackend` 现在提供：

```cpp
initialize(environment, tables, invariant_config);
beginShower(per_shower_config);
```

`initialize()` 只执行一次，并直接开始第一个 shower。`beginShower()` 只更新：

- `random_seed` 与 `shower_id`；
- EM thinning threshold/maximum weight；
- 依赖 primary energy 的 profile fixed-point weight/energy range。

以下资源跨 shower 保持常驻：

- versioned PROPOSAL 物理表与 Molière 插值；
- 双缓冲 SoA 和 CUB scan storage；
- 两个已经扩容的 physical workspace；
- resident photon/lepton cross-species queue；
- ShowerAxis support、profile histogram allocation；
- radio refractivity、observer 和 waveform allocation。

应用的后续事件也不再重复构造 ShowerAxis support 或 radio observer snapshot。

## 3. 安全边界

`beginShower()` 不是无条件清空接口。它在改变任何状态前强制要求：

1. host staging 与 toy wavefront 为空；
2. resident photon/lepton queue 为空；
3. device profile 已下载；
4. 启用 GPU radio 时 waveform 已下载；
5. 新 thinning 与 fixed-point 范围有限且为正。

任一条件不满足都会抛出异常并把当前 shower 标记为 incomplete，不能静默
丢弃前一事件的粒子或输出。

通过边界检查后，它 drain 异步 profile/radio stream，清零 histogram、
profile counters、radio waveform/counters，重置 logical queue head、history
allocator 和所有 per-shower statistics。设备表和分配不会释放。

新增 provenance：

```yaml
backend_lifecycle:
  reused: true
  shower_ordinal: 2
  one_time_initialization_ms: ...
  static_host_to_device_bytes: ...
```

## 4. 测试

真实 RTX 4060 上新增/扩展三项测试：

1. 相同 seed/shower ID 的 interaction selection 在复用前后逐项一致；
2. resident profile 第二个 shower 的四组 histogram、计数器和能量账本与
   第一个 shower 逐位一致；
3. CoREAS/ZHS reset 后 waveform 逐位重复，track/pair 统计不包含上一事件。

结果：

```text
testGpuInteractionSelection  PASS
testGpuPhotonWavefront       PASS
testGpuRadioProjection       PASS
```

应用级 3-shower smoke 也证明 lifecycle ordinal 为 1/2/3，后两个事件
`reused=true`，且 3/3 输出完整。

## 5. 应用级等价性与性能

使用 Phase 51 完全相同的 1 TeV photon、seed 98001 配置重跑 100 个事件：

```text
/tmp/c8_phase52_backend_reuse_photon_1TeV_100_v1
```

把新输出与旧的每事件重新初始化实现的前 100 个同 seed 事件配对后：

```text
per-shower scalar columns       26
bitwise-identical columns       26/26
different scalar cells          0
complete showers                100/100
```

共同 shower timing：

```text
old per-event initialization implementation   91.549 s / 100
resident reusable backend                     56.992 s / 100
speedup from lifecycle refactor                1.606x
```

整个新进程（含一次 CORSIKA/模型初始化、一次表上传和全部输出）为 62.852 s。
这里的额外收益不仅来自避免重复上传，还来自保留已增长的 workspace，使后续
事件不再重复 `cudaMalloc`。

## 6. 当前结论

常驻后端已经从测试级优化进入 production application，且不改变同 seed
物理输出。它显著改善短 shower 系综吞吐量；单个 UHE shower 的原有高性能
路径和失败策略不变。
