# Phase 76：光核备用模型懒加载与当前构建性能复验

> 本阶段的 executable hash 已被 Phase 77 的近阈光致强子闭合修复取代。
> 性能结果保留为回归历史；当前生产性能证据见
> `phase_77_threshold_photoproduction_and_final_performance.md`。

## 1. 本阶段关闭的问题

Phase 75 为 PROPOSAL 光核末态增加了 SIBYLL → QGSJet-II 无损回退。
QGSJet-II 的初始化较重；若每个 application 进程无条件构造备用模型，即使某个
纯电磁事件从未触发氩靶回退，也会支付固定启动代价。

当前实现使用：

```text
LazyHadronicInteractionModel<QGSJetII>
```

适配器先查询 SIBYLL。只有 SIBYLL 拒绝已经选定的 projectile/target 时，才构造
QGSJet-II，并继续生成同一个指定光核顶点的末态。每个 CUDA shower 的 summary
同时记录：

```yaml
photo_hadronic_generator:
  preferred_interactions: ...
  fallback_interactions: ...
  discarded_final_states: 0
  fallback_model_initialized: ...
```

因此可以区分“没有发生回退”“发生了回退并初始化备用模型”和“非法丢弃末态”。

## 2. 当前二进制与物理表

本阶段所有正式证据使用：

```text
c8_air_shower SHA-256
17c0d39a5494e8aaaa8b3866b3b3de764c1f09f05a6c9f80a33ab4d87118e20f

production table SHA-256
140347373b9b5014c8b5bbd6cbcffca5f928eb5190e58c54c725d5843c2b295e
```

性能 runner 在计时前后重新计算二者的 SHA-256；运行期间二进制或表格发生改变
会使验收失败。

## 3. 五次 1 PeV 热缓存性能验收

配置：

```text
primary             electron
energy              1 PeV
events/repetition   1
repetitions         5
EM cut              0.5 MeV
EM thinning         1e-4
maximum weight      100
CPU threads         1
build               Release, sm_89
cache               warm
minimum speedup     5
```

正式输出：

```text
/tmp/c8_phase76_current_physics_lazy_hot_performance_1PeV_5rep_v3
```

结果：

```text
status                                      passed
ratio of external-wall medians              9.37549
ratio of summed-shower-timing medians       13.73049
paired external-wall median                 9.56523
paired external-wall minimum                8.57290
paired shower-timing median                14.26428
paired shower-timing minimum               12.54889
```

五次 external-wall paired speedup 为：

```text
9.57477, 8.57290, 9.94774, 9.56523, 9.44721
```

五个 CUDA 事件均有：

```text
fallback_interactions       0
discarded_final_states      0
fallback_model_initialized  false
```

这证明无实际氩靶回退的短性能作业不再初始化 QGSJet-II，并且当前物理修复后的
可执行文件仍超过原计划的 5× 端到端门禁。

## 4. 真实 UHE 光核回退

使用同一个可执行文件和表格重跑 10 个 \(10^{17}\,\mathrm{eV}\) electron：

```text
/tmp/c8_phase76_lazy_photohadronic_fallback_cuda10_v1
```

结果：

```text
complete showers                         10 / 10
SIBYLL preferred final states           378
QGSJet-II fallback final states           3
discarded final states                    0
旧 "Skipping secondary production"        0
```

前两个 shower 没有回退，`fallback_model_initialized=false`。第 3 个 shower
首次触发两次回退后该值变为 `true`；后续 shower 复用已经初始化的模型。第 4 个
shower 再触发一次回退，总计恰好 3 次。

这同时验证了懒加载的两条路径：

1. 无回退时不构造备用模型；
2. 首次回退时只初始化一次，之后复用，并且不丢失末态。

## 5. 当前源码全量回归

懒加载实现、性能来源门禁和本文档落地后，重新执行了两棵构建树：

```text
CUDA Release all target       PASS
CUDA CTest                    32 / 32 PASS
CPU-only Release all target   PASS
CPU-only CTest                10 / 10 PASS
Python validation tools       53 / 53 PASS
```

CUDA 构建目录：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda
```

CPU-only 构建目录：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_clean
```

这组回归证明新增 header-only 光核适配器同时兼容 CUDA 与默认
`CORSIKA_ENABLE_CUDA=OFF` 构建，并且没有破坏设备物理、环境、wavefront、
PROPOSAL、输出或 validation provenance 测试。

## 6. 结论与剩余工作

当前构建已经重新关闭：

- 光核指定末态无损回退；
- 无回退作业的 QGSJet-II 固定启动开销；
- 当前 executable/table provenance 下的 5× 性能门禁。

剩余生产门禁是用相同来源哈希重建代表性 CPU/CUDA shower 统计矩阵，并完成
当前源码状态的 CUDA、CPU-only 和 Python 全量回归。
