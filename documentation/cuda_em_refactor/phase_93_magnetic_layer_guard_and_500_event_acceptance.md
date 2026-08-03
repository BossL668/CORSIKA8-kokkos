# Phase 93：磁场分层保护带修复与 500 事例验收

## 1. 触发条件

100 TeV、垂直质子、`emthin=1e-6` 的本地 CUDA 扩展验收在固定批次
`seed=10200251` 的第 22 个 shower 中止。严格失败信息为：

```text
pid=11
energy_GeV=0.0011808222196131866
position_m=[-32.82159051860207, 9.385832804803458,
            6382399.0942817135]
direction=[-0.4188050119591029, 0.05284253648793534,
           0.9065373838377857]
reason=magnetic_boundary_failed
```

该失败发生在 USStdBK 大气 11.4 km 分层边界附近。它不是 CUDA error、显存
溢出、FLUKA worker 故障，也与 CPU/CUDA 平均 profile 的归一化差异无关。

## 2. 根因

`intersectUniformMagneticSphere()` 在配置的最大磁偏转步长内正确返回
`NoForwardIntersection`。用日志中的完整粒子状态重放后得到：

- 粒子起点位于 11.4 km 边界内侧约 0.9056 m；
- 0.2 rad 最大磁偏转对应步长约 0.99881155 m；
- 第一个真实球面交点比该步长远约 43 μm；
- 因此磁场步长端点仍在旧层内，但距边界小于 CORSIKA tracking 使用的
  0.1 mm 数值保护带。

`queryAtmosphereLayer()` 有意使用方向感知的 0.1 mm 层归属规则：位于保护带
内且朝外运动的粒子归属于外层。这避免下一步在同一数值边界反复求交。原
CUDA 输运器在 `magnetic_step_wins` 分支中却无条件要求端点层与起点层相同，
于是把一个合法的保护带层转换错误地当成 `magnetic_boundary_failed`。

## 3. 修复

`src/gpu/em/CudaLeptonTransport.cu` 现在把以下情况登记为
`LeptonTransportLimit::LayerBoundary`：

1. 磁偏转步长是最近限制；
2. 端点的方向感知层归属是起点层的直接相邻层；
3. 端点到两层公共球面的距离不超过
   `max(AtmosphereBoundaryGuardM, floating_tolerance)`。

记录同时更新 `limiting_radius_m`、`end_layer_index`、`medium_id` 和端点密度。
非相邻层跳跃、超出保护带、非有限端点或其他层查询错误仍然产生
`MagneticBoundaryFailed` 并使 production shower 硬失败。因此这不是把全部
磁场边界错误降级成静默 CPU fallback。

## 4. 回归测试

新增两层测试：

- `testGpuUniformMagneticField.cu` 保存真实粒子状态，确认球面求交器仍正确返回
  “有界步长内无交点”，且端点落在边界内侧 0.1 mm 保护带；
- `testGpuLeptonTransport.cpp` 用 USStdBK、IGRF13(2025) 的 21CMA 磁场和该
  粒子状态验证 device pipeline 无 fallback，并至少产生一个从第 1 层到第
  2 层的合法 `LayerBoundary` 记录。

修复后的全部 27 项 `testGpu*` 测试通过。随后从同一失败 seed 重跑完整 25
事例批次，结果为：

- 25/25 shower 闭合；
- CoREAS、ZHS、profile、particles、energy loss 和 timing 输出齐全；
- 无 `magnetic_boundary_failed`、critical、CUDA error 或 shower abort；
- 批次运行时间 1579.5 s。

## 5. 500 事例数据的构建分层

修复会改变 `c8_air_shower` 可执行文件哈希。为保持 provenance 可审计，不在
原 campaign 中覆盖或伪造哈希：

- 修复前已闭合的 250 个 CUDA shower 保持只读；
- 修复后 campaign 把这 250 例作为明确的 `existing_cuda` 来源；
- 从失败批次起点 `seed=10200251` 重新运行剩余 250 例；
- 最终统计必须报告并检查修复前 250 与修复后 250 的 build-stratum 相容性。

这两个构建的物理差别只作用于罕见的内部大气边界保护带状态，但不能据此
跳过分层审计。

## 6. 与质子平均 profile 归一化问题的关系

CPU 样本由预先固定的连续 seed/shard 范围定义，不是按完成速度选择。运行
时间与总 EM profile 积分高度相关，因此“选择最快任务”确实会产生向低粒子
数的偏差；当前数据的选择规则没有使用运行时间或 shower observable。

质子的 shower-to-shower 涨落远大于铁核。100 例尺度上约 4% 的 CUDA/CPU
平均归一化差目前不能单独判为后端偏差。最终 500+500 验收将同时检查：

- 预声明的 CPU 前 50 与其余 450；
- CPU runtime 与 EM profile integral 的相关性；
- CUDA 修复前 250 与修复后 250；
- CPU/CUDA 的绝对 profile、逐 shower `Xmax` 对齐后的归一化形状和 bootstrap
  均值比置信区间。
