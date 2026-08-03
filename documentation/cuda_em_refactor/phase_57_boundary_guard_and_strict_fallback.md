# Phase 57：大气层边界保护与生产 strict fallback

## 1. 触发证据

Phase 55 的 5 MeV 200-event CUDA 系综有 4 次
`invalid_mass_density` generic fallback，Phase 56 的 50 MeV 系综有 1 次。
这些事件此前会返回 CPU 标量 PROPOSAL 走一步，保证 shower 不丢失，但不符合
最终生产失败策略：数值输运缺陷不能被笼统的 CPU 重试掩盖。

首先把原原因拆分为：

- `atmosphere_grammage_failed`；
- `atmosphere_inverse_grammage_failed`；
- `atmosphere_vertex_lookup_failed`。

`ProposalFallbackEvent` 同时增加只用于失败诊断的 status 和三个数值槽；正常
粒子 SoA 没有增大。host 对未处理 fallback 输出 PID、能量、history/step、
位置、方向和诊断值。

## 2. 精确重放

50 MeV 的异常位于 seed 68001 的 `shower_57`。由于 Philox 键包含
`shower_id`，重放必须从同一 seed 连续运行 58 个 shower。

分类重放得到：

```text
pid                  electron
energy               987.688077 GeV
history / step       232 / 1
reason               atmosphere_grammage_failed
atmosphere status    GrammageOutOfRange
geometry distance    9,091,625.637 m
limiting radius      6,408,000 m
distance to boundary 45.1 micrometres
```

粒子位于一个层边界外侧约 45 微米并向内运动。

## 3. 根因

三个边界部件使用了不一致的“刚跨过边界”保护：

- `intersectUniformMagneticSphere()` 为匹配
  `TrackingLeapFrogCurved`，忽略小于 0.1 mm 的近根；
- `queryAtmosphereLayer()` 只使用约 \(10^{-7}\) m 的浮点 tolerance；
- straight sphere intersection 也只使用浮点 epsilon。

所以 layer query 把粒子归给外层，磁场求交器却按设计忽略 45 微米的近根，
随后选择同一球面约 9,091 km 之后的远根。把这个距离送入单层指数大气柱深
公式后得到 `GrammageOutOfRange`。

## 4. 修复

新增统一常量：

```cpp
AtmosphereBoundaryGuardM = 1.e-4;
```

以下操作现在共享 0.1 mm 保护：

- 带方向的 layer ownership；
- straight sphere forward root；
- magnetic leapfrog sphere root。

在保护带中，粒子按径向运动方向归给将要进入的层。直线和磁场求交都会忽略
刚离开的近根，继续寻找下一道真实边界。

回归测试显式构造“边界外 50 微米、向内/向外”的两个状态：

- inward 必须属于内层；
- outward 必须属于外层；
- inward 的下一边界距离必须大于 0.1 mm，不能返回刚离开的球面。

## 5. 生产 strict fallback

`PhysicalCudaEmRouter` 新增 production policy。`c8_air_shower --em-backend
cuda` 默认启用：

```cpp
router.setFailOnUnexpectedFallback(true);
```

允许的一次 scalar fallback 只包括预定义能力边界：

- unsupported particle；
- unsupported medium；
- unsupported geometry。

指定的 photoproduction/photonuclear/muon-pair/Epair 低分位仍由
`ProposalCpuFallbackHandler` 处理；memory spill 走独立路径。以下设备数值
失败不再自动标量重试：

- invalid mass density；
- grammage/inverse/vertex failure；
- invalid final state；
- table/LPM/Molière/magnetic failure。

它们会抛出带完整诊断的异常，由应用把当前 shower 标为 incomplete。

## 6. 验证

直接测试：

```text
testGpuHybridRoute             PASS
testGpuUniformMagneticField    PASS
testGpuSphericalAtmosphere     PASS
testGpuLeptonTransport         PASS
testGpuPhotonWavefront         PASS
```

50 MeV 已知事件的 strict 重放：

```text
58 / 58 complete
generic fallback 1 -> 0
queue overflow / spill = 0
```

5 MeV 重放到最后一个旧异常事件：

```text
172 / 172 complete
旧 invalid-mass-density 事件 4 -> 0
```

该重放出现 1 次 `unsupported_geometry`，按原计划属于允许的环境能力边界并
执行一次标量步骤。没有数值失败被静默接受。

## 7. 对物理基线的影响

统一 0.1 mm 边界语义会改变许多粒子在层边界后的第一步，而不只是先前发生
generic fallback 的 5 个事件。50 MeV 的旧/新 200-event 逐项比较显示：

- CPU PROPOSAL 输出完全相同；
- CUDA 有 143/200 个事件至少一个 profile/deposit 标量变化；
- 修复后所有 curve family 仍通过；
- 3 个关键均值超过 1%，但只有 0.85--1.20σ，严格标为统计不充分；
- generic fallback 为 0。

因此 Phase 56 的修复前严格 PASS 只保留为历史证据，不能直接代表当前代码。
当前实现需要扩大样本重新确定 1% key-mean 结论。
