# Phase 14：Molière、均匀磁场与物理 HybridCascade 路由

## 1. 本阶段完成的范围

本阶段把此前彼此独立的轻子物理核连接成一条可由
`HybridCascade` 调用的物理路径：

```text
CPU stack
  ├─ hadron / μ / τ ───────────────> ScalarCascadeStepper
  └─ γ / e- / e+
        │
        v
  PhysicalCudaEmRouter
        │
        ├─ γ  -> selection -> spherical transport -> pair final state
        └─ e± -> selection -> continuous loss
                         -> curved magnetic transport
                         -> Molière scattering
                         -> vertex reselection
                         -> brems final state
```

当前实现已经解决以下基础问题：

1. 解析 Molière 多重散射与 PROPOSAL 7.6.2 的直接实现逐点一致；
2. Molière 参数随 v7 物理表缓存并进行版本、尺寸和介质组分校验；
3. 均匀磁场使用与 CPU `TrackingLeapFrogCurved` 相同的最大偏转步长
   和两半步 leapfrog 公式；
4. 弯曲轨迹与球形大气边界的交点由四次方程求得；
5. Molière 旋转作用在磁场输运后的方向上，顺序与 CPU Step 的连续过程一致；
6. 物理 EM 路由器在 wavefront 之间保有 GPU 粒子所有权；
7. GPU 不支持的过程显式返回 CPU，并且只绕过一次 GPU 路由；
8. CPU 与 GPU 共用 history-ID 分配器，不会产生次级身份碰撞；
9. GPU 轻子轨迹被转换为 `EmStepRecord` 和 `RadioTrackRecord`。

这仍不是最终生产版本。尚未完成的边界列在第 9 节。

## 2. Molière 解析实现

核心文件：

- `corsika/gpu/em/MoliereScattering.hpp`
- `tests/gpu/testGpuMoliere.cu`
- `applications/gpu_em_tablegen.cpp`

设备快照 `MoliereSnapshot` 不含指针和 STL 容器，可以按值传给 CUDA
kernel。它保存：

- 电子质量、精细结构常数、阿伏伽德罗常数等数值常量；
- 介质各组分的电荷和质量权重；
- PROPOSAL `c1`、`c2`、`c2large`、`s2large`、`C1large`
  系数数组。

抽样函数

```cpp
sampleMoliereScatteringAngle2D(
    snapshot, grammage, initial_energy, final_energy, u1, u2)
```

是 PROPOSAL 解析 Molière 算法的 host/device 翻译。随机数不依赖
线程号，而使用

```text
(seed, shower_id, history_id, step_id,
 ContinuousScatteringRandomProcessId, draw_id)
```

形成 Philox 地址。方位角使用第三个独立 draw。

测试网格覆盖多个能量、grammage 和分位点。GPU 解析结果与 PROPOSAL
直接 `Moliere` 的最大相对差约为 `3.4e-13`。与 CPU 生产配置使用的
`MoliereInterpol` 还做了单独一致性检查；插值实现自身允许约百分之一
级坐标误差，因此该比较不用于判断解析公式精度。

## 3. v7 物理表中的 Molière 元数据

表格式升级为：

```text
format version = 7
magic          = C8EMRT07
```

`RateTableMetadata::moliere` 的 `enabled=false` 只允许用于合成测试表。
生产 `gpu_em_tablegen` 会写入完整解析参数。读取时会检查：

- reference mode 必须为 `proposal_analytic`；
- 介质组分 hash 必须与主 metadata 一致且无重复；
- 所有物理常量必须有限且满足正值约束；
- 五个系数数组的长度必须分别为 `70, 70, 13, 13, 15`。

缓存损坏、组分变化或格式不匹配都会使后端初始化失败，不会静默关闭
多重散射。

## 4. 均匀磁场 leapfrog

核心文件：

- `corsika/gpu/em/UniformMagneticField.hpp`
- `tests/gpu/testGpuUniformMagneticField.cu`

`maximumUniformMagneticStep()` 与 CPU tracker 使用相同条件：

- 横向动量低于 `1 eV/c` 时按直线处理；
- 回旋半径大于 `1e9 m` 时按直线处理；
- 默认最大偏转角为 `0.2 rad`；
- 步长为

\[
L_{\max}=2\cos(a)\sin(a)R_g.
\]

`advanceUniformMagneticField()` 实现 CPU `LeapFrogTrajectory` 的二次位置
多项式：

\[
\mathbf{x}(L)=\mathbf{x}_0+
L\mathbf{u}_0+
\frac{L^2}{2}
\frac{q\,0.299792458}{p}
(\mathbf{u}_0\times\mathbf{B}).
\]

末方向才归一化；位置使用未归一化 kick。这一点是为了与 CPU 代码逐项
一致，不能改成简单圆弧旋转。

## 5. 弯曲轨迹与球面交点

将上式代入球面条件

\[
|\mathbf{x}(L)-\mathbf{c}|^2-R^2=0
\]

得到无三次项的四次多项式

\[
A L^4+C L^2+D L+E=0.
\]

设备函数 `intersectUniformMagneticSphere()` 不使用固定空间采样。它先
解析求出四次多项式导数的至多三个实根，将允许步长分成单调区间，再对
发生符号变化的区间做 96 次二分。这种做法能够找到最大磁场步内的第一
个真实穿越，并避免固定采样漏掉近切向交点。

小于 `0.1 mm` 的交点被视为当前边界的数值残差，与
`TrackingLeapFrogCurved` 的体积切换保护一致。

## 6. 轻子最近限制竞争

`CudaLeptonTransport.cu` 现在同时比较：

1. 抽样的离散相互作用 grammage；
2. 最大 10% 连续能损 grammage；
3. 弯曲轨迹的大气层/观测面边界；
4. 最大磁偏转步长；
5. 粒子 cut。

介质 grammage 仍按 CPU `SlidingPlanarExponential` 的规则计算：密度
在球形半径上求值，但一个 Step 内的径向轴冻结在起点，并使用轨迹初始
方向和长度参数积分。因而这不是额外引入的几何近似，而是对当前 CPU
介质实现的复现。

新增的 `LeptonTransportLimit::MagneticStep` 表示没有更近的物理或几何
限制，本次只因最大允许磁偏转而结束。该粒子的 `step_id` 增加后进入
下一 wavefront。

## 7. PhysicalCudaEmRouter

核心文件：

- `corsika/gpu/em/PhysicalCudaEmRouter.hpp`
- `corsika/gpu/em/RouterParticleConversion.hpp`
- `corsika/detail/framework/core/HybridCascade.inl`

路由器实现 `HybridCascade` 所需的四个方法：

```cpp
bool canRoute(particle, step_id);
void stage(particle, history, parent, generation, step);
bool pending() const;
std::size_t advanceOneWavefrontAndReturn(stack);
```

一个 wavefront 内先按 PID 分成 photon 和 lepton 批次。GPU 产生的
EM 次级、层边界 continuation、LPM suppression continuation 和连续步
continuation 直接进入路由器的下一批，不先写回 CORSIKA 主栈。

以下状态才离开设备调度：

- observation surface；
- escape；
- particle cut；
- 显式 `ProposalFallbackEvent`。

### 7.1 CPU fallback 防回环

fallback 粒子按原来的 `(history_id, step_id)` 导入 CPU 栈，并登记在
`cpu_fallback_steps_`。调度器下次取得它时，`canRoute()` 消耗该登记并
返回 false，因此该 transport step 恰好执行一次标量路径。若标量步骤后
粒子仍存活，新的 step 可以再次进入 GPU。

这解决了最基本的 GPU↔CPU 无限回环。当前 fallback 的标量步骤仍会由
CPU 重新执行完整 PROPOSAL 过程；“GPU 已选定 type/component/v、CPU
只生成指定末态”的最终适配尚未接入，见第 9 节。

### 7.2 跨 CPU/GPU 的 history-ID

`TransportIdentityData::reserveHistoryIds(count)` 与
`Stack::reserveTransportHistoryIds(count)` 提供共享连续区间分配。
每次可能分支的 kernel 启动前，路由器从 CPU stack 的同一个 allocator
预留至多 `2 × input_count` 个 ID，并将区间起点交给 GPU。

即使实际只产生少量次级，未使用的 ID 也不会回收。这牺牲少量编号连续性，
换取：

- CPU 和 GPU 次级绝不重号；
- kernel 内可用 scan offset 稳定计算 child history；
- 调度顺序改变不影响某个 child 的身份。

## 8. Step、能量沉积和射电轨迹

`PhysicalCudaEmRouter` 保存三类 host-visible 输出：

```cpp
stepRecords();
radioTracks();
observations();
```

每个 photon/lepton 输运段产生 `EmStepRecord`。轻子记录包含连续能损与
cut 能量沉积，并保留权重。路由器统计的总沉积使用

\[
\sum_i w_i\,\Delta E_i.
\]

每个 e⁻/e⁺ 段同时生成 `RadioTrackRecord`，保存磁场和 Molière 作用前后
的方向、位置、时间、能量和权重。当前只是与 CPU CoREAS 对接所需的
批量数据结构，尚未调用正式 writer。

## 9. 尚未满足的生产条件

本阶段不声称已经达到完整生产验收。下一阶段必须继续完成：

1. 将 `ProposalFallbackEvent` 转为完整
   `proposal::ProposalInteractionRecord`，由 CPU
   `ProposalFinalStateGenerator` 生成指定末态，禁止重新抽过程；
2. 将 router 的 step、deposit、observation 和 radio buffers 接到
   `c8_air_shower` 的正式 writer/CoREAS；
3. 加入 EM thinning，确保 CPU/GPU 使用同一权重语义；
4. GPU 实现或精确 fallback Compton、photoelectric、annihilation、
   EPair 和离散 ionization；
5. 把 photon 和 lepton 合并为真正的设备驻留双队列，消除当前每个物理
   wavefront 的 host round-trip；
6. 增加完整 shower 的能量闭合、纵向 profile、地面分布和性能测试。

## 10. 验证结果

本阶段加入或扩展了以下测试：

- `testGpuMoliere`
- `testGpuUniformMagneticField`
- `testGpuLeptonTransport`
- `testGpuHybridRoute`

验证内容包括：

- host/device 数值一致；
- 弯曲球面交点位于目标半径；
- e⁻/e⁺ 电荷符号导致正确的相反弯曲；
- 固定 Philox key 的逐次重复；
- 磁场后再施加 Molière 的角度关系；
- GPU/CPU fallback 所有权无回环；
- history-ID 预留后 CPU 不重用；
- scheduler acquired/completed 完全相等；
- radio record 与 step record 身份一致。

本次阶段回归中全部 17 个 `testGpu*` 测试通过，0 失败。
