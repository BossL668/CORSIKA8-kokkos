# CUDA 电磁后端重构记录：阶段 12，五层球形大气与物理 photon wavefront

## 1. 本阶段结论

本阶段第一次把阶段 9–11 的物理采样链路放到了真实 CORSIKA 8 五层球形大气
中。一个 photon wavefront 现在可以连续完成：

```text
photon batch
  │
  ├─ PROPOSAL v4 表：采样过程、组分、v 和相互作用 grammage
  │
  ├─ 五层球形大气：计算当前层、局部密度和最近层边界
  │
  ├─ 比较 X_interaction 与 X_boundary
  │    ├─ interaction：反演 grammage 得到顶点和局部密度
  │    ├─ layer boundary：step_id + 1，进入下一层队列
  │    ├─ observation：形成地面 observation record
  │    └─ atmosphere top：形成 escaped record
  │
  └─ interaction
       ├─ GPU photon pair + LPM：生成 e-、e+
       ├─ LPM suppressed：原 photon 的 step_id + 1 后重新排队
       └─ Compton/其它未实现过程：显式 CPU fallback
```

新的 `advancePhotonWavefrontForValidation()` 已经能把边界光子和 LPM 抑制光子
组成稳定的下一轮输入，并实际再次运行下一层。它不再要求测试代码手工填写
`mass_density_g_per_cm3`。

仍需明确：生产入口 `advanceWavefront()` 当前仍是早期 toy branching resident
queue；本阶段的完整物理 wavefront 还是 host-visible integration bridge，会在
三个 CUDA batch 之间返回 host。它证明了物理语义和队列闭合，但尚未达到最终
性能目标。下一阶段要把同一链路改为设备驻留双缓冲调度。

## 2. 与 CORSIKA 8 CPU 大气模型的对应

### 2.1 五层的真实类型

`create_5layer_atmosphere()` 的当前实现并不是五个同类解析层：

- 前四层是 `SlidingPlanarExponential<IMediumModel>`；
- 第五个历史上称为 linear layer，但 CORSIKA 8 实际构造
  `HomogeneousMedium<IMediumModel>`。

`EnvironmentSnapshotBuilder.hpp` 直接使用与
`create_5layer_atmosphere()` 相同的 `atmosphereParameterList`。对前四层写入

\[
\rho(r)=\rho_0\exp\left(\frac{r-R_\oplus}{\lambda}\right),
\quad
\rho_0=\frac{b}{H},
\quad
\lambda=-H,
\]

其中 \(b\) 是 CORSIKA-7 grammage offset，\(H\) 是 scale height。第五层只保存
\(\rho_0=b/H\)，并标记为 `DensityModel::Homogeneous`。

设备快照固定使用：

- 半径和位置：m；
- 密度：\(\mathrm{g\,cm^{-3}}\)；
- grammage：\(\mathrm{g\,cm^{-2}}\)；
- 磁场：T。

构建器会检查层数、半径连续性、密度参数、观测半径和数值有限性。非法快照
不能进入 photon transport。

### 2.2 球面层边界

对相对地心位置 \(\boldsymbol r\)、单位方向 \(\boldsymbol u\) 和边界半径 \(R\)，
设备端求解

\[
\left|\boldsymbol r+s\boldsymbol u\right|^2=R^2,
\]

即

\[
s=-\boldsymbol r\cdot\boldsymbol u
\sqrt{(\boldsymbol r\cdot\boldsymbol u)^2-
\left(|\boldsymbol r|^2-R^2\right)}.
\]

只接受严格为正的前向根。零判别式的切线与 `TrackingStraight` 一样，不认为
发生 volume crossing。

层边界点存在所有权歧义，因此 `queryAtmosphereLayer()` 同时接收方向：

- 径向向内时，公共边界归入较低层；
- 径向向外时，公共边界归入较高层。

第一层还有一个重要特例：观测面可以高于平均地球半径。设备端使用

\[
R_{\rm inner,0}^{\rm effective}
=\max(R_{\rm Earth},R_{\rm observation})
\]

作为向下输运的最近限制，避免 photon 穿过用户配置的观测面后继续到海平面。

## 3. Grammage 和逆积分

密度查询使用精确球面半径。但当前 CPU
`SlidingPlanarExponential::getIntegratedGrammage()` 会在每个 trajectory
segment 起点冻结局部径向轴，因此设备端故意复现这一语义，而不是换成弯曲
路径上的数值积分。

令

\[
\mu=\hat{\boldsymbol r}_0\cdot\boldsymbol u,
\qquad
\rho_s=\rho(r_0),
\]

长度 \(l\) 内的 grammage 为

\[
X(l)=
\rho_s\frac{100\lambda}{\mu}
\operatorname{expm1}\left(\frac{\mu l}{\lambda}\right).
\]

因 \(\rho\) 使用 \(\mathrm{g\,cm^{-3}}\) 而 \(l,\lambda\) 使用 m，式中的 100
完成 m 到 cm 的转换。接近切向的 \(|\mu|<10^{-14}\) 使用稳定极限

\[
X(l)=100\rho_s l.
\]

逆函数为

\[
l(X)=
\frac{\lambda}{\mu}
\log1p\left(
\frac{X\mu}{100\rho_s\lambda}
\right).
\]

均匀第五层也使用 \(X=100\rho l\) 及其直接逆函数。代码使用 `expm1` 和
`log1p`，避免短步长下的抵消误差。

## 4. 最近限制竞争

`CudaPhotonTransport.cu` 的每个线程处理一个 `EmInteractionRecord`：

1. 查询起点层和密度；
2. 求最近球面边界距离 \(l_b\)；
3. 计算 \(X_b=X(l_b)\)；
4. 若选中的有限 \(X_i\le X_b\)，用逆积分求 \(l_i\)；
5. 否则推进到边界。

interaction 分支保持原 `step_id`，因为末态/LPM 随机数必须属于刚刚选中的
同一次 interaction。边界分支执行 `step_id + 1`，利用指数分布的无记忆性在
新介质中重新采样，并避免重复使用 Philox key。

顶点的径向密度直接写入
`record.interaction.mass_density_g_per_cm3`。阶段 11 的 LPM 内核因此使用真实
顶点密度，不再由调用者提供近似值。

每个输入严格产生一个 `PhotonTransportRecord` 或
`ProposalFallbackEvent`。设备端用 CUB exclusive scan 稳定 compact，不使用
原子追加。当前明确回退包括：

- 非 photon 输入；
- 起点不在支持的五层环境；
- 非法/不可反演的 grammage；
- 非正或非有限顶点密度；
- 无法解析的边界所有权。

## 5. 完整 photon wavefront 的队列闭合

`PhotonWavefrontBatchResult` 分开保存：

- 所有成功的 transport records；
- `next_photons`：层边界和 LPM suppressed photon；
- observation/escape records；
- selection fallbacks；
- transport fallbacks；
- final-state records、\(e^-e^+\) secondaries 和 final-state fallbacks。

下一轮 photon 队列按原 source input index 稳定排序。这里的 host sort 只用于
集成 bridge；生产 resident scheduler 将用设备 scan/compaction 生成相同顺序。

单个 interaction 队列满足

\[
N_{\rm interaction}=
N_{\rm GPU\ pair}
+N_{\rm LPM\ suppressed}
+N_{\rm continuation}
+N_{\rm final\ fallback}.
\]

输入选择阶段和输运阶段也分别满足 success/fallback 闭合。未实现的 Compton
等过程不会消失，而是保留过程 ID、component hash、\(v\) 和随机数身份送回
CPU。

## 6. 主要代码改动

| 文件 | 作用 |
|---|---|
| `corsika/gpu/em/EnvironmentSnapshotBuilder.hpp` | 从 CORSIKA-7 参数构造五层设备快照 |
| `corsika/gpu/em/SphericalAtmosphere.hpp` | host/device 层查询、密度、球面求交、grammage 和逆积分 |
| `corsika/gpu/em/CudaPhotonTransport.hpp` | photon transport batch 接口 |
| `src/gpu/em/CudaPhotonTransport.cu` | 最近限制内核和稳定 success/fallback compaction |
| `corsika/gpu/em/Types.hpp` | transport record、wavefront result 和统计 schema |
| `corsika/gpu/em/CudaEmBackend.hpp` | transport 与完整 photon wavefront 公共接口 |
| `src/gpu/em/CudaEmBackend.cu` | 保存环境快照、编排选择/输运/末态和下一轮队列 |
| `tests/gpu/testGpuSphericalAtmosphere.cpp` | CPU–GPU 大气与边界对照 |
| `tests/gpu/testGpuPhotonWavefront.cpp` | 两轮完整物理 photon wavefront 验证 |

## 7. 验证结果

### 7.1 CPU–GPU 五层大气对照

`testGpuSphericalAtmosphere` 在所有层中比较：

- CORSIKA CPU `getMassDensity()` 与 GPU 径向密度；
- CPU `getIntegratedGrammage()` 与设备公式；
- CPU `getArclengthFromGrammage()` 与设备逆积分；
- interaction、向上/向下层边界、100 m 观测面和大气逃逸。

结果：

```text
96 checks
5 successful transport records
1 explicit unsupported-particle fallback
```

密度、grammage 和逆积分对照使用 \(2\times10^{-13}\) 到
\(5\times10^{-13}\) 的相对容差。

### 7.2 两轮物理 photon wavefront

synthetic v4 table 的 4096 个 photon 一半位于 2 km，一半位于 110 km。首轮
结果为：

```text
2141 interaction vertices
1948 layer boundaries
630 accepted GPU photon-pair final states
402 explicit CPU final-state fallbacks
18413 checks
```

fixture 的 loss quantile 能区边缘还产生少量预期的
`LossQuantileOutOfRange` selection fallback；测试要求它们显式存在且原因
正确。首轮的 `next_photons` 被再次输入完整 wavefront，第二层仍满足
selection/transport 闭合。

相同 seed、shower ID、history ID 和 step ID 的整个 wavefront 重复运行时，
transport limit、距离、grammage、下一轮粒子状态和末态队列逐项相同。

### 7.3 回归

CUDA test suite 当前结果：

```text
12/12 passed
0 failed
2.30 s total
```

## 8. 下一阶段

当前最大的性能缺口是三个物理 kernel 之间仍有 host download/upload，并且
`advanceWavefront()` 的 resident SoA 仍运行 toy branching。下一阶段需要：

1. 把 rate-table selection、photon transport 和 photon-pair final state 改成
   接受 resident device pointer/view；
2. 在设备端形成 interaction、boundary、fallback、observation 和 secondary
   双缓冲队列；
3. 只把 CPU fallback 和观测记录回传 host；
4. 用 CUDA event 分离 kernel、scan 和 transfer 时间；
5. 用真实 v4 表测量多轮 photon batch 的吞吐与传输占比。

在这些完成前，不能宣称已经达到端到端 5 倍加速，也不能把 CUDA 设为
`c8_air_shower` 的生产后端。
