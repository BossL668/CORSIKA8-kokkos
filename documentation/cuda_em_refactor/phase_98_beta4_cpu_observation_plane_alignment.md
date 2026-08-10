# Phase 98：beta4 CUDA 观测平面与 CPU 几何对齐

## 1. 问题

原标量 `c8_air_shower` 使用通过 shower core、法向为全局 `+z` 的
`ObservationPlane`。最初 CUDA 环境快照却只保存 `observation_radius_m`，并把它
同时作为五层球形大气的内边界和粒子终止球面。两者只在 `x=y=0` 的切点重合；
离轴粒子会在 CUDA 路径中越过 CPU 平面后继续输运到球面。

该实现从最初 beta1 CUDA 后端延续到 beta2，并非 beta4 连续能损修复引入。
旧 GPU 测试只验证球面求交自身正确，没有比较应用程序实际使用的平面。

## 2. 修复

`EnvironmentSnapshot` 现在分别保存：

- 球形五层大气及其地心；
- 观测平面的点 `observation_plane_point_m[3]`；
- 单位法向 `observation_plane_normal[3]`。

`c8_air_shower` 从同一个 `showerCore` 和 `+z` 法向同时构造 CPU 与 CUDA
观测面。配置输出显式记录 `observation_geometry: plane`、平面点和法向。

直线 photon 和零曲率带电粒子使用与 `TrackingStraight::intersect(Plane)` 相同
的线性平面根。非零磁场带电粒子使用 `TrackingLeapFrogCurved` 的同一个二次
位置轨迹：

\[
\mathbf x(l)=\mathbf x_0+l\mathbf u+
\frac{l^2}{2}\frac{0.299792458q}{p}(\mathbf u\times\mathbf B),
\]

代入平面方程后求最小正二次根。球面求交只负责大气 volume boundary；
transport kernel 以显式布尔状态区分 `ObservationSurface` 与 layer/escape，
不再根据终点半径猜测终止类型。

## 3. 回归测试

新增测试覆盖：

- `theta=80 deg`、横向偏移 600 m 的 photon 平面交点；
- 同一几何下的直线 electron；
- 非零磁场下的弯曲 electron；
- CUDA 磁场平面根与 CPU `solve_quadratic_real` 的逐值比较；
- 终点平面残差以及终点半径明确不等于旧观测球半径；
- 小于 0.1 mm 的正终端根不会被 volume-boundary guard 丢弃。

直线平面根与解析值的相对误差门限为 `2e-13`，弯曲轨迹的 CUDA 根与
CPU `solve_quadratic_real` 根也使用 `2e-13` 门限；最终平面残差要求小于
`1e-8 m`。修改后完整 `testGpu*` 矩阵 27/27 通过，Python 验证工具测试
241/241 通过。补齐 `FLUPRO` 与 `FLUFOR` 后重新运行完整 CTest，34/34
通过（237.84 s）；此前不带 FLUKA 环境的一次中止不是代码失败。

## 4. 高倾角真实运行

生产程序使用 `theta=80 deg`、`phi=180 deg`、10 GeV electron、0.5 MeV
EM cut、`emthin=1e-4` 和 21CMA 天线布局运行。CUDA 输出中的
`gpu_em/config.yaml` 明确记录：

```yaml
environment:
  observation_geometry: plane
  observation_plane_point_m: {x: 0, y: 0, z: 6373680.4441950005}
  observation_plane_normal: {x: 0, y: 0, z: 1}
```

旧默认 CPU 路径和显式 `--em-backend proposal` 控制路径在该事例中逐项
一致：过程记录均为 1951 条，process selection、起始能量、loss fraction、
位置以及全部纵向输出均逐位相同。这证明修复没有改变标量基线。

electron shower 在到达地面前耗尽，因此又增加了一个更直接的穿面测试：
`theta=80 deg`、10 GeV muon、注入高度 3000 m。CPU 和 CUDA 均产生一个
ground record；CUDA 统计为 `observed=1, escaped=0`。由于两后端的随机流、
连续能损和多重散射采样本来就不要求单事例相同，横向坐标不能作为
lockstep 判据；终点是否在同一平面由上述同轨迹 kernel 测试负责严格验证。

## 5. 高倾角射电同轨迹验收

为避免 shower-to-shower 涨落，把完全相同的 CUDA EM 轨迹分别交给 CPU
和 CUDA 射电投影器。配置使用 `theta=80 deg`、IGRF14/2027 和 81 个天线。
CoREAS 与 ZHS 各比较 243 个天线分量，结果全部通过：

| 算法 | 最大归一化逐点差 | 最大相对 L2 差 | 最大相对 fluence 差 |
|---|---:|---:|---:|
| CoREAS | `1.056e-4` | `7.495e-5` | `1.064e-4` |
| ZHS | `2.386e-5` | `4.184e-5` | `1.106e-6` |

再使用 21CMA 的 `pulse_analysis_modular` 提取地磁振幅和脉宽。CoREAS 的
105 组振幅、98 组有效宽度以及 ZHS 的 105 组振幅、101 组有效宽度全部
配对成功；CPU/CUDA 特征的最大差均为 0，未出现有效性、滤波器或脉冲方法
不一致。这一结论仅表示同一批轨迹的两个射电投影后端一致，不把单个独立
CPU/CUDA shower 误当成 shower 统计一致性证明。

## 6. 旧几何的影响尺度

以观测面横向距离 600 m 为例，旧球面终止在 `theta=80 deg` 时比 CPU 平面
多走约 0.163 m，相当于 0.542 ns 和约 `0.0152 g cm^-2`。横向距离 2 km
时增加到约 1.81 m 和 6.03 ns。该差异对整个 shower 的能量纵向发展通常很
小，但已经足以影响高倾角地面粒子的到达时间以及射电脉冲末端的相位，不能
继续作为生产几何近似使用。
