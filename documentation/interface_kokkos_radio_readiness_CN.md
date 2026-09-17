# beta5 跨介质常驻输运接入 Kokkos 射电的源码检查

> 本文保留实现前的检查记录。后续独立模块、山体接线和验证方法见
> [独立跨介质 Kokkos 山体射电模块](interface_kokkos_radio_CN.md)；
> 下文“尚未接通”描述的是检查当时的状态。

日期：2026-09-11。范围：当前工作区 beta5 与 `corsika8-mountain/cpp`。
本轮仅检查源码和已有验证文档，未编译、未运行测试、未修改程序行为。

**可以接入，基础代码已经具备；当前 DEM 跨介质常驻应用尚未接通射电。**
需要增加独立的轨迹接口和跨介质射电传播实现。均匀介质可较直接复用现有累加器；
真实山体外部天线的计算还需要移植 mountain 的传播算法，不能只打开一个配置开关。

## 现有功能及入口

| 代码 | 已有能力 | 与当前任务的关系 |
|---|---|---|
| [KokkosRadioAccumulator.hpp](../corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp)，418 行 | 接收设备 `LeptonTransportRecord`；CoREAS/ZHS 投影、设备波形累加、定点溢出检查、末尾下载 | 可复用均匀介质路径及累加设施；模板支持独立 CUDA/OpenMP 执行空间 |
| [KokkosResidentLeptonCascade.hpp](../corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp)，1627 行 | 大气常驻循环直接将设备输运记录交给射电累加器 | 可作为接线参考，无须修改此文件 |
| [c8_mountain_neutrino.cpp](../applications/c8_mountain_neutrino.cpp)，266–315 行 | 凸山体应用已组合 CPU CoREAS/ZHS、Kokkos 射电配置和最终波形合并 | 这是已有的均匀岩体内部射电路径，不是新 DEM 跨介质应用 |
| [MountainRadio.hpp](../applications/detail/mountain/MountainRadio.hpp)，28、85、110 行 | 均匀介质 `nR/c`；检查源/天线位于凸岩体内部；合并 CoREAS 电场与 ZHS 矢势 | 明确拒绝岩外天线及跨界路径 |
| [c8_terrain_cascade.cpp](../applications/c8_terrain_cascade.cpp)，308 行 | `radio = disabled` | 当前跨介质常驻输运使用这个应用，尚无射电输出 |
| [TerrainScene.hpp](../applications/detail/mountain/TerrainScene.hpp)，62 行 | 已读取 ENU 天线位置，并要求位于山体外大气中 | 几何配置可复用；也说明不能直接使用“只允许岩内天线”的旧适配器 |

当前 [RadioProjectionData.hpp](../corsika/accelerator/radio/detail/RadioProjectionData.hpp)
中的设备 `SignalPath` 仅有传播时间、两端折射率、距离和发射方向。
[RadioProjectionStep.hpp](../corsika/accelerator/radio/detail/RadioProjectionStep.hpp)，192 行的
`propagate()` 只选择均匀折射率或原平面大气表；它不查 DEM，不返回多分支，
也没有接收方向、界面法向、偏振传递及路径衰减。
CoREAS/ZHS 投影函数直接调用这一具体传播函数，现有累加器没有可替换的跨介质传播策略接口。

## mountain 中可以复用的部分

| 原 mountain 文件 | 可移植内容 | 已有实现的边界 |
|---|---|---|
| [cpp/MountainRadio.hpp](../../../corsika8-mountain/cpp/MountainRadio.hpp) | `solveFaceRay()`、Snell/Fresnel、偏振基变换、岩石衰减、射线管幅度；也有分层/频谱响应参考代码 | 主机实现，包含动态容器、异常、CORSIKA 对象及观测器逻辑；不能整体用于设备 kernel |
| [cpp/TerrainRadio.hpp](../../../corsika8-mountain/cpp/TerrainRadio.hpp) | 非凸 DEM 可见性、有限三角面接受、共享边路径去重、岩气单界面透射 | 明确不支持衍射、多次界面或反射路径；地形模式要求均匀岩石/空气 |
| [cpp/TerrainAtmosphereRadio.hpp](../../../corsika8-mountain/cpp/TerrainAtmosphereRadio.hpp) | 空气直线段折射率积分、界面处局部折射率迭代、地形遮挡 | 最低 USStdBK 层，代码要求高度 0–7000 m；没有弯曲空气射线聚焦、反射或过渡辐射 |
| [cpp/TerrainIntervalZHS.hpp](../../../corsika8-mountain/cpp/TerrainIntervalZHS.hpp) | 有限轨迹的连续到达时间区间和矢势面积；避免先除以 Cherenkov 小分母 | CPU 参考；一次母段细分不等于所有子段均已收敛 |
| [cpp/FiniteIntervalMoments.hpp](../../../corsika8-mountain/cpp/FiniteIntervalMoments.hpp) | 区间内时间矩、补偿累加，用于保留亚采样时间结构 | 主机容器和累加方式需要重新设计设备实现 |
| [cpp/TerrainAdaptiveAirIntervalZHS.hpp](../../../corsika8-mountain/cpp/TerrainAdaptiveAirIntervalZHS.hpp) | 每个子段重新求传播路径并检查细分条件 | 专项只接受一条可见空气直达路径，不能直接推广到岩石透射和可见性变化 |

mountain 的构建入口 [cpp/CMakeLists.txt](../../../corsika8-mountain/cpp/CMakeLists.txt)
声明 `LANGUAGES CXX Fortran`。上列传播与区间实现没有设备函数标记，并使用主机缓存、
输出流及异常。因此可复用的是算法、部分数值函数和 CPU 对照实现；仍需 Kokkos 移植。
beta5 已有 [FlatTerrain.hpp](../corsika/geometry/terrain/FlatTerrain.hpp) 与
[KokkosTerrainSession.hpp](../corsika/geometry/terrain/KokkosTerrainSession.hpp) 的设备几何，
可作为移植地形可见性查询的基础，但粒子下一次碰面查询不自动等价于完整光路可见性检查。

## 当前常驻输运需要补的接口

1. **原始射电轨迹段。**
   [InterfaceEmStep.hpp](../corsika/modules/transport/detail/InterfaceEmStep.hpp)，188 行附近
   已在设备 kernel 内生成 `LeptonTransportRecord step`，目前仅将一部分字段写入 `EmStep`。
   `finish()`、顶点继续分支及介质归属更新还会改变综合结果的 `end`。
   应在原始输运记录有效时保存专用设备轨迹缓冲，并标明源介质、粒子权重、history/step。
   不宜把队列中的后继状态直接当作已辐射的轨迹。现有磁场已进入输运，
   射电仍需独立的折射率、衰减等光学配置，并核对弯曲轨迹的分段近似。

2. **同一执行空间内的调用。**
   在 `InterfaceEmSession::runResidentCascade()` 中，步结果验证后、下一轮覆盖输入之前，
   将设备轨迹交给射电计算；静态天线/介质/几何及波形持续驻留设备。
   使用 session 已有运行时与 execution stream，不能另建一个冲突的 Kokkos 运行时。
   主机诊断下载不应成为正常设备射电计算的输入通道。

3. **独立跨介质路径数据和投影。**
   增加传播分支、时延、发射/接收方向、偏振传递、幅度/衰减及失败状态的数据结构；
   移植 mountain 的单界面和 DEM 算法。可在独立 `InterfaceRadio` 模块中复用原数值代码，
   保持大气 `GpuRadioConfig`、`SignalPath` 和既有累加器默认行为不变。
   单纯按粒子所在介质选择一个 `n`，无法描述岩石源到岩外天线的折射路径。

4. **CPU 源和最终波形。**
   当前设备累加器只接受 e±；CPU 推进的 μ/τ/强子带电轨迹需要独立接入相同射电模型，
   或将结果明确限定为 e± 源。CPU 回退前已推进的设备段只能累加一次。
   CoREAS 的 E 与 ZHS 的 A 不能混淆；复用旧时间域路径时，CPU/GPU 的 A 合并后只求导一次。
   若采用 mountain 连续区间/时间矩表示，则应沿该表示单独定义最终频谱/电场转换，
   不能再套一次旧 ZHS 有限差分。

5. **容量与精度。**
   路径候选、子段、波形和临时数组都需计入预算。现有 128 MiB 输运模块预算没有预留完整射电空间；
   现成 OpenMP 定点实现还持有线程私有波形，256 线程时须单独核算。
   mountain 已有短区间时间表示和递归空气细分的专项，若要复现其精度，
   不能仅接上 beta5 原始时间分箱就宣称与该实现等价。

## 可以落实的接入顺序

先用固定轨迹贯通独立模块的均匀介质设备累加，再增加单平面折射/偏振/衰减，
接着迁移 DEM 可见性和空气光程，最后接入完整常驻 shower 与 CPU 源合并。
轨迹与射电接口可以先建好，不必重写输运物理，也不需要修改原有大气模块。

验证应在 PSR 完成：固定同一组轨迹，依次比较 CPU 参考、Kokkos OpenMP、Kokkos CUDA
的有符号三分量波形、绝对到达时间和复频谱，并检查分介质结果、界面两侧连续性、
重复计数、时间窗、容量和实际设备数据传输。开启射电还应保持输运随机历史与物理结果不变。

已有 [时间区间验证](../../../corsika8-mountain/docs/TERRAIN_INTERVAL_ZHS_VALIDATION_CN.md)
及 [空气细分验证](../../../corsika8-mountain/docs/TERRAIN_AIR_SUBDIVISION_VALIDATION_CN.md)
可作为验收设计依据。后者记录空气细分专项已通过，但明确不代表完整 GO、岩石空间近似
或整个射电模块全面通过；这些历史结果不是本轮的 CUDA 验收。
