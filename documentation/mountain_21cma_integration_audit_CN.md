# beta5 山体实例与 21CMA 完整场景的逐项复核

日期：2026-09-08，电脑重启后复测。

后续已接通真实 DEM 的 B=0 多介质 Kokkos 输运；最新结果与尚未完成的跨界射电范围见
[多介质输运验收](terrain_multimaterial_kokkos_transport_validation_CN.md)。

本页记录当时有限凸体原型的复核基线。此后按照用户要求开始真实 DEM 迁移，
新的场景/天线、CPU 跨界和 Kokkos 百万射线结果见
[真实山体迁移与构建记录](terrain_native_scene_build_CN.md)。下表“未实现”描述的是本页审计时的原型，
不是最新几何层状态；但完整 Kokkos 多介质 shower/内外射电仍未完成。

## 结论

**当前版本没有完成“21CMA 真实山体嵌入标准大气 + 地理坐标天线阵 + Kokkos 岩内/岩外 shower 和射电”这条完整链路，不能按这一目标验收通过。**

目前完成并通过小事例接线检查的是：**有限均匀 SiO₂ 凸体内的 CC 中微子/电磁级联、CPU 或 Kokkos 输运、吸收式逃逸边界，以及内部 observer 的均匀介质直达 CoREAS/ZHS**。这不是原 mountain 工程全部功能的移植，也不是完整物理精度验收。

## 1. 对照代码，而不只对照图片

| 要求 | 原 `corsika8-mountain` 中的实现 | 当前 beta5 | 本轮结论 |
|---|---|---|---|
| 21CMA 附近真实地形 | 地理范围 → DEM → 水密非凸 PLY，BVH 查询 | box / tetrahedron / 闭合凸三角 OBJ，设备最多 32 个面平面记录 | **未接入真实 DEM**；不能取凸包冒充山坡/山谷 |
| 嵌入标准大气 | `TerrainAtmosphere.hpp` 建原生 USStdBK 五层环境，将满足高度范围的岩体挂入最低层 | `Universe → 单个均匀 SiO₂ 子节点`，没有空气节点 | **未实现** |
| 经纬度、海拔与天线阵 | 地形准备模块统一 EGM96 正高、WGS84 椭球高、ECEF、地理 ENU，输出天线坐标 | 只读 `radio.observers_m` 的局部米制三元组，并要求点在岩体内部 | **未接入**；不接受真实山外阵列 |
| Kokkos 岩内 shower | 原 DEM 输运/地形射电主要是独立 CPU 路径；firn/ice GPU 不等于 DEM GPU | 原生 PROPOSAL SiO₂ 表 + Kokkos 光子/轻子输运，CPU 保留强子/中微子/回退 | **限定范围小事例通过** |
| 正确岩气边界 | 原工程可记录介质切换并继续空气输运；有独立介质归属检查 | 到凸体表面即 `EscapedEnvironment` / CPU 吸收，并写逃逸记录 | **吸收边界通过；岩气连续输运未实现** |
| Kokkos 山体内外射电 | 原工程另有地形 GO 透射与分层大气 ZHS 轨迹回放；有明确近似和未通过项 | 仅内部 observer，解析 `nR/c`，不求界面光路 | **外部射电未实现**；内部只是功能检查 |

beta5 关键代码入口：

- [应用配置、介质装配和后端接线](../applications/c8_mountain_neutrino.cpp)：只构造一个 `ConvexPolyhedron` 岩石节点；无 `create_5layer_atmosphere()` 或经纬度转换调用。
- [通用凸体几何](../corsika/framework/geometry/ConvexPolyhedron.hpp)、[独立直线 tracking](../corsika/modules/tracking/TrackingConvexPolyhedron.hpp)、[Kokkos 环境快照](../corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp)。
- [光子边界处理](../corsika/accelerator/em/detail/PhotonTransportStep.hpp)、[轻子边界处理](../corsika/accelerator/em/detail/LeptonTransportStep.hpp)：凸体出口是环境逃逸，不是切换为空气。
- [应用诊断与逃逸记录](../applications/detail/mountain/MountainDiagnostics.hpp)：CPU 面交点同样终止；不继续跟踪岩体外粒子。
- [内部射电适配器](../applications/detail/mountain/MountainRadio.hpp)：源点/observer 必须在凸体内，路径使用均匀折射率，无 Fresnel、折射或遮挡算法。
- [原生表导出](../src/accelerator/em/tables/ProposalNativeTable.cpp) 的 `requireSingleComposition()`：拒绝不同组分、平均激发能或密度效应参数混在同一物理表定义内。**空气和 SiO₂ 不能通过仅增加 medium ID 解决。**

## 2. 原 mountain 项目不能被概括为“全部已经验收”

本轮读取了源码和已有机器可读结果，另外复跑其地理配置/天线/大气配置/界面/遮挡单测。

- [地理地形与天线模块](../../../corsika8-mountain/docs/GEOGRAPHIC_TERRAIN_ANTENNA_MODULE_CN.md)区分 `measured` 与 `dem_agl`。此前 80 个站采用网格地表上方 1 m 的**建模高度**，不是已经校准的实测海拔；保留实测经纬度并不等于实测高度完全吻合 DEM。
- [原生 USStdBK 嵌入](../../../corsika8-mountain/docs/TERRAIN_USSTDBK_EMBEDDING_VALIDATION_CN.md)已有独立 CPU 输运结果；源码为 [TerrainAtmosphere.hpp](../../../corsika8-mountain/cpp/TerrainAtmosphere.hpp)。该批 13 例的原记录共审计 79,917 步、零介质错配，**其射电显式关闭**，本轮没有重跑这 13 个完整 shower。
- 后续 [地形分层大气射电回放](../../../corsika8-mountain/cpp/c8_terrain_atmosphere_radio_replay.cpp)使用 stock ZHS 与 `TerrainAtmosphereRadioPropagator`，不是 Kokkos。其明确范围为直线空气路径上的折射率积分时延、局部 Snell/Fresnel；不含空气弯曲/聚焦、反射、绕射或完整过渡辐射。
- [厚山脊中微子射电试验](../../../corsika8-mountain/docs/TERRAIN_THICK_NUTAU_RADIO_PILOT_CN.md)记录了后续部分方向的收敛门未通过。不能把 firn/Meep 的通过结论移用到真实 DEM 山体。

因此应迁移经过范围审计的模块和测试，而不是复制一个输出目录就称为完成。

## 3. 本轮实际验收

### 3.1 重启、构建和设备

重启后 `nvidia-smi` 能识别 RTX 4060 Laptop GPU / 8188 MiB，WSL 约 15 GiB 总内存、检查时约 11 GiB 可用，swap 未使用。没有重启/重置设备或恢复大批生产任务。

使用独立构建的 `build/mountain-openmp/applications/c8_mountain_neutrino` 与 `build/mountain-cuda/applications/c8_mountain_neutrino`；未安装覆盖现有生产程序。二进制 SHA-256：

```text
OpenMP 20daaa84193198893a67cdd5c7fbe4491d734ab8673ccadda21f10e7d025ae04
CUDA   8d50725e16db3d759dfdadb884c7db1d4c09c2591a78aef552ccb789955a4974
```

此前 CUDA 初始化失败保留在 `integration_cuda_v1`，不能算通过。重启后用新目录重跑三类真实 CUDA 事例，均成功；这也避免将失效的 WSL GPU 状态误判为物理错误。

### 3.2 单测与输入门禁

- beta5 四个测试目标 **4/4 通过**：凸体 Kokkos transport、均匀射电、中微子末态、射电适配器。
- 凸体测试包括 10,000 条独立解析 box 射线、四面体斜面、切向/边/角、内外向面起点、柱深/逆柱深/飞行时间，以及默认球形观测面回归。重启后的这次 CTest 是 **OpenMP**，不标记为 CUDA 百万点验证。
- 配置检查 **14 项通过**：11 个拒绝路径和 3 个合法示例；涵盖外部 observer、无效介质、零方向、越能区/其他味中微子、错误强制顶点、非法 cut/时间窗等。
- 原 mountain 的 5 个相关 Python 测试文件合计 **125 项通过，2 个既有 warning**。这是原模块的测试，不是证明这些模块已经接入 beta5。

### 3.3 真实 shower 接线

CPU、Kokkos-OpenMP（2 线程）各 3 例，以及重启后 Kokkos-CUDA 各 3 例；seed 均为 67101。
测试显式限制 batch=64、resident capacity=4096、显存预算=10%，是资源受控的功能检查，不是生产性能配置。

| 事例 | 标量 CPU 步数 | OpenMP CPU / 加速步数 | CUDA CPU / 加速步数 | 检查 |
|---|---:|---:|---:|---|
| 1 GeV 光子，20 m 边长岩体 | 1867 | 1186 / 729 | 1186 / 729 | 有真实 EM 输运、有限且非零的内部 CoREAS/ZHS |
| 10 TeV νe 自然穿四面体 | 1 | 1 / 0 | 1 / 0 | CC=0；逃逸 1 个 νe，10000 GeV；沉积=0 |
| 10 TeV νe 强制 CC，0.2 m 边长岩体 | 422 | 450 / 43 | 450 / 43 | 三者 CC=1；实际次级逃逸，内部射电非零 |

全部 9 例 `complete=true`、边界违规计数 0。自然中微子没有 EM 次级，所以加速步数为 0 是正确行为。
强制例使用短岩柱以限制成本，不代表完整包含在岩中的 10 TeV shower 已验收。
CPU 与 Kokkos 不共享整个随机调用顺序，单例步数不同不能据此判定等价或不等价。

OpenMP 与 CUDA 的同 seed 检查没有峰值归一化或时间平移：

| 事例 | profile 最大绝对数组差* | CoREAS 三分量波形相对 L2 | ZHS 三分量波形相对 L2 |
|---|---:|---:|---:|
| 光子 | 1.55e-13 | 8.20e-11 | 1.09e-9 |
| 强制 CC | 1.31e-14 | 7.80e-10 | 1.86e-10 |

*逐列原单位比较后取最大值，列含 m、GeV 和加权粒子数；这不是无量纲物理误差指标。
波形每组 3×4000 个点。首末点为零，但**首末为零不等于逐贡献时间窗完整性已经验证**。
两个后端使用同一组分表哈希和辅助缓存哈希。这说明已接线算法的跨执行空间重复性良好，不构成对 SiO₂ 场强的独立 Maxwell/逐轨迹 CPU oracle 验证。

## 4. 新发现的未通过项

1. **未知 YAML 字段静默忽略。** 在合法光子配置中加入 `site` 和 `atmosphere: {model: us_standard_bk, ...}`，`--geometry-only` 返回结果与未加入前完全相同。不能据命令成功退出推断地理/大气配置生效。应先为此独立应用加严格 schema/未知字段拒绝门。
2. **岩气接口缺失。** 目前不会生成任何山外空气 shower 轨迹，也不能计算其辐射。
3. **单介质 native 表限制。** 需要分介质的表集合/会话映射；不能删除现有门禁后把岩石率用于空气。
4. **统计 metadata 不完整。** 这些小批次存在加速步数和 radio tracks，但 `wavefronts`、`particles_advanced`、`kernel_time_ms` 为 0。不能拿这几个字段计算 GPU 吞吐或利用率；应补小批次路径的统计覆盖。
5. **能量账本未闭合认证。** 强子/核过程涉及靶核和静质量交换；当前只有沉积与逃逸合计，不能直接要求二者之和等于初级总能量，更不能据此报告已通过 1e-4 能量闭合。
6. **完整场景与大样本均未验收。** 未做真实天线阵三维归属、出山后空气源、透射系数/偏振/遮挡、边界源段分割和最终时窗逐贡献验收；也未做跨后端系综分布验收。

## 5. 建议的补齐顺序（本轮未实施）

1. **独立应用输入门禁与地理准备**：复用原 mountain 的 DEM / EGM96 / ECEF→ENU / 天线高度策略；未知字段报错；为 80 点逐点输出实测值、模型值、基准及介质归属。
2. **独立组合环境**：`StandardAtmosphere + TerrainSolid`，复用原生五层大气，不改空气应用。实际 DEM 需要非凸网格 BVH，32 面凸体不能承担真实地形。验证全部顶点所属层并处理或拒绝跨层岩体。
3. **多介质 Kokkos 输运**：同一 backend 中按介质索引到正确 native 表，或由明确边界路由管理独立介质 session；设备遇边界返回 `MaterialTransition`，不当逃逸。保留 history、时间、能量、权重、父粒子标识，避免重复走步/重复沉积。先做岩→气、气→岩和反复穿越解析薄层，再做 DEM。
4. **独立地形 radio path 接口**：源介质/目的介质、光程、发射/接收方向、偏振传递、界面系数、路径有效性显式进入数据；分开岩石源、空气源及相干总和。先验证平面解析 Snell/Fresnel，再移植有限面、非凸遮挡和空气积分；不拿内部 `nR/c` 冒充跨界传播。
5. **真实 21CMA 场景验收**：小能量出山电子 → 条件 νe CC → 多 seed；每步独立查询介质，输出岩/气源 profile、场分量、时窗和能量账本；物理门通过后再测 GPU 性能。

## 6. 原空气应用保护与复现位置

本轮复核未改动 `c8_air_shower.cpp`、大气参数或生产 build/install；没有把 mountain 的 experimental mesh overlay 全局覆盖到 beta5。
共享 transport / radio 的山体分支以显式 geometry / homogeneous-index 标志启用，默认球形/空气路径单测通过。**这不等于已做旧生产二进制对新构建的完整固定 seed air-shower 回归**；完整升级前仍需该门禁。

当前 `applications/c8_air_shower.cpp` SHA-256：
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。
工作树中有先前的内存、ZHS 等修改，本次没有重置或覆盖它们。

详细结果在项目容器目录下 `build/mountain-validation-20260908/`：

- `integration_openmp_v1/acceptance.json`：CPU/OpenMP 6 例及命令、耗时、完整输出。
- `integration_cuda_v1/acceptance.json`：重启前设备初始化失败，保留。
- `integration_cuda_postreboot_v2/acceptance.json`：重启后 CUDA 3 例及完整输出。
- `configuration_gates_postreboot.json`、`mountain_reference_pytest.log`：配置与原模块单测。

可复现脚本：[小事例接线验收](../validation/mountain/run_smoke.py)、[输入预检](../validation/mountain/check_configuration.py)。

**最终状态：有限岩体原型的接线验收通过；用户要求的完整 21CMA 山体—大气—天线—Kokkos 射电链路未完成。**
