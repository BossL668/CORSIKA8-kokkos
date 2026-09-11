# beta5 真实山体迁移：场景、CPU 参考与 Kokkos 几何

日期：2026-09-08。状态：**真实场景底层已迁移并测试；完整 Kokkos 多介质 shower + 内外射电尚未接通，不能作为整链验收通过。**

后续更新：已新增 **B=0、无射电的有界 Kokkos 岩气输运**，并在本地小显存预算下实测。
该阶段结果见[多介质输运验收](terrain_multimaterial_kokkos_transport_validation_CN.md)。
最新阶段已加入空气 IGRF14/岩石零场及 CPU/Kokkos 曲线边界，见
[磁场与模块化应用验收](terrain_magnetic_application_validation_CN.md)。
2026-09-09 已补充 [一站式地形准备入口](terrain_preparation_workflow_CN.md)：
`c8-terrain --bounds ...` 自动准备并导出 scene，可显式接入现有 CPU/Kokkos 应用；
不再要求用户分别运行准备与 scene 导出命令。下文的旧命令仍兼容。
下文是前一阶段历史记录，不代表新的输运功能仍未实现；跨界射电仍未接通。

这一阶段按照 `corsika8-mountain` 的实际实现迁移，不再用有限凸体代替 21CMA 山谷地形。此前有限凸体示例仍单独保留；它不是本页的真实 DEM 应用。

后续已补充独立的常驻 Kokkos 几何会话与逻辑边界候选，并通过 OpenMP 百万点重复查询。
最新范围、结果及 PSR 驱动阻塞见[常驻会话与 PSR 验收](terrain_resident_session_psr_validation_CN.md)；
下面的阶段记录保留为迁移基线，不代表后续 GPU 整链已通过。

## 1. 已实现的结构

```text
地理范围 + DEM + 天线经纬度/高度策略
                 │
    python/corsika_terrain（从 mountain 迁移）
    EGM96 正高 ↔ WGS84 椭球高 → ECEF → 地理 ENU
                 │
    水密 PLY + 网格 + observer + 哈希 manifest
                 │
             版本化 scene.yaml
                 │
       ┌─────────┴───────────────────────┐
       │                                 │
 Scene + TerrainAtmosphere        ClosedMesh → FlatTerrainExport
 原生 USStdBK 五层 + 岩体子节点       同一 BVH，无几何重新采样
       │                                 │
 c8_terrain_environment           Kokkos OpenMP / CUDA 百万射线
 场景/高度/80 站检查                    几何 oracle（已通过）
       │                                 │
 c8_terrain_cascade                后续：设备多介质输运/射电
 CPU PROPOSAL + FLUKA/QGSJet       （本阶段尚未集成）
 真实跨界、轨迹和介质审计
```

新代码不需要在运行或编译时包含另一个 mountain 工程的源文件。地形数据可以继续存放在既有磁盘目录；没有复制 DEM 大数据。迁移出处和源文件 SHA-256 记录在 [terrain_migration_sources.json](terrain_migration_sources.json)。

scene 导出删除旧均匀空气产品的 `air_world_radius_m` 和 `air_refractive_index`，
本应用空气折射率始终来自 `atmosphere` 配置的原生逐层模型。本轮早期 `scene_v1`
记录仍保留旧 `air_refractive_index` 字段，但未用于求值；逐站折射率对比验证的是实际环境值。

| 层次 | beta5 文件 | 行为/边界 |
|---|---|---|
| 地理准备 | `python/corsika_terrain/` | 复用原高度基准、DEM/地理 ENU、网格面插值和站中心处理；不把椭球高当海拔 |
| CPU 非凸几何 | `corsika/geometry/terrain/` | AABB/Triangle/BVH/PLY/OBJ/ClosedMesh；保留原 BVH 构建优化，隔离于 `corsika::terrain` |
| 输入门禁 | `ValidatedTerrain.hpp`、应用 `TerrainScene.hpp` | 文件哈希、有限坐标、面索引、退化面、闭合/流形/朝向、高度基准和 observer 检查 |
| 标准大气 | `corsika/modules/terrain/TerrainAtmosphere.hpp` | 原生 USStdBK 五层、原生逐层 Gladstone–Dale；水密岩体置于最低层 |
| CPU tracking | `TerrainBoundaryTracking.hpp` | 复用逻辑介质侧与有向法线判断；查询起点小偏移不移动实际粒子；出山不吸收 |
| 设备几何 | `FlatTerrain.hpp`、`FlatTerrainExport.hpp` | double、米制 POD；原 BVH 扁平化为前序节点和前向 skip，设备无递归/固定深度栈 |
| Kokkos 实例 | `src/terrain/TerrainDeviceQuery.cpp` | 具名 functor、View 上传和实际并行查询；不编译 CPU 单位/几何模板到设备翻译单元 |
| CPU 参考应用 | `applications/c8_terrain_cascade.cpp` | photon/e±/νe/anti-νe；PROPOSAL、FLUKA + QGSJet-II.04、CC 模块、cut/thinning；B=0、radio 关闭 |
| 场景/输出适配 | `applications/detail/mountain/TerrainScene.hpp`、`TerrainShowerOutput.hpp` | YAML 留在应用层；流式输出完整轨迹（上限见下），逐步核查岩/气介质，不修正节点 |

不能把“拓扑检查通过”解释为任意输入网格已经排除自相交。这里要求使用通过原地形准备检查的高度场闭合网格，或另行验证的输入。当前岩体全部顶点必须位于海拔 0–7 km；跨大气层的岩体拒绝，不静默截断。当前 scene 中的 80 站要求位于空气中；岩内 observer 的完整地形传播尚待后续接入。

## 2. 构建与运行

先按 beta5 主 README 准备 conda、锁定依赖、许可内 FLUKA、Pythia8/TAUOLA。以下为**复用已有依赖**的独立验收构建，不替换生产 build/install。目录约定：

```text
corsika-21cma-kokkos-beta5/
  corsika8_kokkos_beta5/           源码
  build/openmp/deps/              已配置的 Conan 工具链
  build/cuda/deps/
  build/external/                 当前机器兼容的 Pythia8/TAUOLA
  build/mountain-openmp/          本轮独立构建
  build/mountain-cuda/
  build/mountain-validation-20260908/  测试记录
  install/                       本轮未覆盖
```

在 beta5 父目录运行，`FLUPRO` 应指向当前机器合法安装的 FLUKA：

```bash
conda activate corsika_venv
C8_PROJECT="$PWD"
C8_SOURCE="$C8_PROJECT/corsika8_kokkos_beta5"
cmake -S "$C8_SOURCE" -B build/mountain-openmp \
  -DCMAKE_TOOLCHAIN_FILE="$C8_PROJECT/build/openmp/deps/conan_toolchain.cmake" \
  -DCONAN_CMAKE_DIR="$C8_PROJECT/build/openmp/deps" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCORSIKA_ENABLE_KOKKOS=ON -DCORSIKA_KOKKOS_BACKEND=OPENMP \
  -DCORSIKA_BUILD_MOUNTAIN_APPLICATION=ON -DWITH_FLUKA=ON \
  -DPYTHIA8="$C8_PROJECT/build/external/pythia8" \
  -DC8_TAUOLA_PREFIX="$C8_PROJECT/build/external/tauola"
cmake --build build/mountain-openmp --parallel 1 \
  --target c8_terrain_environment c8_terrain_cascade c8_terrain_device_probe testTerrainEnvironment
```

CUDA 几何 oracle 使用**另一个** build，工具链换成 `build/cuda/deps`、后端改为 `CUDA`，并使用与该 Kokkos 包匹配的 `CORSIKA_KOKKOS_CUDA_ARCHITECTURES`（本机为 89）。不要修改 OpenMP build 的后端。HIP/SYCL 未在本轮硬件实测，不能标记已通过。

从原来已准备并通过门禁的 21CMA 产品生成 scene，不重新下载 DEM：

```bash
PYTHONPATH="$C8_SOURCE/python" python -m corsika_terrain.scene \
  --prepared /path/to/terrain_region_21cma_v2 --output scene.yaml
build/mountain-openmp/applications/c8_terrain_environment \
  --config scene.yaml --output environment_check.yaml
```

如需从新的地理配置准备数据，包内保留 `python -m corsika_terrain.geographic_terrain --config ... --estimate-only` 和 `--offline`。新区域首次仍需 DEM/大地水准面数据；无缓存时不能承诺离线运行。准备程序的旧 run-config 辅助函数不是 beta5 全链启动器，应先导出这里的 scene。

真实 21CMA 网格原点表面 U≈0，本轮穿界案例从下方 1 cm 向上注入；**这不是适用于任何山体的通用默认顶点**：

```bash
build/mountain-openmp/applications/c8_terrain_cascade \
  --scene scene.yaml --output photon_reference \
  --primary photon --energy-GeV 1 --position-m 0 0 -0.01 --direction 0 0 1
```

自然 νe 使用 `--primary nu_e --energy-GeV 10000`；强制顶点加 `--force-vertex-cc`，要求严格位于岩体内。该 CC 模块范围为 10⁴–10¹² GeV、νe/anti-νe；不包含 NC、核阴影或完整 τ 再生，强制事例不能直接解释成探测率。

`c8_terrain_cascade` 不接受虚假的 Kokkos/radio 开关来静默跑 CPU。默认电磁 cut=0.5 MeV、强子/μ/τ cut=0.3 GeV、emthin=1e-6；可通过帮助中的参数显式修改。它是独立参考程序，不改变 `c8_air_shower` 的任何默认值。

几何 oracle：

```bash
OMP_PROC_BIND=false build/mountain-openmp/applications/c8_terrain_device_probe \
  --mesh /path/to/terrain_enu.ply --samples 1000000 --threads 4 --output geometry_openmp.yaml
build/mountain-cuda/applications/c8_terrain_device_probe \
  --mesh /path/to/terrain_enu.ply --samples 1000000 --device 0 --output geometry_cuda.yaml
```

## 3. 实际验收证据

测试根目录为父项目的 `build/mountain-validation-20260908/`。大数据原文件仍在原磁盘位置，本轮不上传/替换生产结果。

### 场景与环境

- 真实网格 **64,694 顶点、129,384 三角面**，不是凸包；高度范围 **2092.508275–3509.064223 m ASL**。
- ENU 原点椭球高 **2750.244591 m**，EGM96 起伏 **−51.755409 m**，对应 ASL **2802 m**。二者不能混用。
- 与原 mountain 已保存的原生环境逐站比较，**80 个站的 ENU、ASL、空气密度、折射率，以及地形最低/最高海拔，数值差均为 0**。
- 保留原 `dem_agl` 地表上方 1 m 高度策略；这是 DEM 上的建模高度，不是已完成实测海拔校准的声明。
- Python 迁移单测 **32 项通过**；环境输入门禁 **11 项通过**；`testTerrainEnvironment` 通过，含拓扑、五层密度、出入/重入边界、扁平 BVH 与边角射线。

证据：[环境逐项比较](../../build/mountain-validation-20260908/environment_reference_comparison.yaml)、[输入门禁](../../build/mountain-validation-20260908/scene_gates.yaml)、[CTest](../../build/mountain-validation-20260908/terrain_ctest_final.log)。

### 真正执行的设备几何

各执行 **1,000,000 条相同射线**，轮流检查最近命中、向外退出、向内进入；10% 为竖直方向。参照是复用的 mountain CPU BVH/三角形算法，不是另一个扁平实现自比。

| 项目 | OpenMP 4 线程 | RTX 4060 Laptop CUDA |
|---|---:|---:|
| 命中数 | 328143 | 328143 |
| 命中/未命中不一致 | 0 | 0 |
| 三角面编号不一致 | 0 | 0 |
| 超过距离门限的点 | 0 | 0 |
| 最大距离差 | 0 m | 2.11e-10 m |

距离门限为 `1e-8 m + 1e-12 * |CPU 距离|`。73,905 个 BVH 节点，扁平几何占 **17,668,320 bytes**。这只是几何内存，不应拿它与完整 shower 的 70% 显存上限比较，也不把一次几何 kernel 计时宣传为 shower 加速比。

证据：[OpenMP](../../build/mountain-validation-20260908/terrain_oriented_openmp_v2.yaml)、[CUDA](../../build/mountain-validation-20260908/terrain_oriented_cuda_v1.yaml)。

### CPU 完整边界参考

同一实际 21CMA 场景、seed=67101、表面下 1 cm 向上注入、B=0、radio off：

| 事例 | 总步数 | 岩内/空气步数 | 岩→气 / 气→岩 | 介质错配 |
|---|---:|---:|---:|---:|
| 1 GeV photon，默认 cut/thinning，出山 | 1548 | 1 / 1547 | 1 / 0 | 0 |
| 1 GeV photon，表面上方 1 cm 向下入山 | 1615 | 1614 / 1 | 0 / 1 | 0 |
| 10 TeV νe 自然穿越 | 6 | 1 / 5 | 1 / 0 | 0 |
| 10 TeV νe 强制 CC，**emthin=0.1** 功能试验 | 170382 | 25 / 170357 | 16 / 1 | 0 |

自然事例无 CC、最终离开物理世界，能损为 0；强制事例使用了真实 Pythia CC 末态和后续强子/电磁过程，546 次物理世界逃逸，轨迹未截断。强薄化案例**不用于高精度物理/探测率/射电统计验收**。有核静质量交换和逃逸时，也不能仅用沉积能量判定闭合。

流式完整性检查（有限数值、正能量/权重、时间方向、单位方向、CSV/summary 计数与介质一致）
使用 `validation/terrain/analyze_reference_runs.py`；结果见
[CPU 轨迹验收](../../build/mountain-validation-20260908/terrain_cpu_track_acceptance.yaml)。
此前有限凸体的四个测试加新增真实地形测试，本轮合计 **5/5 CTest 目标通过**，
记录见 [CTest 汇总](../../build/mountain-validation-20260908/mountain_all_ctest_final.log)。

两次未完成尝试也保留：`forced_coarse_v1` 因改变 cut 触发新缓存生成而停止；`forced_cached_v2` 在实际发展到 200 万步后触发诊断预算。没有把它们算作通过。后一份失败轨迹由约 693 MiB 压为约 221 MiB 的 `tracks.csv.gz`，可解压恢复。

现在诊断流最多写 200,000 行并显式报告 `csv_truncated`；超过 2,000,000 计算步直接失败，`complete=false`。这两个门禁是**参考诊断成本保护**，不是物理 cut；被截断的轨迹不能用于完整 radio replay。后续生产实现应使用受控流式批处理，而不是在 RAM 中积累完整 shower。

## 4. 尚待完成的工作，不能省略

1. **原生表多介质管理**：空气与 SiO₂ 分别导出/校验，建立介质→native view/aux 的对应；不能删除 `requireSingleComposition` 把两种物质当成一张表。
2. **设备跨界 continuation**：在现有 photon/lepton 波前中引入显式材料切换记录，保存 history/time/weight/step ID，边界没有实际反应就不得重复抽随机数或重复计能。当前 `EscapedEnvironment` 不能代表“进入空气”。
3. **长寿命 Kokkos 几何/物理 session**：后续已增加 `KokkosTerrainSession`，借用外部 runtime、常驻只读 BVH、固定查询工作区；几何部分已有 OpenMP 验证，CUDA 编译通过。仍需在 PSR 实测并接入多介质物理 session，不能把几何会话当作完整输运实现。
4. **地形内外 radio**：复用原 mountain 经审计的路径/遮挡、局部 Snell/Fresnel、空气折射率积分语义，再接入 Kokkos CoREAS/ZHS。原地形射电本身存在近似和部分未通过收敛项，不能以 firn 或均匀介质结果代替验收。
5. 对同一 CPU 轨迹做边界、能损、radio 的逐项 replay，再扩展成 CPU/OpenMP/CUDA 整 shower 比较。HIP/SYCL 还需对应硬件。

本轮新增文件和目标均独立；没有安装覆盖生产程序。`applications/c8_air_shower.cpp` 本轮前后 SHA-256 为：

```text
d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e
```

这证明该应用文件未被本轮改动，不代表所有后续通用后端改造可以免做大气回归。
