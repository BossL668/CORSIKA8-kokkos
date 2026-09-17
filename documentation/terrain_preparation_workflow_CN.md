# beta5 通用地形准备入口：经纬度 → DEM → 标准大气中的山体

准备工具实测日期：2026-09-09；安装入口与功能说明核对：2026-09-17。
从空环境开始请先看[当前山体完整教程](terrain_getting_started_CN.md)。
第 6 节保留历史测量，不表示后续物理改动已经复用该验收。

## 1. 哪些已经存在，本轮新增了什么？

beta5 的 `c8_terrain_cascade` 已经使用通用非凸 `ClosedMesh/BVH` 和
`TerrainAtmosphere`：真实 DEM 岩体是 CORSIKA 原生 USStdBK 五层球形大气的子节点。
它不同于 `c8_mountain_neutrino`，后者是有限凸、均匀岩体示例，不能代替真实山谷地形。

原 mountain 项目的下载、高度基准、ENU、闭合网格、天线定位模块此前已迁移到
`python/corsika_terrain/`，但准备产品、导出 scene、运行应用需要分开操作。
本轮新增 `workflow.py` 和 **`c8-terrain`** 入口，将已有实现接成一条可复用工作流，
不重新实现几何算法，也不在运行时依赖另一个 mountain 项目目录。

```text
经纬度范围 / YAML + 天线坐标与高度策略
                  │
              c8-terrain
                  │
      DEM + EGM96 下载 / 校验缓存
                  │
  正高 H → 椭球高 h=H+N → ECEF → 真地理 ENU
                  │
      地表网格 + 侧壁 + 底面 → 闭合 PLY
                  │
    地理天线 + manifest + 相对路径 scene.yaml
                  │
       Scene.build → TerrainAtmosphere
       USStdBK 五层球形大气 + SiO₂ 岩体
                  │
    c8_terrain_environment：C++ 几何/介质检查
    c8_terrain_cascade：标量或 Kokkos 岩气输运
```

源码分工：

| 文件 | 职责 |
|---|---|
| `python/corsika_terrain/workflow.py` | 新的一站式 Python API/CLI；缓存和输出锁、scene 接线、显式启动应用 |
| `python/corsika_terrain/terrain_download.py` | 迁移的 HTTPS、分块 HGT、下载大小限制、SHA-256 缓存、EGM96 |
| `python/corsika_terrain/geographic_terrain.py` | 区域裁剪、地理/高度转换、天线、网格与诊断图；本轮增加内联天线输入 |
| `python/corsika_terrain/terrain.py`、`terrain_array.py` | 闭合岩体、ECEF/ENU 和原 21CMA 测量文件解析 |
| `python/corsika_terrain/scene.py` | 校验产品、导出原生大气 scene，而不是旧的均匀空气球 |
| `corsika/geometry/terrain/` | C++ 水密非凸网格、BVH、边界查询及设备数据 |
| `corsika/modules/terrain/TerrainAtmosphere.hpp` | 原生球形五层大气内嵌岩体 |
| `applications/detail/mountain/TerrainScene.hpp` | YAML 到已封装的环境构建器；observer 与高度门禁 |
| `applications/c8_terrain_cascade.cpp` | 组装物理过程、选择后端、启动 shower，不执行 DEM 下载 |

迁移出处见 [terrain_migration_sources.json](terrain_migration_sources.json)，既有跨介质实现见
[多介质输运](terrain_multimaterial_kokkos_transport_validation_CN.md)与
[磁场/应用边界](terrain_magnetic_application_validation_CN.md)。本轮没有修改上述 C++ 文件。

## 2. 安装与最简准备

在 **beta5 源码目录**运行。Python 安装和 C++ 应用构建是两件事：

```bash
conda activate corsika_venv
python -m pip install ./python

# 只估算下载量、网格和内存，不创建输出、不联网
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" --estimate-only

# 首次自动下载；后续同配置校验并复用
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo"
```

顺序是 **西经度、南纬度、东经度、北纬度**，单位为十进制度。默认网格间距 3 角秒、
闭合底面深度参数 500 m、SiO₂ 密度 2.65 g/cm³。
未提供天线时，明确创建一个名为 `reference_centre` 的**演示观测点**，位于区域中心地表上方
1 m；这不是 21CMA 实际阵列。位置、高度、分辨率、岩体假设均需要按研究目标审定。

默认缓存位于 `${XDG_CACHE_HOME:-$HOME/.cache}/corsika8/terrain`。
可用 `--cache PATH` 指定其他磁盘；`--offline` 禁止网络，缓存不足或损坏则失败。
修改网格、天线或范围时请换输出目录，工具不会覆盖已有不同配置的科研产品。

安装包仅提供 `corsika_terrain`，不替换原有 `corsika` Python 输出读取包。
也可不安装而用 `PYTHONPATH=python python -m corsika_terrain.workflow ...`。

## 3. 接入天线阵列

通用 CSV 最少包含下面三个列名，角度默认十进制度：

```csv
name,longitude_deg,latitude_deg
A01,86.705,42.935
```

在 bounds 命令上加 `--antennas-csv stations.csv`，或使用原 mountain 的 21CMA 测量文件目录：

```bash
c8-terrain --bounds 86.675 42.910 86.750 42.975 \
  --station-directory "$HOME/21cma-stations" \
  --output "$HOME/CorsikaData/terrain/21cma"
```

`--station-directory` 内的数据应为原解析器接受的 `dump.txt` / `20160810.csv` 站中心文件，
不是任意三列 ENU 文本。测量 ECEF 站位转为经纬度后，默认使用 **DEM + AGL=1 m** 建模，
保留原测量值与差值记录，不把 DEM 高度称为实测高度。

需要显式控制高度基准、分辨率、站点选择时使用
[示例 YAML](../configs/mountain/terrain_region_21cma.yaml)，先修改其中的 `station_directory`。
相对路径以 YAML 所在目录为基准；支持 `~`。然后：

```bash
c8-terrain --config configs/mountain/terrain_region_21cma.yaml
```

对于有实测高度的通用 CSV，可在 YAML 内设置
`format: geodetic_csv`、`path: stations.csv`、`height_mode: measured`、
`measured_height_datum: ellipsoidal` 或 `EGM96`，CSV 增加 `height_m` 列。
实测位置落入岩体时直接报错，不暗中上移。范围之外的天线默认报错；只有显式设置
`outside_policy: exclude` 才排除并记录。**地理 ENU 不是磁场 NWU，也不是射电分析的 Ex′/Ey′/Ez′。**

## 4. 接入 C++ 应用

先按主 README 配好合法 FLUKA、PROPOSAL/Kokkos 等依赖，再构建山体目标。
独立构建步骤见[当前山体构建说明](terrain_getting_started_CN.md#1-编译可选应用)，
需 `CORSIKA_BUILD_MOUNTAIN_APPLICATION=ON`、`WITH_FLUKA=ON`。
下面使用当前教程生成的 `../install/openmp/bin`，不再要求历史构建目录。

准备时可直接调用 C++ 门禁，验证**实际** USStdBK 内嵌结果：

```bash
c8-terrain --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" \
  --check-with ../install/openmp/bin/c8_terrain_environment
```

只准备场景不会启动 shower。先做一个空气向岩体入射的 2 MeV 光子小测试：

```bash
# FLUPRO 应已指向当前机器的合法 FLUKA 安装，不能省略该环境配置。
OMP_NUM_THREADS=2 OMP_PROC_BIND=false c8-terrain \
  --bounds 86.700 42.930 86.710 42.940 \
  --output "$HOME/CorsikaData/terrain/demo" --offline \
  --run --application ../install/openmp/bin/c8_terrain_cascade -- \
  --primary photon --energy-GeV 0.002 \
  --position-m 0 0 1 --direction 0 0 -1 \
  --em-backend kokkos --threads 2 --device-memory-MiB 512 --batch 8 \
  --resident-capacity 256 \
  --output "$HOME/CorsikaData/terrain/photon_test"
```

`--` 后参数原样交给应用；省略 `--em-backend kokkos` 则使用标量 PROPOSAL。
使用 GPU 时选择已经编译的 `../install/cuda/bin/c8_terrain_cascade`、`--threads 1`、`--device 0`，
不要给 OpenMP 可执行文件加参数后误认为会启动 GPU。
准备工具不自动设置能量、注入点或后端，不静默回退。
此处 ENU 原点在区域中心地表；其它场景应先核查注入位置。

山体内强制 νe CC 示例可在上述场景中显式选择：

```bash
../install/openmp/bin/c8_terrain_cascade \
  --scene "$HOME/CorsikaData/terrain/demo/terrain/scene.yaml" \
  --primary nu_e --energy-GeV 10000 --force-vertex-cc \
  --position-m 0 0 -1 --direction 0 0 1 \
  --em-backend kokkos --threads 2 \
  --output "$HOME/CorsikaData/terrain/nue_cc"
```

这是一条使用说明命令，**本轮未将该高能事例作为已完成验收**。CC 顶点必须在岩体内部；
当前已接通相应模型能区内的 CC/NC、τ 衰变与再生路由，但不提供强制顶点事件率权重，
也不能声称完整弱物理/事件极化已验收。见[模型适用范围](terrain_original_neutrino_alignment_CN.md)。

## 5. 产物、复现与限制

```text
demo/
  terrain_request.yaml                  bounds 命令保存的请求
  terrain/
    scene.yaml                          应用入口；mesh_path 为相对路径
    terrain_enu.ply                      闭合网格
    terrain_grid.npz                     坐标/高度审计数据
    terrain_manifest.yaml               请求、来源、SHA-256 与定位 QA
    observers.yaml / corsika_geometry.yaml
    01_geographic_and_enu_array.png
    02_terrain_mesh_and_antennas_3d.png
    03_height_datum_and_mesh_clearance.png
    native_environment_check.yaml       仅使用 --check-with 后产生
```

整个 `terrain/` 可一起移动供 C++ 读取，manifest 中原始缓存路径仍保留用于溯源；
再次在 Python 中准备/验证产品需要可访问的原缓存或新输出请求。
下载有 TLS、字节/时间/磁盘限制；缓存复用校验 SHA-256，损坏不覆盖，nodata 不填平。
SHA 是下载内容和产物的一致性记录，**不是上游签名或 DEM 科学精度保证**。
新入口对缓存和输出加进程锁，崩溃后锁自动释放；不要同时运行不遵守这些锁的旧准备命令修改相同缓存。

默认数据源为 [Mapzen terrain tiles / Skadi 公共存储](https://registry.opendata.aws/terrain-tiles/)，
高度转换网格来自 [PROJ CDN](https://cdn.proj.org/)。保留 manifest 的 NASA/NGA/USGS、
Mapzen 或 GRAND 数据署名；Skadi 与 GRAND 数据不能未经核对称为完全相同 DEM。

当前边界必须明确：

- 这是**局部有限 DEM 岩体嵌入球形地球坐标下的标准大气**，不是全球固体地球密度模型。
- ENU 范围每轴最多 2°，拒绝跨日界线/极区；预算默认至多 4 瓦片、120000 顶点、250000 面。
  内存估算是预检而不是 OS 强制内存限额，实际内存应监控。
- native 模块要求全部岩体顶点在海拔 0–7 km；海床、负海拔、跨层山体和更换大气模型不是当前通用入口的已支持能力。
  `--check-with` 或运行应用时做 native 层门禁，仅 Python 准备成功不等于 C++ 门禁通过。
- 底面/侧壁是有限区域人工边界，应扩大范围与底面深度做收敛检查；默认 SiO₂ 不等于测得的岩石组分。
- 大气使用 ASL；ECEF/IGRF 使用椭球高，保留已有球形大气与 WGS84 局部定位约定。
- 默认空气 IGRF14/2027、岩内 B=0。当前 DEM 应用可显式开启 Kokkos 界面射电，
  但天线准备成功不等于传播近似、有限阶矩或时域波形已验收。
  使用与限制见[当前界面射电说明](interface_kokkos_radio_CN.md)。

## 6. 本轮实测（2026-09-09）

结果位于父项目 `build/terrain-prepare-validation-20260909/`，没有移动原始生产数据：

| 验证 | 结果 |
|---|---|
| Python 工作流/地形/天线离线回归 | 40 项通过；含不联网估算、复用、冲突拒绝、scene 篡改、锁、启动参数 |
| 既有 C++ 几何/环境回归 | `testTerrainEnvironment` 通过；未重新编译或改变 C++ 实现 |
| 真实 HTTPS 首次下载 | 新下载 `N42E086.hgt.gz` 12,491,002 bytes；EGM96 自动下载；来源/哈希已保存 |
| 小区域 native 门禁 | 218 顶点、432 面、1 个空气 observer；原点 ASL 2796 m |
| 相同请求离线复用 | `reused: true`，scene 和 native 检查文件一致 |
| 原 21CMA 站中心场景 | 7526 顶点、15048 面、80/80 空气 observer；无排除 |
| 21CMA 位置 QA | 水平往返误差最大 5.83×10⁻¹³ 度；ENU-Up 地表净空约 1 m（3.82×10⁻⁹ m 内） |
| CPU 2 MeV 光子入山 | 完成；9 步，空气→岩体 1 次，介质错配/表面歧义均 0 |
| Kokkos OpenMP 2 线程同设置 | 完成；10 个加速步，7 批，空气→岩体 1 次；无待处理粒子、错配、歧义或指定回退 |
| 两条小输运能量检查 | 均沉积约 0.002 GeV；记录未截断 |

首次 Kokkos 小测试设置了过小的 32 MiB 预算，被 native 双介质表的约 75.8 MiB 预检拒绝；
改为 128 MiB 后通过，门禁未放宽。**CPU/Kokkos 步数不同；这里验证实际后端接线与跨界完整性，
不是逐步相同、GPU 性能或 shower 大样本验收。** 本轮未占用生产 GPU。

对应产物：`online/terrain/`、`21cma/terrain/`、`photon_cpu/`、`photon_kokkos128/`。
`photon_kokkos/` 保留为 32 MiB 失败记录，不混入成功结果。
历史图位于上述本地结果根目录的 `21cma/terrain/`：
`01_geographic_and_enu_array.png`、`02_terrain_mesh_and_antennas_3d.png`、
`03_height_datum_and_mesh_clearance.png`。本机构建目录不随源码包分发，不作为 GitHub 下载链接；
运行准备工具会在自己的输出目录生成对应图。
Python 单测另有 26 条来自既有 pyproj/NumPy 单点转换的弃用警告，本轮没有修改该数学路径。
独立 wheel 已构建并安装到 `corsika_venv`，其 SHA-256 为
`84800915d20da29c485158d76776487f6faf1970dbd53e8814ec779360480716`。
`applications/c8_air_shower.cpp` SHA-256 仍为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`；
`c8_terrain_cascade.cpp` 仍为
`57e4221a06686803cac2ef90e291aae87b76d7b93be1365319594c3dc76b53f8`。

复跑离线单测（源码目录）：

```bash
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  validation/terrain/test_workflow.py \
  validation/terrain/test_geographic_terrain.py \
  validation/terrain/test_terrain_array.py
```
