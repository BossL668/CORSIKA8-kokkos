# 通用双侧介质界面：shower 输运接口

范围：beta5，2026-09-10。只处理粒子输运，不处理界面射电。

## 接口层次

```text
定向封闭网格 + 两侧区域映射
              │
              ▼
nextCrossing：最近交点、面编号、from_region / to_region
              │ 与离散反应、连续步长、磁偏转步长、时间上限竞争
              ▼
InterfaceEmSession：使用当前区域对应的物理 bank 推进一步
              │ 只有实际到达网格界面，才切换逻辑区域
              ▼
InterfaceEmRouter：继续 γ/e± 或回交指定 CPU fallback
              │
              ▼
调用方输出：轨迹、沉积、存活粒子；没有在 crossing 处吸收粒子
```

几何模块不读取材料名称、PROPOSAL 或 YAML；物理公式继续使用原 beta5 共享实现。
物理区域与环境内部大气层编号分开处理，内部层切换不会把粒子误分配给另一材料。

## 源码入口

| 文件 | 职责 |
|---|---|
| `corsika/geometry/interfaces/MaterialInterface.hpp` | 两侧区域和 bank 映射、直线/二次轨迹边界查询 |
| `corsika/modules/transport/HomogeneousMaterial.hpp` | 用显式 CORSIKA Medium、核组分、密度、折射率和磁场构造模型；收集实际靶核；检查 calculator key 冲突 |
| `corsika/modules/transport/InterfaceTracking.hpp` | CPU 跟踪入口；复用既有 SI leapfrog 与网格求交 |
| `corsika/modules/transport/InterfaceMaterialPreparation.hpp` | 从实际 calculator 的只读 view 导出 bank；复用 cut/能区/hash 与辅助缓存流程 |
| `corsika/modules/transport/InterfaceEmSession.hpp` | Kokkos γ/e−/e+ 有界输运会话；保持旧 terrain 类型的源码兼容 |
| `corsika/modules/transport/InterfaceRegionMap.hpp` | 按 CPU 逻辑环境节点映射区域，不按几何类型猜材料 |
| `corsika/modules/transport/InterfaceEmRouter.hpp` | 可复用 HybridCascade 路由；输出与 fallback 由调用方提供 |
| `applications/detail/mountain/TerrainEmRouter.hpp` | 仅保留山体应用 YAML 统计适配 |

会话实现仍位于 `src/terrain/TerrainEmSession.cpp`，链接目标仍为 `CORSIKA8TerrainEm`；
通用头提供兼容入口，未重复实现一套物理 kernel，也未修改空气生产入口。

## 如何复用

```cpp
namespace api = corsika::interfaces;

// 区域 17 为网格外侧，区域 93 为内侧；bank 索引与区域编号无关。
api::EmConfig config;
config.interface = {17, 93, 0, 1};
config.threads = 2;             // OpenMP 示例
config.batch_size = 64;
config.maximum_device_bytes = 256u * 1024u * 1024u;

// mesh 是验证过并导出的 FlatTerrainData。
// materials 的每一项包含实际 PROPOSAL 表、辅助数据及对应环境快照。
api::InterfaceEmSession session(mesh, materials, config);
particle.medium_id = config.interface.outside_region; // 会话内表示逻辑区域
auto records = session.advance({particle}, first_child_history);
```

`advance()` 不负责创建 CORSIKA CPU 环境，也不能把缺失介质的参数猜出来。
调用者必须为两侧提供一致的实际环境模型与快照（坐标单位 m、能量 GeV、磁场 T），
并在收到 crossing 后继续使用返回的区域。不会修改粒子位置来强行越界。

均匀材料模型不依赖山体形状，例如：

```cpp
api::HomogeneousMaterial water{
    corsika::Medium::WaterLiquid,
    corsika::NuclearComposition(
        {corsika::Code::Hydrogen, corsika::Code::Oxygen}, {2./3., 1./3.}),
    1.0, 1.33, {0., 0., 0.}
};
node->setModelProperties(water.makeModel<EnvironmentInterface>(coordinates));
```

组分比例是**核数比例**，不是质量比例。新材料首次运行仍可能生成 PROPOSAL 自身缓存和辅助缓存；
无需手动生成 `.c8emrt`。只改密度不会改变化学组分；本轮不使用折射率计算射电。

## 山体应用的使用方式

原场景和默认 SiO₂ 行为继续保留。`geometry.material` 现在可以选择 `SiO2`、`Water` 或 `Ice`。
例如在准备好的场景中将以下字段改为：

```yaml
geometry:
  material: Water
  rock_density_g_cm3: 1.0
  rock_refractive_index: 1.33
  # mesh_path、rock_reference_enu_m、mesh_sha256 等沿用实际几何准备结果。
```

然后仍通过 `c8_terrain_cascade --scene scene.yaml ... --em-backend kokkos` 运行。
CLI 没有新增必填参数。`rock_*` 配置名及轨迹 CSV 的 `rock` 标签是旧版兼容名称，
现在表示**网格内区域**；实际材料写入 `material_interface.inside_medium`。
FLUKA 靶核自动取实际环境核组分的并集，因此水/冰的氢靶不会遗漏。

## 明确的适用边界

- 一个定向封闭网格及其内外两区；允许非凸地形，也允许两区复用同一个物理 bank。
  不支持任意多个相交界面、三相交线、多重嵌套区域自动导航。不能将未验证的自交网格送入求交器。
- 支持的密度模型仍由 beta5 环境快照决定，不是任意用户函数。
  球层模型可配置均匀磁场；旧凸多面体环境快照仍要求 B=0，非法组合在初始化前拒绝。
  非零磁场时，只在当前有限磁步内查找界面；该范围内没有交点不等于几何损坏。
- 本轮山体应用仍采用 USStdBK 外部大气、网格内均匀材料/零磁场，地形必须位于海拔 0–7 km。
  这是该示例的装配限制，不是 `nextCrossing()` 对材料名称的限制。
- 原 CORSIKA PROPOSAL 包装器按核组分 hash 管理 calculator。同组分、不同 Medium 物性
  （例如同一包装器中同时放水与冰）存在键冲突，现在明确拒绝。
  通用会话可以接收分别构造/导出的独立 bank，但应用的 CPU fallback 也必须匹配独立 calculator；
  不能靠修改 density 或绕过门禁解决。
- 当前 Kokkos 会话处理 γ/e±；μ、τ、强子、中微子继续走现有 CPU 过程。
  本次封装不会自动补齐弱反应低能模型、τ 自旋输运或跨界射电。
- 会话采用有界同步批次；这不是全新多介质生产驻留调度器或生产性能验收。

## 测试

新增 `testInterfaceTransport` 使用实际 CPU PROPOSAL calculator 和编译后的 Kokkos 输运会话，
不使用伪造物理表。测试体积包括箱体和四面体；介质包括水/SiO₂，独立区域编号为 17/93，
并测试同一材料的两侧和两侧非零磁场。

本轮已完成：

- OpenMP Release 的 8 项 CTest 通过；其中 5 种双侧配置各 100 万次解析几何查询，
  总计 500 万次。另对 γ/e−/e+ 的 30 个真实跨界输入检查 CPU/Kokkos 的位置、时间、
  逻辑区域、能量合法性、跨界前后的 grammage 与重复性。
- 实际 DEM 场景中，SiO₂、水、冰各运行 1 个 CPU 和 1 个 OpenMP 光子小事例，
  六例均完整，无介质错配、CSV 截断或未清空的 Kokkos 队列。
  这是小样本装配检查，不是 CPU/Kokkos 相同随机树或介质物理分布的统计验收。
- 原 SiO₂ CPU/OpenMP 的固定种子结果与重构前逐文件比较：
  `tracks.csv`、`deposits.csv`、`window_survivors.csv` 全部 SHA-256 一致。
  `--help` 相同，物理 metadata、计数器和表哈希相同；耗时及描述字段除外。
- CUDA Release 应用和接口测试程序编译通过；本轮没有启动 GPU 测试，不与空气生产任务争用显卡。
  审计版 OpenMP 应用也已编译。
- 非法区域/bank、非法密度/磁场以及同组分不同 Medium 的计算器冲突明确拒绝。
  首次水介质缓存生成曾触及 600 s 保护上限，后续复用已写入的原生缓存完成。
  测试夹具的球层观测参考半径也修正后重新通过；这些失败日志保留且不计为通过。

实际 DEM 小事例为 0.1 GeV 光子、seed=67101、emcut=0.5 MeV、无薄化、20 ns 窗口、
0.1 m 诊断步长，OpenMP 使用 2 线程。单进程峰值 RSS 小于 1 GiB。
测得的端到端时间包括 FLUKA/QGSJet 初始化及新材料缓存生成，不能当作加速比。

测试结果与资源记录保存到 D 盘 `CorsikaData/corsika_validation_results/beta5_generic_interface_20260910_v1/`。
最终完成情况以该目录的验收摘要为准；小样本完整性测试不等同于材料物理系综验收。

OpenMP 编译/测试（已有 beta5 编译环境）：

```bash
conda activate corsika_venv
export FLUPRO=/path/to/fluka
cmake --build ../build/mountain-openmp \
  --target c8_terrain_cascade testInterfaceTransport testTerrainEnvironment -j 1
OMP_NUM_THREADS=2 OMP_PROC_BIND=false \
  ctest --test-dir ../build/mountain-openmp \
  -R 'InterfaceTransport_|testTerrainEnvironment|testTerrainCurvedBoundary|testKokkosTerrainSession' \
  --output-on-failure
```
