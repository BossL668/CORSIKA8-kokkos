# beta5 地形常驻会话与 PSR 验收进度

日期：2026-09-08。范围：继续真实 DEM 山体迁移；本轮不使用本地 GPU 执行测试，不替换生产 build/install。

后续用户授权本地使用不超过 10% 显存后，已接通并补测有界 Kokkos 多介质 EM 输运。
见[最新实现与测试范围](terrain_multimaterial_kokkos_transport_validation_CN.md)。
本文保留此前只做几何会话测试及 PSR 驱动阻塞的历史记录。

**结论：常驻几何与逻辑边界候选已实现，OpenMP 参考测试通过，CUDA 编译通过；PSR GPU 实测被驱动/内核不匹配阻塞。完整多介质 Kokkos shower 和内外射电仍未接通。**

## 1. 本轮交付

| 文件 | 职责 |
|---|---|
| `corsika/geometry/terrain/KokkosTerrainSession.hpp` | 借用已经初始化的 Kokkos runtime；一次上传只读 BVH；返回带生命周期约束的设备 view；固定容量查询工作区，多次调用复用 |
| `corsika/geometry/terrain/TerrainBoundary.hpp` | 与原 CPU `TerrainBoundaryTracking` 一致的逻辑侧求交：岩→气退出、气→岩进入及无交点三种结果 |
| `src/terrain/TerrainDeviceQuery.cpp` | 独立诊断包装器拥有一个 runtime/session；所有重复查询共用它们，而不是每批重新初始化 |
| `applications/c8_terrain_device_probe.cpp` | 新增 `--logical-boundary`、`--batch-size`、`--repeats`，报告几何上传次数、固定缓冲容量、批次数和重复结果差异 |
| `tests/accelerator/testKokkosTerrainSession.cpp` | 实际 Kokkos 执行、解析立方体、错误输入、边界起点、跨批次复用和 runtime 所有权测试 |
| `tests/modules/testTerrainEnvironment.cpp` | 对照实际 CPU `BoundaryTracking::intersect()`，并在存在交点时对照 `getTrack()` 的时长和目标节点 |

接口示意（内部使用，不是完整 shower 启动器）：

```cpp
// 调用方先建立既有 Kokkos runtime。
{
  terrain::KokkosTerrainSession<ExecutionSpace> terrain_session(flat_mesh, 4096);
  auto geometry = terrain_session.deviceView(); // 设备地址，不一定可在 host 解引用
  // 后续 kernel 可借用 geometry；同一会话跨批次/事件复用。
  // 验证工具也可以调用 query() / queryBoundaries() 得到有界分批结果。
} // 所有借用 geometry 的 kernel 必须先完成；先销毁 session，再 finalize runtime。
```

这个会话不创建第二个 Kokkos runtime；其类析构也不会关闭调用方 runtime。
诊断查询是同步的，内部每批返回前 fence；供生产 kernel 借用 `deviceView()` 时，调用方必须保证最后一次设备访问在会话析构前结束。
对象不是多 host 线程并发容器。模板允许调用方显式选择 execution space，不隐式选择另一种后端。

只读几何上传前检查前向 BVH 链接、索引覆盖/重复、有限坐标、法向和退化面数据。
这些是地址/数据一致性门禁，**不是任意网格无自相交的证明**；闭合/拓扑门禁仍由既有地形入口执行。

## 2. 物理语义边界

`nextBoundary()` 只给出候选距离、面编号、`EnterRock / ExitRock / None`，**不执行粒子跨界**：

- 不移动位置，不改变时间、能量、权重、history 或随机状态；接口没有 RNG 输入。
- 与 CPU 一样，仅把求交原点向后移 `1e-8 m`，并保留 CPU 的 `1e-9 m` 最小边界飞行距离；没有把用户 padding 当几何分辨率。
- 边界是材料切换候选，不能当作 `EscapedEnvironment` 吸收粒子。
- 它仍需与大气层边界、相互作用及连续步长竞争；不能把一次最近三角形命中直接当作完整 transport。
- 尚未建立空气/SiO₂ 各自 native 表到设备材料的映射；没有删除 `requireSingleComposition` 门禁，没有把岩石物理率用于空气。

这一步仅补齐后续跨界 continuation 的几何基础，不能宣称已经通过整 shower 能量账本、粒子身份或最终射电验收。

## 3. 本轮验收

独立构建：`build/mountain-openmp`、`build/mountain-cuda`。
结果：`build/mountain-validation-20260908/resident_session_v1/`。

- OpenMP Release：`testKokkosTerrainSession`、`testTerrainEnvironment`、`c8_terrain_device_probe` 构建成功。
- CUDA Release：新 session 测试及 probe 编译/链接成功；只做编译，**没有在本地 GPU 执行测试**。当前这个构建是本机架构 89，不能直接拿去 T400 运行。
- 6 个相关 CTest 目标通过，见 `ctest_openmp_final.log`；包含既有凸体、均匀射电、中微子和地形测试，不能据此声称非凸地形射电已通过。
- 地理准备 Python 测试 32 项通过；保留两个既有 pyproj/NumPy 弃用 warning，见 `python_terrain.xml`。
- 单会话 20 次各 4097 条解析立方体查询，共 81940 条，另有 7 条边界查询；容量固定 31，几何设备指针保持不变，工作区始终 2728 bytes。验证了这条几何查询路径不随批次数扩大工作区，**不是整个 shower 无泄漏的全面证明**。
- 坏索引、重复面、缺面、非有限法向、无 runtime、零容量、非单位方向等错误输入被拒绝。

真实 21CMA 网格为 129384 面、73905 BVH 节点。100 万条随机逻辑侧查询对照原 mountain 迁移的 CPU BVH/有向法线语义，在 **OpenMP 2 线程**下重复 3 次：

| 指标 | 结果 |
|---|---:|
| 每轮命中 | 297414 |
| 命中/未命中差异 | 0 |
| 面编号差异 | 0 |
| 距离超限 | 0 |
| 最大距离差 | 0 m |
| 重复执行差异 | 0 |
| 几何上传 | 1 次 |
| 几何数组 | 17668320 bytes |
| 查询容量 | 4096 |
| 查询/结果工作区 | 360448 bytes |
| 三轮总批数 | 735 |

见 `logical_boundary_openmp_million.yaml`。主机镜像/调用方输入与结果的内存不包含在上述设备工作区数字中；OpenMP 的 HostSpace 镜像可共享分配，GPU 的镜像是额外 host staging。
计时字段改为 `query_fenced_wall_seconds`，包含分批 staging、传输及返回，**不冒充 kernel-only 计时，更不是 shower 加速比**。

另复跑原最近/有向命中模式：100 万条、容量 257、重复 2 次，共 7784 个查询批次，
每轮命中 328143（与迁移基线相同），存在性/面号/距离及重复结果差异均为 0。
几何仍只上传一次，工作区 22616 bytes，见 `oriented_openmp_million.yaml`。

开发时一次 CPU 测试把“不存在交点”错误地解释成“全局 tracking 必须更换节点”；对人为不一致的逻辑侧输入，这个假设不成立。已改为先直接对照 CPU `intersect()` 的存在性，存在交点时再检查全局 tracking。没有为通过测试修改 CPU tracking。
CUDA 编译还修正了新测试目标在 Conan 间接 Kokkos 静态依赖下的 `--as-needed` 驱动库顺序，以及设备端临时聚合对象写法；修改范围均限于地形支持/测试。

可复现（在项目容器目录，先激活 `corsika_venv`）：

```bash
ctest --test-dir build/mountain-openmp \
  -R '^(testKokkosTerrainSession|testTerrainEnvironment)$' --output-on-failure

OMP_PROC_BIND=false build/mountain-openmp/applications/c8_terrain_device_probe \
  --mesh /path/to/terrain_enu.ply --samples 1000000 --threads 2 \
  --logical-boundary --batch-size 4096 --repeats 3 --output logical_boundary.yaml
```

原最近命中/有向命中模式也保留；不加 `--logical-boundary` 即可。输出文件存在时拒绝覆盖。

## 4. PSR GPU 阻塞证据

2026-09-08 只读检查发现：

```text
hostname: psrpku2025
uname -r: 6.8.0-138-generic
lspci: 61:00.0 NVIDIA TU117GL [T400 4GB / T400E] [10de:1ff2]
nvidia-smi: couldn't communicate with the NVIDIA driver
modinfo nvidia: Module nvidia not found
/dev/nvidia*: 不存在
```

包记录有 `nvidia-driver-595-open 595.84`，以及 `6.8.0-136`、`6.8.0-139` 等内核的 NVIDIA 模块，
但当前 `6.8.0-138` 没有可用模块。因此“机器有显卡”和“当前能执行 CUDA”是两回事。
本轮也未在 PATH、常见 CUDA 安装位置或 `corsika_venv/bin` 找到 `nvcc`；GPU 用户态编译环境需一并预检。

没有尝试 `sudo`、装驱动、换内核、卸载模块或重启机器。PSR 的
`c8-beta5-uhe100-openmp-psr-v3-20260908.service` 保持运行；本地 CUDA100 生产队列也未改动。

需要管理员先提供与**正在运行的内核**匹配的 NVIDIA 模块，或在安排停机后使用已有匹配模块的内核启动。
当前还有生产任务，不应直接为本测试重启 PSR。恢复后先确认 `nvidia-smi`、`/dev/nvidia*` 和 CUDA runtime 可用。

## 5. 恢复 GPU 后的下一步

1. 使用独立山体源码快照/build，锁定 Kokkos 4.7.03，并为 T400 配置 `TURING75 / sm_75`；不覆盖 PSR 已冻结的生产二进制、Pythia 数据或运行环境。
2. 在 PSR 执行上述 session 单测、真实 DEM 百万条逻辑边界/重复查询 oracle；明确记录 GPU 型号、CUDA 版本、输出哈希和误差。
3. 再接空气/SiO₂ 多介质 native 表和真实粒子 continuation，逐步核查随机状态、跨界/切层、能损/逃逸账本；最后接原 mountain 经范围审计的跨界 radio 路径。

**本轮不是完整山体 Kokkos 应用的生产验收。** 原空气应用文件 SHA-256 仍为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`；其生产程序、参数、随机流和安装目录未被本轮修改。
