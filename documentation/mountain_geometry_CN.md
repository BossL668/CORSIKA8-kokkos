# 有限凸山体几何接口

本模块把 `corsika8-mountain/src/cma21_askaryan/geometry.py` 中已使用的
`MeshGeometry` 外向面定向、闭合边检查和半空间包含关系，整理为 beta5 可复用的
C++ 接口。它不替换原有的 `IVolume`、`TrackingStraight`、球形大气或空气应用。

## 文件职责

| 文件 | 职责 |
|---|---|
| [ConvexPolyhedron.hpp](../corsika/framework/geometry/ConvexPolyhedron.hpp) | 构造、读取、验证有限凸体；实现 CPU `IVolume` 接口 |
| [ConvexPolyhedronData.hpp](../corsika/framework/geometry/ConvexPolyhedronData.hpp) | 无单位对象、无虚函数的只读平面 POD；CPU/Kokkos 共用包含与射线裁剪公式 |
| [TrackingConvexPolyhedron.hpp](../corsika/modules/tracking/TrackingConvexPolyhedron.hpp) | 显式选择的直线 tracking；扩展凸山体边界，其他形状转发原实现 |

```text
顶点 + 三角面 / OBJ / box / tetrahedron
                    │
             ConvexPolyhedron 验证
               ┌────┴────┐
      CPU IVolume         exportPlanes()
      独立 tracking       Kokkos 只读平面数组
               └────┬────┘
          同一 containsConvex / clipConvexRay
```

几何层只回答“在不在山体内、何时碰到边界”；不决定中微子截面、介质成分、
粒子阈值、出山后的大气演化或射电折射。

## 最小 C++ 用法

```cpp
#include <corsika/framework/geometry/ConvexPolyhedron.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>

using namespace corsika;
auto cs = get_root_CoordinateSystem();

// 中心坐标和半边长均以米输入：完整边长是 2000 × 2000 × 1000 m。
auto mountain = ConvexPolyhedron::box(cs, {0., 0., 0.}, {1000., 1000., 500.});

bool inside = mountain.contains(Point(cs, 0_m, 0_m, 0_m));
auto chord = mountain.intersectRay(Point(cs, -2000_m, 0_m, 0_m),
                                   DirectionVector(cs, {1., 0., 0.}));
// chord.intersects == true；entry_m == 1000，exit_m == 3000。
// 物理弦长为 (chord.exit_m - chord.entry_m) * 1_m。
```

四面体可直接用四个带单位的顶点构造：

```cpp
auto tetra = ConvexPolyhedron::tetrahedron({
    Point(cs, 0_m, 0_m, 0_m), Point(cs, 2000_m, 0_m, 0_m),
    Point(cs, 0_m, 2000_m, 0_m), Point(cs, 0_m, 0_m, 2000_m)});
```

已有三角网格可用 `fromVerticesMeters(cs, vertices, triangles)`；其中顶点是
`std::array<double, 3>`，单位为米，三角面是三个从零开始的顶点索引。
三角形的外向绕序会自动统一，但不会补洞、求凸包或改变顶点。

OBJ 文件调用 `ConvexPolyhedron::fromOBJ(path, cs, 1_m)`：输入坐标的一单位对应
`1_m`；若文件坐标单位为公里，显式传 `1_km`。支持三角面、正/负顶点索引和
`v/vt/vn` 索引写法；不自动三角化多边形。仅几何顶点索引参与体积计算。

## 坐标、边界与安全约定

- `contains(Point)` 与 `intersectRay(Point, DirectionVector)` 自动转换到构造时的
  `cs`；后者自动归一化方向。因此返回的 `entry_m/exit_m` 始终是米，不随输入方向长度变化。
- `exportPlanes()` 返回该 `cs` 中的 `nx, ny, nz, offset_m`，法向是单位外法向；
  内部满足 `nx*x + ny*y + nz*z <= offset_m`。设备位置必须使用同一坐标系，不能
  将局部山体平面直接与另一坐标系下的粒子位置混用。
- 默认包含容差为 `1e-8 m`。容差用于判定近表面点，不会平移真实交面。表面出射射线
  返回零长度区间；调用方必须执行出界处理，不能重新塞回同一个体积无限循环。
- 非有限坐标、零方向、非法索引、退化三角形、重复面、非闭合面、无效体积、未引用
  顶点以及非凸几何都会拒绝。包含容差不会放宽非凸验证。
- 当前 accelerator 环境快照最多接受 **32 个平面记录**。一个三角面导出一个记录，
  因而三角化 box 是 12 条、tetrahedron 是 4 条。CPU 几何对象本身没有此固定上限。
- 首版**不支持非凸 DEM、带洞地形或多个不连通山体**。不能把复杂地形的凸包当作原山体。
- `ConvexPolyhedronTracking` 是零磁场直线策略；带磁场的山体输运不能直接套用此策略。
  出山后的粒子和射电传播应由应用显式处理，几何模块不会偷偷替换为大气算法。

## 验证入口

[testConvexPolyhedron.cpp](../tests/framework/testConvexPolyhedron.cpp) 已加入
`testFramework`，可以仅选择 `[mountain]` 标签。它覆盖四面体的解析交点、表面入射/
出射、非单位方向、粒子速度到时间的换算、平移坐标系、OBJ 缩放及非法输入。
10,000 个随机位置还同时检查轴向和任意方向的 box 射线交点，与独立 slab 解析公式比较。

配套的 `testKokkosConvexEnvironment` 检查同一平面接口在 Kokkos kernel 中的行为；
`testMountainNeutrino` 检查中微子顶点，`testMountainRadio` 与
`testKokkosHomogeneousRadio` 检查 CPU/Kokkos 的均匀介质射电。
这些局部验证不等于已经通过大样本山体中微子物理验收。
