# 山体共享顶点/共边：索引几何候选与独立验收

2026-09-09。承接 [首次浮点分叉诊断](terrain_floating_point_diagnosis_CN.md)。

后续：本页保留独立候选阶段的记录。直线索引查询现已接入真实山体 CPU/Kokkos 输运，
并修正曲线路径中点审计，见[正式接线与小型完整运行验收](terrain_indexed_transport_acceptance_CN.md)。
非零曲率共享特征、严格完整轨迹与跨界射电仍有未完成项。

## 当前结论与范围

已实现**保留原始共享顶点坐标的直线求交候选**，在独立 CPU/CUDA 探针中，
消除了此前 21CMA DEM 顶点 31842 处的面编号和距离分歧。
候选尚未接入山体输运，也未改动大气应用、现有磁偏转求交、物理公式或随机流。
因此这不是“完整 shower 浮点差异已经解决”，也不是跨界射电生产验收。

实现与测试文件：

| 文件 | 职责 |
|---|---|
| `corsika/geometry/terrain/IndexedTerrainIntersection.hpp` | 索引顶点、直线求交、补偿边函数与确定性面归属；当前为待接入候选 |
| `validation/terrain/probe_indexed_boundary.cu` | 独立 host/device 比较、解析平面 oracle、真实 DEM 和遍历顺序检查 |
| `validation/terrain/run_indexed_boundary_acceptance.py` | 三种探针顺序运行、资源保护、二进制 SHA-256 与结果归档 |

探针以 nvcc 验证同一个共享头文件的设备求值，不初始化 Kokkos session；
不能将它称为完整 Kokkos-OpenMP/CUDA shower 测试。

## 1. 为什么不能只按面编号打破平局

旧表示保存 `origin + edge1 + edge2`。不同三角面用 `origin + edge` 重建同一个
共享顶点时，可能得到不同末位；仅增加 `distance == best_distance` 时比较面编号，
不能解决距离本来就不逐位相等的问题。原诊断还表明严格重心坐标判断也参与了分叉。

新表示保存统一的顶点数组，面保存原始三个顶点 ID 和 host 导出的法向。
求交时按顶点 ID 固定局部运算顺序，不改变原始面的有向法向。
将射线变换到以主方向为轴的二维投影平面，再计算有向边函数；该基本方法参考
[PBRT 的射线—三角形求交说明](https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes)。
实现采用显式 `fma` 和补偿运算，不通过全局关闭 FMA 改变其他物理代码。

初版候选的后端结果一致，但 10 个贴边输入中仍有 3 个落在错误的邻面：
`vertex - ray_origin` 的舍入先丢失了一个输入 ULP 的侧向差异。
因此又增加了坐标相减残差和补偿边函数，而不是放宽面内接受条件。
32 epsilon 的判断只决定是否执行补偿计算，**不是几何距离容差或边界扩张**。

## 2. 面归属与物理含义

- 面内交点：选择最近的正向命中。
- 可识别的共同边/顶点：保留共享拓扑标识；在满足出入方向条件的相邻面中，
  采用最小原始面编号，与遍历次序无关。
- 近平行包围盒查询：真正平行方向单独处理；不再用固定大数代替 `1/d`。
- 不移动粒子位置，不平均法向，不抽取随机数，不修改进入/离开材料的规则。

在折角或尖顶处，物理表面法向本来就不唯一。上述最小面编号是可重复的
**几何报告约定**，不是表面平滑、衍射或 Fresnel 模型。
未来射电模块不能把这一约定当作折角电磁边值问题已经解决。

## 3. 已运行的独立检查

在 `corsika_venv` 中分别编译 host、默认 CUDA（sm_89）和 `--fmad=false` CUDA。
没有使用 fast-math。比较距离的 binary64 位表示及面 ID、特征类型、特征顶点 ID；
不对含 padding 的整个 C++ struct 做 `memcmp`。

测试内容：

1. 100 万条解析平面射线：正反向、面内、准确共边和准确共顶点。
2. 20000 条斜射线，覆盖三个主投影轴；与独立 long-double 平面交点比较。
3. 10 个相差一个输入 ULP 的贴边点，检查是否出现裂缝或错侧。
4. 平行、零方向、负飞行距离、方向筛选以及近平行包围盒查询。
5. 真实 21CMA DEM：同一顶点 31842 的六条正反向/不同距离射线；
   32 个倾斜三角面的内部命中，与 long-double 平面 oracle 比较。
6. 原始/反向面遍历、局部顶点顺序变化，以及重新分组的有界 BVH 叶节点。

结果如下；百万点主要是解析平面，**不是百万个真实 DEM 位置或全部病态几何的证明**。

| 检查 | 结果 |
|---|---|
| 每种 CUDA 构建与本机构建的 host 查询比较 | 1020102 次 |
| 已知几何 oracle 检查 | 1020094 次 |
| 10 个贴边样本的错侧数 | 0 |
| 平面距离相对解析解的最大差异 | 2 ULP，绝对值 `2.84217e-14 m` |
| 32 个 DEM 面内点相对 long-double oracle 的最大差异 | `3.51283e-16 m` |
| DEM 顶点 31842 的归属 | 三种构建均为面 62909 |
| 0.01 m 正反向共享顶点查询 | 两端得到相同的 0.01 m 距离表示 |
| 三种构建的结果摘要 | FNV-1a `3640372f1a6dffd3` 相同 |

FNV 仅作便于比较的数值摘要；输入 DEM 与二进制另存 SHA-256。
这次通过只覆盖给定样本，补偿 double 不是任意精度的精确几何谓词。

同一独立探针还通过了 host AddressSanitizer/UndefinedBehaviorSanitizer
（包含 leak detection），错误输出为空。这不是整个 shower 程序的泄漏验收。
最终源码 SHA-256 保存在 `source_sha256.txt`。

## 4. 证据与生产保护

归档在源码同级的 `../build/mountain-validation-20260908/indexed_boundary_v2/`：

- `summary.json`、`{host,cuda,nofma}.json`：数值检查结果。
- `*_resources.json`：执行时间、RSS、设备总体显存增量、停止原因和二进制 SHA-256。
- `*.stderr`：编译完成后的运行错误输出。

每个探针最多运行 120 s，RSS 上限 512 MiB，系统可用内存保留 3 GiB。
额外显存保护为 `min(512 MiB, 设备总显存的 10%)`。
WSL 下记录的是**设备总体占用相对启动前的增量**，包含同期生产活动，
不能当作探针精确独占显存。探针本身只分配小型输入/结果和只读几何数组。
这些耗时含初始化和测试开销，不用于推断生产输运的性能改善。

本轮采样到的探针 RSS 峰值约 105 MiB，设备总体显存增量最大 98 MiB，
未触发任何资源门禁。三种普通构建分别约 0.75 s、3.64 s、3.33 s；
后两项含 CUDA context 初始化及重复的小批量上传，并非 GPU kernel 基准。

正在运行的 UHE 生产服务、归档可执行程序以及 install 没有被替换。
本轮前后 `applications/c8_air_shower.cpp` 的 SHA-256 均为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。

复现时先激活 `corsika_venv`，进入源码目录，并选择新的独立输出目录：

```bash
mkdir -p ../build/indexed-boundary-probe
c++ -O3 -std=c++17 -x c++ -I. validation/terrain/probe_indexed_boundary.cu \
  -o ../build/indexed-boundary-probe/probe_indexed_boundary_host
nvcc -O3 -std=c++17 -arch=sm_89 -I. validation/terrain/probe_indexed_boundary.cu \
  -o ../build/indexed-boundary-probe/probe_indexed_boundary_cuda
nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false -I. \
  validation/terrain/probe_indexed_boundary.cu \
  -o ../build/indexed-boundary-probe/probe_indexed_boundary_nofma
python validation/terrain/run_indexed_boundary_acceptance.py \
  --binary-dir ../build/indexed-boundary-probe \
  --mesh /path/to/validated/terrain_enu.ply \
  --output ../build/indexed-boundary-probe/results
```

该 PLY 读取器特意限定已验收的 21CMA fixture 布局，不是通用生产 PLY 导入器。
其他 NVIDIA 架构需要更改测试编译的 `-arch`；HIP/SYCL 不属于本次实机验证。

## 5. 接入前尚需完成

1. 将原始顶点/面拓扑纳入正式导出器与只读设备数据校验。
2. 对磁场中的二次曲线路径实现相容的面归属，验证双根、切向、薄三角形、
   大坐标和平滑/非光滑边界；不能仅替换直线查询后宣布完整几何已修复。
3. 独立测试构建接入 CPU/Kokkos 山体应用，重跑零场参考、岩气双向跨界、
   IGRF 场及强制 CC 全轨迹；比较第一次分叉和每个材料段。
4. 保留严格数值诊断，再分别设置几何、方向、时间、能量及射电相位误差预算。
5. 验收通过后才考虑正式替换；空气物理与生产程序仍不在本次改动范围。
