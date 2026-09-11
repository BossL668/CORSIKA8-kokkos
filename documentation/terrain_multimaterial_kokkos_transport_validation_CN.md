# beta5 真实 DEM 岩石—大气 Kokkos 输运

日期：2026-09-08。这是独立 `c8_terrain_cascade` 的增量实现，不修改或替换大气生产程序。

后续进展：空气 IGRF14/岩石零场与曲线边界已接入，详见
[带磁场应用验收](terrain_magnetic_application_validation_CN.md)。下文保留本阶段 B=0 数据和限制，
不是最新磁场功能状态；跨界射电仍未完成。

## 1. 本次接通什么

在原 mountain 迁移的 **21CMA 实际 DEM + 原生 USStdBK 五层大气**中，
光子、电子和正电子可以在 Kokkos 上传播、反应、跨越岩气界面，并在另一种介质中继续演化。
中微子、强子、μ/τ、衰变及指定的稀有过程仍交给原 CPU 模块。
本次是**有界、同步、功能验收用输运器**，不是已优化完成的常驻大规模山体生产后端。

**未完成：非零磁场下的曲线/DEM 求交、真实地形内外 CoREAS/ZHS 光路与波形，以及大样本物理验收。**
本应用显式 `B=0, radio=disabled`；不能把另一个有限凸体示例的均匀介质射电结果当作本应用的跨界射电。

```text
CPU：中微子 / 强子 / 衰变 / 不支持的轻子
                    │ 生成 γ、e−、e+
                    ▼
             有界 host 前沿队列
                    │ 每批最多 64（本次测试）
                    ▼
       Kokkos：同一 BVH 求最近逻辑材料界面
                    │
       material 0：干空气 native PROPOSAL
       material 1：SiO₂ native PROPOSAL
                    │
       共享 EM 输运、连续能损、散射和末态函数
          ┌─────────┼───────────────┐
          ▼         ▼               ▼
       跨界继续   存活/次级       指定 CPU 过程
       切材料表   回前沿队列      在实际顶点生成末态
          │         │               └─→ CPU 主栈再分流
          └─────────┴─→ cut / 世界逃逸 / 流式轨迹诊断
```

### 与原 mountain 的对应

- 使用已迁移的三角形网格、BVH、地理 ENU、EGM96/WGS84 高程转换与 `TerrainAtmosphere`，不是用凸体代替山谷。
- 64694 顶点、129384 面、73905 BVH 节点；几何数组 17668320 bytes。
- 场景原点正高约 2802 m；80 个站的地理位置及 `DEM + 1 m` 建模高度保留。
  该高度策略不是经过实测校准的天线海拔；本次不计算这些站的射电。
- 岩石 SiO₂、密度 2.65 g/cm³，位于原生最低大气层内；超出场景支持范围须先经过场景门禁。
- νe/反νe 仅 CC，适用能区沿用原模块；强制 CC 是给定顶点的条件级联，不是探测概率或事件率。
  不含 NC、核阴影或 τ 再生。CPU 强子使用 FLUKA + QGSJet-II.04，80 GeV/n 切换。

## 2. 接口和代码位置

| 位置 | 职责 |
|---|---|
| `corsika/modules/terrain/TerrainEmSession.hpp` | 纯 host/POD 配置、材料 bank、单步结果和 session 接口 |
| `src/terrain/TerrainEmSession.cpp` | 一次初始化 Kokkos、上传两套表及几何、具名 kernel、固定容量输入/输出、指定回退身份补齐 |
| `applications/detail/mountain/TerrainEmPreparation.hpp` | 从当前 CPU calculators 按真实介质 hash 分组、校验、导出原生表和辅助缓存 |
| `applications/detail/mountain/TerrainEmRouter.hpp` | CPU 栈与有界 EM 队列之间分流；主机统一回收，不并发修改 CPU Stack |
| `applications/detail/mountain/TerrainShowerOutput.hpp` | CPU/设备轨迹流式写入；介质、身份、跨界次数与能量沉积审计 |
| `corsika/accelerator/em/common/ExternalTransportBoundary.hpp` | 默认关闭的非吸收材料步长限制；保留旧大气调用的默认行为 |
| `corsika/accelerator/em/detail/{Photon,Lepton}TransportStep.hpp` | 现有共享输运增加可选材料候选，与反应/连续步长/层边界竞争 |
| `applications/c8_terrain_cascade.cpp` | 原 CPU 模型序列、CLI、CPU/Kokkos 分支与输出生命周期 |

不复制另一套电磁物理公式。新 kernel 调用已有的 interaction selection、photon/lepton
transport、Molière、vertex selection、final-state classification/materialization 和 thinning。
每个输入最多三个固定次级槽，history ID 由主机按批次预留；不依赖线程完成顺序或原子追加次序。

空气和岩石是**两套**表，分别保留版本、cut、能区、组分及 SHA-256 校验；
没有删除原生表的单组分定义门禁，也不把岩石的反应率用于空气。
全 host 导出先通过包含 μ 的既有门禁，设备只上传本例路由的 γ/e±，其余仍走 CPU。
辅助缓存自动管理，不要求手动制作 `.c8emrt`。

## 3. 本次发现并修复的问题

1. **材料界面不能算世界逃逸。** 新 `MaterialBoundary` 限制只结束当前步，
   保留位置、时间、能量、权重、history/parent/generation，仅切换材料并增加 step counter。
   与原 CPU 同样采用逻辑内/外侧求交；求交辅助位移不是人为搬动粒子。
2. **最低大气层不是海平面吸收壳。** 原 CPU 最低层为球体。
   从空气向内的极远几何候选可能使正柱深溢出；只在新地形路径中将该候选视为正无穷进行竞争，
   实际选中的有限步仍按原积分计算并强制检查有限 distance/grammage。
   没有给 shower 加一个不存在的海平面 cut。地心 1 mm 区域不在本诊断快照支持范围内。
3. **稀有光子过程必须在反应顶点回退。** native selected-loss 回退可能在传播前提出。
   光子飞行不需要先确定损失分位数，因此先用原候选距离与几何竞争；
   仅当反应胜出后，保留原随机数/过程/组分记录并把回退位置更新到顶点。
   空气/岩石的 CPU calculator 身份分别从对应 bank 补齐。
   未识别的数值错误或不具备合法顶点的回退直接失败，不偷偷重抽一次反应。
4. **有限资源。** 只读数据和固定工作区先估算、再分配；超预算在 runtime 初始化前拒绝。
   全 shower 轨迹写流式 CSV，不留在 RAM；host 前沿最多 200000 粒子，超限输出不完整并终止。
   `--track-row-limit` 只影响磁盘记录行数，默认 200000；完整强制 CC 轨迹测试使用 300000。

进程 registry 明确声明各模块由设备替代、由 CPU 处理或由设备记录重放。
它不允许将未知观察/修改 EM step 的过程静默跳过。

## 4. 构建、使用与资源限制

在项目容器目录激活 `corsika_venv`。依赖/独立 CMake 配置沿用
[真实场景构建说明](terrain_native_scene_build_CN.md)，两个测试 build 不安装覆盖生产目录。

```bash
cmake --build build/mountain-openmp --target c8_terrain_cascade --parallel 1
cmake --build build/mountain-cuda --target c8_terrain_cascade --parallel 1

# 同一个应用源码：OpenMP/CUDA 使用各自构建的二进制。
# 将 scene 路径替换为已验证的场景；输出目录必须不存在。
build/mountain-cuda/applications/c8_terrain_cascade \
  --scene build/mountain-validation-20260908/21cma_scene_v2.yaml \
  --output build/terrain-photon-example \
  --position-m 0 0 -0.01 --direction 0 0 1 \
  --em-backend kokkos --device-memory-MiB 128
```

默认初级为 1 GeV 光子，默认 EM cut 为 0.5 MeV。用 OpenMP 时替换二进制目录并加
`--threads 2`；CPU 参考去掉 `--em-backend kokkos`。位置为场景 ENU，不能照搬到另一个 DEM。
`--force-vertex-cc --primary nu_e --energy-GeV 10000` 开启限定范围的强制 CC 测试。

带监控的测试从**源码目录**运行：

```bash
python validation/terrain/run_multimaterial_acceptance.py \
  --binary ../build/mountain-cuda/applications/c8_terrain_cascade \
  --scene ../build/mountain-validation-20260908/21cma_scene_v2.yaml \
  --output-root ../build/terrain-small-cuda-new \
  --aux-cache ../build/mountain-validation-20260908/aux-cache \
  --mode cuda --cases photon_up photon_down electron_rock positron_rock
```

本次 RTX 4060 Laptop 为 8188 MiB；10% 为 818.8 MiB。
应用所计固定分配为 **97094112 bytes（92.60 MiB）**，限制 128 MiB。
监控器每约 0.25 秒查询，另以 `min(512 MiB, 总显存×10%)` 的设备总占用增量守护、
2 GiB child RSS 守护及 240 s 超时，仅停止自己创建的测试进程组。
WSL 不提供有效的逐进程显存数字，因此增量含其他任务的变化；它不是精确的本进程 NVML 读数，
而分配账本不含 CUDA context/module 开销。两种数字分开记录，不能相互替代。
这轮小测试不进行占满显卡的性能调优。

## 5. 实测结果

两套 Release 的 `c8_terrain_cascade` 编译/链接通过，FLUKA 和 QGSJet 均启用。
相关 OpenMP CTest **8/8 通过**：地形 session、凸体输运、均匀射电、CPU 能量沉积对齐、
CPU 输运对齐、中微子、山体射电适配器、地形环境。
其中射电单测是既有模块回归，**不是本 DEM 应用的射电验收**。

seed=67101，界面附近 ENU `(0,0,±0.01 m)`，OpenMP 两线程、CUDA 一设备，batch=64：

| 事例 | OpenMP 步数 | CUDA 步数 | 完整性/材料检查 |
|---|---:|---:|---|
| 1 GeV 光子，出山 | 1489 | 1489 | 通过，1 次岩→气 |
| 1 GeV 光子，入山 | 2109 | 2109 | 通过 |
| 1 GeV 电子，出山 | 1627 | 1627 | 通过 |
| 1 GeV 正电子，出山 | 1671 | 1671 | 通过 |
| 10 TeV νe 自然穿越 | 6 | 6 | 通过，CC=0，无 EM 是合理结果 |
| 10 TeV νe 强制 CC，emthin=0.1 | 225606 | 225606 | 完整 CSV，通过；194841 加速步、27 次指定 CPU 回退 |

这些成功输出均 `complete=true`、`pending_particles=0`、材料错配=0。
强制 CC 有 16 次岩→气、557 次世界逃逸；光子低能自然测试 emthin=1e-6。
额外 emthin=0.5 的强制 CC 两端各完成 220508 步、56 次岩→气、1 次气→岩、20 次回退，
但其早期 CSV 在 20 万行截断；它只算运行/计数检查，不算完整逐行验收。
随后保留 emthin=0.1、仅将流式记录上限调到 30 万，补齐了上表完整轨迹。

### 跨执行空间差异，不能隐去

六组的 step/PID/medium 序列一致。四个 1 GeV EM 测试的权重也逐行相同，
最大坐标差 `9.20e-9 m`，最大能量差 `1.59e-14 GeV`。
但设置 `rtol=1e-10, atol=1e-12` 的逐字段紧比较**未全部通过**，
且沉积总和不是逐位相同；比较脚本故意返回非零状态保留这个事实。

强制 CC 长轨迹的最大坐标差为 **0.00760 m**、最大能量差 `5.77e-10 GeV`、
最大时间差 `1.86e-12 s`、最大方向分量差 `2.55e-7`。
最大位置差出现在第 214341 行、约 110 km 高度处的低能空气光子长轨迹；
最大权重相对差为 `1.38e-8`。尚未以逐过程 oracle 完全拆解这些累积误差，
因此不能把“完整运行、材料正确、离散轨迹序列一致”提升为严格 shower/radio 等价认证。

原标量 CPU 两个光子参考的 **tracks.csv 与迁移前逐字节一致**。
标量强制 CC 完整记录为 170382 步、材料错配=0；加权沉积约 4793.51 GeV，
而 Kokkos 同 seed 约 8363.52 GeV。CPU/Kokkos 的 RNG/调度不同，且该 CC 测试使用很强的 thinning；
**单个条件事例不能区分统计涨落与系统偏差**，不据此声称二者物理一致，也不能仅凭此判定不一致。
这些数字都保留，后续须降低 thinning/做系综及逐过程对照。

### 资源和失败门

- 已完成 CUDA 小测试所观测显存增量峰值 **426 MiB（约总显存的 5.2%）**；
  完整 CC 子进程峰值 RSS 741320 KiB，未触发守护。测量局限见上一节。
- 32 MiB 分配限制被拒绝：`terrain pre-initialization memory gate requires 97095048 bytes`，
  `complete=false`，没有静默走 CPU 完成。
- 缺失逻辑 DEM 出口、非有限状态、未知回退、超前沿/步数预算都显式失败。
- 开发中的首次预算超限、最低层柱深溢出、回退身份不全失败目录保留；不算成功样本。
- 不进行性能结论：小测试与现有生产 GPU 任务同机，初始化约占数十秒；
  `shower_seconds` 的 Kokkos 分支还含表导出/session 初始化，与 CPU 分支不是纯 kernel 对比。

关键证据（相对项目容器目录）：

```text
build/mountain-validation-20260908/
  multimaterial_{openmp,cuda}_release_build.log
  multimaterial_regression_ctest_final.log
  multimaterial_cpu_fulltrace_v1/
  multimaterial_openmp_fulltrace_v1/
  multimaterial_cuda_fulltrace_v1/acceptance.json
  multimaterial_openmp_v3/                 # 四类 EM + 自然中微子成功；旧 forced 失败
  multimaterial_cuda_cases_v2/             # 同上，旧 forced 失败
  multimaterial_trace_comparison_final_v2.json
  multimaterial_cpu_openmp_integrity_v2.yaml
  multimaterial_budget_rejection_v1/
```

本次未修改 `applications/c8_air_shower.cpp`，其 SHA-256 仍为
`d97913ab311821b6c9f5aa8fe24b0c27d85802a589387b136141fd92c5299d2e`。
生产 service、冻结二进制及 install 不替换。本次独立应用二进制：
OpenMP `f2b6f444ae80b770ff22d643187b8a97321f767a3d961b0d3a301fae260758eb`；
CUDA `05fd2fd91ad1a9815d7e39dcba4ea9996ba9e29e351ae701976373874752065e`。

## 6. 验收边界与后续

验收记录保存在项目容器 `build/mountain-validation-20260908/`；失败的开发中间输出保留，
不得计入通过样本。逐轨迹完整性工具拒绝 `complete=false` 或 `csv_truncated=true`。
比较工具分别报告第一次浮点差异和第一次离散字段差异；没有通过归一化或时间平移掩盖差异。

完整 CPU/Kokkos 系综物理等价、强子静质量账本闭合、所有能区和介质、HIP/SYCL 硬件运行、
整 shower 峰值内存及长期性能仍需后续验收。单例步数不同也不能直接解释为物理偏差。

下一步应继续独立迁移原 mountain 的**带范围约束的地形射电传播接口**：
源/目的介质、光程、出射/接收方向、偏振与 Fresnel 传递、遮挡、路径有效性和时间窗显式传递。
先验平面解析解及同轨迹 CPU/Kokkos，再验真实 DEM；不以均匀 `nR/c` 代替岩气透射。
