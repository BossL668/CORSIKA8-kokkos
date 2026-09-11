# beta5 穿山 ντ：OpenMP 展示事例（2026-09-09）

本轮只添加运行/分析脚本，不修改物理实现或大气应用。使用已验收的
`build/mountain-openmp/applications/c8_terrain_cascade`，4 个 OpenMP 线程；
**不使用 GPU**，现有空气 shower 生产服务继续运行。

## 图和报告在哪里

Windows 目录：

```text
D:\CorsikaData\corsika_validation_results\beta5_terrain_nutau_ppt_openmp_20260909_v1
```

[完整图文说明与复现命令](/mnt/d/CorsikaData/corsika_validation_results/beta5_terrain_nutau_ppt_openmp_20260909_v1/PPT_REPORT_CN.md)

`figures/` 中有六组 PNG + PDF：

1. `01_terrain_and_array`：21CMA 地形、80 个站中心、局部岩石与真实次级轨迹。
2. `02_shower_tracks_and_tau_zoom`：shower 三维/侧视图及真实毫米尺度 τ 轨迹。
3. `03_CC_tau_decay_chain`：地形剖面、CC→τ→次级粒子、自然衰变时间与能量。
4. `04_longitudinal_all_components`：电子、光子、μ、τ、强子、中微子的单事件纵向发展。
5. `05_energy_deposition_and_survivors`：沉积 profile、沉积空间图、窗口末端存活能量。
6. `06_boundary_and_runtime_diagnostics`：自然穿透对照、材质计数、内存和用时。

## 运行内容与结果

沿用原 mountain `terrain_nutau_v1` 的真实 DEM、ENU 原点、标准大气和天线坐标。
射电关闭。SiO₂ 密度 2.65 g/cm³，穿山轴岩石弦长 135.566697 m；
空气使用 IGRF14/2027，岩石内部磁场为零。

| 事例 | 初级 | seed | 轨迹步数 | 全进程 / shower 时间 |
|---|---|---:|---:|---:|
| 条件化首个 CC | 10 TeV ντ | 909350 | 93,571 | 82.30 / 15.84 s |
| 条件化首个 NC | 10 TeV ντ | 909351 | 106,109 | 82.38 / 17.74 s |
| 自然穿透对照 | 10 TeV ντ | 79101 | 3 | 69.01 / 2.14 s |

EM cut 10 MeV、强子/μ/τ kinetic cut 0.3 GeV，EM thinning 10⁻³、max-weight 50，
诊断最大步长 1 m、物理时间窗口 1,100 ns。是快速展示配置，未做参数收敛认证。

三例均正常结束、材质错误为零、轨迹和沉积 CSV 无截断。核对了实际 OpenMP
执行空间、有限值、计数与能量记录、顶点四动量门禁、τ 末态父子关系与轨迹连续性。
CC 例 τ 真实飞行 0.363510 m，在岩石内自然衰变为 ντ、π⁻、π⁻、π⁺；
其衰变点与末轨迹端点差约 5.7×10⁻¹⁴ m。最大进程 RSS 919.91 MiB。

## 展示范围，不能夸大的内容

- CC/NC 顶点在岩石中指定位置条件化；入射轴虚线不是未记录的实际粒子段。
  不提供天然事件率权重。自然穿透对照则没有强制相互作用。
- τ 衰变不强制，不筛选衰变道；τ 没有飞出山体，不是已观察到的 emerging-τ 事件。
- 所有数字诊断使用全部记录；绘制密集轨迹时仅显示子集。
  纵向曲线为端点重建的无符号平面穿越数，不是生产计数，也不是系综均值。
- CC、NC 各有 3、4 条低于 CTW 能区的中微子 history。
  它们继续传播但未包含已验证的低能弱再相互作用，不能称为完整再生链验收。
- TAUOLA 使用原版固定 helicity 约定，不代表完整 CC 自旋密度矩阵验证。
- 沉积与窗口存活能量是分开的部分台账，不以二者之和宣称完整强子能量闭合。
- 六张图不包含射电计算或探测概率结论。

## 脚本

- `validation/terrain/run_nutau_ppt_demo.py`：预先固定三例参数和 seed，检查 DEM 哈希，
  限定 OpenMP-only 构建，复用已存在的 PROPOSAL-native 辅助缓存，逐例带资源门禁运行。
- `validation/terrain/analyze_nutau_ppt_demo.py`：检查输出、重建诊断量、生成图文报告；
  复用 beta5 地形网格对角线约定，不另建 Delaunay 剖面。
- `runs/` 保留命令、二进制哈希、CSV、YAML 和资源日志；`plot_data/` 保留绘图数值。

用户改为 OpenMP 前启动的诊断 CUDA 进程已终止，其未完成输出保留在
`beta5_terrain_nutau_ppt_20260909_v1/` 并标记 `STOPPED_CN.md`，**未混入这些图**。
