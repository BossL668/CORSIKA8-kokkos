**DEM 覆盖边界与 21CMA 80 站验收（2026-09-14）**

读图报告和逐站坐标表已放在 D 盘：

[边界、80 站位置/高度、profile 与射电图解](/mnt/d/CorsikaData/beta5_doublebang_figures_20260914/09_dem_boundary_80stations_20260914/README_CN.md)

[80 站经纬度、测量高程、模型高程与 DEM 净空](/mnt/d/CorsikaData/beta5_doublebang_figures_20260914/09_dem_boundary_80stations_20260914/stations_80_audit.csv)

**配置**

原生山体场景新增：

```yaml
geometry:
  transport_boundary:
    type: dem_coverage
```

`none` 或原生旧场景未指定时保留旧输运范围。新的 Python 地理产品生成器以及 `scene_from_product` 导出默认添加 `dem_coverage`；显式 `none` 保留。站点从产品 `observers.yaml` 全部读取，本轮真实 DEM 使用 E/W/N/S 各 20 站。历史三站场景不会被自动改写成 80 站。

边界由 DEM 顶面三角网格投影的外轮廓生成，沿 ENU Up 延伸。当前支持单一简单轮廓，含洞或不连通网格会拒绝；不是用 mesh 三维包围盒截断高空大气。没有新增顶部或底部输运截断。现有网格底部的输运几何处理仍沿用原行为。

**实现与记录**

`DemCoverage.hpp` 提供 POD 查询及无递归 BVH，`DemCoverageData.hpp` 提取与验证轮廓，`KokkosDemCoverage.hpp` 将不可变查询数据常驻设备。CPU 用与原磁场追踪一致的二次轨迹表达式求交；界面 EM 内核将侧边界与材料界面、物理过程、步长和时间窗共同竞争。同位置侧边界优先归为计算域退出，低于粒子截断的情况仍由原 ParticleCut 处理。

CPU 输出和 Kokkos resident/batched 调度都接入 `DomainEscape`。退出前物理步的轨迹、沉积和带电射电源段保留；末态不重新入队、不发生虚假的侧壁介质翻转。`domain_exits.csv[.gz]` 不受诊断行数上限限制，记录粒子 PDG、权重、介质、时间、坐标、方向、总能量、父子谱系及边编号。`domain_escaped_total_GeV` 纳入总能量账本，独立于沉积、原有世界边界逃逸与有限时间窗幸存者。

开启边界时，轨迹或沉积行数预算不足会明确失败并标记不完整；增加 `--track-row-limit` 后须重新完整运行。这里保证的是计算域内 profile 的完整记录，域外尾部被有意停止，不计作沉积。

射电的直接/透射传播段也检查 DEM 覆盖。人为闭合侧面与底面不作为光学材料界面；地形真实顶面仍使用原跨介质传播算法。仅累计域内物理轨迹段，没有额外添加“计算边界停止脉冲”；缺少域外尾部本身仍可能改变波形，需要扩大地形范围检查收敛。

**PSR 验收**

运行目录：`/data/yhlu/CorsikaData/corsika_validation_results/beta5_dem_boundary_20260914`。在该目录保留独立 `source`、`build-openmp`、`build-cuda`、`runs`、`stations` 和 `report`，未替换原已验收构建。OpenMP 用 `taskset -c 0-255` 与 256 线程；本地仅编辑、读文件和整理结果。

| 检查 | 结果 |
|---|---|
| 几何独立解析对照 | 每后端 100005 条，含直线、磁弯曲、角点、切线、非凸轮廓及高空路径；最大距离差 2.26×10⁻¹³ m |
| 混合常驻队列 | 8 个输入，4 个退出、4 个保留，身份与加权能量检查通过 |
| 完整控制算例 | 32 个：OpenMP 21、CUDA 11；包含早期诊断站/三站控制，实际 80 站射电为另行明确命名的四个算例 |
| 能量账本 | 最大未解释能量/初始加权能量 1.82×10⁻¹⁶；包含相互作用交换、thinning、原有截断和逃逸项 |
| 内部 shower 开关边界 | 同后端完整 tracks/deposits/window CSV 逐字节相同，含 Kokkos OpenMP、CUDA 和标量 CPU |
| 原已验收程序回放 | 关闭边界的三份 CSV 一致；最终行数保护版本的跨界四份 CSV 与保护前一致 |
| 不完整输出保护 | 故意将行数预算设为 1，两个后端均明确失败；外部注入也被拒绝 |
| 真实 DEM 空气源、全部 80 站 | 两后端全部站有有限非零输出，无时间窗外丢失；CoREAS/ZHS 相对 L2 约 2.68×10⁻⁸ |
| 真实 DEM 岩石边缘、全部 80 站 | 全部参与配对，但无接受的顶面透射路径；零场不作为振幅正确性的证据 |
| 非零岩石控制 | 平顶 SiO₂、20 MeV 电子、专用诊断接收点，两算法相对 L2 1.64×10⁻¹⁵ |

EM 截断为 0.0005 GeV，强子/μ/τ 为 0.3 GeV，emthin=1e-5。真实 80 站测试采样率 256 MHz，OpenMP 32768 点、CUDA 16384 点；短轨迹测试的到达时间均在窗内，CUDA 总设备预算实测约 2.48 GiB。不能用这些小规模结果替代高能 double-bang 的完成验收或性能估算。

**站点核对结论**

80 个名称完整唯一、均在 DEM 范围内。绝对 ECEF → 经纬度 → 模型高度下 ECEF → 统一 ENU 的独立核对最大数值差 4.66×10⁻⁹ m，这不是测量精度。站高全部满足 DEM+1 m；相对原始测点/均值高程差 −1.00～+14.09 m，77 个原始测点按现有 DEM 会落在地下。不能把模型高度称为实测天线相位中心。

旧合并相对表的 EW 与 NS 使用不同的 E01 参考定义，EW 还在各站纬经度处旋转。两种 E01 在 DEM 平面相差约 16.7 m。现有山体从原始绝对坐标统一转换，避开了把混合相对表当作同一 ENU 的错误；仍需保留测点定义和 DEM/测量高度差异，不应擅自平移或降低天线来强行匹配。

复现脚本位于 `validation/terrain/dem_boundary/`。构建使用 `build_psr.py` 和 `environment.sh`；输运验收用 `check_geometry.cpp`、`run_psr.py`、`extra_checks_psr.py`；逐站核对与绘图用 `audit_stations_psr.py`、`report_psr.py`。脚本限制在 PSR 运行。原始材料验收环境的 `python-venv/bin/python` 用于包含 pyproj 的坐标核对与绘图。散列和构建来源见服务器 `report/source_manifest.json`。
