# 12 张图：真实光路、s/p、射线束扩散、吸收与波形

直接看 [PPT Markdown 与读图说明](FIGURES_CN.md)，或 [PDF](FIGURES_CN.pdf)、[PPTX](FIGURES_CN.pptx)。正文共 12 页，每页一张图和简短说明；顺序为真实光路（1–3）→ s/p（4–6）→ ray tube 定义与误差（7–8）→ 吸收核对（9）→ 材料衰减（10）→ 波形与双算法对比（11–12）。另附 [s/p 与后续传播因素简短复习](SP_NOTES_CN.md) 和 [Ray tube 定义与误差图详解](RAYTUBE_NOTES_CN.md)。图内全部使用英文，说明保留中文；图片文件名沿用历史编号，不等于新版页码。

第 1–6、11–12 页沿用已完成的 **1 GeV 电子、seed 67101、CUDA 常驻双算、旧岩石 n=2** 事例；Fresnel 扫描是该界面参数下的受控检查。第 7 页新增射线束教学示意；第 8–9 页来自此前射电物理报告的独立传播探针，第 10 页来自材料接口报告。2026-09-15 本次在 PSR 绘制定义示意、复核已保存的 32 个 Jacobian 对照样本并导出，没有重跑簇射。所有原始计算、绘图和本次导出均在 PSR；生产程序、原空气模块及公开库源码未改动。

## 所选光路确实来自哪里

- 空气光路：`electron_rock_resident/terrain/tracks.csv` 的 **step 746、history 2001**，电子轨迹中点 **(21.680794, 10.165493, 988.897241) m**，到原配置的 **E01 (397.881406, −2028.754384, −145.700486) m**。
- 岩石光路：同一事例 **step 1、history 1** 的中点 **(0, 0, −0.008555120) m**，到原配置的 **diagnostic_offset (100, 0, 1000) m**。选择实际有效分支中的 **face 63449**；该源点还存在另一个相邻面的有效分支，图只显示选定这一条。
- 原模拟没有保存逐光路 GPU 调试轨迹。本次读取真实轨迹、实际观测点、原始 DEM 和原射电折射率表，直接调用生产 `directPath / transmittedPath` **重放光路查询**。两个轨迹中点都满足原射电细分条件，是未再细分的叶段；没有人为编造或加深源点。
- 岩石源点到 E01 未获得有效透射分支，因此没有画成有效连线。全部候选与已接受路径分别保存在 `input_provenance.json` 和 `beta5_real_paths.csv`。

山体轮廓由原始三角网格与竖直剖面求交得到，没有平滑拟合。高度坐标是相对场景原点的 ENU 高度，原点海拔约 2802 m。第 2 图左侧是竖直剖面的投影，出射点离此剖面约 0.414 mm；右侧使用真正的光学入射平面和山面法向坐标。毫米放大展示的是模拟网格上的数学光路，不表示实际 DEM 或岩石界面具有毫米精度。

## 外部比较的含义

[RadioPropa](https://github.com/nu-radio/RadioPropa/tree/544a2d6c4e284e3d724cb741dc481245a0f633d7) 使用固定未修改提交 `544a2d6c4e284e3d724cb741dc481245a0f633d7`，沿用前一份报告在 PSR 编译的动态库：

1. **折射方向检查**：在实际出射点、实际法线和界面折射率上调用公开 `Discontinuity`；恒定介质控制下，两条光路的接收屏落点差均小于 `1.2e-11 m`。
2. **实际大气中的弯曲路径**：将原配置的球对称 n(h) 表及其分段线性梯度交给公开 `PropagationCK`。调整发射方向，使 RadioPropa 也到达相同观测点，然后比较路径。
3. **独立核对**：另用 [SciPy solve_ivp 的 DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) 独立积分几何光学方程。SciPy 是通用数值积分库；方程和介质输入由本报告提供，不称作另一个完整射电模拟器。RadioPropa 与该参考在采样点上的最大位置差分别约 **0.293 μm / 0.032 μm**。
4. **重新检查实际山体**：RadioPropa 和独立参考生成的两条路径均重新通过原 DEM 的分段可见性检查；岩石出射点仍位于指定三角面内。见 `terrain_path_checks.csv`。

主图固定两个端点。另存的 `native_forward` 则保持 beta5 初始方向不变，其 E01 接收屏偏离约 **58.7 mm**，与主图的 **15 mm 路径最大偏离**是不同量，不能混为一谈。

公开库的实际大气传播存在一个需要避开的数值限制：此固定版本 [ParticleState::setDirection](https://github.com/nu-radio/RadioPropa/blob/544a2d6c4e284e3d724cb741dc481245a0f633d7/radiopropa/src/ParticleState.cpp) 在 `acos` 得到零夹角时跳过方向更新。弱大气梯度与很小步长组合会丢掉微小转角；本次 0.5 m / 0.125 m 尝试出现这种现象，原始失败结果保存在 `radiopropa_small_step_diagnostic.csv`。最终采用 40 / 20 / 10 m 固定步长检查，展示 10 m 结果，并与连续 ODE 参考核对。没有修改公开库来使结果吻合。

这次比较的是两条真实光路及局部折射，不是完整簇射电场的外部验证。两算法波形继续使用原 beta5 直线光学近似。原始波形未做时间平移、振幅拟合或显示带通；所选射线只对应一个轨迹贡献，不等于整个脉冲。

## 产物

- `figures/`：12 张 PNG；除材料图外均有矢量 PDF。`12_raytube_concept` 用于第 7 页，`09_raytube`、`10_attenuation`、`11_material_attenuation` 用于第 8–10 页。
- `data/propagation_figure_provenance.json`：新增图的原始位置、SHA-256 与核对数据。射线束检查固定源距界面 100 m、接收点高度 300 m，取两组折射率和四个入射角；其误差指 Jacobian，不是电场吸收率。
- 第 9 页为 n=1.33、L_E=100 m 的均匀介质控制，不代表山体折射率；第 10 页为材料卡中的电场衰减长度，SiO₂ 100 m、CaCO₃ 14.4765 m、图示十元素花岗岩混合物 86.8589 m，不是现场测量曲线。
- `data/sp_metrics.json`、`data/sp_selected.csv`、`data/sp_fresnel_scan.csv`：s/p 的实际 beta5 函数调用与 TMM 检查；归一化说明见 SP_NOTES_CN.md。
- `data/selected_ray.json`：具体轨迹、源点、出射点、法线、观测点与传播时间。
- `data/input_provenance.json`：真实输入位置及 SHA-256。
- `data/metrics.json`：光路偏差、外部核对和 CoREAS/ZHS 波形误差。
- `data/*native_shoot.csv`、`data/*scipy_reference.csv`：实际计算出的三维路径。
- `data/terrain_section_*.csv`：实际山体剖面。
- `data/E01_waveforms.npz`：该事例 E01 的两算法时域电场和复频谱。
- `tools/`：全部重放、外部比较、绘图及导出脚本。

新增示意和误差读数记录见 `data/raytube_explanation_metrics.json`；脚本 `tools/make_raytube_concept.py` 只绘制教学示意并读取已有 CSV。报告更新使用低优先级、CPU 510–511、数值库单线程执行，不重启在跑的任务。

## 在 PSR 复现

原始目录仍在 PSR，交付包不包含完整 DEM、全部簇射输出或第三方二进制依赖。PSR 工作目录：

```text
/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/psr-interface-resident-20260911/interface-radio-figures-20260912
```

在上述目录执行：

```bash
bash tools/run_psr.sh
bash tools/render_psr.sh
```

使用 PSR 的 `corsika_venv`，CPU 亲和范围 0–255、OpenMP 线程数 256，关闭 Chromium GPU 加速。几何与光学探针自身是小型串行程序，256 核亲和范围不表示每个探针都会占满全部核心。

真实光路、s/p 和波形页的原始簇射输入来自同一相邻目录 `interface-radio-pair-20260912/app-cuda-p1/electron_rock_resident`。第三方库与 Marp 位于相邻的 `interface-radio-report-20260912`；版本与获取方式见前一报告的 README。初次复制到不同目录复现时，应保留这些相邻目录关系。只更新 PPT 排版时仅执行 `tools/render_psr.sh`，不需要执行光路/簇射计算。
