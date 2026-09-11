# beta5 山体应用：带电轨迹与 PPT 讲稿

2026-09-09：新增四组 OpenMP-only 诊断，不使用 GPU，不修改空气应用或物理算法。

完整的 12 页讲稿、逐图解释及能力边界见
[PPT_TALK_CN.md](/mnt/d/CorsikaData/corsika_validation_results/beta5_terrain_charged_ppt_openmp_20260909_v1/PPT_TALK_CN.md)。
[图片目录](/mnt/d/CorsikaData/corsika_validation_results/beta5_terrain_charged_ppt_openmp_20260909_v1/figures)
包含 10 张 PNG 和对应 PDF：复用此前 6 张 ντ/地形图，新增 4 张带电输运图。

## 新增检查

- 100 GeV μ⁻：空气 → 岩石 → 空气的连续轨迹、能量和偏移。
- 1 GeV e⁻、e⁺ 在 IGRF14 空气中的真实轨迹，以及 e⁻ 零场对照；三图采用相同坐标范围。
- 9,033 个带电粒子步：独立重建局部 leapfrog 位置预测，最大残差 2.65×10⁻¹² m。
- μ⁻ 两次岩气交接：相邻记录的位置、时间、能量差为零，history 和权重一致。

每例约 70–72 s，峰值进程 RSS ≤921 MiB。Kokkos OpenMP 负责电磁组分；μ⁻ 及其余受支持的非电磁过程仍由 CPU 模型负责。
上述检查不能替代完整中微子再生、极化、全能区或射电验收；本批没有开启射电。

## 复现与原始证据

运行脚本：`validation/terrain/run_charged_ppt_demo.py`；绘图脚本：`validation/terrain/analyze_charged_ppt_demo.py`。
输出根目录的 `manifest.yaml`、`runs/*_command.json` 保存输入、命令和二进制哈希，
`charged_acceptance.json` 保存逐项验收；`plot_data/` 保存局部公式检查数据。
请使用独立输出目录，参考脚本 `--help` 和已保存命令；不覆盖生产数据。
