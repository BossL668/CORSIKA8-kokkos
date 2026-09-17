# 不同山体介质：100 PeV ντ shower 与射电复跑

2026-09-13 22:43（北京时间）已在 PSR 启动，共 9 例，顺序执行。
此文件是启动记录，不表示九例已经完成；实时状态以服务器 `progress.json` 为准。

| 山体介质 | 种子 | 密度 kg/m³ | 折射率 | 电场衰减长度 m |
|---|---|---:|---:|---:|
| 二氧化硅 SiO₂ | 158、946、3605 | 2650 | √5 | 100 |
| 方解石／石灰岩 CaCO₃ | 158、946、3605 | 2710 | √6 | 14.4765 |
| 花岗岩混合物：H、C、O、Na、Mg、Al、Si、K、Ca、Fe | 158、946、3605 | 2729 | √5 | 86.8589 |

材料按提供的 `material.md`、`materials.yaml` 封装。射电常数是研究模型参数，不是当地样品实测值。

每例沿用此前完成的 `openmp_seed158`、`openmp_seed946`、`openmp_seed3605` 初始条件：

- 入射粒子 ντ，能量 100 PeV；ENU 位置 `(2516.535431729868, 4294.03815963489, 500.52949201660203)` m，方向 `(-6.123233995736766e-17, -1, 0)`。
- 自然 CC+NC 相互作用、TAUOLA 自然衰变；不固定顶点或衰变通道。不同介质下相同种子可能产生不同末态，按实际输出分类。
- EM/hadron/muon 截断分别为 0.1/10/0.3 GeV，thinning 0.01，最大权重 50000，输运时间窗 100000 ns。
- 原地形、E01/E20/N10 三站、IGRF14/2027 大气磁场，山体磁场仍为零。射电采样率 0.256 GHz，131072 个样本，起始时间 −1000 ns。
- LPM 开启；每例 OpenMP 256 个物理核、常驻 EM 队列，同时计算 CoREAS 与 ZHS。原大气程序未改动。

第一例已实测线程绑定覆盖 CPU 0–255，对应 256 个物理核。总进程线程数为 259，包含运行库服务线程；不是 259 个 OpenMP 计算核。每例另核对最终输出中的 OpenMP 并发数为 256。

服务器工作目录：

```text
/data/yhlu/CorsikaData/corsika_validation_results/beta5_material_doublebang_openmp256_20260913
```

`campaign.json` 保存全部命令、材料卡、地形和程序散列。使用已通过介质验证的 OpenMP 程序快照，SHA-256 为 `320b84f6a200f5954de8a1f13b2c450280fb7483a7e22c433819073c83621ab5`。

`progress.json` 为实时状态；`driver.log` 为总日志。启动时驱动 PID 为 840578，独立核绑定记录进程 PID 为 841284；判断进程是否存活时应同时核对命令，不能只依赖旧 PID。

每例结果位于 `runs/openmp_<完整材料名>_seed<种子>/`：

- `output/terrain/tracks.csv`、`deposits.csv`：真实轨迹和加权能量沉积。
- `output/radio/CoREAS/`、`output/radio/ZHS/`：原始时域电场、复数频谱及配置。
- `checks.json`、`material_checks.json`、`affinity.json`：守恒、队列清空、射电来源完整性、算法一致性、材料/LPM 接线和实际核绑定检查。

每例完成并通过检查后，自动更新服务器 `report/SLIDES_CN.md`，生成英文标注的能量沉积图与 50–100 MHz 时域波形。能量沉积按入射轴投影、25 m 分箱；波形展示完整接收窗及最强脉冲放大图，不归一化、不平移时间。`ALL_NINE_PASSED` 只在全部九例通过后生成。

原始输运窗口、截断和 thinning 保持不变，本批不是这些参数的收敛扫描。三组种子不是材料统计样本；CoREAS/ZHS 重合也不能单独证明全部物理正确。

运行及分析脚本见 [run_psr.py](../../../validation/terrain/material_doublebang/run_psr.py) 与 [analyze_psr.py](../../../validation/terrain/material_doublebang/analyze_psr.py)。编译、测试、模拟与数值绘图均未在本地执行。
