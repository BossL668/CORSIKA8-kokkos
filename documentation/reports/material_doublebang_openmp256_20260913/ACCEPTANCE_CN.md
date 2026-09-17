# 100 PeV ντ 不同介质计算：验收记录

验收时间：2026-09-14 09:35:03（北京时间）。**9 / 9 组完成并通过本次验收。**

每组保留此前的入射条件与随机种子，使用 OpenMP 256 个物理核、常驻队列及 CoREAS/ZHS 双算法。

本次直接读取原始 CSV 复核，未重跑 shower：

- 初始物理参数、程序、运行库和材料输入散列一致；组分、密度、折射率和衰减长度符合提供的材料文件。
- 实际 256 核绑定、OpenMP 并发数、LPM 开关、材料输运与射电接线通过；队列最终清空，无轨迹或沉积输出截断。
- 能量账本最大相对闭合残差：2.15e-12。该值已计入账本中的生成器交换项、静质量处理和 thinning 权重跳变，不等同于全物理模型精度。
- 每事件 CoREAS/ZHS 全频复数谱相对 L2 差异范围：1.81e-09–6.25e-09；逐站最大 7.15e-09。
- 原始时域电场逐站最大相对 L2 差异：7.15e-09；全部采样时刻和 50–100 MHz 峰值与报告一致。
- 射电计算无错误、无接收窗外丢弃，两种算法共用完整带电轨迹来源；累计网格在最终阶段统一导出。

| 材料 | seed | 输运步数 | 计算时间/min | 本次 τ 衰变 |
|---|---:|---:|---:|---|
| Silica (SiO2) | 158 | 8141992 | 33.3 | muonic / air |
| Silica (SiO2) | 946 | 10445731 | 44.2 | electronic / air |
| Silica (SiO2) | 3605 | 8743839 | 40.3 | hadronic / air |
| Calcite / limestone (CaCO3) | 158 | 9833161 | 37.7 | muonic / air |
| Calcite / limestone (CaCO3) | 946 | 11211772 | 46.1 | electronic / air |
| Calcite / limestone (CaCO3) | 3605 | 45020600 | 146.4 | hadronic / air |
| Granite (H, C, O, Na, Mg, Al, Si, K, Ca, Fe) | 158 | 9473099 | 38.1 | muonic / air |
| Granite (H, C, O, Na, Mg, Al, Si, K, Ca, Fe) | 946 | 10912682 | 46.4 | electronic / rock |
| Granite (H, C, O, Na, Mg, Al, Si, K, Ca, Fe) | 3605 | 56822001 | 192.1 | hadronic / air |

[查看 shower 与射电诊断图](SLIDES_CN.md)。图片均为英文标注；完整数值在 [acceptance.json](acceptance.json)。

图中的材料差异同时包含 shower 随机发展、τ 衰变位置和传播参数变化，不能只解释为衰减长度变化。保留了原来的有限输运窗、截断和 thinning，本次通过数值与调度验收，尚不是这些近似的收敛结论或完整物理的独立验证。

旧绘图日志中的 NumExpr 线程上限提示已在分析脚本中修正；本次图表重新生成，原始数据不受改动。模拟及数值验收均在 PSR 完成。
