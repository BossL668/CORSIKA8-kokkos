**21CMA 80 站 double-bang 重跑（2026-09-14）**

PSR 新批次：`/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_21CMA80_dem_openmp256_20260914`。

[D 盘自动同步的状态与图件](/mnt/d/CorsikaData/beta5_doublebang_figures_20260914/10_doublebang_21CMA80_dem_openmp256_20260914/README_CN.md)

以已通过 DEM 边界验收的 OpenMP 二进制（SHA256 `80dcb1a7ac8d2bb0c8febd03d101ebb96319c8d870c44e3a97fc4234758572d3`）为基准。首次启动在参数检查阶段发现旧命令行只允许 16 GiB 预算，尚未输运即退出。随后仅将山体入口的 `--device-memory-MiB` 上限提高至 128 GiB，重新链接入口程序，输运和射电物理库逐字节保持。新二进制及证明见 `campaign.json` 的 `memory_budget_admission`。

`memory_budget_psr.py` 使用新二进制及 96 GiB 预算重跑真实 DEM 的短窗 80 站算例，比较完整的四份输运 CSV 和两算法全部电场采样，并保留首次参数退出记录。该检查通过后才允许生产队列启动。原构建、首次二进制和旧三站结果保留；本次单独建立输入、执行和输出目录。

首先运行 100 PeV、seed 946 回归例，再交替尝试 300 PeV 与 1 EeV 的新种子，每档最多 27 个，目标各 3 个实际 double bang。后两档沿用此前只按近似相互作用深度预选的种子，必须通过实际完整输运和 τ 谱系判定；不将旧种子计入新种子配额，不强制 CC 或衰变。无合格拓扑的完整事件也保留，结果不是无偏事件率样本。

SiO₂ 基准、原入射位置/方向、IGRF14（2027）、LPM、自然 CC/NC 和 TAUOLA 保持。EM 截断 0.0005 GeV，强子/μ/τ 截断 0.3 GeV，emthin=1e-5，最大权重按 0.5×emthin×E/GeV。OpenMP 256 个物理核，单次一例，resident EM 队列。全部 80 站同时计算 CoREAS/ZHS；经纬度及模型 DEM+1 m 高度取自上一轮逐站核对结果。

输运窗 500 μs，接收窗 2048 μs、524288 点、256 MHz，起点 −1000 ns。矩阶数 12 和原传播细分参数保持。三组射电矩数组约 73.125 GiB，射电预算改为 80 GiB，总设备预算 96 GiB，预估结束时拷贝/渲染峰值约 170 GiB。进程 RSS 上限 220 GiB、可用 RAM 下限 16 GiB、剩余磁盘下限 30 GiB；每例启动前检查内存和磁盘余量。达到资源限制会保留不完整输出并停止队列。

`validation/terrain/doublebang_80stations/run_psr.py` 保存输入和构建散列，检查线程亲和性，记录运行时间、RSS 与已落盘轨迹步数。最终模拟输出完整后才运行 `audit_psr.py`，扫描全部轨迹、沉积、出界末态及 80 站全部电场采样，核对能量账本、源段数量、两个算法和源介质分量闭合。使用原有 `high_energy_doublebang/classify_psr.py` 做实际 CC→τ→非 μ 衰变谱系选择。

新流式验收脚本已在 PSR 用已完成的真实 DEM/80 站边界事件验证；跨分箱、反向和零长度沉积积分也有独立解析检查。没有在本地编译、模拟、数值验收或绘图。

完成事件自动生成全部 80 站的双算法/源介质波形、阵列响应、实际顶点几何和完整域内沉积 profile。`archive_psr.py` 在验收后无损压缩射电 CSV 与 moment 二进制，解压 SHA256 与原文件相同时才删除等价的未压缩副本，记录 `radio_archive.json`。`sync_reports.py` 在本地只做文件传输和状态镜像，服务器任务独立于本地同步进程继续执行。

DEM 边界意味着域外 shower 尾部停止输运，能量单列为逃逸，不能把域内完整 profile 解释为无限空间的完整 shower。拓扑通过不等于站点已能分辨两个射电脉冲。站高与实测的差异仍然存在，不能据此直接宣称绝对观测响应已校准。
