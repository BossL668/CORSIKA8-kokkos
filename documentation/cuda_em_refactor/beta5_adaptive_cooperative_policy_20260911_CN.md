# CUDA＋OpenMP：按实测成本分配的实验策略

> 后续审计确认 adaptive-v2 的 100 PeV 性能严重退化；本文保留 v1/v2 研发记录。
> 当前 v3 的协调器响应预算修改、正确的 36.27 min 历史基线和验收状态见
> [v3 诊断](beta5_adaptive_v3_service_budget_20260911_CN.md)。下文“两端相同 200 ms”
> 是旧方案，不再是当前 adaptive 的 CPU 在途策略。

## 范围和使用

只改变空气应用的显式双端调度；不修改共享物理 kernel、PROPOSAL 数据、
磁偏转常数、射电公式、cut、thinning、max-weight 或山体 session。
已有单端和双端生产程序不被替换。新策略仍需性能和统计验收，不设为默认。

在 CUDA_OPENMP 组合构建中使用：

```bash
c8_air_shower <原有初级、能量、方向和输出参数> \
  --em-backend kokkos --radio-backend kokkos \
  --kokkos-execution cuda-openmp --kokkos-num-threads 130 \
  --kokkos-cooperative-policy adaptive
```

省略最后一个选项，仍是原有 `legacy` 双端策略。单 OpenMP、单 CUDA 或标量
PROPOSAL 不接受 `adaptive`，在初始化 shower 前报错。没有新增 `openmp-cuda`
执行空间；快端由实际运行表现决定，不由名称决定。

## 为什么服务器需要改变调度

此前同机 100 TeV 试跑中，纯 OpenMP130 约 52–57 s，T400＋OpenMP130 约
188–189 s。即使两端可独立推进，原双端 CPU 仍只有 16384 容量、8 个轻子波前，
跨种类队列预算也被缩减为普通 OpenMP 的 1/16。GPU 初始保留量和线程数份额
使小显卡与大 CPU 服务器的配比不合适。这些不是 Kokkos 的强制要求。

本轮自适应策略：

1. **CPU 使用纯 OpenMP 的容量及队列预算。** 不再应用双端专属缩减；
   默认 CPU 容量 65536，显式用户安全上限仍有效。一个共享 arena，不按线程复制。
2. **真实小批量校准。** 每端每类粒子在未测量时最多接收 4096 个待校准输入，
   首次只推进一个完整波前。不是重新抽样物理，不添加随机数。
3. **动态份额。** photon/lepton 分别记录实际 transport records/ms，用移动平均
   估算两端当前积压成本，将新输入送往预计更早完成的一端。
   无固定 GPU 保留份额、CPU 百分比上限或线程数先验。
4. **对称的交接粒度。** 根据输入数和实测输入/波前成本调整两端 epoch，
   参考目标 200 ms；首次从 1 开始，逐步增长，最高使用原接口的
   photon 16 / lepton 1024 波前。不是 200 ms 的强制截止，更不是物理时间 cut。
   不打断 kernel，不让快端故意休眠。
   adaptive-v2 不将原来整批输入的 4096 staging 阈值作为子队列立即返回阈值；
   即使尾部小于 4096，也允许继续在本端推进，直至正常 epoch/物理边界。
5. **安全迁移。** 估计两端完成时间差，扣除实测迁移成本和防抖余量后才移动。
   CUDA 在途时可以移动主线程持有的尚未提交粒子；正在执行的设备队列不可访问。
   设备驻留粒子仅在安全批次边界转移。每种粒子每次最多一次搬运，避免来回抖动。

CPU 栈、history 预留、fallback 和最终输出仍由协调器唯一管理；
双端继续使用各自的 EM/radio 累积器，最后按原有检查方式合并。
排队有界，已完成结果只能提交一次；异常仍失败关闭，不改成另一端静默续跑。

## 估计模型的边界

粒子数并不等于未来完整子级联工作量。该策略的排队模型使用近期逐步吞吐近似，
会随 shower 演化更新；不能预知未来分枝或保证双方持续 100%。
校准测量包括 backend 调用内的 EM、射电和传输，但不包含协调器之后的标量 fallback
及文件输出；最终收益必须用完整事件耗时判断，不能只比较这个吞吐估计。
当一端只能贡献少量净吞吐时，动态平衡允许它获得很少工作，而不是强行平均分配。

动态调度和批次改变会影响 history 分配时序，不承诺不同策略同 seed 产生同一棵树。
同一轨迹的物理公式没有另写一套；大样本统计仍是独立验收项。

## 代码位置

- `corsika/accelerator/em/detail/AdaptiveSubshowerControl.hpp`：
  无物理依赖的逐端/逐类成本估计和波前交接策略。
- `corsika/accelerator/em/detail/IndependentSubshowerPump.hpp`：
  有界输入分配、安全迁移、独立推进和完成提交。
- `src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp`：
  adaptive 下沿用纯 OpenMP workspace；legacy 配置不变。
- `applications/c8_air_shower.cpp`：选项和兼容性门禁。
- `applications/detail/air_shower_kokkos/KokkosShowerReport.hpp`：
  仅 adaptive 额外输出策略、逐端吞吐/样本数/波前上限及迁移时间。

## 验收状态

PSR 独立目录 `build/psr-cooperative-balanced-20260911` 编译通过；
编译前二进制及调度源码保存于其 `pre-adaptive/`。不在本地编译或占用本地显卡。

已通过：

- 模拟快 GPU / 快 CPU 的对称控制测试；无 CPU 特殊波前上限；
- 19000 个混合 PID 输入、受控 GPU 阻塞、CPU 连续推进、队列排空和唯一终止身份；
- 原有所有权、spill、history 及异常清理回归；
- PSR 系统 GCC 的 ASan/UBSan（含泄漏检查）调度测试；
- Release 编译和调度/driver 亲和性 CTest。

真实物理/性能流水线：
`validation/accelerator/run_adaptive_cooperative_acceptance.py`。
保存命令、二进制 SHA-256、亲和性、CPU/RSS/GPU 采样和完整结果。
先检查旧/新单端固定 seed N=2 的 11 类数组及 decision trace、非法参数门禁、
真实 EM＋CoREAS/ZHS 能量通量 fixture N=2、自适应 N=32；通过后再测
同二进制、相同物理参数、热缓存、交替顺序的 100 TeV 纯 OpenMP130/双端。
试跑保持 emthin=1e-6 和默认 max-weight，不以改变物理参数获取加速。

初次 help 门禁只发现新增选项引起 citation 日志源码行号变化；仅规范化该行号，
其余帮助文字、顺序和默认值仍严格比较（另显式移除本次新选项块）。
原始诊断保留在 `psr_t400_adaptive130_20260911_v3`，不是成功物理样本。
正式试跑输出位于 PSR：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive130_20260911_v3a`。

当前不能宣称性能门禁或大样本物理验收通过，须以下方后续实测记录为准。

## 真实验收的阶段结果

- 标量 PROPOSAL、单 CUDA、单 OpenMP：各 11 类固定种子物理数组一致，
  非计时 metadata 没有非预期变化，CUDA/OpenMP decision trace 逐字节一致。
- 非双端请求 adaptive 的三种错误组合均在 shower 启动前拒绝。
- 130 线程真实 EM＋CoREAS/ZHS fixture 的 N=2 通过；第二事件能量通量相对
  残差为 `5.41788e-9`，两事件 workspace 均为 `605355966` bytes。
  这是加速器边界的纯 EM 通量检查，不代替强子完整 shower 的总能量账本。
- 100 GeV photon 自适应 N=32 全部完成，用时 49.563 s，队列提交全部回收；
  最后资源采样 RSS 为 823676928 bytes。仍需结合事件工作区和 RSS 曲线判断保留量，
  单个末次 RSS 数字不是“绝无内存泄漏”的证明。
- 100 TeV 端到端计时正在执行；这些正确性检查并不等于加速已达标。

第一轮 adaptive-v1 的两对 100 TeV 已完成：

| seed | 纯 OpenMP130 | 自适应双端 |
|---|---:|---:|
| 2026110001 | 51.002 s | 58.088 s |
| 2026110002 | 56.882 s | 76.441 s |

两对中位时间 53.942 / 67.265 s，双端仍慢 24.7%，不能宣布净加速。
CPU 平均占用分别约 9903% / 9364%，峰值接近 13000%；CPU 无工作等待分别只有
25.97 / 13.63 ms，说明此前长期饿死问题已明显缓解。
但 GPU 结果完成到领取的累计间隔仍有 16.42 / 20.61 s；它与 CPU 工作重叠，
**不能把这个累计数字直接视为可节省的总耗时**。
第二事件 CPU epoch 达到 15063 次，进一步暴露小尾部频繁返回。

因此追加 adaptive-v2：只在自适应模式取消子队列小于整批 staging 阈值时的单波前
返回条件，改由测量驱动的 epoch 控制交接。原单端及 legacy 的阈值不变。
v1 二进制及源码快照保存于独立 build 的 `adaptive-v1/`；
v2 必须使用新结果目录并重新跑门禁，不覆盖或混入 v1 的计时。

## PSR adaptive-v2 结果及本地试验

PSR v4 结果目录对应 adaptive-v2。全部回归门禁再次通过，N=32 用时 30.752 s。
两个 100 TeV 质子种子的纯 OpenMP / 双端端到端时间为：
50.644 / 63.958 s、57.673 / 59.742 s。中位数 54.158 / 61.850 s，双端仍慢约 14.2%。
CPU 平均占用约 9996% / 10338%，CPU epoch 降为 851 / 643 次；
不能再将该结果简单解释为“只用了很少 CPU 核”。仍没有服务器净加速结论。
该结果的小型 JSON 报告已复制到 D 盘同名 `psr_t400_adaptive130_20260911_v4` 目录。

用户随后要求本地 RTX4060＋20 线程验证。此时允许本地独立构建和 GPU 测试，
覆盖此前“只在 PSR 测试”的范围限制：

- 构建目录 `build/cooperative-adaptive-20260911`；不安装或覆盖现有生产版本。
- 本地 Fe500 生产队列暂停，完成样本保留，未完成事件记录为 interrupted，可按原种子重试。
- 本地 Release 和 3 项调度/亲和性 CTest 通过；旧/新单端 N=2 数组/trace、
  自适应 N=32、真实 EM/radio N=2 门禁通过。
- 单 CUDA、legacy 双端、自适应双端使用同一个新二进制计时，历史生产数据仅作背景。
- Fe56：总能量 100 TeV、垂直、emthin=1e-6，其他条件复制原 Fe manifest，
  5 个种子、三模式交替运行。
- proton：100 PeV、47°/180°、emthin=1e-6，复制之前高能命令，
  先运行自适应，再运行单 CUDA 和 legacy 双端；一个种子只是 pilot。
- 双端 20 线程，70% 显存；测试子进程 RSS 上限 5 GiB，
  系统可用内存下限 4 GiB，D 盘保留 15 GiB；达到保护条件立即中止本轮。
- 持久化服务运行，结果写入 D 盘
  `beta5_adaptive20_Fe100TeV_proton100PeV_20260911_v1`。

测试脚本为 `run_local_adaptive_pilots.py`，可显式 `--resume` 复用
命令及二进制身份相同且守护记录成功的阶段；失败/中断的守护记录不会当作成功跳过。
尚未自动恢复旧生产队列，避免与高能对照抢占显卡。

### 本地首批计时（进行中，不是最终验收）

最先完成的两个 Fe56 种子（85000001/85000002），三模式的端到端时间如下：

| seed | 单 CUDA | legacy20 | adaptive20 |
|---|---:|---:|---:|
| 85000001 | 109.558 s | 97.962 s | 69.085 s |
| 85000002 | 75.446 s | 75.892 s | 68.375 s |

这两个共同种子的中位数为 92.502 / 86.927 / 68.730 s。
自适应相对单 CUDA 的描述性时间降幅为 25.7%；当前样本很少，
动态调度并不保证相同 shower 树，不据此宣布一般性的性能达标。
自适应平均占用约 17.48 / 17.41 个逻辑核，采样峰值显存均为 5316 MiB，
峰值 RSS 约 1.29 / 1.25 GiB；没有触发本轮内存保护。

完整 Fe 计划是每模式五个种子；随后自动运行 100 PeV 质子三模式 pilot。
`summarize_local_adaptive_pilots.py` 在每次完成后更新结果目录下的
`PERFORMANCE_REPORT_CN.md` 和 `PERFORMANCE_SUMMARY.json`，只对三模式全部完成
的共同种子汇总中位数，避免只采用已完成的快事件。
资源数据为进程 CPU 时间及设备级 GPU 抽样，并不提供真实 kernel 重叠时间线。

Fe/proton 的现有 GPU 能量账本标记 `complete_coverage=false`：它没有覆盖完整
强子事件。纯 EM fixture 的闭合通过不能替代这里的全 shower 能量或系综验收。
本轮不把该局部账本的差额直接解释成粒子丢失，也不宣布它已完整闭合。

### 本地 Fe 五例三模式验收更新

三模式各五例全部结束。中位总耗时为单CUDA73.760 s、原双端86.518 s、
自适应71.357 s。自适应相对单CUDA仅减少3.26%，相对原双端减少17.52%。
因此上方前两个种子的25.7%不能作为完整五例结论；尚未证明稳定5%净加速。

分析复用了既有profile提取器和原pulse_analysis的逐事件偏振投影，输出
全部组分/parent profile、标量分布、Ex′/Ey′/Ez′平均CoREAS/ZHS及时间分布。
结果位于D盘本轮目录下 `acceptance_Fe5_vs5_vs5_v3/VALIDATION_REPORT_CN.md`。
脚本是 `validation/accelerator/analyze_adaptive_local_pilot.py`；不改变生产程序。

15例完整性、表/辅助数据及物理条件一致性、提交回收、溢出、反解失败和波形有限值检查通过。
尚未进行大样本物理验收：自适应对单CUDA的EM积分差+0.74%，但地面EM数差−11.31%，
逐事件EM Xmax均值差−16.61 g/cm²。全记录平均波形L2差约23.3%（CoREAS）/21.6%（ZHS）。
五例不足以区分涨落和系统差异；不以阴影重叠或“不显著”认定通过。

发现需要区分的既有metadata语义：`cpu_generic_fallbacks`来自router的即时返回数。
独立双端立即执行的指定PROPOSAL完成也计入其中，单CUDA延迟完成则不计入。
分析按reason和`cpu_completed_*`及延迟队列计数逐项核对，不能直接跨模式比较这个总数。
本轮仅记录该命名/口径问题，未修改物理代码或输出schema。
