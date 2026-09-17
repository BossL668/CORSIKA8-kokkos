# adaptive-v4：阻断小批次反馈，限制过长批次

## 范围与当前状态

仅改变空气程序显式`--kokkos-cooperative-policy adaptive`的双端调度。
不改物理公式、原生PROPOSAL、随机数生成函数、cut、thinning、max-weight、
磁偏转、射电kernel、单CUDA、单OpenMP、标量CPU或legacy独立双端的策略。
动态调度/租用history区间会变化，不能要求adaptive前后同种子逐事件一致。
生产install和历史冻结二进制不替换，不自动恢复生产campaign。

本轮基线与日志在D盘`CorsikaData/corsika_validation_results/beta5_adaptive_v4_batching_20260911`。
使用已有隔离build`build/cooperative-adaptive-20260911`，旧v3二进制已独立归档。
当前：Release构建、三项调度/亲和性CTest及ASan/UBSan（含泄漏检测）通过。
单CUDA、单OpenMP、标量CPU的N=2固定种子物理输出完全一致，CUDA/OpenMP的
decision trace一致，help完全一致；adaptive N=32已完成，逐事件检查每端实际步数
之和等于profile.steps，累计CUDA 1,136,453步、OpenMP 3,828,591步。
真实EM＋CoREAS/ZHS N=2通量残差为5.5177e-9/5.46753e-9，通过门禁。
两例铁核交替短测已完成；100PeV性能仍待完成，
不能将这些正确性门禁当作性能或大样本统计通过报告。

新冻结二进制SHA-256：
`b018a78cad6f978d7a4e7abc1f507a061908ae013cc88f20b8aec88e15696826`。
前一版v3对照SHA-256：
`22d64274c6c2fe6de3d815db2c0ba6a7889f33da47890e1fe2e82eb20cccddc1`。

## v3中确认的错误方向

100PeV实际输运步数增加约14%，但OpenMP批次从148848增至1977848，
CUDA提交从11190增至50834；总时间36.27→66.87min。
短的协调器响应目标直接参与`target/(8*ms_per_input_wave)`，缩小物理输入量。
固定调用开销在稀疏批次中占比很大，不应外推成下一次完整批次的单位计算成本。

## 本轮修正

1. **批量与响应间隔不再直接相除。** CPU批量目标独立保存，以机器线程数决定
   保守起点（20线程2048、130线程16384），不是永久容量限制。原arena上限不变。
2. **区分完整批次与稀疏尾部。** 至少填满当前上限的7/8才用于批量调节；得到
   完整批次校准后，不再用尾部固定开销覆盖吞吐估计。单个异常样本不能改变批量。
3. **先减波前数，再减输入宽度。** 连续两次完整的单波前调用仍超过目标1.5倍
   时，输入目标才减半；连续两次完整的8波前调用低于目标一半才加倍。始终服从
   用户显式容量上限，并保留256的小批量安全下限（显式arena更小时服从arena）。
4. **保留有用的工作粒度。** CPU每job最多8个波前；GPU光子最多16、轻子最多
   1024。CPU在GPU在途时的软目标为近期GPU时长的1/8、限制在20–100ms；
   GPU空闲时200ms。这是非抢占的目标，不是严格截止，也不保证GPU零等待。
5. **合并稀疏的粒子种类队列。** 某种类队列不足有效批量且另一种类已有有效
   批量时先处理后者；同种类最多推迟8次，没有另一个可用队列时立即排空。
   GPU已有另一有效种类待处理时，使用原有4096退出checkpoint避免长期推进
   极小残余；这是原kernel的参数，不是改动其物理公式。
6. **增加真实工作量证据。** adaptive metadata记录每端、每种类累计
   `transport_records`和`resident_wavefronts`，以及固定21桶的入口批量直方图
   `floor(log2(inputs))`（末桶>=2^20）。没有增长的逐job内存日志。
   验收检查两端步数之和等于profile.steps/gpu_particles，直方图之和等于完成job数。

这仍是有界独立队列和单CUDA提交线程，并未声称已经实现GPU驱动线程无限自主
分配history和输出。主协调器仍唯一持有CPU栈、history分配与fallback/output
回调。也未将20–100ms响应目标误称为“服务与输运已完全异步解耦”。

## 验证方法

- 调度测试：稀疏尾部不缩小宽度，完整过长批次收缩，快设备批量恢复/增长，
  线程起点与显式容量、小队列排空、跨种类合批、在途所有权、异常恢复、唯一提交。
- 真实回归：重构前后单CUDA/单OpenMP/标量CPU的N=2输出与decision trace，
  help完全一致；adaptive N=32，真实EM+CoREAS/ZHS N=2通量账本。
- 内存：可用内存底线4GiB，测试进程树RSS上限5GiB，GPU预算70%。
- 性能：同一新二进制交替运行两例Fe100TeV legacy/adaptive，然后以相同既有
  seed2026110001跑100PeV。物理参数固定，保留总墙钟时间和实际步数/批量分布。
- 一例100PeV不能替代大样本物理统计，也不能证明稳定中位加速。

本轮使用持久systemd用户服务和独立日志，不使用临时终端维持长任务。

## 已完成的两例Fe短测

同物理条件、20线程；v3为上一轮冻结记录，v4为本轮结果。动态调度会改变树，
不是逐粒子完全相同的受控计算。此表专门检验是否减少过碎批次。

| seed | v3时间s | v4时间s | v3 OpenMP jobs | v4 OpenMP jobs | v3 CPU平均入口 | v4 CPU平均入口 |
|---|---:|---:|---:|---:|---:|---:|
| 85000001 | 69.500 | 67.687 | 28,250 | 5,550 | 163.0 | 1,246.3 |
| 85000002 | 69.970 | 66.247 | 25,175 | 5,722 | 172.3 | 1,171.5 |

CPU批次数减少约77–80%；GPU jobs为674/570，未增加；GPU结果服务等待变为
6.461/5.974s（v3为1.261/1.250s），说明较大的CPU工作粒度也有响应成本，
不能仅凭批次下降就宣布总性能恢复。实际总步数由76,208,272/74,289,374变为
74,019,685/70,949,228，耗时下降部分来自工作量变化，尚不证明稳定单位工作加速。

两例新逐端步数、批量直方图与全局profile计数均对账通过，提交全部回收，
queue/profile/radio溢出为0。100PeV仍须独立验证，不能由短Fe结果外推。
它已在相同持久服务启动，采用seed2026110001、20线程、70%显存预算及原
36.27min事件相同的物理参数。默认自动max-weight仍为50，未通过调节物理
参数制造加速。
试跑服务：`c8-adaptive-v4-pilots-20260911.service`。
实时小报告：[run/SERVICE_PERFORMANCE_CN.md](/mnt/d/CorsikaData/corsika_validation_results/beta5_adaptive_v4_batching_20260911/run/SERVICE_PERFORMANCE_CN.md)。

## 2026-09-12：本地完成后的加速验收

本轮五个新事件均结束，服务退出码0；没有自动恢复旧生产队列。
完整报告、耗时图、资源曲线和批量直方图：
[acceptance_local_20260912/ACCEPTANCE_REPORT_CN.md](/mnt/d/CorsikaData/corsika_validation_results/beta5_adaptive_v4_batching_20260911/acceptance_local_20260912/ACCEPTANCE_REPORT_CN.md)。
脚本：`validation/accelerator/accept_adaptive_v4_performance.py`（只读分析，不修改模拟）。

- 100 PeV v4 总时间 **3190.539 s / 53.18 min**，比v3缩短20.48%，
  但比原独立双端36.27 min慢46.61%；**尚未恢复原独立双端的高能性能**。
- 实际推进3,316,785,422步，比v3多0.89%；此次改善不是由更小的shower造成。
  CUDA/OpenMP jobs为40,693/399,322，比v3减少19.95%/79.81%，
  但仍是原独立双端的3.64/2.68倍。
- GPU完成到协调器领取累计365.42 s，高于v3的75.52 s及旧独立双端的88.13 s。
  此等待与CPU工作重叠，不能当作可从总时间直接扣除的数字。
- CUDA/OpenMP实际输运步数分别2,162,652,895 / 1,154,132,527，占65.20%/34.80%。
  平均GPU77.26%，CPU18.62个逻辑核当量；峰值显存5566 MiB，RSS峰值1.83 GiB，
  系统可用内存最低10.45 GiB，未触发保护。
- Fe100TeV两例同二进制交替测试中位时间86.778→66.967 s，缩短22.83%；
  两例不足以替代500例统计验收。

五个新事件的全部七类Parquet逐批扫描有效，profile/interaction histogram有效，
CoREAS/ZHS数组有效；提交/回收、pending和逐端步数/波前/批次直方图对账通过。
队列/定点溢出和native反解失败为0，指定fallback逐项对账。
100PeV局部能量账本仍明确不完整，残差6.166%，不称为完整能量闭合通过。

历史单CUDA72.82 min仅作参考：对应主粒子、介质/辅助表哈希、磁场、thinning、
max-weight已核对，但其metadata没有磁偏转换算常数，且二进制版本不同。
v4相对该历史耗时的描述性比值为1.369倍，不是严格同版本受控加速比。
