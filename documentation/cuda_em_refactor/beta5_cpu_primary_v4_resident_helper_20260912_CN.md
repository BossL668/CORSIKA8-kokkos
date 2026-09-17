# CPU 优先 v4：原生主端取数顺序与有界 GPU 自主续跑

实验候选，只作用于 `--kokkos-execution openmp-cuda`。不改变单CUDA、
单OpenMP、原GPU优先双端、山体程序或物理kernel；不替换生产安装。

## 为什么还需修改

[v3五种子报告](beta5_cpu_primary_v3_protected_front_20260912_CN.md)证明指定
fallback合批及完整CPU arena保护通过正确性检查，但双端中位时间仍比
纯OpenMP慢12.3%。仅提高两端占用率不等于有效吞吐恢复。

进一步比对实际调用结构：原standalone router先装入本端驻留粒子，再用
剩余arena位置接收新输入；双端pump原来先接新输入。这可能打断已有前沿
的连续处理，完整容量相同不代表取数顺序相同。辅助GPU原来每次调用后
也必须等待协调器领取，在CPU长调用期间无法继续推进自身已经拥有的队列。

## 本轮变更

1. CPU-primary的CPU取数恢复为 **resident first → waiting input**。两端arena、
   CPU photon16/lepton1024 wave上限、CPU min-batch和物理算法不变。
2. 辅助GPU每个独立packet最多4次完整调用：首次加3次自主续跑。只访问
   GPU自身驻留队列，固定photon16/lepton8/minimum1，不执行adaptive时间
   预测或控制器更新，不为喂GPU而拆短CPU工作段。
3. 协调器提交前一次预留最多3个续跑history区间，保持跨PID单调stride bank、
   溢出和重复检查；GPU driver不调用Stack、writer、CPU物理或history分配器。
4. 到达4次调用、无驻留粒子、无进展、结果保留预算或CPU请求工作时，在
   完整调用边界交回。不能抢占正在执行的kernel，因此并非硬实时截止。
5. GPU结果保留预算沿用64MiB，检查发生在下一次调用之前，实际峰值允许
   “预算加一个原有合法结果对象”。不能误写为全部结果严格不超过64MiB。
   收尾逐结果恰好提交一次，最终队列和在途packet全部为空才结束事件。

原GPU优先仍单调用packet，旧adaptive仍为原16调用和原控制器；本轮不修改
它们的物理路径或默认策略。CPU-primary动态调度及history预留时序发生变化，
不承诺与v3同seed长出逐位相同的树；相同物理算法不免除统计验收。

## 验收状态

- 两种PID的resident-first、CPU阻塞时GPU实际独立4次调用、固定wave/minimum、
  单调history、CPU交接、结果内存预算、空队列、零进展及续跑异常测试通过。
- 主机优化构建、ASan/UBSan/泄漏检测通过；原GPU优先单调用断言通过。
- 两机从冻结v3复制为新的独立v4构建；只覆盖已批准的调度/报告和测试文件，
  物理模块、PROPOSAL缓存、生产安装不替换。
- 两机CUDA+OpenMP Release独立构建及三项调度CTest通过。
- 两机PROPOSAL、单CUDA、单OpenMP的N=2输出重新逐项比较通过；CUDA/OpenMP
  decision tape字节相同。本地两例100TeV Fe的单CUDA输出与冻结v3各9个数组
  文件完全一致，非计时metadata、计数和表哈希一致。
- 两机真实双端EM/CoREAS/ZHS fixture各2例通过；加速边界能量残差约
  `5.5e-9`。该边界检查不等于完整强子能量账本覆盖。
- N=32重新检查：两机缓存、工作区恒定、零溢出、系统可用内存底线通过；
  RSS峰值本地约0.774GiB、PSR约0.807GiB。CPU优先这组仅第5个shower实际
  使用辅助GPU（9次调用、4次自主续跑、5个packet、最大3calls），不能称为
  “32次GPU持续复用压力验收”。其余事件任务不足时不强行为GPU分配工作。
- 本地4例Fe输出完整。两例原GPU优先的NVML查询发生超时，保留时间和数据，
  但监控有效性不通过；不从这两例宣布性能通过。
- PSR首个纯OpenMP Fe被其他山体GPU任务的**启动前**互斥门禁拒绝；没有
  启动shower、没有物理崩溃。原拒绝记录可恢复归档，无数据删除、未中断
  山体任务。已在独立systemd服务排队恢复五种子、两模式、AB/BA交替对照，
  保留130线程和`382–511`亲和性及相同`spread/threads`绑定。
- 随后的纯OpenMP seed85000001完整完成，用时56.572s。双端同seed执行18.389s
  时，另一个山体GPU测试启动，独占测试guard终止了**本次被干扰的测量**；
  不是物理异常，也不能把18.389s当成完整shower耗时。该尝试和最初的启动前
  拒绝均保留在`continuations/`下，不用失败样本冒充加速结果。
- 当前续跑等待整个外部山体工作流结束并连续空闲120s，而非仅检查瞬时GPU
  空闲；不重跑已完成的单端样本，不改变五种子计划。每例资源等待最多6h、
  单例计算最多1h，独立服务最多8h，超时明确失败，不无限自动重试。
- 不因正确重叠就宣布有净收益；目标仍是CPU吞吐不被破坏、辅助GPU有
  实际贡献、同机中位时间不回退。完成测量前不将v4推荐为生产加速方案。

本地构建目录：`build/cpu-priority-v4-resident-helper-20260912`。
PSR构建目录：`build/psr-cpu-priority-v4-resident-helper-20260912`。

PSR结果目录：`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v4_resident_helper_130_20260912`。
当前续跑服务：`c8-cpu-priority-v4-psr-resume-isolated-20260912.service`。
查看`STATUS.json`、`RESUME_WAIT.json`和`continuations/*/RESUME.json`可区分
等待、已完成样本以及曾被资源冲突中断的尝试。

独立只读监控服务：`c8-cpu-priority-v4-psr-thread-sample-20260912.service`。
先最多等待6h，确认冻结二进制实际启动后采样最多1800s，JSONL上限64MiB；
记录每TID实际CPU、核时、schedstat排队和wchan及读取错误，不发信号、不调整
亲和性。PID/TID复用按start_ticks区分；同时记录同用户外部CPU活动，方便识别
编译等干扰。结果为`THREAD_CONTENTION.jsonl`，原始采样留在服务器。

独立收尾服务：`c8-cpu-priority-v4-psr-report-20260912.service`。
生产测量服务终止后重读完整输出，生成`performance/`和`diagnosis/`；只有
真实完成才检查全部N=32，失败或部分完成不写成通过。观察超时不重启任务。
续跑、报告、线程采样及其相关宿主机合同测试本轮合计54项通过。

## 后续瓶颈定位边界

### 已取得的有效 v4 对照（2026-09-12，后续检查点）

PSR 已有3个有效完整记录，仅1个完整配对：seed85000001纯OpenMP
56.572s、CPU优先双端64.000s，双端仍慢13.1%；seed85000002双端48.489s，
同seed纯OpenMP尚因其他山体工作流占用资源而排队。不能据此称五种子验收完成。
首个配对主端分别推进75,170,799/71,445,169步，backend调用累计48.444/54.547s，
对应1.552/1.310百万步每秒。动态分树不同，这个吞吐下降是诊断线索，不是
固定相同物理工作量的因果试验。辅助GPU承担1,853,304步，约2.5%。

新线程采样确认实际driver并非被错误固定在某一个核：在允许的后130核内
迁移，但主要落在382/383等OpenMP工作核上。两个双端事件中分别消耗8.97和
6.66 CPU秒，同时记录到8.76和6.42秒runnable wait。driver活跃区间内，
协调器排队比例也较高。该采样未覆盖同轮纯OpenMP，且系统schedstats未开启，
不能将这些计数解释为完整因果证明或全部GPU等待都在自旋。

下一候选[v5阻塞辅助等待](beta5_cpu_primary_v5_blocking_helper_20260912_CN.md)
仅调整CPU优先辅助端的stream等待；保持本轮批次、history和物理模块不变。
它仍需固定相同输入的竞争试验和同机完整shower对照，未替换推荐版本。

v3五例实际活动CUDA driver允许使用后130核全集，并非被错误固定到thread0。
但是它与130个OpenMP工作线程共享这些核，调用期间消耗约52%–56%一个CPU核。
这包含提交和主机处理，不能全部记为自旋，也不能仅凭此证明OpenMP吞吐下降
的因果关系。下一步以每TID实际运行CPU、排队时间和等待点检查竞争。若成立，
可在CPU优先GPU实例单独测试阻塞式CUDA事件等待；不能修改CUDA全局调度标志，
或把原GPU优先与单端的等待方式一并改变。
