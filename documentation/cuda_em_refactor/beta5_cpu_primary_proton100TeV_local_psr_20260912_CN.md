# 本地与PSR：CPU优先策略的100 TeV质子测试

本轮使用[CPU优先v5 a3](beta5_cpu_primary_v5_blocking_helper_20260912_CN.md)，
保持核心算法和生产安装不变，只增加验证脚本的明确质子选择。

| 条件 | 本地 | PSR |
|---|---|---|
| 初级、能量、方向 | proton，100000 GeV，vertical，phi=0 | 相同 |
| emthin / max-weight | 1e-6 / 0（原默认） | 相同 |
| 射电 | 79天线，CoREAS+ZHS，400 ns，1 GHz | 相同 |
| 地磁/物理源 | IGRF14(2027)，proposal-native既有缓存 | 相同 |
| OpenMP | 20线程，CPU0–19 | 130线程，CPU382–511 |
| 辅助GPU | RTX4060 Laptop | T400 4GB |
| 比较模式 | openmp / openmp-cuda | 相同 |
| GPU预算 / min-batch | 70% / 4096 | 相同 |
| 种子 | 85000001–85000005，AB/BA交替 | 相同 |

两机均OMP_PROC_BIND=spread、OMP_PLACES=threads、hadronic-workers=1。
PSR的382–511经lscpu核实为130个不同物理核心。各机器内部比较CPU吞吐、
shower和完整进程时间；不能以跨机器时间差直接评价调度策略。动态模式
同seed不保证相同shower树，因此同时保留固定输入的CPU效率控制测试。

## 本地运行

同一a3二进制SHA：`5871213541f3512647df6b3cee295fcf6e97e00302ac632c5b2fbc31248a18f3`。
既有回归重新核验18份证据，不重跑、不复用Fe计时。新验证工具67项测试
通过；`--timing-primary proton`去掉-Z/-A，实际命令为-p 2212，标签
为proton100TeV。旧`--fe-command`仅为模板接口名，不代表实际初级仍是Fe。

服务：`c8-cpu-priority-v5-a3-local-proton100tev20-20260912`。
输出：`/mnt/d/CorsikaData/corsika_validation_results/beta5_cpu_priority_v5_checkpoint_helper_a3_proton100TeV_local20_20260912`。
独立报告服务在结束后生成`performance/REPORT_CN.md`、
`performance/proton100TeV_timing.png`与`diagnosis/REPORT_CN.md`。
进程树RSS上限6 GiB、系统可用内存下限4 GiB，外层cgroup上限8 GiB。

首对阶段记录：纯OpenMP进程167.295 s、shower157.646 s；双端进程157.993 s、
shower146.996 s。第二对进程138.469 s与125.069 s。这些不是完整五对结论。
后续核对发现首对双端和第三对部分NVML查询超过3秒，guard因此将这些
样本标为performance_valid=false；shower及输出仍正常。上述首对耗时
仅为观测值，不作为严格性能门禁的通过依据，不删除或改写监控失败记录。

## PSR重新启动

旧r3队列在16:56保护退出，因为旧v4结束后的临时服务记录被回收，而严格
门禁不能据此确认成功。已检查旧STATUS的10/10完成记录；用户明确要求
空闲后重启，CPU、T400和内存检查通过。旧失败记录保留，不伪造旧服务状态。

远端14个覆盖文件与本地a3逐项SHA一致。独立128路构建已成功，源码边界
检查没有非预期文件变更，未运行install。107项验证工具测试在两机均通过。
构建单元：`c8-cpu-priority-v5-psr-build-r4-20260912`。
验证单元：`c8-cpu-priority-v5-psr-proton-followon-r4-20260912`。
冻结SPEC SHA：`0516116fe67a9b53ec11472836af1fcd964e63a765301e36e54937684f180d93`。

流程：真实CUDA stream检查 → 固定输入无/有射电 → 单端N=2与双端N=32
回归 → 五对130线程质子测试 → 输出审计、时间图及CPU效率报告。
每步/每例观察15秒已知CPU工作流和GPU空闲，并有最终与运行中检查；
它不是服务器资源锁。其他CPU工作流出现时计时失效但保留正常物理结果；
超时、内存或GPU错误只处理本测试进程树，不停止别人的任务。

结果：`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v5_checkpoint_proton100TeV_130_20260912`。
完整事件报告位于`acceptance/performance/`和`acceptance/diagnosis/`；
固定输入控制位于`fixed-work-radio0-report/`和`fixed-work-radio1-report/`。
原有山体模块、任务及数据不覆盖。编译通过不等于性能通过，完整五对短测
也不替代500例物理分布验收。

### PSR端点阶段记录

真实CUDA stream验证通过，包括32次checkpoint字段精确回传、空队列、
事件复用和不等待其他stream。固定输入5次正式重复也完成，全部模式的
输入、调用及输出哈希一致、资源监控有效：无射电CPU中位时间从helper-off
0.264 s变为blocking-event 0.268 s（约+1.5%）；全射电从2.138 s到2.010 s。
后者不解释为GPU能提高同一CPU工作量的固有性能，也可能受线程调度/频率
波动影响；这些结果均不是完整shower的净加速结论。小报告已同步到本地
`/mnt/d/CorsikaData/corsika_validation_results/psr_cpu_priority_v5_checkpoint_proton100TeV_130_20260912_reports`，
完整原始记录保留在PSR。

## 完成结果：每机5对，全部输出正常

本地与PSR均为10/10事件完成；PSR外层12个步骤均exit 0，单端N=2回归、
双优先级N=32生命周期、真实stream及完整输出检查通过。PSR程序SHA为
`eff87e344991065af2cfdda9accf3ec43578f31b8d36a474b7e1b5b9ef9a9ed1`。
这是独立候选构建，未更新生产install；没有同步无关的山体改动。

| 指标：纯OpenMP → CPU优先双端 | 本地20线程+4060 | PSR130线程+T400 |
|---|---:|---:|
| 完整进程中位/s | 152.411 → 149.781 | 48.413 → 47.911 |
| 完整进程平均/s | 150.555 → 142.449 | 46.637 → 48.450 |
| shower中位/s | 143.637 → 140.219 | 41.727 → 40.771 |
| CPU端百万输运步/s中位 | 0.5935 → 0.5811 | 2.1101 → 1.9752 |
| 双端GPU输运步占比中位 | 3.78% | 3.80% |
| 监控完整配对数 | 1/5 | 5/5 |
| 双端进程树RSS峰值/GiB | 1.778 | 1.871 |

结论：**未证明稳定净加速，也未达到“不损失OpenMP效率”的发布条件。**
PSR中位进程时间仅缩短1.04%，平均时间反而增加3.89%；五对中两对较快、
三对较慢。CPU每步吞吐的中位值降低6.39%，各配对降低约5.0–14.6%。
本地中位进程时间仅缩短1.73%，且6个事件NVML查询超时，不能作严格速度
验收依据。所有慢事件和监控失败事件均保留，没有选择性剔除。

PSR双端CPU平均核当量在103.4–112.0之间（约10342–11197%），纯OpenMP
为103.0–111.8；因此不是“没有启用130线程”。该指标含自旋，不等于有效
物理工作。GPU平均利用率25.4–43.4%，设备峰值显存2526 MiB。

固定输入控制已表明辅助stream的阻塞等待能避免显著的驱动自旋竞争；
然而完整shower中CPU波前仍从5854–8907增加到8797–19620。特别seed85000005，
CPU步数约+8.2%、波前数约+121%、CPU调用墙时约+26.6%，双端进程耗时58.90 s，
纯OpenMP为48.41 s。数据提示粒子拆分后的CPU批次形状/尾段与回退服务时机
仍值得诊断，而非单纯增大线程数或显存。不同动态调度产生不同shower树及
粒子能谱；上述计数关联不是同输入因果证明，下一轮应固定输入队列单独
测量fill、scan、transport、radio和fallback服务，而不是据此修改物理参数。

输出审计确认数组有限、输出关闭、提交与领取相等、回退清空、定点无溢出。
它不证明完整强子能量闭合或500例物理统计等价。N=32检查确认固定工作集
下表哈希及workspace不增长，也不宣称任意能量无内存泄漏。

### 本地查阅

本地20线程目录内`CONCLUSION_CN.md`与`performance/proton100TeV_timing.png`
已生成。PSR图和小报告已同步至上述`_reports/`目录，具体为：

- `acceptance/performance/proton100TeV_timing.png`
- `acceptance/performance/REPORT_CN.md`
- `acceptance/diagnosis/REPORT_CN.md`
- `acceptance/COMPLETED_OUTPUT_AUDIT.json`
- `acceptance/N32_LIFECYCLE_AUDIT.json`

服务器原始shower数据没有下载到WSL或D盘。
