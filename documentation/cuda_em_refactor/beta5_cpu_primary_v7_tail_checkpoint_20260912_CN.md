# CPU优先v7：小尾部返回阈值试验

状态：代码与验证门禁已实现，PSR隔离Release及3项调度/绑定CTest通过。尚未完成真实事件验收，
不推荐替换生产版本。按用户最新要求，本地不再编译或运行测试。

## 依据与假设

v6的物种选择优化在已检查事件中没有触发。更直接的代码机制是：CPU优先
批次固定使用4096作为resident返回阈值，而某些批次初始输入本身不足4096；
`KokkosResidentLeptonCascade.hpp`在至少推进一波后，若存活数低于阈值就返回。
一例本地双端事件记录了11092次below-minimum checkpoint。这不是粒子截断，
但会反复产生host调用、队列打包和跨物种交接。

纯OpenMP也使用原有最小批量，因此这一观察**尚不能单独证明双端比单端慢的
全部原因**。下一步用独立候选测试减少返回次数是否真的改善吞吐和总时间。

## 唯一新增行为

仅在`openmp-cuda`的CPU端，将每批返回阈值设置为：

```text
max(1, min(native minimum, 初始批量 / 4))
```

整数除法。初始批量至少16384且native minimum=4096时仍使用4096。不降低
物理能量cut，不丢弃尾部，不限制输入容量，不新增时间片控制。批次仍受到
原有波数、history范围和内存边界约束，返回时保留全部未完成粒子。

新产生的另一物种可等到本次安全边界才交给协调器，因此减少往返也可能延后
跨物种服务；必须实测，不假设波前越长越好。GPU分配、阻塞等待、fallback
处理不变；单OpenMP、单CUDA及原GPU优先不进入该分支。

## 审计与验收

- 4文件白名单：`CpuPrimarySubshowerPolicy.hpp`、`IndependentSubshowerPump.hpp`、
  `KokkosShowerReport.hpp`、`testIndependentSubshowerPump.cpp`。
- 与各机器冻结v6源码相比，不复制当前工作树中无关的山体、kernel、物理模块。
- 21个floor(log2(input))桶，最后一桶饱和；桶和必须等于CPU已提交工作段数。
- 单元测试覆盖1/3/4/256/4095/4096/16383/16384/65536输入、所有权和恰好一次
  提交；验证脚本拒绝错误语义、非法桶、负数及总和不符。
- 本轮48项Python测试已通过；真实C++与物理门禁需以服务器结果为准。
- PSR构建目录：`build/psr-cpu-priority-v7-tail-20260912`；独立128路编译，
  不更新install，不修改Conan缓存或重新制表。
- 后续：单端N=2输出/decision tape、两种双端fixture、N=32生命周期，再跑
  五对100 TeV质子；130线程固定CPU382–511，T400预算70%，参数与v6一致。
- 所有输出先留PSR，结束后只将小报告和图同步本地。监控不足的事件保留但
  不用于严格性能通过声明。

性能判据仍是纯OpenMP与双端的端到端时间，并同时报告CPU有效步吞吐、批次
分布、checkpoint、GPU工作份额、资源峰值。短样本不代替500例统计物理验收，
两个端点的重叠墙钟不能相加当作事件用时。

## 服务器执行记录

PSR程序SHA-256：`91cebbcf2064824f7518cdb285ef4a7c7836727edea534b9a55857ac8784de5b`。
源码清单确认仅上述4个文件变化，`unexpected=[]`；没有覆盖安装版。

持久验收服务：`c8-cpu-priority-v7-psr-acceptance-20260912`；完成后报告服务
`c8-cpu-priority-v7-psr-report-20260912`审计关闭状态并生成计时图和诊断报告。
任一正确性门禁失败，后续五对计时不启动。

结果目录：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v7_tail_proton100TeV_130_20260912`

本地无v7编译/模拟服务。v6本地未完成的最后一个事件已按用户要求停止；
PSR v6五个双端事件的合并计数均为0，保留这一负结果。
