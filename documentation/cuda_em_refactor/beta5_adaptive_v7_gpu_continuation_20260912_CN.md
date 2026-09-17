# adaptive-v7：有界 GPU 自主续跑

## 后续更新：本地五种子已经完成

2026-09-12约02:21，本地五种子Fe100TeV、完整射电、相同二进制的单CUDA与
adaptive20交替测试全部完成。以下历史进度段落保留实施过程，不表示仍未完成。

| seed | 单CUDA/s | adaptive20/s | 单CUDA/双端 |
|---|---:|---:|---:|
| 85000001 | 109.981 | 56.026 | 1.963 |
| 85000002 | 76.395 | 62.776 | 1.217 |
| 85000003 | 70.485 | 64.122 | 1.099 |
| 85000004 | 73.664 | 69.859 | 1.054 |
| 85000005 | 74.060 | 62.772 | 1.180 |

中位时间74.060→62.776s，描述性比值1.180×，耗时减少15.24%。
五个种子均有正收益，但幅度变化较大；按种子对bootstrap的探索性95%区间
约[1.054,1.963]，少种子区间不能覆盖温度/频率漂移等系统误差。
同seed不是同shower树，尚不宣布高能恢复或广泛硬件适配完成。

10次运行的参数、初级、环境、磁场常数、PROPOSAL/aux哈希和radio配置匹配；
二进制SHA核验一致。70份Parquet逐批检查无空值/非有限值；事件关闭、profile
非负、队列清空、native反解失败与定点溢出为0。单CUDA实际host_threads=1、
OpenMP未启用；双端20线程，不能把相同环境变量误报成单CUDA也用了20核。
局部能量账本仍非完整强子shower账本，本轮不证明全物理或能量闭合。

数据检查与时间图：
`D:/CorsikaData/corsika_validation_results/beta5_adaptive_v7_gpu_continuation_20260912/run/performance_5seeds`。
完整汇总脚本为`validation/accelerator/accept_adaptive_single_dual_pilot.py`；
`summarize_adaptive_service_pilot.py`另提供小型资源/吞吐汇总，不控制任何任务。
本地v8已在这轮完成后开始独立构建，未污染上述五组计时。

## 验收状态

实验策略，不更改默认 legacy、单 CUDA、单 OpenMP、标量 PROPOSAL 或山体流程。
没有替换生产 install；物理 kernel、PROPOSAL 插值、cut、thinning 和磁偏转常数不变。
**尚未完成真实设备性能验收，不能声称快于最有效单端。**

本地 RTX 4060 正在运行冻结 v5 的 100 PeV 事例，不受主工作树 v7 影响。
PSR 独立 v7 Release 构建和三项主机调度测试通过。PSR 排队时显卡有另一个
山体工作流（启动时 PID 200241，创建时间 1789147464.34），v7 实测通过
持久服务等待整个工作流结束且 GPU 连续空闲 10 秒，不停止该任务。
01:45前，单端N=2的11类数组及CUDA/OpenMP trace与v6一致，真实EM+radio
N=2和N=32通过，已开始100TeV计时。完整短测尚未结束。

## 为什么继续优化

PSR T400＋130核，100 TeV质子、47°/180°、emthin=1e-6、完整射电；
同一版本二进制交替运行纯OpenMP与双端，两种子2026110001/2026110002：

| 版本 | OpenMP中位数/s | 双端中位数/s | 双端相对本版OpenMP |
|---|---:|---:|---:|
| v5 | 54.279 | 76.611 | 慢41.1% |
| v6 | 54.112 | 61.415 | 慢13.5% |

v6两例完整时间为OpenMP50.989/57.234s，双端60.011/62.819s。
seed1的CPU驻留波前从v5约41,782降至v6的13,353；仍有28.031s的
GPU完成到主协调器领取延迟。此延迟与CPU工作重叠，不能直接从总耗时扣除。
v6 seed1双端实际输运102,304,187步，纯OpenMP109,727,231步；动态调度
使shower树不同，不能只凭更短时间认定等工作量加速。

v7针对剩余的**结果领取依赖**：主协调器正在运行一个OpenMP驻留调用时，
GPU第一个调用完成后，不必立即闲等，可以从自己的驻留队列再推进一次。

## 代码与所有权

- `corsika/accelerator/em/detail/IndependentSubshowerPump.hpp`：一个driver packet
  包含1或2个完整的驻留调用结果，数量有界；不递归或无限排队。
- 主协调器在提交前预留光子/轻子两种可能续跑的history区间，检查溢出和重叠。
  未使用区间允许空洞，不能被重新分配；history预留、CPU栈和writer仍只由主线程操作。
- GPU驱动只读取它自己的驻留队列、不可变配置和本次控制器快照；不读取主机waiting
  向量或调用CPU fallback。迁移仍只能在安全边界发生。
- 仅显式adaptive且CPU仍有工作时允许续跑。主机可原子通知需要交还；若CPU已空、
  首次调用无进展或已超过500ms软边界，就不续跑。这不是抢占kernel或粒子cut。
- 完成后主线程按原顺序逐份提交；任何第二份结果异常使整个shower失败，不能把第一份
  成功伪装成整个shower完整。所有输入和完成结果为独占所有权。
- profile与CoREAS/ZHS仍在各端积累，只在shower终止后合并；本轮不改变累积公式。

统计继续分别记录**逻辑驻留调用数**和**driver packet数**，不能通过把两次调用
包装成一次而宣称物理批次减少：

```text
CUDA logical calls = driver packets + autonomous continuations
autonomous continuations <= driver packets
```

`cuda_result_service_delay_ms`为最后一次调用结束到领取的真正driver闲置时间；
新增`cuda_completion_buffer_delay_ms`为各结果在缓存中的等待之和，可能包含第二次
GPU调用的有效工作。原`endpoint_window_overlap_ms`仍是host/driver窗口交集，
不是CUDA kernel活动时间；v7的packet内部转接间隙也在该窗口内。

## 已完成的安全检查

1. 主机测试用同步门闩令CPU调用必须等到GPU第二次调用开始后才能返回，因此
   不能靠主线程顺序收取/重新提交蒙混通过。混合光子/轻子续跑，16,384个终止身份
   不重不漏；提交/回收一致，第二次调用异常会禁止输出完成。
2. ASan、UBSan及泄漏检测通过（PSR独立测试程序）。
3. ThreadSanitizer初次因地址布局报`unexpected memory mapping`，不属于代码测试结果。
   使用`setarch x86_64 -R .../pump-tsan`仅关闭该测试进程的ASLR后测试通过、未报data race；
   未修改系统级ASLR设置。这是host调度测试，不覆盖CUDA设备内部race。
4. 独立CUDA_OPENMP Release构建及`testIndependentSubshowerPump`、
   `testAdaptiveSubshowerControl`、`testCooperativeDriverAffinity`三项CTest通过。
5. 代码复核：结果消费回调只读取本份结果；后端statistics在GPU在途期间有明确禁止门禁，
   profile/radio下载要求队列全部排空。未引入对活跃端点的主机读取。

## 排队的真实设备验收

PSR运行绑定382–511（后130物理核每核一硬件线程），T400预算70%，进程内存
24/32GiB软/硬限制，系统至少32GiB可用。构建已使用128并行，与生产install隔离。

服务：`c8-psr-adaptive-v7-tests-20260912`。
构建：`build/psr-adaptive-v7-gpu-continuation-20260912`。
数据：`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v7_gpu_continuation130_20260912`。
等待状态：构建目录中的`WAIT.json`，服务日志`tests.log`。

排队脚本`validation/accelerator/wait_gpu_idle_then_exec.py`同时核验工作流PID创建时间
和实际GPU compute进程；观察失败不算空闲，不会因断线重启任务。验收脚本还会
在每次运行前/运行中抽样检查外部GPU进程，有竞争的时间不进入性能报告。

顺序：单端固定种子N=2数组/trace与v6比较、无效CLI门禁、真实EM+radio N=2、
N=32生命周期，然后同二进制两种子100TeV交替计时。报告必须检查实际触发自主续跑，
按逻辑调用/物理步数/驻留波前及CPU/GPU工作量对账。先通过短测，再决定100PeV长测。
以上仍不替代至少五种子性能复测和500例物理统计，不更改生产推荐。

### PSR首例结果及竞争剔除

seed2026110001：单OpenMP **51.796s**，v7双端 **60.442s**，暂无净收益。
实际触发176次GPU自主续跑，371次逻辑GPU调用由195个driver packet提交；
CPU441次调用，GPU最后完成到领取等待22.526s，低于v6首例的28.031s。
缓存结果等待48.993s包含GPU第二次调用，不能误报成新增GPU闲置。

v7实际总输运106,904,987步，其中GPU5,021,389步（4.70%），
CPU101,883,598步。CPU物理波前11,002，GPU3,555；纯OpenMP为9,572波前。
说明：减少调度job和改善GPU自主续跑，**还没有消除分端后增加的物理波前/
同步成本**。GPU仍只承担少量工作，不能仅凭CPU/GPU利用率宣称恢复单端性能。
过程/能量混合不同，步数归一化只能作描述性诊断。

第二例期间另一山体GPU工作流启动。模拟本身exit=0，但独占检查标记
`performance_valid=false`，该时间不参与加速统计，后续对照未启动。
PSR测试服务退出1是性能污染门禁，不是物理求解失败。未停止其他工作流，
未反复自动重跑，当前不生成两种子性能结论。原始标记和首例JSON已同步到
本地`D:/CorsikaData/corsika_validation_results/psr_t400_adaptive_v7_gpu_continuation130_20260912`。

## 本地后续净收益对照

已冻结v7覆盖文件到`build/adaptive-v7-gpu-continuation-20260912/overlay`。
`c8-adaptive-v7-build-ready-20260912`服务等待当前v5 UHE工作流的PID/创建时间、
成功完成标记及GPU空闲，再复制已验证的独立源码并以2并行构建；不在UHE
计时期间编译。构建内存上限8GiB，启动前至少12GiB系统可用。

`c8-adaptive-v7-pilots-final-20260912`等待该构建完成，再做单端回归、
N=32、真实EM+radio N=2以及五种子Fe100TeV的**单CUDA vs adaptive20**交替测量。
只测短样本，暂不自动再启动100PeV；先根据实测净收益决定长测。
每次独立进程，完整射电、原物理参数、70%显存，至少4GiB系统可用。
数据在`D:/CorsikaData/corsika_validation_results/beta5_adaptive_v7_gpu_continuation_20260912/run`。

启动器现支持`--timing-modes cuda adaptive --timing-seeds 5`，不再仅用旧双端
作为速度分母。首次排队发现覆盖文件夹不是完整源码，已在任何构建/模拟启动前
停止该排队服务，加入显式`--source-root`后重新排队；仅等待状态记录保存在
`queue_attempt_before_source_root_fix`，不是失败的shower样本。
另修复WSL非交互systemd环境缺少`nvidia-smi`搜索路径的问题：等待器可显式
发现`/usr/lib/wsl/lib/nvidia-smi`。限制PATH后的探针已正确看到正在运行的v5
GPU进程；仅重建尚未执行任何构建/模拟的排队服务，v5进程未中断。
