# CPU 优先 v3：保护 OpenMP 波前，再分配辅助 GPU 工作

本轮仅修改显式 `--kokkos-execution openmp-cuda`。目标是先保住纯 OpenMP
的有效吞吐，再利用 GPU 处理富余任务；不是为了同时显示高占用而拆分 CPU
的有效批量。原 `cuda-openmp`、单 CUDA、单 OpenMP、标量 PROPOSAL、山体
程序及物理 kernel 不在本轮修改范围内。生产 install 未替换。

## 已发现的差异

PSR 上此前 CPU 优先 v2 的五例 Fe 100 TeV 进程中位时间为 240.571 s；
已完成的两个同种子纯 OpenMP 对照为 113.761、77.674 s。这不是性能达标。
固定输入的后端隔离测试没有复现同量级的退化，不能把原因简单归结为
“初始化了 CUDA 所以 OpenMP 变慢”或调度公式计算太多。

代码层面存在两处具体差异：

1. CPU 只保留最小有效批量，而不是完整 arena；辅助端可能拆小本可完整
   送给 CPU 的批量。
2. CPU 优先没有使用已有的指定 fallback 延迟合批接口，会立即回到标量栈、
   处理一个过程并重新启动少量产物的波前。实际记录中的标量时间和检查点
   数量明显增加。不同种子的波前变化并不完全一致，不能据此断言全部时间
   回退已经找到唯一原因。

隔离诊断报告位于 D 盘 `CorsikaData/corsika_validation_results/`
`cpu_primary_backend_isolation_20260912/ISOLATION_REPORT_CN.md`。

## 本轮实现

- 对光子、轻子分别保护 OpenMP 原有的完整 arena。新工作只有在保护量之外
  有足够富余时才分给 GPU。
- 新 GPU 领取量至少为 `min(GPU arena, gpu-min-batch)`。默认 min-batch
  仍为 4096；已有驻留粒子、尾部以及有界存储压力 spill 必须允许排空，
  不受这个“新任务领取”门槛阻塞。
- CPU 待接收队列满时，未接受的输入后缀仍归调用方持有，不绕过份额规则
  静默改派 GPU，更不能丢弃。
- CPU 优先复用现有 4096 条指定 fallback 合批，以及最终空前沿的小批量
  flush。仍只有协调器调用 CPU 物理和操作共享栈；每条结果只提交一次。
- 不新增预测控制器。不修改粒子 history 租约、随机数、物理公式、表、cut、
  thinning、max-weight、散射或射电算法。
- CPU 轻子保留 1024-wave 上限，GPU 辅助端暂保留 8-wave 上限；没有在同一
  候选中顺带增加无限自主续跑。若 CPU 吞吐恢复后仍存在辅助 GPU 服务等待，
  再单独测量并处理，而不是延长 GPU 尾部来换取表面占用。

`KokkosShowerReport.hpp` 使用策略标识 `cpu-primary-v3-protected-front`，
并记录完整主端保留、有效辅助领取及 fallback 合批语义。

## 隔离与验收

构建从两台机器各自冻结的 v2 source 复制，严格只覆盖四个调度/报告文件和
两个测试文件。完整源文件 SHA 清单和边界审计随独立 build 保存，不同步
工作树中无关的山体开发修改。复用已构建的物理模型和缓存，不重新制表。

测试安排：

- 主机队列测试：混合 PID、保护门槛两侧、GPU arena 小于 min-batch、满队列
  背压、受阻 GPU 下 CPU 独立推进、身份无丢失或重复。
- ASan/UBSan 与泄漏检查；真实双端 EM/CoREAS/ZHS fixture 和能量闭合。
- 重构前后 CPU/单 CUDA/单 OpenMP 的 N=2 数组、非计时 metadata、decision
  trace 一致；N=32 生命周期和有界内存检查。
- 实际 shower 中指定 fallback 的 queued/flushed 数一致、4096 条整批及
  最终不足一批的 flush 正确，完成时无未提交队列或定点累积溢出。
- PSR 130 线程绑定逻辑 CPU 382–511，以同一新二进制的纯 OpenMP 为对照，
  固定五个 Fe 100 TeV 种子交替运行。两模式使用相同亲和性和物理参数；
  不用改变薄化或权重获得提速。
- 记录实际输运步、CPU/GPU 批次、回退、标量时间、GPU 服务等待、RSS、显存
  及监控完整性。不把等待时段直接当成可实现收益，也不把少量样本的
  “无显著差异”当作 1% 统计等价。

当前主机队列与 sanitizer 测试通过。重复完整旧套件时观察到一次已有的
adaptive 软时限与 handoff 竞争导致的测试断言失败；该旧 fixture 保持不变，
未将其改写混入本候选。真实后端回归和性能测量尚在进行，不能宣称已经
恢复纯 OpenMP 性能或获得 GPU 净收益。

## 隔离构建与首轮真实门禁

两机 Release 和三项调度/亲和性 CTest 均通过。候选目录为本地
`build/cpu-priority-v3-protected-front-a2-20260912`、PSR
`build/psr-cpu-priority-v3-protected-front-a2-20260912`。首次配置遗漏了
冻结 Pythia 安装的非标准 include/lib 布局，配置即终止；补齐原缓存路径
后重新在独立 a2 目录构建，没有更换物理模型或重制表。

PSR CPU PROPOSAL、单 CUDA、单 OpenMP 的 N=2 物理数组及非计时 metadata
均与冻结 v2 一致，两条 Kokkos decision trace 逐字节一致。两种双端真实
EM/CoREAS/ZHS fixture 和 N=32 通过。后两组进程时间分别 29.594、28.314 s，
RSS 峰值均约 0.807 GiB。这是当前固定规模生命周期诊断，不是大样本或
旧版本性能对照；本轮显式使用 `OMP_PROC_BIND=spread`、`OMP_PLACES=threads`，
不能将与历史不同绑定的时间差全部归功于调度修改。

正式 Fe 小样本计时的纯 OpenMP 与 CPU 优先双端使用上述同一绑定、同一
新二进制和同种子，先后顺序逐种子交换。本地三条单端 N=2 回归也通过，
其余测试继续。结果目录：

- PSR：`/data/yhlu/CorsikaData/corsika_validation_results/psr_cpu_priority_v3_protected_front_130_20260912`
- 本地 D 盘：`CorsikaData/corsika_validation_results/beta5_cpu_priority_v3_protected_front_local20_20260912`

各目录 `STATUS.json` 是实时进度，`CORRECTNESS_GATES.json` 仅表示小规模
正确性门禁。性能尚未通过前不更新生产推荐或安装。

## 五种子最终结论：v3 未通过性能门禁

PSR 五对 Fe 100 TeV 全部完成，重新读取十例物理输出通过有限值、空队列、
fallback 全部提交及定点无溢出检查。监控十例均完整。进程中位时间：

| 同机、同二进制、同130线程绑定 | 中位时间/s | 平均时间/s |
|---|---:|---:|
| 纯 OpenMP | 45.528 | 46.208 |
| CPU-primary v3 | 51.144 | 54.741 |

双端中位时间仍增加约12.3%，不能发布为“不损失 OpenMP 性能”的修复。
v3 指定 fallback 每例约5600条，均以4096及后续部分批量完成；标量时间
约0.8s，旧的逐条重入问题已消除，但它不是剩余回退的全部原因。

双端 CPU 平均约107–111核当量，GPU 平均约19%–29%，但GPU仅完成约
2%–3%的输运步，结果领取累计延迟每例约19–20s。CPU完成的步数虽略少，
其端点调用吞吐仍低于纯OpenMP。这些调用计时含射电和同步，不等于纯kernel。

下一候选 v4 单独冻结：主端采用与 standalone 一致的 resident-first 取数；
辅助GPU以最多4次固定安全调用的小包自主推进。两者均不缩短CPU epoch，
不引入新的自适应时间控制器。v3数据完整保留，不用更快的后续数据替换。

本地四例回归也已完成；两例单CUDA Fe各9份物理数组及非计时metadata与
冻结v2完全一致。动态GPU优先的两例不用于认证3%性能门禁。两机N=32均
已重新验证表/辅助缓存哈希、backend复用和工作区大小稳定。

详细诊断与时间图位于本地：
`CorsikaData/corsika_validation_results/psr_cpu_priority_v3_protected_front_130_20260912_report/`
下的 `diagnosis/REPORT_CN.md`、`performance/Fe100TeV_timing.png`。
服务器同名非 `_report` 目录保留完整原始输出。完整强子能量账本覆盖不足
的标记继续保留，本轮没有据此宣称完整能量闭合或500例统计一致性。
