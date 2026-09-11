# beta5：空气 shower 的 CUDA＋OpenMP 双端后端

2026-09-10。承接[轻子前沿与整数射电合并](beta5_cuda_openmp_cooperative_lepton_radio_CN.md)。

## 状态与边界

本阶段把双端协调器接入 `c8_air_shower`，不再只是独立的重叠探针。仍是显式选择的实验后端：没有修改默认后端、正常 install、生产归档二进制或山体应用。**接通功能不等于通过 500 例统计及性能发布门禁。**

```text
c8_air_shower / HybridCascade / PhysicalAcceleratedEmRouter
  单一主线程：CPU 栈、强子、history 预留、fallback、writer
      │
      └─ KokkosEmBackend → CooperativeBackend
          ├─ CUDA instance：原驻留光子/轻子队列、profile、CoREAS/ZHS
          │    提交计算和控制量拷贝 → 等待本流事件时让出主机控制
          └─ OpenMP instance：在等待窗口内执行有界短批次
               │
               └─ 本端次级队列、profile、CoREAS/ZHS
      ↓ 全部粒子及 fallback 排空
  两端最终整数账本 → 身份/尺度/溢出检查 → 一次转换 → 原输出流程
```

### 为什么完整后端采用等待点续执行

已有独立 `PhotonFrontSubmission` / `LeptonFrontSubmission` 验证了可恢复前沿。本阶段保留完整驻留循环及其局部状态，在原来的 **3 个光子等待点、4 个轻子等待点**接入 `ResidentExecutionWait`。CUDA stream event 未完成时，同一协调线程推进一个 OpenMP 前沿；等待结束后原调用栈继续执行。它是有栈的主机续执行，不是第二个线程调用 Kokkos，也不是把 GPU 粒子队列下载成 CPU 队列。

这样保留了原有跨物种驻留队列、workspace、fallback 收集以及射电/统计顺序。没有重写物理 kernel；没有 callback 的单端路径仍调用原 `execution.fence(label)`。为回避显式状态机遗漏完整管线功能的风险，这里没有直接用独立 front helper 拼装另一套 shower 算法。

## 文件职责

| 文件（相对于源码根目录） | 职责 |
|---|---|
| `src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp` | 两实例生命周期、批次划分、互不重叠的 history 范围、短批次进度、结果提交和事件重置 |
| `corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp` | 本 CUDA 流完成事件轮询和 OpenMP 进度回调；异常先排空本流目标，再允许临时缓冲析构 |
| `src/accelerator/em/kokkos/KokkosBackendInstance.inl` | 借用同一 runtime 的实例、预分配 CPU workspace、有界 pending 前缀领取、原始整数输出 |
| `corsika/accelerator/em/detail/CooperativeBackendMerge.hpp` | 明确逐字段合并结果和统计，计数检查溢出，峰值取最大值，表/事件身份一致性门禁 |
| `corsika/accelerator/em/detail/CooperativeProfileMerge.hpp` | 纵向与能量账本整数合并，禁止重复提交 |
| `corsika/accelerator/radio/detail/CooperativeRadioMerge.hpp` | CoREAS/ZHS 整数与元数据事务式合并；ZHS 后续求导仍由原 RadioProcess 完成 |
| `tests/accelerator/testKokkosCooperativeBackend.cpp` | 完整后端、真实 PROPOSAL、混合轻子、跨物种队列、双端射电、连续事件和实际重叠测试 |
| `validation/accelerator/audit_cooperative_wait_extraction.py` | 对冻结源码做 token 审计，仅允许列明的等待点与接口变化 |

## 当前调度与资源规则

- GPU 原有自动显存容量策略不变，默认显存比例 70%；优先保留至少 `gpu-min-batch=4096` 的有效 GPU 批次。
- OpenMP 队列独立有界，前沿最多 2048 个输入；初始分享目标 256，根据已测前沿时间移动平均在 256–2048 调整，目标约 2 ms。**2 ms 不是硬实时上界**，大批射电会产生更长时间片，metadata 记录超时片和最大耗时。
- OpenMP 次级留在本端 pending 队列，CUDA 次级留在 GPU。仅在未提交的安全边界允许 GPU pending 的有限前缀迁移，同时保留 GPU 最小批次。迁移保留完整粒子状态、history、parent 和 step。
- 两端各用协调器预留的一段 history；OpenMP 后续时间片不能顺便消费预留范围之外的 pending 输入。保留未用身份空洞，不回收，不依赖线程完成顺序分配身份。
- 不在 OpenMP kernel 内调用标量 PROPOSAL/强子生成器或 writer；fallback 仍返回唯一协调器，不能由两端并发改 CPU 栈。
- 输出封口后禁止再计算；双端异常使当前事件失败，不静默切成单端继续。下一 shower 开始前要求旧队列已清空。
- 协同模式强子 worker 固定为 1；未显式输入该选项时应用设置为 1。单端已有默认值保持不变。

当前策略是 **GPU 优先＋CPU 短批次自适应**，还没有实现完整的双端吞吐/迁移成本预测，也没有把独立 schedule journal 工具接成应用级调度记录/回放 CLI。不能宣称原计划所有高级调度功能均已完成。

## 运行与构建

只在 `CORSIKA_KOKKOS_BACKEND=CUDA_OPENMP` 组合构建中使用。仍是一份共享物理源码分别实例化两端。普通单 CUDA、OpenMP、HIP、SYCL 构建不能请求协同后端。

从源码目录、已激活 `corsika_venv` 的 shell 中使用隔离测试二进制，例如：

```bash
../build/cooperative-air-20260910/applications/c8_air_shower \
  --em-backend kokkos --radio-backend kokkos \
  --kokkos-execution cuda-openmp --kokkos-num-threads 16 \
  -Z 26 -A 56 -E 100000 --emthin 1e-6 \
  --antenna-file /path/to/antennas.txt -f /path/to/new-output
```

复用项目组合构建依赖时，CMake 的 CUDA/PROPOSAL/Kokkos 参数和普通组合构建相同；本阶段单独使用 `build/cooperative-air-20260910`，没有安装覆盖生产程序。`C8_COOPERATIVE_BACKEND_CUPTI=ON` 只给隔离的 `testKokkosCooperativeBackend` 加 CUPTI，主程序不依赖 CUPTI；当前该诊断使用 CUDA 12 ActivityKernel9。

## 已获得的底层证据

- 完整 CUDA＋OpenMP Release 后端及空气应用构建通过。
- 冻结源码 token 审计通过：原光子/轻子物理和积累 kernel 不变，完整控制循环仅增加列明的可让出等待点。
- 真实 PROPOSAL 58 列测试连续两事件成功：分别为 62,851 / 62,746 个 profile steps；两个事件均有 CUDA 和 OpenMP 输入、非零 CoREAS 与 ZHS 输出。没有重复 terminal history、意外 spill、队列不排空或重复输出提交。
- 两事件的物理工作区合计均为 **98,548,304 bytes**，测试进程峰值 RSS 约 588 MiB。它不是一般 `-N 32` 或高能长期内存验收。
- CUPTI 同一时间基准记录实际 kernel 与首个 OpenMP slice 的交集 **457,342 ns**，丢失 activity record 为 0。该数值是本测试单个窗口的交集，不是整场加速比。
- 完整空气应用 `1 GeV photon, -N 2` 成功，约 8.84 s；此低能样本主要检查生命周期，粒子不足时允许只用 GPU，不能据此证明双端性能收益。

日志在项目 `build/cooperative-air-20260910` 的 `build-test-air-4`、`backend-test-run2`、`air-cooperative-photon2`；输出/图在 D 盘 `CorsikaData/corsika_validation_results/beta5_cooperative_air_acceptance_20260910_v1`。

`cooperative.openmp_while_cuda_pending_ms` 只表示 OpenMP 工作发生在待完成 CUDA 提交窗口内，**不等于实际 kernel 交集时间**；实际重叠需 CUPTI。合并的 phase 时间是两端工作量之和，不是可相加得出的 wall time；传输统计沿用各实例的既有计账，不能都解释为 PCIe 流量。

## 尚待完成的发布门禁

完整调度记录回放、500 例物理统计、山体行为回归，以及相同机器交替五种子的全射电/无射电性能验收。未通过之前，不能称为生产替代后端，也不改变单 CUDA 的生产推荐。

## 16 线程与 Fe500 验收进展（2026-09-10 晚）

按用户最新要求，保留修复后的标量单位制磁偏转常数
`0.2997925146834389 GeV/(T m)`，不再用旧安装二进制的常数作为逐位验收目标。
代码位置为 `corsika/accelerator/ScalarPhysicalConstants.hpp`，编译期对照标量单位系统。
以下统计使用新的归档二进制；旧 CPU/单 CUDA 样本只作为带有明确版本来源的系综参考。

- 真实 PROPOSAL 完整双端测试重新使用 **16 线程**通过，两事件均产生双端工作与 CoREAS/ZHS，工作区总量均为98,548,304bytes。
- 完整空气应用 `1GeV photon, -N32, OpenMP16`：32/32完成；峰值RSS约795MiB。CPU工作区18,892,272bytes保持不变，GPU小工作区扩容后在后16例稳定。仅证明这组生命周期测试，不声称任意高能/任意N都无泄漏。
- 隔离构建中的 `c8_mountain_neutrino` 构建通过；本轮没有修改山体应用或模型，仍需独立行为回归。
- Fe56单例试跑：当前单CUDA约116.55s，协同16线程约75.41s；后者峰值RSS约1.08GiB、显存约5.3GiB。两条shower的物理工作量不同，不能据此宣布稳定1.55倍加速。
- 小样本允许GPU工作不足时OpenMP空闲；Fe试跑实际两端均有工作。`openmp_while_cuda_pending_ms`不等于CUPTI实际重叠时间。

正式500例使用种子85000001–85000500，每次一个新进程 `-N1`，不混用试跑输出：

```text
Fe56，核总能量100000GeV，theta0，phi0
emthin1e-6，max-weight0（与已有样本一致）
IGRF14/2027，同一天线、观测高度、cuts、SIBYLL-2.3d与FLUKA
proposal-native，CUDA+OpenMP16，全CoREAS/ZHS，GPU目标70%，min-batch4096
```

二进制SHA256为 `5561d1dd51c08bedab59d441fd87d3917fcb394f9e209af839b920df600b0d58`。
PROPOSAL/辅助表及SIBYLL/FLUKA动态库哈希已与原Fe参考匹配，不重新制表。
`CORSIKA_DATA`使用当前编译默认的 `modules/data`；首次服务启动误指向不含缓存的
`resources`，已在完成任何事件前中止，全部记录归档到 `preflight_cache_path_interrupted`，不计入样本或时间。

正式目录位于D盘：
`CorsikaData/corsika_validation_results/beta5_cooperative16_Fe56_100TeV_500_emthin1e-6_igrf14_2027_v1`。
服务为 `c8-beta5-cooperative16-fe500-20260910-run2.service`；不依赖临时终端。
每例验收后才增加complete计数，失败种子保留，不换种子补数；仅运行超时允许原种子重试一次。
应用RSS上限3GiB，系统可用内存至少4GiB，另有服务4GiB内存硬上限、禁用该服务swap、禁止异常自动重启。

`run_cooperative_fe_campaign.py`保存每线程CPU ticks、RSS与设备级显存/利用率；6项管理器测试通过。
`analyze_cooperative_fe_campaign.py`复用本地CPU500和单CUDA500的真实紧凑缓存，比较全部纵向组分、
沉积和地面分布、标量、Ex′/Ey′/Ez′平均波形及时间。先逐事件调用原pulse_analysis投影再累计误差，
不旋转已有三分量的边际误差。原参考波形只有均值/SE缓存，不能宣称执行了完整事件级波形bootstrap。
所有500例完成前，任何小样本出图仅标记为分析流程诊断，不能当作正式验收通过。

另注意：当前自动max-weight为0.05，不能薄化单位权重粒子，但本轮按参考条件保留；
历史`energy_closure_fraction`为沉积加地面EM能量份额，不是全shower能量守恒。
加速器自身账本`complete_coverage=false`也不得被描述为全局能量闭合已通过。
