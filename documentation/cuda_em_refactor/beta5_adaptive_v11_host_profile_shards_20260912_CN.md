# v11：自适应双端的 OpenMP 整数 profile 分片

2026-09-12。**实现及验收中，不是性能通过公告；生产默认不变。**

## 依据与目标

PSR 的 v10 五种子 100 TeV 质子测试中，130 线程单 OpenMP 中位
50.810 s，T400＋130 线程双端 62.081 s，双端仍慢约 22.2%。
Kokkos host-call 采样指出，轻子输运统计的融合 kernel 中包含 profile
整数累积，多个线程争用同一组计数器及 bin。工具本身会扰动调度，
因此它只定位热点，不提供可直接扣除的生产耗时。

复用实际 profile 公式的固定输入微基准中，增加独立累积器可以降低
争用，整数输出不变。**该结果不等于整场 shower 加速**；v11 将这一
候选接入真实后端，再验证正确性、生命周期和端到端时间。

## 实现边界

- 仅 `--kokkos-execution cuda-openmp --kokkos-cooperative-policy adaptive`
  的 OpenMP 端启用；单 CUDA、单 OpenMP、旧双端和山体不主动启用。
- 不改变输运公式、粒子身份、RNG、过程选择、PROPOSAL 表、cut、thinning
  或射电算法。不改变 GPU 物理 kernel。
- `HostProfileShards.hpp` 保存主机专属直方图和计数器，调用原有
  `accumulateLeptonProfileStep`；仍使用原 checked atomic，内存不足时
  多个线程共享较少分片也是安全的，不依赖“恰好每线程一片”。
- `HostShardedLeptonProfileFunctor` 继承原融合 reducer 的 init/join/final。
  Molière Newton 迭代和最大迭代仍按原波前归约写入 canonical 计数器，
  没有在分片重复计入。光子及末态累积继续使用 canonical ledger。
- 最终输出顺序为：canonical＋所有主机分片的 checked 整数合并，随后
  与 CUDA 整数快照合并，最后一次浮点转换。任何溢出/无效记录使事件失败；
  不输出部分合并结果，不静默继续。

## 内存和复用

当前候选按实际线程数及剩余后端内存预算选择分片，不再强制2的幂，最多256片，
常驻分片数据硬上限 **16 MiB**。这不是每线程一套粒子栈/物理表/射电缓存。
采用16 MiB是为了覆盖常见纵向 bin 配置、130线程各有一片的情况；
更大的 profile 自动减少分片，预算连2片都不足则使用原 canonical 路径。

首轮归档候选采用2的幂。在实际倾斜100TeV事例中，profile更大，只能选128片，
无法消除130线程中两对线程的争用。后续 `exact-r2` 候选按实际线程数选择，
在预算足够时直接用130片；若预算确实不足，仍保留checked atomic共享保障。
两候选使用独立源码/二进制/结果目录，以SHA和实际分片数区分，不替换正在测试的首轮。

分片在普通工作区预分配之后、任何输运之前分配，计入后端 retained bytes。
最终合并的临时存储与纵向 bin 数成正比，不与线程数相乘。
`beginShower()` 清零全部 bin/计数器并更新固定点尺度，复用原分配。
已导出或失败的快照禁止再次提交，下一事件必须 reset。

新增 adaptive metadata：`host_profile_shards`、`host_profile_shard_bytes`、
`host_profile_shard_semantics`。单端 YAML 不增加这些字段。

## 验收项与当前状态

1. `testKokkosHostProfileShards`：真实 Kokkos OpenMP reducer，对照原始
   canonical reducer 的每个整数 bin、所有 profile 计数器、归约统计。
   包含混合 e±/μ±、cut/观测条件、source holes、空波前、32事件变更尺度、
   正常及强制碰撞分片；溢出、非法记录、布局/尺度错误和重复提交须拒绝。
2. 真实 PROPOSAL 双端 EM＋CoREAS/ZHS N=2，完整加速器边界能量通量闭合，
   检查分片实际启用且重置后 metadata 不丢失。
3. 单 CPU/CUDA/OpenMP 固定种子输出及 replay trace 不变；实际 shower N=32
   检查完整性和分片容量不增长。
4. 通过以上测试后，在 PSR 后130核（逻辑382–511）做同二进制单 OpenMP 与
   双端各五种子100TeV全射电。整场改善中包含 CPU profile 算法优化，不能
   将全部收益称作“加一张GPU的收益”；需与相同优化的CPU路径做归因诊断。

当前 PSR 独立 build service：`c8-psr-adaptive-v11-build-20260912`；
目录 `build/psr-adaptive-v11-host-profile-shards-20260912`。
从冻结v10源码复制，仅叠加本项相关文件；未混入当前山体工作树的新文件，
未替换 install。测试结果待实际完成后补录，不以编译通过代替验收。

本地100PeV仍运行冻结的 **v9**，并未使用v11；不能用它证明本次优化收益。

### 已获得的真实证据（05:02 CST）

- Release 主程序构建成功。原有独立子 shower、adaptive 控制器及线程亲和性
  3项测试通过。新测试目标在组合 Kokkos 包中同样需要由 NVCC 编译；修正了
  这个测试目标的语言配置及测试用动态库搜索路径，没有放宽任何物理门限。
- `testKokkosHostProfileShards 4`、`130` 均通过。130线程正常选择256片
  （128 bin fixture、2,195,456字节）；强制2片也通过，覆盖原子冲突情况。
  每个模式32事件、变更尺度，所有整数结果完全一致；测试显式确认 CUDA
  未初始化。故这里是 **128-bin固定输入验证，不是整个130线程shower加速**。
- 固定种子 `N=2` 的 scalar PROPOSAL、单 CUDA、单 OpenMP 各自前后11个数组
  一致，metadata 无意外差异，CUDA/OpenMP trace 逐字节一致。
- 真实 PROPOSAL 双端 EM＋全射电 N=2 通过，加速器边界能量通量相对残差
  分别约 `5.53e-9`、`5.45e-9`，含显式返回 CPU 的能量通量；两事件工作区
  字节数相同。它不等价于完整强子 shower 能量账本认证。
- 实际100 GeV photon `N=32` 和五对100TeV全射电计时由服务
  `c8-psr-adaptive-v11-acceptance-20260912` 顺序执行；本次记录时尚未完成。
  结果根目录：`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v11_host_profile_shards130_20260912`。

首对实际计时：纯OpenMP 51.442s、首轮128片双端61.875s，尚无净收益。
此时只完成部分配对，不能据此宣布五对验收结果；也不能把微基准收益外推到整场。

`N=32` 实际 shower 已在50.473s完整完成，分片 metadata/容量跨事件保持一致。
精确分片候选使用独立 `build/psr-adaptive-v11-host-profile-exact-r2-20260912`：
`c8-psr-adaptive-v11-exact-build-20260912` 等首轮计时退出成功后才编译，避免
大规模编译污染 CPU 性能；其后由 `c8-psr-adaptive-v11-exact-acceptance-20260912`
重新执行全部回归和五对计时，不跳过门禁。未完成前不宣称该候选有收益。

### 首轮完整五对结果

首轮服务已正常退出（code 0），10/10计时事件完成；但**性能门禁未通过**：

| seed 后四位 | OpenMP 130 / s | 双端（128片）/ s |
|---|---:|---:|
| 0001 | 51.442 | 61.875 |
| 0002 | 57.363 | 56.534 |
| 0003 | 47.178 | 60.175 |
| 0004 | 53.867 | 70.220 |
| 0005 | 47.614 | 67.376 |
| 中位 | **51.442** | **61.875** |

OpenMP/双端比值0.831，双端中位慢约20.3%。4/5种子仍慢，不能把单独
0002的收益当作整组收益。首轮128片的修改没有解决PSR的净加速问题。
精确130片候选正在独立编译/排队验收，尚无其整场性能结果。

小型报告和图已同步到本地：
`/mnt/d/CorsikaData/corsika_validation_results/psr_t400_adaptive_v11_host_profile_shards130_20260912_summary/`。
其中 `PERFORMANCE_RESULT.json`、`CORRECTNESS_GATES.json` 与监测图可核查；
原始shower仍保留在服务器，不搬回WSL。

### exact-r2完成与进一步诊断

精确130片候选Release构建及4/130线程整数fixture通过，包含正常、2片、3片
碰撞情况各32事件。固定种子单CPU/CUDA/OpenMP N=2全部11数组及原有trace
一致；真实EM/radio N=2和实际photon N=32通过（后者44.736s）。

五对100TeV全射电已完成：单OpenMP中位51.484s、双端61.972s，比值0.831，
双端仍慢20.4%。正确性检查通过不等于性能门禁通过。

采样Kokkos Tools诊断已完成（独立运行，不混入上述时间）：原有每步profile
热点已明显下降，但radio、更多小波前及scalar扩展开销仍在。采样/调度扰动
必须单独报告，不能将host计时项目简单求和或直接从生产时间扣除。
新发现的指定回退合批问题与下一候选见
[v12有界回退说明](beta5_adaptive_v12_coalesced_fallback_20260912_CN.md)。

本地小型结果：
`/mnt/d/CorsikaData/corsika_validation_results/psr_t400_adaptive_v11_host_profile_exact130_r2_20260912_summary/`。
`host_call_diagnosis/`保存采样分析；原始数据仍在PSR。
