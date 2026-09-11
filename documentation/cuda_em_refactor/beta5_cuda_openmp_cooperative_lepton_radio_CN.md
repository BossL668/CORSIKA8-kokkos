# beta5 CUDA＋OpenMP：轻子状态机与整数射电合并

2026-09-10，阶段 B2。承接[光子状态机](beta5_cuda_openmp_cooperative_photon_front_CN.md)。

**本页记录底层实现和隔离验收，不是完整双端空气 shower 的发布记录。500 例 Fe 尚未启动，不能据此给出 shower 加速比。**

## 1. 本轮改动

| 文件 | 功能 |
|---|---|
| `corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp` | 从原同步函数抽出 `enqueueResidentLeptonFront` 和 `enqueueResidentLeptonAccumulation`；同步函数仍调用相同物理 kernel |
| `corsika/accelerator/em/kokkos/KokkosLeptonFrontSubmission.hpp` | 电子、正电子、μ± 的可恢复前沿；控制量异步回传、驻留续推、有界工作区、恰好一次领取和失败状态 |
| `corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp` | 事件结束时导出原始整数 profile、尺度和计数；导出后禁止再次积累，reset 后重新开放 |
| `corsika/accelerator/em/detail/CooperativeProfileMerge.hpp` | 校验事件身份、网格、尺度，溢出检查后合并八类 profile 整数及计数，最后转换为原输出类型 |
| `corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp` | 原始 CoREAS/ZHS 整数导出；OpenMP 线程局部积累仅归并一次；传输异常先排空本执行端再释放目标缓冲 |
| `corsika/accelerator/radio/detail/CooperativeRadioMerge.hpp` | 校验天线、时间网格、传播表和定点尺度，事务式合并两个执行端；禁止部分成功、重复提交和整数溢出 |

新实现不引入另外一套物理公式，不改变 PROPOSAL calculator、cut、thinning、磁传播或随机数生成算法。没有修改山体应用，也没有更新正常 build/install 或生产二进制。

## 2. 轻子前沿如何推进

```text
prepare：分配有界工作区、上传新输入、预留射电工作区
    ↓
Ready → submit → WaitingForPhysics
                         ↓ poll：仅查询本 CUDA stream 的完成事件
                  AccumulationReady
                         ↓ submitAccumulation
                  WaitingForAccumulation
                         ↓ poll：profile、统计和射电全部完成
                    ResultsReady
                         ↓ consume：单一协调器恰好领取一次
                     Committed
                       ├─ continueResident → 在原执行端交换粒子双缓冲
                       └─ takeRemaining → 领取容量/history 检查点的剩余粒子
```

物理阶段包含过程选择、连续损失、传播限制、Molière 散射、vertex 选择、末态及稳定 scan/materialization。积累阶段直接使用仍在本执行端的轻子轨迹计算 profile、CoREAS 和 ZHS，不为射电逐步下载 GPU 轨迹。

CUDA 的两个 submit 都在排入 kernel 和预分配 pinned control 拷贝后返回。OpenMP 阶段同步执行，由同一个协调线程在 CUDA 工作期间执行；没有两个普通主机线程同时操作 Kokkos。`continueResident` 不下载再上传粒子队列。达到源粒子容量或 history 预留上限时不重抽随机数，而是保留未处理前沿，等待上层重新分批。

消费者不能在回调中再次领取、重新 prepare 或保存借用的输出 View。异常后 frame 进入 Failed；析构只排空本执行实例。已完成但尚未领取的剩余粒子禁止被下一批覆盖。`secondaryHistoryIdsUsed()` 返回实际消耗的子代身份数量，供协调器审计，不回收已经预留的身份范围。

**这里还不是完整 backend/session。** frame 的调用者仍须管理跨光子/轻子队列、CPU fallback、全局 history 预留、输出 writer 和整个 shower 的完成条件。

## 3. 射电和 profile 合并的语义

两端最终导出相同配置下的整数账本，不能把中间累计快照重复提交：

```text
CUDA 最终整数账本 ───┐
                    ├─ 身份/网格/尺度检查 → checked integer sum → 一次浮点转换
OpenMP 最终整数账本 ─┘
```

- CoREAS 的整数表示场；ZHS 的整数表示势。**先合并 ZHS 势，保留原 `RadioProcess::endOfShower()` 的后续求导顺序。**
- 两端的 observer 位置、采样率、起始时间、bin 数、折射传播数据和固定点尺度必须一致；不是仅比较数组长度。
- 任何数组或计数溢出都使这次提交整体失败；例如 ZHS 合并失败时，不留下已经加过一次的 CoREAS 数组。
- profile 的粒子计数、沉积、介质静质量等预算项按原尺度合并；计数的和与最大迭代数分别使用求和、取最大值。
- 单端已有的浮点输出入口未改成“双端浮点相加”。通过旧接口/新整数接口的精确对照，确认输出排列和转换尺度未变。

## 4. 隔离测试及实际证据

### 4.1 真实 PROPOSAL 轻子测试

测试构造真实空气介质的 photon/e−/e+/μ−/μ+ calculator，导出 **58 列原生样条**；使用原生辅助缓存，不生成 `.c8emrt`。首次运行补齐了这套测试 cut 配置缺失的 PROPOSAL 自身缓存，随后命中缓存。

对照函数是开始本轮之前冻结的 `runResidentLeptonCascadeBefore`，不是新接口调用两遍。源文件 token 审计确认：

1. 抽出的物理 kernel 计算代码不变；
2. profile/统计/射电积累 kernel 计算代码不变；
3. 原同步包装器除调用抽出函数外不变。

逐字段比较包括粒子状态、history/parent/step、过程/组分、随机数及取数标识、连续损失、末态、fallback、观测和衰变候选。分别对每个执行端比较，不宣称不同硬件之间必然逐位一致。

已覆盖：

- 空输入、1、257、4096 个混合 e±/μ± 输入；能量 cut、10 ms 时间 cut、超能区 fallback 和边界逃逸。
- 257 个输入连续 8 个驻留波前：3737 条步进、1436 条末态记录，与冻结代码一致。
- brems、ionization、Epair；新增低能正电子样本，4096 输入、4 个波前中实际执行湮灭过程，不只测试编译分支。
- 首次相互作用诊断和实际消耗的 history ID 数量与旧路径对照。
- 有真实 profile/radio 时，与冻结的完整同步函数产生的整数数组直接比较；不仅比较两个新接口。
- 重复/过早提交、剩余粒子未领取、history 预算不足、混入光子、重入消费者、消费者异常，以及在两个异步阶段直接析构。
- 相同工作区复用 32 次，分配字节数不增长。**这不是应用 `-N 32` 的内存验收。**

### 4.2 实际重叠

使用同一协调线程，先提交 CUDA 的 16384 个混合轻子，再执行 OpenMP 四线程的 2048 个输入。CUPTI 记录实际 kernel 时间戳，而非用 GPU 利用率推断重叠。

`lepton-test-4` 的交集为 **1.652 ms**，`lepton-test-5` 为 **1.440 ms**，两次丢失 activity record 数均为 0。第二次还要求两个执行端的 CoREAS/ZHS 数组均非零，避免“零信号相等”的无效验收。两端分别产生 6007 / 749 条有效射电轨迹；并发与各自顺序执行的物理记录、profile 和射电整数一致。

![实际轻子 kernel 重叠；不是完整 shower 加速比](../../../build/overlap-probe-20260910/lepton-overlap-figures-1/actual_kernel_overlap.png)

这些批次没有完整强子级联、空气 router 或 shower 输出闭合，**不能从 1.4–1.7 ms 交集推算 Fe 单事件提速**。

### 4.3 积累器验证

在每种执行端上，把 257 / 16385 条规定轨迹拆成两个分组；两个整数账本的合并结果与单组计算完全相同，转换结果与旧浮点下载接口完全一致。大批量与两个半批量跨越 CUDA 的分块投影分支。信号必须非零。

覆盖正负整数溢出、第二个数组失败后的回滚、计数溢出、事件混用、尺度/时间网格不符、重复导出，以及 32 次 reset 后无历史波形残留。诊断的最大值不做加法。

## 5. 测试目录和复现

全部使用项目内隔离目录，正常安装没有替换：

```text
../build/overlap-baseline-20260910/             冻结的旧函数及此前二进制基线
../build/overlap-probe-20260910/                CUDA_OPENMP＋CUPTI
../build/overlap-probe-openmp-20260910/         独立 OpenMP
../build/overlap-probe-cuda-20260910/           独立 CUDA＋Serial
```

在阶段 A/B1 的配置参数上，增加 `-DC8_LEPTON_FRONT_TEST=ON`，使用已有 patched PROPOSAL/CubicInterpolation 的 Conan 依赖描述。冻结目录必须同时包含 `ResidentPhotonCascadeBefore.hpp` 和 `ResidentLeptonCascadeBefore.hpp`。本机隔离链接沿用主项目的系统库搜索目录，避免 Conan Boost 与 Conda sysroot 的 libc 符号版本冲突；不修改 Conan 缓存中的库。

```bash
conda activate corsika_venv
cmake --build ../build/overlap-probe-20260910 \
  --target testKokkosLeptonFrontSubmission testKokkosCooperativeAccumulators -j1
python validation/accelerator/run_overlap_guarded.py \
  --output ../build/overlap-probe-20260910/new-b2-check \
  --timeout 480 --rss-limit-gib 3 -- \
  ctest --test-dir ../build/overlap-probe-20260910 --output-on-failure -j1
```

监视器保留至少 **4 GiB 系统可用内存**，超限只停止它新启动的测试进程组。`lepton-test-4/summary.json` 的进程树峰值 RSS 约 413 MiB，最小系统可用内存约 10.44 GiB；整个样本不是70%显存压力测试。

最终隔离构建/CTest 矩阵（包括最新首次相互作用、history 用量、冻结 profile/radio 和异常清理测试）：

| 构建 | CTest | 结果目录 |
|---|---|---|
| CUDA＋OpenMP | **9/9 通过** | `overlap-probe-20260910/ctest-b2-final/` |
| OpenMP-only | **6/6 通过** | `overlap-probe-openmp-20260910/ctest-b2-provenance/` |
| CUDA＋Serial | **6/6 通过** | `overlap-probe-cuda-20260910/ctest-b2-final/` |

这是测试目标矩阵，**不是三个完整应用都已经通过新协同模式验收**。独立 OpenMP 测试的动态依赖不含 CUDA runtime。组合版最后一次 CTest 中，真实轻子 kernel 的重叠为1.433 ms；进程树峰值RSS约427 MiB。编译使用 Release `-O3 -DNDEBUG`，未启用 fast-math。

三种构建记录的依赖及表身份相同：

```text
PROPOSAL 7.6.2; CubicInterpolation 0.1.5
native SHA-256: 14cfaa2f478c8f8e16b1f5a668d9afd5380b87bfb9b7e2dcefe6f355095371cc
auxiliary SHA-256: 916bfc0452b34efd950c630ee30e4dbc9730568ac392259353fc355f8d82388e
```

旧安装的组合 `c8_air_shower` 二进制 SHA-256 仍为 `af247c1ece6c346e97758ef3ff9913fd7e1baa9b60220b733a8aae32a9677b88`，未被隔离构建替换。

失败的实验日志保留：首次冷缓存准备超时；测试 fixture 缺少 photon LPM 导出；过高能量与时间 cut 同时生效导致测试 profile 预算溢出。前两项补齐测试准备后复测；后一项将“超表能区 fallback”和“正常能量时间 cut”分开，**没有关闭生产溢出检查或放宽物理容差**。

## 6. 生产暂停及 Fe 500 门禁

按用户要求停止了 `c8-beta5-uhe100-cuda-persistent.service`。停止时原生产已完成 **41/100**；第42例（index41，seed2026110042）记为 interrupted，可按原种子续跑。已完成数据、旧二进制、manifest 均保留。systemd 的 failed/exit130 是本次人为停止记录，不是新的物理失败；MainPID 为0，Restart=no。

计划采用已有 Fe 500 对照的参数和种子：Fe56，**整核总能量100 TeV**，垂直，emthin1e-6，种子85000001–85000500，其余条件核对原 manifest。不得把核子能量误当成整核能量，也不得通过改变 max-weight 或时间窗制造性能增益。

启动前仍必须完成：

1. 把 photon/lepton 前沿接入真实 backend/session 和空气 router，包括跨 PID 驻留队列、全局 history 预留和 CPU fallback；不能每轮重新分配整套 workspace。
2. 动态分派、每批恰好一次提交、两端 profile/radio 最终合并与 output writer 接线；只有所有队列和在途工作为空才关闭 shower。
3. 完整应用的 CUDA/OpenMP/协同小事件、`N=1/2/32`、能量预算、山体回归，以及同机单 CUDA 性能基线。
4. 通过后才启动500例双端 Fe；记录真实重叠、显存/RSS、波前规模、单事件时间及相对于单 CUDA 的净增益。

**本轮未启动500例、未修改生产默认后端、未提交或推送 GitHub。** 现有应用入口仍是 CUDA/OpenMP 二选一。此前 Fe CUDA 中位约67.2秒来自不同版本/运行条件，不能直接作为新双端实现的受控测速分母。
