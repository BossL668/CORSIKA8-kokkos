# beta5 CUDA＋OpenMP 协同：真实光子波前状态机

日期：2026-09-10。阶段 B1：**光子 producer 与驻留续推已实现并完成隔离测试；完整协同 shower 尚未接通。**

后续进展：见[阶段 B2：轻子状态机、profile 和射电整数合并](beta5_cuda_openmp_cooperative_lepton_radio_CN.md)。本页末尾的待办保留为 B1 时点记录，最新边界以 B2 为准。

## 1. 本轮实现

承接[阶段 A](beta5_cuda_openmp_cooperative_stage_a_CN.md)，从算术探针推进到实际生产光子 kernel：过程选择、传播、稳定 scan、末态及次级生成。

| 文件 | 职责 |
|---|---|
| `corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp` | 抽出 `enqueueResidentPhotonFront()`；原同步 cascade 调用同一实现，保留其原有后处理、计数、profile 和等待顺序 |
| `corsika/accelerator/em/kokkos/KokkosPhotonFrontSubmission.hpp` | 有界私有 workspace，显式 prepare/submit/poll/consume，以及不回传粒子队列的 `continueResident()` |
| `corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp` | 只新增 `next() const` 只读访问重载，供完成后的诊断读取 |
| `tests/accelerator/testKokkosPhotonFrontSubmission.cpp` | 对拆分前源码逐字段比较、连续波前、异常退出、两端真实 EM 重叠测试 |
| `validation/accelerator/audit_photon_front_extraction.py` | 检查迁移的 kernel 计算代码未变；原同步函数除调用抽出函数外未变 |

状态和所有权：

```text
prepare（分配/上传边界）→ Ready → submit → WaitingForControl
                                          ↓ poll
                                     ResultsReady
                                          ↓ consume（恰好一次）
                                      Committed
                                          ↓ continueResident
                        GPU 内交换双缓冲、保留粒子身份 → Ready
```

CUDA `submit()` 在排入原物理 kernel 和 pinned control 拷贝后返回；`poll()` 查询所属 stream 的事件，不调用全局 fence。OpenMP 的同一阶段同步执行，因此协调线程可以先提交 CUDA，再执行短 OpenMP 批次。

`continueResident()` 不下载或重新上传下一轮光子队列。它保留 history/parent/step，按原 scan 给出的子代数量推进 history 起点。表和物理上下文由上层 session 保持生命期。两端 workspace 独立，只有主机协调线程能领取结果。

prepare 仍是显式分配和 staging 边界；目前**不应把每个驻留波前都重新 prepare**。出错析构只等待本执行实例，避免释放仍被 kernel 访问的内存。消费者回调异常后 frame 进入 Failed，不能继续使用。

## 2. 检查范围与结果

这里使用解析、确定性的原生样条测试 fixture：它避免重新生成 PROPOSAL 表，调用的仍是真实生产物理函数，但**不是实际空气介质的 PROPOSAL 精度或 shower 分布 oracle**。辅助 LPM 参数同样仅为诊断 fixture。它们不进入应用或生产表。

比较基准是本轮修改前单独保存的 `runResidentPhotonCascadeBefore()`，不是只用新函数调用两次。比较不使用结构体内存 `memcmp`，避免 padding 伪差异；对物理字段逐个比较，要求相等，不放宽为统计近似。

已通过：

- Compton、光子成对产生、光电效应：过程、目标组分、取数标识、随机数、末态参数、子代身份及运动学、fallback、观测记录逐字段一致。
- 同步包装器与修改前同步代码一致；异步提交/领取与该基准一致。
- 空输入、单粒子、257 和 4096 粒子波前；cut、时间 cut、超能区 fallback、逃逸边界。
- 两个种子各连续 8 轮驻留推进：CUDA/OpenMP 各自与旧同步流程一致；分别有 660、698 条 transport 记录，初级第一次反应记录也一致。
- 同一 workspace 重复 32 次后，所报告的设备分配字节数不增长。4096 粒子案例约 15.1 MiB，不是整次 shower 的显存预算。
- 重复提交、过早领取、重复领取、回调内重新 prepare、history 余量不足被拒绝；在途析构和消费者异常后的失效状态测试通过。
- 实际重叠运行的两端输出，与先后串行执行相同两个批次的输出一致。

上述 32 次是波前复用测试，**不能替代应用 `-N 32` 的内存验收**。

实施中发现并修复了一个状态衔接错误：control 完成后未将队列标记为已同步，重复 upload 会被原有安全检查拒绝。另修正了测试 fixture 的样条数组排列（能量行＋行数×loss 列）；该项仅是新增测试数据错误，没有改动生产样条语义。

## 3. 真实物理 kernel 的重叠证据

使用同一主机线程先提交 CUDA 的 16,384 个光子，再执行 OpenMP 4 线程的短批次。两端在提交前完成 staging；不使用两个普通主机线程同时调用 Kokkos。

初始 256 粒子测试（`photon-test-final/command.log`）记录到约 0.179 ms 的真实交集。但是随后一次全套复测中，CPU 的约 0.198 ms 工作先结束，GPU 约 15 μs 后才开始；该次时间线交集为 **0**，测试明确失败，保留在 `photon-ctest-final/command.log`。不能用“异步提交”或此前成功的单次结果掩盖这种调度空档。

在计划允许的范围内，将诊断默认 CPU 批量改为 2048；原来 256 粒子的实验仍可用测试参数 `--host-batch=256` 重现。这只调整隔离测试，不修改应用默认值或物理条件。

最新 `photon-test-window2048/command.log`：OpenMP 工作约 **0.566 ms**，实际 kernel 交集约 **0.566 ms**，丢失 activity records 为 0。随后组合探针 CTest **7/7** 通过（`photon-ctest-window2048/`）。时间戳使用同一 CUPTI 时钟，没有人为对齐。并发执行前后物理记录保持一致。

![实际光子 producer 重叠，非完整 shower benchmark](../../../build/overlap-probe-20260910/photon-figures-window2048/actual_kernel_overlap.png)

这个小测试只证明真实物理工作可以重叠。生产 GPU 同时有独立任务，不能由该时间线计算加速比，也不能声称 CPU/GPU 已持续满负载。

最新组合测试用 `/usr/bin/time -v` 记录峰值 RSS 为 180,816 KiB（约 177 MiB）。监视器保留至少 4 GiB 系统可用内存，只能停止它新建的测试进程组。现有生产服务和安装二进制没有停止、替换；山体源码没有修改。

独立 CUDA 探针 CTest **4/4**、独立 OpenMP 探针 **4/4** 通过。这是隔离测试目标的构建/运行矩阵，不是完整 air/mountain 程序构建验收。

## 4. 复现与产物

独立测试目录为项目顶层 `build/overlap-probe-20260910`、`build/overlap-probe-openmp-20260910`、`build/overlap-probe-cuda-20260910`。使用已有 Conan/Kokkos 包，不重新制表，不修改正常 build/install。

在阶段 A 的配置命令后增加：

```bash
-DC8_PHOTON_BASELINE_DIR="$PWD/../build/overlap-baseline-20260910"
```

该目录必须含本轮开始前冻结的 `ResidentPhotonCascadeBefore.hpp`；缺失时配置明确失败，不把新实现冒充旧基准。

```bash
conda activate corsika_venv
cmake --build ../build/overlap-probe-20260910 \
  --target testKokkosPhotonFrontSubmission -j1
python validation/accelerator/run_overlap_guarded.py \
  --output ../build/overlap-probe-20260910/check-photon-new -- \
  env OMP_PROC_BIND=false ../build/overlap-probe-20260910/testKokkosPhotonFrontSubmission
```

主要产物：`photon-extraction-audit-final.json`、`photon-boundary-audit.json`、`photon-test-window2048/`、`photon-ctest-window2048/` 和 `photon-figures-window2048/`。初始 256 粒子的成功及失败日志都保留。独立 OpenMP 程序的动态依赖中没有 CUDA runtime。

## 5. 尚未完成，不能当作新生产后端使用

1. 电子/正电子/μ 子驻留状态机，以及其额外 vertex control、连续损失和射电输出阶段。
2. 将两端 frame 接入 backend/session 和空气 router；目前只在隔离测试中使用新异步接口。
3. 动态批次分配、全局 history 预留、cross-species 队列、fallback 提交及调度回放的实际接线。
4. profile、CoREAS/ZHS 原始整数累积的两端合并及溢出门禁。
5. 完整空气/山体回归、N=1/2/32、真实 PROPOSAL decision-tape、500 例统计，以及独占机器热缓存性能验收。

正式入口仍只接受单 CUDA 或单 OpenMP，不新增可运行的 `cuda-openmp` 协同 CLI。正常生产构建未更新；这一步没有完成全射电协同，也没有证明净提速。
