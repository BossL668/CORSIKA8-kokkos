# beta5 OpenMP 超时恢复与补齐记录

日期：2026-09-06。实验：1 GeV 光子、垂直、emthin=1e-6、IGRF14/2027，scalar / Kokkos-CUDA / Kokkos-OpenMP 各 2000 例；OpenMP 每进程 8 线程、每批 `-N 25`。

## 结果

OpenMP 已从 **1900 补齐到 2000 例**。补跑的四批全部正常退出（returncode=0），通过逐事件 metadata、表 hash、队列清空、溢出检查及全部 Parquet 完整性检查。原定种子、二进制和物理参数未改；诊断重放不进入统计样本。

| 批次（零起算） | 原种子 | 例数 | 端到端批次时间 |
|---|---:|---:|---:|
| 019 | 2026091620 | 25 | 127.24 s |
| 035 | 2026091636 | 25 | 120.12 s |
| 051 | 2026091652 | 25 | 127.26 s |
| 067 | 2026091668 | 25 | 125.91 s |

这四批进程峰值 RSS 均低于 1.56 GiB。时间包含启动及文件关闭，不含随后的完整性检查；不能直接当作逐 shower 计算时间或跨机器加速比。

## 已定位的事实与尚未定位的部分

原 batch_019 超过 1800 s 上限后被运行器终止。此前 25 个 shower 都已结束，50 份 interaction histogram 及 7 份 Parquet 都已经写出，所有 Parquet footer 可读，各模块的 summary 已存在，但顶层 `summary.yaml` 缺失。卡住时 8 个线程在 futex 等待，不是还在持续推进 shower。

相同冻结二进制、原种子、8 线程、原 CPU 亲和下：

1. 独立 GDB 重放在 128.78 s 正常退出，没有复现卡顿。
2. 正式原种子重跑在 127.24 s 正常退出。
3. 原失败现场、GDB 重放、正式重跑的 **7 个 Parquet 文件 SHA-256 全部逐字节一致**，包括 CoREAS/ZHS 波形、粒子 profile、production profile、能损、地面粒子和相互作用。

因此，本次超时没有改变这些已写出的物理结果。**但尚未取得故障发生瞬间的 C++ backtrace，不能声称已经确定是 Kokkos::finalize、libgomp 或某一个 writer 的内部缺陷。** 顶层 summary 由 OutputManager 析构写出，其他局部对象的析构也可能发生在它之前；仅凭最后一条日志不能进一步归因。

本次没有修改物理内核、终末态、RNG、OpenMP 算法或 C++ 二进制。修复范围是可验证的运行器故障恢复与诊断能力，并非宣称根治了尚未复现的底层退出卡顿。

## 修复

- 新增 `tools/batch_liveness.py`。仅在全部预定 shower 的 timing callback 已关闭后，若进程 CPU ticks 和输出标记持续 **120 s** 均无变化，才判定退出停滞。活动中的慢 shower 不触发此门禁；原 1800 s 总时限仍保留。
- 终止该运行器自己启动的子进程前，保存各线程 stat/wchan/syscall/kernel stack（权限不足也显式记录）、CPU ticks 和 summary 状态。procfs 记录不是 C++ backtrace 的替代品。
- 明确区分 timeout 与内存、取消、非零退出、物理完整性错误。超时现场先归档，再进行有上限的**同二进制、同参数、同种子**恢复；不删除慢事件、不换种子、不将退出不完整的批次直接标为成功。
- 本次恢复脚本针对批次 019，最多两次新尝试；重复超时仍停止并保留现场。它不覆盖已有完成批次。其余 75 例是原先未启动的三批，按原种子独立补齐。
- 另外修复诊断与恢复进程同时写 `openmp_deployment.json.tmp` 的临时文件竞争：每个写者使用独立临时文件再原子替换。这是诊断启动时发现的第二个运行器问题，**不是原 C++ 超时的原因**。

## 测试

- 6 项工具测试：活动 shower 不被误杀、退出停滞触发、CPU/输出进展重置计时、进程消失处理、4 进程共 120 次并发原子写入等，通过。
- 2 项运行器故障注入测试在 WSL 和 dirac 均通过：真实测试子进程模拟输出关闭后挂起，确认被终止、现场保留、同种子重试；完整/物理失败记录禁止进入超时恢复。测试子进程不是物理样本，未纳入实验。
- 真实零 EM 事件及 7 类非法记录门禁回归通过。
- 四个正式原种子批次共 100 例全部通过原完整性检查；超时批次的 7 个 Parquet 三方 hash 一致。

## 数据与后续验收

服务器实验目录末尾为：

```text
final_beta5rng2_scalar2000_openmp2000_cuda2000_photon_1GeV_vertical_emthin1e-6_igrf14_2027_dirac_v1/
  timeout_recovery/batch_019/attempt_01/  # 原失败现场，不删除
  timeout_recovery/recovery_complete.json # 三方 hash、恢复结果
  timeout_diagnosis/gdb_before_01*       # 额外诊断，不入样本
  scripts/recover_openmp_timeout.py
  scripts/test_openmp_timeout_recovery.py
  openmp_progress.json
  simulation_complete_openmp.json
```

CPU/CUDA 各 2000 例已同步至同一服务器实验目录。完整的 2000×3 统计检验、平均 profile 和带置信区间的平均时域波形由原有服务器分析流程执行；**模拟及完整性检查完成不等于统计一致性已验收通过**。本地仍仅同步分析图片及小型结果，不把服务器 OpenMP 原始样本或大型中间数组下载到 WSL。
