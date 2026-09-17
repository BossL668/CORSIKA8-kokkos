# PSR：adaptive-v4 批量与双端性能测试

## 实验身份

本轮测试的是 `adaptive-v4-bounded-batching`，不是旧结果目录
`psr_t400_adaptive130_20260911_v4` 中的 **adaptive-v2**。不混用这两个“v4”名称。

服务器独立源码以本地 `ede39a1e` 源码快照为基础，覆盖当前 v4 的 controller、
pump、统计结构、应用报告和两项调度测试。未同步无关的山体工作树改动。
上述四个运行源码的 SHA-256 已逐项与本地相同文件核对。
归档源码未附 `.git`，运行 metadata 的 project_revision 为 `unknown`；
实际二进制 SHA-256 在本轮 `CONFIG.json` 中保存，不能仅凭 unknown 识别版本。

在 **PSR** 编译，不在本地为 PSR 编译。使用既有 CUDA 12.6、sm_75、
Kokkos CUDA_OPENMP、FLUKA 和 SIBYLL；不改动物理模型，不重新生成 PROPOSAL 表。
TAUOLA 及 PROPOSAL/辅助缓存复用服务器已有版本。

## 资源与运行顺序

- GPU：NVIDIA T400 4GB，驱动 580.173.02；CUDA 显存预算 70%。
- CPU：双 EPYC 9755，总计 256 物理核、512 逻辑 CPU。
  使用逻辑 CPU **382–511**，对应物理核 **126–255** 各一个硬件线程。
- 130 个 OpenMP 线程，`OMP_PROC_BIND=spread`、`OMP_PLACES=threads`；
  独立 CUDA driver 在同一授权 CPU 集内调度，不继承成只允许一个核。
- 同一新二进制顺序运行单 OpenMP 与 adaptive 双端，不让两项计时争抢服务器。
- 质子：47°/180°、emthin=1e-6、默认 max-weight、IGRF14/2027、相同天线文件，
  全部 EM/CoREAS/ZHS 加速。天线文件与本地测试文件 SHA-256 相同。
- 先做两种子 100 TeV 交替对照，再做 seed 2026110001 的 100 PeV 双端和纯 OpenMP。
  100 PeV 每模式超时保护 12 小时；进程 RSS 上限 24 GiB，系统保留 32 GiB 可用内存。

动态调度允许同 seed 产生不同树；少量事件只用于性能诊断，不宣布统计等价。
CPU 占用包含 OpenMP 忙等，GPU 利用率为设备级采样，不能单独用来证明重叠或净加速。

## 已完成门禁

- Release 独立编译与三项调度/亲和性 CTest 通过。
- 标量 PROPOSAL、单 CUDA、单 OpenMP：重构前后各 11 类数组完全一致；
  非计时 metadata 无非预期差异，CUDA/OpenMP decision trace 完全一致。
- 非法 adaptive/单端组合拒绝，帮助文本仅规范化源码行号后保持一致。
- 130 线程真实 EM＋CoREAS/ZHS fixture `N=2` 通过，约 2.686 s。
- adaptive `N=32` 通过，约 34.538 s；后续完整 shower 还会核对
  两端输运步数、波前数、批量直方图与全局计数。

这不等于强子完整能量账本或大样本物理验收通过。

## 持久化服务与结果

构建：`c8-psr-adaptive-v4-build-20260911.service`（已完成）。
运行：`c8-psr-adaptive-v4-tests-20260911.service`。
只读报告：`c8-psr-adaptive-v4-report-20260911.service`，每 30 s 刷新。
不安装或覆盖任何生产二进制；本地 v4 高能任务未中断。

服务器大数据目录：
`/data/yhlu/CorsikaData/corsika_validation_results/psr_t400_adaptive_v4_batching130_20260911/`。

每事件保留 command/result JSON、日志、资源时序、profile 与波形。
资源约每 0.5 s 采样；每 5 s 记录线程 CPU 时间、实际 CPU 和允许 CPU 集。
GPU 同时记录利用率、显存、P-state、温度和时钟。
批次分布/每端真实推进步数在事件结束时由程序的完整输出对账；不从轮询次数推算。

`MONITOR_REPORT_CN.md/json` 与 `*-resources.png`、`*-batch-inputs.png`
是可同步回本地的轻量监测产物；原始事件仍留在 PSR。

脚本：

- `validation/accelerator/build_psr_adaptive_v4.sh`：隔离构建。
- `validation/accelerator/run_psr_adaptive_v4.sh`：绑定、回归和顺序性能测试入口。
- `validation/accelerator/run_adaptive_cooperative_acceptance.py`：失败即停、资源保护与计数对账。
- `validation/accelerator/report_psr_adaptive_v4.py`：只读流式统计和监测图。

本地只读同步服务 `c8-psr-adaptive-v4-sync-20260911` 每 60 s 拉回轻量报告与图片。
SSH 失联会停止同步并保留错误状态，不会停止服务器模拟。
本地目录：
`/mnt/d/CorsikaData/corsika_validation_results/psr_t400_adaptive_v4_batching130_20260911/`。

## 首对 100 TeV 结果（高能测试仍在后续阶段）

seed 2026110001：纯 OpenMP130 51.481 s，v4 双端 78.953 s，后者慢约 53.4%。
双端实际推进 104,008,556 步，纯 OpenMP 109,727,231 步；此次变慢不是因为双端
总步数更多。两者动态执行下的 shower tree 不保证相同。

双端 CUDA 完成 655 jobs、OpenMP 完成 4355 jobs；CPU 输入量直方图的
3353 批落在 [16384,32768) 区间。CUDA 推进 12,777,001 步，
OpenMP 推进 91,231,555 步，二者合计与全局 profile 步数一致。
完整进程平均 CPU 为 11098%，GPU 平均 74.2%，峰值显存 2568 MiB。
图上主要计算阶段 CPU 约 13000%、GPU 多在 90% 左右，但仍比纯 OpenMP 慢。

GPU 结果领取延迟累计 3.330 s，OpenMP epoch 平均约 14.06 ms。
这些计时存在重叠，不能相加或直接从总时间中扣除。
结论仅为：**正确性基础门禁通过，PSR 小显卡＋130 核的本轮短事件无净加速**；
不能将本地 Fe 结果外推到这台服务器，也不能提前宣称 100 PeV 已达标。

第二种子 2026110002 同样完成：纯 OpenMP130 **57.748 s**，v4 双端 **84.685 s**。
两对中位数为 **54.614 / 81.819 s**，v4 双端慢约 **49.8%**。
100 PeV 双端已在同一持久服务中启动（seed 2026110001），完成后再顺序运行
同机纯 OpenMP130；不能用未完成的高能事件估算最终时间或批次数。
