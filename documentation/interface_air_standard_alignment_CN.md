# 山体跨介质输运按大气常驻执行模型对齐

日期：2026-09-11。

## 大气程序的实际标准

beta5 单 CUDA 大气程序采用**主机协调、设备粒子队列常驻、多 wavefront 后端调用**。
它没有一个 GPU kernel 独立运行到含强子/中微子/CPU 指定过程的整个 shower 结束。
前一次答复以“有没有 persistent kernel”作为山体与大气的区别不够准确；
两者都可以由主机发射 kernel，真正需要补齐的是后端连续推进、控制记录及输出检查点。

源码依据：

| 行为 | 大气源码 |
|---|---|
| 常驻光子循环在主机函数中，队列不空且轮数未满时继续 | [KokkosResidentPhotonCascade.hpp](../corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp)，1349 行 |
| 每轮复制一个控制 POD，并等待执行完成 | 同文件，1397–1400 行 |
| 常驻轻子也使用同类循环 | [KokkosResidentLeptonCascade.hpp](../corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp)，1798 行 |
| 普通单端路由一次光子调用上限 16 轮，轻子上限 1024 轮 | [PhysicalAcceleratedEmRouter.hpp](../corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp)，572、637 行 |
| 指定 CPU 完成事件可暂存到常驻队列排空后处理 | 同文件，`returnFallbacks()` |

因此，本次对齐的是上述实际执行模型，并未宣称取消全部 CPU 参与。

## 独立山体模块的实现

[InterfaceEmSession](../corsika/modules/transport/InterfaceEmSession.hpp) 新增
`runResidentCascade()`，实际实现在 [src/transport/InterfaceEmSession.cpp](../src/transport/InterfaceEmSession.cpp)。
山体的 `--em-backend kokkos` 默认使用这一接口，旧单步及逐轮常驻接口保留作回放参考。

一次调用内重复执行：

```text
显存 FIFO → 输运 → 设备结果验证/后继计数 → 设备扫描
                         ↓
               16 字节控制 POD → 主机后端循环
                         ↓
                后继 scatter/追加到显存 FIFO
                         ↓
               完整步记录追加到设备诊断账本
                         ↓
           继续下一轮，或在检查点统一返回输出
```

控制记录只包含后继数量、错误码和 CPU 回退数量。设备结果验证在队列消费之前执行；
非法粒子或容量超限保留当前 wavefront 输入。多轮调用的前序已完成工作不会回滚，
发生异常的 shower 必须标为失败。

完整诊断账本默认最多 4096 条记录，预分配且计入内存预算；每次调用仅下载有效范围。
停止点为队列排空、指定 CPU 回退、CPU 标量交错、账本容量或轮数上限（默认 1024）。
停止时粒子仍留在设备队列，外层 `HybridCascade` 持续调度，直到 CPU 与设备工作都结束。
无 CPU 回退且账本足够的纯电磁事例可以在一次后端调用内全部完成。

山体保留原混合 γ/e± FIFO 和精确 history 分配顺序。每轮 history 回调只预留编号，
不读取或更新设备粒子；CPU 尚有工作时，在前沿小于一个 batch 时交还标量调度器。
指定 CPU 回退沿用山体的原始完成次序，不改成大气的延后批处理策略，
以保留既有随机历史及诊断顺序。输运物理、材料/磁场约定和空气源码均未由本任务修改。

## 验证范围

所有编译及运行均在 `psrpku2025_PKU`，CPU affinity 为 0–255，共 256 个物理核；
OpenMP 实际并发 256，CUDA 使用 T400 4GB。本地只编辑、核对源码并收取结果。

- 两种后端的队列与六种界面配置检查：包括水/岩石、同材质、四面体、相反磁场、
  交换材料表和同材质不同磁场。
- 新多轮回放与独立主机 FIFO 逐位比较所有粒子、次级、路径和能量记录；
  使用仅 96 条记录的设备账本强制多次容量检查点，同时检查重入、非法编号、
  明确轮数上限、设备验证失败及溢出后输入保持。
- 真实 DEM 的光子向上、光子向下、岩内电子、10 TeV 强制 CC νe：
  每例比较新 batched、新多轮 resident，以及上一轮冻结二进制的完整 CSV、表和诊断。
  冻结二进制 SHA-256 与上一轮存档逐一一致。
- CUPTI 复核新多轮路径的实际 GPU kernel、控制回传、批量输出和粒子常驻。

验收结果在 PSR 的 `build/psr-interface-resident-20260911/air-standard/` 中保存。

## 验收结果

OpenMP、CUDA 分别通过 **7/7 模块测试、8/8 应用运行**。每种后端内，
新 batched、新 resident 与上一轮冻结结果的完整 CSV、材料表、诊断和回退数量一致；
没有未完成粒子、材料误配或输出截断。这里的逐位一致是同后端的新旧实现比较，
不以此宣称任意 CUDA/OpenMP 浮点结果都逐位相同。

两个后端的以下 resident 计数相同：

| 真实 DEM 算例 | 输运步 | wavefront 数 | 后端调用 / 完整记录下载次数 | 单次最多 wavefront | CPU 上传粒子数 | 指定 CPU 回退 |
|---|---:|---:|---:|---:|---:|---:|
| 光子向上 | 1489 | 59 | 1 | 59 | 1 | 0 |
| 光子向下 | 2109 | 74 | 1 | 74 | 1 | 0 |
| 岩内电子 | 1639 | 92 | 1 | 92 | 1 | 0 |
| 10 TeV 强制 CC νe | 194841 | 3095 | 2828 | 90 | 22535 | 27 |

前三例在一次后端调用内排空电磁级联。νe 例继续保留 CPU 标量交错和 CPU 产物注入，
所以调用次数仍较多；它验证了回退后能继续推进到整个混合 shower 结束，
并不代表中微子、强子和所有指定过程已在 GPU 上实现。

默认设备账本上限 4096 条，νe 例实际峰值 4057 条；模块计入的设备分配为
114922048 字节，考虑队列整理时新旧分配重叠的预计峰值为 122262080 字节，
均在设置的 128 MiB 模块预算内。这不是整个 CUDA 进程的显存占用。
96 条账本的专门回放实际触发了 11 或 12 次容量检查点，全部记录与独立主机 FIFO 一致。

## 实际 CUDA 活动

使用 Kokkos Tools 标签和 CUPTI 实际 kernel/memcpy 活动交叉核对；
两个新版 profile 运行的完整物理输出和计数均与未插桩运行一致，trace 没有错误或丢失记录。
基准 batched trace 沿用上一轮验收存档，其物理输出已通过本轮冻结结果检查。

| CUDA 实测项 | 光子向上 resident | 强制 CC νe resident |
|---|---:|---:|
| 输运 kernel | 59 | 3095 |
| 验证/计数 kernel | 59 | 3095 |
| 扫描 kernel（Kokkos 每轮两次实际启动） | 118 | 6190 |
| 后继 scatter kernel | 59 | 3095 |
| 粒子 H2D 上传次数 / 字节 | 1 / 112 | 2825 / 2523920 |
| 控制 D2H 次数 / 字节 | 59 / 944 | 3095 / 49520 |
| 完整记录 D2H 次数 / 字节 | 1 / 1357968 | 2828 / 177694992 |

光子算例从第一次输运 kernel 开始，**实际 H2D memcpy 为零**；每轮从设备 FIFO 取前沿，
后继在 GPU 上验证、扫描、scatter 并追加回 FIFO。所有粒子队列分配均在 CUDA 显存，
没有向主机下载整个队列，结束后分配正常释放。静态材料表和几何数据没有在推进中重传。
Kokkos 对控制清零会记录逻辑 deep-copy 回调，实际执行为填充而非 H2D memcpy；
分析器据真实 CUPTI 活动区分二者。

由于保留完整逐步诊断，完整记录下载的总字节数仍与基准一致，减少的是下载批次。
每轮 16 字节控制回传和主机等待仍然存在；本次没有实现 persistent kernel。
另外 PSR 同时存在其他计算任务，本次只作正确性及常驻性验收，不作加速比结论。

## 交付与限制

- 默认路径：`c8_terrain_cascade --em-backend kokkos --em-scheduler resident`；
  省略 `--em-scheduler` 时 Kokkos 山体同样选择 resident。
- 模块独立封装为 `CORSIKA8::InterfaceEm`，支持封闭界面的两侧材料配置、密度模型和
  各自的均匀磁场矢量；目前不是任意多个嵌套介质区或空间变化磁场的通用几何调度器。
  γ/e± 设备推进，其他粒子及指定回退继续由既有 CPU 过程处理；未加入射电计算。
- 本任务没有编辑大气输运或大气常驻后端。PSR 原始源码 646 个存档文件校验无变化。
  本地同时存在另一项大气协同调度工作，相关修改已保留且未纳入本任务覆盖包。
- 结果摘要：`air-standard/acceptance_summary.json`；实际活动核对：
  `air-standard/residency_analysis.json`；完整 CUPTI trace：`air-standard/profile-v1/`。
  `binary_provenance.json`、`server_source_audit.json`、`source-overlay-v2.json`
  分别记录二进制来源、原始源码保护及最终交付文件哈希。
- 本地存档位于 `../build/interface-resident-validation-20260911/air-standard/results/`，
  包含日志、结果清单及 trace；大体积轨迹 CSV 保存在 PSR，对应 SHA-256 在验收清单内。
