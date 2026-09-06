# beta5 Kokkos 精简与验收

## 来源与运行保护

2026-09-05 从 beta4 当前工作树提取。来源 HEAD 为
`a9b73fb95315c6a0f3615077e5ded5db4fcea244`，包含当时未提交的 Kokkos
容量/内存安全修改，不仅是该 commit。源文件哈希和提取脚本存于项目的
`build/audit/extraction/`。beta4 源码、二进制和正在补跑的最后 25 例不变。

## 实现

- `corsika/accelerator/em/common/`：从旧 gpu 目录提取的设备无关 POD、
  环境、Philox、物理公式、fallback、输出适配和原生样条查询。
- `corsika/accelerator/em/kokkos/`：双缓冲队列、容量、波前和物理 kernel。
- `corsika/accelerator/radio/`：共享射电公式与 Kokkos 波形累积。
- `src/accelerator/em/tables/`：原生样条导出、哈希、辅助缓存、Molière。
- `applications/detail/air_shower_kokkos/`：空气应用接线与报告。
- `tests/accelerator/`：Kokkos 队列、内存、原生查询和架构检查。

部分共享 POD 暂保留 `corsika::gpu` 命名空间和 `Gpu*` 名称，以减少无意义
调用扰动；这些名字不意味着依赖 CUDA。旧 `FlatRateTableView` 精简为
`NativePhysicsView`，只保存 native 样条指针与 cut；二次插值及文件操作
移除，原生查询分支按原顺序提取。`.c8emaux` 材料/散射数据仍保留。

## 测试状态

- 初次提取的 OpenMP 核心库与 probe 已通过编译。
- 根目录已重新整理为源码、build、install；CUDA/OpenMP 的完整应用及探针
  已在新目录完成构建和安装。
- 固定种子 CPU/Kokkos、多 shower、射电及失败门禁验收待完成。
- HIP/SYCL 本机无对应工具链，不标记通过。
- 不启动新的大样本，不推送仓库，不替换 beta4 生产版本。

“能编译”不是物理验收通过。后续实际结果追加到本记录。

统一入口、成对构建及跨设备审查结果见
[BETA5_PORTABLE_ENTRY_AUDIT_CN.md](BETA5_PORTABLE_ENTRY_AUDIT_CN.md)。
