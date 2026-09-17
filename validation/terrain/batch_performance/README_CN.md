# 独立山体模块：常驻批次与输出复核

编译、模拟、数值分析和绘图均在 PSR 执行。脚本带 PSR 主机检查；本地仅编辑和整理报告。使用隔离目录和冻结二进制，原高能任务与空气程序不被覆盖。

关键参数：

|参数|作用|
|---|---|
|`--batch 4096`|CPU 粒子暂存与标量任务交替的阈值|
|`--resident-wavefront-capacity 65536`|可独立提高常驻波前上限；默认跟随 batch|
|`--resident-record-capacity 65536`|完整步记录的有界缓冲；默认 max(4096,16×波前上限)，最高 1048576|
|`--radio-batch 8192`|OpenMP 射电源合批；0 可作即时提交对照|
|`--device-output-threads 32`|主机几何复核线程；0 自动选择，1 串行；不改变 Kokkos 配置的 256 线程|
|`--reference-device-geometry`|对照用原 CPU 几何对象查询；未关闭任何检查|
|`--profile-device-output`|拆分逐条输出与批量几何复核计时|

`run_case_psr.py` 使用完整低能事件、空气默认截止、emthin=1e-6、自动 Wmax、真实 DEM 和 80 站；低能射电接收窗为 128 μs。低能 Wmax<1 时，单位权重粒子实际不触发 thinning。源粒子为电子，目的是隔离 EM 开销；不是 double bang 拓扑测试。

`OutputAuditChecks.cpp` 对照原 CPU 几何与输出，包含边界、错误顺序和批缓存释放；`BufferedRadioChecks.cpp` 对照非零射电、CPU/设备来源及尾批；`InterfaceResidentChecks.hpp` 包含独立常驻队列重放。`phase2_psr.py` 和 `boundary_control_psr.py` 顺序运行完整事件，`analyze_psr.py` 检查逐字节 CSV、末段逃逸记录、能量账本、波形与性能。

本次 PSR 目录：`/data/yhlu/CorsikaData/corsika_validation_results/beta5_interface_batch_performance_20260915`。阶段 1 的较早候选和计时保留用于诊断；最终验收以 `report/acceptance.json` 为准。

图文报告整理到 D 盘：`D:\CorsikaData\beta5_doublebang_figures_20260914\11_interface_batch_performance_20260915`。
