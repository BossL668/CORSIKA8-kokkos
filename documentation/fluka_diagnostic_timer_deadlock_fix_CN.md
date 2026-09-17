# FLUKA 定时诊断死锁修复（2026-09-16）

## 已确认的原因

100 PeV 质子单 Kokkos-CUDA cohort 的 global76（seed `2026110076`）
于约 19:22 卡住。现场调试器读取到同一主线程的调用栈：

```text
SIBYLL doInteraction -> std::set copy -> operator new -> malloc
  [SIGALRM 打断尚未返回的 malloc]
  timer (fpe.c:62) -> quachk (fpe.c:45) -> fwrite
  -> malloc -> __lll_lock_wait_private
```

信号处理函数再次申请内存，等待被自身打断的调用所持有的分配器锁。
这不是内存耗尽、CUDA kernel 无限循环或物理能量不守恒。
此前 `c8_air_shower.cpp` 的保护仅在 `hadronic_process_pool` 存在时执行，
漏掉了 `--hadronic-workers 1` 的路径。

## 修复范围

- `src/modules/fluka/FlukaDiagnosticTimer.cpp`：使用链接器 `--wrap=fpenab_`
  调用原始初始化，然后设置 `SIGALRM` 为忽略并取消 alarm。
- `modules/fluka/CMakeLists.txt`：将包装实现链接进 `libfluka.so`。
  所有通过该模块初始化 FLUKA 的应用、后端和 worker 均获得保护。
- `applications/c8_air_shower.cpp`：删除仅覆盖强子进程池的旧条件保护。
- 不替换 FLUKA 物理生成器，不更改随机流、cut、thinning、磁偏转换算或
  PROPOSAL 数据。原初始化的浮点状态和 SIGTERM 设置均保留。
- 被禁用的只是 `.timer.out` / `.quotacheck` 周期诊断；进程时间、磁盘、
  内存和停滞监控由进程外 runner 执行，不再从异步信号回调调用 stdio。

此实现依赖 ELF 工具链的 `--wrap` 支持；未在不提供该链接选项的工具链上验收。

## 验收

1. 用实际 `modules/fluka/CMakeLists.txt` 隔离配置、构建 FLUKA 模块通过。
2. `testFlukaDiagnosticTimer`：初始化转发、alarm 取消、SIGTERM/浮点状态
   保留、重复初始化及信号与分配并行压力测试通过。
3. 冻结的生产可执行文件不变，仅替换 FLUKA 库，分别运行：
   - 标量 PROPOSAL：10 GeV 质子，修复前后各 `N=2`；
   - Kokkos-CUDA：1 TeV 质子，修复前后各 `N=2`。
   两组均使用 theta=47、phi=180、emthin=1e-6、seed=2026110076、
   原天线表及全射电；七类 Parquet 数组逐项相同，数值列字节比较也通过，
   primary/radio 配置相同。
   CUDA 物理计数、native/aux 哈希、跨事件复用状态相同；初始化耗时不要求相同。
4. 停滞监控新增 6 项测试通过，原有 6 项运行器测试通过；另有 3 项完整
   attempt 循环/内部错误 fail-fast/限定重试测试通过。
5. 100 PeV 第 76 例需重新从同一种子启动；短事件回归通过并不表示
   该高能事件或全 100 例统计验收已经完成。

## 本批数据的安全部署

为避免引入当前工作树其他研发改动，本批不重新构建整套空气程序。
复用冻结 FLUKA 物理 archive；RNG/version 桥接源码逐字验证与 `d5eea700`
相同后重编，仅加入定时器包装。原库、可执行文件和 manifest 不覆盖。

冻结程序使用 DT_RPATH，因此 runner 对子进程显式预载修复版 `libfluka.so`
（相同 SONAME）；运行时确认只映射这一份库、SIGALRM 已忽略、修复日志已出现。
修复库 SHA-256、源代码 SHA-256、回归报告与运行器版本写入独立修复记录和
每次 attempt，不能只凭旧可执行文件哈希声称运行环境未变。

数据目录内 `repairs/fluka_diagnostic_timer_20260916/` 保存调用栈、失败记录、
构建审计、测试输出和恢复检查。global76 的旧 attempt 不删除，不纳入成功样本。
global76--80 使用原种子和 70% 显存上限，每例独立进程；已有 95 例不重跑。

首次部署新监控脚本时还发现 `output_signature` 的局部变量与导入函数重名，
使 76--80 在启动约 0.5 秒时退出（尚无模拟输出）。该脚本错误已改为明确的
函数别名，且完整 attempt 循环纳入测试。此类运行器内部错误现在直接停止
队列，不再连续消耗后续事件。原失败 attempt 原样保留；只对已核实的这一
种启动错误、这五个固定 seed 给出一次显式重试，不放宽物理失败的重试规则。

## 新的停滞检测

旧 runner 只在输出已经关闭后检查退出停滞，运行中死锁只能等待 12 小时总超时。
新 runner 在 shower 关闭前同样检查：连续约 15 分钟没有物理输出变化、
主机用量低于单核 1%，且 GPU 监控持续显示空闲/低功耗，才触发运行超时。
任何 GPU 忙碌或未知状态都会取消这段空闲累计；遥测失败不能作为死锁证据。
不把一次 GPU 利用率截图作为停止依据，不修改物理输出来让事件通过。
保留失败记录并遵守有界重试规则，避免全队列长期卡在一个存活但无进展的进程上。
