# CUDA event 等待计时器：只用于隔离诊断

此 LD_PRELOAD 库用于回答：调用 `cudaEventSynchronize` 的主机线程是否仍在消耗核时？
它不更改生产策略、事件标志、参数、返回值、`errno` 或 CUDA last-error；不增加 CUDA API 调用。
`cudaEventDestroy` 仅额外用于维护已有事件的句柄生命周期，失败的销毁不移除记录，句柄重用重新记录标志。

## 独立编译及 CPU-only 测试

在源码目录执行以下命令，输出放在新建的诊断目录，不覆盖 build/install：

```bash
c8_event_timer_dir=$(mktemp -d /tmp/c8-event-wait-timer-XXXXXX)
cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror \
  validation/accelerator/cuda_event_wait_preload.c \
  -o "$c8_event_timer_dir/libcuda_event_wait_timer.so" -ldl -pthread
sha256sum "$c8_event_timer_dir/libcuda_event_wait_timer.so"

python -B -m unittest discover -s validation/accelerator \
  -p test_cuda_event_wait_preload.py -v
```

编译不需要 CUDA 头文件，也不链接 CUDA。单元测试自行构造 fake libcudart，仅使用主机线程和毫秒级等待，
不启动 GPU、shower、服务或制表。覆盖正常/失败调用顺序、参数、返回、`errno`、last-error、两个 TID、
事件句柄销毁/重用、静态表满、缺符号 127 退出和未观测到 API 时拒绝宣称覆盖。

真实注入须另行安排，不能与未注入的性能测量重叠。仅给目标诊断二进制设置 `LD_PRELOAD`，
不要全局导出给 guard、编译器或整套生产服务。不应修改已归档的无注入 a5 报告。

## 日志及解释

正常进程退出时向 stderr 输出 JSON lines：

- `C8_CUDA_EVENT_TIMER`：按 PID、TID、API、创建 flags、retval 聚合 `calls`、`wall_ns`、`thread_cpu_ns`。
- `C8_CUDA_EVENT_TIMER_SUMMARY`：观测调用数、活动事件数、有界表溢出、未知事件及计时错误。
- 缺失真实动态符号时立即输出 `C8_CUDA_EVENT_TIMER_ERROR`，退出码 127，不静默模拟成功。

`flags & 1` 表示事件请求 `cudaEventBlockingSync`；`thread_cpu_ns` 是调用线程的实际 CPU 核时，
不是 GPU 计算时间，也不是 OpenMP 全部线程的合计。`wall_ns` 是主机 API 调用区间，包含等待流中已有 GPU 工作，
不能当作对应 kernel 的时长。请求阻塞等待不保证驱动内部零核时，初始化、内部轮询、调度与唤醒都可能耗用 CPU；
只能通过相同固定输入与调用形状的诊断对照判断实际影响，不能凭 API 名称推断 CPU 已完全休眠。

计时区间不包含本工具的符号解析、记录锁和日志输出，但时钟读取本身以及区间外的注入开销仍会扰动程序。
因此本工具只解释等待原因，不认证提速比例。可计算同 TID 的 synchronize 核时/墙钟比，并与未注入记录分开保存。

## 明确边界

- Linux/POSIX，CUDA 12 普通动态 libcudart C ABI。仅拦截四个列出的运行时入口，不覆盖静态 cudart、
  `*_ptsz`、`cudaEventRecordWithFlags` 或驱动 `cuEvent*` 入口；不拦截其他等待原语。
- 静态事件表 8192 项、聚合表 4096 项；溢出保留原调用行为并标记覆盖不完整，不扩容。
- `flags=null` 表示未通过受观察入口建立的事件或已销毁的句柄；不会猜测标志。
- `coverage_complete` 仅表示已观测入口的有限记录完整，不表示全部 CUDA 工作或物理覆盖。
- 被信号终止、`_exit()` 或崩溃的进程可能没有正常退出摘要。无摘要不能当作零等待。
- 不支持多个线程同时销毁/操作同一事件的非法应用用法；测试只覆盖 CUDA 允许的正常生命周期。
