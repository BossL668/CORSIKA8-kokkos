# Phase 95：CUDA 主进程 FLUKA `SIGALRM` 死锁修复

## 1. 现象与根因

100 PeV、`emthin=1e-4`、CUDA EM/射电和四个 `fluka-process` worker 的长期
任务中，一个 `c8_air_shower` 主进程停止增加 CPU 时间，主线程永久停在
`futex_wait_queue`。四个 FLUKA worker 仍在 Unix socket 上等待请求，GPU 队列
也不再前进。主进程同时持有一个零字节 `.quotacheck` 文件描述符。

从 `libflukahp.a` 提取并反汇编 `fpe.o` 后可见：`fpenab_` 每 60 秒安装一次
`SIGALRM` timer；handler 调用 `quachk`，后者执行 `fopen`、`fwrite`、
`fflush`、`fseek`、`fclose` 和 `remove`。这些 stdio 调用不是
async-signal-safe。如果 signal 恰好中断持有 libc stream lock 的 CUDA helper
thread，handler 会在同一进程内等待该线程已经持有的锁，形成不可恢复的自死锁。

## 2. 修复边界

`HadronicProcessPool` 必须在父进程首次接触 CUDA runtime 之前 `fork/exec`
worker。worker 启动后，父进程仍构造一个 FLUKA `Interaction`，用于相互作用率和
分类状态，但所有选中的低能 FLUKA 末态均在隔离 worker 中生成。

因此修复位于父进程构造 `Interaction` 之后、创建 CUDA helper thread 之前：

```cpp
if (hadronic_process_pool) {
  if (::signal(SIGALRM, SIG_IGN) == SIG_ERR) {
    throw std::runtime_error(
        "could not disable the parent FLUKA SIGALRM timer");
  }
  ::alarm(0);
}
```

该分支只在同时启用 `WITH_FLUKA`、CUDA EM 和 `fluka-process` 时执行。

- worker 已经 `exec`，不会继承后续的 signal disposition 修改；
- worker 保留 FLUKA 原生 timer、COMMON block、随机数和末态生成；
- 父进程的强子率和分类对象仍保留，只取消与物理无关的 quota timer；
- 标量 `Cascade + PROPOSAL + FLUKA` 默认路径完全不变；
- 无法修改 signal disposition 时 fail closed，当前 shower 不继续运行。

## 3. 验证

独立验证可执行文件：

```text
/home/yuhanglu/21CMA/corsika-21cma-cuda/
corsika8_gpu_refactor_build_cuda/validation_variants/
gpu_parent_fluka_timer_fix_20260802/c8_air_shower
```

SHA-256：

```text
82365bf42b557ed1684fd3a20fdab055f38ed606d9de3ff6af3e4443594ecded
```

使用质子、100 TeV、`theta=47 deg`、`phi=180 deg`、IGRF13/2025、
`emthin=1e-4`、CUDA CoREAS/ZHS 和四个 FLUKA worker 连续运行 3 个 shower：

- 完整墙钟时间：80.518772626 s，跨过原 60 s timer 边界；
- 3 个 shower 全部完成；
- profile、production profile、ground particles、GPU summary、CoREAS 和 ZHS
  输出均存在且非空；
- 主进程运行目录没有生成 `.quotacheck`；
- 正式 100 PeV 进程在 60 s 后的 `SigIgn` 包含 `SIGALRM` 位，CPU 时间继续增加；
- `testFramework`、`testModules` 和 `testOutput` 通过（FLUKA module 测试需设置
  `FLUPRO`）。

正式 100 PeV 新旧构建还复用了相同的首批种子。新 A/B 分片的第一个 shower
分别关闭后，立即对旧/新 `inthist_lab_1.npz` 与 `inthist_cms_1.npz` 做逻辑
逐位比较。四组文件中的数组键、shape、dtype 和全部数值均完全相同；这已经证明
两个独立种子的强子相互作用直方图没有被 signal 修复改变。profile、地面粒子和
CoREAS/ZHS 的大文件比较推迟到 500-event CUDA 生产运行结束，避免验收 I/O 污染
运行时间。

smoke 数据位于：

```text
/mnt/d/CorsikaData/corsika_validation_results/
smoke_gpu_parent_fluka_timer_fix_20260802/
```

## 4. 生产 campaign 处理

旧可执行文件 `c9c811...` 产生的 15 个已完成 CUDA 事例不与新构建混合。
两个旧分片均保留并增加后缀：

```text
_superseded_fluka_parent_timer_deadlock_20260802
```

正式 500-event CUDA 样本用新可执行文件从原始非重叠种子区间重新开始：

- shard A：`10400001..10400250`；
- shard B：`10400251..10400500`。

最终验收器要求两个 manifest 的可执行文件哈希完全相同，并明确排除 superseded
目录，因此不会把修复前后样本混入最终 500-event 统计量。
