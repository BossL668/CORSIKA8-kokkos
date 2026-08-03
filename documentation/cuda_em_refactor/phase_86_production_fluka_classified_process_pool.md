# 阶段 86：生产路径中的强子分类存储与多进程 FLUKA 后端

## 1. 阶段结论

本阶段已经把阶段 85 的进程隔离 FLUKA worker 接入真实的
`HybridCascade` shower 运行路径。实现的不是在 shower 结束后分析耗时，而是：

```text
CPU scalar stepper
  |
  | 已经选定相互作用、靶核、顶点和入射四动量
  v
挂起强子 projectile
  |
  v
model × species × energy bin × A bin 分类队列
  |
  v
按预计计算量形成近等时 batch
  |
  v
多个持久、进程隔离的 FLUKA worker
  |
  v
按 sequence_id 稳定排序
  |
  v
主进程提交末态到唯一 CORSIKA Stack
```

这满足了“强子的分类存储，并尽量让每一批强子的计算时间接近”的核心要求。当前只把
低能强子末态生成交给多进程 FLUKA；高能 SIBYLL、强子的 tracking、相互作用距离
竞争、衰变以及主栈提交仍在主进程执行。

## 2. 为什么使用进程而不是线程

FLUKA 的 Fortran 代码保存可变的进程全局状态和 COMMON block。在一个地址空间内
让多个 C++ 线程同时调用 FLUKA，无法保证：

- COMMON block 不互相覆盖；
- 随机数缓冲区不交叉；
- HEPEVT 末态属于正确的相互作用；
- 初始化和材料表状态是线程安全的。

因此当前实现采用四个持久 worker 进程。每个进程各自拥有：

- 独立 FLUKA 初始化；
- 独立 COMMON block；
- 独立工作目录；
- 与主进程连接的一对 Unix-domain socket。

worker 在 CUDA runtime 初始化之前由主进程 `fork/exec`，避免在已经创建 CUDA
context 后再 `fork`。

## 3. 分类键与存储结构

核心文件：

```text
corsika/framework/core/HadronicWorkQueue.hpp
tests/framework/testHadronicWorkQueue.cpp
```

分类键 `HadronicWorkKey` 有四个维度：

1. `model`：
   - `low_energy`；
   - `high_energy`。
2. `species`：
   - nucleon；
   - pion；
   - kaon；
   - other meson；
   - other baryon；
   - nucleus；
   - other hadron。
3. `energy_bin`：
   - 默认每个 octave 两个 bin；
   - 即相邻边界能量相差 \(\sqrt{2}\)；
   - 用动能分类。
4. `mass_number_bin`：
   - 非原子核为 0；
   - 原子核按 \(A\) 的二进制尺度分段。

完整 PID、四动量和 history 并没有被分类键替代。分类键只决定调度；真实物理请求
仍保留精确 PDG、靶核、四动量和随机数身份。

`ClassifiedHadronicWorkQueue` 使用：

```cpp
std::map<HadronicWorkKey, Bucket>
```

每个 bucket 内部是 FIFO 队列。它具有三个确定性约束：

- 同一分类内保持入队顺序；
- 分类计算不消耗随机数；
- 相同预计负载时用分类键和 `sequence_id` 作为稳定 tie-break。

## 4. 近等时 batch 如何形成

每个分类维护尚未处理的预计总耗时。调度器先选剩余工作量最大的分类，再从该分类取出
一个 batch，直到：

- 预计耗时达到 `--hadronic-target-batch-ms`；或
- 粒子数达到 `--hadronic-max-batch`。

不同分类不会混进同一 batch。因此一个 batch 内的 projectile 使用相同的：

- interaction model；
- 粗粒种；
- 能量尺度；
- 核质量尺度。

batch 形成后使用 longest-processing-time-first 方式分配给当前预计总负载最小的
worker。这既防止昂贵分类在末尾形成长串行尾巴，也比简单 round-robin 更接近
“每个 worker 的总计算时间相近”。

每个分类的预计耗时由粒子种类、能量 bin、核质量 bin 和
`--hadronic-initial-cost-ms` 确定。真实 FLUKA 耗时只写入诊断 metadata，
不会反馈给正在运行的调度器。这样可以避免 OS 调度噪声改变后续 worker 分配，
确保相同 seed 和相同配置的 shower 可重复。真实耗时仍可用于离线校准下一版
确定性系数。

## 5. prepare、dispatch、commit 三阶段

### 5.1 prepare

相关文件：

```text
corsika/framework/core/HadronicInteractionDeferral.hpp
corsika/detail/framework/process/InteractionCounter.inl
corsika/detail/framework/core/HybridCascade.inl
```

`ScalarCascadeStepper` 仍负责原来的物理竞争和顶点推进。当
`InteractionCounter<FLUKA>` 到达已经选定的 FLUKA 相互作用时，它不在主进程调用
事件生成器，而是保存：

- worker model；
- projectile 和 target PDG；
- projectile 和 target 四动量；
- `history_id`、`step_id`；
- 稳定 `sequence_id`；
- history-keyed 随机数键。

此时 projectile 保留在主栈中并被 scheduler 标成 suspended，不能被下一步推进。
队列保存的是请求和主栈句柄的受控组合，不把 C++ 指针发送给 worker。

### 5.2 dispatch

相关文件：

```text
corsika/framework/core/HadronicBatchProtocol.hpp
corsika/framework/core/HadronicProcessPool.hpp
applications/fluka_batch_worker.cpp
```

socket 协议只发送标准布局、可平凡复制的 POD：

- magic `C8HB`；
- 协议版本和 endian marker；
- batch、sequence 和随机数身份；
- PDG；
- 以 GeV 为固定单位的四动量。

worker 返回：

- status；
- `sequence_id`；
- 次级粒子数；
- FLUKA 随机数供应计数；
- 单次事件生成器耗时；
- 每个次级的 PDG、动能和方向。

协议检查失败、worker 退出、PDG 非法、非有限四动量、响应顺序错误或次级数不一致
都会终止当前 shower，不会静默切回另一个物理模型。

### 5.3 commit

所有响应先按 `sequence_id` 排序，然后由主进程依次调用：

```cpp
ScalarCascadeStepper::commitDeferredHadronicInteraction(...)
```

只有主进程能够修改 CORSIKA Stack。提交时再次检查：

- projectile 尚未被删除；
- `history_id` 没有改变；
- worker 响应的 `sequence_id` 匹配；
- FLUKA 耗时为有限非负数。

提交完成后才恢复该 projectile 对应的 scheduler 状态。这样不会出现粒子被重复推进、
同一末态提交两次或 completion-order 改变主栈写回次序。

## 6. 随机数与 worker 数无关

worker 随机流的身份为：

```text
(seed, shower_id, history_id, step_id, process_id, draw_domain)
```

它不包含：

- worker ID；
- batch ID；
- dispatch 顺序；
- 哪个 worker 最先完成。

因此把 worker 数从 1 改成 4 不会改变某次已经确定的 FLUKA 相互作用所看到的随机
序列。请求和响应还分别生成不含计时字段的 FNV-1a fingerprint，用来检查两个运行
是否处理了完全相同的物理请求和末态。

需要区分两种确定性：

- `fluka-process` 的 worker 数从 1 改到 4：目标是逐事例字节一致，当前已经达到；
- 原版 scalar FLUKA 与 `fluka-process`：随机流组织方式不同，不承诺同一个 shower
  逐粒子一致，必须做 ensemble 统计一致性检验。

## 7. 固定种子 1 worker 与 4 worker 的证据

验证输出：

```text
/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_scaling_proton_E100000GeV_seed24680_fluka_process_fingerprint_w1_v4

/home/yuhanglu/21CMA/corsika_validation_results/
  hadronic_scaling_proton_E100000GeV_seed24680_fluka_process_fingerprint_w4_v4
```

设置为 100 TeV proton、相同非零 seed、相同 shower 参数，只改变 worker 数。

| 指标 | 1 worker | 4 workers |
|---|---:|---:|
| FLUKA interactions | 2698 | 2698 |
| worker execute | 338.563 ms | 143.999 ms |
| shower wall | 4742.864 ms | 4509.966 ms |
| worker execute 加速 | 1.000× | 2.351× |
| shower 内部加速 | 1.000× | 1.052× |

两次运行的全部 flush request/response fingerprint 列表一致。下列九个物理输出文件
逐字节 SHA-256 一致：

```text
particles/particles.parquet
profile/longitudinal.parquet
production_profile/production.parquet
energyloss/energyloss.parquet
interactions/interactions.parquet
lab_hist/lab_hist.parquet
cms_hist/cms_hist.parquet
CoREAS/CoREAS.parquet
ZHS/ZHS.parquet
```

四个 worker 的实测 FLUKA 物理负载为：

```text
72.102, 73.197, 71.859, 67.740 ms
```

最大差为 5.456 ms，说明分类加 LPT 分配已经实现较好的负载均衡。

## 8. 为什么强子内核加速没有直接变成端到端 2.35 倍

上面的固定种子测试中，FLUKA worker execute 只占总 shower 的一小部分。即使把它
完全消除，仍有：

- PROPOSAL 的 μ 子输运；
- EM 输运和 CUDA/CPU 路由；
- tracking；
- stack 和 writer；
- batch 积累产生的依赖；
- 初始化。

所以 2.351× 是被并行部分的加速，1.052× 才是该阶段单独带来的 shower 内部
加速。这是正常的 Amdahl 限制，不应把前者写成端到端性能。

该结果也给出了下一步的明确优先级：强子分类和 FLUKA 并行已经不再是主要瓶颈，
必须继续处理 μ 子和输出侧的串行负载。阶段 87 记录后续结果。

## 9. CLI

CUDA hybrid 路径的强子选项为：

```text
--hadronic-backend scalar|fluka-process
--hadronic-workers N
--hadronic-min-batch N
--hadronic-target-batch-ms T
--hadronic-max-batch N
--hadronic-initial-cost-ms T
--hadronic-worker-executable PATH
```

建议的当前本机默认值：

```text
--hadronic-backend fluka-process
--hadronic-workers 4
--hadronic-min-batch 64
--hadronic-target-batch-ms 5
--hadronic-max-batch 256
```

`--em-backend proposal` 的原版 scalar 路径不自动启用 worker pool，CPU 默认行为
保持不变。

## 10. FLUKA 完整性

当前构建链接：

```text
/home/yuhanglu/fluka/libflukahp.a
```

运行时使用：

```text
FLUPRO=/home/yuhanglu/fluka
```

输出 metadata 记录：

```yaml
hadronic_models:
  low_energy:
    name: FLUKA
    version: 2025.0.0
```

`testModules` 在显式设置 `FLUPRO` 后通过。没有设置 `FLUPRO` 时测试会明确失败，
而不是悄悄使用另一个低能强子模型。
