# Phase 42：五次独立重复的性能门禁

## 1. 原计划缺口

原始 CUDA 电磁后端计划要求：

> 性能测试每项运行五次取中位数。

Phase 39 已经用公共逐 shower 计时证明单个 1 PeV 事件的 CUDA 路径超过
5 倍，但当时的正式 runner 只启动一次标量 PROPOSAL 进程和一次 CUDA
进程。这可以证明该次运行通过，却不足以排除一次性的调度、温度、文件缓存
或后台负载波动。

Phase 42 将以下工具升级为进程级重复验收：

```text
validation/gpu_em/run_performance_acceptance.py
```

## 2. 重复定义

`--repetitions` 默认值为 5。每个 repetition 都：

1. 启动一个新的标量 PROPOSAL 进程；
2. 启动一个新的 CUDA EM 进程；
3. 使用完全相同的初级、能量、seed、cut、thinning 和事件数；
4. 分别读取 `simulation_timing/summary.yaml`；
5. 保留进程外部 wall time 和公共逐 shower 时间；
6. 写入独立的输出子目录和日志。

目录结构为：

```text
benchmark_root/
  repetition_000/
    proposal/
    cuda/
    proposal.log
    cuda.log
  repetition_001/
    ...
  benchmark_summary.json
```

重复使用相同 seed 是刻意的：这里测量的是同一工作负载的运行时间波动，
不是建立独立物理 ensemble。物理统计正确性继续由
`run_physics_acceptance.py` 使用独立 CPU/GPU seed 验证。

## 3. 消除固定运行顺序偏置

运行顺序按 repetition 交替：

```text
repetition 0: proposal -> cuda
repetition 1: cuda -> proposal
repetition 2: proposal -> cuda
...
```

这样 CPU 或 CUDA 都不会在五次运行中永久占据“刚启动”或“后运行”的热状态。
每次的真实顺序存入：

```text
samples[i].execution_order
```

## 4. 门禁定义

主门禁采用后端中位数之比：

\[
S_{\rm shower} =
\frac{\operatorname{median}(T_{\rm proposal,shower})}
     {\operatorname{median}(T_{\rm cuda,shower})}.
\]

其中每个 \(T_{\rm shower}\) 是该进程所有 shower 的公共计时之和。
`--minimum-speedup 5` 检查 \(S_{\rm shower}\ge 5\)。

同样报告进程外部 wall time：

\[
S_{\rm external} =
\frac{\operatorname{median}(T_{\rm proposal,external})}
     {\operatorname{median}(T_{\rm cuda,external})}.
\]

此外还保存五个逐次配对比值及其 minimum、maximum、mean 和 median。主结论
不采用最快一次，也不采用“每次比值的最大值”。

## 5. 输出 schema

`benchmark_summary.json` 新增：

```text
configuration.repetitions
samples[]
aggregate.proposal.external_wall
aggregate.proposal.summed_shower_timing
aggregate.cuda.external_wall
aggregate.cuda.summed_shower_timing
aggregate.speedup
```

为兼容已有报告读取器，主加速比仍可由以下稳定路径读取：

```text
speedup.external_wall
speedup.summed_shower_timing
```

但其含义现在是 ratio-of-medians，并由
`speedup.definition = ratio_of_backend_medians` 明确标注。

非法情况立即失败，包括：

- repetition 数不为正；
- 任一进程失败；
- timing summary 未闭合；
- CPU/CUDA 事件数不同；
- 实际 timing record 数与 `--events` 不同；
- 空、负值、NaN、无穷或零 CUDA 分母。

## 6. 自动测试

新增：

```text
validation/gpu_em/tests/test_run_performance_acceptance.py
```

测试覆盖：

- ratio-of-medians 的精确算术；
- paired speedup 序列；
- 空输入；
- 零 CUDA 分母；
- NaN 和无穷；
- 默认五次 repetition；
- proposal/CUDA 交替顺序；
- JSON schema 与 5 倍 hard gate。

执行：

```bash
/home/yuhanglu/miniconda3/envs/corsika_venv/bin/python \
  -m unittest discover \
  -s validation/gpu_em/tests \
  -p 'test_*.py' -v
```

当前结果：

```text
Ran 15 tests
OK
```

## 7. Phase 42 当时尚未宣称完成的证据

本阶段已经完成 runner、schema 和自动测试，但不能用合成测试代替正式性能
证据。原计划中的该项在需求审计中仍保持 `PARTIAL`，直到 RTX 4060 上完成
以下热缓存运行并保存五个真实样本：

```bash
python validation/gpu_em/run_performance_acceptance.py \
  --executable \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v9_1e-3_1EeV.c8emrt \
  --output-root /tmp/c8_phase42_performance_1PeV_x5 \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --seed 24027 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --minimum-speedup 5
```

正式运行应与正在执行的 physics ensemble 错开，避免 CPU/GPU 资源竞争污染
性能结论。

## 8. Phase 49 正式关闭结果

Phase 49 已在 Release、热缓存、单 CPU 数值线程和 RTX 4060 Laptop GPU
条件下完成五次真实重复：

```text
output:
  /tmp/c8_phase49_release_performance_1PeV_5rep_v1

external wall ratio-of-medians:  9.1949x
common shower ratio-of-medians: 13.7978x
status: passed
```

五个配对外部 wall-time 加速比的最小值仍为 7.8721 倍。因此本阶段定义的
五次中位数门禁已经由真实运行关闭；完整命令、样本、构建 provenance 和
严格能量 smoke 见
`phase_49_release_build_and_five_run_performance_acceptance.md`。
