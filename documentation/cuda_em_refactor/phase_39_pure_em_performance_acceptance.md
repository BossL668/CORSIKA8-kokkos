# Phase 39：纯电磁 shower 性能验收与后端无关计时

## 1. 本阶段目标

此前 CUDA 后端已经记录 kernel、传输、CPU fallback 与 HybridCascade
总时间，但标量 PROPOSAL 路径没有使用同一个计时边界。因此，不能仅用
两套后端各自打印的数字做正式加速比验收。

本阶段解决两个问题：

1. 为标量 `Cascade + PROPOSAL` 和 `HybridCascade + CUDA` 增加同一
   `OutputManager` 生命周期内的逐 shower wall-clock 计时；
2. 建立可重复执行的纯电磁性能验收程序，显式固定 CPU 线程数、物理
   cut、thinning、射电开关与 GPU 表格热缓存状态。

本阶段只验收性能，不代替 CPU/GPU 物理分布的集合统计验收。

## 2. 后端无关计时

新增：

```text
corsika/output/SimulationTiming.hpp
```

`SimulationTiming` 是一个普通 `BaseOutput` participant。它由
`OutputManager` 的公共回调启动和停止：

```text
OutputManager::startOfShower()
    -> SimulationTiming::startOfShower()

标量 Cascade 或 HybridCascade 执行

OutputManager::endOfShower()
    -> 各物理 writer 刷新
    -> SimulationTiming::endOfShower()
```

`c8_air_shower` 使用输出键 `simulation_timing` 注册它。当前
`OutputManager` 使用有序 map，该键排在现有物理 writer 之后，所以结束
时间包含这些 writer 的逐 shower 刷新。

每个 shower 输出：

```yaml
shower_0:
  closed: true
  status: closed
  wall_time_ms: 4276.866909
```

这里的 `closed` 只表示 start/end 回调正确配对。它不代表 GPU 物理输出
一定完整；CUDA error、queue overflow、非法状态和 fallback 完整性仍由
`GpuEmRunOutput` 判断。

若 library 在 shower 尚未正常结束时退出，则状态为：

```text
library_ended_during_shower
```

若回调次序或 shower ID 不合法，则状态为：

```text
invalid_callback_order
```

这防止崩溃或中断事件被误计为很快完成的有效事件。

## 3. 单元测试

新增：

```text
tests/output/testSimulationTiming.cpp
```

测试覆盖：

- 配置原样写回；
- shower 启动后的 `in_progress` 状态；
- 正常结束后的非零时间和 `closed` 状态；
- library 在 shower 中途结束时的未闭合状态。

执行：

```bash
cd /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda
./tests/output/testOutput "SimulationTiming"
```

结果：

```text
All tests passed (13 assertions in 1 test case)
```

CPU 与 CUDA 各执行一个 10 GeV smoke shower，也均生成了有效
`simulation_timing/summary.yaml`：

```text
CPU:  62.594 ms
CUDA: 107.967 ms
```

低能 shower 中 GPU 初始化与 batch 开销占主导，CUDA 比 CPU 慢是预期
行为；它同时验证计时器没有只接入某一个后端。

## 4. 可重复的性能验收程序

新增：

```text
validation/gpu_em/run_performance_acceptance.py
validation/gpu_em/README.md
```

runner 的主要约束如下：

- CPU 与 CUDA 使用同一个 `c8_air_shower` 可执行文件；
- CPU 路径明确选择 `--em-backend proposal`；
- GPU 路径明确选择 `--em-backend cuda`；
- `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 和
  `NUMEXPR_NUM_THREADS` 均固定为 1；
- 默认 `--ring 0`，不生成 CoREAS/ZHS 天线投影，防止射电时间混入；
- hadron、muon、tau cut 同时设为远高于初级能量，用来隔离纯 EM
  cascade；
- CPU 与 CUDA 使用完全相同的初级、能量、cut、thinning、最大权重和
  shower 数；
- 正式 CUDA 计时前验证或生成 Molière 辅助热缓存；
- 输出精确命令、环境、外部进程 wall time 和逐 shower 公共计时；
- 输出目录已存在时拒绝覆盖；
- 未闭合 timing record、命令失败或缓存缺失均使验收失败；
- `--minimum-speedup` 可以把最低加速要求变成机器可检查条件。

runner 不用相同 seed 的逐粒子相等性代表物理正确。CPU 与 CUDA 的随机数
实现不同，物理正确性必须通过独立 shower 集合的统计分布判断。

## 5. 正式验收配置

执行命令：

```bash
cd /home/yuhanglu/21CMA/corsika8_gpu_refactor

/home/yuhanglu/miniconda3/bin/conda run -n corsika_venv \
  python validation/gpu_em/run_performance_acceptance.py \
  --executable \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table \
    /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v9_1e-3_1EeV.c8emrt \
  --output-root \
    /tmp/c8_phase39_performance_acceptance_1PeV \
  --energy-gev 1000000 \
  --events 1 \
  --seed 24027 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --minimum-speedup 5
```

关键配置为：

```text
primary                     electron (PDG 11)
energy                      1 PeV
events                      1
EM cut                      0.5 MeV
EM thinning                 1e-4
maximum weight              100
hadron/muon/tau cut         1e13 GeV
radio ring                  0
CPU numerical threads       1
GPU                         RTX 4060 Laptop, sm_89
rate-table tolerance        1e-3
Moliere cache               hot, 539032 bytes
```

将非 EM cut 提高并不会删除 CUDA 内已经明确注册的稀有过程 fallback。
CUDA 的指定末态 fallback 仍被执行并计时，只是非 EM 末态不会进一步形成
一个大的 CPU cascade。

## 6. 正式验收结果

输出：

```text
/tmp/c8_phase39_performance_acceptance_1PeV/
  benchmark_summary.json
  proposal.log
  cuda.log
  proposal/
  cuda/
```

精确结果：

| 指标 | 标量 PROPOSAL | CUDA EM | 加速比 |
|---|---:|---:|---:|
| 公共逐 shower 计时 | 87.808166 s | 4.276867 s | 20.530956× |
| 整个进程 wall time | 90.751904 s | 8.968781 s | 10.118644× |

因此两种计时边界都超过原计划的 5× 要求：

```text
status: passed
minimum required speedup: 5
summed shower timing speedup: 20.530955972284616
external wall speedup: 10.118644032232785
```

CUDA 事件统计还显示：

```text
GPU transported particles     17,211,930
HybridCascade total            4276.915 ms
CPU specified fallback          140.550 ms
profile kernel                 1208.950 ms
all backend kernels            4361.860 ms
radio tracks                          0
```

公共计时与 HybridCascade 自身计时相差约 0.05 ms，说明两者的事件边界
一致。kernel 总时间可以大于 wall time，因为多个 stream 的 kernel 允许
重叠执行，不能把所有 kernel event duration 当作端到端时间。

## 7. 辅助交叉检查

在正式 runner 之前还执行了一个手工、单核、ring 0 的同配置测试：

```text
CPU external wall:  108.34 s
CUDA external wall:  20.16 s
speedup:              5.37×
```

它同样超过 5×。该测试仍包含较早版本的不同初始化与输出路径，因此只作
交叉检查，不作为最终数字。

另一个 ring 1 测试得到约 8.55× 的进程加速，但它包含射电轨迹与天线
输出，不用于纯 EM 性能结论。

## 8. 结论与剩余验收

本阶段完成了原计划中以下性能要求：

- Release 模式；
- 热缓存；
- 标量 CPU 单核对比；
- 0.5 MeV EM cut；
- 纯 EM cascade；
- CPU fallback 保留并计时；
- 端到端加速至少 5×。

当前正式结果在共同 shower 计时边界上为 **20.53×**，在包含程序初始化
的进程边界上为 **10.12×**。

这不表示完整 CUDA EM 后端已经完成科研级验收。仍必须完成：

1. CPU/GPU 独立 shower 集合的统计比较；
2. \(X_{\max}\)、\(N_e(X)\)、\(N_\gamma(X)\)、\(dE/dX\) 的均值与形状；
3. 地面能谱、横向分布和到达时间；
4. 各能区、天顶角、cut 与 thinning 组合；
5. 完整 21CMA 强子初级和 CPU CoREAS 轨迹输入；
6. 关键均值不超过 1% 偏差且差异不超过统计误差。

下一阶段将建立集合输出提取与统计比较工具。只有该测试矩阵通过，才能把
性能通过与物理通过合并为生产后端结论。
