# 阶段 85：进程隔离的 FLUKA batch worker

## 1. 目的

本阶段开始实现强子相互作用的多核后端，而不在同一地址空间并发调用 FLUKA。
目标是建立以下真实边界：

```text
CORSIKA 主进程
  选择相互作用模型、靶核和顶点
        |
        v
按模型 × 粒种 × 能区 × A 分类的 POD 请求
        |
        v
独立 FLUKA worker 进程（每个进程有自己的 COMMON block）
        |
        v
完整末态 POD 响应
        |
        v
主进程按 sequence_id 稳定提交到唯一 CORSIKA Stack
```

本阶段历史上先实现并验证前三层和末态返回协议，当时完整 shower 中“积累多个已经
选定的强子顶点，再异步提交”的调度层尚未接入，因此本阶段记录的 3.43× 只是
FLUKA worker 吞吐，不是 shower 端到端加速。后续生产集成、固定种子 worker 数
等价性和真实 Amdahl 结果见
`phase_86_production_fluka_classified_process_pool.md`。

## 2. FLUKA 末态生成与主栈提交解耦

文件：

```text
corsika/modules/fluka/InteractionModel.hpp
corsika/detail/modules/fluka/InteractionModel.inl
```

新增接口：

```cpp
using FinalStateParticle =
    std::tuple<Code, HEPEnergyType, DirectionVector>;
using FinalState = std::vector<FinalStateParticle>;

FinalState generateFinalState(
    Code projectileId,
    Code targetId,
    FourMomentum const& projectileP4,
    FourMomentum const& targetP4);
```

`generateFinalState()` 完成：

1. 转换 CORSIKA/FLUKA 粒子编号；
2. 变换到靶核静止系；
3. 调用 FLUKA `evtxyz_()`；
4. 读取该进程独有的 `hepevt_` COMMON block；
5. 把末态变换回原 CORSIKA 坐标系；
6. 返回完整末态，但不接触 `SecondaryView`。

原来的 `doInteraction()` 现在只是：

```cpp
auto finalState =
    generateFinalState(projectileId, targetId, projectileP4, targetP4);
for (auto& secondary : finalState) {
  view.addSecondary(std::move(secondary));
}
```

这一步非常关键：worker 进程不再需要访问主进程的 Stack、迭代器、环境节点或
`SecondaryView`。

## 3. 版本化 POD 协议

新增：

```text
corsika/framework/core/HadronicBatchProtocol.hpp
```

协议包含：

- magic：`C8HB`；
- 协议版本；
- endian marker；
- batch ID 和请求数；
- model ID；
- sequence ID；
- `(seed, shower_id, history_id, step_id, process_id, draw_domain)` 随机键；
- projectile/target PDG；
- 以 GeV 表示的入射和靶核四动量；
- status、次级数和 worker 提供给 FLUKA 的随机数数量；
- 次级 PDG、动能和三维方向。

协议对象均满足：

```cpp
std::is_standard_layout_v<T>
std::is_trivially_copyable_v<T>
```

其中没有：

- C++ 指针；
- `shared_ptr<CoordinateSystem>`；
- CORSIKA Stack iterator；
- 带进程地址的对象；
- FLUKA COMMON block 地址。

损坏的 magic、协议版本、字节序、PDG、能量或非有限四动量都会显式失败。

## 4. 与 worker 数无关的随机数

worker seed 由完整物理 history key 经过固定 SplitMix64 组合得到：

```text
seed
shower_id
history_id
step_id
process_id
draw_domain
```

worker ID、batch ID 和调度顺序不参与 seed。每个请求调用 FLUKA 前都会重新连接一个
独立 Philox 流，同时刷新 FLUKA C++ wrapper 的 16 元素随机数缓冲区。因此：

- 同一请求送到不同 worker 应得到相同末态；
- 改变 batch 顺序不改变该请求；
- worker 数量变化不改变该请求；
- 不再依赖各 worker 先前处理过多少相互作用。

这个 keyed 模式保证并行调度确定性，但不会逐事件复刻原版单一 `fluka` 流的随机数
映射。原版兼容路径仍使用原有流且已经逐字节回归；并行 keyed 模式必须用 ensemble
验证统计一致性。

## 5. 成本均衡调度

`HadronicWorkQueue.hpp` 新增：

```cpp
planHadronicWorkerAssignments(...)
```

过程如下：

1. 每个 batch 内保持同一个模型、粒种、能区和质量数区；
2. batch 总估计成本接近 `target_batch_cost`；
3. 把所有 batch 按估计成本从大到小排序；
4. 每次分配给当前累计成本最低的 worker；
5. 成本相同时以 worker ID、分类键和 sequence ID 稳定打破平局。

这相当于确定性的 LPT（longest processing time first）调度，避免昂贵的核子或重核
batch 留在最后形成串行尾部。单元测试构造不同代价的核子和 pion 任务，验证：

- 每个 batch 同类；
- 所有 sequence ID 只出现一次；
- 三个 worker 的负载差不超过最大 batch；
- 重复规划得到完全相同的 worker/batch/sequence 分配。

## 6. 独立 worker 可执行程序

新增：

```text
applications/fluka_batch_worker.cpp
```

构建产物：

```text
/home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda/applications/fluka_batch_worker
```

它只在 `WITH_FLUKA=ON` 时生成，支持：

```bash
fluka_batch_worker --self-test 64

fluka_batch_worker \
  --generate-test-input 128 \
  --output requests.bin

fluka_batch_worker \
  --input requests.bin \
  --output responses.bin
```

当前 worker 初始化 21CMA 五层空气所需的 H、N、O、Ar 靶核。一个进程内仍然串行
调用 FLUKA；多核并行来自多个独立 worker 进程。

## 7. 验证

### 7.1 单元测试

```text
HadronicWorkQueue/HadronicBatchProtocol:
104 assertions in 5 test cases, all passed

FLUKA:
272 assertions in 2 test cases, all passed
FLUKA version 2025.0.0
```

新增 FLUKA 测试直接调用 `generateFinalState()`，不创建 `SecondaryView`，并检查
末态数和动量守恒。

### 7.2 原标量 shower 逐字节回归

参数：

```text
proton
1 TeV
zenith 27 deg
azimuth 180 deg
seed 24680
FLUKA + SIBYLL
CUDA EM
emthin 1e-4
max-weight 100
```

重构前后以下文件 SHA-256 完全一致：

- `particles/particles.parquet`
- `profile/profile.parquet`
- `production_profile/profile.parquet`
- `energyloss/dEdX.parquet`
- `interactions/interactions.parquet`
- `interaction_hist/inthist_lab_1.npz`
- `interaction_hist/inthist_cms_1.npz`
- `CoREAS/observers.parquet`
- `ZHS/observers.parquet`

两边都实际执行：

```text
FLUKA interactions  11
SIBYLL interactions  2
```

这证明拆分 `generateFinalState()` 没有改变旧路径的随机数消费、粒子提交顺序或物理
输出。

### 7.3 worker 自检

64 个 proton/pion/kaon、N/O/Ar、不同能量和 history key 的请求，每个请求连续执行
两次：

```text
FLUKA batch worker self-test passed
requests               64
secondaries           932
random_values_supplied 134182
```

每个重复请求的 PID、动能、方向和随机数供应量逐项一致。

### 7.4 两个独立进程的文件回放

同一个 128 请求文件由两个新启动的 FLUKA worker 执行，响应文件逐字节一致：

```text
SHA-256
7d9dd35a7391ca0ddbad72df79c480d9bd00b86418be1b0dc245befab6151a23
```

### 7.5 四进程吞吐

硬件：

```text
Intel Core i9-14900HX
16 physical cores / 32 hardware threads
```

四个 batch、每个 256 个请求；串行执行四个独立 worker 与同时执行四个独立 worker
对比：

| 模式 | 墙钟 |
|---|---:|
| 4 worker 顺序运行 | 6.14 s |
| 4 worker 并行运行 | 1.79 s |

```text
speedup             3.430x
parallel efficiency 85.8%
```

四个并行响应分别与其串行响应逐字节一致。每个 worker 使用独立工作目录，避免
`fort.*` 和 `tables.dat` 文件冲突。

### 7.6 完整 shower 中的实测末态生成时间和调度上限

为了避免用“所有强子粒子的 scalar step 时间”高估 FLUKA worker 的收益，
`InteractionCounter` 现在只在 `doInteraction()` 外围计时，并为每个真实相互作用保存：

```text
sequence_id
projectile
target
kinetic_energy
final_state_time_ms
```

`c8_air_shower` 会在 shower 结束后把这些已发生的相互作用放入
`HadronicWorkQueue`，按实际测得的单次末态时间执行一次回顾式 LPT 调度。它只回答
“如果这些末态当时能并行生成，计算负载能否均衡”，不会改变当前标量执行路径。

验证事例：

```text
proton
100 TeV
zenith 27 deg
azimuth 180 deg
seed 24680
FLUKA + SIBYLL
CUDA EM
emthin 1e-4
max-weight 100
4 planned workers
5 ms target batch cost
```

真实计时结果：

| 项目 | 数值 |
|---|---:|
| HybridCascade 总时间 | 4019.627 ms |
| FLUKA 相互作用数 | 2206 |
| FLUKA 末态生成时间 | 226.262 ms |
| SIBYLL 相互作用数 | 160 |
| SIBYLL 末态生成时间 | 7.682 ms |
| 强子末态串行总时间 | 233.944 ms |
| 强子末态占 HybridCascade | 5.82% |
| 分类 batch 数 | 104 |
| 4-worker 预测 makespan | 58.487 ms |
| 末态内核理论加速 | 4.000x |
| worker 最大负载差 | 0.0021 ms |

四个 worker 的预测负载分别为：

```text
58.485554 ms
58.485599 ms
58.485184 ms
58.487297 ms
```

这说明“模型 × 粒种 × 能区 × A 分类 + LPT”能够把实际强子末态负载分得几乎完全
均匀。但即使忽略 IPC、等待和提交开销，端到端时间也只能从约 4019.63 ms 降至：

```text
4019.63 - 233.94 + 58.49 = 3844.17 ms
```

即理想加速约为 `1.046x`。就算使用无限多、零开销的 worker，这个参数点的 Amdahl
上限也只有约 `1.062x`。

同一事例中，正负 μ 的 scalar step 时间合计为 2574.982 ms，占 HybridCascade 总时间
约 64.1%，占全部 scalar step 时间约 87.6%。因此目前的真实优先级是：

1. 完成强子顶点的 prepare/dispatch/commit，保留未来高能和重核参数点的并行能力；
2. 同时批处理或并行化 μ 输运；只并行 FLUKA 不足以显著缩短当前单 shower 时间。

加入计时和回顾式调度后，以下文件与改动前的同 seed 100 TeV 基线 SHA-256
逐字节相同：

- 粒子、纵向 profile、产生 profile 和能损；
- 相互作用表和 lab/CMS 相互作用直方图；
- CoREAS 和 ZHS observer 输出。

因此计时采样和分类规划没有改变随机数消费、粒子顺序或物理输出。

## 8. 当前限制与下一步

当前还不能在正式 shower 中启用 `--hadronic-workers 4`，原因不是 FLUKA worker
本身，而是现有 `ScalarCascadeStepper::advance()` 会在发现离散相互作用后立即调用
`ProcessSequence::selectInteraction()` 并马上写入 `SecondaryView`。形成有效 batch
需要继续拆成：

```text
prepareTransport()
  -> 连续输运
  -> 确定 limiter
  -> 若发生强子相互作用，保存已选 process/target/vertex/P4/random key

dispatchPreparedHadronicBatches()
  -> 分类
  -> LPT 分配
  -> 多 worker 并行

commitHadronicResults()
  -> 按 sequence_id 稳定排序
  -> 写入唯一主栈
  -> 运行 doSecondaries()
  -> 删除 projectile
```

下一阶段必须同时满足：

1. 主进程始终是 CORSIKA Stack 的唯一写入者；
2. worker 失败、超时、协议错误或非法末态立即终止 shower；
3. 不能自动切回 UrQMD；
4. `proposal`/旧 FLUKA 标量路径保持默认且逐字节不变；
5. keyed 并行模式与原版做多种子 ensemble，而不是把 worker 微基准当作物理验收；
6. 报告 batch 等待时间、worker 利用率、负载差、IPC 字节和 FLUKA 计算时间；
7. 端到端评估必须计入分类、IPC、稳定合并和 CPU/GPU 同步。

阶段 84 的 1 PeV 剖面中，所有被分类的强子 scalar step 约占 10%–12%；但这些时间还
包含 tracking、连续过程、衰变竞争和未发生相互作用的步骤，不能全部交给 FLUKA
worker。上面的 100 TeV 精确计时表明，真正可由当前 worker 加速的
`doInteraction()` 末态生成只占 5.82%。这项实现为更高能量、重核和不同 cut 下的强子
门槛准备了正确并行边界；当前更大的单事件瓶颈仍是 μ± 标量输运。
